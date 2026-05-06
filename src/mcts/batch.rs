use super::NodeIdx;
use crate::board::Board;
/// Batch accumulator and signal channel for leaf evaluation.
///
/// Instead of evaluating each MCTS leaf immediately (one rollout / one network
/// call per leaf), workers deposit their leaves here.  Once the batch reaches
/// `capacity`, the caller evaluates all non-terminal positions together — one
/// GPU forward pass in M5, parallel rollouts now — and backprops all results.
use std::sync::{Arc, Condvar, Mutex};

// ---------------------------------------------------------------------------
// PendingLeaf
// ---------------------------------------------------------------------------

/// One pending evaluation request deposited by an MCTS worker.
pub struct PendingLeaf {
    /// Arena index of the selected leaf node.
    pub leaf: NodeIdx,
    /// Board position at the leaf — the input to the evaluator.
    pub board: Board,
    /// Path from the leaf to the root, used to remove virtual loss.
    pub vl_path: Vec<NodeIdx>,
    /// True when the leaf is a terminal (no legal moves).
    /// Terminal leaves always get value −1.0 and bypass the evaluator.
    pub is_terminal: bool,
}

/// Evaluator result for a single leaf.
#[derive(Clone, Debug, PartialEq)]
pub struct EvalResult {
    /// Raw policy logits aligned to the canonical move-index space.
    pub policy_logits: Vec<f32>,
    /// Scalar value in [-1, 1] from the perspective of the leaf side to move.
    pub value: f32,
}

// ---------------------------------------------------------------------------
// LeafBatch
// ---------------------------------------------------------------------------

/// Fixed-capacity accumulator for leaf positions.
pub struct LeafBatch {
    pending: Vec<PendingLeaf>,
    capacity: usize,
}

impl LeafBatch {
    /// Create a batch that fires when `capacity` leaves have been deposited.
    /// Capacity is clamped to at least 1.
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.max(1);
        LeafBatch {
            pending: Vec::with_capacity(cap),
            capacity: cap,
        }
    }

    /// Deposit one leaf into the batch.
    pub fn push(&mut self, leaf: NodeIdx, board: Board, vl_path: Vec<NodeIdx>, is_terminal: bool) {
        self.pending.push(PendingLeaf {
            leaf,
            board,
            vl_path,
            is_terminal,
        });
    }

    /// True when the batch has reached its capacity and is ready to evaluate.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.pending.len() >= self.capacity
    }

    /// Number of leaves currently in the batch.
    #[inline]
    pub fn len(&self) -> usize {
        self.pending.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// The configured capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Take all pending leaves out of the batch, leaving it empty and ready
    /// for the next round.
    pub fn drain(&mut self) -> Vec<PendingLeaf> {
        std::mem::take(&mut self.pending)
    }
}

// ---------------------------------------------------------------------------
// BatchChannel — signal-based evaluator/worker handshake
// ---------------------------------------------------------------------------

/// Phase of the shared batch state machine.
///
/// ```text
/// Filling ──(batch full)──► Evaluating ──(post_results)──► Consuming
///    ▲                                                          │
///    └──────────────(last consumer reads result)───────────────┘
/// ```
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
enum Phase {
    /// Accepting deposits from workers.
    Filling,
    /// Batch is full; waiting for the evaluator to score it.
    Evaluating,
    /// Results posted; workers are reading their individual scores.
    Consuming,
}

struct ChannelState {
    boards: Vec<Board>,
    terminals: Vec<bool>,
    capacity: usize,
    /// Evaluator outputs for the last evaluated batch (indexed by deposit order).
    results: Vec<EvalResult>,
    /// Workers that have consumed their result in the current Consuming phase.
    consumed: usize,
    /// Size of the batch currently in Consuming phase.
    batch_len: usize,
    /// Incremented every time `post_results` is called.
    generation: u64,
    phase: Phase,
    /// Set by `close()` after all deposits are done.
    closed: bool,
}

impl ChannelState {
    fn new(capacity: usize) -> Self {
        let cap = capacity.max(1);
        ChannelState {
            boards: Vec::with_capacity(cap),
            terminals: Vec::with_capacity(cap),
            capacity: cap,
            results: Vec::new(),
            consumed: 0,
            batch_len: 0,
            generation: 0,
            phase: Phase::Filling,
            closed: false,
        }
    }
}

/// Signal channel between MCTS worker threads and the evaluator thread.
///
/// # Protocol
///
/// **Workers** (N threads, typically rayon tasks):
/// 1. Call [`deposit`] — blocks until the channel is in the Filling phase,
///    then adds a board and returns `(slot, generation)`.
/// 2. When the deposit fills the batch, the evaluator is notified automatically.
/// 3. Call [`wait_for_result`] with the returned `(slot, generation)` — blocks
///    until the evaluator has posted scores for this generation.
/// 4. The last worker to read its result transitions the channel back to Filling.
///
/// **Evaluator** (main thread or a dedicated thread):
/// 1. Call [`wait_for_batch`] — blocks until the batch is full (or the channel
///    is closed).  Returns `(boards, terminals)` for the batch, or `None` if
///    closed with no pending leaves.
/// 2. Evaluate all boards (neural net forward pass in M5; parallel rollouts now).
/// 3. Call [`post_results`] with the scores — wakes all waiting workers.
/// 4. Repeat until `wait_for_batch` returns `None`.
///
/// **Shutdown**: call [`close`] after all deposits are done (e.g., after the
/// rayon `for_each` returns).  Any partial batch is flushed first.
pub struct BatchChannel {
    state: Mutex<ChannelState>,
    /// Evaluator sleeps here; workers wake it when the batch is full.
    eval_ready: Condvar,
    /// Workers sleep here for results; last consumer wakes workers waiting on
    /// Filling; evaluator not involved in this condvar.
    phase_change: Condvar,
}

impl BatchChannel {
    pub fn new(capacity: usize) -> Arc<Self> {
        Arc::new(BatchChannel {
            state: Mutex::new(ChannelState::new(capacity)),
            eval_ready: Condvar::new(),
            phase_change: Condvar::new(),
        })
    }

    /// Worker: add a leaf to the current batch.
    ///
    /// Blocks if the channel is not in Filling phase.  Returns `(slot, epoch)`
    /// which must be passed to [`wait_for_result`] to retrieve the score.
    /// Signals the evaluator when this deposit fills the batch.
    pub fn deposit(&self, board: Board, is_terminal: bool) -> (usize, u64) {
        let mut s = self
            .phase_change
            .wait_while(self.state.lock().unwrap(), |s| s.phase != Phase::Filling)
            .unwrap();

        let slot = s.boards.len();
        let epoch = s.generation;
        s.boards.push(board);
        s.terminals.push(is_terminal);

        if s.boards.len() >= s.capacity {
            s.phase = Phase::Evaluating;
            drop(s);
            self.eval_ready.notify_one();
        }
        (slot, epoch)
    }

    /// Worker: wait for the evaluator to score the leaf at `(slot, generation)`.
    ///
    /// Blocks until the generation counter advances past `generation`.
    /// The last worker to consume its result transitions the channel back to
    /// Filling and wakes any workers blocked on the next deposit.
    pub fn wait_for_eval_result(&self, slot: usize, generation: u64) -> EvalResult {
        let mut s = self
            .phase_change
            .wait_while(self.state.lock().unwrap(), |s| s.generation == generation)
            .unwrap();

        let result = s.results[slot].clone();
        s.consumed += 1;

        if s.consumed == s.batch_len {
            s.phase = Phase::Filling;
            s.consumed = 0;
            drop(s);
            self.phase_change.notify_all(); // wake workers blocked on next deposit
        }

        result
    }

    /// Compatibility helper returning only the scalar value.
    pub fn wait_for_result(&self, slot: usize, generation: u64) -> f32 {
        self.wait_for_eval_result(slot, generation).value
    }

    /// Evaluator: block until a batch is ready to evaluate.
    ///
    /// "Ready" means the batch is full **or** `timeout` has elapsed with at
    /// least one pending board (partial-batch flush).  The flush handles the
    /// unavoidable case where the number of in-flight workers is smaller than
    /// `capacity` and the batch can never fill on its own.
    ///
    /// Returns `(boards, terminals)` for the batch, or `None` when the channel
    /// is closed with no pending leaves.
    pub fn wait_for_batch(&self, timeout: std::time::Duration) -> Option<(Vec<Board>, Vec<bool>)> {
        loop {
            let (mut s, wait_result) = self
                .eval_ready
                .wait_timeout_while(self.state.lock().unwrap(), timeout, |s| {
                    s.phase != Phase::Evaluating && !s.closed
                })
                .unwrap();

            if s.phase == Phase::Evaluating {
                // Full batch signalled by a worker.
            } else if s.closed {
                if s.boards.is_empty() {
                    return None;
                }
                // Flush remainder before exiting.
                s.phase = Phase::Evaluating;
            } else if wait_result.timed_out() && !s.boards.is_empty() {
                // Workers are blocked waiting for results but couldn't fill
                // the batch (num_threads < batch_size, or last partial batch).
                // Flush whatever is pending so workers can make progress.
                s.phase = Phase::Evaluating;
            } else {
                continue; // spurious wakeup, nothing pending
            }

            let mut boards = Vec::with_capacity(s.capacity);
            let mut terminals = Vec::with_capacity(s.capacity);
            std::mem::swap(&mut s.boards, &mut boards);
            std::mem::swap(&mut s.terminals, &mut terminals);
            return Some((boards, terminals));
        }
    }

    /// Evaluator: post full eval results for the batch returned by [`wait_for_batch`].
    ///
    /// `results[i]` is the policy/value output for the board at slot `i`.
    /// All workers blocked in [`wait_for_result`] are notified.
    pub fn post_eval_results(&self, results: Vec<EvalResult>) {
        let mut s = self.state.lock().unwrap();
        s.batch_len = results.len();
        s.results = results;
        s.generation += 1;
        s.phase = Phase::Consuming;
        drop(s);
        self.phase_change.notify_all();
    }

    /// Compatibility helper posting value-only results.
    pub fn post_results(&self, values: Vec<f32>) {
        let results: Vec<EvalResult> = values
            .into_iter()
            .map(|value| EvalResult {
                policy_logits: Vec::new(),
                value,
            })
            .collect();
        self.post_eval_results(results);
    }

    /// Signal that no more deposits will be made.
    ///
    /// Flushes any partial batch so the evaluator can drain it, then marks
    /// the channel closed.  The evaluator will see `None` from
    /// [`wait_for_batch`] after draining the partial batch.
    pub fn close(&self) {
        let mut s = self.state.lock().unwrap();
        s.closed = true;
        if !s.boards.is_empty() {
            s.phase = Phase::Evaluating; // flush partial batch
            drop(s);
            self.eval_ready.notify_one();
        } else {
            drop(s);
            self.eval_ready.notify_one(); // wake evaluator to observe closed=true
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;

    fn dummy_leaf(leaf: NodeIdx) -> (NodeIdx, Board, Vec<NodeIdx>, bool) {
        (leaf, Board::startpos(), vec![leaf], false)
    }

    #[test]
    fn test_new_batch_is_empty() {
        let b = LeafBatch::new(4);
        assert!(b.is_empty());
        assert_eq!(b.len(), 0);
        assert_eq!(b.capacity(), 4);
    }

    #[test]
    fn test_capacity_clamped_to_one() {
        let b = LeafBatch::new(0);
        assert_eq!(b.capacity(), 1);
    }

    #[test]
    fn test_not_full_until_capacity_reached() {
        let mut b = LeafBatch::new(3);
        let (leaf, board, path, term) = dummy_leaf(1);
        b.push(leaf, board, path, term);
        assert!(!b.is_full());
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_full_at_capacity() {
        let mut b = LeafBatch::new(2);
        for i in 0..2u32 {
            let (leaf, board, path, term) = dummy_leaf(i);
            b.push(leaf, board, path, term);
        }
        assert!(b.is_full());
        assert_eq!(b.len(), 2);
    }

    #[test]
    fn test_drain_returns_all_leaves() {
        let mut b = LeafBatch::new(4);
        for i in 0..3u32 {
            let (leaf, board, path, term) = dummy_leaf(i);
            b.push(leaf, board, path, term);
        }
        let drained = b.drain();
        assert_eq!(drained.len(), 3);
        assert!(b.is_empty(), "batch must be empty after drain");
    }

    #[test]
    fn test_drain_leaves_map_to_correct_leaf_index() {
        let mut b = LeafBatch::new(4);
        for i in 0..3u32 {
            let (leaf, board, path, term) = dummy_leaf(i);
            b.push(leaf, board, path, term);
        }
        let drained = b.drain();
        for (i, p) in drained.iter().enumerate() {
            assert_eq!(p.leaf, i as NodeIdx);
        }
    }

    #[test]
    fn test_terminal_flag_preserved() {
        let mut b = LeafBatch::new(4);
        b.push(0, Board::startpos(), vec![0], true);
        b.push(1, Board::startpos(), vec![1], false);
        let drained = b.drain();
        assert!(drained[0].is_terminal);
        assert!(!drained[1].is_terminal);
    }

    #[test]
    fn test_reuse_after_drain() {
        let mut b = LeafBatch::new(2);
        for i in 0..2u32 {
            let (l, board, p, t) = dummy_leaf(i);
            b.push(l, board, p, t);
        }
        b.drain();
        assert!(b.is_empty());
        let (l, board, p, t) = dummy_leaf(99);
        b.push(l, board, p, t);
        assert_eq!(b.len(), 1);
        assert!(!b.is_full());
    }

    // --- BatchChannel tests ---
    //
    // Each test spawns a simple evaluator thread alongside one or more worker
    // threads so the condvar protocol is exercised under real concurrency.

    /// Spawn a trivial evaluator that returns 0.5 for every leaf.
    fn spawn_evaluator(ch: Arc<BatchChannel>) -> std::thread::JoinHandle<usize> {
        std::thread::spawn(move || {
            let mut batches = 0usize;
            loop {
                match ch.wait_for_batch(std::time::Duration::from_millis(5)) {
                    None => break,
                    Some((boards, _terminals)) => {
                        let values = vec![0.5_f32; boards.len()];
                        ch.post_results(values);
                        batches += 1;
                    }
                }
            }
            batches
        })
    }

    #[test]
    fn test_channel_single_batch_one_worker() {
        let ch = BatchChannel::new(1);
        let eval = spawn_evaluator(Arc::clone(&ch));

        let (slot, epoch) = ch.deposit(Board::startpos(), false);
        let v = ch.wait_for_result(slot, epoch);
        ch.close();

        assert!((v - 0.5).abs() < 1e-6);
        let batches = eval.join().unwrap();
        assert_eq!(batches, 1);
    }

    #[test]
    fn test_channel_batch_size_two_two_workers() {
        let ch = BatchChannel::new(2);
        let eval = spawn_evaluator(Arc::clone(&ch));

        let ch1 = Arc::clone(&ch);
        let w1 = std::thread::spawn(move || {
            let (s, g) = ch1.deposit(Board::startpos(), false);
            ch1.wait_for_result(s, g)
        });

        let ch2 = Arc::clone(&ch);
        let w2 = std::thread::spawn(move || {
            let (s, g) = ch2.deposit(Board::startpos(), false);
            ch2.wait_for_result(s, g)
        });

        let v1 = w1.join().unwrap();
        let v2 = w2.join().unwrap();
        ch.close();
        eval.join().unwrap();

        assert!((v1 - 0.5).abs() < 1e-6);
        assert!((v2 - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_channel_two_rounds() {
        // Two full batches of size 1, sequential.
        let ch = BatchChannel::new(1);
        let eval = spawn_evaluator(Arc::clone(&ch));

        for _ in 0..2 {
            let (s, g) = ch.deposit(Board::startpos(), false);
            ch.wait_for_result(s, g);
        }
        ch.close();
        let batches = eval.join().unwrap();
        assert_eq!(batches, 2);
    }

    #[test]
    fn test_channel_partial_batch_flushed_on_close() {
        // batch_size=4 but only 2 deposits before close.
        let ch = BatchChannel::new(4);

        // Evaluator: count how many boards arrive in total.
        let ch_eval = Arc::clone(&ch);
        let eval = std::thread::spawn(move || {
            let mut total = 0usize;
            loop {
                match ch_eval.wait_for_batch(std::time::Duration::from_millis(5)) {
                    None => break,
                    Some((boards, _)) => {
                        total += boards.len();
                        let v = vec![0.5_f32; boards.len()];
                        ch_eval.post_results(v);
                    }
                }
            }
            total
        });

        let (s1, g1) = ch.deposit(Board::startpos(), false);
        let (s2, g2) = ch.deposit(Board::startpos(), true);

        let ch2 = Arc::clone(&ch);
        let r1 = std::thread::spawn(move || ch2.wait_for_result(s1, g1));
        let ch3 = Arc::clone(&ch);
        let r2 = std::thread::spawn(move || ch3.wait_for_result(s2, g2));

        ch.close(); // flush the partial batch of 2

        r1.join().unwrap();
        r2.join().unwrap();
        let total = eval.join().unwrap();
        assert_eq!(total, 2, "partial batch must be flushed");
    }

    #[test]
    fn test_channel_slot_ordering_preserved() {
        // Deposits from a single thread arrive in order; slot 0 gets the first
        // board, slot 1 gets the second.  The evaluator returns distinct values
        // per slot so we can verify the mapping.
        let ch = BatchChannel::new(2);

        let ch_eval = Arc::clone(&ch);
        std::thread::spawn(move || {
            loop {
                match ch_eval.wait_for_batch(std::time::Duration::from_millis(5)) {
                    None => break,
                    Some((boards, _)) => {
                        // slot 0 → 1.0, slot 1 → -1.0
                        let v: Vec<f32> = (0..boards.len())
                            .map(|i| if i == 0 { 1.0 } else { -1.0 })
                            .collect();
                        ch_eval.post_results(v);
                    }
                }
            }
        });

        let (s0, g) = ch.deposit(Board::startpos(), false);
        let (s1, _) = ch.deposit(Board::startpos(), false);

        let v0 = ch.wait_for_result(s0, g);
        let v1 = ch.wait_for_result(s1, g);
        ch.close();

        assert!((v0 - 1.0).abs() < 1e-6, "slot 0 should get 1.0, got {v0}");
        assert!((v1 - -1.0).abs() < 1e-6, "slot 1 should get -1.0, got {v1}");
    }

    #[test]
    fn test_channel_eval_results_round_trip() {
        let ch = BatchChannel::new(2);

        let ch_eval = Arc::clone(&ch);
        std::thread::spawn(move || {
            loop {
                match ch_eval.wait_for_batch(std::time::Duration::from_millis(5)) {
                    None => break,
                    Some((boards, _)) => {
                        let results: Vec<EvalResult> = (0..boards.len())
                            .map(|i| EvalResult {
                                policy_logits: vec![i as f32, (i + 10) as f32],
                                value: if i == 0 { 1.0 } else { -1.0 },
                            })
                            .collect();
                        ch_eval.post_eval_results(results);
                    }
                }
            }
        });

        let (s0, g) = ch.deposit(Board::startpos(), false);
        let (s1, _) = ch.deposit(Board::startpos(), false);

        let r0 = ch.wait_for_eval_result(s0, g);
        let r1 = ch.wait_for_eval_result(s1, g);
        ch.close();

        assert_eq!(r0.policy_logits, vec![0.0, 10.0]);
        assert!((r0.value - 1.0).abs() < 1e-6);
        assert_eq!(r1.policy_logits, vec![1.0, 11.0]);
        assert!((r1.value - -1.0).abs() < 1e-6);
    }

    #[test]
    fn test_channel_close_empty_returns_none() {
        let ch = BatchChannel::new(4);
        let ch_eval = Arc::clone(&ch);
        let eval =
            std::thread::spawn(move || ch_eval.wait_for_batch(std::time::Duration::from_millis(5)));
        ch.close();
        let result = eval.join().unwrap();
        assert!(
            result.is_none(),
            "empty close must return None to evaluator"
        );
    }
}
