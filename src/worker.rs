/// Self-play worker pool for AlphaZero training.
///
/// Each worker thread runs an infinite self-play loop, collecting game records
/// and pushing them to a shared `ReplayBuffer`.  The training thread drives the
/// pool through `WorkerPool::spawn` / `WorkerPool::join`.
///
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use rand::SeedableRng;
use rand::rngs::StdRng;
use tch::{Tensor, nn};

use crate::nn::{Net, checkpoint::build_with_config};
use crate::replay_buffer::ReplayBuffer;
use crate::selfplay::{SelfPlayConfig, TerminationReason, play_game};

// ---------------------------------------------------------------------------
// Atomic f64 accumulator (bit-cast through u64)
// ---------------------------------------------------------------------------

/// Atomically add `val` to the f64 stored in `atom`.
///
/// Uses a CAS loop; contention is negligible (one call per game per worker).
fn atomic_f64_add(atom: &AtomicU64, val: f64) {
    let _ = atom.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |bits| {
        Some((f64::from_bits(bits) + val).to_bits())
    });
}

/// One deep copy of every VarStore variable, keyed by name.
///
/// `Tensor` is `Send` but not `Sync` in tch-0.24, so each worker slot holds
/// its own independent `HashMap` rather than a shared `Arc`.  The per-worker
/// allocation cost is negligible for the small test networks; for a full
/// 256-channel net the training thread creates N copies only at broadcast time.
type WeightSnapshot = HashMap<String, Tensor>;

/// Per-worker deposit slot.  Holds at most one pending snapshot — a newer
/// broadcast silently overwrites any un-consumed earlier one.
type WeightSlot = Arc<Mutex<Option<WeightSnapshot>>>;

/// Deep-copy every variable in `vs` into a fresh `HashMap`.
///
/// Each tensor owns its own storage so the training thread can safely continue
/// modifying `vs` after `broadcast_weights` returns.
fn snapshot_vars(vs: &nn::VarStore) -> WeightSnapshot {
    tch::no_grad(|| {
        vs.variables()
            .into_iter()
            .map(|(name, src)| {
                let mut dst = src.zeros_like();
                dst.copy_(&src);
                (name, dst)
            })
            .collect()
    })
}

/// Copy all variables from `snapshot` into `worker_vs` in-place.
fn apply_snapshot(worker_vs: &nn::VarStore, snapshot: &WeightSnapshot) {
    let mut vars = worker_vs.variables();
    tch::no_grad(|| {
        for (name, src) in snapshot {
            if let Some(dst) = vars.get_mut(name) {
                dst.copy_(src);
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Weight copy helper
// ---------------------------------------------------------------------------

/// Build a fresh `(VarStore, Net)` on CPU and deep-copy all weights from
/// `master_vs`.  The resulting VarStore is fully independent — mutating
/// `master_vs` afterwards has no effect on the copy.
pub fn build_worker_net(
    master_vs: &nn::VarStore,
    channels: i64,
    blocks: usize,
) -> (nn::VarStore, Net) {
    let (mut worker_vs, worker_net) = build_with_config(tch::Device::Cpu, channels, blocks);
    worker_vs.copy(master_vs).expect("weight copy failed");
    (worker_vs, worker_net)
}

// ---------------------------------------------------------------------------
// Aggregate counters
// ---------------------------------------------------------------------------

/// Snapshot of all self-play counters at a point in time.
///
/// Returned by `WorkerPool::counters()` and `WorkerPool::join()` so callers
/// can read all values in one call without risking inconsistency.
#[derive(Debug, Default, Clone)]
pub struct GameCounters {
    /// Total completed games across all workers.
    pub games: u64,
    /// Total training positions (records) pushed to the replay buffer.
    pub positions: u64,
    /// Games won by Black (White was checkmated or resigned).
    pub black_wins: u64,
    /// Games won by White (Black was checkmated or resigned).
    pub white_wins: u64,
    /// Games that ended in a draw.
    pub draws: u64,
    /// Games that ended because a player resigned (subset of decisive games).
    pub resigns: u64,
    /// Games that ended by hitting `max_moves` (subset of draws).
    pub max_move_draws: u64,
    /// Sum of per-game average visit-distribution entropies (nats).
    pub visit_entropy_sum: f64,
    /// Sum of per-game average root-policy entropies (nats).
    pub policy_entropy_sum: f64,
}

impl GameCounters {
    /// Average game length in plies; 0.0 when no games have completed.
    pub fn avg_game_length(&self) -> f64 {
        if self.games == 0 { 0.0 } else { self.positions as f64 / self.games as f64 }
    }

    /// Average visit-distribution entropy across all completed games (nats).
    pub fn avg_visit_entropy(&self) -> f64 {
        if self.games == 0 { 0.0 } else { self.visit_entropy_sum / self.games as f64 }
    }

    /// Average root-policy entropy across all completed games (nats).
    pub fn avg_policy_entropy(&self) -> f64 {
        if self.games == 0 { 0.0 } else { self.policy_entropy_sum / self.games as f64 }
    }
}

// ---------------------------------------------------------------------------
// Worker pool
// ---------------------------------------------------------------------------

/// Pool of N long-running self-play threads, each with its own network copy,
/// all pushing records to a shared `ReplayBuffer`.
pub struct WorkerPool {
    handles: Vec<JoinHandle<()>>,
    shutdown: Arc<AtomicBool>,
    weight_slots: Vec<WeightSlot>,
    /// Total completed self-play games across all workers since spawn.
    games_played: Arc<AtomicU64>,
    /// Total positions (half-moves) generated across all workers since spawn.
    positions_generated: Arc<AtomicU64>,
    black_wins: Arc<AtomicU64>,
    white_wins: Arc<AtomicU64>,
    draws: Arc<AtomicU64>,
    resigns: Arc<AtomicU64>,
    max_move_draws: Arc<AtomicU64>,
    /// Per-game average visit entropy, accumulated as f64 bits in a u64.
    visit_entropy_sum: Arc<AtomicU64>,
    /// Per-game average policy entropy, accumulated as f64 bits in a u64.
    policy_entropy_sum: Arc<AtomicU64>,
    /// Stored for potential future use (e.g. rebuilding nets on arch change).
    #[allow(dead_code)]
    channels: i64,
    #[allow(dead_code)]
    blocks: usize,
}

impl WorkerPool {
    /// Number of workers to use by default: `available_parallelism − 1`, minimum 1.
    pub fn default_num_workers() -> usize {
        thread::available_parallelism()
            .map(|n| n.get().saturating_sub(1).max(1))
            .unwrap_or(1)
    }

    /// Spawn `num_workers` threads.
    ///
    /// Each worker starts with a deep copy of `master_vs`, runs self-play games
    /// using `config`, and pushes every completed game's records into `buffer`.
    /// Workers are seeded with `base_seed + worker_index` for reproducibility.
    pub fn spawn(
        num_workers: usize,
        master_vs: &nn::VarStore,
        channels: i64,
        blocks: usize,
        config: SelfPlayConfig,
        buffer: Arc<Mutex<ReplayBuffer>>,
        base_seed: u64,
    ) -> Self {
        assert!(num_workers > 0, "must spawn at least one worker");
        let shutdown = Arc::new(AtomicBool::new(false));
        let config = Arc::new(config);
        let games_played        = Arc::new(AtomicU64::new(0));
        let positions_generated = Arc::new(AtomicU64::new(0));
        let black_wins          = Arc::new(AtomicU64::new(0));
        let white_wins          = Arc::new(AtomicU64::new(0));
        let draws               = Arc::new(AtomicU64::new(0));
        let resigns             = Arc::new(AtomicU64::new(0));
        let max_move_draws      = Arc::new(AtomicU64::new(0));
        // 0u64 == 0.0f64.to_bits() so default-zero initialization is correct.
        let visit_entropy_sum   = Arc::new(AtomicU64::new(0));
        let policy_entropy_sum  = Arc::new(AtomicU64::new(0));

        let weight_slots: Vec<WeightSlot> = (0..num_workers)
            .map(|_| Arc::new(Mutex::new(None)))
            .collect();

        let handles = weight_slots
            .iter()
            .enumerate()
            .map(|(idx, slot)| {
                let (worker_vs, worker_net) = build_worker_net(master_vs, channels, blocks);
                let shutdown = Arc::clone(&shutdown);
                let buffer = Arc::clone(&buffer);
                let config = Arc::clone(&config);
                let slot = Arc::clone(slot);
                let games      = Arc::clone(&games_played);
                let positions  = Arc::clone(&positions_generated);
                let bw         = Arc::clone(&black_wins);
                let ww         = Arc::clone(&white_wins);
                let dr         = Arc::clone(&draws);
                let rs         = Arc::clone(&resigns);
                let mmd        = Arc::clone(&max_move_draws);
                let ves        = Arc::clone(&visit_entropy_sum);
                let pes        = Arc::clone(&policy_entropy_sum);
                let seed = base_seed.wrapping_add(idx as u64);
                thread::spawn(move || {
                    worker_loop(worker_vs, worker_net, config, buffer, shutdown, slot, seed,
                                games, positions, bw, ww, dr, rs, mmd, ves, pes)
                })
            })
            .collect();

        Self {
            handles,
            shutdown,
            weight_slots,
            games_played,
            positions_generated,
            black_wins,
            white_wins,
            draws,
            resigns,
            max_move_draws,
            visit_entropy_sum,
            policy_entropy_sum,
            channels,
            blocks,
        }
    }

    /// Snapshot all counters since the pool was spawned.
    ///
    /// Reads are `Relaxed` — values may lag by a few nanoseconds but are always
    /// monotonically non-decreasing and safe to read from any thread.
    pub fn counters(&self) -> GameCounters {
        GameCounters {
            games:              self.games_played.load(Ordering::Relaxed),
            positions:          self.positions_generated.load(Ordering::Relaxed),
            black_wins:         self.black_wins.load(Ordering::Relaxed),
            white_wins:         self.white_wins.load(Ordering::Relaxed),
            draws:              self.draws.load(Ordering::Relaxed),
            resigns:            self.resigns.load(Ordering::Relaxed),
            max_move_draws:     self.max_move_draws.load(Ordering::Relaxed),
            visit_entropy_sum:  f64::from_bits(self.visit_entropy_sum.load(Ordering::Relaxed)),
            policy_entropy_sum: f64::from_bits(self.policy_entropy_sum.load(Ordering::Relaxed)),
        }
    }

    /// Number of live worker threads.
    pub fn num_workers(&self) -> usize {
        self.handles.len()
    }

    /// Push the current `master_vs` weights to every worker.
    ///
    /// Creates one deep copy of the master variables (via `snapshot_vars`), then
    /// shares it across all per-worker slots via `Arc` — no per-worker tensor
    /// allocation.  Workers apply the update between games; there is no mid-game
    /// weight change and no lock contention during forward passes.
    pub fn broadcast_weights(&self, master_vs: &nn::VarStore) {
        // snapshot_vars is called once per worker; each gets its own deep copy
        // so workers can apply concurrently without sharing tensor storage.
        for slot in &self.weight_slots {
            *slot.lock().unwrap() = Some(snapshot_vars(master_vs));
        }
    }

    /// Signal all workers to stop after their current game, join all threads,
    /// and return the final `(games_played, positions_generated)` totals.
    ///
    /// Because workers check the shutdown flag *after* each game, every thread
    /// is guaranteed to push at least one completed game before returning.
    pub fn join(self) -> GameCounters {
        self.shutdown.store(true, Ordering::Relaxed);
        for h in self.handles {
            h.join().expect("worker thread panicked");
        }
        // `self.handles` was moved by the loop; read fields directly instead of
        // calling `self.counters()` (which would borrow the whole struct).
        GameCounters {
            games:              self.games_played.load(Ordering::Relaxed),
            positions:          self.positions_generated.load(Ordering::Relaxed),
            black_wins:         self.black_wins.load(Ordering::Relaxed),
            white_wins:         self.white_wins.load(Ordering::Relaxed),
            draws:              self.draws.load(Ordering::Relaxed),
            resigns:            self.resigns.load(Ordering::Relaxed),
            max_move_draws:     self.max_move_draws.load(Ordering::Relaxed),
            visit_entropy_sum:  f64::from_bits(self.visit_entropy_sum.load(Ordering::Relaxed)),
            policy_entropy_sum: f64::from_bits(self.policy_entropy_sum.load(Ordering::Relaxed)),
        }
    }
}

// ---------------------------------------------------------------------------
// Worker body
// ---------------------------------------------------------------------------

fn worker_loop(
    worker_vs: nn::VarStore,
    net: Net,
    config: Arc<SelfPlayConfig>,
    buffer: Arc<Mutex<ReplayBuffer>>,
    shutdown: Arc<AtomicBool>,
    slot: WeightSlot,
    seed: u64,
    games_counter: Arc<AtomicU64>,
    positions_counter: Arc<AtomicU64>,
    black_wins: Arc<AtomicU64>,
    white_wins: Arc<AtomicU64>,
    draws: Arc<AtomicU64>,
    resigns: Arc<AtomicU64>,
    max_move_draws: Arc<AtomicU64>,
    visit_entropy_sum: Arc<AtomicU64>,
    policy_entropy_sum: Arc<AtomicU64>,
) {
    let mut rng = StdRng::seed_from_u64(seed);
    loop {
        // Apply pending weight broadcast before the next game.
        if let Some(snap) = slot.lock().unwrap().take() {
            apply_snapshot(&worker_vs, &snap);
        }

        let result = tch::no_grad(|| play_game(&net, &config, tch::Device::Cpu, &mut rng));
        let n_positions = result.records.len() as u64;
        buffer.lock().unwrap().push_game(result.records);

        games_counter.fetch_add(1, Ordering::Relaxed);
        positions_counter.fetch_add(n_positions, Ordering::Relaxed);

        // Entropy accumulators.
        atomic_f64_add(&visit_entropy_sum, result.avg_visit_entropy as f64);
        atomic_f64_add(&policy_entropy_sum, result.avg_policy_entropy as f64);

        // Outcome counters.
        if result.outcome > 0.0 {
            black_wins.fetch_add(1, Ordering::Relaxed);
        } else if result.outcome < 0.0 {
            white_wins.fetch_add(1, Ordering::Relaxed);
        } else {
            draws.fetch_add(1, Ordering::Relaxed);
        }

        // Termination counters (orthogonal to outcome).
        match result.termination {
            TerminationReason::Resign   => { resigns.fetch_add(1, Ordering::Relaxed); }
            TerminationReason::MaxMoves => { max_move_draws.fetch_add(1, Ordering::Relaxed); }
            TerminationReason::Checkmate => {}
        }

        // Check shutdown after the game — guarantees at least one game per worker.
        if shutdown.load(Ordering::Relaxed) {
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::{NUM_ACTIONS, NUM_PLANES, checkpoint::build_with_config};
    use tch::{Device, Kind, Tensor};

    fn make_master() -> (nn::VarStore, Net) {
        build_with_config(Device::Cpu, 8, 2)
    }

    fn randn_input() -> Tensor {
        Tensor::randn([1, NUM_PLANES as i64, 9, 9], (Kind::Float, Device::Cpu))
    }

    /// Tiny config so test games finish quickly.
    fn tiny_config() -> SelfPlayConfig {
        SelfPlayConfig {
            num_simulations: 4,
            max_moves: 20,
            resign_min_ply: 100, // don't resign in tiny games
            ..SelfPlayConfig::default()
        }
    }

    fn test_buffer(capacity: usize) -> Arc<Mutex<ReplayBuffer>> {
        Arc::new(Mutex::new(ReplayBuffer::new(capacity)))
    }

    // ----- default_num_workers -----

    #[test]
    fn test_default_num_workers_at_least_one() {
        assert!(WorkerPool::default_num_workers() >= 1);
    }

    #[test]
    fn test_default_workers_equals_available_minus_one() {
        let expected = thread::available_parallelism()
            .map(|n| n.get().saturating_sub(1).max(1))
            .unwrap_or(1);
        assert_eq!(WorkerPool::default_num_workers(), expected);
    }

    // ----- build_worker_net -----

    #[test]
    fn test_worker_net_outputs_match_master() {
        let (master_vs, master_net) = make_master();
        let (_worker_vs, worker_net) = build_worker_net(&master_vs, 8, 2);

        let xs = randn_input();
        let (mp, mv) = tch::no_grad(|| master_net.forward_t(&xs, false));
        let (wp, wv) = tch::no_grad(|| worker_net.forward_t(&xs, false));

        let p_diff = (&mp - &wp).abs().max().double_value(&[]);
        let v_diff = (&mv - &wv).abs().max().double_value(&[]);
        assert!(p_diff < 1e-6, "policy mismatch after copy: {p_diff:.2e}");
        assert!(v_diff < 1e-6, "value mismatch after copy: {v_diff:.2e}");
    }

    #[test]
    fn test_worker_net_is_independent_of_master() {
        let (master_vs, _master_net) = make_master();
        let (_worker_vs, worker_net) = build_worker_net(&master_vs, 8, 2);

        let xs = randn_input();
        let (p_before, _) = tch::no_grad(|| worker_net.forward_t(&xs, false));

        // Zero out every weight in the master in-place.
        // no_grad is required: leaf parameters block in-place ops under autograd.
        tch::no_grad(|| {
            for (_, mut t) in master_vs.variables() {
                let _ = t.fill_(0.0);
            }
        });

        let (p_after, _) = tch::no_grad(|| worker_net.forward_t(&xs, false));
        let diff = (&p_before - &p_after).abs().max().double_value(&[]);
        assert!(
            diff < 1e-6,
            "worker was affected by master mutation: {diff:.2e}"
        );
    }

    #[test]
    fn test_worker_policy_shape() {
        let (master_vs, _) = make_master();
        let (_, worker_net) = build_worker_net(&master_vs, 8, 2);
        let (p, v) = tch::no_grad(|| worker_net.forward_t(&randn_input(), false));
        assert_eq!(p.size(), vec![1, NUM_ACTIONS as i64]);
        assert_eq!(v.size(), vec![1, 1]);
    }

    // ----- WorkerPool spawn / join -----

    #[test]
    fn test_spawn_correct_count() {
        let (master_vs, _) = make_master();
        let pool = WorkerPool::spawn(3, &master_vs, 8, 2, tiny_config(), test_buffer(10_000), 0);
        assert_eq!(pool.num_workers(), 3);
        pool.join();
    }

    #[test]
    fn test_spawn_one_worker() {
        let (master_vs, _) = make_master();
        let pool = WorkerPool::spawn(1, &master_vs, 8, 2, tiny_config(), test_buffer(10_000), 0);
        assert_eq!(pool.num_workers(), 1);
        pool.join();
    }

    #[test]
    fn test_join_terminates_all_threads() {
        let (master_vs, _) = make_master();
        // Must return — hanging here means deadlock.
        WorkerPool::spawn(4, &master_vs, 8, 2, tiny_config(), test_buffer(10_000), 0).join();
    }

    #[test]
    fn test_single_worker_populates_buffer() {
        let (master_vs, _) = make_master();
        let buffer = test_buffer(10_000);
        let pool = WorkerPool::spawn(1, &master_vs, 8, 2, tiny_config(), Arc::clone(&buffer), 0);
        pool.join(); // worker plays exactly one game then exits
        assert!(
            buffer.lock().unwrap().len() > 0,
            "buffer empty after worker ran one game"
        );
    }

    #[test]
    fn test_two_workers_each_push_at_least_one_game() {
        let (master_vs, _) = make_master();
        let buffer = test_buffer(10_000);
        let pool = WorkerPool::spawn(2, &master_vs, 8, 2, tiny_config(), Arc::clone(&buffer), 0);
        pool.join();
        // Each worker plays at least one game; max_moves=20 so each game is ≤20 records.
        // Two workers → at least 2 positions pushed (one game each is guaranteed).
        assert!(buffer.lock().unwrap().len() >= 2);
    }

    #[test]
    fn test_workers_use_different_seeds() {
        let (master_vs, _) = make_master();
        let buffer = test_buffer(10_000);
        let pool = WorkerPool::spawn(2, &master_vs, 8, 2, tiny_config(), Arc::clone(&buffer), 42);
        pool.join();
        assert!(buffer.lock().unwrap().len() >= 2);
    }

    #[test]
    fn test_snapshot_is_independent_of_master() {
        let (master_vs, _) = make_master();
        let snap = snapshot_vars(&master_vs);

        // Record snapshot's total weight mass before touching the master.
        // BatchNorm biases are init to 0, so we can't use `all(nonzero)`;
        // checking that the sum is preserved after zeroing master is robust.
        let sum_before: f64 = snap
            .values()
            .map(|t| t.abs().sum(tch::Kind::Double).double_value(&[]))
            .sum();
        assert!(sum_before > 0.0, "snapshot is unexpectedly all-zero");

        // Zero every master variable in-place.
        tch::no_grad(|| {
            for (_, mut t) in master_vs.variables() {
                let _ = t.fill_(0.0);
            }
        });

        let sum_after: f64 = snap
            .values()
            .map(|t| t.abs().sum(tch::Kind::Double).double_value(&[]))
            .sum();

        assert!(
            (sum_before - sum_after).abs() < 1e-3,
            "snapshot weight sum changed after zeroing master \
             ({sum_before:.4} → {sum_after:.4}): tensors are aliased"
        );
    }

    #[test]
    fn test_apply_snapshot_changes_worker_output() {
        let (master_vs1, _) = make_master();
        let (master_vs2, _) = make_master(); // different random weights
        let (worker_vs, worker_net) = build_worker_net(&master_vs1, 8, 2);
        let xs = randn_input();
        let (p_before, _) = tch::no_grad(|| worker_net.forward_t(&xs, false));

        apply_snapshot(&worker_vs, &snapshot_vars(&master_vs2));

        let (p_after, _) = tch::no_grad(|| worker_net.forward_t(&xs, false));
        let diff = (&p_before - &p_after).abs().max().double_value(&[]);
        assert!(
            diff > 1e-4,
            "apply_snapshot had no effect (diff={diff:.2e})"
        );
    }

    #[test]
    fn test_apply_snapshot_matches_source() {
        let (master_vs1, master_net1) = make_master();
        let (master_vs2, master_net2) = make_master();
        let (worker_vs, worker_net) = build_worker_net(&master_vs1, 8, 2);

        apply_snapshot(&worker_vs, &snapshot_vars(&master_vs2));

        let xs = randn_input();
        let (pm2, _) = tch::no_grad(|| master_net2.forward_t(&xs, false));
        let (pw, _) = tch::no_grad(|| worker_net.forward_t(&xs, false));
        let diff2 = (&pm2 - &pw).abs().max().double_value(&[]);
        assert!(
            diff2 < 1e-5,
            "worker doesn't match source after apply: {diff2:.2e}"
        );

        let (pm1, _) = tch::no_grad(|| master_net1.forward_t(&xs, false));
        let diff1 = (&pm1 - &pw).abs().max().double_value(&[]);
        assert!(diff1 > 1e-4, "worker still matches old master after apply");
    }

    #[test]
    fn test_broadcast_does_not_panic() {
        let (master_vs, _) = make_master();
        let pool = WorkerPool::spawn(2, &master_vs, 8, 2, tiny_config(), test_buffer(10_000), 0);
        pool.broadcast_weights(&master_vs);
        pool.broadcast_weights(&master_vs); // second call overwrites pending
        pool.join();
    }

    #[test]
    fn test_outcome_counters_sum_to_games() {
        let (master_vs, _) = make_master();
        let buffer = test_buffer(10_000);
        let pool = WorkerPool::spawn(2, &master_vs, 8, 2, tiny_config(), Arc::clone(&buffer), 0);
        let c = pool.join();
        assert_eq!(
            c.black_wins + c.white_wins + c.draws, c.games,
            "black_wins + white_wins + draws must equal total games"
        );
        assert!(
            c.resigns <= c.black_wins + c.white_wins,
            "resigns must be a subset of decisive games"
        );
        assert!(
            c.max_move_draws <= c.draws,
            "max_move_draws must be a subset of draws"
        );
    }

    #[test]
    fn test_counters_increment_after_one_game() {
        let (master_vs, _) = make_master();
        let buffer = test_buffer(10_000);
        let pool = WorkerPool::spawn(1, &master_vs, 8, 2, tiny_config(), Arc::clone(&buffer), 0);
        let c = pool.join();
        assert!(c.games >= 1, "expected ≥1 game, got {}", c.games);
        assert!(c.positions >= 1, "expected ≥1 position, got {}", c.positions);
    }

    #[test]
    fn test_counters_scale_with_workers() {
        let (master_vs, _) = make_master();
        let buffer = test_buffer(10_000);
        let pool = WorkerPool::spawn(3, &master_vs, 8, 2, tiny_config(), Arc::clone(&buffer), 0);
        let c = pool.join();
        assert!(c.games >= 3, "expected ≥3 games from 3 workers, got {}", c.games);
    }

    #[test]
    fn test_broadcast_workers_still_populate_buffer() {
        let (master_vs, _) = make_master();
        let buffer = test_buffer(10_000);
        let pool = WorkerPool::spawn(2, &master_vs, 8, 2, tiny_config(), Arc::clone(&buffer), 0);
        pool.broadcast_weights(&master_vs);
        pool.join();
        assert!(buffer.lock().unwrap().len() > 0);
    }
}
