/// Batch accumulator for leaf evaluation.
///
/// Instead of evaluating each MCTS leaf immediately (one rollout / one network
/// call per leaf), workers deposit their leaves here.  Once the batch reaches
/// `capacity`, the caller evaluates all non-terminal positions together — one
/// GPU forward pass in M5, parallel rollouts now — and backprops all results.
use crate::board::Board;
use super::NodeIdx;

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

// ---------------------------------------------------------------------------
// LeafBatch
// ---------------------------------------------------------------------------

/// Fixed-capacity accumulator for leaf positions.
pub struct LeafBatch {
    pending:  Vec<PendingLeaf>,
    capacity: usize,
}

impl LeafBatch {
    /// Create a batch that fires when `capacity` leaves have been deposited.
    /// Capacity is clamped to at least 1.
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.max(1);
        LeafBatch {
            pending:  Vec::with_capacity(cap),
            capacity: cap,
        }
    }

    /// Deposit one leaf into the batch.
    pub fn push(
        &mut self,
        leaf:        NodeIdx,
        board:       Board,
        vl_path:     Vec<NodeIdx>,
        is_terminal: bool,
    ) {
        self.pending.push(PendingLeaf { leaf, board, vl_path, is_terminal });
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
}
