/// Monte Carlo Tree Search
///
/// Nodes are stored in a flat arena (`Vec<Node>`) and reference their
/// children by index, avoiding pointer indirection and keeping allocations
/// contiguous.  All values are in [−1, 1]: +1 = win for the side that just
/// moved into this node, −1 = loss.
pub mod search;
pub mod params;
pub mod batch;

pub use params::MctsConfig;
pub use batch::{LeafBatch, PendingLeaf};

use crate::types::Move;

/// Index into the MCTS arena.
pub type NodeIdx = u32;

/// Sentinel value meaning "no parent" — set only on the root node.
pub const NO_PARENT: NodeIdx = u32::MAX;

/// A single node in the MCTS search tree.
#[derive(Debug)]
pub struct Node {
    /// Move that led to this position from the parent (`None` for the root).
    pub mv: Option<Move>,
    /// Prior probability P(s, a) from the policy network (or 1/N uniform).
    pub prior: f32,
    /// Visit count N(s, a) — incremented during backpropagation.
    pub visit_count: u32,
    /// Accumulated value sum W(s, a) — updated during backpropagation.
    pub total_value: f32,
    /// Indices of child nodes in the arena.  Empty = unexpanded leaf.
    pub children: Vec<NodeIdx>,
    /// Index of the parent node in the arena (`NO_PARENT` for the root).
    pub parent: NodeIdx,
}

impl Node {
    /// Create a new unexpanded leaf node.
    pub fn new(mv: Option<Move>, prior: f32, parent: NodeIdx) -> Self {
        Node {
            mv,
            prior,
            visit_count: 0,
            total_value: 0.0,
            children: Vec::new(),
            parent,
        }
    }

    /// Mean action value Q(s, a) = W / N.
    /// Returns 0.0 for unvisited nodes to avoid division by zero.
    #[inline]
    pub fn mean_value(&self) -> f32 {
        if self.visit_count == 0 {
            0.0
        } else {
            self.total_value / self.visit_count as f32
        }
    }

    /// True when this node has never been expanded (no children yet).
    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Arena allocator
// ---------------------------------------------------------------------------

/// Flat pool of MCTS nodes referenced by `NodeIdx`.
///
/// All nodes for a single search live in one contiguous allocation.  Calling
/// `clear()` between searches resets the node count to zero but keeps the
/// backing memory, so repeated searches avoid repeated heap allocations.
pub struct Arena {
    nodes: Vec<Node>,
}

impl Arena {
    /// Create an arena pre-allocated for `capacity` nodes.
    pub fn new(capacity: usize) -> Self {
        Arena {
            nodes: Vec::with_capacity(capacity),
        }
    }

    /// Allocate `node` in the arena and return its index.
    /// Panics if the index would overflow `NodeIdx` (u32::MAX nodes).
    pub fn alloc(&mut self, node: Node) -> NodeIdx {
        let idx = self.nodes.len() as NodeIdx;
        self.nodes.push(node);
        idx
    }

    /// Immutable access to a node by index.
    #[inline]
    pub fn get(&self, idx: NodeIdx) -> &Node {
        &self.nodes[idx as usize]
    }

    /// Mutable access to a node by index.
    #[inline]
    pub fn get_mut(&mut self, idx: NodeIdx) -> &mut Node {
        &mut self.nodes[idx as usize]
    }

    /// The root node is always allocated first and lives at index 0.
    #[inline]
    pub fn root(&self) -> NodeIdx {
        0
    }

    /// Reset the arena for a new search.
    /// Drops all nodes but retains the backing allocation.
    pub fn clear(&mut self) {
        self.nodes.clear();
    }

    /// Number of nodes currently in the arena.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Select the child of `parent_idx` with the highest PUCT score.
    ///
    /// Uses the AlphaZero PUCT formula:
    ///
    /// ```text
    /// PUCT(s, a) = −Q(s, a) + c_puct · P(s, a) · √N(s) / (1 + N(s, a))
    /// ```
    ///
    /// **Sign convention**: every node stores its accumulated value from the
    /// perspective of the *side to move at that node*.  Backpropagation flips
    /// sign at each level.  Therefore a child's `mean_value()` is the
    /// *opponent's* perspective; the parent must negate it to get its own
    /// perspective before adding the exploration bonus.
    ///
    /// Returns `None` if the parent has no children (unexpanded leaf).
    pub fn best_child(&self, parent_idx: NodeIdx, c_puct: f32) -> Option<NodeIdx> {
        let parent = self.get(parent_idx);
        if parent.children.is_empty() {
            return None;
        }
        let sqrt_n = (parent.visit_count as f32).sqrt();
        parent
            .children
            .iter()
            .copied()
            .max_by(|&a, &b| {
                let sa = self.puct_score(a, sqrt_n, c_puct);
                let sb = self.puct_score(b, sqrt_n, c_puct);
                sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    #[inline]
    fn puct_score(&self, idx: NodeIdx, sqrt_parent_n: f32, c_puct: f32) -> f32 {
        let node = self.get(idx);
        // Negate Q: child stores value from its own (opponent's) perspective.
        let q = -node.mean_value();
        let u = c_puct * node.prior * sqrt_parent_n / (1.0 + node.visit_count as f32);
        q + u
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_new_defaults() {
        let node = Node::new(None, 1.0, NO_PARENT);
        assert_eq!(node.visit_count, 0);
        assert_eq!(node.total_value, 0.0);
        assert!(node.children.is_empty());
        assert_eq!(node.parent, NO_PARENT);
        assert!(node.mv.is_none());
    }

    #[test]
    fn test_node_prior_stored() {
        let node = Node::new(None, 0.35, NO_PARENT);
        assert!((node.prior - 0.35).abs() < 1e-6);
    }

    #[test]
    fn test_mean_value_unvisited() {
        let node = Node::new(None, 0.5, NO_PARENT);
        assert_eq!(node.mean_value(), 0.0);
    }

    #[test]
    fn test_mean_value_after_backprop() {
        let mut node = Node::new(None, 0.5, NO_PARENT);
        node.visit_count = 4;
        node.total_value = 3.0;
        assert!((node.mean_value() - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_is_leaf_new_node() {
        let node = Node::new(None, 1.0, NO_PARENT);
        assert!(node.is_leaf());
    }

    #[test]
    fn test_is_leaf_after_adding_child() {
        let mut node = Node::new(None, 1.0, NO_PARENT);
        node.children.push(1);
        assert!(!node.is_leaf());
    }

    // --- Arena tests ---

    #[test]
    fn test_arena_alloc_sequential_indices() {
        let mut arena = Arena::new(8);
        let i0 = arena.alloc(Node::new(None, 1.0, NO_PARENT));
        let i1 = arena.alloc(Node::new(None, 0.5, i0));
        let i2 = arena.alloc(Node::new(None, 0.5, i0));
        assert_eq!(i0, 0);
        assert_eq!(i1, 1);
        assert_eq!(i2, 2);
    }

    #[test]
    fn test_arena_get_returns_correct_node() {
        let mut arena = Arena::new(4);
        arena.alloc(Node::new(None, 1.0, NO_PARENT));
        arena.alloc(Node::new(None, 0.25, 0));
        assert!((arena.get(1).prior - 0.25).abs() < 1e-6);
        assert_eq!(arena.get(1).parent, 0);
    }

    #[test]
    fn test_arena_get_mut_modifies_node() {
        let mut arena = Arena::new(4);
        arena.alloc(Node::new(None, 1.0, NO_PARENT));
        arena.get_mut(0).visit_count = 7;
        assert_eq!(arena.get(0).visit_count, 7);
    }

    #[test]
    fn test_arena_root_is_zero() {
        let arena = Arena::new(4);
        assert_eq!(arena.root(), 0);
    }

    #[test]
    fn test_arena_clear_resets_len() {
        let mut arena = Arena::new(8);
        arena.alloc(Node::new(None, 1.0, NO_PARENT));
        arena.alloc(Node::new(None, 0.5, 0));
        assert_eq!(arena.len(), 2);
        arena.clear();
        assert_eq!(arena.len(), 0);
        assert!(arena.is_empty());
    }

    #[test]
    fn test_arena_clear_keeps_capacity() {
        let mut arena = Arena::new(64);
        for _ in 0..10 {
            arena.alloc(Node::new(None, 1.0, NO_PARENT));
        }
        let cap_before = arena.nodes.capacity();
        arena.clear();
        assert_eq!(arena.nodes.capacity(), cap_before);
    }

    #[test]
    fn test_arena_reuse_after_clear() {
        let mut arena = Arena::new(8);
        arena.alloc(Node::new(None, 1.0, NO_PARENT));
        arena.clear();
        let idx = arena.alloc(Node::new(None, 0.9, NO_PARENT));
        assert_eq!(idx, 0); // index resets to 0 after clear
        assert!((arena.get(0).prior - 0.9).abs() < 1e-6);
    }

    // --- PUCT / best_child tests ---

    // Build a root + two children, returning (arena, root, child_a, child_b).
    fn two_child_arena(
        prior_a: f32, visits_a: u32, value_a: f32,
        prior_b: f32, visits_b: u32, value_b: f32,
        parent_visits: u32,
    ) -> (Arena, NodeIdx, NodeIdx, NodeIdx) {
        let mut arena = Arena::new(8);
        let root = arena.alloc(Node::new(None, 1.0, NO_PARENT));
        let ca   = arena.alloc(Node::new(None, prior_a, root));
        let cb   = arena.alloc(Node::new(None, prior_b, root));
        arena.get_mut(ca).visit_count  = visits_a;
        arena.get_mut(ca).total_value  = value_a;
        arena.get_mut(cb).visit_count  = visits_b;
        arena.get_mut(cb).total_value  = value_b;
        arena.get_mut(root).visit_count = parent_visits;
        arena.get_mut(root).children.push(ca);
        arena.get_mut(root).children.push(cb);
        (arena, root, ca, cb)
    }

    #[test]
    fn test_best_child_none_for_leaf() {
        let mut arena = Arena::new(4);
        let root = arena.alloc(Node::new(None, 1.0, NO_PARENT));
        assert!(arena.best_child(root, 1.0).is_none());
    }

    #[test]
    fn test_best_child_prefers_higher_prior_when_unvisited() {
        // Both children unvisited: the one with higher prior should win.
        let (arena, root, _ca, cb) =
            two_child_arena(0.3, 0, 0.0, 0.7, 0, 0.0, 5);
        assert_eq!(arena.best_child(root, 1.0), Some(cb));
    }

    #[test]
    fn test_best_child_exploitation_beats_prior_at_low_c_puct() {
        // Child A: prior=0.3, visits=10, total_value=−8.0 → Q=−0.8 from A's
        //   perspective = good for the parent (−Q = +0.8).
        // Child B: prior=0.7, visits=0, Q=0.0 (high prior, unvisited).
        // At low c_puct the exploitation term (−Q) of A should dominate.
        let (arena, root, ca, _cb) =
            two_child_arena(0.3, 10, -8.0,  0.7, 0, 0.0,  10);
        assert_eq!(arena.best_child(root, 0.1), Some(ca));
    }

    #[test]
    fn test_best_child_exploration_beats_q_at_high_c_puct() {
        // Same position — at high c_puct B's larger prior wins despite A's
        // good exploitation score.
        let (arena, root, _ca, cb) =
            two_child_arena(0.3, 10, -8.0,  0.7, 0, 0.0,  10);
        assert_eq!(arena.best_child(root, 5.0), Some(cb));
    }

    #[test]
    fn test_puct_score_manual() {
        // Single child: prior=0.6, visits=4, total_value=2.0 → Q=0.5 from
        //   child's perspective.
        // Parent visits=9 → sqrt_n=3.0, c_puct=1.0.
        // score = −Q + U = −0.5 + (1.0 * 0.6 * 3.0 / 5) = −0.5 + 0.36 = −0.14
        let mut arena = Arena::new(4);
        let root = arena.alloc(Node::new(None, 1.0, NO_PARENT));
        let child = arena.alloc(Node::new(None, 0.6, root));
        arena.get_mut(child).visit_count = 4;
        arena.get_mut(child).total_value = 2.0;
        arena.get_mut(root).visit_count = 9;
        arena.get_mut(root).children.push(child);

        let score = arena.puct_score(child, 3.0, 1.0);
        assert!((score - (-0.14)).abs() < 1e-5, "expected -0.14, got {score}");
    }
}
