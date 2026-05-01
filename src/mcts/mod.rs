/// Monte Carlo Tree Search
///
/// Nodes are stored in a flat arena (`Vec<Node>`) and reference their
/// children by index, avoiding pointer indirection and keeping allocations
/// contiguous.  All values are in [−1, 1]: +1 = win for the side that just
/// moved into this node, −1 = loss.
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
}
