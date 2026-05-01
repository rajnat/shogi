/// MCTS search phases: selection, expansion, evaluation, backpropagation.
use crate::board::Board;
use crate::movegen::generate_legal_moves;
use crate::moves::{make_move_full, UndoState};
use crate::types::Move;
use super::{Arena, Node, NodeIdx};

// ---------------------------------------------------------------------------
// Selection
// ---------------------------------------------------------------------------

/// Walk from `root` down the tree using PUCT until an unexpanded leaf is
/// reached, applying each move to `board` along the way.
///
/// Returns `(leaf, undo_stack)`.  The caller is responsible for reversing
/// `undo_stack` to restore the board to its state before `select` was called.
pub fn select(
    arena: &Arena,
    root: NodeIdx,
    board: &mut Board,
    c_puct: f32,
) -> (NodeIdx, Vec<(Move, UndoState)>) {
    let mut node_idx = root;
    let mut undo_stack: Vec<(Move, UndoState)> = Vec::new();

    loop {
        if arena.get(node_idx).is_leaf() {
            break;
        }
        // best_child returns None only for leaves, already guarded above.
        let child_idx = arena
            .best_child(node_idx, c_puct)
            .expect("non-leaf must have a best child");

        let mv = arena
            .get(child_idx)
            .mv
            .expect("every non-root node must carry the move that created it");

        let undo = make_move_full(board, mv);
        undo_stack.push((mv, undo));
        node_idx = child_idx;
    }

    (node_idx, undo_stack)
}

// ---------------------------------------------------------------------------
// Expansion
// ---------------------------------------------------------------------------

/// Expand `leaf` by generating all legal moves from `board` and allocating
/// one child node per move with uniform prior probabilities.
///
/// Returns `true` if at least one child was created (non-terminal position).
/// Returns `false` if there are no legal moves (checkmate / stalemate) —
/// the caller should treat the leaf as a terminal and score it directly.
///
/// Prior probabilities are set to `1 / N` (uniform) as a placeholder until
/// the policy network (M5) supplies real values.
pub fn expand(arena: &mut Arena, leaf: NodeIdx, board: &mut Board) -> bool {
    let mut moves = Vec::new();
    generate_legal_moves(board, &mut moves);

    if moves.is_empty() {
        return false;
    }

    let prior = 1.0 / moves.len() as f32;

    for mv in moves {
        let child_idx = arena.alloc(Node::new(Some(mv), prior, leaf));
        arena.get_mut(leaf).children.push(child_idx);
    }

    true
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    use crate::mcts::{Node, NO_PARENT};
    use crate::movegen::generate_legal_moves;
    use crate::moves::unmake_move_full;

    /// Build a root node in a fresh arena, return (arena, root_idx).
    fn make_root(board: &Board) -> (Arena, NodeIdx) {
        let _ = board; // board not needed for root alloc, kept for clarity
        let mut arena = Arena::new(64);
        let root = arena.alloc(Node::new(None, 1.0, NO_PARENT));
        (arena, root)
    }

    #[test]
    fn test_select_root_is_leaf_returns_root() {
        let board = Board::startpos();
        let (arena, root) = make_root(&board);
        let mut b = board.clone();
        let (leaf, undo_stack) = select(&arena, root, &mut b, 1.0);
        assert_eq!(leaf, root);
        assert!(undo_stack.is_empty());
    }

    #[test]
    fn test_select_board_unchanged_when_root_is_leaf() {
        let board = Board::startpos();
        let (arena, root) = make_root(&board);
        let mut b = board.clone();
        select(&arena, root, &mut b, 1.0);
        assert_eq!(b.hash, board.hash);
    }

    #[test]
    fn test_select_descends_to_only_child() {
        // Expand the root with one real legal move, then select.
        let board = Board::startpos();
        let (mut arena, root) = make_root(&board);

        let mut moves = Vec::new();
        generate_legal_moves(&mut board.clone(), &mut moves);
        let mv = moves[0];

        let child = arena.alloc(Node::new(Some(mv), 1.0, root));
        arena.get_mut(root).children.push(child);
        // Give the root a visit count so sqrt(N) > 0
        arena.get_mut(root).visit_count = 1;

        let mut b = board.clone();
        let (leaf, undo_stack) = select(&arena, root, &mut b, 1.0);

        assert_eq!(leaf, child, "should descend to the single child");
        assert_eq!(undo_stack.len(), 1, "one move applied");
        assert_ne!(b.hash, board.hash, "board must reflect the applied move");
    }

    #[test]
    fn test_select_undo_stack_restores_board() {
        let board = Board::startpos();
        let (mut arena, root) = make_root(&board);

        let mut moves = Vec::new();
        generate_legal_moves(&mut board.clone(), &mut moves);
        let mv = moves[0];

        let child = arena.alloc(Node::new(Some(mv), 1.0, root));
        arena.get_mut(root).children.push(child);
        arena.get_mut(root).visit_count = 1;

        let mut b = board.clone();
        let (_leaf, undo_stack) = select(&arena, root, &mut b, 1.0);

        // Reverse the undo stack to restore the board.
        for (mv, undo) in undo_stack.into_iter().rev() {
            unmake_move_full(&mut b, mv, &undo);
        }
        assert_eq!(b.hash, board.hash, "board must be fully restored after undoing");
    }

    #[test]
    fn test_select_two_levels_deep() {
        // root → child_a → grandchild; selection should reach grandchild.
        let board = Board::startpos();
        let (mut arena, root) = make_root(&board);

        let mut moves = Vec::new();
        generate_legal_moves(&mut board.clone(), &mut moves);
        let mv1 = moves[0];

        // Apply first move to get second position's moves.
        let mut b = board.clone();
        let undo1 = make_move_full(&mut b, mv1);
        let mut moves2 = Vec::new();
        generate_legal_moves(&mut b, &mut moves2);
        let mv2 = moves2[0];
        unmake_move_full(&mut b, mv1, &undo1);

        let child = arena.alloc(Node::new(Some(mv1), 1.0, root));
        let grandchild = arena.alloc(Node::new(Some(mv2), 1.0, child));
        arena.get_mut(root).children.push(child);
        arena.get_mut(root).visit_count = 1;
        arena.get_mut(child).children.push(grandchild);
        arena.get_mut(child).visit_count = 1;

        let (leaf, undo_stack) = select(&arena, root, &mut b, 1.0);

        assert_eq!(leaf, grandchild);
        assert_eq!(undo_stack.len(), 2);

        for (mv, undo) in undo_stack.into_iter().rev() {
            unmake_move_full(&mut b, mv, &undo);
        }
        assert_eq!(b.hash, board.hash);
    }

    // --- Expansion tests ---

    #[test]
    fn test_expand_startpos_creates_30_children() {
        let mut board = Board::startpos();
        let mut arena = Arena::new(64);
        let root = arena.alloc(Node::new(None, 1.0, NO_PARENT));

        let expanded = expand(&mut arena, root, &mut board);

        assert!(expanded, "startpos is not terminal");
        assert_eq!(arena.get(root).children.len(), 30);
    }

    #[test]
    fn test_expand_uniform_prior() {
        let mut board = Board::startpos();
        let mut arena = Arena::new(64);
        let root = arena.alloc(Node::new(None, 1.0, NO_PARENT));
        expand(&mut arena, root, &mut board);

        let expected = 1.0 / 30.0_f32;
        for &child_idx in &arena.get(root).children.clone() {
            let p = arena.get(child_idx).prior;
            assert!((p - expected).abs() < 1e-6, "prior {p} != {expected}");
        }
    }

    #[test]
    fn test_expand_children_have_leaf_as_parent() {
        let mut board = Board::startpos();
        let mut arena = Arena::new(64);
        let root = arena.alloc(Node::new(None, 1.0, NO_PARENT));
        expand(&mut arena, root, &mut board);

        for &child_idx in &arena.get(root).children.clone() {
            assert_eq!(arena.get(child_idx).parent, root);
        }
    }

    #[test]
    fn test_expand_leaf_becomes_interior_node() {
        let mut board = Board::startpos();
        let mut arena = Arena::new(64);
        let root = arena.alloc(Node::new(None, 1.0, NO_PARENT));
        assert!(arena.get(root).is_leaf());
        expand(&mut arena, root, &mut board);
        assert!(!arena.get(root).is_leaf());
    }

    #[test]
    fn test_expand_children_carry_moves() {
        let mut board = Board::startpos();
        let mut arena = Arena::new(64);
        let root = arena.alloc(Node::new(None, 1.0, NO_PARENT));
        expand(&mut arena, root, &mut board);

        for &child_idx in &arena.get(root).children.clone() {
            assert!(arena.get(child_idx).mv.is_some(), "every child must have a move");
        }
    }

    #[test]
    fn test_expand_board_unchanged_after_expand() {
        let mut board = Board::startpos();
        let hash_before = board.hash;
        let mut arena = Arena::new(64);
        let root = arena.alloc(Node::new(None, 1.0, NO_PARENT));
        expand(&mut arena, root, &mut board);
        assert_eq!(board.hash, hash_before, "expand must not modify the board");
    }
}
