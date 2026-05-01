/// MCTS search phases: selection, expansion, evaluation, backpropagation.
use crate::board::Board;
use crate::moves::{make_move_full, UndoState};
use crate::types::Move;
use super::{Arena, NodeIdx};

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
}
