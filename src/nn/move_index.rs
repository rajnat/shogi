/// Canonical flat policy index for every possible Shogi move.
///
/// ## Layout
///
/// ```text
///   [0, 13121]     board moves — from(81) × to(81) × promote(2)
///                                idx = from * 162 + to * 2 + promote
///
///   [13122, 13688]  drop moves — hand_pt(7) × to(81)
///                                idx = 13122 + pt.index() * 81 + to
/// ```
///
/// `PieceType::HAND_TYPES` indices (Pawn=0 … Rook=6) are used directly for drops
/// because they occupy `PieceType::index()` values 0–6.
///
/// ## Piece type and round-trip
///
/// Piece type is **not** stored in board-move indices (only from, to, promote are).
/// Consequently:
///
/// - `move_to_index` is injective: every legal move maps to a unique slot.
/// - `index_to_move` (board-free) returns a move with `PieceType::Pawn` in the
///   piece-type field for board moves — the field is a placeholder, not meaningful.
///   The important invariant still holds: `move_to_index(index_to_move(i)) == i`.
/// - `index_to_move_on_board` gives the correct piece type by looking it up
///   on the position; use it when the full `Move` is needed (e.g. make-move).
use crate::board::Board;
use crate::types::{Move, PieceType};

/// Total distinct move indices — must equal `net::NUM_ACTIONS`.
pub const NUM_ACTIONS: usize = 13_689;

const DROP_OFFSET: usize = 81 * 81 * 2; // 13_122

/// Maps any `Move` to its canonical policy index `[0, NUM_ACTIONS)`.
///
/// Piece type is ignored for board moves; only `from_sq`, `to_sq`, and the
/// promote flag contribute to the index.
#[inline]
pub fn move_to_index(mv: Move) -> usize {
    if mv.is_drop() {
        DROP_OFFSET + mv.drop_piece().index() * 81 + mv.to_sq() as usize
    } else {
        mv.from_sq() as usize * 81 * 2
            + mv.to_sq() as usize * 2
            + mv.is_promote() as usize
    }
}

/// Decodes a policy index into a `Move` without needing the board.
///
/// For drop moves the result is fully accurate.  For board moves the
/// `piece_type` field is set to `PieceType::Pawn` as a placeholder —
/// only `from_sq`, `to_sq`, and `is_promote` are meaningful.
///
/// Returns `None` if `idx >= NUM_ACTIONS`.
///
/// **Invariant**: `move_to_index(index_to_move(i).unwrap()) == i` for all valid `i`.
pub fn index_to_move(idx: usize) -> Option<Move> {
    if idx >= NUM_ACTIONS {
        return None;
    }
    if idx >= DROP_OFFSET {
        let rel = idx - DROP_OFFSET;
        let pt     = PieceType::HAND_TYPES[rel / 81];
        let to_sq  = (rel % 81) as u8;
        Some(Move::new_drop(pt, to_sq))
    } else {
        let from_sq = (idx / (81 * 2)) as u8;
        let to_sq   = ((idx / 2) % 81) as u8;
        let promote = (idx % 2) != 0;
        Some(Move::new_normal(from_sq, to_sq, PieceType::Pawn, promote))
    }
}

/// Like `index_to_move` but looks up the true piece type from the board.
///
/// Returns `None` when `idx >= NUM_ACTIONS` or when a board-move index
/// encodes a `from_sq` that has no piece of the side to move (i.e. the
/// move is not valid in this position).
pub fn index_to_move_on_board(idx: usize, board: &Board) -> Option<Move> {
    if idx >= NUM_ACTIONS {
        return None;
    }
    if idx >= DROP_OFFSET {
        let rel    = idx - DROP_OFFSET;
        let pt     = PieceType::HAND_TYPES[rel / 81];
        let to_sq  = (rel % 81) as u8;
        Some(Move::new_drop(pt, to_sq))
    } else {
        let from_sq = (idx / (81 * 2)) as u8;
        let to_sq   = ((idx / 2) % 81) as u8;
        let promote = (idx % 2) != 0;
        let color   = board.side_to_move.index();
        let pt_idx  = (0usize..14).find(|&p| board.pieces[color][p].contains(from_sq))?;
        let pt      = PieceType::from_index(pt_idx)?;
        Some(Move::new_normal(from_sq, to_sq, pt, promote))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use crate::board::Board;
    use crate::movegen::generate_legal_moves;
    use crate::types::{Move, PieceType, square};

    // ----- Constants -----

    #[test]
    fn test_num_actions_value() {
        assert_eq!(NUM_ACTIONS, 13_689);
        assert_eq!(DROP_OFFSET, 13_122);
        assert_eq!(NUM_ACTIONS - DROP_OFFSET, 7 * 81);
    }

    // ----- move_to_index / index_to_move (board-free) -----

    #[test]
    fn test_roundtrip_all_drop_moves() {
        for pt in PieceType::HAND_TYPES {
            for to in 0u8..81 {
                let mv  = Move::new_drop(pt, to);
                let idx = move_to_index(mv);
                assert!(idx >= DROP_OFFSET && idx < NUM_ACTIONS);
                let back = index_to_move(idx).unwrap();
                assert!(back.is_drop());
                assert_eq!(back.drop_piece(), pt);
                assert_eq!(back.to_sq(), to);
            }
        }
    }

    #[test]
    fn test_roundtrip_board_moves_geometry() {
        // Piece type is not round-tripped; from/to/promote must be.
        let cases = [(0u8, 1u8, false), (0, 80, true), (80, 0, false), (40, 7, true)];
        for (from, to, promote) in cases {
            for pt in PieceType::HAND_TYPES {
                let mv   = Move::new_normal(from, to, pt, promote);
                let idx  = move_to_index(mv);
                let back = index_to_move(idx).unwrap();
                assert!(!back.is_drop());
                assert_eq!(back.from_sq(),   from);
                assert_eq!(back.to_sq(),     to);
                assert_eq!(back.is_promote(), promote);
            }
        }
    }

    /// Core invariant: move_to_index ∘ index_to_move = id on [0, NUM_ACTIONS).
    #[test]
    fn test_move_to_index_is_left_inverse() {
        for i in 0..NUM_ACTIONS {
            let mv = index_to_move(i).unwrap();
            assert_eq!(move_to_index(mv), i, "roundtrip failed at index {i}");
        }
    }

    #[test]
    fn test_board_move_pawn_placeholder() {
        let from = 10u8;
        let to   = 15u8;
        let idx  = move_to_index(Move::new_normal(from, to, PieceType::Rook, false));
        let back = index_to_move(idx).unwrap();
        // piece_type is Pawn placeholder, NOT Rook
        assert_eq!(back.piece_type(), PieceType::Pawn);
        assert_eq!(back.from_sq(), from);
        assert_eq!(back.to_sq(), to);
    }

    #[test]
    fn test_out_of_range_returns_none() {
        assert!(index_to_move(NUM_ACTIONS).is_none());
        assert!(index_to_move(usize::MAX).is_none());
    }

    // ----- index_to_move_on_board (full piece type) -----

    #[test]
    fn test_legal_moves_full_roundtrip() {
        let mut board = Board::startpos();
        let mut moves = Vec::new();
        generate_legal_moves(&mut board, &mut moves);
        for &mv in &moves {
            let idx  = move_to_index(mv);
            let back = index_to_move_on_board(idx, &board)
                .unwrap_or_else(|| panic!("on_board failed for {}", mv.to_usi_string()));
            assert_eq!(back.from_sq(),    mv.from_sq(),    "{}", mv.to_usi_string());
            assert_eq!(back.to_sq(),      mv.to_sq(),      "{}", mv.to_usi_string());
            assert_eq!(back.is_promote(), mv.is_promote(), "{}", mv.to_usi_string());
            assert_eq!(back.is_drop(),    mv.is_drop(),    "{}", mv.to_usi_string());
            assert_eq!(back.piece_type(), mv.piece_type(), "{}", mv.to_usi_string());
        }
    }

    #[test]
    fn test_on_board_out_of_range_returns_none() {
        let board = Board::startpos();
        assert!(index_to_move_on_board(NUM_ACTIONS, &board).is_none());
        assert!(index_to_move_on_board(usize::MAX,  &board).is_none());
    }

    // ----- Legal moves at startpos -----

    #[test]
    fn test_legal_moves_all_in_range() {
        let mut board = Board::startpos();
        let mut moves = Vec::new();
        generate_legal_moves(&mut board, &mut moves);
        for &mv in &moves {
            let idx = move_to_index(mv);
            assert!(idx < NUM_ACTIONS, "{} → idx {idx}", mv.to_usi_string());
        }
    }

    #[test]
    fn test_legal_moves_no_duplicates() {
        let mut board = Board::startpos();
        let mut moves = Vec::new();
        generate_legal_moves(&mut board, &mut moves);
        let indices: HashSet<usize> = moves.iter().map(|&mv| move_to_index(mv)).collect();
        assert_eq!(indices.len(), moves.len(), "duplicate indices");
    }

    // ----- Specific index arithmetic -----

    #[test]
    fn test_first_board_index() {
        assert_eq!(move_to_index(Move::new_normal(0, 0, PieceType::Pawn, false)), 0);
    }

    #[test]
    fn test_promote_flag_is_lsb() {
        let base  = move_to_index(Move::new_normal(3, 5, PieceType::Lance, false));
        let promo = move_to_index(Move::new_normal(3, 5, PieceType::Lance, true));
        assert_eq!(promo, base + 1);
    }

    #[test]
    fn test_first_drop_index() {
        assert_eq!(move_to_index(Move::new_drop(PieceType::Pawn, 0)), DROP_OFFSET);
    }

    #[test]
    fn test_last_drop_index() {
        assert_eq!(move_to_index(Move::new_drop(PieceType::Rook, 80)), NUM_ACTIONS - 1);
    }

    #[test]
    fn test_piece_type_doesnt_affect_board_index() {
        let from = square(2, 6);
        let to   = square(2, 5);
        let idx_pawn  = move_to_index(Move::new_normal(from, to, PieceType::Pawn,   false));
        let idx_rook  = move_to_index(Move::new_normal(from, to, PieceType::Rook,   false));
        let idx_lance = move_to_index(Move::new_normal(from, to, PieceType::Lance,  false));
        assert_eq!(idx_pawn, idx_rook);
        assert_eq!(idx_pawn, idx_lance);
    }

    #[test]
    fn test_board_and_drop_ranges_disjoint() {
        for pt in PieceType::HAND_TYPES {
            for to in 0u8..81 {
                assert!(move_to_index(Move::new_drop(pt, to)) >= DROP_OFFSET);
            }
        }
        let board_max = move_to_index(Move::new_normal(80, 80, PieceType::Pawn, true));
        assert!(board_max < DROP_OFFSET);
    }
}
