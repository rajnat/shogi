/// Canonical flat policy index for every possible Shogi move.
///
/// ## Layout
///
/// ```text
///   [0, 13121]    board moves  — from(81) × to(81) × promote(2)
///                               idx = from * 162 + to * 2 + promote
///
///   [13122, 13688] drop moves  — hand_pt(7) × to(81)
///                               idx = 13122 + pt.index() * 81 + to
/// ```
///
/// `PieceType::HAND_TYPES` indices (Pawn=0 … Rook=6) are used directly for drops
/// because they already occupy `PieceType::index()` values 0–6.
///
/// ## Round-trip
///
/// `move_to_index` is lossless only when paired with `index_to_move` on the
/// matching board position, because the piece type at `from_sq` is looked up
/// from the board rather than stored in the index.
use crate::board::Board;
use crate::types::{Move, PieceType};

/// Total distinct move indices — must equal `net::NUM_ACTIONS`.
pub const NUM_ACTIONS: usize = 13_689;

const DROP_OFFSET: usize = 81 * 81 * 2; // 13_122

/// Maps any legal `Move` to its canonical policy index `[0, NUM_ACTIONS)`.
#[inline]
pub fn move_to_index(mv: Move) -> usize {
    if mv.is_drop() {
        // drop_piece().index() == hand slot 0..6 (Pawn..Rook)
        DROP_OFFSET + mv.drop_piece().index() * 81 + mv.to_sq() as usize
    } else {
        mv.from_sq() as usize * 81 * 2
            + mv.to_sq() as usize * 2
            + mv.is_promote() as usize
    }
}

/// Reconstructs a `Move` from a policy index and the current board state.
///
/// Returns `None` when:
/// - `idx >= NUM_ACTIONS`
/// - `idx` encodes a board move but no piece of the side to move occupies `from_sq`
pub fn index_to_move(idx: usize, board: &Board) -> Option<Move> {
    if idx >= NUM_ACTIONS {
        return None;
    }
    if idx >= DROP_OFFSET {
        let rel = idx - DROP_OFFSET;
        let pt = PieceType::HAND_TYPES[rel / 81];
        let to_sq = (rel % 81) as u8;
        Some(Move::new_drop(pt, to_sq))
    } else {
        let from_sq = (idx / (81 * 2)) as u8;
        let to_sq = ((idx / 2) % 81) as u8;
        let promote = (idx % 2) != 0;
        let color = board.side_to_move.index();
        let pt_idx = (0usize..14).find(|&p| board.pieces[color][p].contains(from_sq))?;
        let pt = PieceType::from_index(pt_idx)?;
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

    // ----- Drop move roundtrip (self-contained — no board needed) -----

    #[test]
    fn test_drop_roundtrip_all_pieces_all_squares() {
        let board = Board::startpos();
        for pt in PieceType::HAND_TYPES {
            for to in 0u8..81 {
                let mv = Move::new_drop(pt, to);
                let idx = move_to_index(mv);
                assert!(idx >= DROP_OFFSET, "drop idx below DROP_OFFSET");
                assert!(idx < NUM_ACTIONS, "drop idx out of range");
                let back = index_to_move(idx, &board).expect("should reconstruct drop");
                assert!(back.is_drop());
                assert_eq!(back.drop_piece(), pt);
                assert_eq!(back.to_sq(), to);
            }
        }
    }

    #[test]
    fn test_drop_indices_are_disjoint_from_board() {
        for pt in PieceType::HAND_TYPES {
            for to in 0u8..81 {
                let idx = move_to_index(Move::new_drop(pt, to));
                assert!(idx >= DROP_OFFSET);
            }
        }
    }

    // ----- Board move index range -----

    #[test]
    fn test_board_move_indices_below_drop_offset() {
        // A few representative normal moves (we don't care if they're legal here).
        let cases = [
            (0u8, 1u8, false),
            (0, 80, true),
            (80, 0, false),
            (40, 40, true), // from==to is impossible in a game but encodes fine
        ];
        for (from, to, promote) in cases {
            let mv = Move::new_normal(from, to, PieceType::Pawn, promote);
            let idx = move_to_index(mv);
            assert!(idx < DROP_OFFSET, "board move idx={idx} should be < {DROP_OFFSET}");
        }
    }

    // ----- Legal moves at startpos -----

    #[test]
    fn test_legal_moves_all_in_range() {
        let mut board = Board::startpos();
        let mut moves = Vec::new();
        generate_legal_moves(&mut board, &mut moves);
        for &mv in &moves {
            let idx = move_to_index(mv);
            assert!(idx < NUM_ACTIONS, "move {} has idx {idx} >= {NUM_ACTIONS}", mv.to_usi_string());
        }
    }

    #[test]
    fn test_legal_moves_no_duplicates() {
        let mut board = Board::startpos();
        let mut moves = Vec::new();
        generate_legal_moves(&mut board, &mut moves);
        let indices: HashSet<usize> = moves.iter().map(|&mv| move_to_index(mv)).collect();
        assert_eq!(indices.len(), moves.len(), "duplicate policy indices among legal moves");
    }

    #[test]
    fn test_legal_moves_roundtrip() {
        let mut board = Board::startpos();
        let mut moves = Vec::new();
        generate_legal_moves(&mut board, &mut moves);
        for &mv in &moves {
            let idx = move_to_index(mv);
            let back = index_to_move(idx, &board)
                .unwrap_or_else(|| panic!("index_to_move failed for {}", mv.to_usi_string()));
            assert_eq!(back.from_sq(), mv.from_sq(), "from_sq mismatch for {}", mv.to_usi_string());
            assert_eq!(back.to_sq(),   mv.to_sq(),   "to_sq mismatch for {}",   mv.to_usi_string());
            assert_eq!(back.is_promote(), mv.is_promote(), "promote mismatch for {}", mv.to_usi_string());
            assert_eq!(back.is_drop(),    mv.is_drop(),    "drop mismatch for {}",    mv.to_usi_string());
        }
    }

    // ----- Out-of-range guard -----

    #[test]
    fn test_out_of_range_returns_none() {
        let board = Board::startpos();
        assert!(index_to_move(NUM_ACTIONS, &board).is_none());
        assert!(index_to_move(usize::MAX, &board).is_none());
    }

    // ----- Specific index arithmetic -----

    #[test]
    fn test_first_board_index() {
        // from=0, to=0, promote=false → index 0
        let mv = Move::new_normal(0, 0, PieceType::Pawn, false);
        assert_eq!(move_to_index(mv), 0);
    }

    #[test]
    fn test_promote_flag_increments_by_one() {
        let no_promo = Move::new_normal(3, 5, PieceType::Lance, false);
        let promo    = Move::new_normal(3, 5, PieceType::Lance, true);
        assert_eq!(move_to_index(promo), move_to_index(no_promo) + 1);
    }

    #[test]
    fn test_first_drop_index() {
        // Pawn drop to sq 0 → index DROP_OFFSET
        assert_eq!(move_to_index(Move::new_drop(PieceType::Pawn, 0)), DROP_OFFSET);
    }

    #[test]
    fn test_last_drop_index() {
        // Rook drop to sq 80 → index NUM_ACTIONS - 1
        assert_eq!(move_to_index(Move::new_drop(PieceType::Rook, 80)), NUM_ACTIONS - 1);
    }

    #[test]
    fn test_piece_type_doesnt_affect_board_index() {
        // The piece type is NOT part of the board-move index.
        let sq_from = square(2, 6); // arbitrary
        let sq_to   = square(2, 5);
        let idx_pawn  = move_to_index(Move::new_normal(sq_from, sq_to, PieceType::Pawn,   false));
        let idx_rook  = move_to_index(Move::new_normal(sq_from, sq_to, PieceType::Rook,   false));
        let idx_lance = move_to_index(Move::new_normal(sq_from, sq_to, PieceType::Lance,  false));
        assert_eq!(idx_pawn, idx_rook);
        assert_eq!(idx_pawn, idx_lance);
    }
}
