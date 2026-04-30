/// Make and unmake moves on the Board
use crate::board::Board;
use crate::types::{Move, PieceType, Square};
use crate::zobrist::{hand_hash, piece_hash, side_hash};

/// State to restore after unmake_move
#[derive(Clone, Debug)]
pub struct UndoState {
    pub captured: Option<(PieceType, Square)>,
    pub hash: u64,
    pub ply: u16,
}

/// Execute a move and return undo state
pub fn make_move_full(board: &mut Board, mv: Move) -> UndoState {
    let saved_hash = board.hash;
    let saved_ply = board.ply;
    let stm = board.side_to_move;
    let opp = stm.opponent();
    let c = stm.index();
    let oc = opp.index();

    let mut captured_info: Option<(PieceType, Square)> = None;

    if mv.is_drop() {
        let pt = mv.drop_piece();
        let to = mv.to_sq();
        let pt_idx = pt.index();

        let old_count = board.hand[c][pt_idx] as usize;
        board.hash ^= hand_hash(c, pt_idx, old_count);
        board.hand[c][pt_idx] -= 1;
        let new_count = board.hand[c][pt_idx] as usize;
        if new_count > 0 {
            board.hash ^= hand_hash(c, pt_idx, new_count);
        }

        board.pieces[c][pt_idx].set(to);
        board.color_bb[c].set(to);
        board.occ.set(to);
        board.hash ^= piece_hash(to as usize, pt_idx, c);
    } else {
        let from = mv.from_sq();
        let to = mv.to_sq();
        let pt = mv.piece_type();
        let pt_idx = pt.index();

        // Handle capture
        if board.color_bb[oc].contains(to) {
            let cap_pt = board.piece_type_at(to, opp).unwrap();
            let cap_idx = cap_pt.index();

            board.pieces[oc][cap_idx].clear(to);
            board.color_bb[oc].clear(to);
            board.occ.clear(to);
            board.hash ^= piece_hash(to as usize, cap_idx, oc);

            let hand_pt = cap_pt.demoted();
            let hand_idx = hand_pt.index();
            let old_hand = board.hand[c][hand_idx] as usize;
            if old_hand > 0 {
                board.hash ^= hand_hash(c, hand_idx, old_hand);
            }
            board.hand[c][hand_idx] += 1;
            let new_hand = board.hand[c][hand_idx] as usize;
            board.hash ^= hand_hash(c, hand_idx, new_hand);

            captured_info = Some((cap_pt, to));
        }

        // Move piece from -> to
        board.pieces[c][pt_idx].clear(from);
        board.color_bb[c].clear(from);
        board.occ.clear(from);
        board.hash ^= piece_hash(from as usize, pt_idx, c);

        let final_pt_idx = if mv.is_promote() {
            pt.promoted().unwrap().index()
        } else {
            pt_idx
        };

        board.pieces[c][final_pt_idx].set(to);
        board.color_bb[c].set(to);
        board.occ.set(to);
        board.hash ^= piece_hash(to as usize, final_pt_idx, c);
    }

    board.hash ^= side_hash();
    board.side_to_move = opp;
    board.ply += 1;

    UndoState {
        captured: captured_info,
        hash: saved_hash,
        ply: saved_ply,
    }
}

/// Unmake a move, restoring board to prior state
pub fn unmake_move_full(board: &mut Board, mv: Move, undo: &UndoState) {
    let stm = board.side_to_move; // current stm = opponent of who moved
    let mover = stm.opponent();
    let c = mover.index();
    let oc = stm.index();

    board.side_to_move = mover;
    board.ply = undo.ply;

    if mv.is_drop() {
        let pt = mv.drop_piece();
        let to = mv.to_sq();
        let pt_idx = pt.index();

        board.pieces[c][pt_idx].clear(to);
        board.color_bb[c].clear(to);
        board.occ.clear(to);
        board.hand[c][pt_idx] += 1;
    } else {
        let from = mv.from_sq();
        let to = mv.to_sq();
        let pt = mv.piece_type();
        let pt_idx = pt.index();

        let final_pt_idx = if mv.is_promote() {
            pt.promoted().unwrap().index()
        } else {
            pt_idx
        };

        // Remove piece from destination
        board.pieces[c][final_pt_idx].clear(to);
        board.color_bb[c].clear(to);
        board.occ.clear(to);

        // Restore piece at source
        board.pieces[c][pt_idx].set(from);
        board.color_bb[c].set(from);
        board.occ.set(from);

        // Restore captured piece
        if let Some((cap_pt, cap_sq)) = undo.captured {
            let cap_idx = cap_pt.index();
            board.pieces[oc][cap_idx].set(cap_sq);
            board.color_bb[oc].set(cap_sq);
            board.occ.set(cap_sq);

            let hand_pt = cap_pt.demoted();
            let hand_idx = hand_pt.index();
            board.hand[c][hand_idx] -= 1;
        }
    }

    board.hash = undo.hash;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    use crate::types::{Color, Move, PieceType, square};

    #[test]
    fn test_make_unmake_pawn() {
        let mut board = Board::startpos();
        let hash_before = board.hash;
        let sfen_before = board.to_sfen();

        // Move Black pawn from file_idx=0, rank_idx=6 to rank_idx=5
        let from = square(0, 6);
        let to = square(0, 5);
        let mv = Move::new_normal(from, to, PieceType::Pawn, false);

        let undo = make_move_full(&mut board, mv);
        assert_eq!(board.side_to_move, Color::White);
        assert!(board.pieces[0][PieceType::Pawn.index()].contains(to));
        assert!(!board.pieces[0][PieceType::Pawn.index()].contains(from));

        unmake_move_full(&mut board, mv, &undo);
        assert_eq!(board.hash, hash_before);
        assert_eq!(board.to_sfen(), sfen_before);
    }

    #[test]
    fn test_make_unmake_capture() {
        // Create a position where Black can capture a White piece
        // Use a simple custom position
        let sfen = "9/9/9/9/4p4/4P4/9/9/4K3k b - 1";
        let mut board = Board::from_sfen(sfen).expect("valid sfen");
        let hash_before = board.hash;

        // Black pawn at file4, rank5 captures White pawn at file4, rank4
        let from = crate::types::square(4, 5);
        let to = crate::types::square(4, 4);
        let mv = Move::new_normal(from, to, PieceType::Pawn, false);

        let undo = make_move_full(&mut board, mv);
        assert_eq!(board.hand[0][PieceType::Pawn.index()], 1); // Black has captured pawn

        unmake_move_full(&mut board, mv, &undo);
        assert_eq!(board.hash, hash_before);
        assert_eq!(board.hand[0][PieceType::Pawn.index()], 0);
    }
}
