/// Move generation for Shogi
/// Generates pseudo-legal moves, then filters for legality (king safety)
use crate::attacks::{
    bishop_attacks, lance_attacks, piece_attacks, pro_bishop_attacks, pro_rook_attacks,
    rook_attacks, step_attacks,
};
use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::moves::{make_move_full, unmake_move_full};
use crate::types::{Color, Move, PieceType, Square, file_of, rank_of, square};

/// Returns true if the given color's king is in check
pub fn is_in_check(board: &Board, color: Color) -> bool {
    let king_sq = board.king_sq(color);
    is_square_attacked(board, king_sq, color.opponent())
}

/// Returns true if sq is attacked by the given attacker color
pub fn is_square_attacked(board: &Board, sq: Square, by: Color) -> bool {
    let occ = board.occ;
    let by_c = by.index();

    // Check each piece type of attacker
    for pt_idx in 0..14 {
        let pt = PieceType::from_index(pt_idx).unwrap();
        let attackers = board.pieces[by_c][pt_idx];
        if attackers.is_empty() {
            continue;
        }

        let mut bb = attackers;
        while bb.is_not_empty() {
            let attacker_sq = bb.pop_lsb();
            let attacks = piece_attacks(pt, by, attacker_sq, occ);
            if attacks.contains(sq) {
                return true;
            }
        }
    }
    false
}

/// Promotion zone: ranks 1-3 for Black (rank_idx 0-2), ranks 7-9 for White (rank_idx 6-8)
#[inline]
pub fn in_promotion_zone(sq: Square, color: Color) -> bool {
    let rank = rank_of(sq);
    match color {
        Color::Black => rank <= 2,
        Color::White => rank >= 6,
    }
}

/// Returns true if piece MUST promote (would have no legal moves)
#[inline]
fn must_promote(pt: PieceType, to: Square, color: Color) -> bool {
    let rank = rank_of(to);
    match color {
        Color::Black => match pt {
            PieceType::Pawn | PieceType::Lance => rank == 0,
            PieceType::Knight => rank <= 1,
            _ => false,
        },
        Color::White => match pt {
            PieceType::Pawn | PieceType::Lance => rank == 8,
            PieceType::Knight => rank >= 7,
            _ => false,
        },
    }
}

/// Check if a pawn drop on the given file would be illegal (nifu - double pawn)
fn has_pawn_on_file(board: &Board, color: Color, file_idx: u8) -> bool {
    let pawn_bb = board.pieces[color.index()][PieceType::Pawn.index()];
    // Check all squares on this file
    for rank in 0..9u8 {
        let sq = square(file_idx, rank);
        if pawn_bb.contains(sq) {
            return true;
        }
    }
    false
}

/// Check if dropping a pawn delivers immediate checkmate (illegal)
#[allow(dead_code)]
fn pawn_drop_is_checkmate(board: &mut Board, to: Square, color: Color) -> bool {
    let mv = Move::new_drop(PieceType::Pawn, to);
    let undo = make_move_full(board, mv);

    // Check if opponent (who just got the pawn dropped on them) has any legal moves
    let opp = color.opponent();
    let in_check = is_in_check(board, opp);
    let result = if in_check {
        // It's check. Is it checkmate?
        let mut moves = Vec::new();
        generate_moves(board, &mut moves);
        // generate_moves generates for current side (opponent)
        let is_mate = moves.is_empty();
        is_mate
    } else {
        false
    };

    unmake_move_full(board, mv, &undo);
    result
}

/// Generate all legal moves for the side to move
pub fn generate_moves(board: &mut Board, moves: &mut Vec<Move>) {
    let stm = board.side_to_move;
    let mut pseudo = Vec::with_capacity(128);
    generate_pseudo_legal(board, &mut pseudo);

    for mv in pseudo {
        // Make move and check legality
        let undo = make_move_full(board, mv);
        let legal = !is_in_check(board, stm);
        unmake_move_full(board, mv, &undo);

        if legal {
            moves.push(mv);
        }
    }
}

/// Generate pseudo-legal moves (does not check if king is left in check)
pub fn generate_pseudo_legal(board: &Board, moves: &mut Vec<Move>) {
    let stm = board.side_to_move;
    let c = stm.index();
    let own = board.color_bb[c];
    let occ = board.occ;

    // Generate piece moves
    for pt_idx in 0..14 {
        let pt = PieceType::from_index(pt_idx).unwrap();
        let mut bb = board.pieces[c][pt_idx];

        while bb.is_not_empty() {
            let from = bb.pop_lsb();
            gen_piece_moves(board, from, pt, stm, own, occ, moves);
        }
    }

    // Generate drops
    gen_drops(board, stm, occ, moves);
}

fn gen_piece_moves(
    _board: &Board,
    from: Square,
    pt: PieceType,
    stm: Color,
    own: Bitboard,
    occ: Bitboard,
    moves: &mut Vec<Move>,
) {
    let targets: Bitboard = match pt {
        PieceType::Pawn => step_attacks(pt, stm, from),
        PieceType::Knight => step_attacks(pt, stm, from),
        PieceType::Silver => step_attacks(pt, stm, from),
        PieceType::Gold => step_attacks(pt, stm, from),
        PieceType::King => step_attacks(pt, stm, from),
        PieceType::ProPawn
        | PieceType::ProLance
        | PieceType::ProKnight
        | PieceType::ProSilver => step_attacks(PieceType::Gold, stm, from),
        PieceType::Lance => lance_attacks(stm, from, occ),
        PieceType::Bishop => bishop_attacks(from, occ),
        PieceType::Rook => rook_attacks(from, occ),
        PieceType::ProBishop => pro_bishop_attacks(from, occ),
        PieceType::ProRook => pro_rook_attacks(from, occ),
    };

    // Remove own pieces
    let valid = targets - own;
    let from_in_promo_zone = in_promotion_zone(from, stm);

    for to in valid.iter_squares() {
        let to_in_promo_zone = in_promotion_zone(to, stm);
        let can_promo = pt.can_promote() && (from_in_promo_zone || to_in_promo_zone);
        let forced_promo = must_promote(pt, to, stm);

        if forced_promo {
            // Must promote
            moves.push(Move::new_normal(from, to, pt, true));
        } else if can_promo {
            // Optional promotion: add both
            moves.push(Move::new_normal(from, to, pt, true));
            moves.push(Move::new_normal(from, to, pt, false));
        } else {
            moves.push(Move::new_normal(from, to, pt, false));
        }
    }
}

fn gen_drops(board: &Board, stm: Color, occ: Bitboard, moves: &mut Vec<Move>) {
    let c = stm.index();
    let empty = !occ; // all empty squares

    for pt_idx in 0..7usize {
        if board.hand[c][pt_idx] == 0 {
            continue;
        }
        let pt = PieceType::from_index(pt_idx).unwrap();

        // Pawn and Lance: cannot drop on last rank (must_promote would apply)
        // Knight: cannot drop on last 2 ranks
        // We generate all valid drop squares
        for to in empty.iter_squares() {
            let rank = rank_of(to);

            // Skip squares where piece would have no moves
            let invalid = match (pt, stm) {
                (PieceType::Pawn, Color::Black) | (PieceType::Lance, Color::Black) => rank == 0,
                (PieceType::Pawn, Color::White) | (PieceType::Lance, Color::White) => rank == 8,
                (PieceType::Knight, Color::Black) => rank <= 1,
                (PieceType::Knight, Color::White) => rank >= 7,
                _ => false,
            };
            if invalid {
                continue;
            }

            // Nifu: cannot drop pawn if already have one on same file
            if pt == PieceType::Pawn {
                if has_pawn_on_file(board, stm, file_of(to)) {
                    continue;
                }
            }

            moves.push(Move::new_drop(pt, to));
        }
    }
}

/// Check pawn drop checkmate (must use mutable board)
pub fn generate_legal_moves(board: &mut Board, moves: &mut Vec<Move>) {
    let stm = board.side_to_move;
    let c = stm.index();
    let own = board.color_bb[c];
    let occ = board.occ;

    // Piece moves
    let mut pseudo = Vec::with_capacity(128);
    for pt_idx in 0..14 {
        let pt = PieceType::from_index(pt_idx).unwrap();
        let mut bb = board.pieces[c][pt_idx];
        while bb.is_not_empty() {
            let from = bb.pop_lsb();
            gen_piece_moves(board, from, pt, stm, own, occ, &mut pseudo);
        }
    }

    // Filter pseudo-legal piece moves
    for mv in &pseudo {
        let undo = make_move_full(board, *mv);
        let legal = !is_in_check(board, stm);
        unmake_move_full(board, *mv, &undo);
        if legal {
            moves.push(*mv);
        }
    }

    // Drops (must handle pawn drop checkmate)
    let empty = !occ;
    for pt_idx in 0..7usize {
        if board.hand[c][pt_idx] == 0 {
            continue;
        }
        let pt = PieceType::from_index(pt_idx).unwrap();

        for to in empty.iter_squares() {
            let rank = rank_of(to);

            let invalid = match (pt, stm) {
                (PieceType::Pawn, Color::Black) | (PieceType::Lance, Color::Black) => rank == 0,
                (PieceType::Pawn, Color::White) | (PieceType::Lance, Color::White) => rank == 8,
                (PieceType::Knight, Color::Black) => rank <= 1,
                (PieceType::Knight, Color::White) => rank >= 7,
                _ => false,
            };
            if invalid {
                continue;
            }

            // Nifu
            if pt == PieceType::Pawn && has_pawn_on_file(board, stm, file_of(to)) {
                continue;
            }

            let mv = Move::new_drop(pt, to);

            // Check king safety after drop
            let undo = make_move_full(board, mv);
            let legal = !is_in_check(board, stm);
            unmake_move_full(board, mv, &undo);

            if !legal {
                continue;
            }

            // For pawn drops, check for checkmate (uchifuzume)
            if pt == PieceType::Pawn {
                // Check if this creates check on opponent
                let undo2 = make_move_full(board, mv);
                let gives_check = is_in_check(board, stm.opponent());
                if gives_check {
                    // Check if it's checkmate
                    let mut opp_moves = Vec::new();
                    generate_moves(board, &mut opp_moves);
                    unmake_move_full(board, mv, &undo2);
                    if opp_moves.is_empty() {
                        continue; // Pawn drop checkmate is illegal
                    }
                } else {
                    unmake_move_full(board, mv, &undo2);
                }
            }

            moves.push(mv);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;

    #[test]
    fn test_startpos_moves_depth1() {
        let mut board = Board::startpos();
        let mut moves = Vec::new();
        generate_legal_moves(&mut board, &mut moves);
        // Standard startpos has exactly 30 legal moves
        assert_eq!(
            moves.len(),
            30,
            "Expected 30 moves at depth 1, got {}",
            moves.len()
        );
    }

    #[test]
    fn test_is_in_check_startpos() {
        let board = Board::startpos();
        assert!(!is_in_check(&board, Color::Black));
        assert!(!is_in_check(&board, Color::White));
    }
}
