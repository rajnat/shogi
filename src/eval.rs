/// Static evaluation
///
/// Material values are derived from the Tanigawa/Bonanza tables, the most
/// widely cited reference for computer-Shogi piece values.  All values are
/// in centipawns (pawn = 100).
use crate::board::Board;

// ---------------------------------------------------------------------------
// Per-piece named constants
// ---------------------------------------------------------------------------

pub const PAWN_VALUE: i32 = 100;
pub const LANCE_VALUE: i32 = 430;
pub const KNIGHT_VALUE: i32 = 450;
pub const SILVER_VALUE: i32 = 640;
pub const GOLD_VALUE: i32 = 690;
pub const BISHOP_VALUE: i32 = 890;
pub const ROOK_VALUE: i32 = 1040;
// King is never counted in material — its "value" is checkmate.

pub const PRO_PAWN_VALUE: i32 = 530; // tokin
pub const PRO_LANCE_VALUE: i32 = 530;
pub const PRO_KNIGHT_VALUE: i32 = 540;
pub const PRO_SILVER_VALUE: i32 = 650; // +10 over silver; promotes are strictly better
pub const PRO_BISHOP_VALUE: i32 = 1120; // dragon horse
pub const PRO_ROOK_VALUE: i32 = 1310; // dragon king

// ---------------------------------------------------------------------------
// Lookup tables
// ---------------------------------------------------------------------------

/// Material value indexed by `PieceType::index()` (0 = Pawn … 13 = ProRook).
/// King (index 7) is 0 — it is never captured and its loss means checkmate.
pub const PIECE_VALUE: [i32; 14] = [
    PAWN_VALUE,       // 0  Pawn
    LANCE_VALUE,      // 1  Lance
    KNIGHT_VALUE,     // 2  Knight
    SILVER_VALUE,     // 3  Silver
    GOLD_VALUE,       // 4  Gold
    BISHOP_VALUE,     // 5  Bishop
    ROOK_VALUE,       // 6  Rook
    0,                // 7  King
    PRO_PAWN_VALUE,   // 8  ProPawn
    PRO_LANCE_VALUE,  // 9  ProLance
    PRO_KNIGHT_VALUE, // 10 ProKnight
    PRO_SILVER_VALUE, // 11 ProSilver
    PRO_BISHOP_VALUE, // 12 ProBishop
    PRO_ROOK_VALUE,   // 13 ProRook
];

/// Net centipawn gain from promoting each base piece (index 0–6).
/// `PROMOTION_GAIN[pt.index()]` is meaningful only for promotable pieces;
/// it is 0 for King (index 7, out of this array's range).
pub const PROMOTION_GAIN: [i32; 7] = [
    PRO_PAWN_VALUE   - PAWN_VALUE,   // 430
    PRO_LANCE_VALUE  - LANCE_VALUE,  // 100
    PRO_KNIGHT_VALUE - KNIGHT_VALUE, //  90
    PRO_SILVER_VALUE - SILVER_VALUE, //  10
    0,                               // Gold cannot promote
    PRO_BISHOP_VALUE - BISHOP_VALUE, // 230
    PRO_ROOK_VALUE   - ROOK_VALUE,   // 270
];

// ---------------------------------------------------------------------------
// Static evaluation
// ---------------------------------------------------------------------------

/// Static material evaluation from the perspective of `board.side_to_move`.
/// Positive = good for the side to move.
///
/// Sums on-board and in-hand material for both sides using `PIECE_VALUE`.
/// Promoted pieces in hand are impossible in Shogi (they revert on capture),
/// so the hand loop only covers indices 0–6.
pub fn eval(board: &Board) -> i32 {
    let stm = board.side_to_move.index();
    let opp = board.side_to_move.opponent().index();
    let mut score = 0i32;

    for pt_idx in 0..14usize {
        let val = PIECE_VALUE[pt_idx];
        score += board.pieces[stm][pt_idx].count() as i32 * val;
        score -= board.pieces[opp][pt_idx].count() as i32 * val;
    }

    for pt_idx in 0..7usize {
        let val = PIECE_VALUE[pt_idx];
        score += board.hand[stm][pt_idx] as i32 * val;
        score -= board.hand[opp][pt_idx] as i32 * val;
    }

    score
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    use crate::types::PieceType;

    #[test]
    fn test_piece_values_ordered() {
        // Rook > Bishop > Gold > Silver > Knight ≈ Lance > Pawn
        assert!(ROOK_VALUE > BISHOP_VALUE);
        assert!(BISHOP_VALUE > GOLD_VALUE);
        assert!(GOLD_VALUE > SILVER_VALUE);
        assert!(SILVER_VALUE > KNIGHT_VALUE);
        assert!(LANCE_VALUE > PAWN_VALUE);
    }

    #[test]
    fn test_promoted_values_exceed_base() {
        assert!(PRO_PAWN_VALUE   > PAWN_VALUE);
        assert!(PRO_LANCE_VALUE  > LANCE_VALUE);
        assert!(PRO_KNIGHT_VALUE > KNIGHT_VALUE);
        assert!(PRO_SILVER_VALUE > SILVER_VALUE, "ProSilver must exceed Silver");
        assert!(PRO_BISHOP_VALUE > BISHOP_VALUE);
        assert!(PRO_ROOK_VALUE   > ROOK_VALUE);
    }

    #[test]
    fn test_piece_value_array_matches_constants() {
        assert_eq!(PIECE_VALUE[PieceType::Pawn.index()],      PAWN_VALUE);
        assert_eq!(PIECE_VALUE[PieceType::Lance.index()],     LANCE_VALUE);
        assert_eq!(PIECE_VALUE[PieceType::Knight.index()],    KNIGHT_VALUE);
        assert_eq!(PIECE_VALUE[PieceType::Silver.index()],    SILVER_VALUE);
        assert_eq!(PIECE_VALUE[PieceType::Gold.index()],      GOLD_VALUE);
        assert_eq!(PIECE_VALUE[PieceType::Bishop.index()],    BISHOP_VALUE);
        assert_eq!(PIECE_VALUE[PieceType::Rook.index()],      ROOK_VALUE);
        assert_eq!(PIECE_VALUE[PieceType::King.index()],      0);
        assert_eq!(PIECE_VALUE[PieceType::ProPawn.index()],   PRO_PAWN_VALUE);
        assert_eq!(PIECE_VALUE[PieceType::ProLance.index()],  PRO_LANCE_VALUE);
        assert_eq!(PIECE_VALUE[PieceType::ProKnight.index()], PRO_KNIGHT_VALUE);
        assert_eq!(PIECE_VALUE[PieceType::ProSilver.index()], PRO_SILVER_VALUE);
        assert_eq!(PIECE_VALUE[PieceType::ProBishop.index()], PRO_BISHOP_VALUE);
        assert_eq!(PIECE_VALUE[PieceType::ProRook.index()],   PRO_ROOK_VALUE);
    }

    #[test]
    fn test_promotion_gain_array() {
        assert_eq!(PROMOTION_GAIN[PieceType::Pawn.index()],   PRO_PAWN_VALUE   - PAWN_VALUE);
        assert_eq!(PROMOTION_GAIN[PieceType::Lance.index()],  PRO_LANCE_VALUE  - LANCE_VALUE);
        assert_eq!(PROMOTION_GAIN[PieceType::Knight.index()], PRO_KNIGHT_VALUE - KNIGHT_VALUE);
        assert_eq!(PROMOTION_GAIN[PieceType::Silver.index()], PRO_SILVER_VALUE - SILVER_VALUE);
        assert_eq!(PROMOTION_GAIN[PieceType::Bishop.index()], PRO_BISHOP_VALUE - BISHOP_VALUE);
        assert_eq!(PROMOTION_GAIN[PieceType::Rook.index()],   PRO_ROOK_VALUE   - ROOK_VALUE);
        // All gains must be non-negative (promoting is never strictly harmful).
        for &gain in PROMOTION_GAIN.iter() {
            assert!(gain >= 0, "promotion gain must be non-negative: {gain}");
        }
    }

    #[test]
    fn test_eval_startpos_is_zero() {
        let board = Board::startpos();
        assert_eq!(eval(&board), 0, "startpos is perfectly symmetric");
    }

    #[test]
    fn test_eval_sign_convention() {
        let mut board = Board::startpos();
        board.hand[0][PieceType::Pawn.index()] += 1; // Black gains a pawn in hand
        assert!(eval(&board) > 0, "extra pawn in hand is positive for side to move");
    }

    #[test]
    fn test_eval_symmetric_flip() {
        // Swapping side_to_move negates the score.
        let mut board = Board::startpos();
        board.hand[0][PieceType::Rook.index()] += 1; // Black holds a rook
        let score_black = eval(&board);
        board.side_to_move = board.side_to_move.opponent();
        let score_white = eval(&board);
        assert_eq!(score_black, -score_white,
            "flipping side_to_move must negate the score");
    }
}
