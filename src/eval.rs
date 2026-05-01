/// Static evaluation
///
/// Material values are derived from the Tanigawa/Bonanza tables, the most
/// widely cited reference for computer-Shogi piece values.  All values are
/// in centipawns (pawn = 100).
use crate::board::Board;
use crate::types::{add_step, rank_of, PieceType, Square};

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

/// Rank-based positional bonuses indexed by **advancement level** (0–8).
/// ADV_IDX 0 = piece's own back rank; ADV_IDX 8 = deepest enemy territory.
///
/// For a Black piece at square `sq`: adv_idx = 8 − rank_of(sq)
/// For a White piece at square `sq`: adv_idx = rank_of(sq)
///
/// All values are in centipawns. King is the exception — it wants adv_idx 0
/// (own back rank) to stay safe. File-based bonuses (bishop diagonals, rook
/// open files) are deferred to a later evaluation pass.
pub const RANK_PST: [[i32; 9]; 14] = [
    //   adv: [ 0,   1,   2,   3,   4,   5,   6,   7,   8]
    //        [own            midfield            enemy]
    [-15, -5,  0,  5, 10, 15, 20, 25, 30], // 0  Pawn
    [ -5, -5,  0,  5, 10, 15, 20, 20,  0], // 1  Lance  (forced promote at adv=8)
    [ -5, -5,  0,  5, 10, 15, 20, 10,  0], // 2  Knight (restricted at adv=7; forced at adv=8)
    [ -5, -5,  0,  5,  8, 10, 12, 15, 15], // 3  Silver
    [ -5, -5,  0,  5,  8, 10, 12, 12, 12], // 4  Gold
    [  0,  0,  0,  5,  5,  8,  8, 10, 10], // 5  Bishop
    [  0,  0,  0,  5,  5,  8,  8, 10, 10], // 6  Rook
    [ 20, 15, 10,  5,  0,-10,-20,-30,-40], // 7  King   (stay home!)
    [  0,  0,  0,  5, 10, 12, 15, 15, 15], // 8  ProPawn
    [  0,  0,  0,  5, 10, 12, 15, 15, 15], // 9  ProLance
    [  0,  0,  0,  5, 10, 12, 15, 15, 15], // 10 ProKnight
    [  0,  0,  0,  5, 10, 12, 15, 15, 15], // 11 ProSilver
    [  0,  0,  0,  5,  8, 10, 12, 15, 15], // 12 ProBishop
    [  0,  0,  0,  5,  8, 10, 12, 15, 15], // 13 ProRook
];

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

// ---------------------------------------------------------------------------
// King safety
// ---------------------------------------------------------------------------

// Tuning knobs — kept as named constants so they're easy to adjust.
const SHIELD_OWN: i32 = 10;   // friendly piece in king's 3×3 neighbourhood
const SHIELD_GOLD: i32 = 8;   // extra for gold / gold-equivalent defenders
const SHIELD_SILVER: i32 = 4; // extra for silver
const EXPOSED_SQ: i32 = -12;  // penalty for each empty neighbour square

/// Bitboard of gold-equivalent pieces (Gold + all four promoted minors, which
/// move like Gold) for `color_idx`.
#[inline]
fn gold_like_bb(board: &Board, color_idx: usize) -> crate::bitboard::Bitboard {
    board.pieces[color_idx][PieceType::Gold.index()]
        | board.pieces[color_idx][PieceType::ProPawn.index()]
        | board.pieces[color_idx][PieceType::ProLance.index()]
        | board.pieces[color_idx][PieceType::ProKnight.index()]
        | board.pieces[color_idx][PieceType::ProSilver.index()]
}

/// King-safety score for one side (always positive = safer king).
/// Scans the 3×3 neighbourhood of the king:
///   • friendly piece present  → +SHIELD_OWN (+ extra for gold/silver type)
///   • square empty or enemy   → EXPOSED_SQ
fn king_safety(board: &Board, color_idx: usize) -> i32 {
    let king_bb = board.pieces[color_idx][PieceType::King.index()];
    if king_bb.is_empty() {
        return 0;
    }
    let king_sq = king_bb.lsb();
    let own = board.color_bb[color_idx];
    let gold = gold_like_bb(board, color_idx);
    let silver = board.pieces[color_idx][PieceType::Silver.index()];

    let mut score = 0i32;
    for df in [-1i8, 0, 1] {
        for dr in [-1i8, 0, 1] {
            if df == 0 && dr == 0 {
                continue;
            }
            let Some(nsq) = add_step(king_sq, df, dr) else {
                continue;
            };
            if own.contains(nsq) {
                score += SHIELD_OWN;
                if gold.contains(nsq) {
                    score += SHIELD_GOLD;
                } else if silver.contains(nsq) {
                    score += SHIELD_SILVER;
                }
            } else {
                score += EXPOSED_SQ;
            }
        }
    }
    score
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Returns the advancement index (0 = own back rank, 8 = deepest enemy) for
/// a piece belonging to the given side.
#[inline]
fn adv_idx(sq: Square, is_black: bool) -> usize {
    let r = rank_of(sq);
    // Black advances toward rank_idx 0; White toward rank_idx 8.
    if is_black { (8 - r) as usize } else { r as usize }
}

/// Static evaluation from the perspective of `board.side_to_move`.
/// Positive = good for the side to move.
///
/// Combines material (on-board + in-hand) with rank-based positional bonuses
/// from `RANK_PST`. Hand pieces have no positional component (they are off the
/// board). Promoted pieces in hand are impossible in Shogi (they revert on
/// capture), so the hand loop only covers indices 0–6.
pub fn eval(board: &Board) -> i32 {
    let stm = board.side_to_move.index();
    let opp = board.side_to_move.opponent().index();
    let stm_is_black = stm == 0; // Color::Black has index 0
    let mut score = 0i32;

    for pt_idx in 0..14usize {
        let val = PIECE_VALUE[pt_idx];
        // Material
        score += board.pieces[stm][pt_idx].count() as i32 * val;
        score -= board.pieces[opp][pt_idx].count() as i32 * val;
        // Positional (rank-based PST)
        for sq in board.pieces[stm][pt_idx].iter_squares() {
            score += RANK_PST[pt_idx][adv_idx(sq, stm_is_black)];
        }
        for sq in board.pieces[opp][pt_idx].iter_squares() {
            score -= RANK_PST[pt_idx][adv_idx(sq, !stm_is_black)];
        }
    }

    for pt_idx in 0..7usize {
        let val = PIECE_VALUE[pt_idx];
        score += board.hand[stm][pt_idx] as i32 * val;
        score -= board.hand[opp][pt_idx] as i32 * val;
    }

    // King safety
    score += king_safety(board, stm);
    score -= king_safety(board, opp);

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

    // --- PST tests ---

    #[test]
    fn test_pst_pawn_rank_monotone() {
        // Pawn PST should increase with advancement (adv_idx 0 → 8).
        for i in 0..8 {
            assert!(RANK_PST[PieceType::Pawn.index()][i + 1]
                >= RANK_PST[PieceType::Pawn.index()][i],
                "pawn PST must be non-decreasing: idx {} vs {}", i, i + 1);
        }
    }

    #[test]
    fn test_pst_king_rank_monotone_decreasing() {
        // King PST should decrease with advancement (king wants own back rank).
        for i in 0..8 {
            assert!(RANK_PST[PieceType::King.index()][i + 1]
                <= RANK_PST[PieceType::King.index()][i],
                "king PST must be non-increasing: idx {} vs {}", i, i + 1);
        }
    }

    #[test]
    fn test_pst_values_in_reasonable_range() {
        for pt in PieceType::ALL {
            for &v in &RANK_PST[pt.index()] {
                assert!(v >= -50 && v <= 50,
                    "PST value {v} out of [-50, 50] for {:?}", pt);
            }
        }
    }

    #[test]
    fn test_pst_adv_idx_black() {
        use crate::types::square;
        // Black piece at rank_idx 8 (own back rank) → adv_idx 0
        assert_eq!(adv_idx(square(0, 8), true), 0);
        // Black piece at rank_idx 0 (deep enemy) → adv_idx 8
        assert_eq!(adv_idx(square(0, 0), true), 8);
        // Black piece at rank_idx 4 (midfield) → adv_idx 4
        assert_eq!(adv_idx(square(4, 4), true), 4);
    }

    #[test]
    fn test_pst_adv_idx_white() {
        use crate::types::square;
        // White piece at rank_idx 0 (own back rank) → adv_idx 0
        assert_eq!(adv_idx(square(0, 0), false), 0);
        // White piece at rank_idx 8 (deep enemy) → adv_idx 8
        assert_eq!(adv_idx(square(0, 8), false), 8);
        // White piece at rank_idx 4 (midfield) → adv_idx 4
        assert_eq!(adv_idx(square(4, 4), false), 4);
    }

    // --- King safety tests ---

    #[test]
    fn test_king_safety_startpos_symmetric() {
        // Startpos is perfectly symmetric; king_safety(Black) == king_safety(White)
        // so the net contribution to eval is zero.
        let board = Board::startpos();
        let ks_black = king_safety(&board, 0);
        let ks_white = king_safety(&board, 1);
        assert_eq!(ks_black, ks_white,
            "startpos king safety must be symmetric: black={ks_black} white={ks_white}");
    }

    #[test]
    fn test_king_safety_shielded_beats_exposed() {
        // A king at the back rank with two golds beside it should score higher
        // than a bare king alone in the middle of the board.
        //
        // shielded: "k8/9/9/9/9/9/9/9/GKG6 b - 1"
        //   Black king at (file 8, rank 9) = file_idx 1, rank_idx 8
        //   Golds at file 9 and file 7, rank 9
        // bare:     "k8/9/9/9/4K4/9/9/9/9 b - 1"
        //   Black king alone at rank 5 (rank_idx 4)
        let shielded = Board::from_sfen("k8/9/9/9/9/9/9/9/GKG6 b - 1").unwrap();
        let bare     = Board::from_sfen("k8/9/9/9/4K4/9/9/9/9 b - 1").unwrap();
        assert!(king_safety(&shielded, 0) > king_safety(&bare, 0),
            "shielded king must score better than exposed king");
    }

    #[test]
    fn test_king_safety_gold_beats_silver() {
        // Gold-like defenders score higher than silver defenders.
        let with_gold   = Board::from_sfen("k8/9/9/9/9/9/9/9/GK7 b - 1").unwrap();
        let with_silver = Board::from_sfen("k8/9/9/9/9/9/9/9/SK7 b - 1").unwrap();
        assert!(king_safety(&with_gold, 0) > king_safety(&with_silver, 0),
            "gold defender must outscore silver defender");
    }

    #[test]
    fn test_eval_startpos_still_zero_with_king_safety() {
        // Full eval (material + PST + king safety) on startpos must be zero.
        let board = Board::startpos();
        assert_eq!(eval(&board), 0, "startpos eval must remain zero");
    }

    #[test]
    fn test_eval_symmetric_flip_with_king_safety() {
        // eval(pos, stm=A) == -eval(pos, stm=B) must hold with king safety.
        let mut board = Board::startpos();
        board.hand[0][PieceType::Rook.index()] += 1;
        let score_black = eval(&board);
        board.side_to_move = board.side_to_move.opponent();
        let score_white = eval(&board);
        assert_eq!(score_black, -score_white,
            "flipping side_to_move must negate the score");
    }
}
