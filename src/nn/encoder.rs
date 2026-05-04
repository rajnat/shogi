/// Board encoder — `Board` → float32 tensor of shape `[119, 9, 9]`.
///
/// # Plane layout
///
/// ```text
///   0– 13  Black pieces  (binary, one plane per PieceType index 0–13)
///  14– 27  White pieces  (binary, same ordering)
///  28– 65  Black hand, thermometer-coded  (HAND_MAX = [18,4,4,4,4,2,2])
///  66–103  White hand, thermometer-coded
/// 104–110  Current-player combined piece planes  (7 base types, base|promoted)
/// 111–117  Opponent   combined piece planes
///     118  Side to move  (1.0 everywhere = Black, 0.0 = White)
/// ```
///
/// ## Hand thermometer
/// For each (color, piece-type) pair, `HAND_MAX[type]` consecutive planes are
/// allocated.  Plane *k* (0-indexed within the group) is filled with 1.0 iff
/// the player holds **more than k** of that piece.  Holding 3 pawns lights up
/// k = 0, 1, 2; leaves k = 3..17 dark.  Pawn depth 18 covers all 18 pawns in
/// the game (9 per side, all capturable); other maxima match per-side counts.
///
/// ## Combined planes (104–117)
/// Perspective-relative shortcut features keyed to the side to move.  Plane
/// `104 + i` shows every square where the *current player* has a piece of
/// `HAND_TYPES[i]` **or** its promoted form (e.g. Rook | Dragon); planes
/// `111 + i` do the same for the *opponent*.  These collapse the base/promoted
/// distinction and help the network recognise piece-family threats without
/// first summing two sparse planes.
///
/// ## Spatial layout
/// Dim 1 = rank_idx (0 = rank 1 … 8 = rank 9),
/// dim 2 = file_idx (0 = file 9 … 8 = file 1).
/// Square `sq` maps to `[sq % 9][sq / 9]`.
use tch::{Kind, Tensor};
use crate::board::Board;
use crate::types::{Color, PieceType};

/// Total input planes.
pub const NUM_PLANES: usize = 119;

/// Thermometer depth per hand-piece type.
/// Order matches `PieceType::HAND_TYPES`: Pawn, Lance, Knight, Silver, Gold, Bishop, Rook.
pub const HAND_MAX: [usize; 7] = [
    18, // Pawn:   18 in play (9 per side, all capturable)
    4,  // Lance
    4,  // Knight
    4,  // Silver
    4,  // Gold
    2,  // Bishop
    2,  // Rook
];

// Plane 28 + cumsum(HAND_MAX) × 2 = 28 + 38 + 38 = 104  (start of combined)
// 104 + 7 + 7 + 1 = 119  ✓

#[inline]
fn set_sq(data: &mut [f32], plane: usize, sq: u8) {
    let sq = sq as usize;
    data[plane * 81 + (sq % 9) * 9 + (sq / 9)] = 1.0;
}

#[inline]
fn fill_plane(data: &mut [f32], plane: usize) {
    data[plane * 81..plane * 81 + 81].fill(1.0);
}

/// Encode `board` as a `[NUM_PLANES, 9, 9]` float32 tensor on CPU.
pub fn encode(board: &Board) -> Tensor {
    let mut data = vec![0.0_f32; NUM_PLANES * 81];

    // ------------------------------------------------------------------
    // Planes 0–27: board piece planes
    // ------------------------------------------------------------------
    for color in 0..2usize {
        for pt in 0..14usize {
            let plane = color * 14 + pt;
            for sq in board.pieces[color][pt].iter_squares() {
                set_sq(&mut data, plane, sq);
            }
        }
    }

    // ------------------------------------------------------------------
    // Planes 28–103: hand thermometer
    // ------------------------------------------------------------------
    let mut plane = 28usize;
    for color in 0..2usize {
        for (pt_idx, &max_k) in HAND_MAX.iter().enumerate() {
            let count = board.hand[color][pt_idx] as usize;
            for k in 0..max_k {
                if count > k {
                    fill_plane(&mut data, plane);
                }
                plane += 1;
            }
        }
    }
    debug_assert_eq!(plane, 104, "hand planes must end at 104");

    // ------------------------------------------------------------------
    // Planes 104–110: current-player combined (base | promoted)
    // Planes 111–117: opponent combined
    // ------------------------------------------------------------------
    let cur = board.side_to_move.index();
    let opp = board.side_to_move.opponent().index();

    for (i, base_pt) in PieceType::HAND_TYPES.iter().enumerate() {
        for &(player, base_plane) in &[(cur, 104 + i), (opp, 111 + i)] {
            for sq in board.pieces[player][base_pt.index()].iter_squares() {
                set_sq(&mut data, base_plane, sq);
            }
            if let Some(ppt) = base_pt.promoted() {
                for sq in board.pieces[player][ppt.index()].iter_squares() {
                    set_sq(&mut data, base_plane, sq);
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Plane 118: side to move
    // ------------------------------------------------------------------
    if board.side_to_move == Color::Black {
        fill_plane(&mut data, 118);
    }

    Tensor::from_slice(&data)
        .to_kind(Kind::Float)
        .reshape([NUM_PLANES as i64, 9, 9])
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    use crate::movegen::generate_legal_moves;
    use crate::moves::{make_move_full, unmake_move_full};
    use crate::types::square;

    fn get(t: &Tensor, plane: usize, rank: usize, file: usize) -> f32 {
        t.double_value(&[plane as i64, rank as i64, file as i64]) as f32
    }

    fn to_vec(t: Tensor) -> Vec<f32> {
        let flat = t.reshape([-1]).contiguous();
        let n = flat.numel();
        let mut out = vec![0.0f32; n];
        flat.copy_data(&mut out, n);
        out
    }

    fn count_ones(t: &Tensor, plane: usize) -> usize {
        (0..9)
            .flat_map(|r| (0..9).map(move |f| (r, f)))
            .filter(|&(r, f)| (get(t, plane, r, f) - 1.0).abs() < 1e-6)
            .count()
    }

    // ----- Shape / type -----

    #[test]
    fn test_output_shape() {
        assert_eq!(
            encode(&Board::startpos()).size(),
            vec![NUM_PLANES as i64, 9, 9]
        );
    }

    #[test]
    fn test_output_kind_float() {
        assert_eq!(encode(&Board::startpos()).kind(), Kind::Float);
    }

    #[test]
    fn test_all_values_zero_or_one() {
        let flat: Vec<f32> = to_vec(encode(&Board::startpos()));
        for &v in &flat {
            assert!(v == 0.0 || v == 1.0, "unexpected value {v}");
        }
    }

    // ----- Board planes (startpos) -----

    #[test]
    fn test_black_pawn_plane() {
        // Black pawns at rank_idx=6, all 9 files.
        let t = encode(&Board::startpos());
        assert_eq!(count_ones(&t, 0), 9);
        for file in 0..9 {
            assert!(
                (get(&t, 0, 6, file) - 1.0).abs() < 1e-6,
                "missing black pawn at rank=6 file={file}"
            );
        }
    }

    #[test]
    fn test_black_rook_plane() {
        // Rank 8 (rank_idx=7), "1B5R1": Rook at file_idx=7.
        let t = encode(&Board::startpos());
        assert_eq!(count_ones(&t, 6), 1);
        assert!((get(&t, 6, 7, 7) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_black_bishop_plane() {
        // Rank 8 (rank_idx=7), "1B5R1": Bishop at file_idx=1.
        let t = encode(&Board::startpos());
        assert_eq!(count_ones(&t, 5), 1);
        assert!((get(&t, 5, 7, 1) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_white_pawn_plane() {
        // White pawns at rank_idx=2.
        let t = encode(&Board::startpos());
        let plane = 14; // White Pawn = 14 + 0
        assert_eq!(count_ones(&t, plane), 9);
        for file in 0..9 {
            assert!(
                (get(&t, plane, 2, file) - 1.0).abs() < 1e-6,
                "missing white pawn at rank=2 file={file}"
            );
        }
    }

    #[test]
    fn test_white_rook_plane() {
        // Rank 2 (rank_idx=1), "1r5b1": rook at file_idx=1.
        let t = encode(&Board::startpos());
        let plane = 20; // White Rook = 14 + 6
        assert_eq!(count_ones(&t, plane), 1);
        assert!((get(&t, plane, 1, 1) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_white_bishop_plane() {
        // Rank 2 (rank_idx=1), "1r5b1": bishop at file_idx=7.
        let t = encode(&Board::startpos());
        let plane = 19; // White Bishop = 14 + 5
        assert_eq!(count_ones(&t, plane), 1);
        assert!((get(&t, plane, 1, 7) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_total_piece_count() {
        // 40 pieces on the board at startpos (20 per side).
        let t = encode(&Board::startpos());
        let total: usize = (0..28).map(|p| count_ones(&t, p)).sum();
        assert_eq!(total, 40);
    }

    // ----- Hand planes -----
    //
    // Black hand plane offsets (starting at 28):
    //   Pawn  28–45 (18 planes), Lance 46–49, Knight 50–53,
    //   Silver 54–57, Gold 58–61, Bishop 62–63, Rook 64–65.
    // White hand starts at 66, same layout.

    #[test]
    fn test_no_hand_at_startpos() {
        let t = encode(&Board::startpos());
        for plane in 28..104 {
            assert_eq!(
                count_ones(&t, plane),
                0,
                "hand plane {plane} should be 0 at startpos"
            );
        }
    }

    #[test]
    fn test_hand_thermometer_one_pawn() {
        let mut board = Board::startpos();
        board.hand[0][0] = 1; // Black holds 1 pawn
        let t = encode(&board);
        assert!((get(&t, 28, 0, 0) - 1.0).abs() < 1e-6, "plane 28 (k=0) should be lit");
        assert!(get(&t, 29, 0, 0).abs() < 1e-6, "plane 29 (k=1) should be dark");
    }

    #[test]
    fn test_hand_thermometer_three_pawns() {
        let mut board = Board::startpos();
        board.hand[0][0] = 3; // Black holds 3 pawns
        let t = encode(&board);
        for k in 0..3 {
            assert!(
                (get(&t, 28 + k, 0, 0) - 1.0).abs() < 1e-6,
                "plane {} should be lit", 28 + k
            );
        }
        assert!(get(&t, 31, 0, 0).abs() < 1e-6, "plane 31 should be dark");
    }

    #[test]
    fn test_hand_thermometer_white_rook() {
        // White Rook starts at plane 66 + (18+4+4+4+4+2) = 66+36 = 102.
        let mut board = Board::startpos();
        board.hand[1][6] = 2; // White holds 2 rooks
        let t = encode(&board);
        assert!((get(&t, 102, 0, 0) - 1.0).abs() < 1e-6, "plane 102 (k=0) should be lit");
        assert!((get(&t, 103, 0, 0) - 1.0).abs() < 1e-6, "plane 103 (k=1) should be lit");
    }

    // ----- Combined planes (104–117) -----

    #[test]
    fn test_combined_pawn_equals_board_pawn_at_startpos() {
        // No promoted pawns at startpos → combined == board.
        let t = encode(&Board::startpos());
        // Black to move: plane 104 = Black Pawn|ProPawn = Black Pawn only.
        assert_eq!(count_ones(&t, 104), 9);
        for file in 0..9 {
            assert!((get(&t, 104, 6, file) - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_combined_rook_includes_promoted() {
        // Add a ProRook for Black alongside the existing Rook.
        // startpos: Black Rook at file_idx=7, rank_idx=7 (sq = 7*9+7 = 70).
        let mut board = Board::startpos();
        // Place a Black ProRook at a different square.
        board.set_piece(square(3, 3), Color::Black, PieceType::ProRook);
        let t = encode(&board);
        // plane 110 = Black Rook | Black ProRook (Black is to move → current player).
        assert_eq!(count_ones(&t, 110), 2, "Rook + ProRook should both appear in combined");
        // Board plane 6 (Black Rook) still has only 1.
        assert_eq!(count_ones(&t, 6), 1);
        // Board plane 13 (Black ProRook) has the new piece.
        assert_eq!(count_ones(&t, 13), 1);
    }

    #[test]
    fn test_combined_swaps_on_side_change() {
        // After one move White is to move; planes 104–110 now show White's pieces,
        // 111–117 show Black's.
        let mut board = Board::startpos();
        let mut moves = Vec::new();
        generate_legal_moves(&mut board, &mut moves);
        make_move_full(&mut board, moves[0]);

        let t = encode(&board);
        // White pawns at rank_idx=2; plane 104 (current=White) should show them.
        assert_eq!(count_ones(&t, 104), 9, "after Black moves, plane 104 = White's pawns");
        // plane 111 (opponent=Black) should show Black's pawns at rank_idx=6.
        // Note: the pawn that moved may shift the count by 1, but most pawns stay.
        let black_pawn_count = count_ones(&t, 111);
        assert!(black_pawn_count >= 8, "plane 111 = Black's pawns ({black_pawn_count})");
    }

    // ----- Side-to-move plane (118) -----

    #[test]
    fn test_side_to_move_black() {
        assert_eq!(count_ones(&encode(&Board::startpos()), 118), 81);
    }

    #[test]
    fn test_side_to_move_white() {
        let mut board = Board::startpos();
        let mut moves = Vec::new();
        generate_legal_moves(&mut board, &mut moves);
        make_move_full(&mut board, moves[0]);
        assert_eq!(count_ones(&encode(&board), 118), 0);
    }

    #[test]
    fn test_side_to_move_toggles() {
        let mut board = Board::startpos();
        let mut moves = Vec::new();
        generate_legal_moves(&mut board, &mut moves);
        let mv = moves[0];
        let undo = make_move_full(&mut board, mv);
        assert_eq!(count_ones(&encode(&board), 118), 0);
        unmake_move_full(&mut board, mv, &undo);
        assert_eq!(count_ones(&encode(&board), 118), 81);
    }

    // ----- Determinism / regression -----

    #[test]
    fn test_deterministic() {
        let board = Board::startpos();
        assert_eq!(to_vec(encode(&board)), to_vec(encode(&board)));
    }

    #[test]
    fn test_distinct_positions_differ() {
        let board1 = Board::startpos();
        let mut board2 = Board::startpos();
        let mut moves = Vec::new();
        generate_legal_moves(&mut board2, &mut moves);
        make_move_full(&mut board2, moves[0]);
        assert_ne!(to_vec(encode(&board1)), to_vec(encode(&board2)));
    }
}
