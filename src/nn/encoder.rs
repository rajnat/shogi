/// Board encoder — converts a `Board` position to a float32 tensor.
///
/// # Plane layout  (87 planes × 9 × 9)
///
/// ```text
///  planes  0–13   Black pieces,  one binary plane per PieceType (index 0–13)
///  planes 14–27   White pieces,  same ordering
///  planes 28–56   Black hand, thermometer-coded per piece type (see HAND_MAX)
///  planes 57–85   White hand, thermometer-coded per piece type
///  plane  86      Side to move  (1.0 everywhere = Black to move, 0.0 = White)
/// ```
///
/// **Thermometer encoding for hand pieces:** for a type with `max_k` planes, plane
/// k (0-indexed) is filled with 1.0 iff the player holds **more than k** of that
/// piece.  So holding 3 rooks lights up planes k=0, k=1 but not k=2 (since
/// `HAND_MAX[Rook] = 2`, k ranges over 0..2).
///
/// **Spatial layout:** dim 1 = rank_idx (0 = rank 1 … 8 = rank 9),
/// dim 2 = file_idx (0 = file 9 … 8 = file 1).  Square `sq` maps to
/// `[rank_idx = sq % 9][file_idx = sq / 9]`.
///
/// # Why 87 planes and not 119?
/// The roadmap borrowed 119 from the AlphaZero Chess paper (`[119, 8, 8]`),
/// which encodes 8 history timesteps.  Shogi has hand pieces instead of
/// history; this single-position encoding naturally lands at 87 planes.
use tch::{Kind, Tensor};
use crate::board::Board;
use crate::types::Color;

/// Total number of input planes.
pub const NUM_PLANES: usize = 87;

/// Thermometer depth for each of the 7 hand-piece types.
/// Order matches `PieceType::HAND_TYPES`: Pawn, Lance, Knight, Silver, Gold, Bishop, Rook.
///
/// Each player can hold at most `HAND_MAX[i]` of piece type i in realistic play;
/// the encoding saturates silently above this (the network never sees the difference).
pub const HAND_MAX: [usize; 7] = [
    9, // Pawn   — 9 per side, all capturable
    4, // Lance
    4, // Knight
    4, // Silver
    4, // Gold
    2, // Bishop
    2, // Rook
];

/// Encode `board` as a `[NUM_PLANES, 9, 9]` float32 tensor on CPU.
pub fn encode(board: &Board) -> Tensor {
    let mut data = vec![0.0_f32; NUM_PLANES * 81];

    // Inline helpers that borrow `data`.
    // Set one square (sq encodes file_idx=sq/9, rank_idx=sq%9).
    let mut set_sq = |plane: usize, sq: u8| {
        let sq = sq as usize;
        data[plane * 81 + (sq % 9) * 9 + (sq / 9)] = 1.0;
    };

    // Fill an entire 9×9 plane with 1.0.
    let mut fill_plane = |plane: usize| {
        let base = plane * 81;
        for i in 0..81 {
            data[base + i] = 1.0;
        }
    };

    // Planes 0–27: board piece planes.
    for color in 0..2usize {
        for pt in 0..14usize {
            let plane = color * 14 + pt;
            for sq in board.pieces[color][pt].iter_squares() {
                set_sq(plane, sq);
            }
        }
    }

    // Planes 28–85: hand piece thermometer planes.
    let mut plane = 28usize;
    for color in 0..2usize {
        for (pt_idx, &max_k) in HAND_MAX.iter().enumerate() {
            let count = board.hand[color][pt_idx] as usize;
            for k in 0..max_k {
                if count > k {
                    fill_plane(plane);
                }
                plane += 1;
            }
        }
    }
    debug_assert_eq!(plane, 86, "hand plane count does not reach plane 86");

    // Plane 86: side to move.
    if board.side_to_move == Color::Black {
        fill_plane(86);
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
    use crate::moves::{make_move_full, unmake_move_full};
    use crate::movegen::generate_legal_moves;

    /// Read one cell from an encoded tensor.
    fn get(t: &Tensor, plane: usize, rank: usize, file: usize) -> f32 {
        t.double_value(&[plane as i64, rank as i64, file as i64]) as f32
    }

    /// Count how many cells in a plane are 1.0.
    fn count_ones(t: &Tensor, plane: usize) -> usize {
        (0..9)
            .flat_map(|r| (0..9).map(move |f| (r, f)))
            .filter(|&(r, f)| (get(t, plane, r, f) - 1.0).abs() < 1e-6)
            .count()
    }

    // ----- Shape and type -----

    #[test]
    fn test_output_shape() {
        let board = Board::startpos();
        let t = encode(&board);
        assert_eq!(t.size(), vec![NUM_PLANES as i64, 9, 9]);
    }

    #[test]
    fn test_output_kind_is_float() {
        let t = encode(&Board::startpos());
        assert_eq!(t.kind(), Kind::Float);
    }

    #[test]
    fn test_all_values_in_zero_one() {
        let t = encode(&Board::startpos());
        let flat: Vec<f32> = t.reshape([-1]).into();
        for &v in &flat {
            assert!(v == 0.0 || v == 1.0, "unexpected value {v}");
        }
    }

    // ----- Board planes (startpos) -----

    #[test]
    fn test_black_pawn_plane_startpos() {
        // Startpos: Black pawns on rank_idx=6, file_idx=0..8 (all 9 files).
        let t = encode(&Board::startpos());
        let plane = 0; // Black Pawn = PieceType::Pawn = index 0
        assert_eq!(count_ones(&t, plane), 9, "9 black pawns");
        for file in 0..9 {
            assert!(
                (get(&t, plane, 6, file) - 1.0).abs() < 1e-6,
                "black pawn missing at rank=6 file={file}"
            );
        }
    }

    #[test]
    fn test_black_rook_plane_startpos() {
        // Startpos rank 8 (rank_idx=7): "1B5R1" → Rook at file_idx=7.
        let t = encode(&Board::startpos());
        let plane = 6; // PieceType::Rook
        assert_eq!(count_ones(&t, plane), 1);
        assert!((get(&t, plane, 7, 7) - 1.0).abs() < 1e-6, "black rook at [7][7]");
    }

    #[test]
    fn test_black_bishop_plane_startpos() {
        // Startpos rank 8 (rank_idx=7): "1B5R1" → Bishop at file_idx=1.
        let t = encode(&Board::startpos());
        let plane = 5; // PieceType::Bishop
        assert_eq!(count_ones(&t, plane), 1);
        assert!((get(&t, plane, 7, 1) - 1.0).abs() < 1e-6, "black bishop at [7][1]");
    }

    #[test]
    fn test_white_pawn_plane_startpos() {
        // White pawns at rank_idx=2.
        let t = encode(&Board::startpos());
        let plane = 14; // White Pawn = 14 + 0
        assert_eq!(count_ones(&t, plane), 9, "9 white pawns");
        for file in 0..9 {
            assert!(
                (get(&t, plane, 2, file) - 1.0).abs() < 1e-6,
                "white pawn missing at rank=2 file={file}"
            );
        }
    }

    #[test]
    fn test_white_rook_plane_startpos() {
        // Startpos rank 2 (rank_idx=1): "1r5b1" → rook at file_idx=1.
        let t = encode(&Board::startpos());
        let plane = 20; // White Rook = 14 + 6
        assert_eq!(count_ones(&t, plane), 1);
        assert!((get(&t, plane, 1, 1) - 1.0).abs() < 1e-6, "white rook at [1][1]");
    }

    #[test]
    fn test_white_bishop_plane_startpos() {
        // Startpos rank 2 (rank_idx=1): "1r5b1" → bishop at file_idx=7.
        let t = encode(&Board::startpos());
        let plane = 19; // White Bishop = 14 + 5
        assert_eq!(count_ones(&t, plane), 1);
        assert!((get(&t, plane, 1, 7) - 1.0).abs() < 1e-6, "white bishop at [1][7]");
    }

    #[test]
    fn test_total_piece_count_startpos() {
        // Startpos has 40 pieces total (20 per side × 2 = 40).
        // The sum of all 1s in planes 0–27 must be 40.
        let t = encode(&Board::startpos());
        let total: usize = (0..28).map(|p| count_ones(&t, p)).sum();
        assert_eq!(total, 40, "total piece count in startpos should be 40");
    }

    // ----- Hand planes (startpos — empty hand) -----

    #[test]
    fn test_no_hand_pieces_in_startpos() {
        let t = encode(&Board::startpos());
        for plane in 28..86 {
            assert_eq!(
                count_ones(&t, plane),
                0,
                "hand plane {plane} should be empty at startpos"
            );
        }
    }

    #[test]
    fn test_hand_thermometer_one_pawn() {
        // Black has 1 pawn in hand → only hand-pawn plane 0 (k=0, count>0) = 1.
        let mut board = Board::startpos();
        board.hand[0][0] = 1; // Black Pawn count = 1
        let t = encode(&board);
        // Black pawn hand planes start at plane 28; max_k=9.
        assert!((get(&t, 28, 0, 0) - 1.0).abs() < 1e-6, "plane 28 should be lit");
        assert!((get(&t, 29, 0, 0)).abs() < 1e-6, "plane 29 should be dark (count=1 ≤ k=1)");
    }

    #[test]
    fn test_hand_thermometer_three_pawns() {
        // Black has 3 pawns in hand → planes 28,29,30 lit; plane 31 dark.
        let mut board = Board::startpos();
        board.hand[0][0] = 3;
        let t = encode(&board);
        for k in 0..3 {
            assert!(
                (get(&t, 28 + k, 0, 0) - 1.0).abs() < 1e-6,
                "plane {} should be lit (count=3 > k={})", 28 + k, k
            );
        }
        assert!((get(&t, 31, 0, 0)).abs() < 1e-6, "plane 31 should be dark");
    }

    #[test]
    fn test_hand_thermometer_white_rook() {
        // White has 2 rooks in hand → white rook planes (last 2 hand planes before plane 86) lit.
        // White hand planes: 57..85 (29 planes), same layout as Black.
        // White rook planes = 57 + 29 - 2 + offset = plane 83 and 84.
        // Cumulative: pawn=9, lance=4, knight=4, silver=4, gold=4, bishop=2 → 27 planes before rook.
        // So white rook starts at 57 + 27 = 84, HAND_MAX[Rook]=2 → planes 84 and 85.
        let mut board = Board::startpos();
        board.hand[1][6] = 2; // White Rook count = 2
        let t = encode(&board);
        assert!((get(&t, 84, 0, 0) - 1.0).abs() < 1e-6, "white rook plane 84 (k=0) lit");
        assert!((get(&t, 85, 0, 0) - 1.0).abs() < 1e-6, "white rook plane 85 (k=1) lit");
    }

    // ----- Side-to-move plane -----

    #[test]
    fn test_side_to_move_black() {
        let t = encode(&Board::startpos());
        assert_eq!(count_ones(&t, 86), 81, "plane 86 must be all-ones when Black to move");
    }

    #[test]
    fn test_side_to_move_white() {
        // Make one move so White is to move.
        let mut board = Board::startpos();
        let mut moves = Vec::new();
        generate_legal_moves(&mut board, &mut moves);
        make_move_full(&mut board, moves[0]);

        let t = encode(&board);
        assert_eq!(count_ones(&t, 86), 0, "plane 86 must be all-zeros when White to move");
    }

    #[test]
    fn test_side_to_move_toggles_after_move_unmove() {
        let mut board = Board::startpos();
        let mut moves = Vec::new();
        generate_legal_moves(&mut board, &mut moves);
        let mv = moves[0];
        let undo = make_move_full(&mut board, mv);

        let t_white = encode(&board);
        assert_eq!(count_ones(&t_white, 86), 0);

        unmake_move_full(&mut board, mv, &undo);
        let t_black = encode(&board);
        assert_eq!(count_ones(&t_black, 86), 81);
    }

    // ----- Determinism -----

    #[test]
    fn test_encode_is_deterministic() {
        let board = Board::startpos();
        let t1 = encode(&board);
        let t2 = encode(&board);
        let flat1: Vec<f32> = t1.reshape([-1]).into();
        let flat2: Vec<f32> = t2.reshape([-1]).into();
        assert_eq!(flat1, flat2);
    }

    // ----- Regression: distinct positions encode differently -----

    #[test]
    fn test_different_positions_differ() {
        let board1 = Board::startpos();
        let mut board2 = Board::startpos();
        let mut moves = Vec::new();
        generate_legal_moves(&mut board2, &mut moves);
        make_move_full(&mut board2, moves[0]);

        let flat1: Vec<f32> = encode(&board1).reshape([-1]).into();
        let flat2: Vec<f32> = encode(&board2).reshape([-1]).into();
        assert_ne!(flat1, flat2, "different positions must produce different tensors");
    }
}
