/// USI (Universal Shogi Interface) protocol handler
///
/// Implements the stdio loop that lets this engine communicate with any
/// USI-compatible GUI (ShogiGUI, Shogidroid, etc.).
///
/// Protocol flow:
///   GUI → usi          engine → id name / id author / usiok
///   GUI → isready      engine → readyok
///   GUI → position …   (engine sets internal board state, no response)
///   GUI → go …         engine → bestmove <move>
///   GUI → quit         (engine exits)
use std::io::{self, BufRead, Write};

use crate::board::Board;
use crate::movegen::generate_legal_moves;
use crate::moves::make_move_full;
use crate::types::{Color, Move, PieceType};

const ENGINE_NAME: &str = "ShogiCore";
const ENGINE_AUTHOR: &str = "rajnat";

// ---------------------------------------------------------------------------
// USI move parsing
// ---------------------------------------------------------------------------

/// Parse a single USI square token like "7g" → internal Square index.
/// File chars '1'–'9': display file N maps to file_idx = 9 - N.
/// Rank chars 'a'–'i': 'a' = rank_idx 0, 'i' = rank_idx 8.
fn parse_usi_square(file_char: char, rank_char: char) -> Option<u8> {
    let file_n = file_char as i32 - '0' as i32; // 1–9
    let rank_n = rank_char as i32 - 'a' as i32; // 0–8
    if !(1..=9).contains(&file_n) || !(0..=8).contains(&rank_n) {
        return None;
    }
    let file_idx = (9 - file_n) as u8;
    let rank_idx = rank_n as u8;
    Some(file_idx * 9 + rank_idx)
}

/// Convert a USI move string to our internal Move type.
///
/// Formats:
///   "7g7f"   — normal move
///   "8h2b+"  — normal move with promotion
///   "P*5e"   — pawn drop
///
/// Requires the current board to look up the piece on the from-square for
/// normal moves.
pub fn parse_usi_move(s: &str, board: &Board) -> Option<Move> {
    let bytes = s.as_bytes();
    if bytes.len() < 4 {
        return None;
    }

    if bytes[1] == b'*' {
        // Drop move
        let piece_type = match bytes[0] as char {
            'P' => PieceType::Pawn,
            'L' => PieceType::Lance,
            'N' => PieceType::Knight,
            'S' => PieceType::Silver,
            'G' => PieceType::Gold,
            'B' => PieceType::Bishop,
            'R' => PieceType::Rook,
            _ => return None,
        };
        let to = parse_usi_square(bytes[2] as char, bytes[3] as char)?;
        Some(Move::new_drop(piece_type, to))
    } else {
        // Normal (or promotion) move
        let from = parse_usi_square(bytes[0] as char, bytes[1] as char)?;
        let to = parse_usi_square(bytes[2] as char, bytes[3] as char)?;
        let promote = bytes.len() >= 5 && bytes[4] == b'+';

        let (_color, pt) = board.piece_at(from)?;
        Some(Move::new_normal(from, to, pt, promote))
    }
}

// ---------------------------------------------------------------------------
// Position setup
// ---------------------------------------------------------------------------

/// Parse the tokens after the "position" keyword and return the resulting board.
///
/// Handles:
///   position startpos [moves m1 m2 …]
///   position sfen <board> <side> <hand> <ply> [moves m1 m2 …]
fn setup_position(tokens: &[&str]) -> Board {
    let mut idx = 0;

    let mut board = match tokens.get(idx) {
        Some(&"startpos") => {
            idx += 1;
            Board::startpos()
        }
        Some(&"sfen") => {
            idx += 1;
            // SFEN is exactly 4 space-separated tokens: board side hand ply.
            // Find "moves" to know where SFEN ends.
            let moves_pos = tokens[idx..]
                .iter()
                .position(|&t| t == "moves")
                .map(|p| idx + p)
                .unwrap_or(tokens.len());
            let sfen_str = tokens[idx..moves_pos].join(" ");
            idx = moves_pos;
            Board::from_sfen(&sfen_str).unwrap_or_else(|_| Board::startpos())
        }
        _ => return Board::startpos(),
    };

    // Apply the move list if present
    if tokens.get(idx) == Some(&"moves") {
        idx += 1;
        for move_str in &tokens[idx..] {
            if let Some(mv) = parse_usi_move(move_str, &board) {
                make_move_full(&mut board, mv);
            }
        }
    }

    board
}

// ---------------------------------------------------------------------------
// Go command / time management
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
struct GoParams {
    btime_ms: Option<u64>,
    wtime_ms: Option<u64>,
    byoyomi_ms: Option<u64>,
    movetime_ms: Option<u64>,
    infinite: bool,
}

fn parse_go(tokens: &[&str]) -> GoParams {
    let mut p = GoParams::default();
    let mut i = 0;
    while i < tokens.len() {
        let next = || tokens.get(i + 1).and_then(|t| t.parse().ok());
        match tokens[i] {
            "btime"    => { p.btime_ms    = next(); i += 2; }
            "wtime"    => { p.wtime_ms    = next(); i += 2; }
            "byoyomi"  => { p.byoyomi_ms  = next(); i += 2; }
            "movetime" => { p.movetime_ms = next(); i += 2; }
            "infinite" => { p.infinite = true;      i += 1; }
            _          => { i += 1; }
        }
    }
    p
}

/// How many milliseconds the engine may spend on this move.
fn time_budget_ms(params: &GoParams, color: Color) -> u64 {
    if let Some(mt) = params.movetime_ms {
        return mt.saturating_sub(10); // small safety margin
    }
    if params.infinite {
        return u64::MAX / 2;
    }
    let remaining = match color {
        Color::Black => params.btime_ms.unwrap_or(30_000),
        Color::White => params.wtime_ms.unwrap_or(30_000),
    };
    let byoyomi = params.byoyomi_ms.unwrap_or(0);
    // Use ~2% of remaining time, never less than 1 ms
    (remaining / 50).max(1) + byoyomi.saturating_sub(50)
}

// ---------------------------------------------------------------------------
// Move selection (placeholder until M3 adds real search)
// ---------------------------------------------------------------------------

/// Returns the best move for the current position.
///
/// For M2 this is simply the first legal move; M3 will replace this with
/// iterative-deepening alpha-beta within the time budget.
pub fn select_move(board: &mut Board, _budget_ms: u64) -> Option<Move> {
    let mut moves = Vec::with_capacity(128);
    generate_legal_moves(board, &mut moves);
    moves.into_iter().next()
}

// ---------------------------------------------------------------------------
// Main USI loop
// ---------------------------------------------------------------------------

pub fn run_usi_loop() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();

    let mut board = Board::startpos();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }

        let tokens: Vec<&str> = line.split_whitespace().collect();
        match tokens[0] {
            "usi" => {
                writeln!(out, "id name {ENGINE_NAME}").ok();
                writeln!(out, "id author {ENGINE_AUTHOR}").ok();
                writeln!(out, "usiok").ok();
                out.flush().ok();
            }

            "isready" => {
                // Attack tables initialise lazily on first generate_legal_moves call.
                writeln!(out, "readyok").ok();
                out.flush().ok();
            }

            "usinewgame" => {
                board = Board::startpos();
            }

            "position" => {
                board = setup_position(&tokens[1..]);
            }

            "go" => {
                let params = parse_go(&tokens[1..]);
                let budget = time_budget_ms(&params, board.side_to_move);

                let response = select_move(&mut board, budget)
                    .map(|mv| mv.to_usi_string())
                    .unwrap_or_else(|| "resign".to_string());

                writeln!(out, "bestmove {response}").ok();
                out.flush().ok();
            }

            // stop / ponderhit are no-ops until background search is added
            "stop" | "ponderhit" => {}

            "setoption" => {
                // No engine options defined yet; silently accept per spec.
            }

            "quit" => break,

            _ => {
                // Unknown commands must be silently ignored per USI spec.
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;

    #[test]
    fn test_parse_usi_square() {
        // "7g" → file 7 (display) = file_idx 2, rank 'g' = rank_idx 6 → sq = 2*9+6 = 24
        assert_eq!(parse_usi_square('7', 'g'), Some(24));
        // "9a" → file_idx 0, rank_idx 0 → sq = 0
        assert_eq!(parse_usi_square('9', 'a'), Some(0));
        // "1i" → file_idx 8, rank_idx 8 → sq = 80
        assert_eq!(parse_usi_square('1', 'i'), Some(80));
        // out of range
        assert_eq!(parse_usi_square('0', 'a'), None);
        assert_eq!(parse_usi_square('9', 'z'), None);
    }

    #[test]
    fn test_parse_drop_move() {
        let board = Board::startpos();
        let mv = parse_usi_move("P*5e", &board);
        assert!(mv.is_some());
        let mv = mv.unwrap();
        assert!(mv.is_drop());
        assert_eq!(mv.drop_piece(), PieceType::Pawn);
        // "5e" → file 5 (display) = file_idx 4, rank 'e' = rank_idx 4 → sq = 4*9+4 = 40
        assert_eq!(mv.to_sq(), 40);
    }

    #[test]
    fn test_parse_normal_move() {
        let board = Board::startpos();
        // "7g7f" — Black pawn on 7g moves to 7f
        // 7g: file_idx = 9-7=2, rank_idx='g'-'a'=6 → sq=24
        // 7f: file_idx=2, rank_idx=5 → sq=23
        let mv = parse_usi_move("7g7f", &board);
        assert!(mv.is_some());
        let mv = mv.unwrap();
        assert!(!mv.is_drop());
        assert!(!mv.is_promote());
        assert_eq!(mv.from_sq(), 24);
        assert_eq!(mv.to_sq(), 23);
        assert_eq!(mv.piece_type(), PieceType::Pawn);
    }

    #[test]
    fn test_parse_promotion_move() {
        // We test the '+' flag via to_usi_string round-trip on a known move.
        let mv = Move::new_normal(0, 1, PieceType::Bishop, true);
        assert!(mv.is_promote());
        let s = mv.to_usi_string();
        assert!(s.ends_with('+'), "Expected '+' suffix, got {s}");
    }

    #[test]
    fn test_setup_startpos() {
        let board = setup_position(&["startpos"]);
        assert_eq!(board.to_sfen(), Board::startpos().to_sfen());
    }

    #[test]
    fn test_setup_startpos_with_moves() {
        // Apply one move via position command and verify ply advanced
        let board = setup_position(&["startpos", "moves", "7g7f"]);
        assert_eq!(board.ply, 2, "ply should advance after one move");
    }

    #[test]
    fn test_setup_sfen() {
        let sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";
        let tokens: Vec<&str> = ["sfen"]
            .iter()
            .chain(sfen.split_whitespace().collect::<Vec<_>>().iter())
            .cloned()
            .collect();
        let board = setup_position(&tokens);
        assert_eq!(board.to_sfen(), sfen);
    }

    #[test]
    fn test_go_params_movetime() {
        let params = parse_go(&["movetime", "1000"]);
        assert_eq!(params.movetime_ms, Some(1000));
        assert!(!params.infinite);
    }

    #[test]
    fn test_go_params_byoyomi() {
        let params = parse_go(&["btime", "60000", "wtime", "60000", "byoyomi", "5000"]);
        assert_eq!(params.btime_ms, Some(60_000));
        assert_eq!(params.wtime_ms, Some(60_000));
        assert_eq!(params.byoyomi_ms, Some(5_000));
    }

    #[test]
    fn test_time_budget_movetime() {
        let params = parse_go(&["movetime", "1000"]);
        let budget = time_budget_ms(&params, Color::Black);
        assert_eq!(budget, 990); // 1000 - 10ms safety margin
    }

    #[test]
    fn test_time_budget_btime() {
        let params = parse_go(&["btime", "100000", "wtime", "100000"]);
        let budget = time_budget_ms(&params, Color::Black);
        assert_eq!(budget, 2000); // 100000 / 50
    }

    #[test]
    fn test_select_move_startpos() {
        let mut board = Board::startpos();
        let mv = select_move(&mut board, 1000);
        assert!(mv.is_some(), "Should return a move from startpos");
    }

    #[test]
    fn test_usi_move_roundtrip() {
        // Make sure to_usi_string output can be parsed back
        let mut board = Board::startpos();
        let mut moves = Vec::new();
        generate_legal_moves(&mut board, &mut moves);
        for mv in &moves {
            let s = mv.to_usi_string();
            let reparsed = parse_usi_move(&s, &board);
            assert!(
                reparsed.is_some(),
                "Failed to re-parse USI string: {s}"
            );
        }
    }
}
