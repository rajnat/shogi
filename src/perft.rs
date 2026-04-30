/// Perft (performance test) for move generation validation
use crate::board::Board;
use crate::movegen::generate_legal_moves;
use crate::moves::{make_move_full, unmake_move_full};

pub fn perft(board: &mut Board, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }

    let mut moves = Vec::with_capacity(128);
    generate_legal_moves(board, &mut moves);

    if depth == 1 {
        return moves.len() as u64;
    }

    let mut count = 0u64;
    for mv in moves {
        let undo = make_move_full(board, mv);
        count += perft(board, depth - 1);
        unmake_move_full(board, mv, &undo);
    }
    count
}

pub fn perft_divide(board: &mut Board, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }

    let mut moves = Vec::with_capacity(128);
    generate_legal_moves(board, &mut moves);

    let mut total = 0u64;
    for mv in &moves {
        let undo = make_move_full(board, *mv);
        let count = perft(board, depth - 1);
        unmake_move_full(board, *mv, &undo);

        println!("{}: {}", mv.to_usi_string(), count);
        total += count;
    }
    println!("Total: {}", total);
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;

    #[test]
    fn test_perft_depth1() {
        let mut board = Board::startpos();
        let count = perft(&mut board, 1);
        assert_eq!(count, 30, "Perft(1) should be 30");
    }

    #[test]
    fn test_perft_depth2() {
        let mut board = Board::startpos();
        let count = perft(&mut board, 2);
        assert_eq!(count, 900, "Perft(2) should be 900");
    }

    #[test]
    #[ignore] // Slow test, run with --ignored
    fn test_perft_depth3() {
        let mut board = Board::startpos();
        let count = perft(&mut board, 3);
        assert_eq!(count, 25470, "Perft(3) should be 25470");
    }

    #[test]
    #[ignore]
    fn test_perft_depth4() {
        let mut board = Board::startpos();
        let count = perft(&mut board, 4);
        assert_eq!(count, 719731, "Perft(4) should be 719731");
    }

    #[test]
    #[ignore]
    fn test_perft_depth5() {
        let mut board = Board::startpos();
        let count = perft(&mut board, 5);
        assert_eq!(count, 19861490, "Perft(5) should be 19861490");
    }
}
