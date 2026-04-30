/// Board representation for Shogi
use crate::bitboard::Bitboard;
use crate::types::{Color, PieceType, Square, square};
use crate::zobrist::{hand_hash, piece_hash, side_hash};

#[derive(Clone, Debug)]
pub struct Board {
    /// Bitboards for each color and piece type [color][piece_type]
    pub pieces: [[Bitboard; 14]; 2],
    /// Hand piece counts [color][piece_type 0..7]
    pub hand: [[u8; 7]; 2],
    pub side_to_move: Color,
    pub ply: u16,
    pub hash: u64,
    /// Combined occupancy per color (cached)
    pub color_bb: [Bitboard; 2],
    /// All pieces occupancy
    pub occ: Bitboard,
}

impl Board {
    pub fn empty() -> Board {
        Board {
            pieces: [[Bitboard::EMPTY; 14]; 2],
            hand: [[0u8; 7]; 2],
            side_to_move: Color::Black,
            ply: 1,
            hash: 0,
            color_bb: [Bitboard::EMPTY; 2],
            occ: Bitboard::EMPTY,
        }
    }

    /// Add a piece to the board (updates bitboards and hash)
    pub fn set_piece(&mut self, sq: Square, color: Color, pt: PieceType) {
        let c = color.index();
        let p = pt.index();
        self.pieces[c][p].set(sq);
        self.color_bb[c].set(sq);
        self.occ.set(sq);
        self.hash ^= piece_hash(sq as usize, p, c);
    }

    /// Remove a piece from the board
    pub fn remove_piece(&mut self, sq: Square, color: Color, pt: PieceType) {
        let c = color.index();
        let p = pt.index();
        self.pieces[c][p].clear(sq);
        self.color_bb[c].clear(sq);
        self.occ.clear(sq);
        self.hash ^= piece_hash(sq as usize, p, c);
    }

    /// Get piece at square, returns (color, piece_type) if occupied
    pub fn piece_at(&self, sq: Square) -> Option<(Color, PieceType)> {
        if !self.occ.contains(sq) {
            return None;
        }
        for c in 0..2 {
            for p in 0..14 {
                if self.pieces[c][p].contains(sq) {
                    let color = if c == 0 { Color::Black } else { Color::White };
                    return Some((color, PieceType::from_index(p).unwrap()));
                }
            }
        }
        None
    }

    /// Get piece type at square for a given color
    pub fn piece_type_at(&self, sq: Square, color: Color) -> Option<PieceType> {
        let c = color.index();
        if !self.color_bb[c].contains(sq) {
            return None;
        }
        for p in 0..14 {
            if self.pieces[c][p].contains(sq) {
                return PieceType::from_index(p);
            }
        }
        None
    }

    /// Find king square for given color
    pub fn king_sq(&self, color: Color) -> Square {
        self.pieces[color.index()][PieceType::King.index()].lsb()
    }

    /// Recompute hash from scratch
    pub fn recompute_hash(&mut self) {
        self.hash = 0;
        for c in 0..2 {
            for p in 0..14 {
                let mut bb = self.pieces[c][p];
                while bb.is_not_empty() {
                    let sq = bb.pop_lsb();
                    self.hash ^= piece_hash(sq as usize, p, c);
                }
            }
            for pt in 0..7 {
                let count = self.hand[c][pt] as usize;
                if count > 0 {
                    self.hash ^= hand_hash(c, pt, count);
                }
            }
        }
        if self.side_to_move == Color::Black {
            self.hash ^= side_hash();
        }
    }

    /// Parse SFEN string into a Board
    pub fn from_sfen(sfen: &str) -> Result<Board, String> {
        let parts: Vec<&str> = sfen.split_whitespace().collect();
        if parts.len() < 4 {
            return Err(format!("Invalid SFEN: too few parts in '{}'", sfen));
        }

        let mut board = Board::empty();

        // Parse board part
        let board_str = parts[0];
        let ranks: Vec<&str> = board_str.split('/').collect();
        if ranks.len() != 9 {
            return Err(format!("Invalid SFEN: expected 9 ranks, got {}", ranks.len()));
        }

        for (rank_idx, rank_str) in ranks.iter().enumerate() {
            let mut file_idx = 0usize;
            let chars: Vec<char> = rank_str.chars().collect();
            let mut i = 0;
            while i < chars.len() {
                let ch = chars[i];
                if ch.is_ascii_digit() {
                    file_idx += (ch as usize) - ('0' as usize);
                    i += 1;
                } else if ch == '+' {
                    // Promoted piece
                    i += 1;
                    if i >= chars.len() {
                        return Err("Invalid SFEN: '+' at end of rank".to_string());
                    }
                    let pc = chars[i];
                    i += 1;
                    let (color, pt) = parse_sfen_piece_promoted(pc)?;
                    let sq = square(file_idx as u8, rank_idx as u8);
                    board.set_piece(sq, color, pt);
                    file_idx += 1;
                } else {
                    let (color, pt) = parse_sfen_piece(ch)?;
                    let sq = square(file_idx as u8, rank_idx as u8);
                    board.set_piece(sq, color, pt);
                    file_idx += 1;
                    i += 1;
                }
            }
            if file_idx != 9 {
                return Err(format!("Invalid SFEN: rank {} has {} files", rank_idx, file_idx));
            }
        }

        // Parse side to move
        board.side_to_move = match parts[1] {
            "b" => Color::Black,
            "w" => Color::White,
            other => return Err(format!("Invalid side to move: '{}'", other)),
        };

        // Parse hand
        let hand_str = parts[2];
        if hand_str != "-" {
            let chars: Vec<char> = hand_str.chars().collect();
            let mut i = 0;
            while i < chars.len() {
                // May have a count prefix
                let mut count = 0u8;
                while i < chars.len() && chars[i].is_ascii_digit() {
                    count = count * 10 + (chars[i] as u8 - b'0');
                    i += 1;
                }
                if i >= chars.len() {
                    break;
                }
                let ch = chars[i];
                i += 1;
                if count == 0 {
                    count = 1;
                }
                let (color, pt) = parse_sfen_piece(ch)?;
                if pt.index() >= 7 {
                    return Err(format!("Invalid hand piece: '{}'", ch));
                }
                let c = color.index();
                let old_count = board.hand[c][pt.index()];
                board.hand[c][pt.index()] += count;
                let new_count = board.hand[c][pt.index()];
                // Update hash
                if old_count > 0 {
                    board.hash ^= crate::zobrist::hand_hash(c, pt.index(), old_count as usize);
                }
                board.hash ^= crate::zobrist::hand_hash(c, pt.index(), new_count as usize);
            }
        }

        // Parse ply
        board.ply = parts[3].parse::<u16>().unwrap_or(1);

        // Apply side to move to hash
        if board.side_to_move == Color::Black {
            board.hash ^= side_hash();
        }

        Ok(board)
    }

    /// Convert board to SFEN string
    pub fn to_sfen(&self) -> String {
        let mut result = String::new();

        // Board
        for rank_idx in 0..9 {
            let mut empty_count = 0u32;
            for file_idx in 0..9 {
                let sq = square(file_idx as u8, rank_idx as u8);
                match self.piece_at(sq) {
                    None => {
                        empty_count += 1;
                    }
                    Some((color, pt)) => {
                        if empty_count > 0 {
                            result.push_str(&empty_count.to_string());
                            empty_count = 0;
                        }
                        if pt.is_promoted() {
                            result.push('+');
                        }
                        let ch = pt.to_char();
                        if color == Color::Black {
                            result.push(ch);
                        } else {
                            result.push(ch.to_lowercase().next().unwrap());
                        }
                    }
                }
            }
            if empty_count > 0 {
                result.push_str(&empty_count.to_string());
            }
            if rank_idx < 8 {
                result.push('/');
            }
        }

        // Side to move
        result.push(' ');
        result.push(match self.side_to_move {
            Color::Black => 'b',
            Color::White => 'w',
        });

        // Hand
        result.push(' ');
        let hand_order = [
            (0usize, 6usize, 'R'), // Black Rook
            (0, 5, 'B'),           // Black Bishop
            (0, 4, 'G'),           // Black Gold
            (0, 3, 'S'),           // Black Silver
            (0, 2, 'N'),           // Black Knight
            (0, 1, 'L'),           // Black Lance
            (0, 0, 'P'),           // Black Pawn
            (1, 6, 'r'),           // White Rook
            (1, 5, 'b'),           // White Bishop
            (1, 4, 'g'),           // White Gold
            (1, 3, 's'),           // White Silver
            (1, 2, 'n'),           // White Knight
            (1, 1, 'l'),           // White Lance
            (1, 0, 'p'),           // White Pawn
        ];
        let mut has_hand = false;
        for (c, pt, ch) in &hand_order {
            let count = self.hand[*c][*pt];
            if count > 0 {
                has_hand = true;
                if count > 1 {
                    result.push_str(&count.to_string());
                }
                result.push(*ch);
            }
        }
        if !has_hand {
            result.push('-');
        }

        // Ply
        result.push(' ');
        result.push_str(&self.ply.to_string());

        result
    }

    pub fn startpos() -> Board {
        Board::from_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1")
            .expect("startpos SFEN is valid")
    }
}

fn parse_sfen_piece(ch: char) -> Result<(Color, PieceType), String> {
    let color = if ch.is_uppercase() {
        Color::Black
    } else {
        Color::White
    };
    let pt = match ch.to_uppercase().next().unwrap() {
        'P' => PieceType::Pawn,
        'L' => PieceType::Lance,
        'N' => PieceType::Knight,
        'S' => PieceType::Silver,
        'G' => PieceType::Gold,
        'B' => PieceType::Bishop,
        'R' => PieceType::Rook,
        'K' => PieceType::King,
        other => return Err(format!("Unknown piece character: '{}'", other)),
    };
    Ok((color, pt))
}

fn parse_sfen_piece_promoted(ch: char) -> Result<(Color, PieceType), String> {
    let color = if ch.is_uppercase() {
        Color::Black
    } else {
        Color::White
    };
    let pt = match ch.to_uppercase().next().unwrap() {
        'P' => PieceType::ProPawn,
        'L' => PieceType::ProLance,
        'N' => PieceType::ProKnight,
        'S' => PieceType::ProSilver,
        'B' => PieceType::ProBishop,
        'R' => PieceType::ProRook,
        other => return Err(format!("Cannot promote piece: '{}'", other)),
    };
    Ok((color, pt))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_startpos_parse() {
        let board = Board::startpos();
        // Black has rook at file2 (file_idx=7), rank 8 (rank_idx=7) -- wait
        // SFEN: LNSGKGSNL b - 1  (rank 9, Black's back rank)
        // rank 9 = rank_idx 8
        // In SFEN rank 9 (last rank) = rank_idx 8
        // Pieces from file 9 to file 1 (file_idx 0 to 8)
        // LNSGKGSNL -> L at file_idx=0, N at 1, S at 2, G at 3, K at 4, G at 5, S at 6, N at 7, L at 8
        // All at rank_idx=8

        // Black king at file_idx=4, rank_idx=8
        let king_sq = board.king_sq(Color::Black);
        assert_eq!(king_sq, square(4, 8), "Black king should be at file_idx=4, rank_idx=8");

        // White king at file_idx=4, rank_idx=0
        let white_king = board.king_sq(Color::White);
        assert_eq!(white_king, square(4, 0), "White king should be at file_idx=4, rank_idx=0");
    }

    #[test]
    fn test_sfen_roundtrip() {
        let sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";
        let board = Board::from_sfen(sfen).expect("Parse failed");
        let output = board.to_sfen();
        assert_eq!(output, sfen, "SFEN roundtrip failed");
    }

    #[test]
    fn test_piece_at() {
        let board = Board::startpos();
        // Black pawn at file_idx=0, rank_idx=6 (rank 7)
        let sq = square(0, 6);
        let piece = board.piece_at(sq);
        assert_eq!(piece, Some((Color::Black, PieceType::Pawn)));
    }

    #[test]
    fn test_hash_deterministic() {
        let b1 = Board::startpos();
        let b2 = Board::startpos();
        assert_eq!(b1.hash, b2.hash);
    }
}
