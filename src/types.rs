/// Color: Black (Sente) or White (Gote)
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Color {
    Black = 0,
    White = 1,
}

impl Color {
    #[inline]
    pub fn opponent(self) -> Color {
        match self {
            Color::Black => Color::White,
            Color::White => Color::Black,
        }
    }

    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }
}

/// Piece types (0–13)
/// 0–6: unpromotable/base, 7: King, 8–13: promoted
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PieceType {
    Pawn = 0,
    Lance = 1,
    Knight = 2,
    Silver = 3,
    Gold = 4,
    Bishop = 5,
    Rook = 6,
    King = 7,
    ProPawn = 8,
    ProLance = 9,
    ProKnight = 10,
    ProSilver = 11,
    ProBishop = 12,
    ProRook = 13,
}

impl PieceType {
    pub const ALL: [PieceType; 14] = [
        PieceType::Pawn,
        PieceType::Lance,
        PieceType::Knight,
        PieceType::Silver,
        PieceType::Gold,
        PieceType::Bishop,
        PieceType::Rook,
        PieceType::King,
        PieceType::ProPawn,
        PieceType::ProLance,
        PieceType::ProKnight,
        PieceType::ProSilver,
        PieceType::ProBishop,
        PieceType::ProRook,
    ];

    /// Piece types that can be held in hand (0–6)
    pub const HAND_TYPES: [PieceType; 7] = [
        PieceType::Pawn,
        PieceType::Lance,
        PieceType::Knight,
        PieceType::Silver,
        PieceType::Gold,
        PieceType::Bishop,
        PieceType::Rook,
    ];

    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }

    #[inline]
    pub fn from_index(i: usize) -> Option<PieceType> {
        match i {
            0 => Some(PieceType::Pawn),
            1 => Some(PieceType::Lance),
            2 => Some(PieceType::Knight),
            3 => Some(PieceType::Silver),
            4 => Some(PieceType::Gold),
            5 => Some(PieceType::Bishop),
            6 => Some(PieceType::Rook),
            7 => Some(PieceType::King),
            8 => Some(PieceType::ProPawn),
            9 => Some(PieceType::ProLance),
            10 => Some(PieceType::ProKnight),
            11 => Some(PieceType::ProSilver),
            12 => Some(PieceType::ProBishop),
            13 => Some(PieceType::ProRook),
            _ => None,
        }
    }

    /// Returns promoted version of piece, if it can promote
    #[inline]
    pub fn promoted(self) -> Option<PieceType> {
        match self {
            PieceType::Pawn => Some(PieceType::ProPawn),
            PieceType::Lance => Some(PieceType::ProLance),
            PieceType::Knight => Some(PieceType::ProKnight),
            PieceType::Silver => Some(PieceType::ProSilver),
            PieceType::Bishop => Some(PieceType::ProBishop),
            PieceType::Rook => Some(PieceType::ProRook),
            _ => None,
        }
    }

    /// Returns unpromoteed (hand) version of a promoted piece
    #[inline]
    pub fn demoted(self) -> PieceType {
        match self {
            PieceType::ProPawn => PieceType::Pawn,
            PieceType::ProLance => PieceType::Lance,
            PieceType::ProKnight => PieceType::Knight,
            PieceType::ProSilver => PieceType::Silver,
            PieceType::ProBishop => PieceType::Bishop,
            PieceType::ProRook => PieceType::Rook,
            other => other,
        }
    }

    /// Whether this piece can promote in-game
    #[inline]
    pub fn can_promote(self) -> bool {
        matches!(
            self,
            PieceType::Pawn
                | PieceType::Lance
                | PieceType::Knight
                | PieceType::Silver
                | PieceType::Bishop
                | PieceType::Rook
        )
    }

    /// Whether this is a promoted piece
    #[inline]
    pub fn is_promoted(self) -> bool {
        self as u8 >= 8
    }

    pub fn to_char(self) -> char {
        match self {
            PieceType::Pawn => 'P',
            PieceType::Lance => 'L',
            PieceType::Knight => 'N',
            PieceType::Silver => 'S',
            PieceType::Gold => 'G',
            PieceType::Bishop => 'B',
            PieceType::Rook => 'R',
            PieceType::King => 'K',
            PieceType::ProPawn => 'P', // with + prefix
            PieceType::ProLance => 'L',
            PieceType::ProKnight => 'N',
            PieceType::ProSilver => 'S',
            PieceType::ProBishop => 'B',
            PieceType::ProRook => 'R',
        }
    }
}

/// A square index: file_idx * 9 + rank_idx
/// file_idx: 0 (file 9) to 8 (file 1)
/// rank_idx: 0 (rank 1) to 8 (rank 9)
pub type Square = u8;

pub const NUM_SQUARES: usize = 81;

#[inline]
pub fn square(file_idx: u8, rank_idx: u8) -> Square {
    file_idx * 9 + rank_idx
}

#[inline]
pub fn file_of(sq: Square) -> u8 {
    sq / 9
}

#[inline]
pub fn rank_of(sq: Square) -> u8 {
    sq % 9
}

/// Returns the new square after taking a step (df, dr), or None if out of bounds
#[inline]
pub fn add_step(sq: Square, df: i8, dr: i8) -> Option<Square> {
    let f = (sq / 9) as i8 + df;
    let r = (sq % 9) as i8 + dr;
    if f < 0 || f > 8 || r < 0 || r > 8 {
        return None;
    }
    Some(f as u8 * 9 + r as u8)
}

/// Encoded move (packed u32)
/// bits 0–6:   from-square (or piece type for drops)
/// bits 7–13:  to-square
/// bits 14–17: piece type (4 bits)
/// bit 18:     promote flag
/// bit 19:     drop flag
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Move(pub u32);

impl Move {
    #[inline]
    pub fn new_normal(from: Square, to: Square, piece: PieceType, promote: bool) -> Move {
        Move(
            (from as u32)
                | ((to as u32) << 7)
                | ((piece as u32) << 14)
                | ((promote as u32) << 18),
        )
    }

    #[inline]
    pub fn new_drop(piece: PieceType, to: Square) -> Move {
        Move(
            (piece as u32)           // from bits encode piece type
                | ((to as u32) << 7)
                | ((piece as u32) << 14)
                | (1u32 << 19),
        )
    }

    #[inline]
    pub fn from_sq(self) -> u8 {
        (self.0 & 0x7F) as u8
    }

    #[inline]
    pub fn to_sq(self) -> Square {
        ((self.0 >> 7) & 0x7F) as u8
    }

    #[inline]
    pub fn piece_type(self) -> PieceType {
        PieceType::from_index(((self.0 >> 14) & 0xF) as usize).unwrap_or(PieceType::Pawn)
    }

    #[inline]
    pub fn is_promote(self) -> bool {
        (self.0 >> 18) & 1 != 0
    }

    #[inline]
    pub fn is_drop(self) -> bool {
        (self.0 >> 19) & 1 != 0
    }

    /// For drops: the piece type being dropped
    #[inline]
    pub fn drop_piece(self) -> PieceType {
        PieceType::from_index((self.0 & 0x7F) as usize).unwrap_or(PieceType::Pawn)
    }

    pub fn to_usi_string(self) -> String {
        if self.is_drop() {
            let piece = self.drop_piece();
            let to = self.to_sq();
            let file = 9 - file_of(to); // file 1–9 in display (file 9 = idx 0)
            let rank = rank_of(to) + 1; // rank 1–9
            format!("{}*{}{}", piece.to_char(), file, (b'a' + rank - 1) as char)
        } else {
            let from = self.from_sq();
            let to = self.to_sq();
            let from_file = 9 - file_of(from);
            let from_rank = rank_of(from) + 1;
            let to_file = 9 - file_of(to);
            let to_rank = rank_of(to) + 1;
            let promo = if self.is_promote() { "+" } else { "" };
            format!(
                "{}{}{}{}{promo}",
                from_file,
                (b'a' + from_rank - 1) as char,
                to_file,
                (b'a' + to_rank - 1) as char
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_square_encoding() {
        // file_idx=0 (file 9), rank_idx=0 (rank 1) -> sq=0
        assert_eq!(square(0, 0), 0);
        // file_idx=8 (file 1), rank_idx=8 (rank 9) -> sq=80
        assert_eq!(square(8, 8), 80);
        assert_eq!(file_of(0), 0);
        assert_eq!(rank_of(0), 0);
        assert_eq!(file_of(80), 8);
        assert_eq!(rank_of(80), 8);
    }

    #[test]
    fn test_add_step() {
        let sq = square(4, 4); // middle of board
        assert_eq!(add_step(sq, 0, -1), Some(square(4, 3)));
        assert_eq!(add_step(sq, 0, 1), Some(square(4, 5)));
        assert_eq!(add_step(sq, 1, 0), Some(square(5, 4)));
        assert_eq!(add_step(sq, -1, 0), Some(square(3, 4)));
        // off board
        assert_eq!(add_step(square(0, 0), -1, 0), None);
        assert_eq!(add_step(square(0, 0), 0, -1), None);
    }

    #[test]
    fn test_move_encoding() {
        let m = Move::new_normal(10, 20, PieceType::Gold, true);
        assert_eq!(m.from_sq(), 10);
        assert_eq!(m.to_sq(), 20);
        assert_eq!(m.piece_type(), PieceType::Gold);
        assert!(m.is_promote());
        assert!(!m.is_drop());

        let d = Move::new_drop(PieceType::Pawn, 30);
        assert!(d.is_drop());
        assert_eq!(d.to_sq(), 30);
        assert_eq!(d.drop_piece(), PieceType::Pawn);
    }

    #[test]
    fn test_color_opponent() {
        assert_eq!(Color::Black.opponent(), Color::White);
        assert_eq!(Color::White.opponent(), Color::Black);
    }

    #[test]
    fn test_piece_promote_demote() {
        assert_eq!(PieceType::Pawn.promoted(), Some(PieceType::ProPawn));
        assert_eq!(PieceType::King.promoted(), None);
        assert_eq!(PieceType::ProPawn.demoted(), PieceType::Pawn);
        assert_eq!(PieceType::Gold.demoted(), PieceType::Gold);
    }
}
