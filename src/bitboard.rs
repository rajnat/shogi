/// Bitboard for Shogi: 81 squares fit in a u128
/// Square index: sq = file_idx * 9 + rank_idx
/// bit position in u128 = sq
use crate::types::Square;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub struct Bitboard(pub u128);

impl Bitboard {
    pub const EMPTY: Bitboard = Bitboard(0);
    pub const FULL: Bitboard = Bitboard((1u128 << 81) - 1);

    #[inline]
    pub fn from_sq(sq: Square) -> Bitboard {
        Bitboard(1u128 << sq)
    }

    #[inline]
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    #[inline]
    pub fn is_not_empty(self) -> bool {
        self.0 != 0
    }

    #[inline]
    pub fn contains(self, sq: Square) -> bool {
        (self.0 >> sq) & 1 != 0
    }

    #[inline]
    pub fn set(&mut self, sq: Square) {
        self.0 |= 1u128 << sq;
    }

    #[inline]
    pub fn clear(&mut self, sq: Square) {
        self.0 &= !(1u128 << sq);
    }

    #[inline]
    pub fn pop_lsb(&mut self) -> Square {
        let sq = self.0.trailing_zeros() as Square;
        self.0 &= self.0 - 1;
        sq
    }

    #[inline]
    pub fn lsb(self) -> Square {
        self.0.trailing_zeros() as Square
    }

    #[inline]
    pub fn count(self) -> u32 {
        self.0.count_ones()
    }

    /// Iterator over set squares
    #[inline]
    pub fn iter_squares(mut self) -> impl Iterator<Item = Square> {
        std::iter::from_fn(move || {
            if self.0 == 0 {
                None
            } else {
                let sq = self.0.trailing_zeros() as Square;
                self.0 &= self.0 - 1;
                Some(sq)
            }
        })
    }
}

impl std::ops::BitOr for Bitboard {
    type Output = Bitboard;
    #[inline]
    fn bitor(self, rhs: Bitboard) -> Bitboard {
        Bitboard(self.0 | rhs.0)
    }
}

impl std::ops::BitOrAssign for Bitboard {
    #[inline]
    fn bitor_assign(&mut self, rhs: Bitboard) {
        self.0 |= rhs.0;
    }
}

impl std::ops::BitAnd for Bitboard {
    type Output = Bitboard;
    #[inline]
    fn bitand(self, rhs: Bitboard) -> Bitboard {
        Bitboard(self.0 & rhs.0)
    }
}

impl std::ops::BitAndAssign for Bitboard {
    #[inline]
    fn bitand_assign(&mut self, rhs: Bitboard) {
        self.0 &= rhs.0;
    }
}

impl std::ops::BitXor for Bitboard {
    type Output = Bitboard;
    #[inline]
    fn bitxor(self, rhs: Bitboard) -> Bitboard {
        Bitboard(self.0 ^ rhs.0)
    }
}

impl std::ops::BitXorAssign for Bitboard {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Bitboard) {
        self.0 ^= rhs.0;
    }
}

impl std::ops::Not for Bitboard {
    type Output = Bitboard;
    #[inline]
    fn not(self) -> Bitboard {
        Bitboard(!self.0 & ((1u128 << 81) - 1))
    }
}

impl std::ops::Sub for Bitboard {
    type Output = Bitboard;
    #[inline]
    fn sub(self, rhs: Bitboard) -> Bitboard {
        Bitboard(self.0 & !rhs.0)
    }
}

/// File masks: all squares with given file_idx
pub fn file_mask(file_idx: usize) -> Bitboard {
    let mut bb = Bitboard::EMPTY;
    for rank in 0..9 {
        bb.set((file_idx * 9 + rank) as Square);
    }
    bb
}

/// Rank masks: all squares with given rank_idx
pub fn rank_mask(rank_idx: usize) -> Bitboard {
    let mut bb = Bitboard::EMPTY;
    for file in 0..9 {
        bb.set((file * 9 + rank_idx) as Square);
    }
    bb
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_ops() {
        let mut bb = Bitboard::EMPTY;
        bb.set(0);
        bb.set(10);
        assert!(bb.contains(0));
        assert!(bb.contains(10));
        assert!(!bb.contains(5));
        assert_eq!(bb.count(), 2);

        bb.clear(0);
        assert!(!bb.contains(0));
        assert_eq!(bb.count(), 1);
    }

    #[test]
    fn test_pop_lsb() {
        let mut bb = Bitboard::EMPTY;
        bb.set(3);
        bb.set(7);
        bb.set(15);
        let first = bb.pop_lsb();
        assert_eq!(first, 3);
        let second = bb.pop_lsb();
        assert_eq!(second, 7);
    }

    #[test]
    fn test_iter_squares() {
        let mut bb = Bitboard::EMPTY;
        bb.set(5);
        bb.set(20);
        bb.set(80);
        let sqs: Vec<Square> = bb.iter_squares().collect();
        assert_eq!(sqs, vec![5, 20, 80]);
    }

    #[test]
    fn test_not() {
        let bb = Bitboard::FULL;
        let neg = !bb;
        assert_eq!(neg, Bitboard::EMPTY);
    }

    #[test]
    fn test_file_rank_mask() {
        let f = file_mask(0);
        assert_eq!(f.count(), 9);
        let r = rank_mask(0);
        assert_eq!(r.count(), 9);
    }
}
