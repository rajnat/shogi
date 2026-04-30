/// Zobrist hashing for Shogi positions
/// Keys are initialized once with a seeded RNG
use std::sync::OnceLock;

const SEED: u64 = 0x123456789ABCDEF0;

/// Simple xorshift64 RNG for reproducible key generation
struct Xorshift64(u64);

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Xorshift64(if seed == 0 { 1 } else { seed })
    }

    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

/// Maximum hand counts per piece type (index = piece type 0..7)
pub const MAX_HAND: [usize; 7] = [18, 4, 4, 4, 4, 2, 2];
// Pawn=18, Lance=4, Knight=4, Silver=4, Gold=4, Bishop=2, Rook=2

pub struct ZobristKeys {
    /// [sq][piece_type][color] — 81 * 14 * 2 = 2268
    pub piece_keys: Box<[[[u64; 2]; 14]; 81]>,
    /// [color][piece_type][count] — 2 * 7 * 19 (max 18+1 slots for pawns)
    pub hand_keys: Box<[[[u64; 19]; 7]; 2]>,
    /// XOR in when it's Black's turn
    pub side_key: u64,
}

static ZOBRIST: OnceLock<ZobristKeys> = OnceLock::new();

pub fn get_zobrist() -> &'static ZobristKeys {
    ZOBRIST.get_or_init(init_zobrist)
}

fn init_zobrist() -> ZobristKeys {
    let mut rng = Xorshift64::new(SEED);

    let mut piece_keys = Box::new([[[0u64; 2]; 14]; 81]);
    for sq in 0..81 {
        for pt in 0..14 {
            for c in 0..2 {
                piece_keys[sq][pt][c] = rng.next();
            }
        }
    }

    let mut hand_keys = Box::new([[[0u64; 19]; 7]; 2]);
    for c in 0..2 {
        for pt in 0..7 {
            for count in 0..=MAX_HAND[pt] {
                hand_keys[c][pt][count] = rng.next();
            }
        }
    }

    let side_key = rng.next();

    ZobristKeys {
        piece_keys,
        hand_keys,
        side_key,
    }
}

/// Compute hash for a piece placement
#[inline]
pub fn piece_hash(sq: usize, piece_type: usize, color: usize) -> u64 {
    get_zobrist().piece_keys[sq][piece_type][color]
}

/// Compute hash for a hand count
#[inline]
pub fn hand_hash(color: usize, piece_type: usize, count: usize) -> u64 {
    get_zobrist().hand_keys[color][piece_type][count]
}

/// Side to move hash
#[inline]
pub fn side_hash() -> u64 {
    get_zobrist().side_key
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keys_nonzero() {
        let z = get_zobrist();
        assert_ne!(z.piece_keys[0][0][0], 0);
        assert_ne!(z.side_key, 0);
    }

    #[test]
    fn test_keys_unique() {
        let z = get_zobrist();
        // Check a few keys are unique
        assert_ne!(z.piece_keys[0][0][0], z.piece_keys[0][0][1]);
        assert_ne!(z.piece_keys[0][0][0], z.piece_keys[1][0][0]);
    }
}
