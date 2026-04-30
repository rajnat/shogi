/// Precomputed attack tables for all piece types
/// Uses OnceLock for lazy initialization
use crate::bitboard::Bitboard;
use crate::types::{Color, PieceType, Square, add_step};
use std::sync::OnceLock;

/// Attack tables: step attacks for non-sliding pieces
/// [color][piece_type][square] -> Bitboard of attack squares
struct AttackTables {
    /// Step attacks for Pawn, Knight, Silver, Gold, King (per color)
    /// Index: [color][piece_type_idx][square]
    step_attacks: Box<[[[Bitboard; 81]; 8]; 2]>,
    /// Lance attacks (ray) computed on the fly using occupancy
    /// But we store the full ray for empty board per color
    lance_rays: Box<[[Bitboard; 81]; 2]>,
    /// Rook rays (4 directions) per square
    rook_rays: Box<[[Bitboard; 4]; 81]>,
    /// Bishop rays (4 diagonals) per square
    bishop_rays: Box<[[Bitboard; 4]; 81]>,
}

static TABLES: OnceLock<AttackTables> = OnceLock::new();

fn get_tables() -> &'static AttackTables {
    TABLES.get_or_init(init_tables)
}

fn init_tables() -> AttackTables {
    let mut step_attacks = Box::new([[[Bitboard::EMPTY; 81]; 8]; 2]);
    let mut lance_rays = Box::new([[Bitboard::EMPTY; 81]; 2]);
    let mut rook_rays = Box::new([[Bitboard::EMPTY; 4]; 81]);
    let mut bishop_rays = Box::new([[Bitboard::EMPTY; 4]; 81]);

    for sq in 0u8..81 {
        // Black step attacks (forward = rank_idx - 1)
        // Pawn (idx 0): forward 1
        if let Some(to) = add_step(sq, 0, -1) {
            step_attacks[0][0][sq as usize].set(to);
        }
        // Lance: handled via ray
        // Knight (idx 2): (±1, -2)
        for df in [-1i8, 1i8] {
            if let Some(to) = add_step(sq, df, -2) {
                step_attacks[0][2][sq as usize].set(to);
            }
        }
        // Silver (idx 3): (0,-1), (±1,-1), (±1,+1)
        for (df, dr) in [(0i8, -1i8), (1, -1), (-1, -1), (1, 1), (-1, 1)] {
            if let Some(to) = add_step(sq, df, dr) {
                step_attacks[0][3][sq as usize].set(to);
            }
        }
        // Gold (idx 4): (0,-1), (±1,-1), (0,+1), (±1,0)
        for (df, dr) in [(0i8, -1i8), (1, -1), (-1, -1), (0, 1), (1, 0), (-1, 0)] {
            if let Some(to) = add_step(sq, df, dr) {
                step_attacks[0][4][sq as usize].set(to);
            }
        }
        // King (idx 7): all 8 adjacent
        for df in [-1i8, 0, 1] {
            for dr in [-1i8, 0, 1] {
                if df == 0 && dr == 0 {
                    continue;
                }
                if let Some(to) = add_step(sq, df, dr) {
                    step_attacks[0][7][sq as usize].set(to);
                }
            }
        }

        // White step attacks (forward = rank_idx + 1) — mirror of Black
        // Pawn (idx 0): forward = +1 in rank
        if let Some(to) = add_step(sq, 0, 1) {
            step_attacks[1][0][sq as usize].set(to);
        }
        // Knight (idx 2): (±1, +2)
        for df in [-1i8, 1i8] {
            if let Some(to) = add_step(sq, df, 2) {
                step_attacks[1][2][sq as usize].set(to);
            }
        }
        // Silver (idx 3): (0,+1), (±1,+1), (±1,-1)
        for (df, dr) in [(0i8, 1i8), (1, 1), (-1, 1), (1, -1), (-1, -1)] {
            if let Some(to) = add_step(sq, df, dr) {
                step_attacks[1][3][sq as usize].set(to);
            }
        }
        // Gold (idx 4): (0,+1), (±1,+1), (0,-1), (±1,0)
        for (df, dr) in [(0i8, 1i8), (1, 1), (-1, 1), (0, -1), (1, 0), (-1, 0)] {
            if let Some(to) = add_step(sq, df, dr) {
                step_attacks[1][4][sq as usize].set(to);
            }
        }
        // King is symmetric
        step_attacks[1][7][sq as usize] = step_attacks[0][7][sq as usize];

        // Lance rays
        // Black lance: ray decreasing rank_idx (north)
        {
            let mut bb = Bitboard::EMPTY;
            let mut cur = sq;
            loop {
                match add_step(cur, 0, -1) {
                    Some(next) => {
                        bb.set(next);
                        cur = next;
                    }
                    None => break,
                }
            }
            lance_rays[0][sq as usize] = bb;
        }
        // White lance: ray increasing rank_idx (south)
        {
            let mut bb = Bitboard::EMPTY;
            let mut cur = sq;
            loop {
                match add_step(cur, 0, 1) {
                    Some(next) => {
                        bb.set(next);
                        cur = next;
                    }
                    None => break,
                }
            }
            lance_rays[1][sq as usize] = bb;
        }

        // Rook rays: 4 directions: (+1,0), (-1,0), (0,+1), (0,-1)
        let rook_dirs: [(i8, i8); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
        for (d, (df, dr)) in rook_dirs.iter().enumerate() {
            let mut bb = Bitboard::EMPTY;
            let mut cur = sq;
            loop {
                match add_step(cur, *df, *dr) {
                    Some(next) => {
                        bb.set(next);
                        cur = next;
                    }
                    None => break,
                }
            }
            rook_rays[sq as usize][d] = bb;
        }

        // Bishop rays: 4 diagonals: (+1,+1), (+1,-1), (-1,+1), (-1,-1)
        let bishop_dirs: [(i8, i8); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
        for (d, (df, dr)) in bishop_dirs.iter().enumerate() {
            let mut bb = Bitboard::EMPTY;
            let mut cur = sq;
            loop {
                match add_step(cur, *df, *dr) {
                    Some(next) => {
                        bb.set(next);
                        cur = next;
                    }
                    None => break,
                }
            }
            bishop_rays[sq as usize][d] = bb;
        }
    }

    // Copy promoted-piece step attacks from Gold for ProPawn/ProLance/ProKnight/ProSilver
    // These are stored at indices 8–11 but we don't use separate storage — we use gold's idx.

    AttackTables {
        step_attacks,
        lance_rays,
        rook_rays,
        bishop_rays,
    }
}

/// Get step attacks for a piece type and color
/// For promoted pieces that move like gold, returns the gold attacks
#[inline]
pub fn step_attacks(piece: PieceType, color: Color, sq: Square) -> Bitboard {
    let tables = get_tables();
    let c = color.index();
    let pt_idx = match piece {
        PieceType::Pawn => 0,
        PieceType::Knight => 2,
        PieceType::Silver => 3,
        PieceType::Gold
        | PieceType::ProPawn
        | PieceType::ProLance
        | PieceType::ProKnight
        | PieceType::ProSilver => 4,
        PieceType::King => 7,
        _ => return Bitboard::EMPTY,
    };
    tables.step_attacks[c][pt_idx][sq as usize]
}

/// Get full lance ray for empty board
#[inline]
pub fn lance_ray(color: Color, sq: Square) -> Bitboard {
    get_tables().lance_rays[color.index()][sq as usize]
}

/// Lance attacks given occupancy
#[inline]
pub fn lance_attacks(color: Color, sq: Square, occ: Bitboard) -> Bitboard {
    let tables = get_tables();
    let ray = tables.lance_rays[color.index()][sq as usize];
    // Lance ray is in one direction (monotone)
    let c = color.index();
    ray_attacks_single_dir(ray, sq, occ, c == 0)
}

/// Ray attacks in a single monotone direction
/// is_positive: if bit index increases along the ray (decreasing rank for black = negative)
fn ray_attacks_single_dir(ray: Bitboard, sq: Square, occ: Bitboard, is_negative: bool) -> Bitboard {
    let blockers = ray & occ;
    if blockers.is_empty() {
        return ray;
    }
    if is_negative {
        // Negative direction: rank decreases = bit decreases (since sq = file*9 + rank)
        // Actually rank decreases means sq number decreases
        // Find the most significant blocker (largest bit < sq)
        let blocker_sq = 127 - blockers.0.leading_zeros() as u8; // MSB
        // Include from sq-1 down to blocker_sq
        // Mask: bits from sq-1 down to blocker_sq
        if blocker_sq < sq {
            let mask_above = (1u128 << sq) - 1; // bits below sq
            let mask_from_blocker = !((1u128 << blocker_sq) - 1); // bits >= blocker_sq
            Bitboard(ray.0 & mask_above & mask_from_blocker)
        } else {
            ray
        }
    } else {
        // Positive direction: bits increase going along ray
        // Find least significant blocker > sq
        // blockers all have bit > sq (since the ray goes in + direction)
        let blocker_sq = blockers.0.trailing_zeros() as u8; // LSB
        // Include from sq+1 up to blocker_sq
        let mask_up_to_blocker = (1u128 << (blocker_sq + 1)) - 1; // bits <= blocker_sq
        let mask_above_sq = !((1u128 << (sq + 1)) - 1); // bits > sq
        Bitboard(ray.0 & mask_up_to_blocker & mask_above_sq)
    }
}

/// Rook attacks given occupancy
#[inline]
pub fn rook_attacks(sq: Square, occ: Bitboard) -> Bitboard {
    let tables = get_tables();
    let rays = &tables.rook_rays[sq as usize];
    // directions: 0=(+1,0) file+, 1=(-1,0) file-, 2=(0,+1) rank+, 3=(0,-1) rank-
    ray_attacks_single_dir(rays[0], sq, occ, false)
        | ray_attacks_single_dir(rays[1], sq, occ, true)
        | ray_attacks_single_dir(rays[2], sq, occ, false)
        | ray_attacks_single_dir(rays[3], sq, occ, true)
}

/// Bishop attacks given occupancy
#[inline]
pub fn bishop_attacks(sq: Square, occ: Bitboard) -> Bitboard {
    let tables = get_tables();
    let rays = &tables.bishop_rays[sq as usize];
    // directions: 0=(+1,+1), 1=(+1,-1), 2=(-1,+1), 3=(-1,-1)
    // (df, dr) determines which direction the bits go in our linear indexing
    // sq = file*9 + rank; step (+1,+1) means sq+10, (+1,-1) means sq+8, (-1,+1) means sq-8, (-1,-1) means sq-10
    // Since file increases with larger sq groups and rank is within:
    // (+1,+1) -> sq increases (positive)
    // (+1,-1) -> sq = old + 9 - 1 = +8 (positive if 8 > 0)
    // (-1,+1) -> sq = old - 9 + 1 = -8 (negative)
    // (-1,-1) -> sq = old - 9 - 1 = -10 (negative)
    ray_attacks_single_dir(rays[0], sq, occ, false) // (+1,+1): positive direction
        | ray_attacks_single_dir(rays[1], sq, occ, false) // (+1,-1): +8 positive
        | ray_attacks_single_dir(rays[2], sq, occ, true)  // (-1,+1): -8 negative
        | ray_attacks_single_dir(rays[3], sq, occ, true)  // (-1,-1): -10 negative
}

/// ProBishop: bishop moves + 1-step orthogonal
#[inline]
pub fn pro_bishop_attacks(sq: Square, occ: Bitboard) -> Bitboard {
    bishop_attacks(sq, occ) | orthogonal_one_step(sq)
}

/// ProRook: rook moves + 1-step diagonal
#[inline]
pub fn pro_rook_attacks(sq: Square, occ: Bitboard) -> Bitboard {
    rook_attacks(sq, occ) | diagonal_one_step(sq)
}

/// All orthogonal 1-step squares (for ProBishop)
#[inline]
pub fn orthogonal_one_step(sq: Square) -> Bitboard {
    let mut bb = Bitboard::EMPTY;
    for (df, dr) in [(1i8, 0i8), (-1, 0), (0, 1), (0, -1)] {
        if let Some(to) = add_step(sq, df, dr) {
            bb.set(to);
        }
    }
    bb
}

/// All diagonal 1-step squares (for ProRook)
#[inline]
pub fn diagonal_one_step(sq: Square) -> Bitboard {
    let mut bb = Bitboard::EMPTY;
    for (df, dr) in [(1i8, 1i8), (1, -1), (-1, 1), (-1, -1)] {
        if let Some(to) = add_step(sq, df, dr) {
            bb.set(to);
        }
    }
    bb
}

/// All squares attacked by a piece (for is_in_check type checks)
/// Returns attack bitboard for a piece at sq with given occupancy
pub fn piece_attacks(piece: PieceType, color: Color, sq: Square, occ: Bitboard) -> Bitboard {
    match piece {
        PieceType::Pawn | PieceType::Knight | PieceType::Silver | PieceType::King => {
            step_attacks(piece, color, sq)
        }
        PieceType::Gold
        | PieceType::ProPawn
        | PieceType::ProLance
        | PieceType::ProKnight
        | PieceType::ProSilver => step_attacks(PieceType::Gold, color, sq),
        PieceType::Lance => lance_attacks(color, sq, occ),
        PieceType::Bishop => bishop_attacks(sq, occ),
        PieceType::Rook => rook_attacks(sq, occ),
        PieceType::ProBishop => pro_bishop_attacks(sq, occ),
        PieceType::ProRook => pro_rook_attacks(sq, occ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Color, square};

    #[test]
    fn test_king_attacks() {
        let sq = square(4, 4); // center
        let attacks = step_attacks(PieceType::King, Color::Black, sq);
        assert_eq!(attacks.count(), 8);
    }

    #[test]
    fn test_pawn_attacks() {
        // Black pawn at center: attacks one square north (rank-1)
        let sq = square(4, 4);
        let attacks = step_attacks(PieceType::Pawn, Color::Black, sq);
        assert_eq!(attacks.count(), 1);
        assert!(attacks.contains(square(4, 3)));

        // White pawn at center: attacks one square south (rank+1)
        let attacks_w = step_attacks(PieceType::Pawn, Color::White, sq);
        assert_eq!(attacks_w.count(), 1);
        assert!(attacks_w.contains(square(4, 5)));
    }

    #[test]
    fn test_knight_attacks() {
        let sq = square(4, 4);
        let attacks = step_attacks(PieceType::Knight, Color::Black, sq);
        assert_eq!(attacks.count(), 2);
        assert!(attacks.contains(square(3, 2)));
        assert!(attacks.contains(square(5, 2)));
    }

    #[test]
    fn test_rook_attacks_empty() {
        let sq = square(4, 4);
        let attacks = rook_attacks(sq, Bitboard::EMPTY);
        // sq = file_idx=4, rank_idx=4
        // Along rank (same file, different rank): 8 squares
        // Along file (same rank, different file): 8 squares
        // Total = 16
        assert_eq!(attacks.count(), 16);
    }

    #[test]
    fn test_rook_attacks_with_blocker() {
        let sq = square(4, 4);
        let mut occ = Bitboard::EMPTY;
        // Block at file4, rank6 (2 squares south)
        occ.set(square(4, 6));
        let attacks = rook_attacks(sq, occ);
        // South direction: rank5, rank6 (stops at blocker, includes it)
        // Without blocker south: rank5,6,7,8 = 4 squares
        // With blocker at rank6: rank5, rank6 = 2 squares
        // Verify south is limited
        assert!(attacks.contains(square(4, 5)));
        assert!(attacks.contains(square(4, 6))); // blocker included (can capture)
        assert!(!attacks.contains(square(4, 7))); // blocked
    }

    #[test]
    fn test_lance_attacks() {
        let sq = square(4, 4);
        // Black lance goes north (rank decreasing)
        let attacks = lance_attacks(Color::Black, sq, Bitboard::EMPTY);
        assert_eq!(attacks.count(), 4); // rank 3,2,1,0

        let mut occ = Bitboard::EMPTY;
        occ.set(square(4, 2));
        let attacks_blocked = lance_attacks(Color::Black, sq, occ);
        assert_eq!(attacks_blocked.count(), 2); // rank 3,2 (stop at rank2 blocker)
        assert!(attacks_blocked.contains(square(4, 3)));
        assert!(attacks_blocked.contains(square(4, 2)));
        assert!(!attacks_blocked.contains(square(4, 1)));
    }

    #[test]
    fn test_bishop_attacks_empty() {
        let sq = square(4, 4); // file4, rank4
        let attacks = bishop_attacks(sq, Bitboard::EMPTY);
        // Diagonals from center: each diagonal goes to edge
        // (+1,+1): (5,5),(6,6),(7,7),(8,8) = 4 squares
        // (+1,-1): (5,3),(6,2),(7,1),(8,0) = 4 squares
        // (-1,+1): (3,5),(2,6),(1,7),(0,8) = 4 squares
        // (-1,-1): (3,3),(2,2),(1,1),(0,0) = 4 squares
        assert_eq!(attacks.count(), 16);
    }
}
