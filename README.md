# shogi-core

A Shogi (Japanese Chess) engine written in pure Rust, targeting AlphaZero-style
self-play training. No Python. No external engine dependencies.

Full stack goal: bitboard core → USI interface → classical search → MCTS →
neural network (tch-rs / LibTorch) → self-play training loop.

---

## Build

Requires Rust 1.77+ (edition 2024).

```sh
cargo build --release
```

`RUSTFLAGS="-C target-cpu=native"` is set in `.cargo/config.toml` and applies
automatically.

---

## Usage

### USI mode (default)

The binary speaks the [USI protocol](http://shogidojo.net/usi.htm) over
stdin/stdout, making it compatible with any USI-capable GUI (ShogiGUI,
Shogidroid, etc.):

```sh
./target/release/shogi        # starts USI loop
```

Example session:

```
usi
id name ShogiCore
id author rajnat
usiok
isready
readyok
position startpos
go movetime 1000
bestmove 7g7f
quit
```

### Perft (move-generation validation)

```sh
# Count leaf nodes at each depth (validates move generator)
./target/release/shogi perft --depth 5

# Per-move breakdown at a given depth
./target/release/shogi perft --depth 3 --divide
```

Expected output:

```
Perft(1) = 30
Perft(2) = 900
Perft(3) = 25470
Perft(4) = 719731
Perft(5) = 19861490
```

### Show startpos SFEN

```sh
./target/release/shogi startpos
# lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1
```

---

## Architecture

### Crate layout (`src/`)

| File | Contents |
|---|---|
| `types.rs` | `Color`, `PieceType` (14 variants), `Square` (u8, 0–80), `Move` (packed u32) |
| `bitboard.rs` | `Bitboard(u128)` — 81-bit board, full operator set, square iterator |
| `attacks.rs` | Precomputed attack tables in `OnceLock` — step pieces and sliding rays |
| `zobrist.rs` | Deterministic Zobrist keys (xorshift64, fixed seed) |
| `board.rs` | `Board` struct, `startpos()`, SFEN parse/serialize, `make/unmake` |
| `moves.rs` | `make_move_full` / `unmake_move_full` with captured-piece tracking |
| `movegen.rs` | `generate_legal_moves` — pseudo-legal + king-safety filter, nifu, uchifuzume |
| `search.rs` | Evaluation, negamax, alpha-beta (`Searcher`), move ordering |
| `perft.rs` | `perft()` and `perft_divide()` |
| `usi.rs` | Full USI protocol loop, position/go parsing, time management |

### Square encoding

```
sq = file_idx * 9 + rank_idx
file_idx : 0 = file 9 (right, White's lance)  …  8 = file 1 (left)
rank_idx : 0 = rank 1 (top, White's back rank) …  8 = rank 9 (bottom)
```

Black's pieces start at high rank_idx (6–8); White's at low rank_idx (0–2).
Black's "forward" direction is rank_idx − 1.

### Move encoding (packed u32)

```
bits  0– 6  from-square (or drop piece type)
bits  7–13  to-square
bits 14–17  piece type
bit  18     promote flag
bit  19     drop flag
```

### Search

The engine uses negamax with alpha-beta pruning and move ordering.
`Searcher` carries per-search statistics (node count) and search options.

**Move ordering** (best-first):

| Category | Score |
|---|---|
| Capture | `8000 + victim × 8 − attacker` (MVV-LVA) |
| Capture-promotion | capture score + promotion gain |
| Quiet promotion | `6000 + promotion gain` |
| Drop | `4000 + piece value` |
| Quiet | `0` |

**Node reduction from move ordering** (depth 4, startpos, all-quiet position):
unordered alpha-beta: 5 902 nodes → ordered: 4 056 nodes (32% reduction).
Positions with captures available show 60–80% reduction.

**Piece values (centipawns):**

| Piece | Value | Promoted | Value |
|---|---|---|---|
| Pawn | 100 | Tokin | 530 |
| Lance | 430 | Pro-Lance | 530 |
| Knight | 450 | Pro-Knight | 540 |
| Silver | 640 | Pro-Silver | 640 |
| Gold | 690 | — | — |
| Bishop | 890 | Dragon Horse | 1120 |
| Rook | 1040 | Dragon King | 1310 |

---

## Tests

```sh
cargo test               # 62 unit tests
cargo test -- --ignored  # + 3 slow perft tests (depths 3–5)
```

All five perft values match published Shogi perft tables:

```
depth 1 →        30
depth 2 →       900
depth 3 →    25 470
depth 4 →   719 731
depth 5 → 19 861 490
```

---

## References

