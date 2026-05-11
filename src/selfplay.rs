/// Self-play game runner for AlphaZero-style training data generation.
///
/// `play_game` runs a complete game between a network and itself, collecting
/// one `(board_tensor, policy_target, value_target)` record per position.
/// These records feed directly into the training loop.
///
/// # Record layout
///
/// | field | shape | meaning |
/// |-------|-------|---------|
/// | `board_tensor` | `[119, 9, 9]` | encoded board (network input) |
/// | `policy_target` | `Vec<f32>` len `NUM_ACTIONS` | MCTS visit distribution |
/// | `value_target` | `f32` ∈ {−1, 0, +1} | game outcome for the side to move |
///
/// The value target is **backfilled**: after the game ends, every record gets
/// z = +1 (that side won), −1 (that side lost), or 0 (draw).
///
/// # Resign
/// Once `config.resign_min_ply` half-moves have been played, if the network's
/// value estimate at the root stays below `config.resign_threshold` for
/// `config.resign_consecutive` consecutive plies, the side to move resigns.
use rand::Rng;
use tch::{Device, Tensor};

use crate::board::Board;
use crate::mcts::search::{eval_batch_with_net, mcts_search_batched};
use crate::mcts::{Arena, MctsConfig, NodeIdx};
use crate::movegen::generate_legal_moves;
use crate::moves::make_move_full;
use crate::nn::{encode, move_to_index, Net, NUM_ACTIONS};
use crate::types::Color;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Parameters controlling one self-play game.
#[derive(Debug, Clone)]
pub struct SelfPlayConfig {
    /// MCTS simulations per move.
    pub num_simulations: u32,
    /// Move-selection temperature for the first `temperature_drop_ply` half-moves.
    pub temperature_high: f32,
    /// Half-move ply at which temperature drops to `temperature_low`.
    pub temperature_drop_ply: u32,
    /// Temperature after `temperature_drop_ply` (0.0 = greedy argmax).
    pub temperature_low: f32,
    /// Resign when the root value estimate is below this threshold.
    pub resign_threshold: f32,
    /// Do not resign before this many half-moves have been played.
    pub resign_min_ply: u32,
    /// Resign after this many consecutive below-threshold plies.
    pub resign_consecutive: u32,
    /// Maximum half-moves before the game is declared a draw.
    pub max_moves: usize,
    /// Dirichlet α for root exploration noise.
    pub dirichlet_alpha: f32,
    /// ε for Dirichlet noise mixing (P' = (1−ε)·P + ε·η).
    pub dirichlet_epsilon: f32,
    /// c_puct exploration constant for PUCT selection.
    pub c_puct: f32,
    /// Number of leaf positions batched per neural-net MCTS evaluation.
    pub mcts_batch_size: usize,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        SelfPlayConfig {
            num_simulations: 800,
            temperature_high: 1.0,
            temperature_drop_ply: 30,
            temperature_low: 0.0,
            resign_threshold: -0.9,
            resign_min_ply: 30,
            resign_consecutive: 5,
            max_moves: 512,
            dirichlet_alpha: 0.15,
            dirichlet_epsilon: 0.25,
            c_puct: 1.0,
            mcts_batch_size: 8,
        }
    }
}

// ---------------------------------------------------------------------------
// Training record and game result
// ---------------------------------------------------------------------------

/// One training sample produced by self-play.
///
/// - `.0` — encoded board tensor `[119, 9, 9]`.
/// - `.1` — MCTS visit-count distribution over `NUM_ACTIONS`.
/// - `.2` — game outcome for the side to move (+1 win / −1 loss / 0 draw).
pub type GameRecord = (Tensor, Vec<f32>, f32);

/// Why a self-play game ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminationReason {
    /// The side to move had no legal moves (checkmated).
    Checkmate,
    /// The side to move resigned after consecutive below-threshold evaluations.
    Resign,
    /// The game hit `max_moves` without a decisive result (draw).
    MaxMoves,
}

/// Result of a complete self-play game.
pub struct SelfPlayResult {
    /// Training records, one per non-terminal position visited.
    pub records: Vec<GameRecord>,
    /// Game outcome from Black's perspective:
    ///   +1.0  Black wins (White was checkmated or resigned)
    ///   −1.0  White wins (Black was checkmated or resigned)
    ///    0.0  Draw (move limit reached)
    pub outcome: f32,
    /// How the game ended.
    pub termination: TerminationReason,
    /// Number of half-moves (plies) played; equal to `records.len()`.
    pub plies: usize,
    /// Mean Shannon entropy (nats) of the MCTS visit distribution across all plies.
    pub avg_visit_entropy: f32,
    /// Mean Shannon entropy (nats) of the network's softmax policy at the root across all plies.
    pub avg_policy_entropy: f32,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Shannon entropy (nats) of a probability distribution.
///
/// Zero-probability entries are skipped to avoid log(0).
pub fn entropy(dist: &[f32]) -> f32 {
    dist.iter().filter(|&&p| p > 0.0).map(|&p| -p * p.ln()).sum()
}

/// Shannon entropy (nats) of a softmax applied to raw logits.
///
/// Subtracts the max for numerical stability before exponentiating.
fn softmax_entropy(logits: &[f32]) -> f32 {
    if logits.is_empty() {
        return 0.0;
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|&e| e / sum).collect();
    entropy(&probs)
}

/// Extract a visit-count distribution over `NUM_ACTIONS` from the MCTS root.
///
/// Only legal children of the root carry visit counts; all other action
/// slots stay at 0.  The distribution is normalised to sum to 1.
fn visit_distribution(arena: &Arena, root: NodeIdx) -> Vec<f32> {
    let mut dist = vec![0.0f32; NUM_ACTIONS];
    let root_node = arena.get(root);
    let total: u32 = root_node
        .children
        .iter()
        .map(|&c| arena.get(c).visit_count)
        .sum();
    if total == 0 {
        return dist;
    }
    for &child_idx in &root_node.children {
        let child = arena.get(child_idx);
        if let Some(mv) = child.mv {
            dist[move_to_index(mv)] = child.visit_count as f32 / total as f32;
        }
    }
    dist
}

// ---------------------------------------------------------------------------
// play_game
// ---------------------------------------------------------------------------

/// Play one complete self-play game and return training records.
///
/// Both sides are played by `net`.  Dirichlet noise is always added at the
/// root to encourage exploration.  Temperature is `config.temperature_high`
/// for the first `config.temperature_drop_ply` half-moves, then
/// `config.temperature_low`.
///
/// The function runs entirely in inference mode (no gradient tracking).
pub fn play_game(
    net: &Net,
    config: &SelfPlayConfig,
    device: Device,
    rng: &mut impl Rng,
) -> SelfPlayResult {
    let base_cfg = MctsConfig {
        c_puct: config.c_puct,
        rollout_depth: 200, // unused — network replaces rollouts
        dirichlet_alpha: config.dirichlet_alpha,
        dirichlet_epsilon: config.dirichlet_epsilon,
        dirichlet_noise: true, // always on during self-play
        temperature: config.temperature_high,
        batch_size: config.mcts_batch_size,
    };

    let mut board = Board::startpos();
    let mut arena = Arena::new(200_000);

    // Intermediate storage: (board tensor, policy, side to move).
    // Value targets are backfilled once the outcome is known.
    let mut raw: Vec<(Tensor, Vec<f32>, Color)> = Vec::new();

    let mut resign_counter = 0u32;
    let mut resigned = false;

    // Per-ply entropy accumulators.
    let mut visit_entropies: Vec<f32> = Vec::new();
    let mut policy_entropies: Vec<f32> = Vec::new();

    for ply in 0..config.max_moves {
        // Check for terminal before searching.
        let mut legal = Vec::new();
        generate_legal_moves(&mut board, &mut legal);
        if legal.is_empty() {
            break;
        }

        // Temperature schedule.
        let temperature = if ply < config.temperature_drop_ply as usize {
            config.temperature_high
        } else {
            config.temperature_low
        };

        let cfg = MctsConfig {
            temperature,
            ..base_cfg.clone()
        };

        let (mv, root_value, root_policy_logits) = tch::no_grad(|| {
            mcts_search_batched(
                &mut arena,
                &mut board,
                config.num_simulations,
                &cfg,
                rng,
                |boards| eval_batch_with_net(net, device, boards),
            )
        });

        // If search returned None the position is terminal (shouldn't happen —
        // we pre-checked — but handle it defensively).
        let Some(chosen_move) = mv else { break };

        // Policy target: normalised visit counts from the root (index 0).
        let policy = visit_distribution(&arena, 0);

        // Entropy measurements for this ply.
        visit_entropies.push(entropy(&policy));
        policy_entropies.push(softmax_entropy(&root_policy_logits));

        // Record before applying the move.
        raw.push((encode(&board), policy, board.side_to_move));

        // Resign check.
        let v = root_value;
        if ply >= config.resign_min_ply as usize && v < config.resign_threshold {
            resign_counter += 1;
            if resign_counter >= config.resign_consecutive {
                resigned = true;
                break;
            }
        } else {
            resign_counter = 0;
        }

        make_move_full(&mut board, chosen_move);
    }

    // --- Determine outcome and termination from Black's perspective ---
    //
    // `board.side_to_move` at exit tells us who would move next (or who
    // resigned if resigned=true).
    let (outcome_for_black, termination): (f32, TerminationReason) = if resigned {
        // The side that would move next is the one that resigned.
        let outcome = if board.side_to_move == Color::Black { -1.0 } else { 1.0 };
        (outcome, TerminationReason::Resign)
    } else {
        let mut probe = Vec::new();
        generate_legal_moves(&mut board, &mut probe);
        if probe.is_empty() {
            // No legal moves → the side to move is checkmated.
            let outcome = if board.side_to_move == Color::Black { -1.0 } else { 1.0 };
            (outcome, TerminationReason::Checkmate)
        } else {
            (0.0, TerminationReason::MaxMoves)
        }
    };

    // --- Backfill value targets ---
    //
    // Each record stores the side to move at that ply.  Convert the
    // Black-perspective outcome to the current-player perspective.
    let records: Vec<GameRecord> = raw
        .into_iter()
        .map(|(tensor, policy, side)| {
            let z = if side == Color::Black {
                outcome_for_black
            } else {
                -outcome_for_black
            };
            (tensor, policy, z)
        })
        .collect();

    let plies = records.len();
    let avg_visit_entropy = if visit_entropies.is_empty() {
        0.0
    } else {
        visit_entropies.iter().sum::<f32>() / visit_entropies.len() as f32
    };
    let avg_policy_entropy = if policy_entropies.is_empty() {
        0.0
    } else {
        policy_entropies.iter().sum::<f32>() / policy_entropies.len() as f32
    };
    SelfPlayResult {
        records,
        outcome: outcome_for_black,
        termination,
        plies,
        avg_visit_entropy,
        avg_policy_entropy,
    }
}

// ---------------------------------------------------------------------------
// play_pit_game
// ---------------------------------------------------------------------------

/// Play one game between two distinct networks for evaluation purposes.
///
/// `net_black` controls Black (Sente, moves first); `net_white` controls White.
/// Unlike `play_game`, no Dirichlet noise is added — this is pure evaluation.
/// Returns the outcome from Black's perspective (+1 Black wins, −1 White wins, 0 draw).
pub fn play_pit_game(
    net_black: &Net,
    net_white: &Net,
    config: &SelfPlayConfig,
    device: Device,
    rng: &mut impl Rng,
) -> f32 {
    let base_cfg = MctsConfig {
        c_puct: config.c_puct,
        rollout_depth: 200,
        dirichlet_alpha: config.dirichlet_alpha,
        dirichlet_epsilon: 0.0, // no noise during evaluation
        dirichlet_noise: false,
        temperature: config.temperature_low,
        batch_size: config.mcts_batch_size,
    };

    let mut board = Board::startpos();
    let mut arena = Arena::new(200_000);

    let mut resign_counter = 0u32;
    let mut resigned = false;

    for ply in 0..config.max_moves {
        let mut legal = Vec::new();
        generate_legal_moves(&mut board, &mut legal);
        if legal.is_empty() {
            break;
        }

        let net = if board.side_to_move == Color::Black {
            net_black
        } else {
            net_white
        };

        let (mv, root_value, _) = tch::no_grad(|| {
            mcts_search_batched(
                &mut arena,
                &mut board,
                config.num_simulations,
                &base_cfg,
                rng,
                |boards| eval_batch_with_net(net, device, boards),
            )
        });

        let Some(chosen_move) = mv else { break };

        let v = root_value;
        if ply >= config.resign_min_ply as usize && v < config.resign_threshold {
            resign_counter += 1;
            if resign_counter >= config.resign_consecutive {
                resigned = true;
                break;
            }
        } else {
            resign_counter = 0;
        }

        make_move_full(&mut board, chosen_move);
    }

    if resigned {
        if board.side_to_move == Color::Black {
            -1.0
        } else {
            1.0
        }
    } else {
        let mut probe = Vec::new();
        generate_legal_moves(&mut board, &mut probe);
        if probe.is_empty() {
            if board.side_to_move == Color::Black {
                -1.0
            } else {
                1.0
            }
        } else {
            0.0
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::checkpoint::build_with_config;
    use tch::Device;

    fn small_config() -> SelfPlayConfig {
        SelfPlayConfig {
            num_simulations: 4,
            temperature_high: 1.0,
            temperature_drop_ply: 10,
            temperature_low: 0.0,
            resign_threshold: -0.9,
            resign_min_ply: 20, // won't trigger in a short game
            resign_consecutive: 3,
            max_moves: 20,
            ..SelfPlayConfig::default()
        }
    }

    // ----- entropy helper -----

    #[test]
    fn test_entropy_uniform_distribution() {
        // Uniform over N outcomes → H = ln(N).
        let n = 4usize;
        let uniform = vec![1.0f32 / n as f32; n];
        let h = entropy(&uniform);
        let expected = (n as f32).ln();
        assert!((h - expected).abs() < 1e-5, "uniform entropy: got {h}, expected {expected}");
    }

    #[test]
    fn test_entropy_deterministic_distribution() {
        // One outcome with probability 1 → H = 0.
        let mut dist = vec![0.0f32; 8];
        dist[3] = 1.0;
        let h = entropy(&dist);
        assert!(h.abs() < 1e-6, "deterministic entropy must be 0, got {h}");
    }

    #[test]
    fn test_entropy_two_outcomes() {
        // p=0.5, q=0.5 → H = ln(2) ≈ 0.6931.
        let dist = vec![0.5f32, 0.5];
        let h = entropy(&dist);
        assert!((h - 2.0f32.ln()).abs() < 1e-5, "binary entropy: got {h}");
    }

    #[test]
    fn test_entropy_is_non_negative() {
        let dist = vec![0.1f32, 0.3, 0.6];
        assert!(entropy(&dist) >= 0.0);
    }

    // ----- visit_distribution -----

    #[test]
    fn test_visit_distribution_sums_to_one() {
        use crate::board::Board;
        use crate::mcts::search::expand;
        use crate::mcts::{Arena, Node, NO_PARENT};
        use crate::nn::move_index::NUM_ACTIONS;

        let mut board = Board::startpos();
        let mut arena = Arena::new(512);
        let root = arena.alloc(Node::new(None, 1.0, NO_PARENT));
        expand(&mut arena, root, &mut board);

        // Give every child a visit count of 1.
        let children: Vec<_> = arena.get(root).children.clone();
        for &c in &children {
            arena.get_mut(c).visit_count = 1;
        }

        let dist = visit_distribution(&arena, root);
        let sum: f32 = dist.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "distribution should sum to 1, got {sum}"
        );
        assert_eq!(dist.len(), NUM_ACTIONS);
    }

    #[test]
    fn test_visit_distribution_zero_visits_returns_zeros() {
        use crate::board::Board;
        use crate::mcts::search::expand;
        use crate::mcts::{Arena, Node, NO_PARENT};

        let mut board = Board::startpos();
        let mut arena = Arena::new(512);
        let root = arena.alloc(Node::new(None, 1.0, NO_PARENT));
        expand(&mut arena, root, &mut board);
        // No visits assigned → all zero.

        let dist = visit_distribution(&arena, root);
        assert!(dist.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_visit_distribution_correct_action_slot() {
        use crate::board::Board;
        use crate::mcts::search::expand;
        use crate::mcts::{Arena, Node, NO_PARENT};
        use crate::nn::move_to_index;

        let mut board = Board::startpos();
        let mut arena = Arena::new(512);
        let root = arena.alloc(Node::new(None, 1.0, NO_PARENT));
        expand(&mut arena, root, &mut board);

        // Give only the first child a visit.
        let first_child_idx = arena.get(root).children[0];
        let first_mv = arena.get(first_child_idx).mv.unwrap();
        arena.get_mut(first_child_idx).visit_count = 1;

        let dist = visit_distribution(&arena, root);
        let expected_slot = move_to_index(first_mv);
        assert!(
            (dist[expected_slot] - 1.0).abs() < 1e-6,
            "expected slot {expected_slot} to be 1.0"
        );
        let nonzero: Vec<usize> = dist
            .iter()
            .enumerate()
            .filter(|&(_, &v)| v > 0.0)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(nonzero, vec![expected_slot]);
    }

    // ----- play_game smoke tests -----

    #[test]
    fn test_play_game_returns_records() {
        let (_vs, net) = build_with_config(Device::Cpu, 8, 2);
        let config = small_config();
        let mut rng = rand::thread_rng();
        let result = play_game(&net, &config, Device::Cpu, &mut rng);
        assert!(
            !result.records.is_empty(),
            "play_game should return at least one record"
        );
    }

    #[test]
    fn test_play_game_tensor_shape() {
        let (_vs, net) = build_with_config(Device::Cpu, 8, 2);
        let config = small_config();
        let mut rng = rand::thread_rng();
        let result = play_game(&net, &config, Device::Cpu, &mut rng);
        let (tensor, _, _) = &result.records[0];
        assert_eq!(tensor.size(), vec![119, 9, 9]);
    }

    #[test]
    fn test_play_game_policy_length() {
        let (_vs, net) = build_with_config(Device::Cpu, 8, 2);
        let config = small_config();
        let mut rng = rand::thread_rng();
        let result = play_game(&net, &config, Device::Cpu, &mut rng);
        for (_, policy, _) in &result.records {
            assert_eq!(policy.len(), NUM_ACTIONS);
        }
    }

    #[test]
    fn test_play_game_policy_sums_to_one() {
        let (_vs, net) = build_with_config(Device::Cpu, 8, 2);
        let config = small_config();
        let mut rng = rand::thread_rng();
        let result = play_game(&net, &config, Device::Cpu, &mut rng);
        for (_, policy, _) in &result.records {
            let sum: f32 = policy.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-4,
                "policy should sum to ~1.0, got {sum}"
            );
        }
    }

    #[test]
    fn test_play_game_value_targets_in_range() {
        let (_vs, net) = build_with_config(Device::Cpu, 8, 2);
        let config = small_config();
        let mut rng = rand::thread_rng();
        let result = play_game(&net, &config, Device::Cpu, &mut rng);
        for (_, _, z) in &result.records {
            assert!(
                *z == -1.0 || *z == 0.0 || *z == 1.0,
                "value target must be -1, 0, or +1, got {z}"
            );
        }
    }

    #[test]
    fn test_play_game_respects_max_moves() {
        let (_vs, net) = build_with_config(Device::Cpu, 8, 2);
        let config = small_config(); // max_moves = 20
        let mut rng = rand::thread_rng();
        let result = play_game(&net, &config, Device::Cpu, &mut rng);
        assert!(
            result.records.len() <= config.max_moves,
            "records ({}) must not exceed max_moves ({})",
            result.records.len(),
            config.max_moves,
        );
    }

    #[test]
    fn test_play_game_value_consistent_across_plies() {
        let (_vs, net) = build_with_config(Device::Cpu, 8, 2);
        let config = small_config();
        let mut rng = rand::thread_rng();
        let result = play_game(&net, &config, Device::Cpu, &mut rng);

        // If the game ended decisively, all z are ±1 (no mixing of 0 and ±1).
        let has_zero = result.records.iter().any(|(_, _, z)| *z == 0.0);
        let has_nonzero = result.records.iter().any(|(_, _, z)| *z != 0.0);
        if has_zero {
            assert!(!has_nonzero, "draw outcome should make all z = 0");
        }
    }

    // ----- outcome-specific tests -----

    /// Draw by move limit: outcome must be 0.0 and all z values must be 0.0.
    #[test]
    fn test_outcome_draw_by_move_limit() {
        let (_vs, net) = build_with_config(Device::Cpu, 8, 2);
        // max_moves = 2 guarantees a draw: no game ends that quickly.
        let config = SelfPlayConfig {
            num_simulations: 4,
            max_moves: 2,
            resign_min_ply: 999, // no resign
            ..SelfPlayConfig::default()
        };
        let mut rng = rand::thread_rng();
        let result = play_game(&net, &config, Device::Cpu, &mut rng);

        assert_eq!(result.outcome, 0.0, "move-limit game must be a draw");
        for (_, _, z) in &result.records {
            assert_eq!(*z, 0.0, "all z values must be 0 in a draw");
        }
    }

    /// Resign: when the resign threshold is impossible to avoid, the first move
    /// triggers resignation and the outcome is decisive (±1.0).
    #[test]
    fn test_outcome_resign_is_decisive() {
        let (_vs, net) = build_with_config(Device::Cpu, 8, 2);
        let config = SelfPlayConfig {
            num_simulations: 4,
            resign_threshold: 2.0, // always triggers (tanh output < 1.0 always)
            resign_min_ply: 0,
            resign_consecutive: 1,
            max_moves: 50,
            ..SelfPlayConfig::default()
        };
        let mut rng = rand::thread_rng();
        let result = play_game(&net, &config, Device::Cpu, &mut rng);

        assert!(
            result.outcome == 1.0 || result.outcome == -1.0,
            "resign must produce a decisive outcome, got {}",
            result.outcome
        );
    }

    /// After a resign the outcome must be reflected in every z value.
    #[test]
    fn test_resign_z_consistent_with_outcome() {
        let (_vs, net) = build_with_config(Device::Cpu, 8, 2);
        let config = SelfPlayConfig {
            num_simulations: 4,
            resign_threshold: 2.0,
            resign_min_ply: 0,
            resign_consecutive: 1,
            max_moves: 50,
            ..SelfPlayConfig::default()
        };
        let mut rng = rand::thread_rng();
        let result = play_game(&net, &config, Device::Cpu, &mut rng);

        // Every z must be ±1; none should be 0 in a resigned game.
        for (_, _, z) in &result.records {
            assert!(
                *z == 1.0 || *z == -1.0,
                "resigned game must have z ∈ {{-1, +1}}, got {z}"
            );
        }
    }

    // ----- backfill tests -----
    //
    // The backfill rule:
    //   z(ply) = +outcome_for_black   when side_to_move == Black
    //   z(ply) = -outcome_for_black   when side_to_move == White
    //
    // We control the exact resign ply to make these tests deterministic.

    fn immediate_resign_config(consecutive: u32) -> SelfPlayConfig {
        SelfPlayConfig {
            num_simulations: 4,
            resign_threshold: 2.0, // tanh output is always < 1 < 2, so always triggers
            resign_min_ply: 0,
            resign_consecutive: consecutive,
            max_moves: 50,
            ..SelfPlayConfig::default()
        }
    }

    /// resign_consecutive=1 → resign fires on ply 0 (Black to move).
    /// Black resigned → outcome = −1.0; the one record (Black's turn) gets z = −1.0.
    #[test]
    fn test_backfill_black_resigns_at_ply0() {
        let (_vs, net) = build_with_config(Device::Cpu, 8, 2);
        let mut rng = rand::thread_rng();
        let result = play_game(&net, &immediate_resign_config(1), Device::Cpu, &mut rng);

        assert_eq!(result.outcome, -1.0, "Black resigned → outcome must be −1");
        assert_eq!(result.records.len(), 1);
        assert_eq!(
            result.records[0].2, -1.0,
            "Black-to-move record must carry z = −1"
        );
    }

    /// resign_consecutive=2 → resign fires on ply 1 (White to move).
    /// White resigned → outcome = +1.0 (Black wins).
    /// Record 0 (Black to move): z = +1.0. Record 1 (White to move): z = −1.0.
    #[test]
    fn test_backfill_white_resigns_at_ply1() {
        let (_vs, net) = build_with_config(Device::Cpu, 8, 2);
        let mut rng = rand::thread_rng();
        let result = play_game(&net, &immediate_resign_config(2), Device::Cpu, &mut rng);

        assert_eq!(
            result.outcome, 1.0,
            "White resigned → outcome must be +1 for Black"
        );
        assert_eq!(result.records.len(), 2);
        assert_eq!(
            result.records[0].2, 1.0,
            "Black-to-move record must carry z = +1"
        );
        assert_eq!(
            result.records[1].2, -1.0,
            "White-to-move record must carry z = −1"
        );
    }

    /// In a decisive game, consecutive records always have opposite z values
    /// because colors alternate every half-move.
    #[test]
    fn test_backfill_z_alternates_each_ply() {
        let (_vs, net) = build_with_config(Device::Cpu, 8, 2);
        // resign_consecutive=4 → 4 records before resign.
        let mut rng = rand::thread_rng();
        let result = play_game(&net, &immediate_resign_config(4), Device::Cpu, &mut rng);

        // Only test alternation when the game was decisive.
        if result.outcome == 0.0 {
            return;
        }
        for w in result.records.windows(2) {
            let z0 = w[0].2;
            let z1 = w[1].2;
            assert_eq!(
                z0, -z1,
                "consecutive z values must be opposite (got {z0} then {z1})"
            );
        }
    }

    /// Draw by move limit: every z must equal 0.0, matching outcome = 0.0.
    #[test]
    fn test_backfill_draw_gives_zero_for_every_ply() {
        let (_vs, net) = build_with_config(Device::Cpu, 8, 2);
        let config = SelfPlayConfig {
            num_simulations: 4,
            max_moves: 4,
            resign_min_ply: 999, // no resign
            ..SelfPlayConfig::default()
        };
        let mut rng = rand::thread_rng();
        let result = play_game(&net, &config, Device::Cpu, &mut rng);

        assert_eq!(result.outcome, 0.0, "move-limit game is a draw");
        for (i, (_, _, z)) in result.records.iter().enumerate() {
            assert_eq!(*z, 0.0, "ply {i}: draw must give z = 0, got {z}");
        }
    }

    /// outcome is always one of the three legal values.
    #[test]
    fn test_outcome_is_legal_value() {
        let (_vs, net) = build_with_config(Device::Cpu, 8, 2);
        let config = small_config();
        let mut rng = rand::thread_rng();
        let result = play_game(&net, &config, Device::Cpu, &mut rng);
        assert!(
            result.outcome == -1.0 || result.outcome == 0.0 || result.outcome == 1.0,
            "outcome must be -1, 0, or +1, got {}",
            result.outcome
        );
    }

    /// The move chosen by MCTS must appear in the recorded policy distribution.
    ///
    /// This is the central invariant of the data pipeline: the policy target
    /// π recorded for training always assigns non-zero weight to the move
    /// that was actually played.
    #[test]
    fn test_chosen_move_has_nonzero_policy_weight() {
        use crate::board::Board;
        use crate::mcts::search::{eval_with_net, mcts_search_with_evaluator};
        use crate::mcts::{Arena, MctsConfig};
        use std::cell::Cell;

        let (_vs, net) = build_with_config(Device::Cpu, 8, 2);
        let mut arena = Arena::new(50_000);
        let mut board = Board::startpos();
        let cfg = MctsConfig {
            temperature: 1.0,
            ..MctsConfig::default()
        };
        let mut rng = rand::thread_rng();
        let call_count = Cell::new(0u32);

        let mv = tch::no_grad(|| {
            mcts_search_with_evaluator(
                &mut arena,
                &mut board,
                8, // small but enough to build the tree
                &cfg,
                &mut rng,
                |b| {
                    let r = eval_with_net(&net, Device::Cpu, b);
                    call_count.set(call_count.get() + 1);
                    r
                },
            )
        });

        let chosen = mv.expect("startpos is not terminal");
        let policy = visit_distribution(&arena, 0);
        let slot = move_to_index(chosen);
        assert!(
            policy[slot] > 0.0,
            "chosen move (slot {slot}) must have non-zero weight in policy, got {:.6}",
            policy[slot]
        );
    }

    // ----- termination reason and plies -----

    #[test]
    fn test_plies_matches_records_length() {
        let (_vs, net) = build_with_config(Device::Cpu, 8, 2);
        let config = small_config();
        let mut rng = rand::thread_rng();
        let result = play_game(&net, &config, Device::Cpu, &mut rng);
        assert_eq!(result.plies, result.records.len());
    }

    #[test]
    fn test_termination_max_moves() {
        let (_vs, net) = build_with_config(Device::Cpu, 8, 2);
        let config = SelfPlayConfig {
            num_simulations: 4,
            max_moves: 2,
            resign_min_ply: 999,
            ..SelfPlayConfig::default()
        };
        let mut rng = rand::thread_rng();
        let result = play_game(&net, &config, Device::Cpu, &mut rng);
        assert_eq!(result.termination, TerminationReason::MaxMoves);
        assert_eq!(result.outcome, 0.0);
    }

    #[test]
    fn test_termination_resign() {
        let (_vs, net) = build_with_config(Device::Cpu, 8, 2);
        let config = SelfPlayConfig {
            num_simulations: 4,
            resign_threshold: 2.0, // always fires
            resign_min_ply: 0,
            resign_consecutive: 1,
            max_moves: 50,
            ..SelfPlayConfig::default()
        };
        let mut rng = rand::thread_rng();
        let result = play_game(&net, &config, Device::Cpu, &mut rng);
        assert_eq!(result.termination, TerminationReason::Resign);
        assert!(result.outcome == 1.0 || result.outcome == -1.0);
    }

    // ----- resign logic tests -----
    //
    // resign_threshold = 2.0 makes the condition (v < threshold) always true,
    // because the value head uses tanh whose output is in (−1, 1).
    // This lets us precisely control when resign fires without needing a mock net.

    fn always_resign_config(min_ply: u32, consecutive: u32) -> SelfPlayConfig {
        SelfPlayConfig {
            num_simulations: 4,
            resign_threshold: 2.0, // always satisfied
            resign_min_ply: min_ply,
            resign_consecutive: consecutive,
            max_moves: 50,
            ..SelfPlayConfig::default()
        }
    }

    // Plies and their colors from startpos (Black always moves first):
    //   ply 0 → Black,  ply 1 → White,  ply 2 → Black,  ply 3 → White, …
    // Colors alternate so ply N is Black when N is even, White when N is odd.

    /// Resign fires on ply 0 (Black to move): one record, outcome = −1.
    #[test]
    fn test_resign_fires_on_first_consecutive_ply() {
        let (_vs, net) = build_with_config(Device::Cpu, 8, 2);
        let mut rng = rand::thread_rng();
        let result = play_game(&net, &always_resign_config(0, 1), Device::Cpu, &mut rng);

        // resign_consecutive = 1 → resign on the very first ply that satisfies the threshold.
        assert_eq!(result.records.len(), 1, "exactly one record before resign");
        assert_eq!(
            result.outcome, -1.0,
            "Black (ply 0) resigned → outcome = −1"
        );
    }

    /// Resign requires N consecutive plies below the threshold.
    /// With consecutive=3 and threshold always satisfied, resign fires on ply 2 (Black).
    #[test]
    fn test_resign_fires_after_exactly_n_consecutive_plies() {
        let (_vs, net) = build_with_config(Device::Cpu, 8, 2);
        let mut rng = rand::thread_rng();
        let result = play_game(&net, &always_resign_config(0, 3), Device::Cpu, &mut rng);

        // Plies 0,1,2 all satisfy threshold → resign on ply 2 (Black to move).
        assert_eq!(result.records.len(), 3, "three records for plies 0–2");
        assert_eq!(
            result.outcome, -1.0,
            "Black (ply 2) resigned → outcome = −1"
        );
    }

    /// resign_min_ply prevents early resignation.
    /// With min_ply=5 and consecutive=1, resign fires on ply 5 (White to move).
    #[test]
    fn test_resign_not_before_min_ply() {
        let (_vs, net) = build_with_config(Device::Cpu, 8, 2);
        let mut rng = rand::thread_rng();
        let result = play_game(&net, &always_resign_config(5, 1), Device::Cpu, &mut rng);

        // Plies 0–4 are guarded by resign_min_ply=5 (counter stays 0).
        // Ply 5 (White, first eligible ply): counter=1 ≥ 1 → resign.
        assert_eq!(result.records.len(), 6, "six records for plies 0–5");
        assert_eq!(
            result.outcome, 1.0,
            "White (ply 5) resigned → outcome = +1 for Black"
        );
    }

    /// Resign never fires when the threshold is impossible to satisfy.
    /// tanh output ∈ (−1, 1), so threshold = −2.0 is never met; game ends by draw.
    #[test]
    fn test_resign_never_fires_when_threshold_not_met() {
        let (_vs, net) = build_with_config(Device::Cpu, 8, 2);
        let config = SelfPlayConfig {
            num_simulations: 4,
            resign_threshold: -2.0, // never satisfied
            resign_min_ply: 0,
            resign_consecutive: 1,
            max_moves: 4,
            ..SelfPlayConfig::default()
        };
        let mut rng = rand::thread_rng();
        let result = play_game(&net, &config, Device::Cpu, &mut rng);

        assert_eq!(
            result.outcome, 0.0,
            "threshold never met → draw by move limit"
        );
    }

    /// If the counter has not yet reached consecutive, resign must not fire.
    /// consecutive=3, max_moves=2 → counter reaches 2 but the loop ends first.
    #[test]
    fn test_resign_requires_full_consecutive_count() {
        let (_vs, net) = build_with_config(Device::Cpu, 8, 2);
        let config = SelfPlayConfig {
            num_simulations: 4,
            resign_threshold: 2.0,
            resign_min_ply: 0,
            resign_consecutive: 3,
            max_moves: 2, // loop ends after 2 plies, counter reaches 2 < 3
            ..SelfPlayConfig::default()
        };
        let mut rng = rand::thread_rng();
        let result = play_game(&net, &config, Device::Cpu, &mut rng);

        // Counter: ply 0 → 1, ply 1 → 2; loop exits; no resign.
        assert_eq!(result.outcome, 0.0, "consecutive count not reached → draw");
    }

    /// Each move changes the board — consecutive records should have different tensors.
    #[test]
    fn test_sequential_records_have_distinct_boards() {
        let (_vs, net) = build_with_config(Device::Cpu, 8, 2);
        let config = small_config();
        let mut rng = rand::thread_rng();
        let result = play_game(&net, &config, Device::Cpu, &mut rng);

        if result.records.len() < 2 {
            return; // game ended on the first move — nothing to compare
        }
        for w in result.records.windows(2) {
            let (t0, _, _) = &w[0];
            let (t1, _, _) = &w[1];
            let max_diff = (t0 - t1).abs().max().double_value(&[]);
            assert!(
                max_diff > 0.0,
                "consecutive board tensors should differ (max diff = {max_diff})"
            );
        }
    }
}
