/// Strength benchmark: pit two MCTS agents against each other in a match.
///
/// Usage (from main.rs):
/// ```text
///   shogi bench                         # random-weight net vs rollout, 20 games
///   shogi bench --games 100 --sims 400  # more games / deeper search
///   shogi bench --checkpoint run/weights.safetensors --channels 256 --blocks 20
/// ```
use std::time::Duration;

use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::board::Board;
use crate::mcts::{Arena, MctsConfig};
use crate::mcts::search::{mcts_search, mcts_search_with_net};
use crate::movegen::generate_legal_moves;
use crate::moves::make_move_full;
use crate::nn::Net;
use crate::types::{Color, Move};

// ---------------------------------------------------------------------------
// Agent
// ---------------------------------------------------------------------------

/// Which evaluation backend to use for an MCTS agent.
pub enum AgentKind<'a> {
    /// Uniform prior + random rollout (the classical baseline).
    Rollout,
    /// Policy + value network (the AlphaZero-style agent).
    Network { net: &'a Net, device: tch::Device },
}

fn agent_move<'a, R: rand::Rng>(
    kind: &AgentKind<'a>,
    arena: &mut Arena,
    board: &mut Board,
    sims: u32,
    cfg: &MctsConfig,
    rng: &mut R,
) -> Option<Move> {
    match kind {
        AgentKind::Rollout => mcts_search(arena, board, sims, cfg, rng),
        AgentKind::Network { net, device } => {
            mcts_search_with_net(arena, board, sims, cfg, rng, net, *device)
        }
    }
}

// ---------------------------------------------------------------------------
// Game outcome
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
enum GameOutcome {
    BlackWins,
    WhiteWins,
    Draw,
}

fn play_game<'a>(
    black: &AgentKind<'a>,
    white: &AgentKind<'a>,
    sims: u32,
    cfg: &MctsConfig,
    max_half_moves: usize,
    rng: &mut StdRng,
) -> GameOutcome {
    let mut board = Board::startpos();
    let mut black_arena = Arena::new(100_000);
    let mut white_arena = Arena::new(100_000);

    for _ in 0..max_half_moves {
        let mut legal = Vec::new();
        generate_legal_moves(&mut board, &mut legal);

        if legal.is_empty() {
            return if board.side_to_move == Color::Black {
                GameOutcome::WhiteWins
            } else {
                GameOutcome::BlackWins
            };
        }

        let (kind, arena) = if board.side_to_move == Color::Black {
            (black, &mut black_arena)
        } else {
            (white, &mut white_arena)
        };

        let mv = agent_move(kind, arena, &mut board, sims, cfg, rng)
            .unwrap_or(*legal.first().unwrap());

        make_move_full(&mut board, mv);
    }

    GameOutcome::Draw
}

// ---------------------------------------------------------------------------
// Match result
// ---------------------------------------------------------------------------

/// Aggregated result of a multi-game match between two agents.
#[derive(Debug, Default)]
pub struct MatchResult {
    pub wins_a: u32,
    pub wins_b: u32,
    pub draws: u32,
}

impl MatchResult {
    pub fn total(&self) -> u32 {
        self.wins_a + self.wins_b + self.draws
    }

    /// Score for A: (wins + 0.5*draws) / total, the standard chess-style score.
    pub fn score_a(&self) -> f64 {
        let n = self.total();
        if n == 0 {
            return 0.5;
        }
        (self.wins_a as f64 + 0.5 * self.draws as f64) / n as f64
    }

    /// 95 % Wilson score confidence interval for `score_a`.
    pub fn wilson_ci_95(&self) -> (f64, f64) {
        let n = self.total() as f64;
        if n == 0.0 {
            return (0.0, 1.0);
        }
        let p = self.score_a();
        let z = 1.96_f64;
        let z2 = z * z;
        let denom = 1.0 + z2 / n;
        let center = (p + z2 / (2.0 * n)) / denom;
        let margin = (z / denom) * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt();
        ((center - margin).max(0.0), (center + margin).min(1.0))
    }

    /// Print a formatted summary to stdout.
    pub fn print_summary(&self, name_a: &str, name_b: &str, elapsed: Duration) {
        let n = self.total();
        let pct = |v: u32| 100.0 * v as f64 / n.max(1) as f64;
        let (lo, hi) = self.wilson_ci_95();
        println!();
        println!("═══════════════════════════════════════════════");
        println!("{:>14} wins: {:3}/{} ({:4.1}%)", name_a, self.wins_a, n, pct(self.wins_a));
        println!("{:>14} wins: {:3}/{} ({:4.1}%)", name_b, self.wins_b, n, pct(self.wins_b));
        println!("         Draws:      {:3}/{} ({:4.1}%)", self.draws, n, pct(self.draws));
        println!(
            "{} score: {:.1}%  [{:.1}%, {:.1}%] 95% CI",
            name_a,
            self.score_a() * 100.0,
            lo * 100.0,
            hi * 100.0,
        );
        println!(
            "Elapsed: {:.1}s  ({:.1}s/game)",
            elapsed.as_secs_f64(),
            elapsed.as_secs_f64() / n.max(1) as f64,
        );
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Play `num_games` games between `agent_a` (labelled `name_a`) and
/// `agent_b`, alternating colors so each agent plays an equal number of
/// games as Black.
///
/// All games are sequential.  `verbose` prints a one-line summary per game.
pub fn run_match<'a>(
    agent_a: &AgentKind<'a>,
    name_a: &str,
    agent_b: &AgentKind<'a>,
    name_b: &str,
    num_games: u32,
    sims: u32,
    max_half_moves: usize,
    verbose: bool,
    seed: u64,
) -> MatchResult {
    let mut result = MatchResult::default();
    let mut rng = StdRng::seed_from_u64(seed);
    let cfg = MctsConfig::default();

    for game_idx in 0..num_games {
        let a_is_black = game_idx % 2 == 0;
        let (black, white) = if a_is_black {
            (agent_a, agent_b)
        } else {
            (agent_b, agent_a)
        };

        let outcome = play_game(black, white, sims, &cfg, max_half_moves, &mut rng);

        match outcome {
            GameOutcome::Draw => {
                result.draws += 1;
                if verbose {
                    println!("Game {:3}: Draw (move limit)", game_idx + 1);
                }
            }
            other => {
                let a_wins = match other {
                    GameOutcome::BlackWins => a_is_black,
                    GameOutcome::WhiteWins => !a_is_black,
                    GameOutcome::Draw => unreachable!(),
                };
                if a_wins {
                    result.wins_a += 1;
                } else {
                    result.wins_b += 1;
                }
                if verbose {
                    let winner = if a_wins { name_a } else { name_b };
                    let color_a = if a_is_black { "B" } else { "W" };
                    let color_b = if a_is_black { "W" } else { "B" };
                    println!(
                        "Game {:3}: {}/{} vs {}/{} — {}",
                        game_idx + 1,
                        name_a, color_a,
                        name_b, color_b,
                        winner,
                    );
                }
            }
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_match_result_score_symmetry() {
        let r = MatchResult { wins_a: 5, wins_b: 5, draws: 0 };
        assert!((r.score_a() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_match_result_score_all_a() {
        let r = MatchResult { wins_a: 10, wins_b: 0, draws: 0 };
        assert!((r.score_a() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_match_result_score_draws_half() {
        let r = MatchResult { wins_a: 0, wins_b: 0, draws: 10 };
        assert!((r.score_a() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_wilson_ci_contains_50_for_equal() {
        let r = MatchResult { wins_a: 5, wins_b: 5, draws: 0 };
        let (lo, hi) = r.wilson_ci_95();
        assert!(lo < 0.5 && 0.5 < hi);
    }

    #[test]
    fn test_wilson_ci_zero_games() {
        let r = MatchResult::default();
        let (lo, hi) = r.wilson_ci_95();
        assert_eq!(lo, 0.0);
        assert_eq!(hi, 1.0);
    }

    /// Smoke test: two rollout agents playing a 4-game mini-match should not panic.
    #[test]
    fn test_rollout_vs_rollout_smoke() {
        let a = AgentKind::Rollout;
        let b = AgentKind::Rollout;
        let result = run_match(&a, "RolloutA", &b, "RolloutB", 4, 50, 80, false, 1234);
        assert_eq!(result.total(), 4);
    }

    #[test]
    fn test_play_game_terminates() {
        let mut rng = StdRng::seed_from_u64(42);
        let cfg = MctsConfig::default();
        let a = AgentKind::Rollout;
        let b = AgentKind::Rollout;
        let outcome = play_game(&a, &b, 50, &cfg, 100, &mut rng);
        let _ = outcome; // just check it doesn't hang or panic
    }
}
