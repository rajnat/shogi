/// Tunable parameters for MCTS search.
#[derive(Debug, Clone)]
pub struct MctsConfig {
    /// Exploration constant in the PUCT formula.
    /// Higher values push the search toward less-visited, high-prior moves;
    /// lower values exploit the current Q estimates more aggressively.
    /// AlphaZero used values in the range 1.0–5.0. Default: 1.0.
    pub c_puct: f32,

    /// Maximum depth of a single random rollout.
    /// Rollouts that reach this depth without a terminal are scored as draws.
    /// Default: 200.
    pub rollout_depth: usize,

    /// Dirichlet concentration parameter α for root noise.
    /// Each root child receives noise drawn from Dirichlet(α, …, α).
    /// AlphaZero used 0.3 for chess; 0.15 is a common starting point.
    /// Default: 0.15.
    pub dirichlet_alpha: f32,

    /// Fraction of Dirichlet noise mixed into root priors: P' = (1−ε)·P + ε·η.
    /// AlphaZero used 0.25. Default: 0.25.
    pub dirichlet_epsilon: f32,

    /// Enable Dirichlet noise injection at the root.
    /// Should be `true` during self-play, `false` for analysis/search.
    /// Default: false.
    pub dirichlet_noise: bool,

    /// Move-selection temperature τ applied to root visit counts.
    ///
    /// Move a is chosen with probability proportional to N(a)^(1/τ):
    ///   τ = 0  — greedy argmax (deterministic, best for analysis)
    ///   τ = 1  — sample proportional to visit count (AlphaZero early-game)
    ///   τ > 1  — flatter distribution, more exploratory
    ///
    /// AlphaZero uses τ=1 for the first ~30 moves of each self-play game,
    /// then drops to τ→0. Default: 0.0 (greedy).
    pub temperature: f32,
}

impl Default for MctsConfig {
    fn default() -> Self {
        MctsConfig {
            c_puct: 1.0,
            rollout_depth: 200,
            dirichlet_alpha: 0.15,
            dirichlet_epsilon: 0.25,
            dirichlet_noise: false,
            temperature: 0.0,
        }
    }
}

impl MctsConfig {
    pub fn new(c_puct: f32, rollout_depth: usize) -> Self {
        MctsConfig { c_puct, rollout_depth, ..MctsConfig::default() }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_values() {
        let cfg = MctsConfig::default();
        assert!((cfg.c_puct - 1.0).abs() < 1e-6);
        assert_eq!(cfg.rollout_depth, 200);
        assert!((cfg.dirichlet_alpha - 0.15).abs() < 1e-6);
        assert!((cfg.dirichlet_epsilon - 0.25).abs() < 1e-6);
        assert!(!cfg.dirichlet_noise);
    }

    #[test]
    fn test_custom_values() {
        let cfg = MctsConfig::new(2.5, 100);
        assert!((cfg.c_puct - 2.5).abs() < 1e-6);
        assert_eq!(cfg.rollout_depth, 100);
        // new() only sets c_puct and rollout_depth; rest default.
        assert!(!cfg.dirichlet_noise);
    }

    #[test]
    fn test_dirichlet_fields_via_struct_literal() {
        let cfg = MctsConfig {
            dirichlet_alpha: 0.3,
            dirichlet_epsilon: 0.1,
            dirichlet_noise: true,
            ..MctsConfig::default()
        };
        assert!((cfg.dirichlet_alpha - 0.3).abs() < 1e-6);
        assert!((cfg.dirichlet_epsilon - 0.1).abs() < 1e-6);
        assert!(cfg.dirichlet_noise);
    }

    #[test]
    fn test_clone() {
        let cfg = MctsConfig::new(3.0, 50);
        let cfg2 = cfg.clone();
        assert!((cfg2.c_puct - 3.0).abs() < 1e-6);
        assert_eq!(cfg2.rollout_depth, 50);
    }
}
