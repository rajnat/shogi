/// Tunable parameters for MCTS search.
///
/// Passed to `mcts_search`; fields will grow as M4-03 progresses
/// (Dirichlet noise, temperature).
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
}

impl Default for MctsConfig {
    fn default() -> Self {
        MctsConfig {
            c_puct: 1.0,
            rollout_depth: 200,
        }
    }
}

impl MctsConfig {
    pub fn new(c_puct: f32, rollout_depth: usize) -> Self {
        MctsConfig { c_puct, rollout_depth }
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
    }

    #[test]
    fn test_custom_values() {
        let cfg = MctsConfig::new(2.5, 100);
        assert!((cfg.c_puct - 2.5).abs() < 1e-6);
        assert_eq!(cfg.rollout_depth, 100);
    }

    #[test]
    fn test_clone() {
        let cfg = MctsConfig::new(3.0, 50);
        let cfg2 = cfg.clone();
        assert!((cfg2.c_puct - 3.0).abs() < 1e-6);
        assert_eq!(cfg2.rollout_depth, 50);
    }
}
