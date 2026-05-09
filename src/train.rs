/// Training step for AlphaZero-style self-play training.
use std::sync::{Arc, Mutex};

use rand::Rng;
use tch::{Device, Tensor, nn, nn::OptimizerConfig};

use crate::nn::{Net, checkpoint::build_with_config};
use crate::replay_buffer::ReplayBuffer;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Hyper-parameters for one training run.
#[derive(Debug, Clone)]
pub struct TrainConfig {
    /// Mini-batch size drawn from the replay buffer each step.
    pub batch_size: usize,
    /// Adam learning rate.
    pub learning_rate: f64,
    /// L2 weight-decay coefficient (added to Adam's weight decay).
    pub weight_decay: f64,
    /// Minimum number of positions in the buffer before training starts.
    pub min_buffer_size: usize,
    /// Log a loss line every this many steps.
    pub log_every: u64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        TrainConfig {
            batch_size: 512,
            learning_rate: 1e-3,
            weight_decay: 1e-4,
            min_buffer_size: 10_000,
            log_every: 100,
        }
    }
}

// ---------------------------------------------------------------------------
// Trainer
// ---------------------------------------------------------------------------

/// Owns the master network, optimizer, and training step counter.
pub struct Trainer {
    pub vs: nn::VarStore,
    pub net: Net,
    opt: nn::Optimizer,
    pub device: Device,
    config: TrainConfig,
    pub step: u64,
}

impl Trainer {
    /// Create a `Trainer` with a freshly-built network on `device`.
    pub fn new(device: Device, channels: i64, blocks: usize, config: TrainConfig) -> Self {
        let (vs, net) = build_with_config(device, channels, blocks);
        let opt = nn::Adam {
            wd: config.weight_decay,
            ..nn::Adam::default()
        }
        .build(&vs, config.learning_rate)
        .expect("failed to build Adam optimizer");
        Trainer {
            vs,
            net,
            opt,
            device,
            config,
            step: 0,
        }
    }

    /// Draw one mini-batch from `buffer` and move it to the training device.
    ///
    /// Returns `(boards, policies, values)`:
    /// - `boards`:   `[B, 119, 9, 9]` f32
    /// - `policies`: `[B, NUM_ACTIONS]` f32
    /// - `values`:   `[B, 1]` f32
    ///
    /// Panics if the buffer has fewer than `config.min_buffer_size` entries.
    pub fn sample_batch<R: Rng>(
        &self,
        buffer: &Arc<Mutex<ReplayBuffer>>,
        rng: &mut R,
    ) -> (Tensor, Tensor, Tensor) {
        let buf = buffer.lock().unwrap();
        assert!(
            buf.len() >= self.config.min_buffer_size,
            "buffer too small to train ({} < {})",
            buf.len(),
            self.config.min_buffer_size,
        );
        buf.sample_batch(self.config.batch_size, self.device, rng)
    }

    /// Run `boards` through the network in training mode.
    ///
    /// Returns `(policy_logits, values)`:
    /// - `policy_logits`: `[B, NUM_ACTIONS]` — raw logits (no softmax).
    /// - `values`:        `[B, 1]`           — tanh output ∈ (−1, 1).
    ///
    /// BatchNorm running stats are updated and dropout (if any) is active.
    pub fn forward(&self, boards: &Tensor) -> (Tensor, Tensor) {
        self.net.forward_t(boards, true)
    }

    /// Returns the minimum buffer size required before training can start.
    pub fn min_buffer_size(&self) -> usize {
        self.config.min_buffer_size
    }

    /// Policy loss: cross-entropy between `policy_logits` and `policy_targets`.
    ///
    /// `policy_targets` is the MCTS visit distribution — a proper probability
    /// distribution (sums to 1) over all actions.  Because the target is soft
    /// (not a one-hot class index) we compute the loss manually:
    ///
    /// ```text
    /// L_policy = -mean( Σ_a  π(a) · log softmax(logits)(a) )
    /// ```
    ///
    /// `policy_logits`: `[B, NUM_ACTIONS]` — raw network output, no softmax.
    /// `policy_targets`: `[B, NUM_ACTIONS]` — MCTS visit distribution.
    ///
    /// Returns a scalar tensor.
    pub fn policy_loss(&self, policy_logits: &Tensor, policy_targets: &Tensor) -> Tensor {
        let log_probs = policy_logits.log_softmax(-1, tch::Kind::Float);
        -(log_probs * policy_targets)
            .sum_dim_intlist([-1].as_slice(), false, tch::Kind::Float)
            .mean(tch::Kind::Float)
    }

    /// Returns the configured batch size.
    pub fn batch_size(&self) -> usize {
        self.config.batch_size
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use tch::Tensor;

    use crate::nn::NUM_ACTIONS;
    use crate::replay_buffer::ReplayBuffer;

    fn small_config() -> TrainConfig {
        TrainConfig {
            batch_size: 4,
            min_buffer_size: 4,
            ..TrainConfig::default()
        }
    }

    fn trainer() -> Trainer {
        Trainer::new(Device::Cpu, 8, 2, small_config())
    }

    /// Push `n` dummy records into a new buffer of capacity 10_000.
    fn filled_buffer(n: usize) -> Arc<Mutex<ReplayBuffer>> {
        let buf = Arc::new(Mutex::new(ReplayBuffer::new(10_000)));
        {
            let mut b = buf.lock().unwrap();
            for i in 0..n {
                let board = Tensor::zeros([119, 9, 9], (tch::Kind::Float, Device::Cpu));
                let policy = vec![0.0f32; NUM_ACTIONS];
                let value = if i % 2 == 0 { 1.0 } else { -1.0 };
                b.push_game(vec![(board, policy, value)]);
            }
        }
        buf
    }

    #[test]
    fn test_sample_batch_board_shape() {
        let t = trainer();
        let buf = filled_buffer(t.min_buffer_size());
        let mut rng = StdRng::seed_from_u64(0);
        let (boards, _, _) = t.sample_batch(&buf, &mut rng);
        assert_eq!(boards.size(), vec![t.batch_size() as i64, 119, 9, 9]);
    }

    #[test]
    fn test_sample_batch_policy_shape() {
        let t = trainer();
        let buf = filled_buffer(t.min_buffer_size());
        let mut rng = StdRng::seed_from_u64(0);
        let (_, policies, _) = t.sample_batch(&buf, &mut rng);
        assert_eq!(
            policies.size(),
            vec![t.batch_size() as i64, NUM_ACTIONS as i64]
        );
    }

    #[test]
    fn test_sample_batch_value_shape() {
        let t = trainer();
        let buf = filled_buffer(t.min_buffer_size());
        let mut rng = StdRng::seed_from_u64(0);
        let (_, _, values) = t.sample_batch(&buf, &mut rng);
        assert_eq!(values.size(), vec![t.batch_size() as i64, 1]);
    }

    #[test]
    fn test_sample_batch_values_in_range() {
        let t = trainer();
        let buf = filled_buffer(t.min_buffer_size());
        let mut rng = StdRng::seed_from_u64(0);
        let (_, _, values) = t.sample_batch(&buf, &mut rng);
        let min = values.min().double_value(&[]);
        let max = values.max().double_value(&[]);
        assert!(
            min >= -1.0 && max <= 1.0,
            "values out of range: [{min}, {max}]"
        );
    }

    #[test]
    #[should_panic(expected = "buffer too small to train")]
    fn test_sample_batch_panics_when_buffer_too_small() {
        let t = trainer();
        let buf = filled_buffer(t.min_buffer_size() - 1);
        let mut rng = StdRng::seed_from_u64(0);
        let _ = t.sample_batch(&buf, &mut rng);
    }

    #[test]
    fn test_trainer_accessors() {
        let t = trainer();
        assert_eq!(t.batch_size(), 4);
        assert_eq!(t.min_buffer_size(), 4);
        assert_eq!(t.step, 0);
    }

    fn sample(t: &Trainer) -> (Tensor, Tensor, Tensor) {
        let buf = filled_buffer(t.min_buffer_size());
        let mut rng = StdRng::seed_from_u64(0);
        t.sample_batch(&buf, &mut rng)
    }

    #[test]
    fn test_forward_policy_shape() {
        let t = trainer();
        let (boards, _, _) = sample(&t);
        let (policy, _) = t.forward(&boards);
        assert_eq!(
            policy.size(),
            vec![t.batch_size() as i64, NUM_ACTIONS as i64]
        );
    }

    #[test]
    fn test_forward_value_shape() {
        let t = trainer();
        let (boards, _, _) = sample(&t);
        let (_, value) = t.forward(&boards);
        assert_eq!(value.size(), vec![t.batch_size() as i64, 1]);
    }

    #[test]
    fn test_forward_value_in_tanh_range() {
        let t = trainer();
        let (boards, _, _) = sample(&t);
        let (_, value) = t.forward(&boards);
        let min = value.min().double_value(&[]);
        let max = value.max().double_value(&[]);
        assert!(
            min > -1.0 && max < 1.0,
            "value outside (-1, 1): [{min}, {max}]"
        );
    }

    #[test]
    fn test_forward_policy_finite() {
        let t = trainer();
        let (boards, _, _) = sample(&t);
        let (policy, _) = t.forward(&boards);
        assert_eq!(policy.isnan().any().int64_value(&[]), 0, "policy has NaN");
        assert_eq!(policy.isinf().any().int64_value(&[]), 0, "policy has Inf");
    }

    // ----- policy_loss -----

    /// Uniform target distribution over all actions.
    fn uniform_policy(batch: i64) -> Tensor {
        let p = 1.0 / NUM_ACTIONS as f64;
        Tensor::full(
            [batch, NUM_ACTIONS as i64],
            p,
            (tch::Kind::Float, Device::Cpu),
        )
    }

    #[test]
    fn test_policy_loss_is_scalar() {
        let t = trainer();
        let (boards, policy_targets, _) = sample(&t);
        let (logits, _) = t.forward(&boards);
        let loss = t.policy_loss(&logits, &policy_targets);
        assert_eq!(
            loss.size(),
            Vec::<i64>::new(),
            "policy loss should be a scalar"
        );
    }

    #[test]
    fn test_policy_loss_non_negative() {
        let t = trainer();
        let (boards, policy_targets, _) = sample(&t);
        let (logits, _) = t.forward(&boards);
        let loss = t.policy_loss(&logits, &policy_targets);
        assert!(loss.double_value(&[]) >= 0.0, "cross-entropy must be ≥ 0");
    }

    #[test]
    fn test_policy_loss_uniform_target_equals_log_num_actions() {
        // H(uniform) = log(NUM_ACTIONS); the minimum achievable loss with
        // a uniform target is exactly log(N) when the predicted probs are also uniform.
        let t = trainer();
        let b = t.batch_size() as i64;
        let uniform_logits =
            Tensor::zeros([b, NUM_ACTIONS as i64], (tch::Kind::Float, Device::Cpu));
        let uniform_target = uniform_policy(b);
        let loss = t.policy_loss(&uniform_logits, &uniform_target);
        let expected = (NUM_ACTIONS as f64).ln();
        let diff = (loss.double_value(&[]) - expected).abs();
        assert!(
            diff < 1e-4,
            "H(uniform) mismatch: got {:.4}, expected {expected:.4}",
            loss.double_value(&[])
        );
    }

    #[test]
    fn test_policy_loss_lower_when_logits_match_target() {
        // Loss should be strictly lower when the logits favour the target action
        // compared to uniform logits.
        let t = trainer();
        let b = t.batch_size() as i64;
        // One-hot target on action 0.
        let mut target_data = vec![0.0f32; b as usize * NUM_ACTIONS];
        for i in 0..b as usize {
            target_data[i * NUM_ACTIONS] = 1.0;
        }
        let target = Tensor::from_slice(&target_data).reshape([b, NUM_ACTIONS as i64]);

        // Logits that strongly favour action 0.
        let mut logit_data = vec![0.0f32; b as usize * NUM_ACTIONS];
        for i in 0..b as usize {
            logit_data[i * NUM_ACTIONS] = 10.0;
        }
        let good_logits = Tensor::from_slice(&logit_data).reshape([b, NUM_ACTIONS as i64]);

        let uniform_logits =
            Tensor::zeros([b, NUM_ACTIONS as i64], (tch::Kind::Float, Device::Cpu));

        let loss_good = t.policy_loss(&good_logits, &target).double_value(&[]);
        let loss_uniform = t.policy_loss(&uniform_logits, &target).double_value(&[]);
        assert!(
            loss_good < loss_uniform,
            "matched logits ({loss_good:.4}) should have lower loss than uniform ({loss_uniform:.4})"
        );
    }

    #[test]
    fn test_policy_loss_near_zero_for_perfect_prediction() {
        // When logits perfectly match a one-hot target, loss ≈ 0.
        let t = trainer();
        let b = t.batch_size() as i64;
        let mut target_data = vec![0.0f32; b as usize * NUM_ACTIONS];
        for i in 0..b as usize {
            target_data[i * NUM_ACTIONS] = 1.0;
        }
        let target = Tensor::from_slice(&target_data).reshape([b, NUM_ACTIONS as i64]);

        // Very large logit on the target action → softmax ≈ 1 there, ≈ 0 elsewhere.
        let mut logit_data = vec![0.0f32; b as usize * NUM_ACTIONS];
        for i in 0..b as usize {
            logit_data[i * NUM_ACTIONS] = 100.0;
        }
        let logits = Tensor::from_slice(&logit_data).reshape([b, NUM_ACTIONS as i64]);

        let loss = t.policy_loss(&logits, &target).double_value(&[]);
        assert!(
            loss < 1e-3,
            "loss should be near 0 for perfect prediction, got {loss:.6}"
        );
    }
}
