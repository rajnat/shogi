/// Training step for AlphaZero-style self-play training.
///
/// `Trainer` owns the master network and optimizer and exposes one method per
/// training-step bullet:
///
///   ✓ sample_batch        — draw a random batch from the replay buffer  (bullet 1)
///   • forward             — run the network in training mode             (bullet 2)
///   • policy_loss         — cross-entropy vs MCTS visit distribution     (bullet 3)
///   • value_loss          — MSE vs game outcome                         (bullet 4)
///   • total_loss          — weighted sum + L2 weight decay               (bullet 5)
///   • step                — backward pass + optimizer update             (bullet 6)
///   • log                 — print losses every N steps                   (bullet 7)
use std::sync::{Arc, Mutex};

use rand::Rng;
use tch::{nn, nn::OptimizerConfig, Device, Tensor};

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
        Trainer { vs, net, opt, device, config, step: 0 }
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

    /// Returns the minimum buffer size required before training can start.
    pub fn min_buffer_size(&self) -> usize {
        self.config.min_buffer_size
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
        assert_eq!(policies.size(), vec![t.batch_size() as i64, NUM_ACTIONS as i64]);
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
        assert!(min >= -1.0 && max <= 1.0, "values out of range: [{min}, {max}]");
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
}
