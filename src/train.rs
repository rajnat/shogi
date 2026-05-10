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

#[derive(Debug, Clone, Copy)]
pub struct TrainStepMetrics {
    pub step: u64,
    pub total_loss: f64,
    pub policy_loss: f64,
    pub value_loss: f64,
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
    pub channels: i64,
    pub blocks: usize,
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
            channels,
            blocks,
        }
    }

    /// Resume training from a saved checkpoint.
    ///
    /// Loads weights from `path` into the VarStore and, if the filename follows the
    /// `step_XXXXXXXX.ot` convention produced by `checkpoint_path`, restores
    /// `self.step` so logging and checkpoint intervals remain correct.
    pub fn resume(&mut self, path: &std::path::Path) {
        self.vs.load(path).unwrap_or_else(|e| panic!("failed to load checkpoint {path:?}: {e}"));
        if let Some(step) = crate::orchestrate::parse_step_from_filename(path) {
            self.step = step;
            println!("Resumed from {path:?} at step {step}.");
        } else {
            println!("Loaded weights from {path:?} (step counter not restored).");
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

    /// Value loss: MSE between `pred_values` and `value_targets`.
    ///
    /// ```text
    /// L_value = mean( (v_pred − z)² )
    /// ```
    ///
    /// `pred_values`:   `[B, 1]` — network tanh output ∈ (−1, 1).
    /// `value_targets`: `[B, 1]` — game outcome z ∈ {−1, 0, +1}.
    ///
    /// Returns a scalar tensor.
    pub fn value_loss(&self, pred_values: &Tensor, value_targets: &Tensor) -> Tensor {
        (pred_values - value_targets)
            .pow_tensor_scalar(2)
            .mean(tch::Kind::Float)
    }

    /// Total loss: policy cross-entropy + value MSE.
    ///
    /// ```text
    /// L = L_policy + L_value
    /// ```
    ///
    /// L2 weight decay is applied by the Adam optimizer (via `TrainConfig::weight_decay`)
    /// and does not appear as an explicit term in the returned tensor.
    ///
    /// Returns `(total, policy_component, value_component)` so the caller can
    /// log each term independently without recomputing them.
    pub fn total_loss(
        &self,
        policy_logits: &Tensor,
        policy_targets: &Tensor,
        pred_values: &Tensor,
        value_targets: &Tensor,
    ) -> (Tensor, Tensor, Tensor) {
        let lp = self.policy_loss(policy_logits, policy_targets);
        let lv = self.value_loss(pred_values, value_targets);
        let total = &lp + &lv;
        (total, lp, lv)
    }

    /// Run one complete training step: sample → forward → loss → backward → optimizer.
    ///
    /// `opt.backward_step` zeros gradients, runs backprop, then applies the Adam
    /// update in a single call — the idiomatic tch-rs pattern.
    ///
    /// Prints a loss line to stdout when `step % log_every == 0`.
    ///
    /// Returns loss values as `f64` scalars so the caller can log them without
    /// holding a live tensor.  `self.step` is incremented after each call.
    pub fn train_step<R: Rng>(
        &mut self,
        buffer: &Arc<Mutex<ReplayBuffer>>,
        rng: &mut R,
    ) -> TrainStepMetrics {
        let (boards, policy_targets, value_targets) = self.sample_batch(buffer, rng);
        let (policy_logits, pred_values) = self.forward(&boards);
        let (total, lp, lv) = self.total_loss(
            &policy_logits,
            &policy_targets,
            &pred_values,
            &value_targets,
        );
        self.opt.backward_step(&total);
        self.step += 1;
        let metrics = TrainStepMetrics {
            step: self.step,
            total_loss: total.double_value(&[]),
            policy_loss: lp.double_value(&[]),
            value_loss: lv.double_value(&[]),
        };
        if self.step % self.config.log_every == 0 {
            println!(
                "{}",
                Self::format_log_line(
                    metrics.step,
                    metrics.total_loss,
                    metrics.policy_loss,
                    metrics.value_loss,
                )
            );
        }
        metrics
    }

    /// Format a single log line for a training step.
    ///
    /// Extracted so the format can be tested without running a full training step
    /// or capturing stdout.
    pub fn format_log_line(step: u64, total: f64, policy: f64, value: f64) -> String {
        format!("step {step:6}  total={total:.4}  policy={policy:.4}  value={value:.4}")
    }

    /// Returns `true` if losses should be logged after `step`.
    pub fn should_log(&self, step: u64) -> bool {
        step % self.config.log_every == 0
    }

    /// Returns the configured logging interval.
    pub fn log_every(&self) -> u64 {
        self.config.log_every
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

    // ----- value_loss -----

    fn scalar(v: f32) -> Tensor {
        Tensor::from_slice(&[v]).reshape([1, 1])
    }

    fn batch_values(vals: &[f32]) -> Tensor {
        let n = vals.len() as i64;
        Tensor::from_slice(vals).reshape([n, 1])
    }

    #[test]
    fn test_value_loss_is_scalar() {
        let t = trainer();
        let pred = batch_values(&[0.5, -0.3, 0.1, 0.9]);
        let target = batch_values(&[1.0, -1.0, 0.0, 1.0]);
        let loss = t.value_loss(&pred, &target);
        assert_eq!(
            loss.size(),
            Vec::<i64>::new(),
            "value loss should be a scalar"
        );
    }

    #[test]
    fn test_value_loss_non_negative() {
        let t = trainer();
        let pred = batch_values(&[0.5, -0.5, 0.0, 0.8]);
        let target = batch_values(&[1.0, -1.0, 0.0, 1.0]);
        let loss = t.value_loss(&pred, &target).double_value(&[]);
        assert!(loss >= 0.0, "MSE must be ≥ 0, got {loss}");
    }

    #[test]
    fn test_value_loss_zero_for_perfect_prediction() {
        let t = trainer();
        let vals = batch_values(&[1.0, -1.0, 0.0, 1.0]);
        let loss = t.value_loss(&vals, &vals).double_value(&[]);
        assert!(loss < 1e-6, "MSE(x, x) must be 0, got {loss:.2e}");
    }

    #[test]
    fn test_value_loss_matches_manual_mse() {
        let t = trainer();
        let pred = &[0.3f32, -0.7, 0.5, -0.2];
        let target = &[1.0f32, -1.0, 0.0, 1.0];
        let manual: f64 = pred
            .iter()
            .zip(target.iter())
            .map(|(p, z)| ((p - z) as f64).powi(2))
            .sum::<f64>()
            / 4.0;
        let loss = t
            .value_loss(&batch_values(pred), &batch_values(target))
            .double_value(&[]);
        assert!(
            (loss - manual).abs() < 1e-5,
            "MSE mismatch: got {loss:.6}, expected {manual:.6}"
        );
    }

    #[test]
    fn test_value_loss_increases_with_error() {
        let t = trainer();
        let target = scalar(1.0);
        let close = scalar(0.9); // error = 0.1
        let far = scalar(-0.9); // error = 1.9
        let loss_close = t.value_loss(&close, &target).double_value(&[]);
        let loss_far = t.value_loss(&far, &target).double_value(&[]);
        assert!(
            loss_far > loss_close,
            "larger error should give larger loss: {loss_far:.4} vs {loss_close:.4}"
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

    // ----- total_loss -----

    fn dummy_logits_and_targets(b: i64) -> (Tensor, Tensor, Tensor, Tensor) {
        let logits = Tensor::randn([b, NUM_ACTIONS as i64], (tch::Kind::Float, Device::Cpu));
        let target_p = uniform_policy(b);
        let pred_v = batch_values(&vec![0.5f32; b as usize]);
        let target_v = batch_values(&vec![1.0f32; b as usize]);
        (logits, target_p, pred_v, target_v)
    }

    #[test]
    fn test_total_loss_is_scalar() {
        let t = trainer();
        let (logits, tp, pv, tv) = dummy_logits_and_targets(t.batch_size() as i64);
        let (total, _, _) = t.total_loss(&logits, &tp, &pv, &tv);
        assert_eq!(
            total.size(),
            Vec::<i64>::new(),
            "total loss should be a scalar"
        );
    }

    #[test]
    fn test_total_loss_equals_sum_of_components() {
        let t = trainer();
        let (logits, tp, pv, tv) = dummy_logits_and_targets(t.batch_size() as i64);
        let (total, lp, lv) = t.total_loss(&logits, &tp, &pv, &tv);
        let expected = lp.double_value(&[]) + lv.double_value(&[]);
        let got = total.double_value(&[]);
        assert!(
            (got - expected).abs() < 1e-6,
            "total ({got:.6}) != lp + lv ({expected:.6})"
        );
    }

    #[test]
    fn test_total_loss_non_negative() {
        let t = trainer();
        let (logits, tp, pv, tv) = dummy_logits_and_targets(t.batch_size() as i64);
        let (total, _, _) = t.total_loss(&logits, &tp, &pv, &tv);
        assert!(total.double_value(&[]) >= 0.0, "total loss must be ≥ 0");
    }

    #[test]
    fn test_total_loss_components_independently_accessible() {
        // Verify lp and lv match what the individual methods return.
        let t = trainer();
        let b = t.batch_size() as i64;
        let (logits, tp, pv, tv) = dummy_logits_and_targets(b);
        let (_, lp, lv) = t.total_loss(&logits, &tp, &pv, &tv);
        let lp_direct = t.policy_loss(&logits, &tp).double_value(&[]);
        let lv_direct = t.value_loss(&pv, &tv).double_value(&[]);
        assert!(
            (lp.double_value(&[]) - lp_direct).abs() < 1e-6,
            "policy component mismatch"
        );
        assert!(
            (lv.double_value(&[]) - lv_direct).abs() < 1e-6,
            "value component mismatch"
        );
    }

    #[test]
    fn test_total_loss_dominated_by_larger_component() {
        // When value loss is zero, total should equal policy loss.
        let t = trainer();
        let b = t.batch_size() as i64;
        let logits = Tensor::randn([b, NUM_ACTIONS as i64], (tch::Kind::Float, Device::Cpu));
        let tp = uniform_policy(b);
        let perfect_v = batch_values(&vec![0.5f32; b as usize]);
        let (total, lp, _) = t.total_loss(&logits, &tp, &perfect_v, &perfect_v);
        let diff = (total.double_value(&[]) - lp.double_value(&[])).abs();
        assert!(
            diff < 1e-6,
            "when value loss = 0, total should equal policy loss"
        );
    }

    // ----- train_step -----

    fn trainer_mut() -> Trainer {
        Trainer::new(Device::Cpu, 8, 2, small_config())
    }

    #[test]
    fn test_train_step_returns_structured_finite_metrics() {
        let mut t = trainer_mut();
        let buf = filled_buffer(t.min_buffer_size());
        let mut rng = StdRng::seed_from_u64(0);
        let metrics = t.train_step(&buf, &mut rng);
        assert_eq!(metrics.step, 1);
        assert!(
            metrics.total_loss.is_finite(),
            "total loss is not finite: {}",
            metrics.total_loss
        );
        assert!(
            metrics.policy_loss.is_finite(),
            "policy loss is not finite: {}",
            metrics.policy_loss
        );
        assert!(
            metrics.value_loss.is_finite(),
            "value loss is not finite: {}",
            metrics.value_loss
        );
    }

    #[test]
    fn test_train_step_increments_counter() {
        let mut t = trainer_mut();
        let buf = filled_buffer(t.min_buffer_size());
        let mut rng = StdRng::seed_from_u64(0);
        assert_eq!(t.step, 0);
        t.train_step(&buf, &mut rng);
        assert_eq!(t.step, 1);
        t.train_step(&buf, &mut rng);
        assert_eq!(t.step, 2);
    }

    #[test]
    fn test_train_step_losses_non_negative() {
        let mut t = trainer_mut();
        let buf = filled_buffer(t.min_buffer_size());
        let mut rng = StdRng::seed_from_u64(0);
        let metrics = t.train_step(&buf, &mut rng);
        assert!(
            metrics.total_loss >= 0.0,
            "total loss < 0: {}",
            metrics.total_loss
        );
        assert!(
            metrics.policy_loss >= 0.0,
            "policy loss < 0: {}",
            metrics.policy_loss
        );
        assert!(
            metrics.value_loss >= 0.0,
            "value loss < 0: {}",
            metrics.value_loss
        );
    }

    #[test]
    fn test_train_step_loss_decreases_over_iterations() {
        // Run many steps on a fixed buffer and verify the network is learning.
        // All value targets are +1.0: the optimal prediction is +1 everywhere,
        // so MSE starts near 1.0 (random init ≈ 0) and should decrease as the
        // value head learns to output +1 for every position.
        let mut t = trainer_mut();
        let buf = {
            let buf = Arc::new(Mutex::new(ReplayBuffer::new(10_000)));
            let mut b = buf.lock().unwrap();
            for _ in 0..32 {
                let board = Tensor::zeros([119, 9, 9], (tch::Kind::Float, Device::Cpu));
                let policy = vec![1.0 / NUM_ACTIONS as f32; NUM_ACTIONS];
                b.push_game(vec![(board, policy, 1.0f32)]);
            }
            drop(b);
            buf
        };
        let mut rng = StdRng::seed_from_u64(42);

        let first_loss = t.train_step(&buf, &mut rng).total_loss;
        for _ in 1..199 {
            t.train_step(&buf, &mut rng);
        }
        let last_loss = t.train_step(&buf, &mut rng).total_loss;

        assert!(
            last_loss < first_loss,
            "loss did not decrease after 200 steps ({first_loss:.4} → {last_loss:.4}): \
             gradients may not be flowing"
        );
    }

    // ----- log -----

    #[test]
    fn test_log_every_accessor() {
        let t = trainer();
        assert_eq!(t.log_every(), small_config().log_every);
    }

    #[test]
    fn test_should_log_fires_at_multiples() {
        let t = trainer(); // log_every = small_config().log_every (default 100)
        let every = t.log_every();
        assert!(t.should_log(every), "should log at step {every}");
        assert!(t.should_log(2 * every), "should log at step {}", 2 * every);
        assert!(
            !t.should_log(every - 1),
            "should not log at step {}",
            every - 1
        );
        assert!(
            !t.should_log(every + 1),
            "should not log at step {}",
            every + 1
        );
    }

    #[test]
    fn test_should_log_at_zero() {
        let t = trainer();
        // Step 0 satisfies 0 % N == 0 — fires on initialisation before any step.
        assert!(t.should_log(0));
    }

    #[test]
    fn test_format_log_line_contains_all_fields() {
        let line = Trainer::format_log_line(100, 8.1234, 7.0001, 1.1233);
        assert!(line.contains("100"), "missing step:   {line}");
        assert!(line.contains("8.1234"), "missing total:  {line}");
        assert!(line.contains("7.0001"), "missing policy: {line}");
        assert!(line.contains("1.1233"), "missing value:  {line}");
    }

    #[test]
    fn test_format_log_line_field_labels() {
        let line = Trainer::format_log_line(200, 1.0, 0.5, 0.5);
        assert!(line.contains("step"), "missing 'step' label");
        assert!(line.contains("total"), "missing 'total' label");
        assert!(line.contains("policy"), "missing 'policy' label");
        assert!(line.contains("value"), "missing 'value' label");
    }

    #[test]
    fn test_train_step_logs_at_configured_interval() {
        // Use log_every=3 so the test runs fast.  Run exactly 3 steps and verify
        // the step counter reaches 3 (logging would have fired on step 3).
        // We don't capture stdout — testing the side-effect format is enough.
        let mut t = Trainer::new(
            Device::Cpu,
            8,
            2,
            TrainConfig {
                batch_size: 4,
                min_buffer_size: 4,
                log_every: 3,
                ..TrainConfig::default()
            },
        );
        let buf = filled_buffer(4);
        let mut rng = StdRng::seed_from_u64(0);
        for _ in 0..3 {
            t.train_step(&buf, &mut rng);
        }
        assert_eq!(t.step, 3);
        assert!(t.should_log(t.step), "step {} should trigger a log", t.step);
    }

    // ----- resume -----

    #[test]
    fn test_resume_restores_step_from_filename() {
        use crate::orchestrate::checkpoint_path;

        let dir = tempfile::tempdir().unwrap();
        let path = checkpoint_path(dir.path().to_str().unwrap(), 7500);

        // Save a checkpoint at step 7500.
        let t_save = Trainer::new(Device::Cpu, 8, 2, small_config());
        t_save.vs.save(&path).unwrap();

        // Load into a fresh trainer — step should jump to 7500.
        let mut t_load = Trainer::new(Device::Cpu, 8, 2, small_config());
        assert_eq!(t_load.step, 0);
        t_load.resume(&path);
        assert_eq!(t_load.step, 7500, "step should be restored from filename");
    }

    #[test]
    fn test_resume_loads_weights() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("weights.ot");

        // Train a few steps so weights differ from init.
        let buf = filled_buffer(4);
        let mut rng = StdRng::seed_from_u64(0);
        let mut t_trained = Trainer::new(Device::Cpu, 8, 2, small_config());
        for _ in 0..5 {
            t_trained.train_step(&buf, &mut rng);
        }
        t_trained.vs.save(&path).unwrap();

        // Load into a fresh trainer and verify weights match.
        let mut t_loaded = Trainer::new(Device::Cpu, 8, 2, small_config());
        t_loaded.resume(&path);

        let vars_trained = t_trained.vs.variables();
        let vars_loaded = t_loaded.vs.variables();
        for (name, tensor) in &vars_trained {
            let loaded = vars_loaded.get(name).expect("variable missing after resume");
            let max_diff = (tensor - loaded).abs().max().double_value(&[]);
            assert!(
                max_diff < 1e-6,
                "variable {name} differs after resume (max diff {max_diff})"
            );
        }
    }
}
