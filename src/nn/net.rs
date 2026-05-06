/// AlphaZero-style policy + value network for Shogi.
///
/// Architecture:
/// ```text
///   input [batch, 119, 9, 9]
///    │
///   Conv2d(119→channels, 3×3, pad=1) → BN → ReLU          ← stem
///    │
///   ResBlock × num_blocks                                   ← tower
///    │
///    ├─── Conv2d(channels→2, 1×1) → BN → ReLU              ← policy head
///    │    Flatten → Linear(162 → NUM_ACTIONS)
///    │    → [batch, NUM_ACTIONS]  (raw logits)
///    │
///    └─── Conv2d(channels→1, 1×1) → BN → ReLU              ← value head
///         Flatten → Linear(81→channels) → ReLU
///         → Linear(channels→1) → tanh
///         → [batch, 1]  (∈ (-1, 1))
/// ```
use tch::{nn, nn::ModuleT, Tensor};
use super::{ResBlock, NUM_PLANES};
use super::move_index::NUM_ACTIONS;

#[derive(Debug)]
pub struct Net {
    // Stem
    stem_conv: nn::Conv2D,
    stem_bn:   nn::BatchNorm,
    // Residual tower
    tower:     Vec<ResBlock>,
    // Policy head
    ph_conv:   nn::Conv2D,
    ph_bn:     nn::BatchNorm,
    ph_fc:     nn::Linear,
    // Value head
    vh_conv:   nn::Conv2D,
    vh_bn:     nn::BatchNorm,
    vh_fc1:    nn::Linear,
    vh_fc2:    nn::Linear,
}

impl Net {
    /// Standard AlphaZero-Shogi config: 256 channels, 20 residual blocks.
    pub fn new(vs: &nn::Path) -> Self {
        Self::with_config(vs, 256, 20)
    }

    /// Configurable constructor — use for tests and ablations.
    pub fn with_config(vs: &nn::Path, channels: i64, num_blocks: usize) -> Self {
        let conv3 = nn::ConvConfig { padding: 1, bias: false, ..Default::default() };
        let conv1 = nn::ConvConfig { bias: false, ..Default::default() };

        let stem_conv = nn::conv2d(vs / "stem_conv", NUM_PLANES as i64, channels, 3, conv3);
        let stem_bn   = nn::batch_norm2d(vs / "stem_bn", channels, Default::default());

        let tower_path = vs / "tower";
        let tower = (0..num_blocks)
            .map(|i| {
                let name = i.to_string();
                let block_vs = &tower_path / name.as_str();
                ResBlock::new(&block_vs, channels)
            })
            .collect();

        let ph_conv = nn::conv2d(vs / "ph_conv", channels, 2, 1, conv1);
        let ph_bn   = nn::batch_norm2d(vs / "ph_bn", 2, Default::default());
        let ph_fc   = nn::linear(vs / "ph_fc", 2 * 9 * 9, NUM_ACTIONS as i64, Default::default());

        let vh_conv = nn::conv2d(vs / "vh_conv", channels, 1, 1, conv1);
        let vh_bn   = nn::batch_norm2d(vs / "vh_bn", 1, Default::default());
        let vh_fc1  = nn::linear(vs / "vh_fc1", 9 * 9, channels, Default::default());
        let vh_fc2  = nn::linear(vs / "vh_fc2", channels, 1, Default::default());

        Net { stem_conv, stem_bn, tower, ph_conv, ph_bn, ph_fc, vh_conv, vh_bn, vh_fc1, vh_fc2 }
    }

    /// Forward pass under training or inference mode.
    ///
    /// Returns `(policy_logits, value)`:
    /// - `policy_logits`: `[batch, NUM_ACTIONS]` — raw logits, no softmax applied.
    /// - `value`:         `[batch, 1]`           — tanh output, ∈ (-1, 1).
    pub fn forward_t(&self, xs: &Tensor, train: bool) -> (Tensor, Tensor) {
        // Stem
        let x = xs
            .apply(&self.stem_conv)
            .apply_t(&self.stem_bn, train)
            .relu();

        // Tower
        let x = self.tower.iter().fold(x, |acc, block| block.forward_t(&acc, train));

        // Policy head
        let policy = x
            .apply(&self.ph_conv)
            .apply_t(&self.ph_bn, train)
            .relu()
            .flatten(1, -1)
            .apply(&self.ph_fc);

        // Value head
        let value = x
            .apply(&self.vh_conv)
            .apply_t(&self.vh_bn, train)
            .relu()
            .flatten(1, -1)
            .apply(&self.vh_fc1)
            .relu()
            .apply(&self.vh_fc2)
            .tanh();

        (policy, value)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{Device, Kind};

    /// Small network (8 channels, 2 blocks) for fast shape tests.
    fn make_net() -> (nn::VarStore, Net) {
        let vs = nn::VarStore::new(Device::Cpu);
        let net = Net::with_config(&vs.root(), 8, 2);
        (vs, net)
    }

    fn randn_input(batch: i64) -> Tensor {
        Tensor::randn([batch, NUM_PLANES as i64, 9, 9], (Kind::Float, Device::Cpu))
    }

    // ----- Output shapes -----

    #[test]
    fn test_policy_shape_batch1() {
        let (_vs, net) = make_net();
        let (policy, _) = net.forward_t(&randn_input(1), false);
        assert_eq!(policy.size(), vec![1, NUM_ACTIONS as i64]);
    }

    #[test]
    fn test_value_shape_batch1() {
        let (_vs, net) = make_net();
        let (_, value) = net.forward_t(&randn_input(1), false);
        assert_eq!(value.size(), vec![1, 1]);
    }

    #[test]
    fn test_policy_shape_batch4() {
        let (_vs, net) = make_net();
        let (policy, _) = net.forward_t(&randn_input(4), false);
        assert_eq!(policy.size(), vec![4, NUM_ACTIONS as i64]);
    }

    #[test]
    fn test_value_shape_batch4() {
        let (_vs, net) = make_net();
        let (_, value) = net.forward_t(&randn_input(4), false);
        assert_eq!(value.size(), vec![4, 1]);
    }

    // ----- Numerical properties -----

    #[test]
    fn test_value_tanh_range() {
        let (_vs, net) = make_net();
        let (_, value) = net.forward_t(&randn_input(4), false);
        let min = value.min().double_value(&[]);
        let max = value.max().double_value(&[]);
        assert!(min > -1.0, "value below -1 ({min})");
        assert!(max < 1.0, "value above +1 ({max})");
    }

    #[test]
    fn test_policy_finite() {
        let (_vs, net) = make_net();
        let (policy, _) = net.forward_t(&randn_input(2), false);
        assert!(!policy.isnan().any().int64_value(&[]) != 0, "policy has NaN");
        assert!(!policy.isinf().any().int64_value(&[]) != 0, "policy has Inf");
    }

    #[test]
    fn test_value_finite() {
        let (_vs, net) = make_net();
        let (_, value) = net.forward_t(&randn_input(2), false);
        assert!(!value.isnan().any().int64_value(&[]) != 0, "value has NaN");
        assert!(!value.isinf().any().int64_value(&[]) != 0, "value has Inf");
    }

    // ----- Train / eval mode -----

    #[test]
    fn test_train_eval_same_shapes() {
        let (_vs, net) = make_net();
        let xs = randn_input(2);
        let (p_train, v_train) = net.forward_t(&xs, true);
        let (p_eval,  v_eval)  = net.forward_t(&xs, false);
        assert_eq!(p_train.size(), p_eval.size());
        assert_eq!(v_train.size(), v_eval.size());
    }
}
