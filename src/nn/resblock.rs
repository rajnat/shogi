/// Residual block for the AlphaZero-style policy+value network.
///
/// Architecture (AlphaZero convention):
/// ```text
///   input
///    │
///    ├──────────────────────────── skip ──────────────────────────────┐
///    │                                                                 │
///   Conv2d(channels→channels, 3×3, pad=1, no bias)                   │
///   BatchNorm2d                                                        │
///   ReLU                                                               │
///   Conv2d(channels→channels, 3×3, pad=1, no bias)                   │
///   BatchNorm2d                                                        │
///    │                                                                 │
///    └────────────────────── Add ─────────────────────────────────────┘
///    │
///   ReLU
///    │
///   output  [batch, channels, 9, 9]
/// ```
///
/// Padding = 1 keeps the 9×9 spatial size unchanged throughout.
/// Bias is disabled on the convolutions because BatchNorm already shifts the mean.
use tch::{nn, Tensor};

#[derive(Debug)]
pub struct ResBlock {
    conv1: nn::Conv2D,
    bn1:   nn::BatchNorm,
    conv2: nn::Conv2D,
    bn2:   nn::BatchNorm,
}

impl ResBlock {
    /// Create a residual block with `channels` filters, registered under `vs`.
    pub fn new(vs: &nn::Path, channels: i64) -> Self {
        let conv_cfg = nn::ConvConfig {
            padding: 1,
            bias: false,
            ..Default::default()
        };
        ResBlock {
            conv1: nn::conv2d(vs / "conv1", channels, channels, 3, conv_cfg),
            bn1:   nn::batch_norm2d(vs / "bn1", channels, Default::default()),
            conv2: nn::conv2d(vs / "conv2", channels, channels, 3, conv_cfg),
            bn2:   nn::batch_norm2d(vs / "bn2", channels, Default::default()),
        }
    }
}

impl nn::ModuleT for ResBlock {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let residual = xs.shallow_clone();
        let ys = xs
            .apply(&self.conv1)
            .apply_t(&self.bn1, train)
            .relu()
            .apply(&self.conv2)
            .apply_t(&self.bn2, train);
        (residual + ys).relu()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{Device, Kind, nn::ModuleT};

    fn make_block(channels: i64) -> (nn::VarStore, ResBlock) {
        let vs = nn::VarStore::new(Device::Cpu);
        let block = ResBlock::new(&vs.root(), channels);
        (vs, block)
    }

    fn randn(shape: &[i64]) -> Tensor {
        Tensor::randn(shape, (Kind::Float, Device::Cpu))
    }

    // ----- Shape (the primary spec requirement) -----

    #[test]
    fn test_output_shape_batch1() {
        let (_vs, block) = make_block(8);
        let ys = block.forward_t(&randn(&[1, 8, 9, 9]), false);
        assert_eq!(ys.size(), vec![1, 8, 9, 9]);
    }

    #[test]
    fn test_output_shape_batch4() {
        let (_vs, block) = make_block(8);
        let ys = block.forward_t(&randn(&[4, 8, 9, 9]), false);
        assert_eq!(ys.size(), vec![4, 8, 9, 9]);
    }

    #[test]
    fn test_standard_256_channels_shape() {
        // Spec requires [batch, 256, 9, 9] → [batch, 256, 9, 9].
        let (_vs, block) = make_block(256);
        let ys = block.forward_t(&randn(&[1, 256, 9, 9]), false);
        assert_eq!(ys.size(), vec![1, 256, 9, 9]);
    }

    // ----- Numerical properties -----

    #[test]
    fn test_output_is_finite() {
        let (_vs, block) = make_block(8);
        let ys = block.forward_t(&randn(&[2, 8, 9, 9]), false);
        // isnan/isinf as scalar count checks
        let has_nan  = ys.isnan().any().int64_value(&[]) != 0;
        let has_inf  = ys.isinf().any().int64_value(&[]) != 0;
        assert!(!has_nan, "output contains NaN");
        assert!(!has_inf, "output contains Inf");
    }

    #[test]
    fn test_output_nonnegative() {
        // The final ReLU clamps all outputs to ≥ 0.
        let (_vs, block) = make_block(8);
        let ys = block.forward_t(&randn(&[2, 8, 9, 9]), false);
        let min_val = ys.min().double_value(&[]);
        assert!(min_val >= 0.0, "output has negative values ({min_val}); final ReLU should clamp them");
    }

    // ----- Train / eval mode -----

    #[test]
    fn test_train_and_eval_same_shape() {
        let (_vs, block) = make_block(8);
        let xs = randn(&[2, 8, 9, 9]);
        assert_eq!(
            block.forward_t(&xs, true).size(),
            block.forward_t(&xs, false).size(),
        );
    }

    // ----- Skip connection -----

    #[test]
    fn test_skip_connection_active() {
        // Zero-initialise both conv weights so conv(x) = 0 for any x.
        // Then the block output is ReLU(x + BN(0)) ≈ ReLU(x) ≠ 0 for positive x,
        // which confirms the skip path is carrying signal.
        let vs = nn::VarStore::new(Device::Cpu);
        let block = ResBlock::new(&vs.root(), 4);
        tch::no_grad(|| {
            for (_, mut v) in vs.variables() {
                let _ = v.zero_();
            }
        });
        // All-positive input → skip ensures output is also non-zero.
        let xs = Tensor::ones([1, 4, 9, 9], (Kind::Float, Device::Cpu));
        let ys = block.forward_t(&xs, false);
        let max_val = ys.max().double_value(&[]);
        assert!(max_val > 0.0, "skip connection should keep signal alive (max={max_val})");
    }
}
