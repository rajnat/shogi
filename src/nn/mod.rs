/// Neural network: policy + value head for AlphaZero-style MCTS.
///
/// M5-01: tch-rs setup and device detection.
/// Subsequent submodules add the encoder, ResBlock, and full network.
pub use tch::Device;

/// Return the device to use for all tensor ops.
///
/// Prefers CUDA if available, otherwise falls back to CPU.  On Apple Silicon
/// the MPS backend is not exposed through tch-rs, so CPU is the correct
/// fallback here.
pub fn device() -> Device {
    Device::cuda_if_available()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Tensor;

    #[test]
    fn test_device_is_valid() {
        let d = device();
        // Must be either Cpu or a Cuda device — no panic means linkage is good.
        let _ = d;
    }

    #[test]
    fn test_tensor_add() {
        // Basic sanity check that LibTorch is linked and ops work.
        let a = Tensor::from_slice(&[1.0_f32, 2.0, 3.0]);
        let b = Tensor::from_slice(&[4.0_f32, 5.0, 6.0]);
        let c = a + b;
        let vals: Vec<f32> = c.into();
        assert_eq!(vals, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_tensor_matmul_shape() {
        // 2×3 @ 3×4 → 2×4
        let a = Tensor::randn([2, 3], (tch::Kind::Float, device()));
        let b = Tensor::randn([3, 4], (tch::Kind::Float, device()));
        let c = a.matmul(&b);
        assert_eq!(c.size(), vec![2, 4]);
    }

    #[test]
    fn test_tensor_relu() {
        let t = Tensor::from_slice(&[-1.0_f32, 0.0, 1.0, 2.0]);
        let r = t.relu();
        let vals: Vec<f32> = r.into();
        assert_eq!(vals, vec![0.0, 0.0, 1.0, 2.0]);
    }
}
