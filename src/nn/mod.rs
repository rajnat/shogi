/// Neural network: policy + value head for AlphaZero-style MCTS.
pub use tch::Device;
pub mod encoder;
pub use encoder::{encode, NUM_PLANES, HAND_MAX};
pub mod resblock;
pub use resblock::ResBlock;
pub mod move_index;
pub use move_index::{move_to_index, index_to_move, index_to_move_on_board, NUM_ACTIONS};
pub mod net;
pub use net::Net;

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

    fn copy_to_vec(t: &Tensor) -> Vec<f32> {
        let n = t.numel();
        let mut out = vec![0.0f32; n];
        t.copy_data(&mut out, n);
        out
    }

    #[test]
    fn test_device_is_valid() {
        let d = device();
        let _ = d;
    }

    #[test]
    fn test_tensor_add() {
        let a = Tensor::from_slice(&[1.0_f32, 2.0, 3.0]);
        let b = Tensor::from_slice(&[4.0_f32, 5.0, 6.0]);
        let c = a + b;
        assert_eq!(copy_to_vec(&c), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_tensor_matmul_shape() {
        let a = Tensor::randn([2, 3], (tch::Kind::Float, device()));
        let b = Tensor::randn([3, 4], (tch::Kind::Float, device()));
        let c = a.matmul(&b);
        assert_eq!(c.size(), vec![2, 4]);
    }

    #[test]
    fn test_tensor_relu() {
        let t = Tensor::from_slice(&[-1.0_f32, 0.0, 1.0, 2.0]);
        let r = t.relu();
        assert_eq!(copy_to_vec(&r), vec![0.0, 0.0, 1.0, 2.0]);
    }
}
