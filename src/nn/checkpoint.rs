/// Model checkpoint — save and load `Net` weights.
///
/// # File format
///
/// Checkpoint files **must use the `.safetensors` extension**.
///
/// `tch`'s `VarStore` dispatches on the file extension at both save and load
/// time.  With `.pt` / `.bin`, the save path calls
/// `torch::jit::_save_parameters` while the load path calls
/// `_load_parameters_bytes`; in LibTorch 2.x these formats diverged (save
/// emits a JIT *Object*, load expects a *GenericDict*), causing a hard error.
/// The `.safetensors` path uses an entirely separate, stable code path that is
/// not affected by that mismatch.
///
/// # Typical usage
///
/// ```rust,ignore
/// use shogi_core::nn::{checkpoint, device};
///
/// // Create a network and save
/// let (vs, net) = checkpoint::build(device());
/// // ... training loop ...
/// checkpoint::save(&vs, "run/weights.safetensors").unwrap();
///
/// // Restore on any device
/// let (mut vs, net) = checkpoint::build(device());
/// checkpoint::load(&mut vs, "run/weights.safetensors").unwrap();
/// let (policy, value) = net.forward_t(&board_tensor, false);
/// ```
use std::path::Path;
use tch::{nn, Device, TchError};
use super::Net;

/// Creates a `(VarStore, Net)` pair on `device` using the standard
/// AlphaZero-Shogi configuration (256 channels, 20 residual blocks).
pub fn build(device: Device) -> (nn::VarStore, Net) {
    let vs = nn::VarStore::new(device);
    let net = Net::new(&vs.root());
    (vs, net)
}

/// Same as `build` but with explicit channel width and tower depth.
///
/// Use this for tests, ablations, or lightweight inference models.
pub fn build_with_config(
    device: Device,
    channels: i64,
    num_blocks: usize,
) -> (nn::VarStore, Net) {
    let vs = nn::VarStore::new(device);
    let net = Net::with_config(&vs.root(), channels, num_blocks);
    (vs, net)
}

/// Writes all `vs` variables to `path`.
///
/// `path` must end with `.safetensors`; see the module-level note for why.
pub fn save(vs: &nn::VarStore, path: impl AsRef<Path>) -> Result<(), TchError> {
    vs.save(path)
}

/// Reads variables from `path` into `vs`.
///
/// `path` must end with `.safetensors`.  `vs` must have been built with the
/// same architecture used when saving so that variable names align.
pub fn load(vs: &mut nn::VarStore, path: impl AsRef<Path>) -> Result<(), TchError> {
    vs.load(path)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{Kind, Tensor};
    use crate::nn::{NUM_PLANES, NUM_ACTIONS};

    fn cpu_input(batch: i64) -> Tensor {
        Tensor::randn(
            [batch, NUM_PLANES as i64, 9, 9],
            (Kind::Float, Device::Cpu),
        )
    }

    /// Temp path with the required `.safetensors` extension.
    fn tmp(stem: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("{stem}.safetensors"))
    }

    // ----- build helpers -----

    #[test]
    fn test_build_with_config_output_shapes() {
        let (_vs, net) = build_with_config(Device::Cpu, 8, 2);
        let (policy, value) = net.forward_t(&cpu_input(1), false);
        assert_eq!(policy.size(), vec![1, NUM_ACTIONS as i64]);
        assert_eq!(value.size(), vec![1, 1]);
    }

    // ----- save -----

    #[test]
    fn test_save_creates_nonempty_file() {
        let path = tmp("shogi_ckpt_save_test");
        let (vs, _net) = build_with_config(Device::Cpu, 8, 2);
        save(&vs, &path).expect("save failed");
        assert!(path.exists(), "checkpoint file missing");
        assert!(path.metadata().unwrap().len() > 0, "checkpoint file is empty");
        let _ = std::fs::remove_file(&path);
    }

    // ----- load -----

    #[test]
    fn test_load_nonexistent_returns_err() {
        let (mut vs, _net) = build_with_config(Device::Cpu, 8, 2);
        let result = load(&mut vs, tmp("shogi_no_such_file_xyzzy"));
        assert!(result.is_err(), "loading a missing file should return Err");
    }

    // ----- roundtrip -----

    #[test]
    fn test_save_load_weights_match() {
        let path = tmp("shogi_ckpt_roundtrip");
        let xs = cpu_input(2);

        // Build net1 and save.
        let (vs1, net1) = build_with_config(Device::Cpu, 8, 2);
        let (p1, v1) = net1.forward_t(&xs, false);
        save(&vs1, &path).expect("save failed");

        // Build net2 (fresh random weights) and load.
        let (mut vs2, net2) = build_with_config(Device::Cpu, 8, 2);
        load(&mut vs2, &path).expect("load failed");
        let (p2, v2) = net2.forward_t(&xs, false);

        // After loading, outputs must be bit-for-bit identical.
        let p_diff = (&p1 - &p2).abs().max().double_value(&[]);
        let v_diff = (&v1 - &v2).abs().max().double_value(&[]);
        assert!(p_diff < 1e-6, "policy mismatch after load (max diff {p_diff:.2e})");
        assert!(v_diff < 1e-6, "value mismatch after load (max diff {v_diff:.2e})");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_load_changes_weights() {
        // Sanity: two independently initialised nets diverge before any load.
        let xs = cpu_input(1);
        let (_vs1, net1) = build_with_config(Device::Cpu, 8, 2);
        let (_vs2, net2) = build_with_config(Device::Cpu, 8, 2);
        let (p1, _) = net1.forward_t(&xs, false);
        let (p2, _) = net2.forward_t(&xs, false);
        let diff = (&p1 - &p2).abs().max().double_value(&[]);
        assert!(diff > 1e-4, "two fresh nets should differ (diff={diff:.2e})");
    }
}
