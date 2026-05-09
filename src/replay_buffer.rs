/// Replay buffer for AlphaZero-style training.
///
/// Stores a fixed-capacity ring of `(board_tensor, policy_target, value_target)`
/// records collected from self-play games.  When the buffer is full the oldest
/// records are evicted to make room for new ones.
///
/// The buffer is wrapped in `Arc<Mutex<ReplayBuffer>>` by callers that need
/// concurrent access from multiple self-play workers and the training thread.
use std::collections::VecDeque;

use rand::seq::index::sample;
use rand::Rng;
use tch::{Device, Kind, Tensor};

use crate::nn::NUM_ACTIONS;
use crate::nn::move_index::MIRROR_ACTION_TABLE;

// ---------------------------------------------------------------------------
// Entry type
// ---------------------------------------------------------------------------

/// One training sample: board encoding, MCTS visit distribution, game outcome.
pub struct ReplayEntry {
    /// Board tensor of shape `[C, 9, 9]` (on CPU, f32).
    pub board: Tensor,
    /// MCTS visit distribution, length `NUM_ACTIONS`.
    pub policy: Vec<f32>,
    /// Game outcome z ∈ {−1, 0, +1} from the side-to-move's perspective.
    pub value: f32,
}

// ---------------------------------------------------------------------------
// Buffer
// ---------------------------------------------------------------------------

/// Fixed-capacity ring buffer of replay entries.
pub struct ReplayBuffer {
    capacity: usize,
    entries: VecDeque<ReplayEntry>,
}

impl ReplayBuffer {
    /// Create an empty buffer with the given capacity.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "ReplayBuffer capacity must be > 0");
        Self { capacity, entries: VecDeque::with_capacity(capacity.min(65_536)) }
    }

    /// Number of entries currently stored.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Append records from a completed self-play game.
    ///
    /// Older entries are evicted if the buffer would exceed `capacity`.
    pub fn push_game(&mut self, records: Vec<(Tensor, Vec<f32>, f32)>) {
        for (board, policy, value) in records {
            if self.entries.len() == self.capacity {
                self.entries.pop_front();
            }
            self.entries.push_back(ReplayEntry { board, policy, value });
        }
    }

    /// Sample `n` entries uniformly at random (with replacement if n > len).
    ///
    /// Returns `(boards, policies, values)` as batched tensors on `device`:
    /// - `boards`:   `[n, C, 9, 9]`  f32
    /// - `policies`: `[n, NUM_ACTIONS]` f32
    /// - `values`:   `[n, 1]`          f32
    ///
    /// Panics if the buffer is empty.
    pub fn sample_batch<R: Rng>(
        &self,
        n: usize,
        device: Device,
        rng: &mut R,
    ) -> (Tensor, Tensor, Tensor) {
        assert!(!self.is_empty(), "cannot sample from empty ReplayBuffer");

        let len = self.entries.len();

        // Draw indices with replacement when n > len, without replacement otherwise.
        let indices: Vec<usize> = if n <= len {
            let idx_set = sample(rng, len, n);
            idx_set.into_vec()
        } else {
            (0..n).map(|_| rng.gen_range(0..len)).collect()
        };

        // Collect samples, applying left-right board flip augmentation with p=0.5.
        let mut board_list  = Vec::with_capacity(n);
        let mut policy_flat = Vec::with_capacity(n * NUM_ACTIONS);
        let mut value_flat  = Vec::with_capacity(n);

        for &i in &indices {
            let entry = &self.entries[i];
            if rng.gen_bool(0.5) {
                // Flip board along the file dimension (dim 2).
                board_list.push(entry.board.to_device(device).flip([2_i64]));
                // Permute policy: flipped_policy[j] = original_policy[mirror(j)].
                for j in 0..NUM_ACTIONS {
                    policy_flat.push(entry.policy[MIRROR_ACTION_TABLE[j]]);
                }
            } else {
                board_list.push(entry.board.to_device(device));
                policy_flat.extend_from_slice(&entry.policy);
            }
            value_flat.push(entry.value);
        }

        let boards = Tensor::stack(&board_list, 0); // [n, C, 9, 9]

        let policies = Tensor::from_slice(&policy_flat)
            .to_kind(Kind::Float)
            .to_device(device)
            .reshape([n as i64, NUM_ACTIONS as i64]);

        let values = Tensor::from_slice(&value_flat)
            .to_kind(Kind::Float)
            .to_device(device)
            .reshape([n as i64, 1]);

        (boards, policies, values)
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
    use tch::Device;

    fn dummy_entry(value: f32) -> (Tensor, Vec<f32>, f32) {
        let board = Tensor::zeros([119, 9, 9], (tch::Kind::Float, Device::Cpu));
        let policy = vec![0.0f32; NUM_ACTIONS];
        (board, policy, value)
    }

    #[test]
    fn test_new_buffer_is_empty() {
        let buf = ReplayBuffer::new(1000);
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
    }

    #[test]
    fn test_push_game_increases_len() {
        let mut buf = ReplayBuffer::new(1000);
        let records = vec![dummy_entry(1.0), dummy_entry(-1.0)];
        buf.push_game(records);
        assert_eq!(buf.len(), 2);
    }

    #[test]
    fn test_ring_evicts_oldest() {
        let mut buf = ReplayBuffer::new(3);
        // Push 5 entries across two games; only the last 3 should remain.
        buf.push_game(vec![dummy_entry(1.0), dummy_entry(2.0), dummy_entry(3.0)]);
        buf.push_game(vec![dummy_entry(4.0), dummy_entry(5.0)]);
        assert_eq!(buf.len(), 3);
        // Remaining entries should be values 3, 4, 5.
        let values: Vec<f32> = buf.entries.iter().map(|e| e.value).collect();
        assert_eq!(values, vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_sample_batch_shapes() {
        let mut buf = ReplayBuffer::new(1000);
        for i in 0..20 {
            buf.push_game(vec![dummy_entry(if i % 2 == 0 { 1.0 } else { -1.0 })]);
        }
        let mut rng = StdRng::seed_from_u64(0);
        let (boards, policies, values) = buf.sample_batch(8, Device::Cpu, &mut rng);
        assert_eq!(boards.size(), vec![8, 119, 9, 9]);
        assert_eq!(policies.size(), vec![8, NUM_ACTIONS as i64]);
        assert_eq!(values.size(), vec![8, 1]);
    }

    #[test]
    fn test_sample_with_replacement_when_n_gt_len() {
        let mut buf = ReplayBuffer::new(1000);
        buf.push_game(vec![dummy_entry(1.0), dummy_entry(-1.0)]);
        // n=10 > len=2: should sample with replacement and not panic.
        let mut rng = StdRng::seed_from_u64(0);
        let (boards, _, _) = buf.sample_batch(10, Device::Cpu, &mut rng);
        assert_eq!(boards.size()[0], 10);
    }

    #[test]
    fn test_capacity_exactly_respected() {
        let mut buf = ReplayBuffer::new(5);
        for i in 0..10 {
            buf.push_game(vec![dummy_entry(i as f32)]);
        }
        assert_eq!(buf.len(), 5);
        let values: Vec<f32> = buf.entries.iter().map(|e| e.value).collect();
        assert_eq!(values, vec![5.0, 6.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_sample_returns_values_from_buffer() {
        let mut buf = ReplayBuffer::new(100);
        // Single entry with a distinctive value.
        buf.push_game(vec![dummy_entry(0.42)]);
        let mut rng = StdRng::seed_from_u64(0);
        let (_, _, values) = buf.sample_batch(1, Device::Cpu, &mut rng);
        let v: f32 = values.double_value(&[0, 0]) as f32;
        assert!((v - 0.42).abs() < 1e-5);
    }

    // ----- augmentation -----

    #[test]
    fn test_augmented_batch_shapes_unchanged() {
        // Augmentation must not change the output tensor shapes.
        let mut buf = ReplayBuffer::new(100);
        for i in 0..20 {
            buf.push_game(vec![dummy_entry(if i % 2 == 0 { 1.0 } else { -1.0 })]);
        }
        let mut rng = StdRng::seed_from_u64(99);
        let (boards, policies, values) = buf.sample_batch(8, Device::Cpu, &mut rng);
        assert_eq!(boards.size(),   vec![8, 119, 9, 9]);
        assert_eq!(policies.size(), vec![8, NUM_ACTIONS as i64]);
        assert_eq!(values.size(),   vec![8, 1]);
    }

    #[test]
    fn test_augmented_policy_sums_preserved() {
        // Mirroring is a permutation — the policy sum must remain 1.
        let uniform: Vec<f32> = vec![1.0 / NUM_ACTIONS as f32; NUM_ACTIONS];
        let mut buf = ReplayBuffer::new(10);
        let board = Tensor::zeros([119, 9, 9], (tch::Kind::Float, Device::Cpu));
        buf.push_game(vec![(board, uniform, 1.0)]);

        let mut rng = StdRng::seed_from_u64(0);
        let (_, policies, _) = buf.sample_batch(16, Device::Cpu, &mut rng);
        // Each row of `policies` should sum to 1.0.
        let sums: Vec<f64> = (0..16)
            .map(|i| policies.get(i).sum(tch::Kind::Float).double_value(&[]))
            .collect();
        for (i, &s) in sums.iter().enumerate() {
            assert!((s - 1.0).abs() < 1e-4, "row {i}: policy sum = {s}");
        }
    }

    #[test]
    fn test_augmented_board_flip_is_correct() {
        // Place a 1.0 in plane 0 at (rank=0, file_idx=0) and verify that
        // after a flip the 1.0 appears at (rank=0, file_idx=8).
        let mut data = vec![0.0f32; 119 * 9 * 9];
        // Spatial layout: [plane, rank, file] — dim1=rank, dim2=file
        // plane 0, rank 0, file_idx 0 → flat index = 0*81 + 0*9 + 0 = 0
        data[0] = 1.0;
        let board = Tensor::from_slice(&data)
            .reshape([119, 9, 9])
            .to_kind(tch::Kind::Float);

        let policy = vec![0.0f32; NUM_ACTIONS];
        let mut buf = ReplayBuffer::new(10);
        buf.push_game(vec![(board, policy, 0.0)]);

        // Sample many times; at least one should be flipped (p=0.5).
        // On a flip, the 1.0 should move from file_idx=0 to file_idx=8.
        let mut found_flipped = false;
        let mut found_original = false;
        for seed in 0..50u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let (boards, _, _) = buf.sample_batch(1, Device::Cpu, &mut rng);
            // plane 0, rank 0, file_idx=0 → [0, 0, 0, 0]
            let at_orig   = boards.double_value(&[0, 0, 0, 0]);
            // plane 0, rank 0, file_idx=8 → [0, 0, 0, 8]
            let at_mirror = boards.double_value(&[0, 0, 0, 8]);
            if at_orig > 0.5   { found_original = true; }
            if at_mirror > 0.5 { found_flipped  = true; }
            if found_original && found_flipped { break; }
        }
        assert!(found_original, "original orientation never sampled");
        assert!(found_flipped,  "flipped orientation never sampled");
    }
}
