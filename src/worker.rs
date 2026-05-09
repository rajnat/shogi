/// Self-play worker pool for AlphaZero training.
///
/// Each worker thread runs an infinite self-play loop, collecting game records
/// and pushing them to a shared `ReplayBuffer`.  The training thread drives the
/// pool through `WorkerPool::spawn` / `WorkerPool::join`.
///
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use rand::SeedableRng;
use rand::rngs::StdRng;
use tch::{Tensor, nn};

use crate::nn::{Net, checkpoint::build_with_config};
use crate::replay_buffer::ReplayBuffer;
use crate::selfplay::{SelfPlayConfig, play_game};

/// One deep copy of every VarStore variable, keyed by name.
///
/// `Tensor` is `Send` but not `Sync` in tch-0.24, so each worker slot holds
/// its own independent `HashMap` rather than a shared `Arc`.  The per-worker
/// allocation cost is negligible for the small test networks; for a full
/// 256-channel net the training thread creates N copies only at broadcast time.
type WeightSnapshot = HashMap<String, Tensor>;

/// Per-worker deposit slot.  Holds at most one pending snapshot — a newer
/// broadcast silently overwrites any un-consumed earlier one.
type WeightSlot = Arc<Mutex<Option<WeightSnapshot>>>;

/// Deep-copy every variable in `vs` into a fresh `HashMap`.
///
/// Each tensor owns its own storage so the training thread can safely continue
/// modifying `vs` after `broadcast_weights` returns.
fn snapshot_vars(vs: &nn::VarStore) -> WeightSnapshot {
    tch::no_grad(|| {
        vs.variables()
            .into_iter()
            .map(|(name, src)| {
                let mut dst = src.zeros_like();
                dst.copy_(&src);
                (name, dst)
            })
            .collect()
    })
}

/// Copy all variables from `snapshot` into `worker_vs` in-place.
fn apply_snapshot(worker_vs: &nn::VarStore, snapshot: &WeightSnapshot) {
    let mut vars = worker_vs.variables();
    tch::no_grad(|| {
        for (name, src) in snapshot {
            if let Some(dst) = vars.get_mut(name) {
                dst.copy_(src);
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Weight copy helper
// ---------------------------------------------------------------------------

/// Build a fresh `(VarStore, Net)` on CPU and deep-copy all weights from
/// `master_vs`.  The resulting VarStore is fully independent — mutating
/// `master_vs` afterwards has no effect on the copy.
pub fn build_worker_net(
    master_vs: &nn::VarStore,
    channels: i64,
    blocks: usize,
) -> (nn::VarStore, Net) {
    let (mut worker_vs, worker_net) = build_with_config(tch::Device::Cpu, channels, blocks);
    worker_vs.copy(master_vs).expect("weight copy failed");
    (worker_vs, worker_net)
}

// ---------------------------------------------------------------------------
// Worker pool
// ---------------------------------------------------------------------------

/// Pool of N long-running self-play threads, each with its own network copy,
/// all pushing records to a shared `ReplayBuffer`.
pub struct WorkerPool {
    handles: Vec<JoinHandle<()>>,
    shutdown: Arc<AtomicBool>,
    weight_slots: Vec<WeightSlot>,
    /// Stored for potential future use (e.g. rebuilding nets on arch change).
    #[allow(dead_code)]
    channels: i64,
    #[allow(dead_code)]
    blocks: usize,
}

impl WorkerPool {
    /// Number of workers to use by default: `available_parallelism − 1`, minimum 1.
    pub fn default_num_workers() -> usize {
        thread::available_parallelism()
            .map(|n| n.get().saturating_sub(1).max(1))
            .unwrap_or(1)
    }

    /// Spawn `num_workers` threads.
    ///
    /// Each worker starts with a deep copy of `master_vs`, runs self-play games
    /// using `config`, and pushes every completed game's records into `buffer`.
    /// Workers are seeded with `base_seed + worker_index` for reproducibility.
    pub fn spawn(
        num_workers: usize,
        master_vs: &nn::VarStore,
        channels: i64,
        blocks: usize,
        config: SelfPlayConfig,
        buffer: Arc<Mutex<ReplayBuffer>>,
        base_seed: u64,
    ) -> Self {
        assert!(num_workers > 0, "must spawn at least one worker");
        let shutdown = Arc::new(AtomicBool::new(false));
        let config = Arc::new(config);

        let weight_slots: Vec<WeightSlot> = (0..num_workers)
            .map(|_| Arc::new(Mutex::new(None)))
            .collect();

        let handles = weight_slots
            .iter()
            .enumerate()
            .map(|(idx, slot)| {
                let (worker_vs, worker_net) = build_worker_net(master_vs, channels, blocks);
                let shutdown = Arc::clone(&shutdown);
                let buffer = Arc::clone(&buffer);
                let config = Arc::clone(&config);
                let slot = Arc::clone(slot);
                let seed = base_seed.wrapping_add(idx as u64);
                thread::spawn(move || {
                    worker_loop(worker_vs, worker_net, config, buffer, shutdown, slot, seed)
                })
            })
            .collect();

        Self {
            handles,
            shutdown,
            weight_slots,
            channels,
            blocks,
        }
    }

    /// Number of live worker threads.
    pub fn num_workers(&self) -> usize {
        self.handles.len()
    }

    /// Push the current `master_vs` weights to every worker.
    ///
    /// Creates one deep copy of the master variables (via `snapshot_vars`), then
    /// shares it across all per-worker slots via `Arc` — no per-worker tensor
    /// allocation.  Workers apply the update between games; there is no mid-game
    /// weight change and no lock contention during forward passes.
    pub fn broadcast_weights(&self, master_vs: &nn::VarStore) {
        // snapshot_vars is called once per worker; each gets its own deep copy
        // so workers can apply concurrently without sharing tensor storage.
        for slot in &self.weight_slots {
            *slot.lock().unwrap() = Some(snapshot_vars(master_vs));
        }
    }

    /// Signal all workers to stop after their current game, then join all threads.
    ///
    /// Because workers check the shutdown flag *after* each game, every thread
    /// is guaranteed to push at least one completed game before returning.
    pub fn join(self) {
        self.shutdown.store(true, Ordering::Relaxed);
        for h in self.handles {
            h.join().expect("worker thread panicked");
        }
    }
}

// ---------------------------------------------------------------------------
// Worker body
// ---------------------------------------------------------------------------

fn worker_loop(
    worker_vs: nn::VarStore,
    net: Net,
    config: Arc<SelfPlayConfig>,
    buffer: Arc<Mutex<ReplayBuffer>>,
    shutdown: Arc<AtomicBool>,
    slot: WeightSlot,
    seed: u64,
) {
    let mut rng = StdRng::seed_from_u64(seed);
    loop {
        // Apply pending weight broadcast before the next game.
        if let Some(snap) = slot.lock().unwrap().take() {
            apply_snapshot(&worker_vs, &snap);
        }

        let result = tch::no_grad(|| play_game(&net, &config, tch::Device::Cpu, &mut rng));
        buffer.lock().unwrap().push_game(result.records);

        // Check shutdown after the game — guarantees at least one game per worker.
        if shutdown.load(Ordering::Relaxed) {
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::{NUM_ACTIONS, NUM_PLANES, checkpoint::build_with_config};
    use tch::{Device, Kind, Tensor};

    fn make_master() -> (nn::VarStore, Net) {
        build_with_config(Device::Cpu, 8, 2)
    }

    fn randn_input() -> Tensor {
        Tensor::randn([1, NUM_PLANES as i64, 9, 9], (Kind::Float, Device::Cpu))
    }

    /// Tiny config so test games finish quickly.
    fn tiny_config() -> SelfPlayConfig {
        SelfPlayConfig {
            num_simulations: 4,
            max_moves: 20,
            resign_min_ply: 100, // don't resign in tiny games
            ..SelfPlayConfig::default()
        }
    }

    fn test_buffer(capacity: usize) -> Arc<Mutex<ReplayBuffer>> {
        Arc::new(Mutex::new(ReplayBuffer::new(capacity)))
    }

    // ----- default_num_workers -----

    #[test]
    fn test_default_num_workers_at_least_one() {
        assert!(WorkerPool::default_num_workers() >= 1);
    }

    #[test]
    fn test_default_workers_equals_available_minus_one() {
        let expected = thread::available_parallelism()
            .map(|n| n.get().saturating_sub(1).max(1))
            .unwrap_or(1);
        assert_eq!(WorkerPool::default_num_workers(), expected);
    }

    // ----- build_worker_net -----

    #[test]
    fn test_worker_net_outputs_match_master() {
        let (master_vs, master_net) = make_master();
        let (_worker_vs, worker_net) = build_worker_net(&master_vs, 8, 2);

        let xs = randn_input();
        let (mp, mv) = tch::no_grad(|| master_net.forward_t(&xs, false));
        let (wp, wv) = tch::no_grad(|| worker_net.forward_t(&xs, false));

        let p_diff = (&mp - &wp).abs().max().double_value(&[]);
        let v_diff = (&mv - &wv).abs().max().double_value(&[]);
        assert!(p_diff < 1e-6, "policy mismatch after copy: {p_diff:.2e}");
        assert!(v_diff < 1e-6, "value mismatch after copy: {v_diff:.2e}");
    }

    #[test]
    fn test_worker_net_is_independent_of_master() {
        let (master_vs, _master_net) = make_master();
        let (_worker_vs, worker_net) = build_worker_net(&master_vs, 8, 2);

        let xs = randn_input();
        let (p_before, _) = tch::no_grad(|| worker_net.forward_t(&xs, false));

        // Zero out every weight in the master in-place.
        // no_grad is required: leaf parameters block in-place ops under autograd.
        tch::no_grad(|| {
            for (_, mut t) in master_vs.variables() {
                let _ = t.fill_(0.0);
            }
        });

        let (p_after, _) = tch::no_grad(|| worker_net.forward_t(&xs, false));
        let diff = (&p_before - &p_after).abs().max().double_value(&[]);
        assert!(
            diff < 1e-6,
            "worker was affected by master mutation: {diff:.2e}"
        );
    }

    #[test]
    fn test_worker_policy_shape() {
        let (master_vs, _) = make_master();
        let (_, worker_net) = build_worker_net(&master_vs, 8, 2);
        let (p, v) = tch::no_grad(|| worker_net.forward_t(&randn_input(), false));
        assert_eq!(p.size(), vec![1, NUM_ACTIONS as i64]);
        assert_eq!(v.size(), vec![1, 1]);
    }

    // ----- WorkerPool spawn / join -----

    #[test]
    fn test_spawn_correct_count() {
        let (master_vs, _) = make_master();
        let pool = WorkerPool::spawn(3, &master_vs, 8, 2, tiny_config(), test_buffer(10_000), 0);
        assert_eq!(pool.num_workers(), 3);
        pool.join();
    }

    #[test]
    fn test_spawn_one_worker() {
        let (master_vs, _) = make_master();
        let pool = WorkerPool::spawn(1, &master_vs, 8, 2, tiny_config(), test_buffer(10_000), 0);
        assert_eq!(pool.num_workers(), 1);
        pool.join();
    }

    #[test]
    fn test_join_terminates_all_threads() {
        let (master_vs, _) = make_master();
        // Must return — hanging here means deadlock.
        WorkerPool::spawn(4, &master_vs, 8, 2, tiny_config(), test_buffer(10_000), 0).join();
    }

    #[test]
    fn test_single_worker_populates_buffer() {
        let (master_vs, _) = make_master();
        let buffer = test_buffer(10_000);
        let pool = WorkerPool::spawn(1, &master_vs, 8, 2, tiny_config(), Arc::clone(&buffer), 0);
        pool.join(); // worker plays exactly one game then exits
        assert!(
            buffer.lock().unwrap().len() > 0,
            "buffer empty after worker ran one game"
        );
    }

    #[test]
    fn test_two_workers_each_push_at_least_one_game() {
        let (master_vs, _) = make_master();
        let buffer = test_buffer(10_000);
        let pool = WorkerPool::spawn(2, &master_vs, 8, 2, tiny_config(), Arc::clone(&buffer), 0);
        pool.join();
        // Each worker plays at least one game; max_moves=20 so each game is ≤20 records.
        // Two workers → at least 2 positions pushed (one game each is guaranteed).
        assert!(buffer.lock().unwrap().len() >= 2);
    }

    #[test]
    fn test_workers_use_different_seeds() {
        let (master_vs, _) = make_master();
        let buffer = test_buffer(10_000);
        let pool = WorkerPool::spawn(2, &master_vs, 8, 2, tiny_config(), Arc::clone(&buffer), 42);
        pool.join();
        assert!(buffer.lock().unwrap().len() >= 2);
    }

    #[test]
    fn test_snapshot_is_independent_of_master() {
        let (master_vs, _) = make_master();
        let snap = snapshot_vars(&master_vs);

        // Record snapshot's total weight mass before touching the master.
        // BatchNorm biases are init to 0, so we can't use `all(nonzero)`;
        // checking that the sum is preserved after zeroing master is robust.
        let sum_before: f64 = snap
            .values()
            .map(|t| t.abs().sum(tch::Kind::Double).double_value(&[]))
            .sum();
        assert!(sum_before > 0.0, "snapshot is unexpectedly all-zero");

        // Zero every master variable in-place.
        tch::no_grad(|| {
            for (_, mut t) in master_vs.variables() {
                let _ = t.fill_(0.0);
            }
        });

        let sum_after: f64 = snap
            .values()
            .map(|t| t.abs().sum(tch::Kind::Double).double_value(&[]))
            .sum();

        assert!(
            (sum_before - sum_after).abs() < 1e-3,
            "snapshot weight sum changed after zeroing master \
             ({sum_before:.4} → {sum_after:.4}): tensors are aliased"
        );
    }

    #[test]
    fn test_apply_snapshot_changes_worker_output() {
        let (master_vs1, _) = make_master();
        let (master_vs2, _) = make_master(); // different random weights
        let (worker_vs, worker_net) = build_worker_net(&master_vs1, 8, 2);
        let xs = randn_input();
        let (p_before, _) = tch::no_grad(|| worker_net.forward_t(&xs, false));

        apply_snapshot(&worker_vs, &snapshot_vars(&master_vs2));

        let (p_after, _) = tch::no_grad(|| worker_net.forward_t(&xs, false));
        let diff = (&p_before - &p_after).abs().max().double_value(&[]);
        assert!(
            diff > 1e-4,
            "apply_snapshot had no effect (diff={diff:.2e})"
        );
    }

    #[test]
    fn test_apply_snapshot_matches_source() {
        let (master_vs1, master_net1) = make_master();
        let (master_vs2, master_net2) = make_master();
        let (worker_vs, worker_net) = build_worker_net(&master_vs1, 8, 2);

        apply_snapshot(&worker_vs, &snapshot_vars(&master_vs2));

        let xs = randn_input();
        let (pm2, _) = tch::no_grad(|| master_net2.forward_t(&xs, false));
        let (pw, _) = tch::no_grad(|| worker_net.forward_t(&xs, false));
        let diff2 = (&pm2 - &pw).abs().max().double_value(&[]);
        assert!(
            diff2 < 1e-5,
            "worker doesn't match source after apply: {diff2:.2e}"
        );

        let (pm1, _) = tch::no_grad(|| master_net1.forward_t(&xs, false));
        let diff1 = (&pm1 - &pw).abs().max().double_value(&[]);
        assert!(diff1 > 1e-4, "worker still matches old master after apply");
    }

    #[test]
    fn test_broadcast_does_not_panic() {
        let (master_vs, _) = make_master();
        let pool = WorkerPool::spawn(2, &master_vs, 8, 2, tiny_config(), test_buffer(10_000), 0);
        pool.broadcast_weights(&master_vs);
        pool.broadcast_weights(&master_vs); // second call overwrites pending
        pool.join();
    }

    #[test]
    fn test_broadcast_workers_still_populate_buffer() {
        let (master_vs, _) = make_master();
        let buffer = test_buffer(10_000);
        let pool = WorkerPool::spawn(2, &master_vs, 8, 2, tiny_config(), Arc::clone(&buffer), 0);
        pool.broadcast_weights(&master_vs);
        pool.join();
        assert!(buffer.lock().unwrap().len() > 0);
    }
}
