/// Self-play worker pool for AlphaZero training.
///
/// Each worker thread runs an infinite self-play loop, collecting game records
/// and pushing them to a shared `ReplayBuffer`.  The training thread drives the
/// pool through `WorkerPool::spawn` / `WorkerPool::join`.
///
/// Bullet roadmap:
///   ✓ Spawn N threads with shutdown signal         (bullet 1)
///   ✓ Per-worker network copy                      (bullet 2)
///   • Workers push to shared replay buffer          (bullet 3)
///   • Weight broadcast from training thread         (bullet 4)
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use tch::nn;

use crate::nn::{Net, checkpoint::build_with_config};

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

/// Pool of N long-running self-play threads, each with its own network copy.
pub struct WorkerPool {
    handles: Vec<JoinHandle<()>>,
    shutdown: Arc<AtomicBool>,
    /// Architecture parameters — stored for weight broadcast (bullet 4).
    channels: i64,
    blocks: usize,
}

impl WorkerPool {
    /// Number of workers to use by default: `available_parallelism − 1`, minimum 1.
    pub fn default_num_workers() -> usize {
        thread::available_parallelism()
            .map(|n| n.get().saturating_sub(1).max(1))
            .unwrap_or(1)
    }

    /// Spawn `num_workers` threads, each starting with a copy of `master_vs`.
    pub fn spawn(
        num_workers: usize,
        master_vs: &nn::VarStore,
        channels: i64,
        blocks: usize,
    ) -> Self {
        assert!(num_workers > 0, "must spawn at least one worker");
        let shutdown = Arc::new(AtomicBool::new(false));

        let handles = (0..num_workers)
            .map(|_| {
                let (worker_vs, worker_net) = build_worker_net(master_vs, channels, blocks);
                let shutdown = Arc::clone(&shutdown);
                thread::spawn(move || worker_loop(worker_vs, worker_net, shutdown))
            })
            .collect();

        Self { handles, shutdown, channels, blocks }
    }

    /// Number of live worker threads.
    pub fn num_workers(&self) -> usize {
        self.handles.len()
    }

    /// Signal all workers to stop, then wait for every thread to exit.
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

fn worker_loop(_vs: nn::VarStore, _net: Net, shutdown: Arc<AtomicBool>) {
    while !shutdown.load(Ordering::Relaxed) {
        // Placeholder: bullet 3 replaces this with a self-play + buffer push.
        thread::sleep(Duration::from_millis(1));
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{Device, Kind, Tensor};
    use crate::nn::{NUM_PLANES, NUM_ACTIONS, checkpoint::build_with_config};

    fn make_master() -> (nn::VarStore, Net) {
        build_with_config(Device::Cpu, 8, 2)
    }

    fn randn_input() -> Tensor {
        Tensor::randn([1, NUM_PLANES as i64, 9, 9], (Kind::Float, Device::Cpu))
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
        assert!(diff < 1e-6, "worker was affected by master mutation: {diff:.2e}");
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
        let pool = WorkerPool::spawn(3, &master_vs, 8, 2);
        assert_eq!(pool.num_workers(), 3);
        pool.join();
    }

    #[test]
    fn test_spawn_one_worker() {
        let (master_vs, _) = make_master();
        let pool = WorkerPool::spawn(1, &master_vs, 8, 2);
        assert_eq!(pool.num_workers(), 1);
        pool.join();
    }

    #[test]
    fn test_join_terminates_all_threads() {
        let (master_vs, _) = make_master();
        let pool = WorkerPool::spawn(4, &master_vs, 8, 2);
        pool.join(); // must return; hanging == deadlock
    }
}
