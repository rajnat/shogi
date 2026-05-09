/// Self-play worker pool for AlphaZero training.
///
/// Each worker thread runs an infinite self-play loop, collecting game records
/// and pushing them to a shared `ReplayBuffer`.  The training thread drives the
/// pool through `WorkerPool::spawn` / `WorkerPool::join`.
///
/// Subsequent bullets add:
///   • per-worker network copy
///   • replay-buffer push
///   • weight broadcast
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};

// ---------------------------------------------------------------------------
// Worker pool
// ---------------------------------------------------------------------------

/// Pool of N long-running self-play threads.
pub struct WorkerPool {
    handles: Vec<JoinHandle<()>>,
    shutdown: Arc<AtomicBool>,
}

impl WorkerPool {
    /// Number of workers to use by default: `available_parallelism − 1`, minimum 1.
    pub fn default_num_workers() -> usize {
        thread::available_parallelism()
            .map(|n| n.get().saturating_sub(1).max(1))
            .unwrap_or(1)
    }

    /// Spawn `num_workers` worker threads.
    ///
    /// Each thread loops until `join()` is called.  The actual self-play body
    /// will be added in the next bullets.
    pub fn spawn(num_workers: usize) -> Self {
        assert!(num_workers > 0, "must spawn at least one worker");
        let shutdown = Arc::new(AtomicBool::new(false));

        let handles = (0..num_workers)
            .map(|_| {
                let shutdown = Arc::clone(&shutdown);
                thread::spawn(move || worker_loop(shutdown))
            })
            .collect();

        Self { handles, shutdown }
    }

    /// Number of live worker threads.
    pub fn num_workers(&self) -> usize {
        self.handles.len()
    }

    /// Signal all workers to stop, then wait for every thread to exit.
    ///
    /// Blocks until all workers have completed their current game (or the
    /// idle sleep) and returned.
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

fn worker_loop(shutdown: Arc<AtomicBool>) {
    while !shutdown.load(Ordering::Relaxed) {
        // Placeholder: real self-play + buffer push will replace this sleep.
        thread::sleep(std::time::Duration::from_millis(1));
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_num_workers_at_least_one() {
        assert!(WorkerPool::default_num_workers() >= 1);
    }

    #[test]
    fn test_spawn_correct_count() {
        let pool = WorkerPool::spawn(3);
        assert_eq!(pool.num_workers(), 3);
        pool.join();
    }

    #[test]
    fn test_join_terminates_all_threads() {
        // If join() hangs this test will time out — a live deadlock detector.
        let pool = WorkerPool::spawn(4);
        pool.join(); // must return
    }

    #[test]
    fn test_spawn_one_worker() {
        let pool = WorkerPool::spawn(1);
        assert_eq!(pool.num_workers(), 1);
        pool.join();
    }

    #[test]
    fn test_default_workers_equals_available_minus_one() {
        let expected = thread::available_parallelism()
            .map(|n| n.get().saturating_sub(1).max(1))
            .unwrap_or(1);
        assert_eq!(WorkerPool::default_num_workers(), expected);
    }
}
