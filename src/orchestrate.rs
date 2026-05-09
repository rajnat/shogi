/// Training orchestration: the outer loop that coordinates self-play workers,
/// the replay buffer, and the training thread.
///
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use rand::Rng;

use crate::replay_buffer::ReplayBuffer;
use crate::train::Trainer;
use crate::worker::WorkerPool;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Parameters for the outer training loop.
#[derive(Debug, Clone)]
pub struct OrchestrationConfig {
    /// Number of gradient steps between each weight broadcast to workers.
    pub steps_per_broadcast: u64,
    /// Stop after this many total training steps (0 = run until shutdown).
    pub total_steps: u64,
    /// How often to poll the buffer size while waiting for it to fill (ms).
    pub fill_poll_ms: u64,
}

impl Default for OrchestrationConfig {
    fn default() -> Self {
        OrchestrationConfig {
            steps_per_broadcast: 100,
            total_steps: 0,
            fill_poll_ms: 200,
        }
    }
}

// ---------------------------------------------------------------------------
// Buffer fill wait
// ---------------------------------------------------------------------------

/// Block until `buffer` contains at least `min_size` entries.
///
/// Polls every `poll_ms` milliseconds and prints a status line once per second.
pub fn wait_for_buffer(buffer: &Arc<Mutex<ReplayBuffer>>, min_size: usize, poll_ms: u64) {
    if buffer.lock().unwrap().len() >= min_size {
        return;
    }
    println!("Waiting for replay buffer to reach {min_size} positions…");
    let mut last_print = Instant::now();
    loop {
        std::thread::sleep(Duration::from_millis(poll_ms));
        let len = buffer.lock().unwrap().len();
        if last_print.elapsed() >= Duration::from_secs(1) {
            println!("  buffer: {len}/{min_size}");
            last_print = Instant::now();
        }
        if len >= min_size {
            println!("Buffer ready ({len} positions). Starting training.");
            return;
        }
    }
}

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------

/// Run the outer training loop.
///
/// Phase 1 — fill: wait until the buffer has at least
/// `trainer.min_buffer_size()` positions (workers generate them in the
/// background).
///
/// Phase 2 — alternate:
///   1. Run `config.steps_per_broadcast` gradient steps.
///   2. Broadcast updated weights to `pool` (if provided).
///   3. Repeat until `config.total_steps` is reached or `shutdown` is set.
///
/// `pool` is `None` in tests or when running without workers.
pub fn run_loop<R: Rng>(
    trainer: &mut Trainer,
    buffer: Arc<Mutex<ReplayBuffer>>,
    pool: Option<&WorkerPool>,
    config: &OrchestrationConfig,
    rng: &mut R,
    shutdown: Arc<AtomicBool>,
) {
    wait_for_buffer(&buffer, trainer.min_buffer_size(), config.fill_poll_ms);

    loop {
        for _ in 0..config.steps_per_broadcast {
            if shutdown.load(Ordering::Relaxed) {
                return;
            }
            if config.total_steps > 0 && trainer.step >= config.total_steps {
                return;
            }
            trainer.train_step(&buffer, rng);
        }

        if let Some(p) = pool {
            p.broadcast_weights(&trainer.vs);
        }

        if shutdown.load(Ordering::Relaxed) {
            return;
        }
        if config.total_steps > 0 && trainer.step >= config.total_steps {
            return;
        }
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
    use tch::{Device, Tensor};

    use crate::nn::NUM_ACTIONS;
    use crate::train::TrainConfig;

    fn no_shutdown() -> Arc<AtomicBool> {
        Arc::new(AtomicBool::new(false))
    }

    fn small_trainer() -> Trainer {
        Trainer::new(
            Device::Cpu,
            8,
            2,
            TrainConfig {
                batch_size: 4,
                min_buffer_size: 4,
                log_every: 9999,
                ..TrainConfig::default()
            },
        )
    }

    fn filled_buffer(n: usize) -> Arc<Mutex<ReplayBuffer>> {
        let buf = Arc::new(Mutex::new(ReplayBuffer::new(100_000)));
        let mut b = buf.lock().unwrap();
        for i in 0..n {
            let board = Tensor::zeros([119, 9, 9], (tch::Kind::Float, Device::Cpu));
            let policy = vec![0.0f32; NUM_ACTIONS];
            let value = if i % 2 == 0 { 1.0f32 } else { -1.0 };
            b.push_game(vec![(board, policy, value)]);
        }
        drop(b);
        buf
    }

    fn fast_config(total_steps: u64, steps_per_broadcast: u64) -> OrchestrationConfig {
        OrchestrationConfig {
            total_steps,
            steps_per_broadcast,
            fill_poll_ms: 1,
        }
    }

    // ----- wait_for_buffer -----

    #[test]
    fn test_wait_for_buffer_returns_immediately_when_full() {
        let buf = filled_buffer(10);
        // Should return without sleeping — if it hangs, the test times out.
        wait_for_buffer(&buf, 10, 1);
    }

    #[test]
    fn test_wait_for_buffer_returns_when_threshold_met() {
        // Start with an empty buffer; a background thread fills it after 10 ms.
        let buf = Arc::new(Mutex::new(ReplayBuffer::new(100_000)));
        let buf2 = Arc::clone(&buf);
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(10));
            let board = Tensor::zeros([119, 9, 9], (tch::Kind::Float, Device::Cpu));
            let policy = vec![0.0f32; NUM_ACTIONS];
            buf2.lock().unwrap().push_game(vec![(board, policy, 1.0)]);
        });
        wait_for_buffer(&buf, 1, 2); // poll every 2ms
        assert!(buf.lock().unwrap().len() >= 1);
    }

    // ----- run_loop -----

    #[test]
    fn test_run_loop_runs_exact_total_steps() {
        let mut t = small_trainer();
        let buf = filled_buffer(t.min_buffer_size());
        let mut rng = StdRng::seed_from_u64(0);
        run_loop(
            &mut t,
            buf,
            None,
            &fast_config(7, 3),
            &mut rng,
            no_shutdown(),
        );
        assert_eq!(t.step, 7, "expected 7 steps, got {}", t.step);
    }

    #[test]
    fn test_run_loop_respects_shutdown() {
        let mut t = small_trainer();
        let buf = filled_buffer(t.min_buffer_size());
        let mut rng = StdRng::seed_from_u64(0);
        let shutdown = Arc::new(AtomicBool::new(false));

        // Signal shutdown before the loop even starts — it should run 0 steps.
        shutdown.store(true, Ordering::Relaxed);
        run_loop(
            &mut t,
            Arc::clone(&buf),
            None,
            &fast_config(100, 10),
            &mut rng,
            Arc::clone(&shutdown),
        );
        assert_eq!(t.step, 0, "shutdown-before-start should run 0 steps");
    }

    #[test]
    fn test_run_loop_steps_per_broadcast_alignment() {
        // With steps_per_broadcast=5 and total_steps=10, we get exactly 2 broadcasts.
        // Verified indirectly: step counter reaches 10.
        let mut t = small_trainer();
        let buf = filled_buffer(t.min_buffer_size());
        let mut rng = StdRng::seed_from_u64(0);
        run_loop(
            &mut t,
            buf,
            None,
            &fast_config(10, 5),
            &mut rng,
            no_shutdown(),
        );
        assert_eq!(t.step, 10);
    }

    #[test]
    fn test_run_loop_partial_last_broadcast_block() {
        // total_steps=7 with steps_per_broadcast=5: first block runs 5, second block
        // runs 2 (hits total_steps limit mid-block) → total = 7.
        let mut t = small_trainer();
        let buf = filled_buffer(t.min_buffer_size());
        let mut rng = StdRng::seed_from_u64(0);
        run_loop(
            &mut t,
            buf,
            None,
            &fast_config(7, 5),
            &mut rng,
            no_shutdown(),
        );
        assert_eq!(t.step, 7);
    }

    #[test]
    fn test_run_loop_zero_total_steps_runs_until_shutdown() {
        // total_steps=0 → run until shutdown.  We fire shutdown from a thread
        // after 20 ms so the loop doesn't hang forever.
        let mut t = small_trainer();
        let buf = filled_buffer(32); // plenty of data
        let mut rng = StdRng::seed_from_u64(0);
        let shutdown = Arc::new(AtomicBool::new(false));
        let sd2 = Arc::clone(&shutdown);
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            sd2.store(true, Ordering::Relaxed);
        });
        run_loop(&mut t, buf, None, &fast_config(0, 1), &mut rng, shutdown);
        assert!(
            t.step > 0,
            "should have run at least one step before shutdown"
        );
    }
}
