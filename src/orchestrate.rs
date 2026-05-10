/// Training orchestration: the outer loop that coordinates self-play workers,
/// the replay buffer, and the training thread.
///
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use rand::Rng;
use tch::Device;

use crate::metrics::{CheckpointEvent, JsonlWriter, MetricEvent, TrainEvent};
use crate::nn::checkpoint::build_with_config;
use crate::replay_buffer::ReplayBuffer;
use crate::selfplay::{SelfPlayConfig, play_pit_game};
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
    /// Save a checkpoint every this many steps (0 = never).
    pub checkpoint_every: u64,
    /// Directory where checkpoint files are written.
    pub checkpoint_dir: String,
    /// Number of pit games to play after each checkpoint (0 = skip).
    pub pit_games: u64,
}

impl Default for OrchestrationConfig {
    fn default() -> Self {
        OrchestrationConfig {
            steps_per_broadcast: 100,
            total_steps: 0,
            fill_poll_ms: 200,
            checkpoint_every: 1000,
            checkpoint_dir: "checkpoints".to_string(),
            pit_games: 100,
        }
    }
}

// ---------------------------------------------------------------------------
// Checkpointing
// ---------------------------------------------------------------------------

/// Return the path for a checkpoint at the given step.
pub fn checkpoint_path(dir: &str, step: u64) -> PathBuf {
    Path::new(dir).join(format!("step_{step:08}.ot"))
}

/// Parse the training step encoded in a checkpoint filename produced by `checkpoint_path`.
///
/// Expects filenames of the form `step_XXXXXXXX.ot` (8 zero-padded digits).
/// Returns `None` for any other filename format.
pub fn parse_step_from_filename(path: &Path) -> Option<u64> {
    let stem = path.file_stem()?.to_str()?;
    let digits = stem.strip_prefix("step_")?;
    digits.parse().ok()
}

/// Save a checkpoint if `step` just crossed a `checkpoint_every` boundary.
///
/// Returns the path of the written checkpoint, or `None` if no checkpoint was due.
/// Creates `dir` if it does not exist.
pub fn maybe_checkpoint(vs: &tch::nn::VarStore, step: u64, config: &OrchestrationConfig) -> Option<PathBuf> {
    if config.checkpoint_every == 0 || step == 0 {
        return None;
    }
    if step % config.checkpoint_every != 0 {
        return None;
    }
    let path = checkpoint_path(&config.checkpoint_dir, step);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("failed to create checkpoint dir");
    }
    vs.save(&path).expect("failed to save checkpoint");
    println!("Checkpoint saved: {}", path.display());
    Some(path)
}

// ---------------------------------------------------------------------------
// ELO estimation via pit games
// ---------------------------------------------------------------------------

/// Compute the expected ELO delta of the "new" player over "old" given win/draw/loss counts.
///
/// Uses the standard formula: Δ = 400 · log₁₀(score / (1 − score)),
/// where score = (wins + 0.5·draws) / total.
/// Returns 0.0 if total is zero or score would be 0 or 1 (avoid ±∞).
pub fn elo_delta(wins: u32, draws: u32, losses: u32) -> f64 {
    let total = (wins + draws + losses) as f64;
    if total == 0.0 {
        return 0.0;
    }
    let score = (wins as f64 + 0.5 * draws as f64) / total;
    if score <= 0.0 || score >= 1.0 {
        return if score >= 1.0 { f64::INFINITY } else { f64::NEG_INFINITY };
    }
    400.0 * (score / (1.0 - score)).log10()
}

/// A minimal `SelfPlayConfig` for pit games: greedy, no noise, fast.
fn pit_sp_config(num_simulations: u32) -> SelfPlayConfig {
    SelfPlayConfig {
        num_simulations,
        temperature_high: 0.0,
        temperature_drop_ply: 0,
        temperature_low: 0.0,
        dirichlet_epsilon: 0.0,
        resign_threshold: -0.95,
        resign_min_ply: 30,
        resign_consecutive: 5,
        max_moves: 512,
        ..SelfPlayConfig::default()
    }
}

/// Play `games` pit games between networks loaded from `new_path` and `old_path`.
///
/// Half the games have new as Black, half as White (to cancel first-mover bias).
/// Returns `(wins, draws, losses)` from the new network's perspective.
pub fn pit_networks<R: Rng>(
    new_path: &Path,
    old_path: &Path,
    channels: i64,
    blocks: usize,
    games: u64,
    device: Device,
    rng: &mut R,
) -> (u32, u32, u32) {
    let (mut new_vs, new_net) = build_with_config(device, channels, blocks);
    new_vs.load(new_path).expect("failed to load new checkpoint");

    let (mut old_vs, old_net) = build_with_config(device, channels, blocks);
    old_vs.load(old_path).expect("failed to load old checkpoint");

    let cfg = pit_sp_config(200);
    let half = games / 2;
    let mut wins = 0u32;
    let mut draws = 0u32;
    let mut losses = 0u32;

    // new as Black
    for _ in 0..half {
        let outcome = tch::no_grad(|| play_pit_game(&new_net, &old_net, &cfg, device, rng));
        match outcome.partial_cmp(&0.0) {
            Some(std::cmp::Ordering::Greater) => wins += 1,
            Some(std::cmp::Ordering::Less)    => losses += 1,
            _                                  => draws += 1,
        }
    }

    // new as White (outcome for Black = old's outcome → flip sign for new)
    for _ in 0..(games - half) {
        let outcome = tch::no_grad(|| play_pit_game(&old_net, &new_net, &cfg, device, rng));
        match outcome.partial_cmp(&0.0) {
            Some(std::cmp::Ordering::Less)    => wins += 1,   // old won as Black → new lost? no: outcome is Black's perspective; outcome < 0 means Black lost → new (White) won
            Some(std::cmp::Ordering::Greater) => losses += 1,
            _                                  => draws += 1,
        }
    }

    (wins, draws, losses)
}

/// Run a pit match and print the result if `config.pit_games > 0` and both paths are present.
pub fn pit_and_log<R: Rng>(
    new_path: &Path,
    old_path: &Path,
    trainer: &Trainer,
    config: &OrchestrationConfig,
    rng: &mut R,
) {
    if config.pit_games == 0 {
        return;
    }
    println!(
        "Pitting {} vs {} ({} games)…",
        new_path.display(),
        old_path.display(),
        config.pit_games
    );
    let (w, d, l) = pit_networks(
        new_path,
        old_path,
        trainer.channels,
        trainer.blocks,
        config.pit_games,
        trainer.device,
        rng,
    );
    let delta = elo_delta(w, d, l);
    println!(
        "Pit result: +{w}={d}-{l}  ELO Δ = {delta:+.1}"
    );
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
    mut metrics_writer: Option<&mut JsonlWriter>,
) {
    wait_for_buffer(&buffer, trainer.min_buffer_size(), config.fill_poll_ms);

    let started_at = Instant::now();
    let mut prev_ckpt: Option<PathBuf> = None;

    loop {
        for _ in 0..config.steps_per_broadcast {
            if shutdown.load(Ordering::Relaxed) {
                return;
            }
            if config.total_steps > 0 && trainer.step >= config.total_steps {
                return;
            }
            let metrics = trainer.train_step(&buffer, rng);
            if metrics.step == 1 || trainer.should_log(metrics.step) {
                if let Some(writer) = metrics_writer.as_deref_mut() {
                    let buffer_size = buffer.lock().unwrap().len();
                    writer
                        .write(&MetricEvent::Train(TrainEvent {
                            step: metrics.step,
                            wall_time_sec: started_at.elapsed().as_secs_f64(),
                            total_loss: metrics.total_loss,
                            policy_loss: metrics.policy_loss,
                            value_loss: metrics.value_loss,
                            buffer_size,
                            checkpoint_every: config.checkpoint_every,
                            batch_size: trainer.batch_size(),
                        }))
                        .expect("failed to write training metrics JSONL");
                }
            }
            if let Some(new_ckpt) = maybe_checkpoint(&trainer.vs, trainer.step, config) {
                if let Some(writer) = metrics_writer.as_deref_mut() {
                    writer
                        .write(&MetricEvent::Checkpoint(CheckpointEvent {
                            step: trainer.step,
                            path: new_ckpt.display().to_string(),
                            wall_time_sec: started_at.elapsed().as_secs_f64(),
                        }))
                        .expect("failed to write checkpoint metrics JSONL");
                }
                if let Some(ref old_ckpt) = prev_ckpt {
                    pit_and_log(&new_ckpt, old_ckpt, trainer, config, rng);
                }
                prev_ckpt = Some(new_ckpt);
            }
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
            checkpoint_every: 0, // disabled by default in fast tests
            checkpoint_dir: String::new(),
            pit_games: 0,
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
            None,
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
            None,
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
            None,
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
            None,
        );
        assert_eq!(t.step, 7);
    }

    #[test]
    fn test_run_loop_writes_train_metrics_jsonl_at_first_step_and_log_interval() {
        let mut t = Trainer::new(
            Device::Cpu,
            8,
            2,
            TrainConfig {
                batch_size: 4,
                min_buffer_size: 4,
                log_every: 2,
                ..TrainConfig::default()
            },
        );
        let buf = filled_buffer(t.min_buffer_size());
        let mut rng = StdRng::seed_from_u64(0);
        let dir = tempfile::tempdir().unwrap();
        let metrics_path = dir.path().join("metrics.jsonl");
        let mut metrics_writer = crate::metrics::JsonlWriter::new(&metrics_path).unwrap();

        run_loop(
            &mut t,
            buf,
            None,
            &fast_config(3, 1),
            &mut rng,
            no_shutdown(),
            Some(&mut metrics_writer),
        );

        let contents = std::fs::read_to_string(metrics_path).unwrap();
        let lines: Vec<_> = contents.lines().collect();
        assert_eq!(lines.len(), 2, "expected events at steps 1 and 2");

        let first: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(first["type"], "train");
        assert_eq!(first["step"], 1);
        assert_eq!(first["buffer_size"], 4);
        assert!(first["wall_time_sec"].as_f64().unwrap() >= 0.0);
        assert!(first["total_loss"].as_f64().unwrap().is_finite());
        assert!(first["policy_loss"].as_f64().unwrap().is_finite());
        assert!(first["value_loss"].as_f64().unwrap().is_finite());

        let second: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(second["type"], "train");
        assert_eq!(second["step"], 2);
        assert_eq!(second["batch_size"], 4);
        assert_eq!(second["checkpoint_every"], 0);
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
        run_loop(&mut t, buf, None, &fast_config(0, 1), &mut rng, shutdown, None);
        assert!(
            t.step > 0,
            "should have run at least one step before shutdown"
        );
    }

    // ----- checkpoint_path -----

    #[test]
    fn test_checkpoint_path_format() {
        let p = checkpoint_path("checkpoints", 1000);
        assert_eq!(p, PathBuf::from("checkpoints/step_00001000.ot"));
    }

    #[test]
    fn test_checkpoint_path_zero_padded() {
        let p = checkpoint_path("out/ckpt", 42);
        assert_eq!(p, PathBuf::from("out/ckpt/step_00000042.ot"));
    }

    // ----- parse_step_from_filename -----

    #[test]
    fn test_parse_step_roundtrip() {
        let path = checkpoint_path("checkpoints", 5000);
        assert_eq!(parse_step_from_filename(&path), Some(5000));
    }

    #[test]
    fn test_parse_step_leading_zeros() {
        let path = PathBuf::from("dir/step_00000042.ot");
        assert_eq!(parse_step_from_filename(&path), Some(42));
    }

    #[test]
    fn test_parse_step_unrecognised_filename() {
        let path = PathBuf::from("dir/model_final.ot");
        assert_eq!(parse_step_from_filename(&path), None);
    }

    #[test]
    fn test_parse_step_no_extension() {
        let path = PathBuf::from("step_00001000");
        // file_stem strips nothing useful here — digits are still parseable
        assert_eq!(parse_step_from_filename(&path), Some(1000));
    }

    // ----- maybe_checkpoint -----

    #[test]
    fn test_maybe_checkpoint_writes_file_at_boundary() {
        let t = small_trainer();
        let dir = tempfile::tempdir().unwrap();
        let cfg = OrchestrationConfig {
            checkpoint_every: 10,
            checkpoint_dir: dir.path().to_str().unwrap().to_string(),
            ..OrchestrationConfig::default()
        };
        let result = maybe_checkpoint(&t.vs, 10, &cfg);
        assert!(result.is_some(), "should return Some path at boundary");
        let expected = dir.path().join("step_00000010.ot");
        assert!(expected.exists(), "checkpoint file should exist at step 10");
    }

    #[test]
    fn test_maybe_checkpoint_no_file_between_boundaries() {
        let t = small_trainer();
        let dir = tempfile::tempdir().unwrap();
        let cfg = OrchestrationConfig {
            checkpoint_every: 10,
            checkpoint_dir: dir.path().to_str().unwrap().to_string(),
            ..OrchestrationConfig::default()
        };
        let result = maybe_checkpoint(&t.vs, 7, &cfg);
        assert!(result.is_none(), "should return None at non-boundary step");
        let not_expected = dir.path().join("step_00000007.ot");
        assert!(!not_expected.exists(), "should not write file at non-boundary step");
    }

    #[test]
    fn test_maybe_checkpoint_disabled_when_zero() {
        let t = small_trainer();
        let dir = tempfile::tempdir().unwrap();
        let cfg = OrchestrationConfig {
            checkpoint_every: 0,
            checkpoint_dir: dir.path().to_str().unwrap().to_string(),
            ..OrchestrationConfig::default()
        };
        let result = maybe_checkpoint(&t.vs, 1000, &cfg);
        assert!(result.is_none(), "should return None when checkpoint_every=0");
        assert_eq!(
            std::fs::read_dir(dir.path()).unwrap().count(),
            0,
            "no files should be written when checkpoint_every=0"
        );
    }

    #[test]
    fn test_run_loop_writes_checkpoint_at_boundary() {
        let mut t = small_trainer();
        let buf = filled_buffer(t.min_buffer_size());
        let mut rng = StdRng::seed_from_u64(0);
        let dir = tempfile::tempdir().unwrap();
        let cfg = OrchestrationConfig {
            total_steps: 10,
            steps_per_broadcast: 5,
            fill_poll_ms: 1,
            checkpoint_every: 5,
            checkpoint_dir: dir.path().to_str().unwrap().to_string(),
            pit_games: 0, // disabled: pit would be too slow for a unit test
        };
        run_loop(&mut t, buf, None, &cfg, &mut rng, no_shutdown(), None);
        assert_eq!(t.step, 10);
        assert!(dir.path().join("step_00000005.ot").exists(), "checkpoint at step 5");
        assert!(dir.path().join("step_00000010.ot").exists(), "checkpoint at step 10");
    }

    #[test]
    fn test_run_loop_writes_checkpoint_metrics_jsonl_at_boundary() {
        let mut t = small_trainer();
        let buf = filled_buffer(t.min_buffer_size());
        let mut rng = StdRng::seed_from_u64(0);
        let dir = tempfile::tempdir().unwrap();
        let metrics_path = dir.path().join("metrics.jsonl");
        let mut metrics_writer = crate::metrics::JsonlWriter::new(&metrics_path).unwrap();
        let cfg = OrchestrationConfig {
            total_steps: 2,
            steps_per_broadcast: 100,
            fill_poll_ms: 1,
            checkpoint_every: 1,
            checkpoint_dir: dir.path().join("checkpoints").to_str().unwrap().to_string(),
            pit_games: 0,
        };

        run_loop(
            &mut t,
            buf,
            None,
            &cfg,
            &mut rng,
            no_shutdown(),
            Some(&mut metrics_writer),
        );

        let contents = std::fs::read_to_string(metrics_path).unwrap();
        let events: Vec<serde_json::Value> = contents
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        let checkpoint_events: Vec<_> = events
            .iter()
            .filter(|event| event["type"] == "checkpoint")
            .collect();

        assert_eq!(checkpoint_events.len(), 2, "expected one checkpoint event per checkpoint");
        assert_eq!(checkpoint_events[0]["step"], 1);
        assert_eq!(checkpoint_events[1]["step"], 2);
        assert!(checkpoint_events[0]["path"].as_str().unwrap().ends_with("step_00000001.ot"));
        assert!(checkpoint_events[0]["wall_time_sec"].as_f64().unwrap() >= 0.0);
    }

    // ----- elo_delta -----

    #[test]
    fn test_elo_delta_even_score_is_zero() {
        // 50% score → Δ = 0
        let delta = elo_delta(5, 0, 5);
        assert!((delta).abs() < 1e-9, "even score should give Δ=0, got {delta}");
    }

    #[test]
    fn test_elo_delta_all_wins_is_positive_infinity() {
        let delta = elo_delta(10, 0, 0);
        assert!(delta.is_infinite() && delta > 0.0, "all wins → +∞");
    }

    #[test]
    fn test_elo_delta_all_losses_is_negative_infinity() {
        let delta = elo_delta(0, 0, 10);
        assert!(delta.is_infinite() && delta < 0.0, "all losses → −∞");
    }

    #[test]
    fn test_elo_delta_known_value() {
        // 75% score (3W 0D 1L): Δ = 400·log10(3) ≈ 190.8
        let delta = elo_delta(3, 0, 1);
        let expected = 400.0 * 3.0f64.log10();
        assert!(
            (delta - expected).abs() < 0.1,
            "expected {expected:.1}, got {delta:.1}"
        );
    }

    #[test]
    fn test_elo_delta_symmetry() {
        // Swapping W and L should negate the delta.
        let pos = elo_delta(7, 2, 3);
        let neg = elo_delta(3, 2, 7);
        assert!((pos + neg).abs() < 1e-9, "elo_delta should be antisymmetric");
    }

    #[test]
    fn test_elo_delta_zero_games_is_zero() {
        let delta = elo_delta(0, 0, 0);
        assert_eq!(delta, 0.0);
    }

    // ----- pit_networks -----

    #[test]
    fn test_pit_networks_counts_sum_to_games() {
        // Use 2 games (1 as Black, 1 as White) with identical networks —
        // result is unpredictable but counts must sum to 2.
        let t = small_trainer();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("net.ot");
        t.vs.save(&path).unwrap();

        let mut rng = StdRng::seed_from_u64(42);
        let (w, d, l) = pit_networks(&path, &path, t.channels, t.blocks, 2, Device::Cpu, &mut rng);
        assert_eq!(w + d + l, 2, "win+draw+loss must equal games");
    }
}
