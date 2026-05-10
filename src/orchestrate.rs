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
    /// Fixed anchor checkpoint to evaluate every new checkpoint against.
    /// When set, emits eval events with opponent_kind = "anchor" and tracks
    /// the best-so-far checkpoint by anchor score.
    pub eval_anchor: Option<PathBuf>,
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
            eval_anchor: None,
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

/// Score rate: (wins + 0.5·draws) / total.  Returns 0.0 for zero games.
pub fn score_rate(wins: u32, draws: u32, losses: u32) -> f64 {
    let total = (wins + draws + losses) as f64;
    if total == 0.0 {
        return 0.0;
    }
    (wins as f64 + 0.5 * draws as f64) / total
}

/// 95% normal-approximate confidence interval for the score rate.
///
/// Uses the standard error `se = sqrt(s·(1−s)/n)` and returns
/// `(s − 1.96·se, s + 1.96·se)` clamped to `[0, 1]`.
/// Returns `(0.0, 1.0)` (maximum uncertainty) when total is zero.
pub fn score_ci95(wins: u32, draws: u32, losses: u32) -> (f64, f64) {
    let total = (wins + draws + losses) as f64;
    if total == 0.0 {
        return (0.0, 1.0);
    }
    let s = score_rate(wins, draws, losses);
    let se = (s * (1.0 - s) / total).sqrt();
    let low  = (s - 1.96 * se).clamp(0.0, 1.0);
    let high = (s + 1.96 * se).clamp(0.0, 1.0);
    (low, high)
}

/// Convert a score in (0, 1) to an ELO delta using the logistic formula.
/// Returns ±∞ at the boundary; callers should clamp as needed.
fn elo_from_score(score: f64) -> f64 {
    if score <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if score >= 1.0 {
        return f64::INFINITY;
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

/// Run a pit match, print the result, and return a structured `EvalEvent`.
///
/// `opponent_kind` is written verbatim into the event (e.g. "previous", "anchor", "best").
/// Returns `None` when `config.pit_games == 0` (pit disabled).
pub fn pit_and_log<R: Rng>(
    new_path: &Path,
    old_path: &Path,
    trainer: &Trainer,
    config: &OrchestrationConfig,
    rng: &mut R,
    opponent_kind: &str,
    wall_time_sec: f64,
) -> Option<crate::metrics::EvalEvent> {
    if config.pit_games == 0 {
        return None;
    }
    println!(
        "Pitting {} vs {} [{}] ({} games)…",
        new_path.display(),
        old_path.display(),
        opponent_kind,
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
    let score = score_rate(w, d, l);
    let (ci_low, ci_high) = score_ci95(w, d, l);
    let delta      = elo_delta(w, d, l).clamp(-800.0, 800.0);
    let elo_ci_low  = elo_from_score(ci_low).clamp(-800.0, 800.0);
    let elo_ci_high = elo_from_score(ci_high).clamp(-800.0, 800.0);
    println!(
        "Pit [{opponent_kind}]: +{w}={d}-{l}  score={score:.3} [{ci_low:.3},{ci_high:.3}]  \
         ELO Δ = {delta:+.1} [{elo_ci_low:+.1},{elo_ci_high:+.1}]"
    );
    Some(crate::metrics::EvalEvent {
        step: trainer.step,
        new_checkpoint: new_path.display().to_string(),
        opponent_checkpoint: old_path.display().to_string(),
        opponent_kind: opponent_kind.to_string(),
        games: config.pit_games,
        wins: w,
        draws: d,
        losses: l,
        score,
        score_ci_low: ci_low,
        score_ci_high: ci_high,
        elo_delta: delta,
        elo_ci_low,
        elo_ci_high,
        wall_time_sec,
    })
}

// ---------------------------------------------------------------------------
// Best-checkpoint tracking
// ---------------------------------------------------------------------------

/// Return `true` if `current_score` is strictly better than `best_score`,
/// meaning the current checkpoint should become the new best.
pub fn should_update_best(current_score: f64, best_score: f64) -> bool {
    current_score > best_score
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
    mut eval_writer: Option<&mut JsonlWriter>,
) {
    wait_for_buffer(&buffer, trainer.min_buffer_size(), config.fill_poll_ms);

    let started_at = Instant::now();
    let mut prev_ckpt: Option<PathBuf> = None;
    // Best checkpoint by anchor score; only tracked when eval_anchor is configured.
    let mut best_ckpt: Option<PathBuf> = None;
    let mut best_score: f64 = -1.0;
    // Throughput tracking: snapshot at the previous log event for window rates.
    let mut last_log_wall_sec: f64 = 0.0;
    let mut last_log_games: u64 = 0;
    let mut last_log_positions: u64 = 0;

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
                    let wall_time_sec = started_at.elapsed().as_secs_f64();
                    let (total_games, total_positions) =
                        pool.map_or((0, 0), |p| p.counters());
                    let dt = (wall_time_sec - last_log_wall_sec).max(1e-9);
                    let games_per_sec = (total_games - last_log_games) as f64 / dt;
                    let positions_per_sec = (total_positions - last_log_positions) as f64 / dt;
                    last_log_wall_sec = wall_time_sec;
                    last_log_games = total_games;
                    last_log_positions = total_positions;
                    writer
                        .write(&MetricEvent::Train(TrainEvent {
                            step: metrics.step,
                            wall_time_sec,
                            total_loss: metrics.total_loss,
                            policy_loss: metrics.policy_loss,
                            value_loss: metrics.value_loss,
                            buffer_size,
                            checkpoint_every: config.checkpoint_every,
                            batch_size: trainer.batch_size(),
                            selfplay_games: total_games,
                            selfplay_positions: total_positions,
                            games_per_sec,
                            positions_per_sec,
                        }))
                        .expect("failed to write training metrics JSONL");
                }
            }
            if let Some(new_ckpt) = maybe_checkpoint(&trainer.vs, trainer.step, config) {
                let wt = started_at.elapsed().as_secs_f64();
                if let Some(writer) = metrics_writer.as_deref_mut() {
                    writer
                        .write(&MetricEvent::Checkpoint(CheckpointEvent {
                            step: trainer.step,
                            path: new_ckpt.display().to_string(),
                            wall_time_sec: wt,
                        }))
                        .expect("failed to write checkpoint metrics JSONL");
                }

                // Pit vs previous checkpoint.
                if let Some(ref old_ckpt) = prev_ckpt {
                    if let Some(event) = pit_and_log(&new_ckpt, old_ckpt, trainer, config, rng, "previous", wt) {
                        if let Some(writer) = eval_writer.as_deref_mut() {
                            writer.write(&event).expect("failed to write eval JSONL");
                        }
                    }
                }

                // Pit vs anchor (if configured) and update best-so-far.
                if let Some(ref anchor) = config.eval_anchor.clone() {
                    if let Some(event) = pit_and_log(&new_ckpt, anchor, trainer, config, rng, "anchor", wt) {
                        if let Some(writer) = eval_writer.as_deref_mut() {
                            writer.write(&event).expect("failed to write eval JSONL");
                        }
                        if should_update_best(event.score, best_score) {
                            // Pit vs current best before replacing it.
                            if let Some(ref old_best) = best_ckpt.clone() {
                                if let Some(best_event) = pit_and_log(&new_ckpt, old_best, trainer, config, rng, "best", wt) {
                                    if let Some(writer) = eval_writer.as_deref_mut() {
                                        writer.write(&best_event).expect("failed to write eval JSONL");
                                    }
                                }
                            }
                            best_ckpt = Some(new_ckpt.clone());
                            best_score = event.score;
                        }
                    }
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
            eval_anchor: None,
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
            None,
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
        run_loop(&mut t, buf, None, &fast_config(0, 1), &mut rng, shutdown, None, None);
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
            eval_anchor: None,
        };
        run_loop(&mut t, buf, None, &cfg, &mut rng, no_shutdown(), None, None);
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
            eval_anchor: None,
        };

        run_loop(
            &mut t,
            buf,
            None,
            &cfg,
            &mut rng,
            no_shutdown(),
            Some(&mut metrics_writer),
            None,
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

    // ----- eval JSONL -----

    #[test]
    fn test_run_loop_writes_eval_event_after_second_checkpoint() {
        // checkpoint_every=1, pit_games=2, total_steps=2:
        // step 1 → first checkpoint (no previous → no pit)
        // step 2 → second checkpoint (previous exists → pit fires → eval event)
        let mut t = small_trainer();
        let buf = filled_buffer(t.min_buffer_size());
        let mut rng = StdRng::seed_from_u64(0);
        let dir = tempfile::tempdir().unwrap();
        let eval_path = dir.path().join("eval.jsonl");
        let mut eval_writer = crate::metrics::JsonlWriter::new(&eval_path).unwrap();
        let cfg = OrchestrationConfig {
            total_steps: 2,
            steps_per_broadcast: 100,
            fill_poll_ms: 1,
            checkpoint_every: 1,
            checkpoint_dir: dir.path().join("checkpoints").to_str().unwrap().to_string(),
            pit_games: 2,
            eval_anchor: None,
        };

        run_loop(
            &mut t,
            buf,
            None,
            &cfg,
            &mut rng,
            no_shutdown(),
            None,
            Some(&mut eval_writer),
        );

        let contents = std::fs::read_to_string(&eval_path).unwrap();
        let events: Vec<serde_json::Value> = contents
            .lines()
            .filter(|l| !l.is_empty())
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();

        assert_eq!(events.len(), 1, "exactly one eval event (step 2 vs step 1)");
        let ev = &events[0];
        assert_eq!(ev["step"], 2);
        assert_eq!(ev["opponent_kind"], "previous");
        assert_eq!(ev["games"], 2);
        let w = ev["wins"].as_u64().unwrap();
        let d = ev["draws"].as_u64().unwrap();
        let l = ev["losses"].as_u64().unwrap();
        assert_eq!(w + d + l, 2);
        assert!(ev["score"].as_f64().unwrap() >= 0.0);
        assert!(ev["score_ci_low"].as_f64().is_some());
        assert!(ev["score_ci_high"].as_f64().is_some());
        assert!(ev["elo_delta"].as_f64().is_some());
        assert!(ev["elo_ci_low"].as_f64().is_some());
        assert!(ev["elo_ci_high"].as_f64().is_some());
        assert!(ev["wall_time_sec"].as_f64().unwrap() >= 0.0);
        assert!(ev["new_checkpoint"].as_str().unwrap().ends_with("step_00000002.ot"));
        assert!(ev["opponent_checkpoint"].as_str().unwrap().ends_with("step_00000001.ot"));
    }

    // ----- score_rate -----

    #[test]
    fn test_score_rate_zero_games() {
        assert_eq!(score_rate(0, 0, 0), 0.0);
    }

    #[test]
    fn test_score_rate_all_wins() {
        assert_eq!(score_rate(10, 0, 0), 1.0);
    }

    #[test]
    fn test_score_rate_all_losses() {
        assert_eq!(score_rate(0, 0, 10), 0.0);
    }

    #[test]
    fn test_score_rate_all_draws() {
        assert!((score_rate(0, 10, 0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_score_rate_mixed() {
        // 3W 2D 5L → (3 + 1) / 10 = 0.4
        assert!((score_rate(3, 2, 5) - 0.4).abs() < 1e-10);
    }

    // ----- score_ci95 -----

    #[test]
    fn test_score_ci95_zero_games_returns_maximum_uncertainty() {
        let (lo, hi) = score_ci95(0, 0, 0);
        assert_eq!(lo, 0.0);
        assert_eq!(hi, 1.0);
    }

    #[test]
    fn test_score_ci95_is_within_unit_interval() {
        for (w, d, l) in [(0u32,0,1),(1,0,0),(0,10,0),(5,0,5),(100,0,0),(0,0,100)] {
            let (lo, hi) = score_ci95(w, d, l);
            assert!(lo >= 0.0 && lo <= 1.0, "lo={lo} out of [0,1] for ({w},{d},{l})");
            assert!(hi >= 0.0 && hi <= 1.0, "hi={hi} out of [0,1] for ({w},{d},{l})");
            assert!(lo <= hi, "lo > hi for ({w},{d},{l})");
        }
    }

    #[test]
    fn test_score_ci95_symmetric_at_half() {
        // Equal wins and losses → score = 0.5; CI should be symmetric around 0.5.
        let (lo, hi) = score_ci95(50, 0, 50);
        assert!((lo + hi - 1.0).abs() < 1e-10, "CI not symmetric: [{lo:.4},{hi:.4}]");
    }

    #[test]
    fn test_score_ci95_narrows_with_more_games() {
        let (lo10, hi10)     = score_ci95(5, 0, 5);
        let (lo1000, hi1000) = score_ci95(500, 0, 500);
        assert!(hi1000 - lo1000 < hi10 - lo10, "CI should narrow with more games");
    }

    #[test]
    fn test_score_ci95_all_wins_clamped() {
        let (lo, hi) = score_ci95(10, 0, 0);
        assert_eq!(lo, 1.0, "lower bound for all-wins should clamp to 1.0");
        assert_eq!(hi, 1.0, "upper bound for all-wins should clamp to 1.0");
    }

    #[test]
    fn test_score_ci95_all_losses_clamped() {
        let (lo, hi) = score_ci95(0, 0, 10);
        assert_eq!(lo, 0.0);
        assert_eq!(hi, 0.0);
    }

    // ----- elo CI via pit_and_log -----

    #[test]
    fn test_elo_ci_bounds_ordered() {
        // The ELO CI endpoints must satisfy low ≤ elo_delta ≤ high.
        let t = small_trainer();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("net.ot");
        t.vs.save(&path).unwrap();

        let cfg = OrchestrationConfig {
            pit_games: 10,
            checkpoint_every: 0,
            checkpoint_dir: String::new(),
            ..OrchestrationConfig::default()
        };
        let mut rng = StdRng::seed_from_u64(7);
        if let Some(ev) = pit_and_log(&path, &path, &t, &cfg, &mut rng, "previous", 0.0) {
            assert!(
                ev.elo_ci_low <= ev.elo_delta,
                "elo_ci_low ({:.1}) > elo_delta ({:.1})",
                ev.elo_ci_low, ev.elo_delta
            );
            assert!(
                ev.elo_delta <= ev.elo_ci_high,
                "elo_delta ({:.1}) > elo_ci_high ({:.1})",
                ev.elo_delta, ev.elo_ci_high
            );
            assert!(ev.score_ci_low <= ev.score);
            assert!(ev.score <= ev.score_ci_high);
        }
    }

    // ----- pit_and_log opponent_kind -----

    #[test]
    fn test_pit_and_log_records_opponent_kind() {
        let t = small_trainer();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("net.ot");
        t.vs.save(&path).unwrap();

        let cfg = OrchestrationConfig {
            pit_games: 2,
            checkpoint_every: 0,
            checkpoint_dir: String::new(),
            ..OrchestrationConfig::default()
        };
        let mut rng = StdRng::seed_from_u64(99);
        let ev = pit_and_log(&path, &path, &t, &cfg, &mut rng, "anchor", 1.5)
            .expect("pit_and_log should return Some when pit_games > 0");
        assert_eq!(ev.opponent_kind, "anchor");
        assert_eq!(ev.wall_time_sec, 1.5);
    }

    // ----- should_update_best -----

    #[test]
    fn test_should_update_best_when_strictly_better() {
        assert!(should_update_best(0.6, 0.5));
        assert!(should_update_best(0.5, -1.0)); // first checkpoint: best_score sentinel = -1
        assert!(should_update_best(1.0, 0.999));
    }

    #[test]
    fn test_should_update_best_when_equal_or_worse() {
        assert!(!should_update_best(0.5, 0.5));  // equal: no update
        assert!(!should_update_best(0.4, 0.5));  // worse: no update
        assert!(!should_update_best(0.0, 0.0));
    }

    // ----- anchor eval events in run_loop -----

    #[test]
    fn test_run_loop_writes_anchor_eval_events() {
        // Saves a network as the "anchor" before training starts, then
        // runs 2 steps with checkpoint_every=1.  Each checkpoint should
        // emit an "anchor" eval event in eval.jsonl.
        let mut t = small_trainer();
        let buf = filled_buffer(t.min_buffer_size());
        let mut rng = StdRng::seed_from_u64(0);
        let dir = tempfile::tempdir().unwrap();

        // Anchor = random-weight network saved before the loop.
        let anchor_path = dir.path().join("anchor.ot");
        t.vs.save(&anchor_path).unwrap();

        let eval_path = dir.path().join("eval.jsonl");
        let mut eval_writer = crate::metrics::JsonlWriter::new(&eval_path).unwrap();
        let cfg = OrchestrationConfig {
            total_steps: 2,
            steps_per_broadcast: 100,
            fill_poll_ms: 1,
            checkpoint_every: 1,
            checkpoint_dir: dir.path().join("checkpoints").to_str().unwrap().to_string(),
            pit_games: 2,
            eval_anchor: Some(anchor_path),
        };

        run_loop(&mut t, buf, None, &cfg, &mut rng, no_shutdown(), None, Some(&mut eval_writer));

        let contents = std::fs::read_to_string(&eval_path).unwrap();
        let events: Vec<serde_json::Value> = contents
            .lines()
            .filter(|l| !l.is_empty())
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();

        let anchor_events: Vec<_> = events.iter()
            .filter(|e| e["opponent_kind"] == "anchor")
            .collect();
        // Step 1 → anchor pit; step 2 → anchor pit (+ possibly "best" pit).
        assert!(
            anchor_events.len() >= 1,
            "expected at least one anchor eval event, got {}; events: {events:?}",
            anchor_events.len()
        );
        for ev in &anchor_events {
            assert_eq!(ev["opponent_kind"], "anchor");
            let w = ev["wins"].as_u64().unwrap();
            let d = ev["draws"].as_u64().unwrap();
            let l = ev["losses"].as_u64().unwrap();
            assert_eq!(w + d + l, 2, "game counts must sum to pit_games");
        }
    }
}
