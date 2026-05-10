use serde::Serialize;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize)]
pub struct TrainMetricEvent {
    pub step: u64,
    pub game: u64,
    pub loss: f64,
    pub policy_loss: f64,
    pub value_loss: f64,
    pub learning_rate: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct TrainEvent {
    pub step: u64,
    pub wall_time_sec: f64,
    pub total_loss: f64,
    pub policy_loss: f64,
    pub value_loss: f64,
    pub buffer_size: usize,
    pub checkpoint_every: u64,
    pub batch_size: usize,
    // Self-play throughput counters (cumulative since run start).
    pub selfplay_games: u64,
    pub selfplay_positions: u64,
    // Per-second rates over the window since the previous log event.
    pub games_per_sec: f64,
    pub positions_per_sec: f64,
    // Outcome distribution (cumulative since run start).
    pub selfplay_black_wins: u64,
    pub selfplay_white_wins: u64,
    pub selfplay_draws: u64,
    pub selfplay_resigns: u64,
    pub selfplay_max_move_draws: u64,
    /// Mean game length in plies (positions / games).
    pub avg_game_length: f64,
    /// Mean Shannon entropy (nats) of the MCTS visit distribution, averaged across all games.
    pub avg_visit_entropy: f64,
    /// Mean Shannon entropy (nats) of the network's softmax policy at the root, averaged across all games.
    pub avg_policy_entropy: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CheckpointEvent {
    pub step: u64,
    pub path: String,
    pub wall_time_sec: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct EvalEvent {
    pub step: u64,
    pub new_checkpoint: String,
    pub opponent_checkpoint: String,
    pub opponent_kind: String,
    pub games: u64,
    pub wins: u32,
    pub draws: u32,
    pub losses: u32,
    pub score: f64,
    pub score_ci_low: f64,
    pub score_ci_high: f64,
    pub elo_delta: f64,
    pub elo_ci_low: f64,
    pub elo_ci_high: f64,
    pub wall_time_sec: f64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MetricEvent {
    #[serde(rename = "train")]
    Train(TrainEvent),
    TrainMetric(TrainMetricEvent),
    Checkpoint(CheckpointEvent),
    Eval(EvalEvent),
}

// ---------------------------------------------------------------------------
// Run configuration snapshot
// ---------------------------------------------------------------------------

/// Complete record of all hyperparameters for one training run.
///
/// Written to `<run_dir>/run_config.json` at startup so every output
/// directory is self-describing.
#[derive(Debug, Clone, Serialize)]
pub struct RunConfig {
    // Provenance
    pub version: String,
    pub unix_timestamp: u64,
    // Network architecture
    pub channels: i64,
    pub blocks: usize,
    // Parallelism / reproducibility
    pub workers: usize,
    pub seed: u64,
    // Output paths
    pub run_dir: String,
    pub metrics_jsonl: String,
    pub eval_jsonl: String,
    pub checkpoint_dir: String,
    // Training hyper-parameters
    pub batch_size: usize,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub min_buffer_size: usize,
    pub log_every: u64,
    // Orchestration
    pub checkpoint_every: u64,
    pub total_steps: u64,
    pub pit_games: u64,
    // Self-play / MCTS
    pub sims: u32,
    pub temperature_high: f32,
    pub temperature_low: f32,
    pub temperature_drop_ply: u32,
    pub dirichlet_alpha: f32,
    pub dirichlet_epsilon: f32,
    pub c_puct: f32,
    pub resign_threshold: f32,
    pub resign_min_ply: u32,
    pub resign_consecutive: u32,
    pub max_moves: usize,
    pub mcts_batch_size: usize,
    // Resume
    pub resume: Option<String>,
}

impl RunConfig {
    pub fn unix_now() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }
}

/// Serialize `config` as pretty-printed JSON and write it to `path`.
///
/// Creates parent directories if they do not exist.
pub fn write_run_config(path: &Path, config: &RunConfig) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(config).map_err(std::io::Error::other)?;
    std::fs::write(path, json)
}

pub struct JsonlWriter {
    writer: BufWriter<File>,
}

impl JsonlWriter {
    pub fn new(path: &Path) -> std::io::Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self {
            writer: BufWriter::new(file),
        })
    }

    pub fn write<T: Serialize>(&mut self, event: &T) -> std::io::Result<()> {
        serde_json::to_writer(&mut self.writer, event).map_err(std::io::Error::other)?;
        self.writer.write_all(b"\n")?;
        self.writer.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{CheckpointEvent, JsonlWriter, MetricEvent, RunConfig, TrainMetricEvent, write_run_config};
    use serde_json::Value;
    use std::fs;

    fn minimal_run_config() -> RunConfig {
        RunConfig {
            version: "0.1.0".to_string(),
            unix_timestamp: 1_700_000_000,
            channels: 64,
            blocks: 6,
            workers: 2,
            seed: 42,
            run_dir: "runs/test".to_string(),
            metrics_jsonl: "runs/test/metrics.jsonl".to_string(),
            eval_jsonl: "runs/test/eval.jsonl".to_string(),
            checkpoint_dir: "runs/test/checkpoints".to_string(),
            batch_size: 256,
            learning_rate: 1e-3,
            weight_decay: 1e-4,
            min_buffer_size: 5000,
            log_every: 50,
            checkpoint_every: 500,
            total_steps: 10_000,
            pit_games: 20,
            sims: 100,
            temperature_high: 1.0,
            temperature_low: 0.0,
            temperature_drop_ply: 30,
            dirichlet_alpha: 0.15,
            dirichlet_epsilon: 0.25,
            c_puct: 1.0,
            resign_threshold: -0.9,
            resign_min_ply: 30,
            resign_consecutive: 5,
            max_moves: 512,
            mcts_batch_size: 8,
            resume: None,
        }
    }

    #[test]
    fn run_config_is_valid_json() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("run_config.json");
        write_run_config(&path, &minimal_run_config()).unwrap();
        let contents = fs::read_to_string(&path).unwrap();
        let _: Value = serde_json::from_str(&contents).expect("run_config.json must be valid JSON");
    }

    #[test]
    fn run_config_contains_all_hyperparameter_fields() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("run_config.json");
        write_run_config(&path, &minimal_run_config()).unwrap();
        let v: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();

        let required = [
            "version", "unix_timestamp",
            "channels", "blocks", "workers", "seed",
            "run_dir", "metrics_jsonl", "eval_jsonl", "checkpoint_dir",
            "batch_size", "learning_rate", "weight_decay", "min_buffer_size", "log_every",
            "checkpoint_every", "total_steps", "pit_games",
            "sims", "temperature_high", "temperature_low", "temperature_drop_ply",
            "dirichlet_alpha", "dirichlet_epsilon", "c_puct",
            "resign_threshold", "resign_min_ply", "resign_consecutive",
            "max_moves", "mcts_batch_size", "resume",
        ];
        for field in required {
            assert!(v.get(field).is_some(), "missing field: {field}");
        }
    }

    #[test]
    fn run_config_values_are_correct() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("run_config.json");
        let cfg = minimal_run_config();
        write_run_config(&path, &cfg).unwrap();
        let v: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();

        assert_eq!(v["version"], "0.1.0");
        assert_eq!(v["unix_timestamp"], 1_700_000_000u64);
        assert_eq!(v["channels"], 64);
        assert_eq!(v["blocks"], 6);
        assert_eq!(v["seed"], 42);
        assert_eq!(v["batch_size"], 256);
        assert_eq!(v["total_steps"], 10_000u64);
        assert_eq!(v["sims"], 100);
        assert_eq!(v["resume"], Value::Null);
    }

    #[test]
    fn run_config_resume_path_is_serialized() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("run_config.json");
        let mut cfg = minimal_run_config();
        cfg.resume = Some("checkpoints/step_00001000.ot".to_string());
        write_run_config(&path, &cfg).unwrap();
        let v: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(v["resume"], "checkpoints/step_00001000.ot");
    }

    #[test]
    fn run_config_creates_parent_directories() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested").join("deep").join("run_config.json");
        write_run_config(&path, &minimal_run_config()).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn jsonl_writer_writes_one_valid_json_object_per_line() {
        let tempdir = tempfile::tempdir().unwrap();
        let metrics_path = tempdir.path().join("nested").join("metrics.jsonl");

        let mut writer = JsonlWriter::new(&metrics_path).unwrap();
        writer
            .write(&MetricEvent::TrainMetric(TrainMetricEvent {
                step: 1,
                game: 2,
                loss: 0.25,
                policy_loss: 0.125,
                value_loss: 0.0625,
                learning_rate: 0.001,
            }))
            .unwrap();
        writer
            .write(&MetricEvent::Checkpoint(CheckpointEvent {
                step: 1,
                path: "checkpoints/step-1.pt".to_string(),
                wall_time_sec: 0.5,
            }))
            .unwrap();

        let contents = fs::read_to_string(metrics_path).unwrap();
        let lines: Vec<_> = contents.lines().collect();
        assert_eq!(lines.len(), 2);

        let train_line: Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(train_line["type"], "train_metric");
        assert_eq!(train_line["step"], 1);
        assert_eq!(train_line["game"], 2);
        assert_eq!(train_line["loss"], 0.25);

        let checkpoint_line: Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(checkpoint_line["type"], "checkpoint");
        assert_eq!(checkpoint_line["step"], 1);
        assert_eq!(checkpoint_line["path"], "checkpoints/step-1.pt");
        assert_eq!(checkpoint_line["wall_time_sec"], 0.5);
    }
}
