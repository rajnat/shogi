use serde::Serialize;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;

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
    pub games: u64,
    pub win_rate: f64,
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
    use super::{CheckpointEvent, JsonlWriter, MetricEvent, TrainMetricEvent};
    use serde_json::Value;
    use std::fs;

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
