"""Stream Shogi training JSONL files to Weights & Biases."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Iterator

import wandb


# ---------------------------------------------------------------------------
# JSONL streaming
# ---------------------------------------------------------------------------

def follow_jsonl(path: Path, from_start: bool = True) -> Iterator[dict]:
    """Yield JSON objects from *path*, blocking until new lines arrive.

    Set *from_start=False* to skip events already in the file and only
    forward new ones (useful when attaching to a run in progress).
    """
    with open(path) as fh:
        if not from_start:
            fh.seek(0, 2)
        while True:
            line = fh.readline()
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    pass  # guard against partially-written lines
            else:
                time.sleep(0.25)


# ---------------------------------------------------------------------------
# W&B dispatch
# ---------------------------------------------------------------------------

def _log_metrics_event(event: dict, run: wandb.sdk.wandb_run.Run) -> None:
    etype = event.get("type")
    step = event.get("step", 0)
    if etype == "train":
        run.log(
            {
                "train/loss": event["total_loss"],
                "train/policy_loss": event["policy_loss"],
                "train/value_loss": event["value_loss"],
                "train/buffer_size": event["buffer_size"],
                "train/wall_time_sec": event["wall_time_sec"],
            },
            step=step,
        )
    elif etype == "checkpoint":
        run.log({"checkpoint/saved_at_step": step}, step=step)


def _log_eval_event(event: dict, run: wandb.sdk.wandb_run.Run) -> None:
    kind = event.get("opponent_kind", "unknown")
    step = event.get("step", 0)
    run.log(
        {
            f"eval/{kind}/score": event["score"],
            f"eval/{kind}/score_ci_low": event["score_ci_low"],
            f"eval/{kind}/score_ci_high": event["score_ci_high"],
            f"eval/{kind}/elo_delta": event["elo_delta"],
            f"eval/{kind}/elo_ci_low": event["elo_ci_low"],
            f"eval/{kind}/elo_ci_high": event["elo_ci_high"],
            f"eval/{kind}/wins": event["wins"],
            f"eval/{kind}/draws": event["draws"],
            f"eval/{kind}/losses": event["losses"],
            f"eval/{kind}/games": event["games"],
        },
        step=step,
    )


# ---------------------------------------------------------------------------
# Main watch entry point
# ---------------------------------------------------------------------------

def watch_run(
    run_dir: Path,
    *,
    project: str = "shogi",
    entity: str | None = None,
    from_start: bool = True,
    log_fn=print,
) -> None:
    """Tail *run_dir* and stream all metric/eval events to W&B.

    Blocks until interrupted with Ctrl-C.  Both files are streamed
    concurrently via daemon threads so neither blocks the other.
    """
    run_dir = Path(run_dir)
    config: dict = {}
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())

    metrics_path = run_dir / "metrics.jsonl"
    eval_path    = run_dir / "eval.jsonl"

    for path in (metrics_path, eval_path):
        while not path.exists():
            log_fn(f"Waiting for {path} to appear…")
            time.sleep(2)

    log_fn(f"Initialising W&B run (project={project!r}, dir={run_dir})…")
    wb_run = wandb.init(
        project=project,
        entity=entity,
        config=config,
        dir=str(run_dir),
        resume="allow",
    )
    log_fn(f"W&B run URL: {wb_run.get_url()}")

    def _stream(path: Path, dispatch):
        for event in follow_jsonl(path, from_start=from_start):
            try:
                dispatch(event, wb_run)
            except Exception as exc:  # noqa: BLE001
                log_fn(f"[warn] failed to log event from {path.name}: {exc}")

    t_metrics = threading.Thread(
        target=_stream, args=(metrics_path, _log_metrics_event), daemon=True
    )
    t_eval = threading.Thread(
        target=_stream, args=(eval_path, _log_eval_event), daemon=True
    )
    t_metrics.start()
    t_eval.start()
    log_fn("Streaming — press Ctrl-C to stop.")

    try:
        while t_metrics.is_alive() or t_eval.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        log_fn("\nStopped.")
    finally:
        wb_run.finish()
