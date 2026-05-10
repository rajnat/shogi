#!/usr/bin/env python3
"""Convert a YAML experiment config into a shogi train CLI command.

Usage (from project root):
    uv run --project experiments python experiments/run_experiment.py \\
        --config experiments/configs/smoke.yaml [--dry-run]
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, TextIO

import yaml

# ---------------------------------------------------------------------------
# YAML train-section key → CLI flag
# ---------------------------------------------------------------------------
_TRAIN_FLAGS: dict[str, str] = {
    "channels":             "--channels",
    "blocks":               "--blocks",
    "workers":              "--workers",
    "sims":                 "--sims",
    "batch_size":           "--batch-size",
    "min_buffer_size":      "--min-buffer-size",
    "learning_rate":        "--learning-rate",
    "weight_decay":         "--weight-decay",
    "c_puct":               "--c-puct",
    "dirichlet_alpha":      "--dirichlet-alpha",
    "dirichlet_epsilon":    "--dirichlet-epsilon",
    "temperature_high":     "--temperature-high",
    "temperature_low":      "--temperature-low",
    "temperature_drop_ply": "--temperature-drop-ply",
    "max_moves":            "--max-moves",
    "mcts_batch_size":      "--mcts-batch-size",
    "total_steps":          "--total-steps",
    "checkpoint_every":     "--checkpoint-every",
    "pit_games":            "--pit-games",
    "log_every":            "--log-every",
    "resign_threshold":     "--resign-threshold",
    "resign_min_ply":       "--resign-min-ply",
    "resign_consecutive":   "--resign-consecutive",
}


# ---------------------------------------------------------------------------
# Run-dir helpers
# ---------------------------------------------------------------------------

def make_run_dir(cfg: dict) -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    return Path("runs") / f"{ts}_{cfg['run_name']}_seed{cfg['seed']}"


def build_command(cfg: dict, run_dir: Path) -> list[str]:
    train = cfg.get("train", {})
    cmd: list[str] = [
        cfg["rust_binary"],
        "train",
        "--seed", str(cfg["seed"]),
        "--run-dir", str(run_dir),
        "--checkpoint-dir", str(run_dir / "checkpoints"),
    ]
    for key, flag in _TRAIN_FLAGS.items():
        if key in train:
            cmd += [flag, str(train[key])]
    if "resume" in cfg:
        cmd += ["--resume", str(cfg["resume"])]
    if "eval_anchor" in cfg:
        cmd += ["--eval-anchor", str(cfg["eval_anchor"])]
    return cmd


def format_command(cmd: list[str]) -> str:
    """Return a shell-pasteable, line-wrapped representation."""
    parts: list[str] = []
    i = 0
    while i < len(cmd):
        tok = cmd[i]
        if tok.startswith("--") and i + 1 < len(cmd) and not cmd[i + 1].startswith("--"):
            parts.append(f"{tok} {cmd[i + 1]}")
            i += 2
        else:
            parts.append(tok)
            i += 1
    return " \\\n  ".join(parts)


# ---------------------------------------------------------------------------
# JSONL tailing
# ---------------------------------------------------------------------------

def _tail_jsonl(
    path: Path,
    stop: threading.Event,
    on_event: Callable[[dict], None],
) -> None:
    """Tail *path* and call *on_event* for each complete JSON line.

    Handles:
    - File not yet existing: polls until it appears.
    - Partial writes: if readline() returns a line without a trailing newline
      the writer hasn't flushed yet; we seek back to the last good offset
      and retry after a short sleep.
    - Duplicate reads: a monotonically advancing offset prevents re-processing.
    """
    fh = None
    offset = 0
    try:
        while not stop.is_set():
            if fh is None:
                if not path.exists():
                    stop.wait(0.25)
                    continue
                fh = open(path)

            line = fh.readline()

            if not line:
                # Genuine EOF — nothing new yet; wait and try again.
                stop.wait(0.25)
                continue

            if not line.endswith("\n"):
                # Partial write: the writer is mid-flush.  Seek back and wait.
                fh.seek(offset)
                stop.wait(0.1)
                continue

            # Complete line: advance offset and dispatch.
            offset = fh.tell()
            stripped = line.strip()
            if stripped:
                try:
                    on_event(json.loads(stripped))
                except json.JSONDecodeError:
                    pass
    finally:
        if fh:
            fh.close()


# Shared print lock so tee threads and tail threads don't interleave lines.
_print_lock = threading.Lock()


def _locked_print(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def _on_metrics_event(event: dict) -> None:
    etype = event.get("type")
    step  = event.get("step", "?")
    if etype == "train":
        _locked_print(
            f"  \033[36m[metrics]\033[0m "
            f"step={step:>6}  "
            f"loss={event['total_loss']:.4f}  "
            f"policy={event['policy_loss']:.4f}  "
            f"value={event['value_loss']:.4f}  "
            f"buf={event['buffer_size']}"
        )
    elif etype == "checkpoint":
        _locked_print(
            f"  \033[33m[ckpt]\033[0m    "
            f"step={step:>6}  saved → {event.get('path', '')}"
        )


def _on_eval_event(event: dict) -> None:
    kind = event.get("opponent_kind", "?")
    step = event.get("step", "?")
    w, d, l = event["wins"], event["draws"], event["losses"]
    _locked_print(
        f"  \033[32m[eval/{kind}]\033[0m "
        f"step={step:>6}  "
        f"score={event['score']:.3f} "
        f"[{event['score_ci_low']:.3f},{event['score_ci_high']:.3f}]  "
        f"elo={event['elo_delta']:+.1f}  "
        f"+{w}={d}-{l}"
    )


# ---------------------------------------------------------------------------
# W&B metric logging
# ---------------------------------------------------------------------------

def _wb_log_metrics(event: dict, run) -> None:
    etype = event.get("type")
    step  = event.get("step")
    try:
        if etype == "train":
            run.log(
                {
                    "train/total_loss":    event["total_loss"],
                    "train/policy_loss":   event["policy_loss"],
                    "train/value_loss":    event["value_loss"],
                    "data/buffer_size":    event["buffer_size"],
                    "time/wall_time_sec":  event["wall_time_sec"],
                },
                step=step,
            )
        elif etype == "checkpoint":
            run.log({"checkpoint/step": step}, step=step)
            # Path is not a scalar; store it as a summary value.
            run.summary["checkpoint/latest_step"] = step
            run.summary["checkpoint/latest_path"] = event.get("path", "")
    except Exception as exc:  # noqa: BLE001
        _locked_print(f"  [warn] W&B log failed ({etype}): {exc}")


def _wb_log_eval(event: dict, run) -> None:
    kind = event.get("opponent_kind", "unknown")
    step = event.get("step")
    try:
        run.log(
            {
                f"eval/{kind}/score":      event["score"],
                f"eval/{kind}/elo_delta":  event["elo_delta"],
                f"eval/{kind}/elo_ci_low": event["elo_ci_low"],
                f"eval/{kind}/elo_ci_high":event["elo_ci_high"],
                f"eval/{kind}/wins":       event["wins"],
                f"eval/{kind}/draws":      event["draws"],
                f"eval/{kind}/losses":     event["losses"],
            },
            step=step,
        )
    except Exception as exc:  # noqa: BLE001
        _locked_print(f"  [warn] W&B eval log failed ({kind}): {exc}")


# ---------------------------------------------------------------------------
# W&B artifact upload
# ---------------------------------------------------------------------------

_LOG_FILES = [
    "resolved_config.yaml",
    "command.txt",
    "metrics.jsonl",
    "eval.jsonl",
    "stdout.log",
    "stderr.log",
]


def _find_best_checkpoint(run_dir: Path) -> Path | None:
    """Return the path of the best checkpoint inferred from eval.jsonl.

    The last eval event with opponent_kind="best" records the checkpoint that
    beat the previous best; its new_checkpoint field is the current best.
    Returns None when no such event exists or the file is absent.
    """
    eval_path = run_dir / "eval.jsonl"
    if not eval_path.exists():
        return None
    best: Path | None = None
    for line in eval_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
            if event.get("opponent_kind") == "best":
                candidate = Path(event["new_checkpoint"])
                if candidate.exists():
                    best = candidate
        except (json.JSONDecodeError, KeyError):
            pass
    return best


def _find_final_checkpoint(run_dir: Path) -> Path | None:
    """Return the highest-step checkpoint in run_dir/checkpoints/, or None."""
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None
    checkpoints = sorted(ckpt_dir.glob("step_*.ot"))
    return checkpoints[-1] if checkpoints else None


def upload_run_artifacts(run_dir: Path, wb_run, cfg: dict) -> None:
    """Upload selected run outputs to W&B as artifacts.

    Controlled by wandb.upload_artifacts and wandb.upload_checkpoints in cfg.
    Does nothing when upload_artifacts is falsy or wb_run is None.
    """
    if wb_run is None:
        return
    wb = cfg.get("wandb", {})
    if not wb.get("upload_artifacts", False):
        return

    import wandb

    # ---- Log files ----
    log_artifact = wandb.Artifact(
        name="run-logs",
        type="run-outputs",
        description=f"Logs and configs for run '{cfg.get('run_name', 'unknown')}'",
    )
    added = 0
    for fname in _LOG_FILES:
        path = run_dir / fname
        if path.exists() and path.stat().st_size > 0:
            log_artifact.add_file(str(path), name=fname)
            added += 1
    if added:
        wb_run.log_artifact(log_artifact)
        _locked_print(f"  [wandb] uploaded run-logs artifact ({added} files)")

    # ---- Checkpoints ----
    upload_ckpts = wb.get("upload_checkpoints", "none")
    if upload_ckpts == "none":
        return

    # final checkpoint
    final = _find_final_checkpoint(run_dir)
    if final:
        fa = wandb.Artifact(name="checkpoint-final", type="model")
        fa.add_file(str(final), name=final.name)
        wb_run.log_artifact(fa)
        _locked_print(f"  [wandb] uploaded checkpoint-final → {final.name}")

    # best checkpoint (only if distinct from final)
    best = _find_best_checkpoint(run_dir)
    if best and best != final:
        ba = wandb.Artifact(name="checkpoint-best", type="model")
        ba.add_file(str(best), name=best.name)
        wb_run.log_artifact(ba)
        _locked_print(f"  [wandb] uploaded checkpoint-best → {best.name}")


# ---------------------------------------------------------------------------
# W&B initialisation
# ---------------------------------------------------------------------------

def init_wandb(cfg: dict, run_dir: Path) -> "wandb.sdk.wandb_run.Run | None":
    """Initialise a W&B run if wandb.enabled is true in *cfg*.

    Returns the run object, or None when W&B is disabled.
    The run URL and ID are written to *run_dir*/wandb_run.txt for reference.
    """
    import os

    wb = cfg.get("wandb", {})
    if not wb.get("enabled", False):
        return None

    import wandb  # deferred — not required when disabled

    # WANDB_MODE env var takes precedence over the config value (standard W&B
    # convention), so honour it here too.
    effective_mode = os.environ.get("WANDB_MODE") or wb.get("mode", "online")

    # W&B ≥0.18 requires an API key even in offline mode.  Set a placeholder
    # so offline / CI runs don't error-out prompting for credentials.
    if effective_mode in ("offline", "disabled") and not os.environ.get("WANDB_API_KEY"):
        os.environ["WANDB_API_KEY"] = "local-xxx"

    run = wandb.init(
        project=wb.get("project", "shogi"),
        entity=wb.get("entity") or None,   # None falls back to personal account
        name=cfg.get("run_name"),
        group=cfg.get("group"),
        tags=wb.get("tags") or [],
        config=cfg,                         # full resolved config as hyperparams
        mode=effective_mode,
        dir=str(run_dir),
        resume="allow",
    )

    url = getattr(run, "url", None) or "(offline — sync with `wandb sync`)"
    print(f"W&B:      {url}")
    (run_dir / "wandb_run.txt").write_text(f"id:  {run.id}\nurl: {url}\n")
    return run


# ---------------------------------------------------------------------------
# Subprocess launch with tee + tail
# ---------------------------------------------------------------------------

def _tee(src: TextIO, *dests: TextIO) -> None:
    """Read lines from *src* and write every line to each destination."""
    for line in src:
        with _print_lock:
            for dest in dests:
                dest.write(line)
                dest.flush()


def _env_with_libtorch() -> dict[str, str]:
    """Return os.environ extended with DYLD_LIBRARY_PATH for libtorch on macOS."""
    import os
    env = os.environ.copy()
    candidates = [
        os.path.expanduser("~/libtorch/lib"),
        os.path.expanduser("~/.local/lib"),
    ]
    if libtorch := os.environ.get("LIBTORCH"):
        candidates.insert(0, str(Path(libtorch) / "lib"))
    extra = ":".join(c for c in candidates if Path(c).is_dir())
    if extra:
        existing = env.get("DYLD_LIBRARY_PATH", "")
        env["DYLD_LIBRARY_PATH"] = f"{extra}:{existing}" if existing else extra
    return env


def launch(cmd: list[str], run_dir: Path, wb_run=None) -> int:
    """Run *cmd*, tee-ing stdout/stderr to logs while tailing JSONL metrics.

    When *wb_run* is provided, each parsed JSONL event is also logged to W&B.
    """
    stdout_path  = run_dir / "stdout.log"
    stderr_path  = run_dir / "stderr.log"
    metrics_path = run_dir / "metrics.jsonl"
    eval_path    = run_dir / "eval.jsonl"

    stop = threading.Event()

    def on_metrics(event: dict) -> None:
        _on_metrics_event(event)
        if wb_run is not None:
            _wb_log_metrics(event, wb_run)

    def on_eval(event: dict) -> None:
        _on_eval_event(event)
        if wb_run is not None:
            _wb_log_eval(event, wb_run)

    # JSONL tail threads — start before the process so we catch the first write.
    t_metrics_tail = threading.Thread(
        target=_tail_jsonl,
        args=(metrics_path, stop, on_metrics),
        daemon=True,
    )
    t_eval_tail = threading.Thread(
        target=_tail_jsonl,
        args=(eval_path, stop, on_eval),
        daemon=True,
    )
    t_metrics_tail.start()
    t_eval_tail.start()

    with open(stdout_path, "w") as fout, open(stderr_path, "w") as ferr:
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=_env_with_libtorch(),
            )
        except FileNotFoundError:
            msg = f"error: binary not found: {cmd[0]!r}\n"
            sys.stderr.write(msg)
            ferr.write(msg)
            stop.set()
            return 127

        t_out = threading.Thread(
            target=_tee, args=(proc.stdout, sys.stdout, fout), daemon=True
        )
        t_err = threading.Thread(
            target=_tee, args=(proc.stderr, sys.stderr, ferr), daemon=True
        )
        t_out.start()
        t_err.start()

        try:
            t_out.join()
            t_err.join()
        except KeyboardInterrupt:
            proc.terminate()
            t_out.join()
            t_err.join()

        proc.wait()

    # Give tail threads time to drain any events written just before exit.
    time.sleep(0.5)
    stop.set()
    t_metrics_tail.join(timeout=3)
    t_eval_tail.join(timeout=3)

    return proc.returncode


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build (and optionally launch) a shogi training experiment."
    )
    parser.add_argument("--config", required=True, type=Path, metavar="YAML",
                        help="Experiment config file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Write files but do not launch training")
    args = parser.parse_args(argv)

    cfg: dict = yaml.safe_load(args.config.read_text())
    run_dir = make_run_dir(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_command(cfg, run_dir)
    cmd_pretty = format_command(cmd)

    (run_dir / "resolved_config.yaml").write_text(
        yaml.dump(cfg, default_flow_style=False, sort_keys=False)
    )
    (run_dir / "command.txt").write_text(cmd_pretty + "\n")

    print(f"run_dir:  {run_dir}")
    print(f"command:\n  {cmd_pretty}\n")
    print(f"wrote:    {run_dir}/resolved_config.yaml")
    print(f"wrote:    {run_dir}/command.txt")

    if args.dry_run:
        print("\n(dry run — training not launched)")
        return

    if cfg.get("build_release"):
        print("\nBuilding release binary…")
        rc = subprocess.run(["cargo", "build", "--release"]).returncode
        if rc != 0:
            print(f"\nFAILED: cargo build exited {rc}", file=sys.stderr)
            sys.exit(rc)

    wb_run = init_wandb(cfg, run_dir)

    print("\nLaunching training…\n")
    rc = launch(cmd, run_dir, wb_run=wb_run)

    if wb_run is not None:
        upload_run_artifacts(run_dir, wb_run, cfg)
        wb_run.finish(exit_code=rc)

    if rc == 0:
        print(f"\nDone. Logs → {run_dir}/")
    else:
        print(f"\nFAILED (exit {rc}). Logs → {run_dir}/", file=sys.stderr)
    sys.exit(rc)


if __name__ == "__main__":
    main()
