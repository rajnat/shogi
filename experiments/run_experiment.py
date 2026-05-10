#!/usr/bin/env python3
"""Convert a YAML experiment config into a shogi train CLI command.

Usage (from project root):
    uv run --project experiments python experiments/run_experiment.py \\
        --config experiments/configs/smoke.yaml [--dry-run]
"""

from __future__ import annotations

import argparse
import datetime
import subprocess
import sys
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# YAML train-section key → CLI flag
# All keys are optional; only present ones are emitted.
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


def make_run_dir(cfg: dict) -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    return Path("runs") / f"{ts}_{cfg['run_name']}_seed{cfg['seed']}"


def build_command(cfg: dict, run_dir: Path) -> list[str]:
    train = cfg.get("train", {})
    checkpoint_dir = run_dir / "checkpoints"

    cmd: list[str] = [
        cfg["rust_binary"],
        "train",
        "--seed", str(cfg["seed"]),
        "--run-dir", str(run_dir),
        "--checkpoint-dir", str(checkpoint_dir),
    ]

    for key, flag in _TRAIN_FLAGS.items():
        if key in train:
            cmd += [flag, str(train[key])]

    # Top-level optional passthrough fields
    if "resume" in cfg:
        cmd += ["--resume", str(cfg["resume"])]
    if "eval_anchor" in cfg:
        cmd += ["--eval-anchor", str(cfg["eval_anchor"])]

    return cmd


def format_command(cmd: list[str]) -> str:
    """Return a human-readable, shell-pasteable representation."""
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build (and optionally launch) a shogi training experiment."
    )
    parser.add_argument("--config", required=True, type=Path, metavar="YAML",
                        help="Experiment config file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Create run dir and write files, but do not launch training")
    args = parser.parse_args(argv)

    cfg: dict = yaml.safe_load(args.config.read_text())
    run_dir = make_run_dir(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_command(cfg, run_dir)
    cmd_pretty = format_command(cmd)

    # Persist resolved config and command for reproducibility.
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
        subprocess.run(["cargo", "build", "--release"], check=True)

    print("\nLaunching training…")
    sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
