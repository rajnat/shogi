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
import threading
from io import TextIOWrapper
from pathlib import Path
from typing import TextIO

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
# Subprocess launch with tee
# ---------------------------------------------------------------------------

def _tee(src: TextIO, *dests: TextIO) -> None:
    """Read lines from *src* and write every line to each destination."""
    for line in src:
        for dest in dests:
            dest.write(line)
            dest.flush()


def _env_with_libtorch() -> dict[str, str]:
    """Return os.environ extended with DYLD_LIBRARY_PATH for libtorch on macOS.

    On macOS, debug binaries built with tch-rs don't embed an rpath, so the
    dynamic linker can't find libtorch unless we point it explicitly.  We
    probe $LIBTORCH/lib and the tch-rs default (~/.local/lib) in addition to
    whatever DYLD_LIBRARY_PATH the caller already set.
    """
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


def launch(cmd: list[str], run_dir: Path) -> int:
    """Run *cmd*, tee-ing stdout/stderr to terminal and log files.

    Returns the process exit code.
    """
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"

    with open(stdout_path, "w") as fout, open(stderr_path, "w") as ferr:
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # line-buffered (requires text=True)
                env=_env_with_libtorch(),
            )
        except FileNotFoundError:
            msg = f"error: binary not found: {cmd[0]!r}\n"
            sys.stderr.write(msg)
            ferr.write(msg)
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

    print("\nLaunching training…\n")
    rc = launch(cmd, run_dir)

    if rc == 0:
        print(f"\nDone. Logs → {run_dir}/")
    else:
        print(f"\nFAILED (exit {rc}). Logs → {run_dir}/", file=sys.stderr)
    sys.exit(rc)


if __name__ == "__main__":
    main()
