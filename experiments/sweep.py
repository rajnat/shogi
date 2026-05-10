#!/usr/bin/env python3
"""Run a grid of experiments from a YAML sweep config.

Each combination of parameter values is crossed with every seed, producing
one run per (combo, seed) pair.  Runs execute sequentially so the Rust
binary can use all available CPU cores.

Usage (from project root):
    uv run --project experiments python experiments/sweep.py \\
        --config experiments/configs/cpuct_sweep.yaml [--dry-run]

Sweep config format:
    base_config: experiments/configs/baseline_fast.yaml
    sweep_name: cpuct_fast
    seeds: [1, 2, 3]
    parameters:
      train.c_puct: [0.5, 1.0, 1.5, 2.5]

Keys in 'parameters' use dot-notation to address nested config fields.
All value lists are crossed via Cartesian product.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from run_experiment import (
    make_run_dir,
    build_command,
    format_command,
    launch,
    init_wandb,
    upload_run_artifacts,
    _make_plateau_detector,
    _env_with_libtorch,
)


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def _set_nested(cfg: dict, dotkey: str, value: Any) -> dict:
    """Return a copy of *cfg* with the dot-separated *dotkey* set to *value*."""
    head, _, tail = dotkey.partition(".")
    result = dict(cfg)
    if not tail:
        result[head] = value
    else:
        result[head] = _set_nested(dict(cfg.get(head, {})), tail, value)
    return result


def expand_grid(parameters: dict[str, list]) -> list[dict]:
    """Return the Cartesian product of all parameter value lists.

    >>> expand_grid({"a": [1, 2], "b": ["x", "y"]})
    [{"a": 1, "b": "x"}, {"a": 1, "b": "y"}, {"a": 2, "b": "x"}, {"a": 2, "b": "y"}]
    """
    if not parameters:
        return [{}]
    keys = list(parameters.keys())
    return [
        dict(zip(keys, combo))
        for combo in itertools.product(*(parameters[k] for k in keys))
    ]


def _combo_tag(combo: dict) -> str:
    """Short human-readable label for one parameter combination."""
    parts = []
    for dotkey, val in combo.items():
        short = dotkey.rsplit(".", 1)[-1]
        val_str = f"{val:g}" if isinstance(val, float) else str(val)
        parts.append(f"{short}{val_str}")
    return "_".join(parts)


def apply_combo(base_cfg: dict, combo: dict, seed: int, run_name: str) -> dict:
    """Return a deep copy of *base_cfg* with *combo* overrides, *seed*, and *run_name*."""
    cfg = copy.deepcopy(base_cfg)
    for dotkey, value in combo.items():
        cfg = _set_nested(cfg, dotkey, value)
    cfg["seed"] = seed
    cfg["run_name"] = run_name
    return cfg


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run a Cartesian grid of shogi training experiments."
    )
    parser.add_argument("--config", required=True, type=Path, metavar="YAML",
                        help="Sweep config file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print resolved configs without launching training")
    parser.add_argument("--max-runtime-sec", type=float, default=None, metavar="SEC",
                        help="Per-run time cap passed to each launch()")
    args = parser.parse_args(argv)

    sweep_cfg: dict = yaml.safe_load(args.config.read_text())
    base_cfg: dict  = yaml.safe_load(Path(sweep_cfg["base_config"]).read_text())

    sweep_name: str       = sweep_cfg.get("sweep_name", "sweep")
    seeds: list[int]      = sweep_cfg.get("seeds", [base_cfg.get("seed", 42)])
    parameters: dict      = sweep_cfg.get("parameters", {})

    combos = expand_grid(parameters)
    runs   = [(combo, seed) for combo in combos for seed in seeds]
    total  = len(runs)

    print(
        f"Sweep: {sweep_name!r}  "
        f"{len(combos)} combo(s) × {len(seeds)} seed(s) = {total} run(s)"
    )
    print(f"Base:  {sweep_cfg['base_config']}\n")

    # Build release binary once up front rather than for each individual run.
    if base_cfg.get("build_release") and not args.dry_run:
        print("Building release binary…")
        rc = subprocess.run(
            ["cargo", "build", "--release"],
            env=_env_with_libtorch(),
        ).returncode
        if rc != 0:
            print(f"FAILED: cargo build exited {rc}", file=sys.stderr)
            sys.exit(rc)

    failed: list[str] = []

    for i, (combo, seed) in enumerate(runs, 1):
        tag      = _combo_tag(combo) if combo else "default"
        run_name = f"{sweep_name}_{tag}"
        cfg      = apply_combo(base_cfg, combo, seed, run_name)
        cfg["build_release"] = False   # already built above

        run_dir     = make_run_dir(cfg)
        cmd         = build_command(cfg, run_dir)
        cmd_pretty  = format_command(cmd)

        print(f"[{i}/{total}] {run_name}")
        print(f"  params:  {combo}  seed={seed}")
        print(f"  run_dir: {run_dir}")

        if args.dry_run:
            print(f"  command: {cmd[0]} {cmd[1]} …  (dry run)\n")
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "resolved_config.yaml").write_text(
            yaml.dump(cfg, default_flow_style=False, sort_keys=False)
        )
        (run_dir / "command.txt").write_text(cmd_pretty + "\n")

        wb_run  = init_wandb(cfg, run_dir)
        plateau = _make_plateau_detector(cfg)

        rc = launch(
            cmd, run_dir,
            wb_run=wb_run,
            max_runtime_sec=args.max_runtime_sec,
            plateau=plateau,
        )

        if wb_run is not None:
            upload_run_artifacts(run_dir, wb_run, cfg)
            wb_run.finish(exit_code=rc)

        if rc == 0:
            status = "done"
        elif rc < 0:
            # Negative codes are Unix signal terminations (SIGINT/SIGTERM/SIGKILL).
            # These are intentional stops (plateau, timeout, Ctrl-C) — not failures.
            status = f"stopped (signal {-rc})"
        else:
            status = f"FAILED (exit {rc})"
            failed.append(f"{run_name}: exit {rc}")

        print(f"  → {status}\n")

    print(f"Sweep complete: {total - len(failed)}/{total} succeeded")
    if failed:
        for f in failed:
            print(f"  FAILED: {f}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
