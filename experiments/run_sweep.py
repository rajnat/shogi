#!/usr/bin/env python3
"""Run a W&B hyperparameter sweep over a base experiment config.

Each sweep run merges the base config with the W&B-assigned parameters,
builds a Rust training command, and streams metrics back to the live run.

Usage (from project root):
    uv run --project experiments python experiments/run_sweep.py \\
        --base-config experiments/configs/baseline_fast.yaml \\
        --sweep-config experiments/configs/sweep_exploration.yaml \\
        [--count 20]

The sweep metric (eval/anchor/elo_delta) requires --eval-anchor in the
base config or passed as a top-level key.  Example baseline_fast.yaml:

    eval_anchor: checkpoints/anchor.ot
"""

from __future__ import annotations

import argparse
import datetime
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

# Shared helpers from the sibling script — no package install needed.
sys.path.insert(0, str(Path(__file__).parent))
from run_experiment import (
    build_command,
    format_command,
    launch,
    upload_run_artifacts,
    _make_plateau_detector,
)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def deep_merge(base: dict, overrides: dict) -> dict[str, Any]:
    """Recursively merge *overrides* into a copy of *base*."""
    result = dict(base)
    for key, val in overrides.items():
        if isinstance(val, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _make_sweep_run_dir(base_cfg: dict) -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    return Path("runs") / f"{ts}_sweep_seed{base_cfg['seed']}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Launch a W&B hyperparameter sweep."
    )
    parser.add_argument("--base-config", required=True, type=Path,
                        metavar="YAML",
                        help="Fixed experiment config (channels, steps, …)")
    parser.add_argument("--sweep-config", required=True, type=Path,
                        metavar="YAML",
                        help="W&B sweep definition (method, parameters, metric)")
    parser.add_argument("--count", type=int, default=None,
                        help="Number of sweep runs to execute (default: unlimited)")
    args = parser.parse_args(argv)

    base_cfg: dict = yaml.safe_load(args.base_config.read_text())
    sweep_def: dict = yaml.safe_load(args.sweep_config.read_text())

    import wandb

    wb_cfg = base_cfg.get("wandb", {})

    # Build the release binary once before the sweep starts rather than
    # inside each agent function.
    if base_cfg.get("build_release"):
        print("Building release binary…")
        rc = subprocess.run(["cargo", "build", "--release"]).returncode
        if rc != 0:
            print(f"FAILED: cargo build exited {rc}", file=sys.stderr)
            sys.exit(rc)

    sweep_id = wandb.sweep(
        sweep_def,
        project=wb_cfg.get("project", "shogi"),
        entity=wb_cfg.get("entity") or None,
    )
    print(f"Sweep ID: {sweep_id}\n")

    def agent_fn() -> None:
        run_dir = _make_sweep_run_dir(base_cfg)
        run_dir.mkdir(parents=True, exist_ok=True)

        # wandb.init() is called by the agent before invoking this function.
        # wandb.config holds the sweep-assigned parameter values.
        run = wandb.init(
            project=wb_cfg.get("project", "shogi"),
            entity=wb_cfg.get("entity") or None,
            group=base_cfg.get("group"),
            tags=wb_cfg.get("tags") or [],
            dir=str(run_dir),
        )

        # Sweep params are flat (c_puct, dirichlet_alpha, …); map them into
        # the train section of the base config.
        sweep_params = {k: v for k, v in wandb.config.items()}
        cfg = deep_merge(base_cfg, {"train": sweep_params})
        cfg["run_name"] = run.name   # use W&B-generated name for this run

        cmd = build_command(cfg, run_dir)
        cmd_pretty = format_command(cmd)
        (run_dir / "resolved_config.yaml").write_text(
            yaml.dump(cfg, default_flow_style=False, sort_keys=False)
        )
        (run_dir / "command.txt").write_text(cmd_pretty + "\n")

        url = getattr(run, "url", None) or "(offline — sync with `wandb sync`)"
        (run_dir / "wandb_run.txt").write_text(f"id:  {run.id}\nurl: {url}\n")

        print(f"\n[sweep] run: {run.name}  {url}")
        print(f"[sweep] params: {sweep_params}")
        print(f"[sweep] run_dir: {run_dir}\n")

        rc = launch(cmd, run_dir, wb_run=run, plateau=_make_plateau_detector(cfg))
        upload_run_artifacts(run_dir, run, cfg)
        run.finish(exit_code=rc)

    wandb.agent(sweep_id, function=agent_fn, count=args.count)


if __name__ == "__main__":
    main()
