"""CLI entry point: shogi-watch."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Optional rich output
# ---------------------------------------------------------------------------

try:
    from rich.console import Console
    _console = Console()
    _log = _console.print
except ImportError:
    _log = print


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def _cmd_watch(args: argparse.Namespace) -> None:
    from experiments.watcher import watch_run

    watch_run(
        args.run_dir,
        project=args.project,
        entity=args.entity,
        from_start=not args.tail,
        log_fn=_log,
    )


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shogi-watch",
        description="Stream Shogi training metrics to Weights & Biases.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    watch = sub.add_parser("watch", help="Tail a run directory and log to W&B")
    watch.add_argument(
        "run_dir",
        type=Path,
        help="Path to the run directory (must contain run_config.json, metrics.jsonl, eval.jsonl)",
    )
    watch.add_argument(
        "--project",
        default="shogi",
        help="W&B project name (default: shogi)",
    )
    watch.add_argument(
        "--entity",
        default=None,
        help="W&B entity / team (default: personal account)",
    )
    watch.add_argument(
        "--tail",
        action="store_true",
        help="Skip events already in the file; only forward new ones",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "watch":
        _cmd_watch(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
