"""Argument parsing — runs before any heavy imports."""

from __future__ import annotations

import argparse

from ezwork import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ezwork",
        description="Ezwork — a lean coding agent (REPL or one-shot).",
    )
    parser.add_argument(
        "-p", "--prompt",
        help="one-shot prompt; use '-' to read it from stdin; omit for REPL.",
    )
    parser.add_argument("-s", "--session", help="session id to resume/continue.")
    parser.add_argument("--model", help="override the model for this run.")
    parser.add_argument(
        "--no-session",
        action="store_true",
        help="one-shot mode: do not persist a session.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="show version and exit.",
    )
    return parser


__all__ = ["build_parser"]
