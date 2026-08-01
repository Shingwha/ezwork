"""Ezwork app layer — CLI, config, sessions, prompts, export.

One opinionated way to run the `ezwork` kernel:

    app/
    ├── config.py     Config (~/.ezwork/config.json, single provider)
    ├── session.py    Session + SessionStore + SessionManager
    ├── prompt.py     build_system_prompt (plain text + XML)
    ├── utils.py      JSON I/O + text shaping + tool-call grouping
    ├── export.py     session -> Markdown / JSON
    └── cli/          entry point: args, stdin, CLIApp, Display, Input,
                      slash commands

Public imports stay stable: `from ezwork.app import Config, Session,
SessionStore, build_system_prompt, ...` — the CLI entry (`ezwork.app.cli:main`)
is unchanged.
"""

from __future__ import annotations

from .config import Config, DEFAULT_CONFIG_PATH
from .session import Session, SessionManager, SessionStore, DEFAULT_SESSIONS_DIR
from .prompt import build_system_prompt

__all__ = [
    "Config",
    "DEFAULT_CONFIG_PATH",
    "Session",
    "SessionStore",
    "SessionManager",
    "DEFAULT_SESSIONS_DIR",
    "build_system_prompt",
]
