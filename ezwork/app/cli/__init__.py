"""Ezwork CLI — entry point for the `ezwork` command.

    ezwork                                  REPL: streaming, multi-turn
    ezwork -p "your question"               oneshot: answer to stdout,
                                         session id + tokens to stderr
    ezwork -p -                             oneshot, prompt read from stdin
                                         (e.g. `git diff | ezwork -p -`)
    cat file | ezwork -p "summarize"        oneshot; piped content is appended
                                         as context (Claude Code convention)
    ezwork -p "continue..." -s <id>         oneshot, continue an existing session
    ezwork --model X                        override model for this run
    ezwork --no-session                     oneshot without persisting a session

Sub-agents: there is no separate sub-agent subsystem. Every `-p` run creates a
session and prints its id to stderr; any session can be continued with `-s <id>`.
To run a sub-agent, just start another one-shot session and keep its id:

    SID=$(ezwork -p "analyse the auth module" 2>&1 >/dev/null | grep '^session:' | cut -d' ' -f2)
    ezwork -p "now list its tests" -s "$SID"

Error policy: the kernel never raises on LLM failures — provider errors arrive
as ErrorEvent. This CLI prints them as a readable message and keeps going; the
process never crashes on an LLM error. Only truly unexpected exceptions are
caught at the top level and reported.

The module stays a thin shell: argument parsing (args.py), stdin handling
(stdin.py), the composition root (app.py) and the renderer (display.py) live
in sibling modules. `main()` is the pyproject entry point.
"""

from __future__ import annotations

import asyncio
import sys
import threading

from .app import CLIApp
from .args import build_parser
from .stdin import _read_piped_stdin, _prompt_from_stdin

# Re-exported for tests that patch timing constants / stdin.
from .stdin import _STDIN_GRACE, _STDIN_TIMEOUT  # noqa: F401

from ..config import DEFAULT_CONFIG_PATH, Config


# ─── config bootstrap ──────────────────────────────────────────────────────


def load_config_or_exit() -> Config:
    """Load ~/.ezwork/config.json; write a template + exit gracefully if missing
    or not yet filled in. Never raises."""
    if not DEFAULT_CONFIG_PATH.exists():
        Config.write_template(DEFAULT_CONFIG_PATH)
        print(
            f"Created config template: {DEFAULT_CONFIG_PATH}\n"
            "Edit it (set api_key, provider, model), then run again.",
            file=sys.stderr,
        )
        sys.exit(0)

    config = Config.load(DEFAULT_CONFIG_PATH)
    if config is None:
        print(
            f"Could not parse {DEFAULT_CONFIG_PATH} — check the JSON syntax.",
            file=sys.stderr,
        )
        sys.exit(0)
    if not config.is_filled_in():
        print(
            f"Edit {DEFAULT_CONFIG_PATH} and set a valid api_key before running.",
            file=sys.stderr,
        )
        sys.exit(0)
    return config


# ─── modes ─────────────────────────────────────────────────────────────────


async def oneshot(
    prompt_text: str,
    *,
    session_id: str | None = None,
    no_session: bool = False,
    model_override: str | None = None,
) -> int:
    """One-shot turn: answer to stdout, session id + tokens to stderr."""
    config = load_config_or_exit()
    app = CLIApp(config, interactive=False, model_override=model_override)
    return await app.run_oneshot(
        prompt_text, session_id=session_id, no_session=no_session
    )


async def repl(
    *,
    session_id: str | None = None,
    model_override: str | None = None,
) -> int:
    """Interactive loop with slash commands."""
    config = load_config_or_exit()
    app = CLIApp(
        config, interactive=True, model_override=model_override, resume_id=session_id
    )
    return await app.repl()


# ─── entry ─────────────────────────────────────────────────────────────────


def _preload_openai() -> None:
    """Start the openai SDK import in a daemon thread (no client built).

    The heavy import (~2s) then overlaps with arg parsing, the stdin drain,
    and the REPL startup instead of sitting in the first stream() call.
    """

    def _work() -> None:
        from ezwork.providers.openai import preload

        preload()

    threading.Thread(target=_work, daemon=True, name="ezwork-openai-preload").start()


def _enable_ansi() -> None:
    """Enable ANSI escape processing on Windows 10+ consoles (no-op elsewhere).

    The well-known `os.system("")` trick switches the console into VT mode;
    without it the styled UI shows raw escape codes on cmd/PowerShell. Only
    needed when stdout is an interactive terminal — oneshot/piped output
    never emits ANSI, so skip the cmd.exe spawn there.
    """
    if sys.platform == "win32" and sys.stdout.isatty():
        import os

        os.system("")


def main() -> int:
    # Kick off the openai SDK import in a background thread so it overlaps
    # with arg parsing, the stdin drain (oneshot) and startup (REPL) instead
    # of sitting in the first stream() call.
    _preload_openai()

    args = build_parser().parse_args()

    _enable_ansi()

    if args.prompt is not None:
        prompt = args.prompt
        if prompt == "-":
            # `-p -`: stdin is the whole prompt (like `cat -`).
            try:
                prompt = _prompt_from_stdin()
            except SystemExit as exc:
                return int(exc.code)
        else:
            # `-p "query"` with piped stdin: append the content as context,
            # matching `cat file | claude -p "query"` (Claude Code convention).
            piped = _read_piped_stdin()
            if piped:
                prompt = f"{prompt}\n\n{piped}"
        try:
            return asyncio.run(
                oneshot(
                    prompt,
                    session_id=args.session,
                    no_session=args.no_session,
                    model_override=args.model,
                )
            )
        except KeyboardInterrupt:
            return 130

    try:
        return asyncio.run(
            repl(session_id=args.session, model_override=args.model)
        )
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
