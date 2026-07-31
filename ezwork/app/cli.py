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

Piped stdin is appended as context, so a sub-agent can receive data without
a file round-trip (e.g. `git diff | ezwork -p "summarize"`).

A sub-agent is therefore simply a session you spin up from a parent workflow —
same config, same tools, isolated history. No extra settings, no extra files.

Error policy: the kernel never raises on LLM failures — provider errors arrive
as ErrorEvent. This CLI prints them as a readable message and keeps going; the
process never crashes on an LLM error. Tool errors are surfaced by the renderer
too. Only truly unexpected exceptions are caught at the top level and reported.
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from collections.abc import Callable
from pathlib import Path

from ezwork import __version__
from ezwork.core import Agent, LoopConfig, ToolRegistry
from ezwork.tools import BashTool, EditTool, ReadTool, WriteTool

from .config import DEFAULT_CONFIG_PATH, Config
from .prompt import build_system_prompt
from .session import Session, SessionStore
from .ui import Palette, UI

# ─── paths ────────────────────────────────────────────────────────────────

HOME = str(Path.home())
SESSIONS_DIR = str(Path.home() / ".ezwork" / "sessions")


def _cwd() -> str:
    """Current working directory (lazy, may not exist in sandbox)."""
    try:
        return str(Path.cwd())
    except OSError:
        return HOME


def _skills_dirs(cwd: str) -> list[Path]:
    return [Path(HOME) / ".ezwork" / "skills", Path(cwd) / ".ezwork" / "skills"]


# ─── streaming renderer ────────────────────────────────────────────────────
# The UI class lives in ui.py (Palette + streaming renderer). Attached only in
# interactive REPL mode. Oneshot does NOT use a renderer at all — it just
# prints the final answer afterwards — so there is no "silent" mode and
# streaming output can never leak into oneshot stdout.


# ─── agent construction ────────────────────────────────────────────────────


def _build_tools(tool_timeout: int) -> ToolRegistry:
    reg = ToolRegistry()
    for t in [ReadTool(), WriteTool(), EditTool(), BashTool(timeout=tool_timeout)]:
        reg.register(t)
    return reg


def build_agent(
    config: Config,
    *,
    render: bool,
    model_override: str | None = None,
    on_error: Callable[[str], None] | None = None,
) -> Agent:
    """Build a fully-wired AgentLoop.

    render=True  -> attach UI (REPL: live streaming + tool events)
    render=False -> no renderer at all (oneshot: nothing prints during the run;
                    the caller prints the final answer afterwards). This avoids
                    the whole "silent flag" branch — oneshot simply has no
                    renderer to leak anything.
    on_error    -> optional callback invoked once per provider ErrorEvent.
                   Oneshot uses this to surface LLM failures on stderr (the
                   renderer only exists in REPL mode, so without it a provider
                   error would otherwise be completely silent).
    """
    provider = config.build_provider()
    if model_override:
        provider.model = model_override  # temporary run override

    cfg = LoopConfig(
        max_retries=2,
        thinking=config.thinking,
        reasoning_effort=config.reasoning_effort or None,
        max_tokens=config.max_tokens,
        tool_timeout=config.tool_timeout,
    )
    if render:
        cfg.emit.append(UI())
    if on_error is not None:
        cfg.emit.append(
            lambda e: on_error(e.error)
            if getattr(e, "type", None) == "error" and getattr(e, "error", None)
            else None
        )

    tools = _build_tools(config.tool_timeout)
    system_prompt = build_system_prompt(
        cwd=_cwd(),
        home=HOME,
        config_path=str(DEFAULT_CONFIG_PATH),
        sessions_dir=SESSIONS_DIR,
        skills_dirs=_skills_dirs(_cwd()),
    )

    return (
        Agent()
        .provider(provider)
        .prompt(system_prompt)
        .tools(tools)
        .config(cfg)
        .build()
    )


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
    """Run one turn; print the answer to stdout, session id + tokens to stderr."""
    config = load_config_or_exit()
    errors: list[str] = []
    agent = build_agent(
        config, render=False, model_override=model_override, on_error=errors.append
    )
    store = SessionStore()

    # Load prior history if continuing a session.
    session: Session
    if session_id:
        loaded = store.load(_cwd(), session_id)
        if loaded is None:
            print(f"[error] session {session_id} not found (cwd {_cwd()})", file=sys.stderr)
            return 1
        session = loaded
        agent.messages = list(session.messages)
    else:
        session = Session.new(_cwd(), model=config.model, provider=config.provider)

    try:
        answer = await agent.chat(prompt_text)
    except asyncio.CancelledError:
        print("\n(interrupted)", file=sys.stderr)
        _persist_session(store, session, agent.messages, no_session=no_session)
        return 130
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        _persist_session(store, session, agent.messages, no_session=no_session)
        return 1

    # Output layout (human view in a terminal):
    #   <answer>            <- stdout (clean, script-friendly)
    #   ───────────────     <- stderr separator
    #   session: <id>       <- stderr
    #   [tokens: ...]       <- stderr
    # stdout flushes before stderr, so the answer appears on top. Scripts that
    # want only the answer just read stdout; those wanting the session id grep
    # stderr for '^session:'.
    if errors:
        # Provider failed: the loop stops with an empty/partial answer and no
        # renderer is attached in oneshot mode, so surface the error here and
        # exit non-zero — a script must not mistake this for a real answer.
        print(f"[provider error] {errors[0]}", file=sys.stderr)
        _persist_session(store, session, agent.messages, no_session=no_session)
        return 1
    print(answer, flush=True)
    u = agent.total_usage
    meta_lines = [f"[tokens: prompt={u.prompt_tokens} completion={u.completion_tokens}]"]
    if not no_session:
        _persist_session(store, session, agent.messages)
        meta_lines.append(f"session: {session.id}")
    print("────────────────", file=sys.stderr)
    for line in meta_lines:
        print(line, file=sys.stderr)
    return 0


def _persist_session(
    store: SessionStore,
    session: Session | None,
    messages,
    *,
    no_session: bool = False,
) -> None:
    """Save the session if there is one and anything to save. Never raises."""
    if no_session or session is None or not messages:
        return
    session.messages = list(messages)
    try:
        store.save(session)
    except Exception:
        pass


async def repl(
    *,
    session_id: str | None = None,
    model_override: str | None = None,
) -> int:
    """Interactive loop with slash commands."""
    config = load_config_or_exit()
    agent = build_agent(config, render=True, model_override=model_override)
    store = SessionStore()

    session: Session | None = None
    ui = UI()
    if session_id:
        loaded = store.load(_cwd(), session_id)
        if loaded is None:
            print(f"[error] session {session_id} not found (cwd {_cwd()})", file=sys.stderr)
            return 1
        session = loaded
        agent.messages = list(session.messages)
        ui.info(f"(resumed session {session.id}: {session.title or 'untitled'})")

    model = config.model or config.provider
    ui.info(f"Ezwork {__version__} — {model}")

    while True:
        try:
            line = ui.prompt().strip()
        except (EOFError, KeyboardInterrupt):
            print()
            _persist_session(store, session, agent.messages)
            return 0
        if not line:
            continue

        cmd, _, arg = line.partition(" ")
        cmd_lower = cmd.lower()

        if cmd_lower in ("/exit", "/quit"):
            _persist_session(store, session, agent.messages)
            return 0
        if cmd_lower == "/clear":
            agent.reset()
            session = None
            ui.info("(history cleared)")
            continue
        if cmd_lower in ("/help", "?"):
            _print_help()
            continue
        if cmd_lower == "/sessions":
            # /sessions       -> default 10; /sessions N -> show N (capped at 100).
            limit = 10
            n = arg.strip()
            if n.isdigit():
                limit = max(1, min(int(n), 100))
            _print_sessions(store, limit=limit)
            continue
        if cmd_lower == "/resume":
            resumed = _resume_cmd(store, arg.strip())
            if resumed is not None:
                session = resumed
                agent.messages = list(session.messages)
                ui.info(
                    f"(resumed session {session.id}: {_format_preview(session.title)})"
                )
            continue

        # Normal chat turn. Create the session lazily on first real message.
        if session is None:
            session = Session.new(_cwd(), model=config.model, provider=config.provider)

        task = asyncio.ensure_future(agent.chat(line))
        cancelled = False

        def _on_sigint(_s, _f):
            nonlocal cancelled
            cancelled = True
            if not task.done():
                task.cancel()

        original = signal.signal(signal.SIGINT, _on_sigint)
        try:
            await task
        except asyncio.CancelledError:
            ui.info("(interrupted)")
        except Exception as e:
            ui.error(f"[error] {e}")
        finally:
            signal.signal(signal.SIGINT, original)

        _persist_session(store, session, agent.messages)
        ui.divider()  # separator before the next prompt


def _print_help() -> None:
    print(
        "commands:\n"
        "  /exit /quit        exit\n"
        "  /clear             clear current conversation history\n"
        "  /help              show this help\n"
        "  /sessions [N]      list sessions for this directory (default 10)\n"
        "  /resume [id]       resume a session (lists recent ones if no id)"
    )


def _format_time(iso: str) -> str:
    """Trim an ISO timestamp to a readable 'YYYY-MM-DD HH:MM' form."""
    if not iso:
        return "?"
    # datetime.isoformat() -> '2026-07-29T19:41:23.456789'; keep date + HH:MM.
    return iso[:16].replace("T", " ")


def _format_preview(title: str, limit: int = 40) -> str:
    """One-line preview of a session's content, collapsed + truncated."""
    text = " ".join((title or "untitled").split())
    return text[:limit] + ("..." if len(text) > limit else "")


def _format_session_line(s: Session) -> str:
    """One line for the /sessions and /resume listings."""
    return (
        f"  {Palette.paint(s.id, 'accent')}  "
        f"{Palette.paint(_format_time(s.updated_at), 'muted')}  "
        f"{_format_preview(s.title)}"
    )


def _print_sessions(store: SessionStore, limit: int = 10) -> None:
    sessions = store.list(_cwd())
    if not sessions:
        print("(no sessions for this directory)")
        return
    shown = sessions[:limit]
    print(f"sessions for this directory (showing {len(shown)} of {len(sessions)}):")
    for s in shown:
        print(_format_session_line(s))


def _resume_cmd(store: SessionStore, arg: str) -> Session | None:
    """Resume by id; with no id, list recent sessions and hint at /resume <id>."""
    if arg:
        loaded = store.load(_cwd(), arg)
        if loaded is None:
            print(f"[error] session {arg} not found")
        return loaded
    sessions = store.list(_cwd())
    if not sessions:
        print("(no sessions for this directory)")
        return None
    print("recent sessions (use /resume <id> to resume one):")
    for s in sessions[:10]:
        print(_format_session_line(s))
    return None


# ─── entry ─────────────────────────────────────────────────────────────────

# How long `-p -` waits for piped stdin before giving up (see _read_stdin).
_STDIN_TIMEOUT = 2.0


def _enable_ansi() -> None:
    """Enable ANSI escape processing on Windows 10+ consoles (no-op elsewhere).

    The well-known `os.system("")` trick switches the console into VT mode;
    without it the styled UI shows raw escape codes on cmd/PowerShell.
    """
    if sys.platform == "win32":
        import os

        os.system("")


def _read_piped_stdin() -> str | None:
    """Return piped/redirected stdin content, or None if there is none.

    Used for two purposes: `-p -` (stdin is the whole prompt) and piped
    context with `-p "query"` (stdin content is appended as context, matching
    `cat file | claude -p "query"`).

    Never blocks indefinitely: piped input is drained with a short timeout so
    a never-closing pipe (sandbox, CI) cannot hang the process. Returns None
    when stdin is a TTY, has no data, or select() can't probe it (Windows).
    """
    if sys.stdin.isatty():
        return None

    import os
    import time

    fd = sys.stdin.fileno()
    chunks: list[str] = []
    deadline = time.monotonic() + _STDIN_TIMEOUT

    # select() only works on sockets on Windows; probe it and fall back to a
    # non-blocking drain loop there (os.set_blocking supports pipes on win32).
    use_select = True
    try:
        import select

        select.select([sys.stdin], [], [], 0)
    except (OSError, ValueError):
        use_select = False

    if use_select:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            ready, _, _ = select.select([sys.stdin], [], [], remaining)
            if not ready:
                break
            data = os.read(fd, 65536)
            if not data:  # EOF — the pipe writer closed
                break
            chunks.append(data.decode("utf-8", errors="replace"))
    else:
        try:
            os.set_blocking(fd, False)
        except OSError:
            return None  # cannot probe this stdin (e.g. a console handle)
        try:
            while True:
                if time.monotonic() >= deadline:
                    break
                try:
                    data = os.read(fd, 65536)
                except BlockingIOError:
                    time.sleep(0.01)
                    continue
                if not data:  # EOF — the pipe writer closed
                    break
                chunks.append(data.decode("utf-8", errors="replace"))
        finally:
            os.set_blocking(fd, True)

    return "".join(chunks) if chunks else None


def _prompt_from_stdin() -> str:
    """Read the whole prompt from stdin for `-p -` (like `cat -`).

    On a TTY this reads interactively until EOF (Ctrl-D); piped input is
    drained via _read_piped_stdin(). Raises SystemExit(2) on empty input so
    we never send an empty prompt (or a literal '-') to the model.
    """
    if sys.stdin.isatty():
        data = sys.stdin.read()
    else:
        data = _read_piped_stdin() or ""
    if not data.strip():
        print("error: -p - received no input on stdin", file=sys.stderr)
        raise SystemExit(2)
    return data


def main() -> int:
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
    args = parser.parse_args()

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
