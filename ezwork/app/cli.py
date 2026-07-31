"""Ezwork CLI — entry point for the `ezwork` command.

    ezwork                                  REPL: streaming, multi-turn
    ezwork -p "your question"               oneshot: answer to stdout,
                                         session id + tokens to stderr
    ezwork -p "continue..." -s <id>         oneshot, continue an existing session
    ezwork --model X                        override model for this run
    ezwork --no-session                     oneshot without persisting a session

Sub-agents: there is no separate sub-agent subsystem. Every `-p` run creates a
session and prints its id to stderr; any session can be continued with `-s <id>`.
To run a sub-agent, just start another one-shot session and keep its id:

    SID=$(ezwork -p "analyse the auth module" 2>&1 >/dev/null | grep '^session:' | cut -d' ' -f2)
    ezwork -p "now list its tests" -s "$SID"

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
import json
import signal
import sys
from collections.abc import Callable
from pathlib import Path

from ezwork import __version__
from ezwork.core import Agent, LoopConfig, ToolRegistry
from ezwork.tools import BashTool, EditTool, GlobTool, GrepTool, ReadTool, WriteTool

from .config import DEFAULT_CONFIG_PATH, Config
from .prompt import build_system_prompt
from .session import Session, SessionStore

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


class StreamRenderer:
    """Print streaming text + tool events as the loop runs (REPL only).

    Only attached in interactive REPL mode. Oneshot does NOT use a renderer at
    all — it just prints the final answer afterwards — so this class never needs
    a "silent" mode and can't leak streaming output into oneshot stdout.

    Layout per iteration:
        [thinking]  dim, only when reasoning_delta arrives
        answer      normal text
        tools         -> name(args)
                      ok result  /  x error
    """

    _DIM = "\033[2m"
    _RESET = "\033[0m"

    def __init__(self) -> None:
        self._section: str | None = None  # "thinking" | "answer" | None
        self._has_error = False
        if not sys.stdout.isatty():
            self._DIM = self._RESET = ""

    def __call__(self, event) -> None:
        handler = self._HANDLERS.get(event.type)
        if handler:
            handler(self, event)

    # -- section state --

    def _enter(self, section: str) -> None:
        if self._section is not None and self._section != section:
            self._close()
        if self._section == section:
            return
        self._section = section
        if section == "thinking":
            print(f"{self._DIM}[thinking] ", end="", flush=True)

    def _close(self) -> None:
        if self._section == "thinking":
            print(self._RESET, end="", flush=True)
        if self._section is not None:
            print(flush=True)
        self._section = None

    # -- handlers (one per event type) --

    def _on_iter_start(self, event) -> None:
        if event.iteration > 0:
            self._close()
            print("— next iteration —", flush=True)

    def _on_stream_chunk(self, event) -> None:
        chunk = event.chunk
        if chunk.type == "text_delta" and chunk.text:
            self._enter("answer")
            print(chunk.text, end="", flush=True)
        elif chunk.type == "reasoning_delta" and chunk.text:
            self._enter("thinking")
            print(chunk.text, end="", flush=True)

    def _on_response(self, _event) -> None:
        self._close()

    def _on_tool_start(self, event) -> None:
        self._close()
        tc = event.tool_call.get("function", {})
        args = _format_args(tc.get("arguments", ""))
        print(f"  -> {tc.get('name', '?')}({args})", flush=True)

    def _on_tool_complete(self, event) -> None:
        content = event.tool_result.get("content", "")
        tag = "x" if content.startswith("[error]") else "ok"
        print(f"    {tag} {_oneline(content, 120)}", flush=True)

    def _on_error(self, event) -> None:
        # Errors go to stderr so they never pollute oneshot's stdout answer.
        self._has_error = True
        self._close()
        print(f"[provider error] {event.error}", file=sys.stderr, flush=True)

    _HANDLERS = {
        "iter_start": _on_iter_start,
        "stream_chunk": _on_stream_chunk,
        "response": _on_response,
        "tool_start": _on_tool_start,
        "tool_complete": _on_tool_complete,
        "error": _on_error,
    }


def _format_args(args_raw: str, limit: int = 80) -> str:
    try:
        d = json.loads(args_raw) if args_raw else {}
    except Exception:
        return args_raw[:limit]
    s = ", ".join(f"{k}={v!r}" for k, v in d.items())
    return s[:limit] + ("..." if len(s) > limit else "")


def _oneline(text: str, limit: int) -> str:
    s = " ".join(text.split())
    return s[:limit] + ("..." if len(s) > limit else "")


# ─── agent construction ────────────────────────────────────────────────────


def _build_tools() -> ToolRegistry:
    reg = ToolRegistry()
    for t in [ReadTool(), WriteTool(), EditTool(), BashTool(), GlobTool(), GrepTool()]:
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

    render=True  -> attach StreamRenderer (REPL: live streaming + tool events)
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
        provider._model = model_override  # temporary run override

    cfg = LoopConfig(
        max_retries=2,
        thinking=config.thinking,
        reasoning_effort=config.effective_reasoning_effort(),
        max_tokens=config.max_tokens,
    )
    if render:
        cfg.emit.append(StreamRenderer())
    if on_error is not None:
        cfg.emit.append(
            lambda e: on_error(e.error)
            if getattr(e, "type", None) == "error" and getattr(e, "error", None)
            else None
        )

    tools = _build_tools()
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
    if session_id:
        loaded = store.load(_cwd(), session_id)
        if loaded is None:
            print(f"[error] session {session_id} not found (cwd {_cwd()})", file=sys.stderr)
            return 1
        session = loaded
        agent.messages = list(session.messages)
        print(f"(resumed session {session.id}: {session.title or 'untitled'})")

    print(
        f"Ezwork CLI — model={config.model}  provider={config.provider}  "
        f"(commands: /exit /clear /help /sessions /resume)"
    )

    while True:
        try:
            line = input("\n> ").strip()
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
            print("(history cleared)")
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
                print(f"(resumed session {session.id}: {_format_preview(session.title)})")
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
            print("\n(interrupted)")
        except Exception as e:
            print(f"\n[error] {e}", file=sys.stderr)
        finally:
            signal.signal(signal.SIGINT, original)

        _persist_session(store, session, agent.messages)


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
    return f"  {s.id}  {_format_time(s.updated_at)}  {_format_preview(s.title)}"


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
    if arg:
        loaded = store.load(_cwd(), arg)
        if loaded is None:
            print(f"[error] session {arg} not found")
        return loaded
    sessions = store.list(_cwd())
    if not sessions:
        print("(no sessions for this directory)")
        return None
    print("recent sessions (enter an id to resume, or blank to cancel):")
    for s in sessions[:10]:
        print(_format_session_line(s))
    try:
        choice = input("\nresume> ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    if not choice:
        return None
    loaded = store.load(_cwd(), choice)
    if loaded is None:
        print(f"[error] session {choice} not found")
    return loaded


# ─── entry ─────────────────────────────────────────────────────────────────


def _read_stdin_if_piped() -> str | None:
    if sys.stdin.isatty():
        return None
    try:
        import select

        r, _, _ = select.select([sys.stdin], [], [], 0)
        if not r:
            return None
        return sys.stdin.read()
    except Exception:
        return None


def _compose_prompt(prompt: str, stdin_text: str | None) -> str:
    """Attach piped stdin as context to the prompt (carried over from the
    legacy app: `cat file | ezwork -p "explain this"`)."""
    if not stdin_text or not stdin_text.rstrip():
        return prompt
    return f"{stdin_text.rstrip()}\n\n---\n\n{prompt}"


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="ezwork",
        description="Ezwork — a lean coding agent (REPL or one-shot).",
    )
    parser.add_argument("-p", "--prompt", help="one-shot prompt; omit for REPL. Use '-' to read from stdin.")
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

    if args.prompt is not None:
        stdin_text = _read_stdin_if_piped()
        prompt = args.prompt
        if prompt == "-" and stdin_text:
            prompt = stdin_text.rstrip()
            stdin_text = None
        prompt = _compose_prompt(prompt, stdin_text)
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
