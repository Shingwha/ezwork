"""CLIApp — composition root for the Ezwork CLI.

Owns config, agent construction, session management, slash commands and the
display, and exposes two entry points:
  - `run()`            interactive REPL (sync wrapper over asyncio.run)
  - `run_oneshot()`    one-shot turn: answer to stdout, session id + tokens
                       to stderr (the script-friendly contract)

Borrowed shape from MoCode's CLIApp, trimmed: no workflow engine, no skill
manager, no provider hot-swap. The module-level `build_agent()` stays
available standalone for tests and embedding.
"""

from __future__ import annotations

import asyncio
import signal
import sys
from collections.abc import Callable
from pathlib import Path

from ezwork import __version__
from ezwork.core import Agent, LoopConfig, ToolRegistry
from ezwork.tools import BashTool, EditTool, ReadTool, WriteTool

from ..config import DEFAULT_CONFIG_PATH, Config
from ..prompt import build_system_prompt
from ..session import Session, SessionManager
from .commands import CommandContext, CommandRegistry, CommandResult, commands as builtin_commands
from .input import Input
from .display import Display

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
    renderer: Display | None = None,
) -> Agent:
    """Build a fully-wired AgentLoop.

    render=True  -> attach the display renderer (REPL: live streaming + tool
                    events; pass your Display via renderer, or one is created)
    render=False -> no renderer at all (oneshot: nothing prints during the
                    run; the caller prints the final answer afterwards)
    on_error    -> optional callback invoked once per provider ErrorEvent.
                    Oneshot uses this to surface LLM failures on stderr.
    """
    provider = config.build_provider()
    if model_override:
        provider.model = model_override  # temporary run override
    # Warm the LLM client in a background thread so the first stream() call
    # doesn't pay the ~2-3s SDK import + client construction inline. Optional
    # hook — custom providers that don't implement it stay cold-lazy.
    if hasattr(provider, "warmup"):
        provider.warmup()

    cfg = LoopConfig(
        max_retries=2,
        thinking=config.thinking,
        reasoning_effort=config.reasoning_effort or None,
        max_tokens=config.max_tokens,
        tool_timeout=config.tool_timeout,
    )
    if render:
        cfg.emit.append(renderer or Display())
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


# ─── CLIApp ────────────────────────────────────────────────────────────────


class CLIApp:
    """Composable entry point: config + agent + sessions + commands + display."""

    def __init__(
        self,
        config: Config,
        *,
        interactive: bool = True,
        model_override: str | None = None,
        resume_id: str | None = None,
    ):
        self.config = config
        self.interactive = interactive
        self.model_override = model_override
        self.errors: list[str] = []  # provider errors (oneshot mode)

        # Session management (oneshot and REPL both use it).
        self.session_mgr = SessionManager(
            _cwd(),
            model=config.model,
            provider=config.provider,
        )

        # Interactive-only: commands + display + input.
        self.commands = CommandRegistry()
        self.display: Display | None = None
        self._input: Input | None = None
        if interactive:
            for cmd in builtin_commands:
                self.commands.register(cmd)
            self._input = Input(self.commands)
            self.display = Display(self._input)

        self.agent = self._build_agent()
        self._resume_id = resume_id

    # ── agent construction ──

    def _build_agent(self) -> Agent:
        return build_agent(
            self.config,
            render=self.interactive,
            model_override=self.model_override,
            on_error=None if self.interactive else self.errors.append,
            renderer=self.display,
        )

    # ── session lifecycle ──

    def _save_current_session(self) -> None:
        """Persist current agent messages to the active session.

        Sync by design: called both from the async REPL loop and from the
        sync KeyboardInterrupt path of run(), where awaiting is impossible.
        The write happens between turns — a few ms of JSON I/O is fine."""
        if not self.agent.messages:
            return
        self.session_mgr.save(
            self.agent.messages,
            model=self.config.model,
            provider=self.config.provider,
        )

    def resume_session(self, session: Session) -> None:
        """Resume an existing session — preserves its identity."""
        self._save_current_session()
        self.session_mgr.switch_to(session)
        self.agent.messages = list(session.messages)
        if self.display is not None:
            self.display.info(
                f"(resumed session {session.id}: {session.title or 'untitled'})"
            )
            self.display.divider()
            self.display.render_messages(session.messages)

    def clear_conversation(self) -> None:
        """Save and clear the current conversation."""
        self._save_current_session()
        self.agent.reset()
        self.session_mgr.clear()

    # ── dispatch ──

    async def _dispatch(self, text: str) -> CommandResult:
        """Resolve input: run a slash command, or mark for chat."""
        parts = text.split(None, 1)
        cmd_text = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        cmd = self.commands.get(cmd_text)
        if cmd is not None:
            ctx = CommandContext(app=self, args=args, display=self.display)
            return await cmd.run(ctx)

        if text.startswith("/"):
            # Fuzzy matching for unknown commands.
            import difflib

            matches = difflib.get_close_matches(
                cmd_text, [c.name for c in self.commands.all()], n=1, cutoff=0.6
            )
            if matches:
                self.display.warn(
                    f"Unknown command: {cmd_text} — did you mean {matches[0]}?"
                )
            else:
                self.display.warn(f"Unknown command: {cmd_text}")
            return CommandResult.CONTINUE

        return CommandResult.chat(text)

    # ── chat helper ──

    async def _run_chat(self, prompt: str) -> None:
        """Send prompt to the agent with SIGINT cancellation."""
        task = asyncio.ensure_future(self.agent.chat(prompt))
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
            self.display.info("(interrupted)")
        except Exception as e:
            self.display.error(f"[error] {e}")
        finally:
            signal.signal(signal.SIGINT, original)

    # ── REPL ──

    async def repl(self) -> int:
        """Interactive loop with slash commands."""
        if self.display is None:
            raise RuntimeError("repl() requires interactive=True")

        model = self.config.model or self.config.provider
        self.display.info(f"Ezwork {__version__} — {model}")

        # Resume a session passed via -s before the loop starts.
        if self._resume_id:
            session = self.session_mgr.resume(self._resume_id)
            if session is None:
                self.display.error(
                    f"session {self._resume_id} not found (cwd {_cwd()})"
                )
                return 1
            self.agent.messages = list(session.messages)
            self.display.info(
                f"(resumed session {session.id}: {session.title or 'untitled'})"
            )
            self.display.divider()
            self.display.render_messages(session.messages)

        while True:
            try:
                line = await self.display.prompt()
            except (EOFError, KeyboardInterrupt):
                print()
                self._save_current_session()
                return 0
            if not line:
                continue

            result = await self._dispatch(line)
            if result.kind == "exit":
                self._save_current_session()
                return 0
            if result.kind == "chat":
                # Lazy session: created in memory on the first real message.
                if self.session_mgr.active is None:
                    self.session_mgr.create()
                await self._run_chat(result.prompt)
                self._save_current_session()
                self.display.divider()  # separator before the next prompt

    def run(self) -> int:
        """Sync entry point for the interactive CLI."""
        try:
            return asyncio.run(self.repl())
        except KeyboardInterrupt:
            self._save_current_session()
            return 130

    # ── oneshot ──

    async def run_oneshot(
        self,
        prompt_text: str,
        *,
        session_id: str | None = None,
        no_session: bool = False,
    ) -> int:
        """Run one turn; print the answer to stdout, session id + tokens to
        stderr. Never raises — errors map to non-zero exit codes."""
        if session_id:
            session = self.session_mgr.resume(session_id)
            if session is None:
                print(
                    f"[error] session {session_id} not found (cwd {_cwd()})",
                    file=sys.stderr,
                )
                return 1
            self.agent.messages = list(session.messages)

        try:
            answer = await self.agent.chat(prompt_text)
        except asyncio.CancelledError:
            print("\n(interrupted)", file=sys.stderr)
            await self._persist(no_session=no_session)
            return 130
        except Exception as e:
            print(f"[error] {e}", file=sys.stderr)
            await self._persist(no_session=no_session)
            return 1

        # Output layout (human view in a terminal):
        #   <answer>            <- stdout (clean, script-friendly)
        #   ───────────────     <- stderr separator
        #   session: <id>       <- stderr
        #   [tokens: ...]       <- stderr
        if self.errors:
            # Provider failed: the loop stops with an empty/partial answer and
            # no renderer is attached in oneshot mode, so surface the error
            # here and exit non-zero — a script must not mistake this for a
            # real answer.
            print(f"[provider error] {self.errors[0]}", file=sys.stderr)
            await self._persist(no_session=no_session)
            return 1
        print(answer, flush=True)
        u = self.agent.total_usage
        meta_lines = [f"[tokens: prompt={u.prompt_tokens} completion={u.completion_tokens}]"]
        if not no_session:
            await self._persist()
            meta_lines.append(f"session: {self.session_mgr.active_id}")
        print("────────────────", file=sys.stderr)
        for line in meta_lines:
            print(line, file=sys.stderr)
        return 0

    async def _persist(self, *, no_session: bool = False) -> None:
        """Save the active session if there is anything to save. Never raises.

        The write runs in a worker thread: a long history (large tool outputs)
        can be megabytes of JSON, and saving synchronously would stall the
        loop before the next prompt."""
        if no_session or not self.agent.messages:
            return
        try:
            await asyncio.to_thread(
                self.session_mgr.save,
                self.agent.messages,
                model=self.config.model,
                provider=self.config.provider,
            )
        except Exception:
            pass


__all__ = ["CLIApp", "build_agent", "_build_tools"]
