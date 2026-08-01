"""Slash commands — Command / CommandRegistry and the built-in set.

Borrowed from MoCode's command system, trimmed to what ezwork needs: a
command is a name + description + optional aliases + one async handler.
`/help` is generated from the registry, so adding a command never requires
touching the help text by hand.

Dispatch contract: handlers return a CommandResult — CONTINUE (keep the
REPL running), EXIT (leave the REPL), or chat(text) (send text to the agent).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable, Callable

from ..utils import oneline

if TYPE_CHECKING:
    from .app import CLIApp
    from .display import Display


# ─── Result & Context ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class CommandResult:
    """Outcome of a command dispatch."""

    kind: str  # "continue" | "exit" | "chat"
    prompt: str | None = None

    @classmethod
    def chat(cls, text: str) -> CommandResult:
        return cls(kind="chat", prompt=text)


CommandResult.CONTINUE = CommandResult(kind="continue")
CommandResult.EXIT = CommandResult(kind="exit")


@dataclass
class CommandContext:
    app: CLIApp
    args: str  # everything after the command name, stripped
    display: Display


# ─── Command ────────────────────────────────────────────────────────────────


@dataclass
class Command:
    name: str  # "/sessions"
    description: str
    aliases: tuple[str, ...] = ()
    handler: Callable[[CommandContext], Awaitable[CommandResult]] | None = None

    async def run(self, ctx: CommandContext) -> CommandResult:
        if self.handler is None:
            return CommandResult.CONTINUE
        return await self.handler(ctx)


class CommandRegistry:
    """Single source of truth for slash commands (name + aliases)."""

    def __init__(self) -> None:
        self._by_name: dict[str, Command] = {}
        self._order: list[Command] = []

    def register(self, cmd: Command) -> None:
        self._by_name[cmd.name] = cmd
        for a in cmd.aliases:
            self._by_name[a] = cmd
        self._order.append(cmd)

    def get(self, text: str) -> Command | None:
        return self._by_name.get(text.lower())

    def all(self) -> list[Command]:
        """All registered commands, sorted alphabetically by name."""
        return sorted(self._order, key=lambda c: c.name)


# ─── Formatting helpers ─────────────────────────────────────────────────────


def _format_time(iso: str) -> str:
    """Trim an ISO timestamp to a readable 'YYYY-MM-DD HH:MM' form."""
    if not iso:
        return "?"
    return iso[:16].replace("T", " ")


def _format_preview(title: str, limit: int = 40) -> str:
    """One-line preview of a session's content, collapsed + truncated."""
    text = " ".join((title or "untitled").split())
    return text[:limit] + ("..." if len(text) > limit else "")


def _format_session_line(session) -> str:
    """One line for the /sessions and /resume listings (palette applied)."""
    from .display import Palette

    return (
        f"  {Palette.paint(session.id, 'accent')}  "
        f"{Palette.paint(_format_time(session.updated_at), 'muted')}  "
        f"{_format_preview(session.title)}"
    )


# ─── Built-in commands ──────────────────────────────────────────────────────


async def _quit(ctx: CommandContext) -> CommandResult:
    return CommandResult.EXIT


async def _help(ctx: CommandContext) -> CommandResult:
    commands = ctx.app.commands.all()
    if not commands:
        ctx.display.info("No commands available.")
        return CommandResult.CONTINUE
    max_len = max(len(c.name) for c in commands)
    lines = [f"  {c.name:<{max_len}}  {c.description}" for c in commands]
    ctx.display.info("commands:\n" + "\n".join(lines))
    return CommandResult.CONTINUE


async def _clear(ctx: CommandContext) -> CommandResult:
    ctx.app.clear_conversation()
    ctx.display.info("(history cleared)")
    return CommandResult.CONTINUE


async def _sessions(ctx: CommandContext) -> CommandResult:
    # /sessions       -> default 10; /sessions N -> show N (capped at 100).
    limit = 10
    n = ctx.args.strip()
    if n.isdigit():
        limit = max(1, min(int(n), 100))
    sessions = ctx.app.session_mgr.list()
    if not sessions:
        ctx.display.info("(no sessions for this directory)")
        return CommandResult.CONTINUE
    shown = sessions[:limit]
    ctx.display.info(
        f"sessions for this directory (showing {len(shown)} of {len(sessions)}):"
    )
    for s in shown:
        ctx.display.print(_format_session_line(s))
    return CommandResult.CONTINUE


async def _resume(ctx: CommandContext) -> CommandResult:
    """Resume by id; with no id, list recent sessions and hint at /resume <id>."""
    arg = ctx.args.strip()
    if arg:
        session = ctx.app.session_mgr.resume(arg)
        if session is None:
            ctx.display.error(f"session {arg} not found")
            return CommandResult.CONTINUE
        ctx.app.resume_session(session)
        return CommandResult.CONTINUE
    sessions = ctx.app.session_mgr.list()
    if not sessions:
        ctx.display.info("(no sessions for this directory)")
        return CommandResult.CONTINUE
    ctx.display.info("recent sessions (use /resume <id> to resume one):")
    for s in sessions[:10]:
        ctx.display.print(_format_session_line(s))
    return CommandResult.CONTINUE


async def _copy(ctx: CommandContext) -> CommandResult:
    """Copy the last plain assistant answer to the clipboard."""
    for msg in reversed(ctx.app.agent.messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content") or ""
        if content and not msg.get("tool_calls"):
            try:
                import pyperclip

                pyperclip.copy(content)
                ctx.display.info(f"Copied: {oneline(content, 60)}")
            except ImportError:
                ctx.display.error("pyperclip is not installed — cannot copy")
            except Exception as e:  # noqa: BLE001 — clipboard backends vary
                ctx.display.error(f"Clipboard error: {e}")
            return CommandResult.CONTINUE
    ctx.display.warn("No assistant response to copy.")
    return CommandResult.CONTINUE


async def _export(ctx: CommandContext) -> CommandResult:
    """Export the active session to cwd as JSON (portable) or Markdown."""
    from ..export import export_session

    session = ctx.app.session_mgr.active
    if session is None or not session.messages:
        ctx.display.warn("No active session to export.")
        return CommandResult.CONTINUE
    fmt = ctx.args.strip().lower() or "json"
    if fmt not in ("json", "md"):
        ctx.display.warn("usage: /export [json|md]")
        return CommandResult.CONTINUE
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path.cwd() / f"session_{ts}.{fmt}"
    export_session(
        session, path, fmt=fmt, system_prompt=ctx.app.agent.system_prompt
    )
    ctx.display.info(f"Exported {len(session.messages)} msgs → {path}")
    return CommandResult.CONTINUE


async def _model(ctx: CommandContext) -> CommandResult:
    ctx.display.info(
        f"provider: {ctx.app.config.provider}  model: {ctx.app.config.model or '(default)'}"
    )
    return CommandResult.CONTINUE


# ─── Command list ───────────────────────────────────────────────────────────


commands: list[Command] = [
    Command("/quit", "Exit the application", aliases=("/exit",), handler=_quit),
    Command("/help", "Show available commands", handler=_help),
    Command("/clear", "Clear the current conversation history", handler=_clear),
    Command(
        "/sessions",
        "List sessions for this directory (optional count: /sessions 20)",
        handler=_sessions,
    ),
    Command(
        "/resume",
        "Resume a session by id; with no id, list recent ones",
        handler=_resume,
    ),
    Command(
        "/copy",
        "Copy the last assistant response to the clipboard",
        handler=_copy,
    ),
    Command(
        "/export",
        "Export the current session to a file: /export [json|md]",
        handler=_export,
    ),
    Command("/model", "Show the active provider and model", handler=_model),
]


__all__ = [
    "Command",
    "CommandRegistry",
    "CommandContext",
    "CommandResult",
    "commands",
]
