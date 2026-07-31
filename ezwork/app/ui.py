"""UI — terminal rendering for the Ezwork CLI (REPL only).

Deliberately tiny: a semantic color palette + one streaming renderer class.
The visual vocabulary is borrowed from MoCode's CLI (icons, semantic colors,
compact one-line tool status) without its theme / spinner / questionary
machinery. One palette, one renderer, no plugin points.

Turn layout (REPL):

    ❯ how does auth work?                  <- typed at the input() prompt
    ┊ scanning the routes first…           <- thinking, dim, per line
    answer streams here as plain text…
    ✓ read(path='auth.py')           0.2s  <- one line per tool call
    ✗ edit(file='x.py') old_string not found
    ────────────────────────────────       <- divider between iterations
    ↑1,234 ↓567 tokens                     <- dim usage, end of turn

Design rules:
  - The tool start line (`→ name(args)`) is overwritten in place by the
    status line (`✓`/`✗`) when it completes, so a turn shows exactly one
    line per tool call — no result echo, no double lines.
  - All color decisions live in Palette; when stdout is not a TTY every
    code is empty, so call sites never branch.
  - Oneshot mode never constructs a UI — stdout stays clean for scripts.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from dataclasses import dataclass


# ─── palette ────────────────────────────────────────────────────────────────


class Palette:
    """Semantic ANSI colors. `enabled=False` (non-TTY stdout) makes every
    code an empty string — call sites need no branching."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GRAY = "\033[90m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    SOFT_CYAN = "\033[36m"

    _STYLES = {
        "bold": BOLD,
        "dim": DIM,
        "muted": GRAY,
        "success": GREEN,
        "error": RED,
        "warn": YELLOW,
        "accent": CYAN,
        "info": SOFT_CYAN,
    }

    enabled = True

    @classmethod
    def paint(cls, text: str, *styles: str) -> str:
        """Apply semantic styles: paint('done', 'success', 'bold')."""
        if not cls.enabled or not styles:
            return text
        codes = "".join(cls._STYLES[s] for s in styles)
        return f"{codes}{text}{cls.RESET}" if codes else text


# ─── formatting helpers (pure) ──────────────────────────────────────────────


def _format_args(args_raw: str, limit: int = 60) -> str:
    """Tool-call arguments JSON string -> compact 'k=v, k=v' one-liner."""
    try:
        d = json.loads(args_raw) if args_raw else {}
    except Exception:
        return args_raw[:limit]
    s = ", ".join(f"{k}={v!r}" for k, v in d.items())
    return s[:limit] + ("..." if len(s) > limit else "")


def _oneline(text: str, limit: int) -> str:
    """Collapse whitespace + truncate to one line."""
    s = " ".join(text.split())
    return s[:limit] + ("..." if len(s) > limit else "")


# ─── renderer ────────────────────────────────────────────────────────────────


@dataclass
class _ToolLine:
    """State for one in-flight tool call, so its start line can be replaced."""

    name: str
    args: str
    started: float
    length: int  # chars of the start line — if it wrapped, don't overwrite


class UI:
    """Streaming turn renderer. Attach to LoopConfig.emit (REPL only)."""

    def __init__(self) -> None:
        self._tty = sys.stdout.isatty()
        Palette.enabled = self._tty
        self._width = self._terminal_width()
        self._section: str | None = None  # "thinking" | "answer" | None
        self._partial = ""               # unflushed thinking line buffer
        self._pending_tool: _ToolLine | None = None
        self._usage = None

    # ── public helpers for the REPL (non-streaming) ──

    def prompt(self) -> str:
        return input(Palette.paint("❯ ", "accent", "bold"))

    def info(self, text: str) -> None:
        self._print(Palette.paint(text, "dim"))

    def warn(self, text: str) -> None:
        self._print(Palette.paint(text, "warn"))

    def error(self, text: str) -> None:
        self._print(Palette.paint(f"✗ {text}", "error"))

    # ── event dispatch (LoopConfig.emit callback) ──

    def __call__(self, event) -> None:
        handler = _HANDLERS.get(event.type)
        if handler:
            handler(self, event)

    # ── output primitives ──

    def _print(self, text: str = "", *, end: str = "\n") -> None:
        print(text, end=end, flush=True)

    @staticmethod
    def _terminal_width() -> int:
        try:
            return shutil.get_terminal_size().columns or 80
        except Exception:
            return 80

    # ── section state (thinking / answer streaming) ──

    def _enter(self, section: str) -> None:
        if self._section is not None and self._section != section:
            self._close_section()
        self._section = section

    def _close_section(self) -> None:
        if self._section == "thinking" and self._partial:
            self._print(Palette.paint(f"┊ {self._partial}", "dim"))
        self._partial = ""
        if self._section is not None:
            self._print()  # end the streamed section with a newline
        self._section = None

    # ── event handlers ──

    def _on_iter_start(self, event) -> None:
        if event.iteration > 0:
            self._close_section()
            self._print(Palette.paint("─" * min(self._width, 48), "muted"))

    def _on_stream_chunk(self, event) -> None:
        chunk = event.chunk
        if chunk.type == "text_delta" and chunk.text:
            self._enter("answer")
            self._print(chunk.text, end="")
        elif chunk.type == "reasoning_delta" and chunk.text:
            self._enter("thinking")
            self._thinking(chunk.text)

    def _thinking(self, text: str) -> None:
        """Stream reasoning text, printing one dim '┊ line' per complete line."""
        parts = text.split("\n")
        parts[0] = self._partial + parts[0]
        self._partial = ""
        for line in parts[:-1]:
            self._print(Palette.paint(f"┊ {line}", "dim"))
        self._partial = parts[-1]

    def _on_response(self, event) -> None:
        self._close_section()
        if event.usage:
            self._usage = event.usage

    def _on_tool_start(self, event) -> None:
        self._close_section()
        tc = event.tool_call.get("function", {})
        name = tc.get("name", "?")
        args = _format_args(tc.get("arguments", ""))
        self._pending_tool = _ToolLine(
            name, args, time.monotonic(), len(name) + len(args) + 4
        )
        self._print(Palette.paint(f"→ {name}({args})", "dim"))

    def _on_tool_complete(self, event) -> None:
        p = self._pending_tool
        self._pending_tool = None
        if p is None:
            return
        content = event.tool_result.get("content", "")
        ok = not content.startswith("[error]")
        elapsed = time.monotonic() - p.started
        parts = [
            Palette.paint("✓" if ok else "✗", "success" if ok else "error"),
            p.name + (f"({p.args})" if p.args else ""),
        ]
        if ok:
            if elapsed >= 0.1:
                parts.append(Palette.paint(f"{elapsed:.1f}s", "dim"))
        else:
            msg = _oneline(content[len("[error]") :].strip(), 90)
            if msg:
                parts.append(Palette.paint(msg, "error"))
        line = " ".join(parts)
        # Replace the start line in place — nothing else printed since it was
        # emitted, so it is guaranteed to be the last line on screen.
        if self._tty and p.length <= self._width:
            self._print("\033[1A\033[K", end="")
        self._print(line)

    def _on_iter_end(self, event) -> None:
        # End-of-turn summary: dim usage line (final iteration only).
        if event.final_content is not None and self._usage:
            u = self._usage
            self._print(
                Palette.paint(
                    f"↑{u.prompt_tokens:,} ↓{u.completion_tokens:,} tokens", "muted"
                )
            )
            self._usage = None

    def _on_error(self, event) -> None:
        # Provider errors go to stderr so they never pollute stdout.
        self._close_section()
        print(
            Palette.paint(f"✗ provider error: {event.error}", "error"),
            file=sys.stderr,
            flush=True,
        )


_HANDLERS = {
    "iter_start": UI._on_iter_start,
    "stream_chunk": UI._on_stream_chunk,
    "response": UI._on_response,
    "tool_start": UI._on_tool_start,
    "tool_complete": UI._on_tool_complete,
    "iter_end": UI._on_iter_end,
    "error": UI._on_error,
}


__all__ = ["Palette", "UI"]
