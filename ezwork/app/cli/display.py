"""Display — terminal rendering for the Ezwork REPL.

Palette (semantic ANSI colors) + Display (streaming turn renderer attached
to LoopConfig.emit). Visual vocabulary borrowed from MoCode's CLI (icons,
semantic colors, grouped parallel tool calls) without its theme / spinner /
questionary machinery. One palette, one renderer, no plugin points.

Turn layout:

    ❯ how does auth work?                  <- typed at the prompt
    ┊ scanning the routes first…           <- thinking, dim, per line
    answer streams here as plain text…
    → running 3 tools: read(auth.py), read(config.py), bash(uv run pytest -q)
    ✓ read (auth.py)  0.3s                 <- replaces the running line
    ✓ read (config.py)  0.2s
    ✓ bash (uv run pytest -q)  8.1s
    ↑1,234 ↓567 tokens                     <- dim usage, end of turn
    ────────────────────────────────       <- divider at end of turn

Design rules:
  - Every tool call of a batch gets its own result line (no grouping).
  - All color decisions live in Palette; when stdout is not a TTY every
    code is empty, so call sites never branch.
  - Oneshot mode never constructs a Display — stdout stays clean for scripts.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
import time
from dataclasses import dataclass

from ..utils import oneline, tool_summary


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
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    SOFT_CYAN = "\033[36m"

    _STYLES = {
        "bold": BOLD,
        "dim": DIM,
        "muted": GRAY,
        "success": GREEN,
        "error": RED,
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


def _args_dict(args_raw: str) -> dict:
    try:
        return json.loads(args_raw) if args_raw else {}
    except Exception:
        return {}


# ─── renderer ───────────────────────────────────────────────────────────────


@dataclass
class _BatchCall:
    """One tool call of the currently running batch."""

    call_id: str  # tool_call id — the reliable match key
    name: str
    args: dict  # parsed arguments (grouping/summary source)
    started: float
    ok: bool = True
    error: str = ""
    elapsed: float = 0.0


class Display:
    """Streaming turn renderer. Attach to LoopConfig.emit (REPL only)."""

    # Streamed chunks are batched to this interval before flushing — one
    # syscall per token is measurable on slow terminals, while <50ms of
    # buffering is imperceptible. Event boundaries always flush.
    _FLUSH_INTERVAL = 0.05

    def __init__(self, input_=None) -> None:
        self._tty = sys.stdout.isatty()
        Palette.enabled = self._tty
        self._width = self._terminal_width()
        self._input = input_
        self._section: str | None = None  # "thinking" | "answer" | None
        self._partial = ""               # unflushed thinking line buffer
        self._at_line_start = True       # is stdout at the start of a fresh line?
        self._usage = None
        self._last_flush = time.monotonic()
        self._batch: list[_BatchCall] | None = None
        self._running_line = False       # is the "→ running…" line the last line?

    # ── public helpers for the REPL (non-streaming) ──

    async def prompt(self) -> str:
        if self._input is not None:
            return await self._input.prompt()
        return (await asyncio.to_thread(input, "❯ ")).strip()

    def print(self, text: str = "", *, end: str = "\n", flush: bool = True) -> None:
        """Raw print (bypasses info/error styling)."""
        self._print(text, end=end, flush=flush)

    def info(self, text: str) -> None:
        self._print(Palette.paint(text, "dim"))

    def warn(self, text: str) -> None:
        self._print(Palette.paint(f"! {text}", "muted"))

    def error(self, text: str) -> None:
        self._print(Palette.paint(f"✗ {text}", "error"))

    def divider(self) -> None:
        """Separator line shown between turns, before the next prompt."""
        self._print(Palette.paint("─" * min(self._width, 48), "muted"))

    def clear_screen(self) -> None:
        import os

        os.system("cls" if os.name == "nt" else "clear")

    # ── event dispatch (LoopConfig.emit callback) ──

    def __call__(self, event) -> None:
        handler = _HANDLERS.get(event.type)
        if handler:
            handler(self, event)

    # ── output primitives ──

    def _print(self, text: str = "", *, end: str = "\n", flush: bool = True) -> None:
        print(text, end=end, flush=flush)
        if flush:
            self._last_flush = time.monotonic()
        self._at_line_start = end == "\n" or text.endswith("\n")
        self._running_line = False  # any output invalidates the in-place replace

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
        # Only terminate a streamed answer line — never print a blank line.
        if self._section == "answer" and not self._at_line_start:
            self._print()
        self._section = None

    # ── tool batch rendering ──

    def _start_tool_batch(self, tool_calls: list[dict]) -> None:
        """A new batch of parallel tool calls is about to run."""
        calls = []
        for tc in tool_calls:
            fn = tc.get("function", {})
            calls.append(
                _BatchCall(
                    call_id=tc.get("id", ""),
                    name=fn.get("name", "?"),
                    args=_args_dict(fn.get("arguments", "")),
                    started=0.0,
                )
            )
        if not calls:
            return
        self._batch = calls
        total = len(calls)
        label = "running 1 tool" if total == 1 else f"running {total} tools"
        parts = []
        for c in calls:
            summary = tool_summary(c.name, c.args)
            parts.append(f"{c.name}({summary})" if summary else c.name)
        line = f"→ {label}: {', '.join(parts)}"
        self._print(Palette.paint(line, "dim"))
        self._running_line = True

    def _flush_tool_batch(self) -> None:
        """Print one result line per call, replacing the running line in place."""
        batch = self._batch
        self._batch = None
        if not batch:
            return
        # Replace the "→ running…" line only if it is still the last line.
        if self._tty and self._running_line:
            self._print("\033[1A\033[K", end="", flush=True)
        self._running_line = False
        for call in batch:
            label = call.name
            summary = tool_summary(call.name, call.args)
            if summary:
                label += f" ({summary})"
            parts = [
                Palette.paint("✓" if call.ok else "✗", "success" if call.ok else "error"),
                label,
            ]
            if call.ok:
                if call.elapsed >= 0.1:
                    parts.append(Palette.paint(f"{call.elapsed:.1f}s", "dim"))
            elif call.error:
                parts.append(Palette.paint(call.error, "error"))
            self._print(" ".join(parts))

    # ── event handlers ──

    def _on_iter_start(self, event) -> None:
        if event.iteration > 0:
            self._close_section()

    def _on_stream_chunk(self, event) -> None:
        chunk = event.chunk
        # Debounced flush for text deltas — see _FLUSH_INTERVAL.
        flush = time.monotonic() - self._last_flush >= self._FLUSH_INTERVAL
        if chunk.type == "text_delta" and chunk.text:
            self._enter("answer")
            self._print(chunk.text, end="", flush=flush)
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
        self._flush_tool_batch()  # safety: a pending batch flushes here too
        response = event.response
        if response is not None and getattr(response, "tool_calls", None):
            self._start_tool_batch(response.tool_calls)

    def _on_tool_start(self, event) -> None:
        self._close_section()
        tc = event.tool_call or {}
        call_id = tc.get("id", "")
        for call in self._batch or []:
            if call.call_id == call_id and call.started == 0.0:
                call.started = time.monotonic()
                break

    def _on_tool_complete(self, event) -> None:
        tc = event.tool_call or {}
        call_id = tc.get("id", "")
        content = (event.tool_result or {}).get("content", "")
        ok = not content.startswith("[error]")
        for call in self._batch or []:
            if call.call_id == call_id and call.started != 0.0:
                call.ok = ok
                call.elapsed = time.monotonic() - call.started
                if not ok:
                    call.error = oneline(content[len("[error]") :].strip(), 90)
                call.started = -1.0  # mark consumed
                break

    def _on_iter_end(self, event) -> None:
        # End-of-turn summary: flush tools, then the dim usage line.
        self._flush_tool_batch()
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

    # ── history re-rendering (session resume) ──

    def render_messages(self, messages: list[dict]) -> None:
        """Re-render a message history as if it were live output."""
        i = 0
        while i < len(messages):
            msg = messages[i]
            role = msg.get("role")
            if role == "user":
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        p.get("text", "[image]") for p in content if isinstance(p, dict)
                    )
                self._print(Palette.paint(f"❯ {content}", "bold"))
                i += 1
            elif role == "assistant":
                if msg.get("reasoning_content"):
                    for line in str(msg["reasoning_content"]).splitlines():
                        self._print(Palette.paint(f"┊ {line}", "dim"))
                tcs = msg.get("tool_calls")
                if msg.get("content") and not tcs:
                    self._print(str(msg["content"]))
                j = i + 1
                tool_msgs = []
                while j < len(messages) and messages[j].get("role") == "tool":
                    tool_msgs.append(messages[j])
                    j += 1
                if tcs:
                    self._render_tool_calls(tcs, tool_msgs)
                    i = j
                else:
                    i += 1
            else:
                i += 1

    def _render_tool_calls(self, tcs: list[dict], tool_messages: list[dict]) -> None:
        """Re-render tool calls from history, one ✓/✗ line per call."""
        errors: dict[str, str] = {}
        for msg in tool_messages:
            content = msg.get("content", "")
            if content.startswith("[error]"):
                tcid = msg.get("tool_call_id", "")
                for tc in tcs:
                    if tc.get("id") == tcid:
                        errors[tc.get("function", {}).get("name", "?")] = content[:80]
                        break
        for tc in tcs:
            fn = tc.get("function", {})
            name = fn.get("name", "?")
            summary = tool_summary(name, _args_dict(fn.get("arguments", "")))
            label = name + (f" ({summary})" if summary else "")
            if name in errors:
                msg = errors[name]
                if msg.startswith("[error] "):
                    msg = msg[len("[error] ") :]
                self._print(
                    " ".join(
                        [
                            Palette.paint("✗", "error"),
                            label,
                            Palette.paint(msg, "error"),
                        ]
                    )
                )
            else:
                self._print(
                    " ".join(
                        [Palette.paint("✓", "success"), label]
                    )
                )


_HANDLERS = {
    "iter_start": Display._on_iter_start,
    "stream_chunk": Display._on_stream_chunk,
    "response": Display._on_response,
    "tool_start": Display._on_tool_start,
    "tool_complete": Display._on_tool_complete,
    "iter_end": Display._on_iter_end,
    "error": Display._on_error,
}


__all__ = ["Palette", "Display"]
