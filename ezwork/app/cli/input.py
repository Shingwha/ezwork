"""Input — REPL prompt with prompt_toolkit (primary) and input() fallback.

prompt_toolkit is a primary dependency: it provides multi-line input
(Esc+Enter / Ctrl-J inserts a newline), Tab completion for slash commands,
bracketed-paste handling and a stable history. If the library is missing or
fails to initialise (exotic terminals), we fall back to plain input() so the
REPL stays usable.

Large pastes (>5 lines or >200 chars) are replaced on screen by a
`[paste:N]` marker and resolved on submit — keeps the terminal responsive
for big diffs while the model still receives the full content.
"""

from __future__ import annotations

import asyncio
import re

_PASTE_LINE_THRESHOLD = 5
_PASTE_CHAR_THRESHOLD = 200
_PASTE_RE = re.compile(r"\[paste:(\d+)]")


class PasteStore:
    """Indexed store for pasted content with marker resolution."""

    def __init__(self) -> None:
        self._store: dict[int, str] = {}
        self._counter = 0

    def put(self, data: str) -> str:
        self._counter += 1
        self._store[self._counter] = data
        return f"[paste:{self._counter}]"

    def resolve(self, text: str) -> str:
        def _repl(m: re.Match) -> str:
            return self._store.get(int(m.group(1)), m.group(0))

        return _PASTE_RE.sub(_repl, text)


class _SlashCompleter:
    """Prefix-matches /commands from a CommandRegistry.

    Duck-typed for prompt_toolkit's Completer protocol (get_completions_async)
    so prompt_toolkit is only imported lazily, inside the prompt session.
    """

    def __init__(self, registry) -> None:
        self._registry = registry

    async def get_completions_async(self, document, complete_event):
        text = document.text
        if not text.startswith("/") or " " in text:
            return
        from prompt_toolkit.completion import Completion

        for cmd in self._registry.all():
            if cmd.name.startswith(text):
                yield Completion(
                    cmd.name,
                    start_position=-len(text),
                    display_meta=cmd.description,
                )


class Input:
    """REPL input. `prompt()` returns one submitted string (multi-line
    content included, pastes resolved, stripped)."""

    def __init__(self, registry=None, ps1: str = "❯") -> None:
        self._registry = registry
        self._ps1 = ps1
        self._session = None  # PromptSession (lazy)
        self._fallback = False
        self._pastes = PasteStore()

    def _ensure_session(self) -> None:
        if self._session is not None or self._fallback:
            return
        try:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.key_binding import KeyBindings
            from prompt_toolkit.keys import Keys
        except ImportError:
            self._fallback = True
            return

        kb = KeyBindings()

        @kb.add("tab")
        def _(event):
            buf = event.current_buffer
            if buf.complete_state:
                self._apply_completion(buf)
            else:
                buf.start_completion(select_first=True)

        @kb.add("enter")
        def _(event):
            buf = event.current_buffer
            if buf.complete_state and buf.complete_state.completions:
                self._apply_completion(buf)
            else:
                buf.validate_and_handle()

        @kb.add("escape", "enter")
        def _(event):
            event.current_buffer.insert_text("\n")

        @kb.add("c-j")
        def _(event):
            event.current_buffer.insert_text("\n")

        @kb.add(Keys.BracketedPaste)
        def _(event):
            self._handle_paste(event)

        completer = _SlashCompleter(self._registry) if self._registry else None
        try:
            self._session = PromptSession(
                completer=completer,
                complete_while_typing=False,
                key_bindings=kb,
            )
        except Exception:
            # Any initialisation failure (e.g. NoConsoleScreenBufferError on
            # Git Bash / MSYS terminals) degrades to plain input() — the REPL
            # must stay usable everywhere.
            self._session = None
            self._fallback = True

    @staticmethod
    def _apply_completion(buf) -> None:
        completion = (
            buf.complete_state.current_completion or buf.complete_state.completions[0]
        )
        buf.apply_completion(completion)

    def _handle_paste(self, event) -> None:
        data = event.data.replace("\r\n", "\n").replace("\r", "\n")
        lines, chars = data.count("\n") + 1, len(data)
        if lines < _PASTE_LINE_THRESHOLD or chars < _PASTE_CHAR_THRESHOLD:
            event.current_buffer.insert_text(data)
            return
        event.current_buffer.insert_text(self._pastes.put(data))

    def _fallback_prompt(self, default: str = "") -> str:
        return input(f"{self._ps1} ")

    async def prompt(self, default: str = "") -> str:
        """Return one submitted input string (stripped, pastes resolved)."""
        self._ensure_session()
        if self._fallback or self._session is None:
            return (await asyncio.to_thread(self._fallback_prompt, default)).strip()
        text = await self._session.prompt_async(f"{self._ps1} ", default=default)
        text = self._pastes.resolve(text)
        # Sanitize surrogates from prompt_toolkit on Windows
        return text.encode("utf-16-le", errors="surrogatepass").decode(
            "utf-16-le", errors="replace"
        ).strip()


__all__ = ["Input", "PasteStore"]
