"""Tests for the CLI layer: stdin handling for `-p` / `-p -`.

Semantics follow the Claude Code convention (`cat file | claude -p "query"`):
piped stdin is appended to a `-p "query"` prompt as context, while `-p -`
treats stdin as the whole prompt.
"""

from __future__ import annotations

import os
import time

import pytest

import ezwork.app.cli as cli  # entry: main, oneshot, _enable_ansi
from ezwork.app.cli import app as cli_app  # build_agent
from ezwork.app.cli import stdin as cli_stdin  # _read_piped_stdin / _prompt_from_stdin


class _FakeStdin:
    """Minimal stand-in for sys.stdin with isatty/fileno/read."""

    def __init__(self, fd: int | None = None, text: str = "", is_tty: bool = False):
        self._fd = fd
        self._text = text
        self._tty = is_tty

    def isatty(self) -> bool:
        return self._tty

    def fileno(self) -> int:
        assert self._fd is not None
        return self._fd

    def read(self) -> str:
        return self._text


def _pipe_stdin(monkeypatch, payload: bytes, close_write: bool) -> int:
    """Create a real OS pipe, write payload, return the read fd."""
    r, w = os.pipe()
    os.write(w, payload)
    if close_write:
        os.close(w)
    monkeypatch.setattr(cli_stdin.sys, "stdin", _FakeStdin(fd=r, is_tty=False))
    return r


# ─── _read_piped_stdin ────────────────────────────────────────────────────


def test_piped_stdin_drains_until_eof(monkeypatch) -> None:
    """Piped input (`echo x | ezwork`) is read completely."""
    _pipe_stdin(monkeypatch, b"git diff output\n", close_write=True)
    assert cli_stdin._read_piped_stdin() == "git diff output\n"


def test_piped_stdin_never_closed_times_out(monkeypatch) -> None:
    """A never-closing pipe (sandbox/CI) must not hang: drain + timeout."""
    _pipe_stdin(monkeypatch, b"partial data", close_write=False)
    monkeypatch.setattr(cli_stdin, "_STDIN_TIMEOUT", 0.1)
    got = cli_stdin._read_piped_stdin()
    assert "partial data" in got  # whatever arrived is returned


def test_piped_stdin_silent_open_pipe_returns_quickly(monkeypatch) -> None:
    """A silent inherited pipe (CI, sub-agents) must not wait out the full
    2s drain window — the first-byte grace period is all it costs."""
    r, w = os.pipe()  # nothing written, writer kept open
    monkeypatch.setattr(cli_stdin.sys, "stdin", _FakeStdin(fd=r, is_tty=False))
    start = time.monotonic()
    assert cli_stdin._read_piped_stdin(first_byte_timeout=0.05) is None
    assert time.monotonic() - start < 1.0


def test_piped_stdin_tty_returns_none(monkeypatch) -> None:
    """Interactive terminals must never block or consume stdin."""
    monkeypatch.setattr(cli_stdin.sys, "stdin", _FakeStdin(text="ignored", is_tty=True))
    assert cli_stdin._read_piped_stdin() is None


# ─── _prompt_from_stdin (`-p -`) ──────────────────────────────────────────


def test_prompt_from_stdin_pipe(monkeypatch) -> None:
    _pipe_stdin(monkeypatch, b"summarize this diff", close_write=True)
    assert cli_stdin._prompt_from_stdin() == "summarize this diff"


def test_prompt_from_stdin_tty_reads_to_eof(monkeypatch) -> None:
    """TTY stdin behaves like `cat -`: read until EOF (Ctrl-D)."""
    monkeypatch.setattr(cli_stdin.sys, "stdin", _FakeStdin(text="multi\nline\n", is_tty=True))
    assert cli_stdin._prompt_from_stdin() == "multi\nline\n"


@pytest.mark.parametrize(
    "payload,close_write",
    [(b"", True), (b"", False)],
    ids=["empty-closed-pipe", "silent-open-pipe"],
)
def test_prompt_from_stdin_empty_input_exits_2(monkeypatch, payload, close_write) -> None:
    """`-p -` with empty stdin exits 2 — whether the pipe is closed or a
    silent (open) pipe, since the whole prompt is expected on stdin."""
    _pipe_stdin(monkeypatch, payload, close_write=close_write)
    monkeypatch.setattr(cli_stdin, "_STDIN_TIMEOUT", 0.05)
    with pytest.raises(SystemExit) as exc:
        cli_stdin._prompt_from_stdin()
    assert exc.value.code == 2


# ─── _enable_ansi / build_agent ───────────────────────────────────────────


def test_enable_ansi_only_spawns_on_tty(monkeypatch) -> None:
    """The `os.system("")` cmd.exe spawn must only run when stdout is a TTY —
    piped/oneshot output never emits ANSI and shouldn't pay for it."""
    import os as _os

    calls = {"n": 0}
    monkeypatch.setattr(cli.sys, "platform", "win32")
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(
        _os, "system", lambda _cmd: calls.__setitem__("n", calls["n"] + 1)
    )
    cli._enable_ansi()
    assert calls["n"] == 1
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: False)
    cli._enable_ansi()
    assert calls["n"] == 1  # no spawn for non-TTY stdout


def test_build_agent_warms_up_provider(monkeypatch) -> None:
    """build_agent() starts provider.warmup() so the SDK import overlaps
    startup instead of sitting in the first stream() call."""
    from ezwork.app.config import Config
    from ezwork.core import ToolRegistry

    calls = {"n": 0}

    class FakeProvider:
        model = "m"

        def warmup(self):
            calls["n"] += 1

    monkeypatch.setattr(Config, "build_provider", lambda self: FakeProvider())
    monkeypatch.setattr(cli_app, "build_system_prompt", lambda **kw: "system")
    monkeypatch.setattr(cli_app, "_build_tools", lambda timeout: ToolRegistry())
    agent = cli_app.build_agent(Config(), render=False)
    assert calls["n"] == 1
    assert agent is not None


# ─── main() dispatch ──────────────────────────────────────────────────────


def test_main_prompt_dash_dispatches_to_oneshot(monkeypatch) -> None:
    """`-p -` feeds stdin text into oneshot() instead of the literal '-'."""
    calls: dict = {}

    async def fake_oneshot(prompt: str, **kwargs) -> int:
        calls["prompt"] = prompt
        calls["kwargs"] = kwargs
        return 0

    _pipe_stdin(monkeypatch, b"summarize this diff", close_write=True)
    monkeypatch.setattr(cli, "oneshot", fake_oneshot)
    monkeypatch.setattr(cli.sys, "argv", ["ezwork", "-p", "-"])
    assert cli.main() == 0
    assert calls["prompt"] == "summarize this diff"
    assert calls["kwargs"]["session_id"] is None


def test_main_prompt_dash_empty_stdin_is_error(monkeypatch) -> None:
    """`-p -` with no input exits 2 without calling the model."""
    called = {"n": 0}

    async def fake_oneshot(prompt: str, **kwargs) -> int:
        called["n"] += 1
        return 0

    _pipe_stdin(monkeypatch, b"", close_write=True)
    monkeypatch.setattr(cli, "oneshot", fake_oneshot)
    monkeypatch.setattr(cli.sys, "argv", ["ezwork", "-p", "-"])
    assert cli.main() == 2
    assert called["n"] == 0


def test_main_prompt_query_merges_piped_context(monkeypatch) -> None:
    """`cat file | ezwork -p "query"` appends piped content as context."""
    calls: dict = {}

    async def fake_oneshot(prompt: str, **kwargs) -> int:
        calls["prompt"] = prompt
        return 0

    _pipe_stdin(monkeypatch, b"line1\nline2\n", close_write=True)
    monkeypatch.setattr(cli, "oneshot", fake_oneshot)
    monkeypatch.setattr(cli.sys, "argv", ["ezwork", "-p", "summarize"])
    assert cli.main() == 0
    assert calls["prompt"] == "summarize\n\nline1\nline2\n"


@pytest.mark.parametrize(
    "stdin_mode",
    ["tty", "empty-pipe"],
    ids=["tty", "empty-pipe"],
)
def test_main_prompt_query_without_content_is_unchanged(monkeypatch, stdin_mode) -> None:
    """`-p "query"` stays exactly the query when there is no piped content
    (interactive TTY or empty pipe)."""
    calls: dict = {}

    async def fake_oneshot(prompt: str, **kwargs) -> int:
        calls["prompt"] = prompt
        return 0

    if stdin_mode == "tty":
        monkeypatch.setattr(cli.sys, "stdin", _FakeStdin(text="", is_tty=True))
    else:
        _pipe_stdin(monkeypatch, b"", close_write=True)
    monkeypatch.setattr(cli, "oneshot", fake_oneshot)
    monkeypatch.setattr(cli.sys, "argv", ["ezwork", "-p", "hello"])
    assert cli.main() == 0
    assert calls["prompt"] == "hello"
