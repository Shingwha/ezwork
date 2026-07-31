"""Tests for the CLI layer: stdin handling for `-p` / `-p -`.

Semantics follow the Claude Code convention (`cat file | claude -p "query"`):
piped stdin is appended to a `-p "query"` prompt as context, while `-p -`
treats stdin as the whole prompt.
"""

from __future__ import annotations

import os
import subprocess
import sys

from ezwork.app import cli


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
    monkeypatch.setattr(cli.sys, "stdin", _FakeStdin(fd=r, is_tty=False))
    return r


# ─── _read_piped_stdin ────────────────────────────────────────────────────


def test_piped_stdin_drains_until_eof(monkeypatch) -> None:
    """Piped input (`echo x | ezwork`) is read completely."""
    _pipe_stdin(monkeypatch, b"git diff output\n", close_write=True)
    assert cli._read_piped_stdin() == "git diff output\n"


def test_piped_stdin_never_closed_times_out(monkeypatch) -> None:
    """A never-closing pipe (sandbox/CI) must not hang: drain + timeout."""
    _pipe_stdin(monkeypatch, b"partial data", close_write=False)
    monkeypatch.setattr(cli, "_STDIN_TIMEOUT", 0.1)
    got = cli._read_piped_stdin()
    assert "partial data" in got  # whatever arrived is returned


def test_piped_stdin_empty_pipe_returns_none(monkeypatch) -> None:
    r, w = os.pipe()
    os.close(w)  # nothing written, writer closed
    monkeypatch.setattr(cli.sys, "stdin", _FakeStdin(fd=r, is_tty=False))
    assert cli._read_piped_stdin() is None


def test_piped_stdin_tty_returns_none(monkeypatch) -> None:
    """Interactive terminals must never block or consume stdin."""
    monkeypatch.setattr(cli.sys, "stdin", _FakeStdin(text="ignored", is_tty=True))
    assert cli._read_piped_stdin() is None


# ─── _prompt_from_stdin (`-p -`) ──────────────────────────────────────────


def test_prompt_from_stdin_pipe(monkeypatch) -> None:
    _pipe_stdin(monkeypatch, b"summarize this diff", close_write=True)
    assert cli._prompt_from_stdin() == "summarize this diff"


def test_prompt_from_stdin_tty_reads_to_eof(monkeypatch) -> None:
    """TTY stdin behaves like `cat -`: read until EOF (Ctrl-D)."""
    monkeypatch.setattr(cli.sys, "stdin", _FakeStdin(text="multi\nline\n", is_tty=True))
    assert cli._prompt_from_stdin() == "multi\nline\n"


def test_prompt_from_stdin_empty_exits_2(monkeypatch) -> None:
    _pipe_stdin(monkeypatch, b"", close_write=True)
    try:
        cli._prompt_from_stdin()
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected SystemExit(2)")


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


def test_main_prompt_query_without_pipe_is_unchanged(monkeypatch) -> None:
    """No piped stdin: `-p "query"` stays exactly the query."""
    calls: dict = {}

    async def fake_oneshot(prompt: str, **kwargs) -> int:
        calls["prompt"] = prompt
        return 0

    monkeypatch.setattr(cli.sys, "stdin", _FakeStdin(text="", is_tty=True))
    monkeypatch.setattr(cli, "oneshot", fake_oneshot)
    monkeypatch.setattr(cli.sys, "argv", ["ezwork", "-p", "hello"])
    assert cli.main() == 0
    assert calls["prompt"] == "hello"


def test_main_prompt_query_empty_pipe_is_unchanged(monkeypatch) -> None:
    """Empty piped stdin: `-p "query"` stays exactly the query."""
    calls: dict = {}

    async def fake_oneshot(prompt: str, **kwargs) -> int:
        calls["prompt"] = prompt
        return 0

    _pipe_stdin(monkeypatch, b"", close_write=True)
    monkeypatch.setattr(cli, "oneshot", fake_oneshot)
    monkeypatch.setattr(cli.sys, "argv", ["ezwork", "-p", "hello"])
    assert cli.main() == 0
    assert calls["prompt"] == "hello"


def test_main_prompt_dash_empty_stdin_e2e() -> None:
    """End-to-end: `ezwork -p - < /dev/null` exits 2 without a model call."""
    result = subprocess.run(
        [sys.executable, "-m", "ezwork.app.cli", "-p", "-"],
        input=b"",
        capture_output=True,
        timeout=30,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    assert result.returncode == 2
    assert b"no input on stdin" in result.stderr
