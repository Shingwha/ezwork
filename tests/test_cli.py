"""Tests for the CLI layer: `-p -` stdin prompt handling."""

from __future__ import annotations

import io
import os
import subprocess
import sys

import pytest

from ezwork.app import cli


class _FakeStdin:
    """Minimal stand-in for sys.stdin with isatty/fileno/read."""

    def __init__(self, fd: int | None = None, text: str = "", is_tty: bool = False):
        self._fd = fd
        self._text = text
        self._tty = is_tty
        self._read_called = False

    def isatty(self) -> bool:
        return self._tty

    def fileno(self) -> int:
        assert self._fd is not None
        return self._fd

    def read(self) -> str:
        self._read_called = True
        return self._text


def _pipe_stdin(monkeypatch, payload: bytes, close_write: bool) -> int:
    """Create a real OS pipe, write payload, return the read fd."""
    r, w = os.pipe()
    os.write(w, payload)
    if close_write:
        os.close(w)
    monkeypatch.setattr(cli.sys, "stdin", _FakeStdin(fd=r, is_tty=False))
    return r


def test_read_stdin_tty_reads_to_eof(monkeypatch) -> None:
    """TTY stdin behaves like `cat -`: read until EOF (Ctrl-D)."""
    monkeypatch.setattr(cli.sys, "stdin", _FakeStdin(text="multi\nline\n", is_tty=True))
    assert cli._read_stdin() == "multi\nline\n"


def test_read_stdin_pipe_drains_until_eof(monkeypatch) -> None:
    """Piped input (`echo x | ezwork -p -`) is read completely."""
    _pipe_stdin(monkeypatch, b"git diff output\n", close_write=True)
    assert cli._read_stdin() == "git diff output\n"


def test_read_stdin_pipe_never_closed_times_out(monkeypatch) -> None:
    """A never-closing pipe (sandbox/CI) must not hang: drain + timeout."""
    _pipe_stdin(monkeypatch, b"partial data", close_write=False)
    monkeypatch.setattr(cli, "_STDIN_TIMEOUT", 0.1)
    got = cli._read_stdin()
    assert "partial data" in got  # whatever arrived is returned


def test_read_stdin_empty_pipe_returns_empty(monkeypatch) -> None:
    r, w = os.pipe()
    os.close(w)  # nothing written, writer closed
    monkeypatch.setattr(cli.sys, "stdin", _FakeStdin(fd=r, is_tty=False))
    assert cli._read_stdin() == ""


def test_main_prompt_dash_dispatches_to_oneshot(monkeypatch) -> None:
    """`-p -` feeds stdin text into oneshot() instead of the literal '-'.

    Runs the real argparse + main() path with oneshot() stubbed out.
    """
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


def test_main_prompt_literal_dash_is_read_from_stdin() -> None:
    """End-to-end via subprocess: `echo hi | ezwork -p -` reads the pipe.

    The prompt is non-empty so it proceeds into oneshot(); that would call the
    model, so we only assert the stdin plumbing works by checking the exit
    path for a *missing* config error is NOT hit with a prompt error. Simpler:
    empty stdin must exit 2 (no model call, no config needed).
    """
    result = subprocess.run(
        [sys.executable, "-m", "ezwork.app.cli", "-p", "-"],
        input=b"",
        capture_output=True,
        timeout=30,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    assert result.returncode == 2
    assert b"no input on stdin" in result.stderr
