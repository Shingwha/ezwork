"""Tests for the slash-command system and CLIApp dispatch."""

from __future__ import annotations

import pytest

from ezwork.app.cli.commands import (
    Command,
    CommandRegistry,
    CommandResult,
    commands as builtin_commands,
)
from ezwork.app.cli.input import PasteStore


# ─── CommandRegistry ────────────────────────────────────────────────────────


def test_registry_get_by_name_and_alias() -> None:
    reg = CommandRegistry()
    cmd = Command("/quit", "exit", aliases=("/exit",), handler=None)
    reg.register(cmd)
    assert reg.get("/quit") is cmd
    assert reg.get("/exit") is cmd
    assert reg.get("/QUIT") is cmd  # case-insensitive
    assert reg.get("/nope") is None


def test_registry_all_sorted() -> None:
    reg = CommandRegistry()
    reg.register(Command("/zeta", "z"))
    reg.register(Command("/alpha", "a"))
    assert [c.name for c in reg.all()] == ["/alpha", "/zeta"]


def test_builtin_commands_registered() -> None:
    reg = CommandRegistry()
    for c in builtin_commands:
        reg.register(c)
    for name in ("/quit", "/help", "/clear", "/sessions", "/resume", "/copy", "/export", "/model"):
        assert reg.get(name) is not None


# ─── CommandResult ──────────────────────────────────────────────────────────


def test_command_result_sentinels() -> None:
    assert CommandResult.CONTINUE.kind == "continue"
    assert CommandResult.EXIT.kind == "exit"
    chat = CommandResult.chat("hello")
    assert chat.kind == "chat"
    assert chat.prompt == "hello"


# ─── PasteStore ─────────────────────────────────────────────────────────────


def test_paste_store_roundtrip() -> None:
    ps = PasteStore()
    marker = ps.put("multi\nline content")
    assert ps.resolve(f"prefix {marker} suffix") == "prefix multi\nline content suffix"
    assert ps.resolve("no marker here") == "no marker here"


# ─── Input degradation ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_input_falls_back_when_prompt_toolkit_fails(monkeypatch) -> None:
    """Any prompt_toolkit initialisation failure (e.g. Git Bash on Windows)
    degrades to plain input() so the REPL stays usable."""
    from ezwork.app.cli.input import Input

    def _boom(*args, **kwargs):
        raise RuntimeError("NoConsoleScreenBufferError")

    monkeypatch.setattr("builtins.input", lambda prompt="": "hi there")
    monkeypatch.setattr(
        "prompt_toolkit.PromptSession", _boom  # imported lazily inside Input
    )
    inp = Input(registry=None)
    assert await inp.prompt() == "hi there"
    assert inp._fallback is True
