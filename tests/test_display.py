"""Tests for the Display renderer (non-TTY: Palette codes are empty)."""

from __future__ import annotations

import sys

from ezwork.app.cli.display import Display
from ezwork.core.event import (
    IterEndEvent,
    ResponseEvent,
    StreamChunkEvent,
    ToolCompleteEvent,
    ToolStartEvent,
)
from ezwork.core.provider import Response, StreamChunk, Usage


def _tc(call_id: str, name: str, arguments: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def test_thinking_streams_dim_lines(capsys) -> None:
    d = Display()
    d(StreamChunkEvent(chunk=StreamChunk.reasoning_delta("line one\nline two\n")))
    out = capsys.readouterr().out
    assert "┊ line one" in out
    assert "┊ line two" in out


def test_answer_streams_plain(capsys) -> None:
    d = Display()
    d(StreamChunkEvent(chunk=StreamChunk.text_delta("hello ")))
    d(StreamChunkEvent(chunk=StreamChunk.text_delta("world")))
    out = capsys.readouterr().out
    assert out == "hello world"


def test_tool_batch_groups_parallel_reads(capsys) -> None:
    """Two parallel reads each get their own ✓ line (no grouping)."""
    d = Display()
    response = Response(
        content=None,
        tool_calls=[_tc("c1", "read", '{"path": "a.py"}'), _tc("c2", "read", '{"path": "b.py"}')],
        finish_reason="tool_calls",
        usage=Usage(prompt_tokens=10, completion_tokens=5),
    )
    d(ResponseEvent(response=response, usage=response.usage))
    d(ToolStartEvent(tool_call=_tc("c1", "read", '{"path": "a.py"}')))
    d(ToolStartEvent(tool_call=_tc("c2", "read", '{"path": "b.py"}')))
    d(ToolCompleteEvent(tool_call=_tc("c1", "read", ""), tool_result={"role": "tool", "content": "ok"}))
    d(ToolCompleteEvent(tool_call=_tc("c2", "read", ""), tool_result={"role": "tool", "content": "ok"}))
    d(IterEndEvent(final_content=None))

    out = capsys.readouterr().out
    assert "✓ read (a.py)" in out
    assert "✓ read (b.py)" in out
    assert "read×2" not in out
    # No batch announcement line and no placeholders on non-TTY output.
    assert "running" not in out
    assert "· read" not in out
    # NB: no usage line here — tool-call iterations never print it (the
    # final text iteration does, see test_usage_line_on_final_iteration).


def test_tool_batch_tty_placeholder_lit_in_place(monkeypatch, capsys) -> None:
    """TTY: dim placeholders up front, each replaced in place on completion."""
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    d = Display()
    response = Response(
        content=None,
        tool_calls=[_tc("c1", "read", '{"path": "a.py"}'), _tc("c2", "read", '{"path": "b.py"}')],
        finish_reason="tool_calls",
    )
    d(ResponseEvent(response=response, usage=None))
    out = capsys.readouterr().out
    assert "· read (a.py)…" in out
    assert "· read (b.py)…" in out

    d(ToolStartEvent(tool_call=_tc("c1", "read", '{"path": "a.py"}')))
    d(ToolCompleteEvent(tool_call=_tc("c1", "read", ""), tool_result={"role": "tool", "content": "ok"}))
    out = capsys.readouterr().out
    # Row 1 of 2: move up 2 rows, clear, write the ✓ line, back to bottom.
    assert "\x1b[2A\x1b[K" in out and "✓" in out and "read (a.py)" in out
    assert "\x1b[2B\r" in out

    d(ToolStartEvent(tool_call=_tc("c2", "read", '{"path": "b.py"}')))
    d(ToolCompleteEvent(tool_call=_tc("c2", "read", ""), tool_result={"role": "tool", "content": "ok"}))
    out = capsys.readouterr().out
    # Row 2 of 2: move up 1 row, clear, write the ✓ line, back to bottom.
    assert "\x1b[1A\x1b[K" in out and "✓" in out and "read (b.py)" in out
    d(IterEndEvent(final_content=None))


def test_tool_batch_shows_error(capsys) -> None:
    d = Display()
    response = Response(
        content=None,
        tool_calls=[_tc("c1", "edit", '{"path": "x.py"}')],
        finish_reason="tool_calls",
    )
    d(ResponseEvent(response=response, usage=None))
    d(ToolStartEvent(tool_call=_tc("c1", "edit", '{"path": "x.py"}')))
    d(
        ToolCompleteEvent(
            tool_call=_tc("c1", "edit", ""),
            tool_result={"role": "tool", "content": "[error] old_string not found"},
        )
    )
    d(IterEndEvent(final_content=None))

    out = capsys.readouterr().out
    assert "✗ edit (x.py)" in out
    assert "old_string not found" in out


def test_usage_line_on_final_iteration(capsys) -> None:
    d = Display()
    d(
        ResponseEvent(
            response=Response(content="done", finish_reason="stop"),
            usage=Usage(prompt_tokens=1234, completion_tokens=56),
        )
    )
    d(IterEndEvent(final_content="done"))
    out = capsys.readouterr().out
    assert "↑1,234 ↓56 tokens" in out


# ─── history re-rendering ───────────────────────────────────────────────────


def test_render_messages_full_history(capsys) -> None:
    d = Display()
    d.render_messages(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "plain answer"},
            {
                "role": "assistant",
                "content": "thinking text",
                "reasoning_content": "reasoning line",
                "tool_calls": [_tc("c1", "bash", '{"command": "ls"}')],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        ]
    )
    out = capsys.readouterr().out
    assert "❯ hi" in out
    assert "plain answer" in out
    assert "┊ reasoning line" in out
    assert "✓ bash (ls)" in out


def test_render_messages_tool_error(capsys) -> None:
    d = Display()
    d.render_messages(
        [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [_tc("c1", "bash", '{"command": "nope"}')],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "[error] command not found"},
        ]
    )
    out = capsys.readouterr().out
    assert "✗ bash (nope)" in out
    assert "command not found" in out
