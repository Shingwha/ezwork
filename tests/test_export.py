"""Tests for session export (JSON / Markdown)."""

from __future__ import annotations

import json

from ezwork.app.export import export_session, render_session_md
from ezwork.app.session import Session, SessionStore


def _session() -> Session:
    s = Session.new("/work/dir", model="m", provider="p")
    s.messages = [
        {"role": "user", "content": "how does it work?"},
        {"role": "assistant", "content": "like this", "reasoning_content": "let me think"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "read", "arguments": '{"path": "a.py"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "file content"},
    ]
    return s


def test_export_json_roundtrips(tmp_path) -> None:
    path = export_session(_session(), tmp_path / "s.json", fmt="json", system_prompt="sp")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["system_prompt"] == "sp"
    assert len(data["messages"]) == 4
    assert data["model"] == "m"
    # A fresh store can load the exported file back as a session.
    store = SessionStore(tmp_path / "import")
    s = Session.from_dict({k: v for k, v in data.items() if k != "system_prompt"})
    store.save(s)
    assert store.load("/work/dir", s.id) is not None


def test_export_md_structure(tmp_path) -> None:
    path = export_session(_session(), tmp_path / "s.md", fmt="md", system_prompt="sp")
    md = path.read_text(encoding="utf-8")
    assert md.startswith("---\n")
    assert "id: session_" in md
    assert "## System prompt" in md and "```text\nsp\n```" in md
    assert "## User" in md and "how does it work?" in md
    assert "<details><summary>Reasoning</summary>" in md
    assert "### Tool: read" in md and '"path": "a.py"' in md
    assert "### Tool result" in md and "file content" in md


def test_render_session_md_empty_messages() -> None:
    md = render_session_md(Session.new("wd"), system_prompt="")
    assert "messages: 0" in md
    assert "## User" not in md
