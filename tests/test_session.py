"""Tests for the JSONL session log (SessionStore / SessionManager)."""

from __future__ import annotations

from pathlib import Path

from ezwork.app.session import Session, SessionManager, SessionStore, _extract_title, _parse_line


def _store(tmp_path) -> SessionStore:
    return SessionStore(tmp_path / "sessions")


def _manager(tmp_path, **kwargs) -> SessionManager:
    return SessionManager("wd", _store(tmp_path), **kwargs)


def _meta(session: Session) -> dict:
    return {
        "type": "meta",
        "schema": "ezwork.session.v1",
        "id": session.id,
        "created_at": session.created_at,
        "workdir": session.workdir,
        "title": session.title,
        "model": session.model,
        "provider": session.provider,
    }


def _round(session: Session, round_no: int, *, title: str = "", msgs: int = 0,
           system_prompt: str = "", updated_at: str = "2026-01-01T00:00:00") -> dict:
    return {
        "type": "round",
        "round": round_no,
        "updated_at": updated_at,
        "title": title,
        "message_count": msgs,
        "usage_total": {"prompt_tokens": 10, "completion_tokens": 5},
        "tool_calls": {"read": 1},
        "tool_failures": {},
        "interrupted": False,
        "error": None,
        "system_prompt": system_prompt,
        "system_prompt_hash": "h" if system_prompt else "",
    }


def _write_session(store: SessionStore, s: Session, *, msgs: list[dict] | None = None,
                   system_prompt: str = "", title: str = "") -> None:
    events = [_meta(s)]
    for m in msgs or []:
        events.append({"type": "message", **m})
    events.append(_round(s, 1, title=title or s.title, msgs=len(msgs or []), system_prompt=system_prompt))
    store.append(s.workdir, s.id, events)


# ─── store ──────────────────────────────────────────────────────────────────


def test_store_append_load_roundtrip(tmp_path) -> None:
    store = _store(tmp_path)
    s = Session.new("wd", model="m", provider="p")
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    _write_session(store, s, msgs=msgs, system_prompt="sys")

    loaded = store.load("wd", s.id)
    assert loaded is not None
    assert loaded.id == s.id
    assert loaded.model == "m"
    assert loaded.provider == "p"
    assert loaded.messages == msgs
    assert loaded.system_prompt == "sys"
    assert loaded.usage_total == {"prompt_tokens": 10, "completion_tokens": 5}
    assert loaded.tool_stats == {"read": 1}


def test_store_load_missing_returns_none(tmp_path) -> None:
    assert _store(tmp_path).load("wd", "session_nope") is None


def test_store_list_newest_first_and_derives_title(tmp_path) -> None:
    store = _store(tmp_path)
    a = Session.new("wd")
    _write_session(store, a, title="alpha")
    b = Session.new("wd")
    events = [_meta(b), _round(b, 1, title="beta", updated_at="2026-01-02T00:00:00")]
    store.append(b.workdir, b.id, events)

    sessions = store.list("wd")
    assert [s.id for s in sessions] == [b.id, a.id]
    assert sessions[0].title == "beta"
    assert sessions[0].round_count == 1
    assert sessions[0].message_count == 0


def test_store_delete(tmp_path) -> None:
    store = _store(tmp_path)
    s = Session.new("wd")
    _write_session(store, s)
    assert store.load("wd", s.id) is not None
    assert store.delete("wd", s.id) is True
    assert store.load("wd", s.id) is None
    assert store.delete("wd", s.id) is False


def test_store_tolerates_malformed_lines(tmp_path) -> None:
    store = _store(tmp_path)
    s = Session.new("wd")
    _write_session(store, s, msgs=[{"role": "user", "content": "keep me"}])
    path = store.path_for("wd", s.id)
    # A corrupted line in the middle and a truncated trailing line (interrupted
    # write) must not lose the rest.
    raw = path.read_text(encoding="utf-8").rstrip("\n")
    lines = raw.split("\n")
    lines.insert(1, "{not json at all")
    lines.append('{"type": "message", "role": "assistant", "content": "half')
    path.write_text("\n".join(lines), encoding="utf-8")

    loaded = store.load("wd", s.id)
    assert loaded is not None
    assert len(loaded.messages) == 1
    assert loaded.messages[0]["content"] == "keep me"


def test_store_empty_or_unreadable(tmp_path) -> None:
    store = _store(tmp_path)
    s = Session.new("wd")
    store.append(s.workdir, s.id, [_meta(s)])
    loaded = store.load("wd", s.id)
    assert loaded is not None and loaded.id == s.id and loaded.messages == []


def test_parse_line_rejects_garbage() -> None:
    assert _parse_line("not json") is None
    assert _parse_line('{"type": "round"}') == {"type": "round"}
    assert _parse_line("") is None


# ─── manager ────────────────────────────────────────────────────────────────


def test_manager_lazy_create_and_save(tmp_path) -> None:
    mgr = _manager(tmp_path)
    mgr.create()
    assert mgr.active is not None
    assert not (Path(tmp_path) / "sessions").exists()  # nothing on disk yet

    mgr.save(
        [{"role": "user", "content": "hi"}],
        usage_total={"prompt_tokens": 3, "completion_tokens": 1},
        system_prompt="sys",
    )
    loaded = mgr.list()[0]
    assert loaded.id == mgr.active_id
    assert loaded.messages == [{"role": "user", "content": "hi"}]
    assert loaded.system_prompt == "sys"
    assert loaded.usage_total == {"prompt_tokens": 3, "completion_tokens": 1}


def test_manager_save_appends_incrementally(tmp_path) -> None:
    mgr = _manager(tmp_path)
    mgr.create()
    mgr.save([{"role": "user", "content": "one"}], system_prompt="s1")
    path = _store(tmp_path).path_for("wd", mgr.active_id)
    lines_after_first = len(path.read_text(encoding="utf-8").splitlines())
    assert lines_after_first == 3  # meta + message + round

    mgr.save(
        [{"role": "user", "content": "one"}, {"role": "assistant", "content": "two"}],
        system_prompt="s2",
    )
    assert len(path.read_text(encoding="utf-8").splitlines()) == lines_after_first + 2
    loaded = mgr.list()[0]
    assert [m["content"] for m in loaded.messages] == ["one", "two"]


def test_manager_keeps_one_system_prompt_per_round(tmp_path) -> None:
    mgr = _manager(tmp_path)
    mgr.create()
    mgr.save([{"role": "user", "content": "one"}], system_prompt="prompt-A")
    mgr.save(
        [{"role": "user", "content": "one"}, {"role": "assistant", "content": "two"}],
        system_prompt="prompt-B",
    )
    loaded = mgr.list()[0]
    assert loaded.round_count == 2
    assert [r["system_prompt"] for r in loaded.rounds] == ["prompt-A", "prompt-B"]
    assert loaded.system_prompt == "prompt-B"  # latest round wins on resume


def test_manager_save_noop_without_new_messages(tmp_path) -> None:
    mgr = _manager(tmp_path)
    mgr.create()
    msgs = [{"role": "user", "content": "one"}]
    mgr.save(msgs, system_prompt="s")
    path = _store(tmp_path).path_for("wd", mgr.active_id)
    before = len(path.read_text(encoding="utf-8").splitlines())
    mgr.save(msgs, system_prompt="s")  # nothing new
    assert len(path.read_text(encoding="utf-8").splitlines()) == before


def test_manager_records_per_call_usage_and_tool_events(tmp_path) -> None:
    mgr = _manager(tmp_path)
    mgr.create()
    mgr.save(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}],
        system_prompt="s",
        usages=[
            {"iteration": 0, "usage": {"prompt_tokens": 100, "completion_tokens": 10},
             "ts": "2026-01-01T00:00:00"},
            {"iteration": 1, "usage": {"prompt_tokens": 110, "completion_tokens": 12},
             "ts": "2026-01-01T00:00:01"},
        ],
        tools=[
            {"tool_call_id": "c1", "name": "read", "ok": True, "elapsed_ms": 5,
             "ts": "2026-01-01T00:00:00"},
            {"tool_call_id": "c2", "name": "bash", "ok": False, "elapsed_ms": 20,
             "ts": "2026-01-01T00:00:02"},
        ],
        elapsed_ms=1500,
    )
    loaded = mgr.list()[0]
    assert loaded.round_count == 1
    assert len(loaded.iterations) == 2
    assert loaded.iterations[0]["usage"]["prompt_tokens"] == 100
    assert loaded.iterations[1]["round"] == 1
    assert len(loaded.tools) == 2
    assert loaded.tools[1]["ok"] is False
    r = loaded.rounds[0]
    # Round-level aggregates derived from the per-call events.
    assert r["iterations"] == 2
    assert r["elapsed_ms"] == 1500
    assert r["usage"] == {"prompt_tokens": 210, "completion_tokens": 22}
    assert r["model"] == ""
    # Persisted lines: meta + 2 messages + 2 usage + 2 tool + 1 round.
    path = _store(tmp_path).path_for("wd", mgr.active_id)
    assert len(path.read_text(encoding="utf-8").splitlines()) == 8


def test_manager_round_records_model_and_usage_delta(tmp_path) -> None:
    mgr = _manager(tmp_path, model="m1", provider="p")
    mgr.create()
    mgr.save([{"role": "user", "content": "one"}], system_prompt="s",
             usages=[{"iteration": 0, "usage": {"prompt_tokens": 5, "completion_tokens": 1},
                      "ts": "t"}],
             model="m2")
    mgr.save([{"role": "user", "content": "one"}, {"role": "assistant", "content": "two"}],
             system_prompt="s",
             usage_total={"prompt_tokens": 12, "completion_tokens": 3},
             usages=[{"iteration": 0, "usage": {"prompt_tokens": 7, "completion_tokens": 2},
                      "ts": "t"}],
             model="m2")
    loaded = mgr.list()[0]
    assert loaded.rounds[0]["model"] == "m2"
    assert loaded.rounds[1]["usage"] == {"prompt_tokens": 7, "completion_tokens": 2}
    assert loaded.rounds[1]["usage_total"] == {"prompt_tokens": 12, "completion_tokens": 3}


def test_manager_records_interrupted_and_error(tmp_path) -> None:
    mgr = _manager(tmp_path)
    mgr.create()
    mgr.save([{"role": "user", "content": "hi"}], interrupted=True)
    mgr.save(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "x"}],
        error="boom",
    )
    rounds = mgr.list()[0].rounds
    assert rounds[0]["interrupted"] is True
    assert rounds[0]["error"] is None
    assert rounds[1]["error"] == "boom"


def test_manager_create_is_in_memory_only(tmp_path) -> None:
    mgr = _manager(tmp_path)
    mgr.create()
    assert mgr.list() == []
    assert mgr.active is not None


def test_manager_resume_and_switch(tmp_path) -> None:
    store = _store(tmp_path)
    mgr = SessionManager("wd", store)
    mgr.create()
    mgr.save(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}],
        system_prompt="sp",
    )
    sid = mgr.active_id

    mgr2 = SessionManager("wd", store)
    s = mgr2.resume(sid)
    assert s is not None
    assert s.system_prompt == "sp"
    # Appending after resume continues the same log (no duplication).
    mgr2.save(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"},
         {"role": "user", "content": "more"}],
        system_prompt="sp",
    )
    loaded = store.load("wd", sid)
    assert [m["content"] for m in loaded.messages] == ["hi", "yo", "more"]
    assert loaded.round_count == 2


def test_manager_clear_and_delete_active(tmp_path) -> None:
    mgr = _manager(tmp_path)
    mgr.create()
    mgr.save([{"role": "user", "content": "hi"}])
    sid = mgr.active_id
    mgr.clear()
    assert mgr.active is None
    assert mgr.list()[0].id == sid  # still on disk

    mgr.resume(sid)
    assert mgr.delete(sid) is True
    assert mgr.active is None
    assert mgr.list() == []


def test_extract_title_first_user_message() -> None:
    assert _extract_title([{"role": "assistant", "content": "x"},
                           {"role": "user", "content": "  hello world  "}]) == "hello world"
    assert _extract_title([{"role": "tool", "content": "r"}]) == ""
