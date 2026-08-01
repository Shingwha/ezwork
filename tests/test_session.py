"""Tests for Session / SessionStore / SessionManager."""

from __future__ import annotations

from ezwork.app.session import Session, SessionManager, SessionStore, _extract_title


def _store(tmp_path) -> SessionStore:
    return SessionStore(tmp_path / "sessions")


# ─── SessionStore ───────────────────────────────────────────────────────────


def test_store_save_load_roundtrip(tmp_path) -> None:
    store = _store(tmp_path)
    s = Session.new("wd", model="m", provider="p")
    s.messages = [{"role": "user", "content": "hi"}]
    store.save(s)
    loaded = store.load("wd", s.id)
    assert loaded is not None
    assert loaded.id == s.id
    assert loaded.messages == s.messages
    assert loaded.model == "m"
    assert loaded.provider == "p"


def test_store_load_missing_returns_none(tmp_path) -> None:
    assert _store(tmp_path).load("wd", "session_nope") is None


def test_store_list_newest_first_and_derives_title(tmp_path) -> None:
    store = _store(tmp_path)
    a = store.save(Session.new("wd"))
    a.messages = [{"role": "user", "content": "first question"}]
    store.save(a)
    b = store.save(Session.new("wd"))
    b.messages = [{"role": "user", "content": "second question"}]
    store.save(b)
    listed = store.list("wd")
    assert [s.id for s in listed] == [b.id, a.id]
    assert listed[0].title == "second question"


def test_store_delete(tmp_path) -> None:
    store = _store(tmp_path)
    s = store.save(Session.new("wd"))
    assert store.delete("wd", s.id) is True
    assert store.delete("wd", s.id) is False


# ─── SessionManager ─────────────────────────────────────────────────────────


def test_manager_lazy_create_and_save(tmp_path) -> None:
    mgr = SessionManager(str(tmp_path), _store(tmp_path), model="m", provider="p")
    assert mgr.active is None
    assert mgr.active_id is None

    s = mgr.save([{"role": "user", "content": "hello"}])  # first save creates
    assert s.id == mgr.active_id
    assert s.title == "hello"
    assert s.model == "m"
    assert s.messages[0]["content"] == "hello"


def test_manager_save_updates_in_place(tmp_path) -> None:
    store = _store(tmp_path)
    mgr = SessionManager(str(tmp_path), store)
    s1 = mgr.save([{"role": "user", "content": "one"}])
    s2 = mgr.save([{"role": "user", "content": "one"}, {"role": "assistant", "content": "two"}])
    assert s1.id == s2.id  # same active session, not a new file
    assert len(store.list(str(tmp_path))) == 1


def test_manager_create_is_in_memory_only(tmp_path) -> None:
    """create() must not write a file — an interrupted first turn leaves
    no empty session behind."""
    store = _store(tmp_path)
    mgr = SessionManager(str(tmp_path), store)
    mgr.create()
    assert mgr.active_id is not None
    assert store.list(str(tmp_path)) == []


def test_manager_resume_and_switch(tmp_path) -> None:
    store = _store(tmp_path)
    mgr = SessionManager(str(tmp_path), store)
    s = mgr.save([{"role": "user", "content": "hi"}])

    mgr2 = SessionManager(str(tmp_path), store)
    assert mgr2.resume("session_nope") is None
    loaded = mgr2.resume(s.id)
    assert loaded is not None
    assert loaded.messages == s.messages
    assert mgr2.active_id == s.id

    other = Session.new(str(tmp_path))
    mgr2.switch_to(other)
    assert mgr2.active_id == other.id


def test_manager_clear_and_delete_active(tmp_path) -> None:
    store = _store(tmp_path)
    mgr = SessionManager(str(tmp_path), store)
    s = mgr.save([{"role": "user", "content": "hi"}])
    mgr.clear()
    assert mgr.active is None
    assert mgr.list() == [s]  # file still on disk

    mgr.resume(s.id)
    assert mgr.delete(s.id) is True
    assert mgr.active is None  # deleting the active session clears it


# ─── title extraction ───────────────────────────────────────────────────────


def test_extract_title_first_user_message() -> None:
    msgs = [
        {"role": "assistant", "content": "ignored"},
        {"role": "user", "content": "  how does\n auth work?  "},
    ]
    assert _extract_title(msgs) == "how does  auth work?"
