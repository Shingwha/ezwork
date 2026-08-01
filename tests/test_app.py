"""Tests for the CLIApp composition root (oneshot contract, resume, errors)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from ezwork.app.cli.app import CLIApp
from ezwork.app.config import Config
from ezwork.app.session import Session, SessionStore

from tests import MockProvider


def _make_app(provider, tmp_path, **kwargs) -> CLIApp:
    """CLIApp with a scripted provider and an isolated sessions dir."""
    import ezwork.app.session as session_mod

    def _patch(_self) -> MockProvider:
        return provider

    return _make_app_patched(provider, tmp_path, _patch, **kwargs)


def _make_app_patched(provider, tmp_path, patch_fn, **kwargs) -> CLIApp:
    import ezwork.app.session as session_mod

    orig = Config.build_provider
    Config.build_provider = patch_fn
    try:
        session_mod.DEFAULT_SESSIONS_DIR = tmp_path / "sessions"
        return CLIApp(Config(), **kwargs)
    finally:
        Config.build_provider = orig
        session_mod.DEFAULT_SESSIONS_DIR = Path.home() / ".ezwork" / "sessions"


def _run(coro) -> int:
    return asyncio.run(coro)


# ─── oneshot contract ───────────────────────────────────────────────────────


def test_oneshot_answer_to_stdout_session_to_stderr(tmp_path, capsys) -> None:
    from tests import usage

    provider = MockProvider([{"content": "hello world", "usage": usage(10, 5)}])
    app = _make_app(provider, tmp_path, interactive=False)

    rc = _run(app.run_oneshot("hi"))

    out, err = capsys.readouterr()
    assert rc == 0
    assert out == "hello world\n"  # stdout: answer only
    assert "session: session_" in err
    assert "[tokens: prompt=10 completion=5]" in err


def test_oneshot_no_session_flag(tmp_path, capsys) -> None:
    provider = MockProvider(["answer"])
    app = _make_app(provider, tmp_path, interactive=False)

    rc = _run(app.run_oneshot("hi", no_session=True))

    out, err = capsys.readouterr()
    assert rc == 0
    assert "session:" not in err
    assert not (tmp_path / "sessions").exists()  # nothing persisted


def test_oneshot_provider_error_exits_1(tmp_path, capsys) -> None:
    provider = MockProvider([("error", "boom")])
    app = _make_app(provider, tmp_path, interactive=False)

    rc = _run(app.run_oneshot("hi"))

    out, err = capsys.readouterr()
    assert rc == 1
    assert out == ""  # no fake answer on stdout
    assert "[provider error] boom" in err


def test_oneshot_unknown_session_exits_1(tmp_path, capsys) -> None:
    provider = MockProvider(["answer"])
    app = _make_app(provider, tmp_path, interactive=False)

    rc = _run(app.run_oneshot("hi", session_id="session_nope"))

    _, err = capsys.readouterr()
    assert rc == 1
    assert "session session_nope not found" in err


def test_oneshot_interrupt_returns_130(tmp_path, capsys) -> None:
    class CancellingProvider(MockProvider):
        async def stream(self, *args, **kwargs):  # type: ignore[override]
            raise asyncio.CancelledError()
            yield  # pragma: no cover — makes this an async generator

    app = _make_app(CancellingProvider([]), tmp_path, interactive=False)
    rc = _run(app.run_oneshot("hi"))
    _, err = capsys.readouterr()
    assert rc == 130
    assert "(interrupted)" in err


# ─── resume / session persistence ───────────────────────────────────────────


def test_oneshot_resume_loads_history(tmp_path, capsys) -> None:
    store = SessionStore(tmp_path / "sessions")
    s = Session.new(str(Path.cwd()), model="m", provider="p")
    s.messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "first answer"},
    ]
    store.save(s)

    provider = MockProvider(["second answer"])
    app = _make_app(provider, tmp_path, interactive=False)

    rc = _run(app.run_oneshot("continue", session_id=s.id))

    out, err = capsys.readouterr()
    assert rc == 0
    assert out == "second answer\n"
    # The resumed history was sent to the provider as prior context.
    sent = provider.calls[0]["messages"]
    assert any(m.get("content") == "first" for m in sent)
    assert any(m.get("content") == "first answer" for m in sent)


def test_oneshot_persists_session_on_success(tmp_path, capsys) -> None:
    provider = MockProvider(["answer"])
    app = _make_app(provider, tmp_path, interactive=False)

    _run(app.run_oneshot("hi"))

    store = SessionStore(tmp_path / "sessions")
    sessions = store.list(str(Path.cwd()))
    assert len(sessions) == 1
    assert sessions[0].messages[-1]["content"] == "answer"
    assert sessions[0].title == "hi"
