"""Session — conversation persistence for the Ezwork app.

Three pieces:
  - Session: pure serialisable data (id, timestamps, workdir, model/provider,
    title, messages).
  - SessionStore: file-based storage partitioned by workdir hash.
  - SessionManager: lifecycle state machine over one active session
    (create / save / resume / switch / clear). The CLI talks to this, never
    to the store directly.

Sessions live under ~/.ezwork/sessions/<sha256(cwd)[:16]>/<session_id>.json so
that history from the same project directory clusters together. The CLI uses
these directly — oneshot creates a session and prints its id to stderr so a
caller can continue it with `ezwork -p "..." -s <id>`.

Messages are stored verbatim in OpenAI format (the same dicts the agent loop
appends). To resume, they are loaded straight back into the agent's message
list.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .utils import read_json, write_json

DEFAULT_SESSIONS_DIR = Path.home() / ".ezwork" / "sessions"


def _hash_workdir(workdir: str) -> str:
    return hashlib.sha256(workdir.encode()).hexdigest()[:16]


@dataclass
class Session:
    """Conversation session — pure data, serialisable."""

    id: str
    created_at: str
    updated_at: str
    workdir: str
    messages: list[dict[str, Any]]
    title: str = ""
    model: str = ""
    provider: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        return cls(
            id=str(data.get("id", "")),
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
            workdir=str(data.get("workdir", "")),
            messages=list(data.get("messages", []) or []),
            title=str(data.get("title", "")),
            model=str(data.get("model", "")),
            provider=str(data.get("provider", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "workdir": self.workdir,
            "messages": self.messages,
            "title": self.title,
            "model": self.model,
            "provider": self.provider,
        }

    @classmethod
    def new(cls, workdir: str, *, model: str = "", provider: str = "") -> "Session":
        """Create a fresh empty session (not yet persisted)."""
        now = datetime.now().isoformat()
        return cls(
            id=f"session_{uuid4().hex[:12]}",
            created_at=now,
            updated_at=now,
            workdir=workdir,
            messages=[],
            model=model,
            provider=provider,
        )


def _extract_title(messages: list[dict[str, Any]]) -> str:
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content[:60].replace("\n", " ").strip()
    return ""


class SessionStore:
    """File-based session storage, partitioned by workdir hash."""

    def __init__(self, base_dir: Path | None = None):
        self._base_dir = base_dir or DEFAULT_SESSIONS_DIR

    def _dir_for(self, workdir: str) -> Path:
        return self._base_dir / _hash_workdir(workdir)

    def list(self, workdir: str) -> list[Session]:
        """Return sessions for workdir, newest first by updated_at."""
        d = self._dir_for(workdir)
        if not d.exists():
            return []
        sessions: list[Session] = []
        for f in d.glob("session_*.json"):
            data = read_json(f)
            if data is None:
                continue
            s = Session.from_dict(data)
            if s.id:
                sessions.append(s)
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions

    def save(self, session: Session) -> Session:
        """Upsert a session. Refreshes updated_at and (re)derives the title."""
        session.updated_at = datetime.now().isoformat()
        title = _extract_title(session.messages)
        if title:
            session.title = title
        d = self._dir_for(session.workdir)
        write_json(d / f"{session.id}.json", session.to_dict())
        return session

    def load(self, workdir: str, session_id: str) -> Session | None:
        data = read_json(self._dir_for(workdir) / f"{session_id}.json")
        if data is None:
            return None
        s = Session.from_dict(data)
        return s if s.id else None

    def delete(self, workdir: str, session_id: str) -> bool:
        path = self._dir_for(workdir) / f"{session_id}.json"
        if not path.exists():
            return False
        try:
            path.unlink()
            return True
        except OSError:
            return False


class SessionManager:
    """Lifecycle state machine over one active session.

    The active session lives in memory (no reload-from-disk on every save)
    and is persisted by SessionStore.save(). `create()` is deliberately
    in-memory only — the first real save() writes the file, so an
    interrupted first turn leaves no empty session file behind.
    """

    def __init__(
        self,
        workdir: str,
        store: SessionStore | None = None,
        *,
        model: str = "",
        provider: str = "",
    ):
        self._workdir = workdir
        self._store = store or SessionStore()
        self._model = model
        self._provider = provider
        self._active: Session | None = None

    @property
    def workdir(self) -> str:
        return self._workdir

    @property
    def active(self) -> Session | None:
        return self._active

    @property
    def active_id(self) -> str | None:
        return self._active.id if self._active else None

    def create(self) -> Session:
        """Start a fresh in-memory session (not persisted until first save)."""
        self._active = Session.new(
            self._workdir, model=self._model, provider=self._provider
        )
        return self._active

    def resume(self, session_id: str) -> Session | None:
        """Load a session from the store and make it active."""
        session = self._store.load(self._workdir, session_id)
        if session is None:
            return None
        self._active = session
        return session

    def switch_to(self, session: Session) -> None:
        """Make an already-loaded session active (used by /resume)."""
        self._active = session

    def save(self, messages: list[dict[str, Any]], *, model: str = "", provider: str = "") -> Session:
        """Persist messages into the active session; creates one on first save."""
        if self._active is None:
            self._active = Session.new(
                self._workdir, model=model or self._model, provider=provider or self._provider
            )
        s = self._active
        s.messages = list(messages)
        if model:
            s.model = model
        if provider:
            s.provider = provider
        return self._store.save(s)

    def get_active(self) -> Session | None:
        return self._active

    def list(self) -> list[Session]:
        return self._store.list(self._workdir)

    def delete(self, session_id: str) -> bool:
        result = self._store.delete(self._workdir, session_id)
        if result and self._active is not None and self._active.id == session_id:
            self._active = None
        return result

    def clear(self) -> None:
        self._active = None


__all__ = ["Session", "SessionStore", "SessionManager", "DEFAULT_SESSIONS_DIR"]
