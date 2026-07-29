"""Session — conversation persistence for the Ezwork app.

Two pieces only (kept deliberately small):
  - Session: pure serialisable data (id, timestamps, workdir, model/provider,
    title, messages).
  - SessionStore: file-based storage partitioned by workdir hash.

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
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

DEFAULT_SESSIONS_DIR = Path.home() / ".ezwork" / "sessions"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


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
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        return cls(
            id=data["id"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            workdir=data["workdir"],
            messages=data.get("messages", []),
            title=data.get("title", ""),
            model=data.get("model", ""),
            provider=data.get("provider", ""),
            metadata=data.get("metadata", {}),
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
            "metadata": self.metadata,
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
            data = _read_json(f)
            if data is None:
                continue
            try:
                sessions.append(Session.from_dict(data))
            except KeyError:
                continue
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions

    def save(self, session: Session) -> Session:
        """Upsert a session. Refreshes updated_at and (re)derives the title."""
        session.updated_at = datetime.now().isoformat()
        title = _extract_title(session.messages)
        if title:
            session.title = title
        d = self._dir_for(session.workdir)
        _write_json(d / f"{session.id}.json", session.to_dict())
        return session

    def load(self, workdir: str, session_id: str) -> Session | None:
        data = _read_json(self._dir_for(workdir) / f"{session_id}.json")
        if data is None:
            return None
        try:
            return Session.from_dict(data)
        except KeyError:
            return None

    def delete(self, workdir: str, session_id: str) -> bool:
        path = self._dir_for(workdir) / f"{session_id}.json"
        if not path.exists():
            return False
        try:
            path.unlink()
            return True
        except OSError:
            return False


__all__ = ["Session", "SessionStore", "DEFAULT_SESSIONS_DIR"]
