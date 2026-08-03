"""Session — conversation persistence for the Ezwork app (JSONL event log).

Design
------
Each session is an append-only JSONL file under
`~/.ezwork/sessions/<sha256(workdir)[:16]>/<session_id>.jsonl`. Every line is
one self-describing event; nothing is ever rewritten, so an interrupted write
can at worst truncate the last line (skipped on read) and can never corrupt
or lose earlier turns.

Events (the first JSON key is always "type", so readers can classify a line
cheaply):

  meta      first line, written with the first save: static identity
            (id, schema, created_at, workdir, model, provider).
  message   one OpenAI-format message dict per line, appended in order.
  round     one per completed turn: cumulative usage / tool stats, the
            interrupted/error state, and the FULL system prompt that was in
            effect for that turn (with its hash) — a session file therefore
            holds one system prompt per round. Resuming uses the latest
            round's prompt (prefix-cache friendly); replaying a specific
            round can use the exact prompt that produced it.
  compact   RESERVED for future compaction: a compaction is planned as one
            more appended event (summary + rebuilt system prompt) followed by
            normal message/round lines — lines are never deleted, so the full
            history stays recoverable for display/archive purposes.

Reading tolerates malformed lines: a partial trailing line (interrupted
write) is skipped; earlier turns are unaffected.

Session is the in-memory view: messages plus the raw round events, with the
latest round's aggregates (usage, tool stats, system prompt) promoted onto
the object for direct access.
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
SCHEMA = "ezwork.session.v1"

# Message keys that belong to the event envelope, not the OpenAI message dict.
_MESSAGE_ENVELOPE_KEYS = ("type",)


def _hash_workdir(workdir: str) -> str:
    return hashlib.sha256(workdir.encode()).hexdigest()[:16]


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _dump_line(event: dict) -> str:
    """Serialise one event as a compact single-line JSON record."""
    return json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"


def _parse_line(line: str) -> dict | None:
    """Parse one event line. Returns None for malformed lines (interrupted
    writes, stray bytes) — callers skip them."""
    try:
        obj = json.loads(line)
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def _extract_title(messages: list[dict[str, Any]]) -> str:
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content[:60].replace("\n", " ").strip()
    return ""


# ─── Session (in-memory view) ───────────────────────────────────────────────


@dataclass
class Session:
    """Conversation session — pure data, reconstructed from the JSONL log."""

    id: str
    created_at: str
    updated_at: str
    workdir: str
    title: str = ""
    model: str = ""
    provider: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    rounds: list[dict[str, Any]] = field(default_factory=list)
    system_prompt: str = ""
    system_prompt_hash: str = ""
    usage_total: dict[str, int] = field(default_factory=dict)
    tool_stats: dict[str, int] = field(default_factory=dict)
    tool_failures: dict[str, int] = field(default_factory=dict)

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def round_count(self) -> int:
        return len(self.rounds)

    @classmethod
    def new(cls, workdir: str, *, model: str = "", provider: str = "") -> "Session":
        """Create a fresh in-memory session (not persisted until first save)."""
        now = datetime.now().isoformat()
        return cls(
            id=f"session_{uuid4().hex[:12]}",
            created_at=now,
            updated_at=now,
            workdir=workdir,
            model=model,
            provider=provider,
        )

    @classmethod
    def from_events(cls, events: list[dict[str, Any]]) -> "Session":
        """Reconstruct a session from parsed JSONL events (last meta wins;
        the latest round's aggregates are promoted)."""
        meta: dict[str, Any] = {}
        messages: list[dict[str, Any]] = []
        rounds: list[dict[str, Any]] = []
        for ev in events:
            t = ev.get("type")
            if t == "meta":
                meta = ev
            elif t == "message":
                messages.append(
                    {k: v for k, v in ev.items() if k not in _MESSAGE_ENVELOPE_KEYS}
                )
            elif t == "round":
                rounds.append(ev)
        last = rounds[-1] if rounds else {}
        return cls(
            id=str(meta.get("id", "")),
            created_at=str(meta.get("created_at", "")),
            updated_at=str(last.get("updated_at") or meta.get("updated_at", "")),
            workdir=str(meta.get("workdir", "")),
            title=str(last.get("title") or meta.get("title", "")),
            model=str(meta.get("model", "")),
            provider=str(meta.get("provider", "")),
            messages=messages,
            rounds=rounds,
            system_prompt=str(last.get("system_prompt", "")),
            system_prompt_hash=str(last.get("system_prompt_hash", "")),
            usage_total=dict(last.get("usage_total") or {}),
            tool_stats=dict(last.get("tool_calls") or {}),
            tool_failures=dict(last.get("tool_failures") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Full portable view (used by /export json)."""
        return {
            "id": self.id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "workdir": self.workdir,
            "title": self.title,
            "model": self.model,
            "provider": self.provider,
            "messages": self.messages,
            "rounds": self.rounds,
            "system_prompt": self.system_prompt,
            "system_prompt_hash": self.system_prompt_hash,
            "usage_total": self.usage_total,
            "tool_stats": self.tool_stats,
            "tool_failures": self.tool_failures,
        }


# ─── Store ──────────────────────────────────────────────────────────────────


class SessionStore:
    """Append-only JSONL storage, partitioned by workdir hash."""

    def __init__(self, base_dir: Path | None = None):
        self._base_dir = base_dir or DEFAULT_SESSIONS_DIR

    def _dir_for(self, workdir: str) -> Path:
        return self._base_dir / _hash_workdir(workdir)

    def path_for(self, workdir: str, session_id: str) -> Path:
        return self._dir_for(workdir) / f"{session_id}.jsonl"

    def append(self, workdir: str, session_id: str, events: list[dict]) -> None:
        """Append events as one write. Line-atomic: an interrupted write can
        truncate the final line but never earlier ones."""
        path = self.path_for(workdir, session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = "".join(_dump_line(ev) for ev in events)
        with open(path, "a", encoding="utf-8") as f:
            f.write(payload)

    def load(self, workdir: str, session_id: str) -> Session | None:
        """Read the full log, skipping malformed lines. None when missing or
        unreadable."""
        path = self.path_for(workdir, session_id)
        if not path.exists():
            return None
        events: list[dict] = []
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    ev = _parse_line(line)
                    if ev is not None:
                        events.append(ev)
        except OSError:
            return None
        s = Session.from_events(events)
        return s if s.id else None

    def list(self, workdir: str) -> list[Session]:
        """Return sessions for workdir, newest first by updated_at.

        Each file is fully parsed (simple and correct; the log is line-based,
        so a future optimisation can scan only meta/round lines for large
        files)."""
        d = self._dir_for(workdir)
        if not d.exists():
            return []
        sessions: list[Session] = []
        for f in sorted(d.glob("session_*.jsonl")):
            s = self.load(workdir, f.stem)
            if s is not None:
                sessions.append(s)
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions

    def delete(self, workdir: str, session_id: str) -> bool:
        path = self.path_for(workdir, session_id)
        if not path.exists():
            return False
        try:
            path.unlink()
            return True
        except OSError:
            return False


# ─── Manager ────────────────────────────────────────────────────────────────


class SessionManager:
    """Lifecycle state machine over one active session.

    The active session lives in memory and is persisted incrementally by
    SessionStore.append(). `create()` is deliberately in-memory only — the
    first real save() writes the meta line, so an interrupted first turn
    leaves no empty session file behind. An append cursor tracks how many of
    the agent's messages are already on disk; save() writes only the new
    ones plus one round event.
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
        self._written = 0  # messages already persisted for the active session
        self._save_failed = False

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
        self._written = 0
        self._save_failed = False
        return self._active

    def resume(self, session_id: str) -> Session | None:
        """Load a session from the store and make it active."""
        session = self._store.load(self._workdir, session_id)
        if session is None:
            return None
        self._active = session
        self._written = len(session.messages)
        self._save_failed = False
        return session

    def switch_to(self, session: Session) -> None:
        """Make an already-loaded session active (used by /resume)."""
        self._active = session
        self._written = len(session.messages)
        self._save_failed = False

    def save(
        self,
        messages: list[dict[str, Any]],
        *,
        usage_total: dict[str, int] | None = None,
        tool_stats: dict[str, int] | None = None,
        tool_failures: dict[str, int] | None = None,
        system_prompt: str = "",
        model: str = "",
        provider: str = "",
        interrupted: bool = False,
        error: str | None = None,
    ) -> Session:
        """Append new messages + one round event for the active session.

        No-op when there is nothing new (no messages, no interrupted/error
        state). The round event records the full system prompt in effect for
        this turn, so the file keeps one system prompt per round. A failed
        write marks the session dirty; the next save reconciles the append
        cursor against disk before retrying, so messages are never silently
        dropped or duplicated."""
        if self._active is None:
            self._active = Session.new(
                self._workdir, model=model or self._model, provider=provider or self._provider
            )
        s = self._active
        if model:
            s.model = model
        if provider:
            s.provider = provider

        if self._save_failed:
            self._written = self._sync_written()

        new_msgs = messages[self._written :]
        if not new_msgs and not interrupted and error is None:
            return s  # nothing new to record

        events: list[dict[str, Any]] = []
        if not s.rounds:
            events.append(
                {
                    "type": "meta",
                    "schema": SCHEMA,
                    "id": s.id,
                    "created_at": s.created_at,
                    "workdir": s.workdir,
                    "title": s.title,
                    "model": s.model,
                    "provider": s.provider,
                }
            )
        for msg in new_msgs:
            events.append({"type": "message", **msg})

        now = datetime.now().isoformat()
        title = s.title or _extract_title(messages)
        round_ev = {
            "type": "round",
            "round": len(s.rounds) + 1,
            "updated_at": now,
            "title": title,
            "message_count": len(messages),
            "usage_total": dict(usage_total or {}),
            "tool_calls": dict(tool_stats or {}),
            "tool_failures": dict(tool_failures or {}),
            "interrupted": bool(interrupted),
            "error": error,
            "system_prompt": system_prompt,
            "system_prompt_hash": _hash_text(system_prompt) if system_prompt else "",
        }
        events.append(round_ev)

        try:
            self._store.append(s.workdir, s.id, events)
            self._save_failed = False
        except OSError:
            # Keep memory state; next save reconciles the cursor with disk.
            self._save_failed = True
            return s

        s.messages = list(messages)
        s.rounds.append(round_ev)
        s.updated_at = now
        s.title = title
        s.system_prompt = system_prompt
        s.system_prompt_hash = round_ev["system_prompt_hash"]
        s.usage_total = dict(usage_total or {})
        s.tool_stats = dict(tool_stats or {})
        s.tool_failures = dict(tool_failures or {})
        self._written = len(messages)
        return s

    def _sync_written(self) -> int:
        """Reconcile the append cursor with disk after a failed write."""
        try:
            loaded = self._store.load(self._workdir, self._active.id)
        except Exception:
            return self._written
        if loaded is None:
            return 0
        return len(loaded.messages)

    def get_active(self) -> Session | None:
        return self._active

    def list(self) -> list[Session]:
        return self._store.list(self._workdir)

    def delete(self, session_id: str) -> bool:
        result = self._store.delete(self._workdir, session_id)
        if result and self._active is not None and self._active.id == session_id:
            self._active = None
            self._written = 0
        return result

    def clear(self) -> None:
        self._active = None
        self._written = 0
        self._save_failed = False


__all__ = ["Session", "SessionStore", "SessionManager", "DEFAULT_SESSIONS_DIR", "SCHEMA"]
