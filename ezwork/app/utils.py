"""Shared utilities for the ezwork app layer — JSON I/O, text shaping,
and tool-call summarising/grouping for the CLI display."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# ─── JSON I/O ───────────────────────────────────────────────────────────────


def read_json(path: Path | str, *, encoding: str = "utf-8") -> dict[str, Any] | None:
    """Read a JSON file. Returns None if missing, invalid, or unreadable."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding=encoding))
    except (json.JSONDecodeError, OSError):
        return None


def write_json(
    path: Path | str,
    data: Any,
    *,
    indent: int = 2,
    ensure_ascii: bool = False,
    encoding: str = "utf-8",
) -> None:
    """Write data as JSON, creating parent directories as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(data, indent=indent, ensure_ascii=ensure_ascii),
        encoding=encoding,
    )


# ─── Text shaping ───────────────────────────────────────────────────────────


def oneline(text: str, limit: int) -> str:
    """Collapse whitespace and truncate to a single line."""
    s = " ".join(text.split())
    return s[:limit] + ("..." if len(s) > limit else "")


def ellipsize(text: str, limit: int) -> str:
    """Truncate keeping the tail: 'abcdefghij' → 'abcde…hij' (middle ellipsis)."""
    if len(text) <= limit:
        return text
    if limit < 7:
        return text[:limit]
    head = limit // 2 - 1
    tail = limit - head - 1  # one char for '…'
    return text[:head] + "…" + text[-tail:]


# ─── Tool-call summarising & grouping ───────────────────────────────────────

_CONTENT_LIMIT = 60
# Tools whose parallel calls collapse into one grouped line (read×2 …).
_MERGE_TOOLS = frozenset({"read", "write", "append", "edit", "glob", "grep"})
# The argument shown as a tool call's summary.
_TOOL_KEY = {
    "read": "path",
    "write": "path",
    "append": "path",
    "edit": "path",
    "bash": "command",
    "glob": "pattern",
    "grep": "pattern",
}


def tool_summary(name: str, args: dict) -> str:
    """Short summary of a tool call's arguments (the interesting value)."""
    key = _TOOL_KEY.get(name)
    if not key:
        if not args:
            return ""
        key = next(iter(args))
    return ellipsize(str(args.get(key, "")), _CONTENT_LIMIT)


def merge_summaries(summaries: list[str]) -> str:
    """Join summaries, truncating with a '… +N' suffix when too long."""
    if not summaries:
        return ""
    joined = ", ".join(summaries)
    if len(joined) <= _CONTENT_LIMIT:
        return joined
    total, count = 0, 0
    for s in summaries:
        add = len(s) + (2 if count else 0)
        if total + add > _CONTENT_LIMIT - 10:
            break
        total += add
        count += 1
    count = max(count, 1)
    shown = ", ".join(summaries[:count])
    remaining = len(summaries) - count
    return shown + f"… +{remaining}" if remaining else shown


def group_items(items: list, get_name, get_args) -> list[tuple[str, list[str]]]:
    """Group items by tool name; mergeable tools combine their summaries."""
    merged: dict[str, list[str]] = {}
    singles: list[tuple[str, list[str]]] = []
    for item in items:
        name = get_name(item)
        summary = tool_summary(name, get_args(item))
        if name in _MERGE_TOOLS:
            merged.setdefault(name, []).append(summary)
        else:
            singles.append((name, [summary]))
    return list(merged.items()) + singles


__all__ = [
    "read_json",
    "write_json",
    "oneline",
    "ellipsize",
    "tool_summary",
    "merge_summaries",
    "group_items",
]
