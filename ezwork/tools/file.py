"""File operation tools — ReadTool, WriteTool, EditTool.

Ported from the old Ezwork with VFS support stripped (the kernel has no
virtual filesystem). All tools operate directly on the local filesystem.
"""

from __future__ import annotations

import os
import threading
from difflib import SequenceMatcher
from pathlib import Path

from ezwork.core import Tool, ToolError

from .utils import IGNORE_DIRS, decode_bytes, require_file


def _format_lines(content: str, label: str, offset: int, limit: int) -> str:
    """Format content with line numbers (cat -n style).

    offset is 1-based; limit = 0 means all lines.
    """
    all_lines = content.splitlines(keepends=True)
    total = len(all_lines)
    size_kb = len(content.encode("utf-8")) / 1024
    start = max(0, offset - 1)
    end = start + limit if limit else total
    selected = all_lines[start:end]
    if not selected:
        raise ToolError(
            f"Line {offset} is beyond end of file (file has {total} lines)",
            "out_of_range",
        )
    header = f"[{label} | {total} lines | {size_kb:.1f} KB]"
    lines_text = "".join(
        f"{start + idx + 1:>5} | {line}" for idx, line in enumerate(selected)
    )
    end_line = start + len(selected)
    if end_line < total:
        footer = (
            f"\n[Showing lines {start + 1}-{end_line} of {total}. "
            f"Use offset={end_line + 1} to read more.]"
        )
    else:
        footer = ""
    return header + "\n" + lines_text + footer


def _list_directory(p: Path) -> str:
    """List directory contents when read() is called on a directory."""
    with os.scandir(p) as it:
        # DirEntry caches is_dir()/stat() after the first call, so sorting
        # and listing here costs one syscall per entry instead of O(n log n).
        entries = sorted(it, key=lambda e: (not e.is_dir(), e.name.lower()))
    dirs: list[str] = []
    files: list[str] = []
    for entry in entries:
        if entry.name in IGNORE_DIRS:
            continue
        if entry.is_dir():
            dirs.append(f"{entry.name}/")
        else:
            size = entry.stat().st_size
            if size < 1024:
                size_str = f"{size} B"
            elif size < 1024 * 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size / (1024 * 1024):.1f} MB"
            files.append(f"{entry.name}  ({size_str})")
    lines = dirs + files
    header = f"[{p}/ — {len(dirs)} directories, {len(files)} files]"
    hint = "\nThis is a directory, not a file."
    return header + "\n" + "\n".join(lines) + hint


# ---- read ----

_READ_PARAMS = {
    "path": {"type": "string", "description": "File path to read"},
    "offset": {
        "type": "integer",
        "description": "Line number to start from (1-based, default 1)",
        "default": 1,
    },
    "limit": {
        "type": "integer",
        "description": "Max lines to read (0 = all lines)",
        "default": 0,
    },
}
_READ_DESC = (
    "Read a file and return its contents with line numbers. "
    "Supports text files with UTF-8/GBK encoding. "
    "Use offset and limit to read specific line ranges. Line numbers are 1-based."
)


class ReadTool(Tool):
    """Read a file and return its contents with line numbers."""

    def __init__(self) -> None:
        super().__init__(name="read", description=_READ_DESC, params=_READ_PARAMS, func=self._execute)

    def _execute(self, args: dict) -> str:
        path = args["path"]
        offset = max(1, int(args.get("offset", 1)))
        limit = int(args.get("limit", 0))
        p = Path(path)
        if p.is_dir():
            return _list_directory(p)
        p = require_file(p)
        raw = p.read_bytes()
        if b"\x00" in raw[:8192]:
            raise ToolError(f"File appears to be binary: {p}", "binary_file")
        content = decode_bytes(raw)
        return _format_lines(content, str(p), offset, limit)


# ---- write ----

_WRITE_PARAMS = {
    "path": {"type": "string", "description": "File path to write"},
    "content": {"type": "string", "description": "Content to write (UTF-8)"},
    "append": {
        "type": "boolean",
        "description": "Append to end of file instead of overwriting (default: false)",
        "default": False,
    },
}
_WRITE_DESC = (
    "Write content to a file. Creates the file and any parent directories if they "
    "don't exist. By default, overwrites existing content entirely. Set append=true "
    "to append to the end of the file instead."
)


class WriteTool(Tool):
    """Write content to a file."""

    def __init__(self) -> None:
        super().__init__(name="write", description=_WRITE_DESC, params=_WRITE_PARAMS, func=self._execute)

    def _execute(self, args: dict) -> str:
        p = Path(args["path"])
        if p.is_dir():
            raise ToolError(f"Path is a directory: {p}", "invalid_path")
        content = args["content"]
        append = args.get("append", False)
        p.parent.mkdir(parents=True, exist_ok=True)
        if append and p.exists():
            existing = p.read_text(encoding="utf-8")
            if existing and not existing.endswith("\n"):
                content = "\n" + content
            with open(p, "a", encoding="utf-8") as f:
                f.write(content)
            action = "Appended"
        else:
            p.write_text(content, encoding="utf-8")
            action = "Wrote"
        line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
        return f"{action} {line_count} lines to {p.name}"


# ---- edit ----


_EDIT_LOCKS: dict[str, threading.Lock] = {}
_EDIT_LOCKS_GUARD = threading.Lock()


def _edit_lock_for(path: Path) -> threading.Lock:
    """Per-file lock — tool calls in one turn run in parallel
    (asyncio.gather), so concurrent edits to the same file would otherwise
    read-modify-write from the same snapshot and lose updates."""
    key = str(path.resolve())
    if os.name == "nt":
        key = key.lower()
    with _EDIT_LOCKS_GUARD:
        lock = _EDIT_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _EDIT_LOCKS[key] = lock
        return lock


def _closest_line_hint(text: str, old: str) -> str:
    """Best-matching line vs the first non-blank line of old_string."""
    target = old.strip().splitlines()[0] if old.strip() else ""
    if not target:
        return ""
    best, best_ratio = "", 0.0
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        r = SequenceMatcher(None, target, s).ratio()
        if r > best_ratio:
            best, best_ratio = s, r
    if best and best_ratio >= 0.5:
        return f"\nClosest line in file: {best[:120]!r}"
    return ""


def _normalise(text: str) -> tuple[str, list[int]]:
    """Return CRLF→LF text plus a map from each norm index to its raw index.

    A CRLF pair collapses to a single '\n' whose raw index points at the '\r',
    so a norm span maps back to raw text losslessly (see _raw_span).
    """
    norm: list[str] = []
    idx: list[int] = []
    i, n = 0, len(text)
    while i < n:
        if text[i] == "\r" and i + 1 < n and text[i + 1] == "\n":
            norm.append("\n")
            idx.append(i)
            i += 2
        else:
            norm.append(text[i])
            idx.append(i)
            i += 1
    return "".join(norm), idx


def _raw_span(text: str, idx: list[int], s: int, e: int) -> tuple[int, int]:
    """Raw-text span corresponding to the normalised segment [s, e)."""
    last = idx[e - 1]
    return idx[s], last + (2 if text[last] == "\r" else 1)


_EDIT_PARAMS = {
    "path": {"type": "string", "description": "File path to edit"},
    "old_string": {
        "type": "string",
        "description": "Exact text to find (must be unique unless all=true)",
    },
    "new_string": {"type": "string", "description": "Replacement text"},
    "all": {
        "type": "boolean",
        "description": "Replace all occurrences instead of just the first",
        "default": False,
    },
}
_EDIT_DESC = (
    "Find and replace text in a file. The old_string must match exactly (including "
    "whitespace and indentation). By default, old_string must appear exactly once in "
    "the file — the tool will fail if it matches multiple locations. Use all=true to "
    "replace every occurrence. The file must already exist."
)


class EditTool(Tool):
    """Find and replace text in a file (concurrency-safe per file)."""

    def __init__(self) -> None:
        super().__init__(name="edit", description=_EDIT_DESC, params=_EDIT_PARAMS, func=self._execute)

    @staticmethod
    def _not_found_msg(text: str, old: str, new: str, name: str) -> str:
        # The classic mistake: old_string/new_string swapped.
        if new and new in text:
            return (
                f"old_string not found in {name} — but new_string IS present in "
                "the file. Did you swap old_string and new_string?"
            )
        return "old_string not found in file" + _closest_line_hint(text, old)

    @staticmethod
    def _not_unique_msg(text: str, old: str, count: int) -> str:
        lines: list[int] = []
        start = 0
        while True:
            i = text.find(old, start)
            if i < 0:
                break
            lines.append(text.count("\n", 0, i) + 1)
            start = i + 1
        return (
            f"old_string appears {count} times (lines {lines}) — "
            "must be unique (use all=true to replace every occurrence)"
        )

    def _execute(self, args: dict) -> str:
        p = require_file(Path(args["path"]))
        with _edit_lock_for(p):
            # Read with newline="" so CRLF survives the round-trip; the default
            # universal-newline mode would strip \r before we ever match.
            with p.open("r", encoding="utf-8", newline="") as fh:
                raw = fh.read()
            # Match in CRLF-normalised space, then map spans back onto raw text.
            norm, idx = _normalise(raw)
            old = args["old_string"].replace("\r\n", "\n")
            new = args["new_string"]
            if not old:
                raise ToolError("old_string must not be empty", "not_found")
            if old not in norm:
                raise ToolError(self._not_found_msg(norm, old, new, p.name), "not_found")
            count = norm.count(old)
            replace_all = args.get("all", False)
            if not replace_all and count > 1:
                raise ToolError(self._not_unique_msg(norm, old, count), "not_unique")
            actual = count if replace_all else 1
            spans: list[tuple[int, int]] = []
            pos = 0
            for _ in range(actual):
                s = norm.index(old, pos)
                e = s + len(old)
                spans.append(_raw_span(raw, idx, s, e))
                pos = e
            # Apply back-to-front so earlier spans stay valid.
            for s, e in reversed(spans):
                raw = raw[:s] + new + raw[e:]
            p.write_text(raw, encoding="utf-8", newline="")
            return f"Replaced {actual} occurrence(s) in {p.name}"
