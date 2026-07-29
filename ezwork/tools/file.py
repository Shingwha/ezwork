"""File operation tools — ReadTool, WriteTool, EditTool.

Ported from the old Ezwork with VFS support stripped (the kernel has no
virtual filesystem). All tools operate directly on the local filesystem.
"""

from __future__ import annotations

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
    entries = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
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
    """Find and replace text in a file."""

    def __init__(self) -> None:
        super().__init__(name="edit", description=_EDIT_DESC, params=_EDIT_PARAMS, func=self._execute)

    def _execute(self, args: dict) -> str:
        p = require_file(Path(args["path"]))
        text = p.read_text(encoding="utf-8")
        old, new = args["old_string"], args["new_string"]
        if old not in text:
            raise ToolError("old_string not found in file", "not_found")
        count = text.count(old)
        replace_all = args.get("all", False)
        if not replace_all and count > 1:
            raise ToolError(
                f"old_string appears {count} times, must be unique (use all=true)",
                "not_unique",
            )
        replacement = text.replace(old, new) if replace_all else text.replace(old, new, 1)
        p.write_text(replacement, encoding="utf-8")
        actual = count if replace_all else 1
        return f"Replaced {actual} occurrence(s) in {p.name}"
