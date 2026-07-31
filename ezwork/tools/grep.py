"""Grep tool — search file contents by regex pattern.

Ported from the legacy MoCode design, with the VFS layer stripped. Pure
Python + re — cross-platform, no dependency on shell `grep`/`findstr`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from ezwork.core import Tool

from .utils import (
    _get_type_filter,
    _walk_text_files,
    expand_context_indices,
    format_grep_content,
    format_grep_count,
    format_grep_files,
    require_dir,
)


def _search_files(
    pattern: re.Pattern,
    file_iter: Iterable[tuple[str, list[str]]],
    output_mode: str,
    context_lines: int,
    max_results: int,
    not_found_msg: str,
) -> str:
    """Core grep logic — operates on ``(display_path, lines)`` pairs."""
    if output_mode == "files":
        found: list[str] = []
        for display_path, file_lines in file_iter:
            if any(pattern.search(line) for line in file_lines):
                found.append(display_path)
                if len(found) >= max_results:
                    break
        return format_grep_files(found, pattern.pattern)
    if output_mode == "count":
        entries: list[str] = []
        for display_path, file_lines in file_iter:
            count = sum(1 for line in file_lines if pattern.search(line))
            if count > 0:
                entries.append(f"{display_path}:{count}")
                if len(entries) >= max_results:
                    break
        return format_grep_count(entries, pattern.pattern)
    # content mode
    hits: list[str] = []
    for display_path, file_lines in file_iter:
        match_indices = [i for i, line in enumerate(file_lines) if pattern.search(line)]
        if not match_indices:
            continue
        display_indices = expand_context_indices(match_indices, len(file_lines), context_lines)
        result = format_grep_content(
            file_lines, match_indices, display_indices,
            display_path, pattern.pattern, max_results, hits,
        )
        if result:
            return result
    if not hits:
        return not_found_msg
    return f"[Showing {len(hits)} match(es) for '{pattern.pattern}']\n" + "\n".join(hits)


def _grep_single_file(
    pattern: re.Pattern, filepath: Path, output_mode: str, context_lines: int, max_results: int
) -> str:
    try:
        lines = filepath.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return f"Cannot read file: {filepath}"
    return _search_files(
        pattern, [(str(filepath), lines)], output_mode, context_lines, max_results,
        f"No matches for '{pattern.pattern}' in {filepath}",
    )


def _grep_real_fs(
    pattern: re.Pattern,
    base_path: Path,
    type_filter: set[str] | None,
    output_mode: str,
    max_results: int,
    context_lines: int,
) -> str:
    cwd = Path.cwd()

    def _iter() -> Iterable[tuple[str, list[str]]]:
        for filepath in _walk_text_files(base_path, type_filter):
            try:
                lines = Path(filepath).read_text(encoding="utf-8", errors="replace").splitlines()
            except Exception:
                continue
            p = Path(filepath)
            dp = str(p.relative_to(cwd)) if p.is_relative_to(cwd) else filepath
            yield dp, lines

    return _search_files(
        pattern, _iter(), output_mode, context_lines, max_results,
        f"No matches for '{pattern.pattern}' in {base_path}",
    )


_GREP_PARAMS = {
    "pattern": {"type": "string", "description": "Regex pattern to search for"},
    "path": {
        "type": "string",
        "description": "Directory to search in (defaults to current directory). Also accepts a single file path",
        "default": ".",
    },
    "type": {
        "type": "string",
        "description": "File extension filter, e.g. 'py', 'js', 'go' (comma-separated for multiple)",
        "default": "",
    },
    "output_mode": {
        "type": "string",
        "description": "Output format: 'content' shows lines, 'files' shows file paths, 'count' shows match counts",
        "enum": ["content", "files", "count"],
        "default": "content",
    },
    "context": {
        "type": "integer",
        "description": "Number of context lines before and after each match",
        "default": 0,
    },
    "limit": {
        "type": "integer",
        "description": "Max results (default 100)",
        "default": 100,
    },
    "ignore_case": {
        "type": "boolean",
        "description": "Case-insensitive matching (default false)",
        "default": False,
    },
}
_GREP_DESC = (
    "Search file contents for a regex pattern across a directory tree. "
    "Automatically excludes .git, node_modules, __pycache__, and other common non-project directories. "
    "Only searches text files (skips binary files by extension). "
    "Use 'type' to filter by file extension (e.g. 'py' for Python files). "
    "Use 'context' to show surrounding lines. Use 'output_mode' to control output format. "
    "Use 'ignore_case' for case-insensitive matching."
)


class GrepTool(Tool):
    """Search file contents for a regex pattern."""

    def __init__(self) -> None:
        super().__init__(
            name="grep", description=_GREP_DESC, params=_GREP_PARAMS, func=self._execute
        )

    def _execute(self, args: dict) -> str:
        raw_ic = args.get("ignore_case", False)
        ignore_case = (
            str(raw_ic).lower() in ("true", "1", "yes") if isinstance(raw_ic, str) else bool(raw_ic)
        )
        flags = re.IGNORECASE if ignore_case else 0
        pattern = re.compile(args["pattern"], flags)
        base_path = args.get("path", ".")
        max_results = int(args.get("limit", 100)) or 100
        type_filter = _get_type_filter(args.get("type", ""))
        output_mode = args.get("output_mode", "content")
        context_lines = int(args.get("context", 0))

        target = Path(base_path).resolve()
        if target.is_file():
            return _grep_single_file(pattern, target, output_mode, context_lines, max_results)
        real_path = require_dir(target)
        return _grep_real_fs(pattern, real_path, type_filter, output_mode, max_results, context_lines)


__all__ = ["GrepTool"]
