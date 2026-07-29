"""Shared tool utilities used by the built-in tools."""

from __future__ import annotations

from pathlib import Path

from ezwork.core import ToolError


def decode_bytes(data: bytes) -> str:
    """Decode bytes trying common encodings (utf-8 first, then GBK family
    for Windows/CJK environments)."""
    if not data:
        return ""
    for encoding in ("utf-8", "gbk", "cp936", "gb2312"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def require_file(p: Path) -> Path:
    """Validate path exists and is not a directory."""
    if not p.exists():
        raise ToolError(f"File not found: {p}", "file_not_found")
    if p.is_dir():
        raise ToolError(f"Path is a directory: {p}", "invalid_path")
    return p


# Directories ignored when listing / walking.
IGNORE_DIRS = frozenset(
    {
        ".git", ".svn", ".hg",
        "__pycache__", ".venv", "venv", "env", ".tox",
        ".pytest_cache", ".mypy_cache", ".ruff_cache",
        "node_modules", ".next", ".nuxt",
        "target", "dist", "build",
        ".idea", ".vscode",
        ".cache", "coverage",
    }
)
