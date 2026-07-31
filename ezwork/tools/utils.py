"""Shared tool utilities used by the built-in tools."""

from __future__ import annotations

import os
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


def require_dir(p: Path) -> Path:
    """Validate path exists and is a directory."""
    if not p.is_dir():
        raise ToolError(f"Directory not found: {p}", "path_not_found")
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
        ".cache", "coverage", ".gradle", ".terraform",
    }
)


# ─── grep output formatting ───────────────────────────────────────────────


def expand_context_indices(
    match_indices: list[int], total_lines: int, context: int
) -> list[int]:
    """Expand match indices to include surrounding context lines.

    Returns sorted list of all line indices to display (including matches and
    context)."""
    if context <= 0:
        return match_indices
    expanded: set[int] = set()
    for idx in match_indices:
        for j in range(
            max(0, idx - context),
            min(total_lines, idx + context + 1),
        ):
            expanded.add(j)
    return sorted(expanded)


def format_grep_content(
    file_lines: list[str],
    match_indices: list[int],
    display_indices: list[int],
    display_path: str,
    pattern_str: str,
    max_results: int,
    hits: list[str],
) -> str:
    """Format grep content-mode output for one file's matches.

    Appends formatted hit lines to *hits* (mutated in-place). Returns the
    final formatted string if max_results is reached, else empty string."""
    match_set = set(match_indices)
    for idx in display_indices:
        sep = ":" if idx in match_set else "-"
        hits.append(f"{display_path}{sep}{idx + 1}{sep}{file_lines[idx]}")
        if len(hits) >= max_results:
            return (
                f"[Showing {len(hits)} matches for '{pattern_str}']\n"
                + "\n".join(hits)
            )
    return ""


def format_grep_files(files: list[str], pattern_str: str) -> str:
    """Format grep files-mode output."""
    if not files:
        return f"No files matching '{pattern_str}'"
    header = f"[Found {len(files)} file(s)]"
    return header + "\n" + "\n".join(files)


def format_grep_count(entries: list[str], pattern_str: str) -> str:
    """Format grep count-mode output. *entries* are 'path:count' strings."""
    if not entries:
        return f"No matches for '{pattern_str}'"
    return "\n".join(entries)


# ─── search constants ─────────────────────────────────────────────────────

TYPE_MAP = {
    "py": ".py", "js": ".js", "ts": ".ts", "tsx": ".tsx", "jsx": ".jsx",
    "go": ".go", "rs": ".rs", "java": ".java", "c": ".c", "cpp": ".cpp",
    "h": ".h", "rb": ".rb", "php": ".php", "cs": ".cs", "swift": ".swift",
    "kt": ".kt", "scala": ".scala", "lua": ".lua", "r": ".r",
    "html": ".html", "css": ".css", "vue": ".vue", "svelte": ".svelte",
    "json": ".json", "yaml": ".yaml", "yml": ".yml", "toml": ".toml",
    "md": ".md", "txt": ".txt", "sql": ".sql", "xml": ".xml", "sh": ".sh",
}

TEXT_EXTENSIONS = frozenset(
    set(TYPE_MAP.values())
    | {
        ".hpp", ".m", ".mm", ".bash", ".zsh", ".ps1", ".bat", ".cmd", ".fish",
        ".ini", ".cfg", ".conf", ".env", ".rst", ".adoc", ".tex", ".org",
        ".scss", ".less", ".sass", ".svg", ".csv", ".tsv",
        ".dockerfile", ".makefile", ".cmake",
        ".gitignore", ".gitattributes", ".editorconfig",
    }
)

_GLOB_MAX = 200


def _is_text_file(path: str) -> bool:
    suffix = Path(path).suffix.lower()
    return suffix in TEXT_EXTENSIONS or not suffix


def _get_type_filter(type_str: str) -> set[str] | None:
    if not type_str:
        return None
    extensions: set[str] = set()
    for t in type_str.split(","):
        t = t.strip().lower()
        if t in TYPE_MAP:
            extensions.add(TYPE_MAP[t])
        else:
            ext = t if t.startswith(".") else f".{t}"
            extensions.add(ext)
    return extensions


def _walk_text_files(base_path: Path, type_filter: set[str] | None):
    """Yield text file paths under base_path, respecting IGNORE_DIRS and
    the optional extension filter."""
    for root, dirs, _files in os.walk(base_path):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for filename in _files:
            if not _is_text_file(filename):
                continue
            if type_filter and Path(filename).suffix.lower() not in type_filter:
                continue
            yield os.path.join(root, filename)
