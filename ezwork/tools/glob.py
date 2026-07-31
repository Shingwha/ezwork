"""Glob tool — find files by pattern, sorted by modification time.

Ported from the legacy MoCode design, with the VFS layer stripped (ezwork has
no virtual filesystem). Pure Python, cross-platform — no dependency on shell
`find`/`ls` behaviour.
"""

from __future__ import annotations

import os
from pathlib import Path

from ezwork.core import Tool

from .utils import IGNORE_DIRS, _GLOB_MAX, require_dir

_GLOB_PARAMS = {
    "pattern": {
        "type": "string",
        "description": "Glob pattern (e.g. '**/*.py', 'src/**/*.ts')",
    },
    "path": {
        "type": "string",
        "description": "Base directory to search in (defaults to current directory)",
        "default": ".",
    },
}
_GLOB_DESC = (
    "Find files matching a glob pattern, sorted by modification time (newest first). "
    "Automatically excludes .git, node_modules, __pycache__, and other common "
    "non-project directories."
)


class GlobTool(Tool):
    """Find files matching a glob pattern."""

    def __init__(self) -> None:
        super().__init__(
            name="glob", description=_GLOB_DESC, params=_GLOB_PARAMS, func=self._execute
        )

    def _execute(self, args: dict) -> str:
        pattern = args["pattern"]
        base = require_dir(Path(args.get("path", ".")).resolve())
        files = sorted(
            (
                p
                for p in base.glob(pattern)
                if p.is_file() and not any(part in IGNORE_DIRS for part in p.relative_to(base).parts)
            ),
            key=lambda f: os.path.getmtime(f),
            reverse=True,
        )
        if not files:
            return f"No files matching '{pattern}' in {base}"
        truncated = len(files) > _GLOB_MAX
        files = files[:_GLOB_MAX]
        cwd = Path.cwd()
        paths = [str(p.relative_to(base)) if base == cwd else str(p) for p in files]
        result = f"[Found {len(files)}{'+' if truncated else ''} file(s) matching '{pattern}']\n"
        result += "\n".join(paths)
        if truncated:
            result += f"\n... and more files not shown (showing first {_GLOB_MAX})"
        return result


__all__ = ["GlobTool"]
