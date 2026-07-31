"""Built-in tools: read, write, edit, bash, glob, grep.

A sibling package to `providers/`. The kernel ships NO tools; these are the
reference implementations an app would bundle. Register them like:

    from ezwork.core import ToolRegistry
    from tools import ReadTool, WriteTool, EditTool, BashTool, GlobTool, GrepTool

    reg = ToolRegistry()
    for t in [ReadTool(), WriteTool(), EditTool(), BashTool(), GlobTool(), GrepTool()]:
        reg.register(t)
"""

from __future__ import annotations

__all__ = [
    "ReadTool",
    "WriteTool",
    "EditTool",
    "BashTool",
    "GlobTool",
    "GrepTool",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "ReadTool": (".file", "ReadTool"),
    "WriteTool": (".file", "WriteTool"),
    "EditTool": (".file", "EditTool"),
    "BashTool": (".bash", "BashTool"),
    "GlobTool": (".glob", "GlobTool"),
    "GrepTool": (".grep", "GrepTool"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        module_path, attr = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path, __name__)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'tools' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY_IMPORTS.keys()))
