"""Prompt building — section-based, plain-text rendering.

Deliberately minimal. The kernel only flattens sections into text; it does no
nesting, no markup, no tags. Each enabled section renders as:

    name: <content>

…for single-line content, or:

    name:
    <content>

…for multi-line content. Sections are joined by a blank line and sorted by
(priority, name) so stable, low-priority sections come first (better prefix
cache hit rate) and dynamic, high-churn sections come last.

The kernel adds NO tags of its own. If an app wants XML, delimiters, or any
other markup, it writes them directly into the section content strings — the
kernel never parses or wraps content. This is the least-coupled shape: swap the
app, keep the kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self

Content = "str | Callable[[dict[str, Any]], str]"


@dataclass
class Section:
    name: str
    content: Any  # str | Callable[[dict[str, Any]], str]
    priority: int = 0
    enabled: bool = True


class Prompt:
    """Flat section-based prompt builder — dict-backed, API aligned with
    ToolRegistry (register/unregister/get/all). Renders plain text only."""

    def __init__(self, sections: list[Section] | None = None) -> None:
        self._sections: dict[str, Section] = {}
        self._context: dict[str, Any] = {}
        if sections:
            for s in sections:
                self._sections[s.name] = s

    def register(self, section: Section) -> Self:
        self._sections[section.name] = section
        return self

    def unregister(self, name: str) -> Section | None:
        return self._sections.pop(name, None)

    def get(self, name: str) -> Section | None:
        return self._sections.get(name)

    def all(self) -> list[Section]:
        return list(self._sections.values())

    def enable(self, name: str) -> Self:
        s = self.get(name)
        if s:
            s.enabled = True
        return self

    def disable(self, name: str) -> Self:
        s = self.get(name)
        if s:
            s.enabled = False
        return self

    def context(self, **kwargs: Any) -> Self:
        self._context.update(kwargs)
        return self

    def build(self) -> str:
        """Render all enabled sections, sorted by (priority, name), joined by
        blank lines. Plain text only — no markup, no nesting."""
        sorted_sections = sorted(
            self._sections.values(), key=lambda s: (s.priority, s.name)
        )
        parts: list[str] = []
        for s in sorted_sections:
            if not s.enabled:
                continue
            content = s.content(self._context) if callable(s.content) else s.content
            if not content:
                continue
            if "\n" in content:
                parts.append(f"{s.name}:\n{content}")
            else:
                parts.append(f"{s.name}: {content}")
        return "\n\n".join(parts)

    def __repr__(self) -> str:
        sections = ", ".join(
            f"{s.name}({'on' if s.enabled else 'off'})" for s in self._sections.values()
        )
        return f"Prompt([{sections}])"


__all__ = ["Section", "Prompt", "Content"]
