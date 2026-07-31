"""LongCat provider (OpenAI-compatible).

Endpoint: https://api.longcat.chat/openai/v1
Default model: LongCat-2.0 (128K context, 128K max output)
Thinking: thinking.type=enabled|disabled, no effort control.
"""

from __future__ import annotations

from typing import Any

from ezwork.core import ThinkingParams
from .openai import OpenAIProvider

BASE_URL = "https://api.longcat.chat/openai/v1"
DEFAULT_MODEL = "LongCat-2.0"


class LongCatPreset:
    """thinking.type only, no effort control."""

    default_enabled = True
    effort_levels: list[str] = []
    default_effort = ""

    def build_params(self, enabled: bool, effort: str | None = None) -> ThinkingParams:
        return ThinkingParams(body={"thinking": {"type": "enabled" if enabled else "disabled"}})


def LongCat(
    api_key: str,
    *,
    model: str | None = None,
    extra_body: dict[str, Any] | None = None,
    **kwargs: Any,
) -> OpenAIProvider:
    """Build a LongCat OpenAIProvider. Pass model= to override the default."""
    base_url = kwargs.pop("base_url", None) or BASE_URL
    return OpenAIProvider(
        api_key=api_key,
        model=model or DEFAULT_MODEL,
        base_url=base_url,
        thinking_preset=LongCatPreset(),
        extra_body=extra_body,
        **kwargs,
    )


__all__ = ["LongCat", "LongCatPreset", "BASE_URL", "DEFAULT_MODEL"]
