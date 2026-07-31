"""DeepSeek provider (OpenAI-compatible).

Endpoint: https://api.deepseek.com
Default model: deepseek-v4-pro
Thinking: thinking.type=enabled|disabled + top-level reasoning_effort
          (effort levels: high, max).
"""

from __future__ import annotations

from typing import Any

from ezwork.core import ThinkingParams
from .openai import OpenAIProvider

BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-pro"


class DeepSeekPreset:
    """thinking.type + reasoning_effort (high, max)."""

    default_enabled = True
    effort_levels = ["high", "max"]
    default_effort = "high"

    def build_params(self, enabled: bool, effort: str | None = None) -> ThinkingParams:
        body: dict = {"thinking": {"type": "enabled" if enabled else "disabled"}}
        top: dict = {}
        if enabled and effort:
            top["reasoning_effort"] = effort
        return ThinkingParams(body=body, top_params=top)


def DeepSeek(
    api_key: str,
    *,
    model: str | None = None,
    extra_body: dict[str, Any] | None = None,
    **kwargs: Any,
) -> OpenAIProvider:
    """Build a DeepSeek OpenAIProvider. Pass model= to override the default."""
    base_url = kwargs.pop("base_url", None) or BASE_URL
    return OpenAIProvider(
        api_key=api_key,
        model=model or DEFAULT_MODEL,
        base_url=base_url,
        thinking_preset=DeepSeekPreset(),
        extra_body=extra_body,
        **kwargs,
    )


__all__ = ["DeepSeek", "DeepSeekPreset", "BASE_URL", "DEFAULT_MODEL"]
