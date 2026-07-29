"""MiniMax provider (OpenAI-compatible).

Endpoint: https://api.minimax.chat/v1
Default model: MiniMax-M3
Thinking: thinking.type=adaptive|disabled + reasoning_split.

`reasoning_split=true` makes the API return thinking via the standard
`reasoning_content` stream field, so OpenAIProvider's stream parser picks
it up unchanged.
"""

from __future__ import annotations

from typing import Any

from ezwork.core import ThinkingParams
from .openai import OpenAIProvider

BASE_URL = "https://api.minimax.chat/v1"
DEFAULT_MODEL = "MiniMax-M3"


class MiniMaxPreset:
    """thinking.type=adaptive|disabled + reasoning_split."""

    default_enabled = True
    effort_levels: list[str] = []
    default_effort = ""

    def build_params(self, enabled: bool, effort: str | None = None) -> ThinkingParams:
        return ThinkingParams(
            body={
                "thinking": {"type": "adaptive" if enabled else "disabled"},
                "reasoning_split": enabled,
            }
        )


def MiniMax(
    api_key: str,
    *,
    model: str | None = None,
    extra_body: dict[str, Any] | None = None,
    **kwargs: Any,
) -> OpenAIProvider:
    """Build a MiniMax OpenAIProvider. Pass model= to override the default."""
    return OpenAIProvider(
        api_key=api_key,
        model=model or DEFAULT_MODEL,
        base_url=BASE_URL,
        thinking_preset=MiniMaxPreset(),
        extra_body=extra_body,
        **kwargs,
    )


__all__ = ["MiniMax", "MiniMaxPreset", "BASE_URL", "DEFAULT_MODEL"]
