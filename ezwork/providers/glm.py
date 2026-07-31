"""Zhipu GLM provider (OpenAI-compatible).

Endpoint: https://open.bigmodel.cn/api/paas/v4
Default model: glm-5.2
Thinking: same shape as DeepSeek (effort: high, max).
"""

from __future__ import annotations

from typing import Any

from ezwork.core import ThinkingParams
from .openai import OpenAIProvider

BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
DEFAULT_MODEL = "glm-5.2"


class GLMPreset:
    """thinking.type + reasoning_effort (high, max). Default leans to 'max'."""

    default_enabled = True
    effort_levels = ["high", "max"]
    default_effort = "max"

    def build_params(self, enabled: bool, effort: str | None = None) -> ThinkingParams:
        body: dict = {"thinking": {"type": "enabled" if enabled else "disabled"}}
        top: dict = {}
        if enabled and effort:
            top["reasoning_effort"] = effort
        return ThinkingParams(body=body, top_params=top)


def GLM(
    api_key: str,
    *,
    model: str | None = None,
    extra_body: dict[str, Any] | None = None,
    **kwargs: Any,
) -> OpenAIProvider:
    """Build a GLM OpenAIProvider. Pass model= to override the default."""
    base_url = kwargs.pop("base_url", None) or BASE_URL
    return OpenAIProvider(
        api_key=api_key,
        model=model or DEFAULT_MODEL,
        base_url=base_url,
        thinking_preset=GLMPreset(),
        extra_body=extra_body,
        **kwargs,
    )


__all__ = ["GLM", "GLMPreset", "BASE_URL", "DEFAULT_MODEL"]
