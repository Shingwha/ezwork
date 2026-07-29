"""Xiaomi Mimo provider (OpenAI-compatible).

Endpoint: https://token-plan-cn.xiaomimimo.com/v1
Default model: mimo-v2.5-pro
Thinking: thinking.type only, no effort control.
"""

from __future__ import annotations

from typing import Any

from ezwork.core import ThinkingParams
from .openai import OpenAIProvider

BASE_URL = "https://token-plan-cn.xiaomimimo.com/v1"
DEFAULT_MODEL = "mimo-v2.5-pro"


class MimoPreset:
    """thinking.type only, no effort control."""

    default_enabled = True
    effort_levels: list[str] = []
    default_effort = ""

    def build_params(self, enabled: bool, effort: str | None = None) -> ThinkingParams:
        return ThinkingParams(body={"thinking": {"type": "enabled" if enabled else "disabled"}})


def Mimo(
    api_key: str,
    *,
    model: str | None = None,
    extra_body: dict[str, Any] | None = None,
    **kwargs: Any,
) -> OpenAIProvider:
    """Build a Mimo OpenAIProvider. Pass model= to override the default."""
    return OpenAIProvider(
        api_key=api_key,
        model=model or DEFAULT_MODEL,
        base_url=BASE_URL,
        thinking_preset=MimoPreset(),
        extra_body=extra_body,
        **kwargs,
    )


__all__ = ["Mimo", "MimoPreset", "BASE_URL", "DEFAULT_MODEL"]
