"""ThinkingPreset integration tests.

We don't hit a real API. Instead we drive OpenAIProvider._build_request
directly to assert that preset output is merged into extra_body correctly,
and that the provider itself hardcodes no thinking field names.

Key design property under test: all vendor-specific (non-OpenAI-standard)
fields must go through extra_body, because the typed openai SDK rejects
unknown kwargs like `thinking`, `reasoning_split`, etc.
"""

from dataclasses import dataclass, field
from typing import Any

from ezwork.core.provider import ThinkingParams
from ezwork.providers.openai import OpenAIProvider


# ---- a couple of representative presets (mirrors ezwork-ts shapes) ----


class _DeepSeekStylePreset:
    """thinking.type (body) + reasoning_effort (top-level)."""

    default_enabled = True
    effort_levels = ["high", "max"]
    default_effort = "high"

    def build_params(self, enabled: bool, effort: str | None = None) -> ThinkingParams:
        body = {"thinking": {"type": "enabled" if enabled else "disabled"}}
        top = {}
        if enabled and effort:
            top["reasoning_effort"] = effort
        return ThinkingParams(body=body, top_params=top)


class _MiniMaxStylePreset:
    """thinking.type=adaptive + reasoning_split."""

    default_enabled = True
    effort_levels = []
    default_effort = ""

    def build_params(self, enabled: bool, effort: str | None = None) -> ThinkingParams:
        return ThinkingParams(
            body={
                "thinking": {"type": "adaptive" if enabled else "disabled"},
                "reasoning_split": enabled,
            }
        )


# ---- tests -------------------------------------------------------------


def _provider_with(preset) -> OpenAIProvider:
    return OpenAIProvider(
        api_key="sk-test",
        model="m",
        base_url="http://x",
        extra_body={"existing": 1},
        thinking_preset=preset,
    )


def test_kwargs_never_contain_vendor_fields():
    """The SDK-typed kwargs must never carry vendor-specific keys."""
    p = _provider_with(_DeepSeekStylePreset())
    kwargs, extra_body = p._build_request(
        [], "sys", [], 100, True, thinking=True, reasoning_effort="high"
    )
    for forbidden in ("thinking", "reasoning_effort", "reasoning_split"):
        assert forbidden not in kwargs, f"{forbidden} leaked into SDK kwargs"


def test_no_preset_no_thinking_fields_anywhere():
    p = OpenAIProvider(api_key="k", model="m")  # no preset
    kwargs, extra_body = p._build_request(
        [], "sys", [], 100, False, thinking=True, reasoning_effort="high"
    )
    assert "thinking" not in kwargs
    assert extra_body is None or "thinking" not in extra_body


def test_no_thinking_arg_preset_not_consulted():
    p = _provider_with(_DeepSeekStylePreset())
    # thinking=None → preset not consulted
    kwargs, extra_body = p._build_request(
        [], "sys", [], 100, False, thinking=None, reasoning_effort="high"
    )
    assert "thinking" not in extra_body
    assert "reasoning_effort" not in extra_body


def test_deepseek_style_enabled_with_effort():
    p = _provider_with(_DeepSeekStylePreset())
    kwargs, extra_body = p._build_request(
        [], "sys", [], 100, False, thinking=True, reasoning_effort="max"
    )
    assert extra_body["thinking"] == {"type": "enabled"}
    assert extra_body["reasoning_effort"] == "max"


def test_deepseek_style_disabled_drops_effort():
    p = _provider_with(_DeepSeekStylePreset())
    kwargs, extra_body = p._build_request(
        [], "sys", [], 100, False, thinking=False, reasoning_effort="max"
    )
    assert extra_body["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in extra_body


def test_minimax_style_merges_body():
    p = _provider_with(_MiniMaxStylePreset())
    kwargs, extra_body = p._build_request(
        [], "sys", [], 100, False, thinking=True, reasoning_effort=None
    )
    assert extra_body["thinking"] == {"type": "adaptive"}
    assert extra_body["reasoning_split"] is True


def test_existing_extra_body_preserved():
    p = _provider_with(_DeepSeekStylePreset())
    kwargs, extra_body = p._build_request(
        [], "sys", [], 100, False, thinking=True, reasoning_effort="high"
    )
    assert extra_body["existing"] == 1
    assert extra_body["thinking"] == {"type": "enabled"}


def test_kwargs_carry_sdk_standard_fields_only():
    p = _provider_with(_DeepSeekStylePreset())
    kwargs, extra_body = p._build_request(
        [], "sys", [{"type": "function", "function": {"name": "t"}}], 100,
        stream=True, thinking=True, reasoning_effort="high",
    )
    # only these keys are valid openai SDK kwargs
    assert set(kwargs.keys()) <= {
        "model", "messages", "max_tokens", "stream", "tools", "stream_options",
        "tool_choice", "temperature", "top_p", "n", "presence_penalty",
        "frequency_penalty", "user", "seed", "stop", "response_format",
        "logit_bias", "logprobs", "top_logprobs", "service_tier",
    }


def test_preset_returned_as_thinking_params_type():
    tp = _DeepSeekStylePreset().build_params(True, "high")
    assert isinstance(tp, ThinkingParams)
    assert tp.body == {"thinking": {"type": "enabled"}}
    assert tp.top_params == {"reasoning_effort": "high"}


def test_preset_satisfies_protocol():
    from ezwork.core.provider import ThinkingPreset

    assert isinstance(_DeepSeekStylePreset(), ThinkingPreset)
    assert isinstance(_MiniMaxStylePreset(), ThinkingPreset)
