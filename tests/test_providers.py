"""Built-in provider factory tests.

Verifies each factory returns a correctly configured OpenAIProvider:
correct base_url, default model, and matching preset — without hitting
any real API (we drive _build_request directly).
"""

from __future__ import annotations

import pytest

from ezwork.core.provider import ThinkingPreset
from ezwork.providers import (
    DeepSeek,
    DeepSeekPreset,
    GLM,
    GLMPreset,
    LongCat,
    LongCatPreset,
    Mimo,
    MimoPreset,
    MiniMax,
    MiniMaxPreset,
    OpenAIProvider,
)


# ---- factory return types ----------------------------------------------


@pytest.mark.parametrize("factory", [LongCat, DeepSeek, GLM, Mimo, MiniMax])
def test_factory_returns_openai_provider(factory):
    p = factory(api_key="sk-test")
    assert isinstance(p, OpenAIProvider)
    assert p.model  # non-empty default


@pytest.mark.parametrize(
    "factory,expected_url",
    [
        (LongCat, "https://api.longcat.chat/openai/v1"),
        (DeepSeek, "https://api.deepseek.com"),
        (GLM, "https://open.bigmodel.cn/api/paas/v4"),
        (Mimo, "https://token-plan-cn.xiaomimimo.com/v1"),
        (MiniMax, "https://api.minimax.chat/v1"),
    ],
)
def test_factory_base_url(factory, expected_url):
    p = factory(api_key="sk-test")
    assert p._base_url == expected_url


@pytest.mark.parametrize(
    "factory,expected_model",
    [
        (LongCat, "LongCat-2.0"),
        (DeepSeek, "deepseek-v4-pro"),
        (GLM, "glm-5.2"),
        (Mimo, "mimo-v2.5-pro"),
        (MiniMax, "MiniMax-M3"),
    ],
)
def test_factory_default_model(factory, expected_model):
    assert factory(api_key="sk-test").model == expected_model


def test_factory_model_override():
    p = DeepSeek(api_key="sk-test", model="deepseek-v4-flash")
    assert p.model == "deepseek-v4-flash"


def test_factory_model_none_uses_default():
    p = LongCat(api_key="sk-test", model=None)
    assert p.model == "LongCat-2.0"


@pytest.mark.parametrize(
    "factory,preset_cls",
    [
        (LongCat, LongCatPreset),
        (DeepSeek, DeepSeekPreset),
        (GLM, GLMPreset),
        (Mimo, MimoPreset),
        (MiniMax, MiniMaxPreset),
    ],
)
def test_factory_attaches_preset(factory, preset_cls):
    p = factory(api_key="sk-test")
    assert isinstance(p._thinking_preset, preset_cls)
    assert isinstance(p._thinking_preset, ThinkingPreset)


# ---- thinking shapes per vendor ----------------------------------------


def test_longcat_thinking_no_effort():
    p = LongCat(api_key="sk-test")
    _, extra = p._build_request([], "s", [], 100, True, thinking=True, reasoning_effort="high")
    assert extra["thinking"] == {"type": "enabled"}
    # LongCat has no effort — even if caller passes one, it's dropped
    assert "reasoning_effort" not in extra


def test_deepseek_thinking_with_effort():
    p = DeepSeek(api_key="sk-test")
    _, extra = p._build_request([], "s", [], 100, True, thinking=True, reasoning_effort="max")
    assert extra["thinking"] == {"type": "enabled"}
    assert extra["reasoning_effort"] == "max"


def test_deepseek_thinking_disabled_drops_effort():
    p = DeepSeek(api_key="sk-test")
    _, extra = p._build_request([], "s", [], 100, True, thinking=False, reasoning_effort="max")
    assert extra["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in extra


def test_minimax_thinking_uses_adaptive_and_split():
    p = MiniMax(api_key="sk-test")
    _, extra = p._build_request([], "s", [], 100, True, True, None)
    assert extra["thinking"] == {"type": "adaptive"}
    assert extra["reasoning_split"] is True


def test_no_thinking_arg_leaves_preset_silent():
    """thinking=None means 'provider default'; preset should not inject anything."""
    p = LongCat(api_key="sk-test")
    _, extra = p._build_request([], "s", [], 100, True, None, None)
    assert extra is None or "thinking" not in extra


# ---- extra_body passthrough --------------------------------------------


def test_factory_extra_body_merged():
    p = LongCat(api_key="sk-test", extra_body={"custom_flag": True})
    _, extra = p._build_request([], "s", [], 100, True, True, None)
    assert extra["custom_flag"] is True
    assert extra["thinking"] == {"type": "enabled"}


# ---- presets are protocols ---------------------------------------------


@pytest.mark.parametrize(
    "preset_cls",
    [LongCatPreset, DeepSeekPreset, GLMPreset, MimoPreset, MiniMaxPreset],
)
def test_preset_satisfies_protocol(preset_cls):
    assert isinstance(preset_cls(), ThinkingPreset)
