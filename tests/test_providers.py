"""Built-in provider factory tests.

Verifies each factory returns a correctly configured OpenAIProvider:
correct base_url, default model, and matching preset — without hitting
any real API (we drive _build_request directly).
"""

from __future__ import annotations

import pytest

from ezwork.core.provider import ThinkingPreset
from ezwork.app.config import Config
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


@pytest.mark.parametrize("factory", [LongCat, DeepSeek, GLM, Mimo, MiniMax])
def test_factory_base_url_override(factory):
    """Config base_url must override the vendor default.

    Regression: the factories hardcoded base_url=BASE_URL while forwarding
    **kwargs, so a non-empty base_url crashed with "got multiple values
    for keyword argument 'base_url'" (seen when pointing DeepSeek at the
    opencode.ai zen gateway).
    """
    p = factory(api_key="sk-test", base_url="https://opencode.ai/zen/go/v1")
    assert p._base_url == "https://opencode.ai/zen/go/v1"


def test_config_build_provider_passes_overrides():
    """Config.build_provider() must forward base_url/model/extra_body.

    Regression: the full config→factory path crashed on a non-empty
    base_url (kwargs collision in the vendor factories).
    """
    cfg = Config.from_dict(
        {
            "provider": "deepseek",
            "api_key": "sk-test",
            "base_url": "https://opencode.ai/zen/go/v1/chat/completions",
            "model": "deepseek-v4-flash",
            "extra_body": {"custom_flag": True},
        }
    )
    p = cfg.build_provider()
    assert p._base_url == "https://opencode.ai/zen/go/v1/chat/completions"
    assert p.model == "deepseek-v4-flash"
    assert p._extra_body == {"custom_flag": True}


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


# ---- client warmup / preload ---------------------------------------------


def test_warmup_builds_client_in_background_thread(monkeypatch):
    """warmup() constructs the client (and preloads error classes) in a
    daemon thread, so the first stream() call doesn't pay the ~2-3s SDK
    import + construction inline."""
    import threading
    import time as _time

    p = OpenAIProvider(api_key="sk-test")
    calls: dict = {}

    def fake_ensure():
        calls["n"] = calls.get("n", 0) + 1
        calls["thread"] = threading.current_thread().name

    monkeypatch.setattr(p, "_ensure_client", fake_ensure)
    monkeypatch.setattr(
        OpenAIProvider, "_load_exc_classes", classmethod(lambda cls: None)
    )
    p.warmup()
    deadline = _time.monotonic() + 2
    while not calls.get("n") and _time.monotonic() < deadline:
        _time.sleep(0.01)
    assert calls.get("n") == 1
    assert calls.get("thread") != threading.main_thread().name


def test_warmup_is_noop_after_client_exists(monkeypatch):
    """Calling warmup() twice must not spawn a second thread/client."""
    import threading
    import time as _time

    p = OpenAIProvider(api_key="sk-test")
    calls = {"n": 0}

    def fake_ensure():
        calls["n"] += 1
        p._client = object()  # warmup()'s no-op guard checks this

    monkeypatch.setattr(p, "_ensure_client", fake_ensure)
    monkeypatch.setattr(
        OpenAIProvider, "_load_exc_classes", classmethod(lambda cls: None)
    )
    p.warmup()
    deadline = _time.monotonic() + 2
    while not calls["n"] and _time.monotonic() < deadline:
        _time.sleep(0.01)
    p.warmup()  # client is now set — must not spawn again
    _time.sleep(0.05)
    assert calls["n"] == 1


def test_preload_imports_openai():
    """preload() pulls the SDK into sys.modules without building a client."""
    import sys

    from ezwork.providers.openai import preload

    sys.modules.pop("openai", None)  # force a real (re)import
    preload()
    assert "openai" in sys.modules
