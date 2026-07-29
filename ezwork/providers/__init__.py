"""Provider implementations.

- `OpenAIProvider` is the generic OpenAI-compatible implementation.
- One module per built-in vendor (`longcat.py`, `deepseek.py`, etc.). Each
  module bundles the vendor's preset, base_url, default model, and a factory
  function of the same name as the vendor. To add a vendor, copy one of these
  modules — see docs/providers.md.
"""

from __future__ import annotations

__all__ = [
    "OpenAIProvider",
    # built-in vendors (factory + preset live in each vendor's module)
    "LongCat",
    "LongCatPreset",
    "DeepSeek",
    "DeepSeekPreset",
    "GLM",
    "GLMPreset",
    "Mimo",
    "MimoPreset",
    "MiniMax",
    "MiniMaxPreset",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "OpenAIProvider": (".openai", "OpenAIProvider"),
    "LongCat": (".longcat", "LongCat"),
    "LongCatPreset": (".longcat", "LongCatPreset"),
    "DeepSeek": (".deepseek", "DeepSeek"),
    "DeepSeekPreset": (".deepseek", "DeepSeekPreset"),
    "GLM": (".glm", "GLM"),
    "GLMPreset": (".glm", "GLMPreset"),
    "Mimo": (".mimo", "Mimo"),
    "MimoPreset": (".mimo", "MimoPreset"),
    "MiniMax": (".minimax", "MiniMax"),
    "MiniMaxPreset": (".minimax", "MiniMaxPreset"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        module_path, attr = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path, __name__)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'providers' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY_IMPORTS.keys()))
