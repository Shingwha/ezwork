"""Config — minimal single-provider configuration for the Ezwork app.

Lives at ~/.ezwork/config.json. On first run, if the file is missing, a template
is written and the user is told to edit it (the app never crashes on a missing
config). Only one provider is configured at a time; switching vendors means
editing the `provider` field and the corresponding credentials. This keeps the
config surface tiny and friendly for cloud-server one-shot installs.

JSON shape:

    {
      "provider": "longcat",          # longcat|deepseek|glm|mimo|minimax|openai
      "api_key": "your-api-key-here",
      "base_url": "",                 # optional override; "" = vendor default
      "model": "LongCat-2.0",
      "thinking": true,               # enable extended thinking
      "reasoning_effort": "",         # "" = vendor default; else e.g. "high"/"max"
      "max_tokens": 32768
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path.home() / ".ezwork" / "config.json"

# Placeholder written into the template; if api_key still equals this, the
# config is treated as not-yet-filled-in.
_API_KEY_PLACEHOLDER = "your-api-key-here"

# Default model per vendor, used when the template is generated.
_VENDOR_DEFAULTS: dict[str, dict[str, Any]] = {
    "longcat": {"model": "LongCat-2.0", "base_url": ""},
    "deepseek": {"model": "deepseek-v4-pro", "base_url": ""},
    "glm": {"model": "glm-5.2", "base_url": ""},
    "mimo": {"model": "mimo-v2.5-pro", "base_url": ""},
    "minimax": {"model": "MiniMax-M3", "base_url": ""},
    "openai": {"model": "gpt-4o", "base_url": ""},
}


@dataclass
class Config:
    """Single-provider configuration."""

    provider: str = "longcat"
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    thinking: bool = True
    reasoning_effort: str = ""
    max_tokens: int = 32768
    extra_body: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "provider": self.provider,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "model": self.model,
            "thinking": self.thinking,
            "reasoning_effort": self.reasoning_effort,
            "max_tokens": self.max_tokens,
        }
        if self.extra_body:
            d["extra_body"] = self.extra_body
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        return cls(
            provider=str(data.get("provider", "longcat")),
            api_key=str(data.get("api_key", "")),
            base_url=str(data.get("base_url", "") or ""),
            model=str(data.get("model", "")),
            thinking=bool(data.get("thinking", True)),
            reasoning_effort=str(data.get("reasoning_effort", "") or ""),
            max_tokens=int(data.get("max_tokens", 32768)),
            extra_body=dict(data.get("extra_body", {}) or {}),
        )

    # ---- Load / Save ----

    @classmethod
    def load(cls, path: Path | str = DEFAULT_CONFIG_PATH) -> "Config | None":
        """Load config from JSON. Returns None if the file is missing or
        unreadable. Use ensure_config() to also handle the template-create
        flow expected from the CLI."""
        p = Path(path)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8-sig"))
        except (json.JSONDecodeError, OSError):
            return None
        return cls.from_dict(data)

    def save(self, path: Path | str = DEFAULT_CONFIG_PATH) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @staticmethod
    def write_template(path: Path | str = DEFAULT_CONFIG_PATH) -> None:
        """Write the default config template (does not overwrite an existing file)."""
        p = Path(path)
        if p.exists():
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        c = Config()
        c.api_key = _API_KEY_PLACEHOLDER
        c.model = _VENDOR_DEFAULTS[c.provider]["model"]
        p.write_text(
            json.dumps(c.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def is_filled_in(self) -> bool:
        """True once the user has replaced the placeholder api_key."""
        return bool(self.api_key) and self.api_key != _API_KEY_PLACEHOLDER

    def build_provider(self) -> Any:
        """Construct the provider for this config. Factory names are registered
        in ezwork.providers.FACTORIES; base_url override applies only when
        non-empty. Raises ValueError for an unknown provider name."""
        import ezwork.providers as providers

        factory_name = providers.FACTORIES.get(self.provider)
        if factory_name is None:
            raise ValueError(
                f"unknown provider: {self.provider!r}. "
                f"valid values: {', '.join(providers.FACTORIES)}"
            )
        factory = getattr(providers, factory_name)  # triggers lazy import

        base_url = self.base_url or None
        # Every factory (OpenAIProvider and the vendor factories) accepts
        # api_key / model / base_url / extra_body as keywords — no special
        # cases needed per provider.
        kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.model:
            kwargs["model"] = self.model
        if base_url:
            kwargs["base_url"] = base_url
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        return factory(**kwargs)


__all__ = ["Config", "DEFAULT_CONFIG_PATH"]
