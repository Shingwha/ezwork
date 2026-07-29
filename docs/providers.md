# Adding a Provider

This guide covers everything you need to add a new LLM provider to Ezwork.
There are two distinct paths depending on whether the provider is
OpenAI-compatible or has its own native API.

## TL;DR — pick a path

| Provider speaks… | Path | Effort |
|------------------|------|--------|
| OpenAI Chat Completions (DeepSeek, GLM, LongCat, Mimo, MiniMax, Ollama, vLLM, …) | **Path A** — factory + optional preset | ~10 lines |
| Its own native API (Anthropic Messages, Gemini generateContent, …) | **Path B** — implement `Provider` protocol | ~150 lines |

Both paths end with the same shape: an object you pass to `Agent.provider(...)`.

---

## Path A — OpenAI-compatible provider

This is by far the most common case. The provider accepts OpenAI-format
messages, tool_calls, and (for streaming) OpenAI SSE chunks. You don't write
any network code — `OpenAIProvider` already does it. You just tell it where
to point and (optionally) how that vendor's "thinking" feature is shaped.

### A.1 — No thinking support (simplest)

If the vendor has no extended-thinking feature, just instantiate
`OpenAIProvider` with a `base_url`:

```python
from ezwork.providers import OpenAIProvider

# Example: a local Ollama server exposing an OpenAI-compatible API.
provider = OpenAIProvider(
    api_key="ollama",                 # any non-empty string; Ollama ignores it
    model="llama3.2",
    base_url="http://localhost:11434/v1",
)
agent = Agent().provider(provider).prompt("...").build()
```

Done. No new file, no preset.

### A.2 — With thinking support (add a ThinkingPreset)

If the vendor has extended thinking, the only thing that varies is the
request shape. Write a `ThinkingPreset` class — ONE method, `build_params`:

```python
# my_package/my_presets.py
from ezwork.core import ThinkingParams

class AcmePreset:
    default_enabled = True
    effort_levels = ["low", "medium", "high"]   # empty list = no effort control
    default_effort = "medium"

    def build_params(self, enabled: bool, effort: str | None = None) -> ThinkingParams:
        # Return whatever fields the vendor expects, in body and/or top_params.
        # Both end up merged into the request's extra_body (the typed openai
        # SDK rejects unknown kwargs, so all vendor-specific fields ride there).
        body = {"acme_thinking": {"mode": "on" if enabled else "off"}}
        top = {"acme_effort": effort} if enabled and effort else {}
        return ThinkingParams(body=body, top_params=top)
```

Then pass it to `OpenAIProvider`:

```python
from ezwork.providers import OpenAIProvider
from my_package.my_presets import AcmePreset

provider = OpenAIProvider(
    api_key="...",
    model="acme-1",
    base_url="https://api.acme.com/v1",
    thinking_preset=AcmePreset(),
)
```

Toggle thinking at runtime via the agent's `LoopConfig`:

```python
agent = (
    Agent()
    .provider(provider)
    .prompt("...")
    .thinking(True)              # or .thinking(False), .reasoning_effort("high")
    .build()
)
# or mutate between turns:
agent.config.thinking = False
```

**Key property**: `OpenAIProvider` itself hardcodes no thinking field names.
Every vendor's quirk is isolated in its preset. Switching vendors = swapping
one preset class.

### A.3 — Wrap it in a vendor module (recommended for reusable providers)

If you'll use the same vendor repeatedly, or want to ship it as a built-in,
create one module per vendor that bundles the preset, base_url, default
model, and a factory function together. This is the layout the built-ins
use:

```python
# providers/acme.py
from typing import Any
from ezwork.core import ThinkingParams
from ezwork.providers.openai import OpenAIProvider

BASE_URL = "https://api.acme.com/v1"
DEFAULT_MODEL = "acme-1"

class AcmePreset:
    default_enabled = True
    effort_levels = ["low", "medium", "high"]
    default_effort = "medium"

    def build_params(self, enabled: bool, effort: str | None = None) -> ThinkingParams:
        body = {"acme_thinking": {"mode": "on" if enabled else "off"}}
        top = {"acme_effort": effort} if enabled and effort else {}
        return ThinkingParams(body=body, top_params=top)

def Acme(
    api_key: str,
    *,
    model: str | None = None,
    extra_body: dict[str, Any] | None = None,
    **kwargs: Any,
) -> OpenAIProvider:
    """Build an Acme OpenAIProvider. Pass model= to override the default."""
    return OpenAIProvider(
        api_key=api_key,
        model=model or DEFAULT_MODEL,
        base_url=BASE_URL,
        thinking_preset=AcmePreset(),
        extra_body=extra_body,
        **kwargs,
    )
```

Then export both from `providers/__init__.py`:

```python
_LAZY_IMPORTS["Acme"] = (".acme", "Acme")
_LAZY_IMPORTS["AcmePreset"] = (".acme", "AcmePreset")
```

Callers then do:

```python
from ezwork.providers import Acme
provider = Acme(api_key="...")          # model defaults to DEFAULT_MODEL
provider = Acme(api_key="...", model="acme-1-mini")  # override
```

Why bundle preset + factory in one file: a vendor's thinking shape, endpoint,
and default model are facets of the same thing. Splitting them across
`presets.py` and `builtins.py` (an earlier design) forced readers to hop
between files. One vendor, one module.

### A.4 — Common thinking shapes (steal these)

The built-in vendor modules cover the shapes we've encountered. Copy the
closest one and tweak:

| Shape | Vendors | Body | Top params | Module to copy |
|-------|---------|------|------------|----------------|
| `thinking.type=enabled\|disabled` + `reasoning_effort` | DeepSeek, GLM | `{"thinking": {"type": ...}}` | `{"reasoning_effort": ...}` | `deepseek.py` |
| `thinking.type=enabled\|disabled` only | LongCat, Mimo | `{"thinking": {"type": ...}}` | — | `longcat.py` |
| `thinking.type=adaptive\|disabled` + `reasoning_split` | MiniMax | `{"thinking": {...}, "reasoning_split": ...}` | — | `minimax.py` |

Each module is ~50 lines including docstring and factory.

---

## Path B — Native-API provider (Anthropic, Gemini, …)

When the provider's wire format is fundamentally different from OpenAI's
(e.g. Anthropic's content blocks, Gemini's `generateContent`), implement the
`Provider` protocol directly.

### B.1 — The protocol

```python
# ezwork/provider.py (already defined — this is what you implement)
class Provider(Protocol):
    @property
    def model(self) -> str: ...

    def is_retriable(self, exc: Exception) -> bool: ...

    def stream(
        self,
        messages: list[dict],          # OpenAI-format dicts (see message.py)
        system: str,
        tools: list[dict],             # OpenAI function-calling schemas
        max_tokens: int,
        *,
        thinking: bool | None = None,
        reasoning_effort: str | None = None,
    ) -> AsyncIterator[StreamChunk]: ...
```

### B.2 — Your job in three steps

1. **Translate messages** at your boundary: convert the OpenAI-format dicts
   you receive into the vendor's native request shape.
2. **Stream chunks back** as `StreamChunk` instances. The agent loop's
   `collect_stream` will assemble them into a `Response`. Chunk types:
   - `StreamChunk.text_delta(text)` — incremental assistant text
   - `StreamChunk.reasoning_delta(text)` — incremental reasoning
   - `StreamChunk.tool_call_delta(index, id=, name=, arguments_delta=)` —
     tool call fragments; multiple deltas per call accumulate by index
   - `StreamChunk.usage(Usage(...))` — token counts (last one wins)
   - `StreamChunk.done(finish_reason)` — end of a successful stream
   - `StreamChunk.error(msg)` — stream failed; iterator should then end
3. **Encode errors, don't raise.** Transient failures become a `StreamChunk.error()`
   chunk followed by ending the iterator. The agent loop sees this as
   `Response(finish_reason="error")` and surfaces an `ErrorEvent`. Only
   `CancelledError` should propagate as an exception (that's the user's intent).

### B.3 — Skeleton

```python
# my_package/anthropic.py
from typing import Any, AsyncIterator
from ezwork.core import StreamChunk, Usage

class AnthropicProvider:
    def __init__(self, api_key: str, model: str = "claude-..."):
        self._api_key = api_key
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    def is_retriable(self, exc: Exception) -> bool:
        # typically: rate-limit, 5xx, connection errors
        ...

    async def stream(
        self,
        messages: list[dict],
        system: str,
        tools: list[dict],
        max_tokens: int,
        *,
        thinking: bool | None = None,
        reasoning_effort: str | None = None,
    ) -> AsyncIterator[StreamChunk]:
        # 1. translate OpenAI messages → Anthropic /v1/messages format
        req = self._to_anthropic(messages, system, tools, max_tokens, thinking)
        try:
            async with self._client.messages.stream(**req) as stream:
                async for event in stream:
                    for chunk in self._map_event(event):  # your mapping
                        yield chunk
        except Exception as exc:
            yield StreamChunk.error(str(exc))
            return

    def _to_anthropic(self, ...): ...
    def _map_event(self, event) -> list[StreamChunk]: ...
```

### B.4 — Why there's no base class

`Provider` is a `Protocol` (structural typing, not inheritance). Your class
just needs the right method signatures — no `extends`, no `super()` calls.
This keeps providers decoupled from kernel internals.

---

## Shipping and discovery

### As part of the `providers/` package

Create one module per vendor (e.g. `providers/acme.py`) containing the
preset, base_url, default model, and factory. Export both the factory and
the preset class from `providers/__init__.py`. The next `uv sync` makes them
available as `from ezwork.providers import Acme`.

### Via entry points

For pip-installed packages:

```toml
# pyproject.toml
[project.entry-points."ezwork.providers"]
acme = "my_package:Acme"
```

---

## Built-in providers

These ship with the `providers/` package:

| Factory | Vendor | Default model | Thinking shape |
|---------|--------|---------------|----------------|
| `LongCat(api_key)` | LongCat | `LongCat-2.0` | type only, no effort |
| `DeepSeek(api_key)` | DeepSeek | `deepseek-v4-pro` | type + `reasoning_effort` (high, max) |
| `GLM(api_key)` | Zhipu GLM | `glm-5.2` | type + `reasoning_effort` (high, max) |
| `Mimo(api_key)` | Xiaomi Mimo | `mimo-v2.5-pro` | type only, no effort |
| `MiniMax(api_key)` | MiniMax | `MiniMax-M3` | type=adaptive + `reasoning_split` |

All accept optional `model=`, `extra_body=`, and any other kwargs forwarded
to `OpenAIProvider` (e.g. `max_retries=`).

```python
from ezwork.providers import DeepSeek, LongCat
from ezwork.core import Agent

provider = DeepSeek(api_key="sk-...", model="deepseek-v4-flash")
agent = (
    Agent()
    .provider(provider)
    .prompt("You are concise.")
    .thinking(True)
    .reasoning_effort("max")
    .build()
)
```

---

## Testing your provider

Two layers:

1. **Unit test the request building** without hitting the network. Drive
   `_build_request` (Path A) or your translation function (Path B) and assert
   on the produced params. See `tests/test_thinking.py` and
   `tests/test_providers.py` — every vendor field should land in `extra_body`,
   never in SDK-typed kwargs.

2. **Integration test with a MockProvider** at the agent level. Script a
   sequence of `StreamChunk`s and verify the agent loop handles them. See
   `tests/__init__.py::MockProvider`.

Never put real API calls in the test suite.

---

## Troubleshooting

**"AsyncCompletions.create() got an unexpected keyword argument 'X'"**
You're leaking a vendor-specific field into SDK-typed kwargs. All
non-standard fields must go through `extra_body`. If using a preset, ensure
`build_params` returns everything in `body`/`top_params` (both end up in
extra_body). The test `test_kwargs_never_contain_vendor_fields` guards this.

**Reasoning tokens show in stream but not in message history.**
Ensure your preset is attached (`OpenAIProvider(thinking_preset=...)`) and
`thinking=True` is set on the LoopConfig. The provider only emits
`reasoning_delta` chunks when the API actually returns reasoning; check the
vendor's API requires an explicit opt-in (most do, via the thinking body).

**Orphan tool_calls cause 400 errors.**
`OpenAIProvider._sanitize_messages` already strips them. If you're on Path B,
replicate that logic: an assistant message's `tool_calls` must have matching
`role=tool` results later in the history, or the API rejects them.
