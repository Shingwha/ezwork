"""OpenAI-compatible provider.

Implements the Provider protocol via a single `stream()` method. Messages
are already OpenAI-format dicts (see ezwork/message.py), so this provider
passes them through with only light sanitisation (dropping orphan
tool_calls — some endpoints reject them).

Works against any OpenAI-compatible endpoint (OpenAI, DeepSeek, GLM, MiniMax,
Mimo, Step, LongCat, …) via base_url override. The `openai` package is
required (`pip install ezwork[openai]`) and imported lazily so the kernel
itself stays dependency-free.

Thinking: an optional ThinkingPreset can be supplied at construction. When
the caller passes thinking/reasoning_effort to stream(), the preset's
build_params() output is merged into the request body. The provider impl
itself hardcodes no thinking field names.

Error model: SDK exceptions during streaming are caught and converted into a
single StreamChunk(type="error"), then the iterator ends. No exceptions
escape stream() after the first chunk. The kernel never raises on LLM
failures — the loop surfaces them via ErrorEvent and stops gracefully, so
the app layer decides how to present them (it never crashes the process).
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, AsyncIterator

from ezwork.core import StreamChunk, ThinkingPreset, Usage


class OpenAIProvider:
    """OpenAI-compatible API provider — implements the Provider protocol."""

    # Lazy-loaded retriable exception classes
    _exc_classes: tuple[type[Exception], ...] | None = None

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: str | None = None,
        extra_body: dict[str, Any] | None = None,
        thinking_preset: ThinkingPreset | None = None,
        max_retries: int = 2,
    ):
        self._api_key = api_key
        self._base_url = base_url
        self._client: Any = None  # lazy
        self._client_lock = threading.Lock()
        self._model = model
        self._extra_body = dict(extra_body) if extra_body else {}
        self._thinking_preset = thinking_preset
        self._max_retries = max_retries

    def _ensure_client(self):
        if self._client is None:
            # Double-checked locking: warmup() may construct the client from a
            # background thread while stream() runs on the event loop.
            with self._client_lock:
                if self._client is None:
                    from openai import AsyncOpenAI

                    self._client = AsyncOpenAI(
                        api_key=self._api_key,
                        base_url=self._base_url,
                        max_retries=self._max_retries,
                    )
        return self._client

    def warmup(self) -> None:
        """Eagerly import the SDK and build the client in a daemon thread.

        The first stream() call otherwise pays the full SDK import + client
        construction (~2-3s with the openai package) inline. The app layer
        calls this at startup so that cost overlaps with user typing / stdin
        drain. No-op once the client exists; safe to call from any thread.
        """
        if self._client is None:
            threading.Thread(
                target=self._warmup_work, daemon=True, name="ezwork-openai-warmup"
            ).start()

    def _warmup_work(self) -> None:
        self._ensure_client()
        # Preload the retriable exception classes too, so the error path never
        # does a second lazy import.
        self._load_exc_classes()

    @property
    def model(self) -> str:
        return self._model

    @model.setter
    def model(self, value: str) -> None:
        """Runtime model override (e.g. `ezwork --model X`)."""
        self._model = value

    @classmethod
    def _load_exc_classes(cls) -> tuple[type[Exception], ...]:
        if cls._exc_classes is None:
            from openai import (
                APIConnectionError,
                APITimeoutError,
                InternalServerError,
                RateLimitError,
            )

            cls._exc_classes = (
                RateLimitError,
                InternalServerError,
                APIConnectionError,
                APITimeoutError,
            )
        return cls._exc_classes

    def is_retriable(self, exc: Exception) -> bool:
        return isinstance(exc, self._load_exc_classes())

    # ---- request body ----

    def _build_request(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]],
        max_tokens: int,
        stream: bool,
        thinking: bool | None,
        reasoning_effort: str | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return (kwargs, extra_body) for chat.completions.create.

        `kwargs` holds only params the openai SDK knows (model, messages,
        max_tokens, stream, tools, stream_options, ...). Anything vendor-
        specific (thinking, reasoning_split, prompt cache fields, ...) goes
        into `extra_body`, which the SDK passes through to the wire verbatim
        without validation. This is the only robust way to carry non-standard
        fields through the typed SDK.
        """
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                *self._sanitize_messages(messages),
            ],
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if tools:
            kwargs["tools"] = tools
        if stream:
            kwargs["stream_options"] = {"include_usage": True}

        # extra_body starts from constructor-supplied defaults
        extra_body: dict[str, Any] = dict(self._extra_body) if self._extra_body else {}

        # Thinking — delegated to the preset. Zero hardcoded field names here.
        # All preset output goes into extra_body (vendors use non-standard
        # field names; the SDK would reject them as kwargs).
        if thinking is not None and self._thinking_preset is not None:
            tp = self._thinking_preset.build_params(thinking, reasoning_effort)
            if tp.body:
                _deep_merge(extra_body, tp.body)
            if tp.top_params:
                _deep_merge(extra_body, tp.top_params)

        return kwargs, extra_body

    @staticmethod
    def _sanitize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop orphan tool_calls (assistant entries whose ids have no matching
        role=tool result). Some endpoints reject these."""
        has_orphans = any(
            m.get("role") == "assistant" and m.get("tool_calls") for m in messages
        )
        if not has_orphans:
            return messages
        result_ids = {
            m["tool_call_id"]
            for m in messages
            if m.get("role") == "tool" and m.get("tool_call_id")
        }
        if not result_ids:
            return [
                {k: v for k, v in m.items() if k != "tool_calls"}
                if m.get("role") == "assistant" and "tool_calls" in m
                else m
                for m in messages
            ]
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.get("role") == "assistant" and "tool_calls" in m:
                valid = [tc for tc in m["tool_calls"] if tc.get("id") in result_ids]
                if not valid:
                    out.append({k: v for k, v in m.items() if k != "tool_calls"})
                else:
                    out.append({**m, "tool_calls": valid})
            else:
                out.append(m)
        return out

    # ---- stream ----

    async def stream(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]],
        max_tokens: int,
        *,
        thinking: bool | None = None,
        reasoning_effort: str | None = None,
    ) -> AsyncIterator[StreamChunk]:
        kwargs, extra_body = self._build_request(
            messages, system, tools, max_tokens, stream=True,
            thinking=thinking, reasoning_effort=reasoning_effort,
        )
        try:
            # Build the client in a worker thread: even if warmup() hasn't
            # finished yet, the SDK import must not block the event loop.
            client = await asyncio.to_thread(self._ensure_client)
            stream = await client.chat.completions.create(
                **kwargs, extra_body=extra_body or None
            )
            async for chunk in stream:
                for produced in self._map_chunk(chunk):
                    yield produced
        except Exception as exc:
            yield StreamChunk.error(str(exc))
            return

    # ---- chunk mapping ----

    def _map_chunk(self, chunk: Any) -> list[StreamChunk]:
        out: list[StreamChunk] = []

        if getattr(chunk, "usage", None):
            out.append(StreamChunk.usage(_extract_usage(chunk.usage)))

        choices = getattr(chunk, "choices", None) or []
        if not choices:
            return out

        choice = choices[0]
        delta = getattr(choice, "delta", None)

        if delta is not None:
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                out.append(StreamChunk.reasoning_delta(reasoning))

            text = getattr(delta, "content", None)
            if text:
                out.append(StreamChunk.text_delta(text))

            tool_calls = getattr(delta, "tool_calls", None) or []
            for tc in tool_calls:
                idx = getattr(tc, "index", 0) or 0
                tc_id = getattr(tc, "id", None)
                fn = getattr(tc, "function", None)
                name = getattr(fn, "name", None) if fn else None
                args_delta = getattr(fn, "arguments", None) if fn else None
                out.append(
                    StreamChunk.tool_call_delta(
                        idx, id=tc_id, name=name, arguments_delta=args_delta,
                    )
                )

        finish = getattr(choice, "finish_reason", None)
        if finish:
            out.append(StreamChunk.done(finish_reason=finish))

        return out


# ---- helpers ------------------------------------------------------------


def _extract_usage(usage: Any) -> Usage:
    """Build a Usage from an OpenAI usage object, surfacing prompt cache
    fields across vendors (OpenAI prompt_tokens_details.cached_tokens and
    DeepSeek prompt_cache_hit_tokens / prompt_cache_miss_tokens)."""
    prompt = getattr(usage, "prompt_tokens", 0) or 0
    completion = getattr(usage, "completion_tokens", 0) or 0
    hit: int | None = None
    miss: int | None = None

    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", None) if details else None
    if cached:
        hit = cached
        miss = max(prompt - cached, 0)
    else:
        # DeepSeek / compatible extension fields (not in SDK types)
        raw = usage if isinstance(usage, dict) else getattr(usage, "__dict__", {})
        dh = raw.get("prompt_cache_hit_tokens")
        dm = raw.get("prompt_cache_miss_tokens")
        if isinstance(dh, int) or isinstance(dm, int):
            hit = dh if isinstance(dh, int) else 0
            miss = dm if isinstance(dm, int) else 0

    return Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        prompt_cache_hit_tokens=hit,
        prompt_cache_miss_tokens=miss,
    )


def preload() -> None:
    """Import the openai package in the current thread (no client built).

    The CLI starts this in a background thread at process start so the heavy
    SDK import (~2s) overlaps with arg parsing, stdin drain, and the REPL
    prompt. Idempotent; import failures are swallowed here — stream() will
    surface the real error if the package is missing.
    """
    try:
        from openai import AsyncOpenAI  # noqa: F401
    except Exception:
        pass


def _deep_merge(target: dict[str, Any], src: dict[str, Any]) -> None:
    """Recursive dict merge: src into target, nested dicts merged, scalars overwritten."""
    for k, v in src.items():
        if k in target and isinstance(target[k], dict) and isinstance(v, dict):
            _deep_merge(target[k], v)
        else:
            target[k] = v


__all__ = ["OpenAIProvider"]
