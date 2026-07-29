"""Provider protocol + stream chunk DTO + accumulation helpers.

Every provider implements a single method: `stream()`, returning an async
iterator of StreamChunk. Non-streaming is just "collect the stream".

Messages are OpenAI-format dicts (see message.py). Providers that are
themselves OpenAI-compatible pass them through (with light sanitisation);
providers targeting a different native API convert at their boundary.

Thinking: providers carry an optional ThinkingPreset that knows how to render
`(enabled, effort)` into provider-specific API params. The agent loop passes
`thinking` / `reasoning_effort` through; the provider merges the preset's
output. This isolates every vendor's thinking vocabulary in one place.

Error model: providers do NOT raise on transient LLM failures. They emit a
StreamChunk with type="error" and finish the iterator. The accumulated
Response carries finish_reason="error". (Cancellation still propagates as
CancelledError — that's the caller's intent, not an LLM failure.)

Retry is offered as a standalone helper `retry_stream` but the agent loop
itself does no retrying — that's a caller-side policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Protocol, runtime_checkable


# ---- DTOs ----------------------------------------------------------------


@dataclass
class Usage:
    """Token accounting. Cache fields are optional (vendor-specific)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_cache_hit_tokens: int | None = None
    prompt_cache_miss_tokens: int | None = None


@dataclass
class StreamChunk:
    """One increment from a provider stream.

    types:
      - text_delta            incremental assistant text (text)
      - reasoning_delta       incremental reasoning text (text)
      - tool_call_delta       incremental tool call (tool_call_index + optional
                              id/name on first delta + arguments_delta JSON fragment)
      - usage                 usage reported mid/end of stream (usage)
      - done                  stream complete; carry finish_reason
      - error                 stream failed; carry error message; iterator ends
    """

    type: str
    text: str | None = None
    tool_call_index: int | None = None
    tool_call_id: str | None = None
    tool_call_name: str | None = None
    tool_call_arguments_delta: str | None = None
    usage: Usage | None = None
    finish_reason: str | None = None
    error: str | None = None

    @staticmethod
    def text_delta(text: str) -> StreamChunk:
        return StreamChunk(type="text_delta", text=text)

    @staticmethod
    def reasoning_delta(text: str) -> StreamChunk:
        return StreamChunk(type="reasoning_delta", text=text)

    @staticmethod
    def tool_call_delta(
        index: int,
        *,
        id: str | None = None,
        name: str | None = None,
        arguments_delta: str | None = None,
    ) -> StreamChunk:
        return StreamChunk(
            type="tool_call_delta",
            tool_call_index=index,
            tool_call_id=id,
            tool_call_name=name,
            tool_call_arguments_delta=arguments_delta,
        )

    @staticmethod
    def usage(usage: Usage) -> StreamChunk:
        return StreamChunk(type="usage", usage=usage)

    @staticmethod
    def done(finish_reason: str = "stop") -> StreamChunk:
        return StreamChunk(type="done", finish_reason=finish_reason)

    @staticmethod
    def error(message: str) -> StreamChunk:
        return StreamChunk(type="error", error=message)


@dataclass
class Response:
    """Accumulated full response. The only shape agent loop consumes.

    `content` is the assembled assistant text. `tool_calls` (if any) are
    OpenAI-format dicts: [{"id","type":"function","function":{"name","arguments": json_str}}]
    — arguments is a JSON *string*, matching the wire format.
    """

    content: str | None = None
    tool_calls: list[dict] | None = None
    usage: Usage | None = None
    finish_reason: str | None = None
    reasoning_content: str | None = None
    error: str | None = None

    @property
    def is_error(self) -> bool:
        return self.finish_reason == "error" or self.error is not None


# ---- thinking preset protocol -------------------------------------------


@dataclass
class ThinkingParams:
    """Output of a ThinkingPreset.build_params().

    `body` is deep-merged into the request's extra_body region; `top_params`
    goes to the top level of the API request (e.g. reasoning_effort)."""

    body: dict[str, Any] = field(default_factory=dict)
    top_params: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ThinkingPreset(Protocol):
    """How a specific provider renders thinking on/off + effort into API params.

    Each vendor has its own vocabulary (thinking.type=enabled/disabled,
    reasoning_effort=low/medium/high, reasoning_split, etc.). Implementations
    encapsulate that; the provider impl itself hardcodes no field names.

    The kernel defines ONLY this protocol. Concrete presets live in provider
    packages or app config (the kernel ships no PRESETS registry).
    """

    default_enabled: bool
    effort_levels: list[str]
    default_effort: str

    def build_params(self, enabled: bool, effort: str | None = None) -> ThinkingParams: ...


# ---- provider protocol --------------------------------------------------


@runtime_checkable
class Provider(Protocol):
    """LLM provider protocol. Implementations live in providers/."""

    @property
    def model(self) -> str: ...

    def is_retriable(self, exc: Exception) -> bool: ...

    def stream(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]],
        max_tokens: int,
        *,
        thinking: bool | None = None,
        reasoning_effort: str | None = None,
    ) -> AsyncIterator[StreamChunk]: ...


# ---- stream accumulation -------------------------------------------------


async def collect_stream(chunks: AsyncIterator[StreamChunk]) -> Response:
    """Consume a StreamChunk stream into a single Response.

    - text deltas concatenated into content
    - reasoning deltas concatenated into reasoning_content
    - tool-call deltas accumulated per index; arguments assembled as a JSON
      string (matches the OpenAI wire format — NOT parsed to a dict)
    - usage: last-one-wins
    - finish_reason captured from done/error chunk
    """
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tc_index_order: list[int] = []
    tc_ids: dict[int, str] = {}
    tc_names: dict[int, str] = {}
    tc_args: dict[int, list[str]] = {}
    usage: Usage | None = None
    finish_reason: str | None = None
    error: str | None = None

    async for chunk in chunks:
        t = chunk.type
        if t == "text_delta" and chunk.text:
            text_parts.append(chunk.text)
        elif t == "reasoning_delta" and chunk.text:
            reasoning_parts.append(chunk.text)
        elif t == "tool_call_delta":
            idx = chunk.tool_call_index if chunk.tool_call_index is not None else 0
            if idx not in tc_args:
                tc_index_order.append(idx)
                tc_args[idx] = []
            if chunk.tool_call_id is not None:
                tc_ids[idx] = chunk.tool_call_id
            if chunk.tool_call_name is not None:
                tc_names[idx] = chunk.tool_call_name
            if chunk.tool_call_arguments_delta:
                tc_args[idx].append(chunk.tool_call_arguments_delta)
        elif t == "usage" and chunk.usage:
            usage = chunk.usage
        elif t == "done":
            finish_reason = chunk.finish_reason or "stop"
        elif t == "error":
            error = chunk.error or "unknown error"
            finish_reason = "error"

    tool_calls: list[dict] | None = None
    if tc_index_order:
        tool_calls = []
        for idx in tc_index_order:
            args_str = "".join(tc_args[idx])
            tool_calls.append(
                {
                    "id": tc_ids.get(idx, f"call_{idx}"),
                    "type": "function",
                    "function": {"name": tc_names.get(idx, ""), "arguments": args_str},
                }
            )

    return Response(
        content="".join(text_parts) if text_parts else None,
        tool_calls=tool_calls,
        usage=usage,
        finish_reason=finish_reason,
        reasoning_content="".join(reasoning_parts) if reasoning_parts else None,
        error=error,
    )


# ---- optional retry helper ----------------------------------------------


_MAX_RETRIES = 6  # 7 total attempts
_BASE_DELAY = 1.0
_MAX_DELAY = 60.0
_JITTER_MAX = 0.5


async def retry_stream(
    factory: Callable[[], AsyncIterator[StreamChunk]],
    is_retriable: Callable[[Exception], bool],
    *,
    max_retries: int = _MAX_RETRIES,
) -> AsyncIterator[StreamChunk]:
    """Retry a stream-producing factory on retriable errors raised *before*
    the first chunk is emitted. Once a chunk has been emitted, the stream is
    considered live and any subsequent error propagates to the caller.

    Usage:
        chunks = retry_stream(lambda: provider.stream(...), provider.is_retriable)
        response = await collect_stream(chunks)
    """
    import asyncio
    import random

    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        iterator: AsyncIterator[StreamChunk] | None = None
        try:
            iterator = factory()
            first = await iterator.__anext__()
        except StopAsyncIteration:
            return
        except Exception as exc:
            if not is_retriable(exc) or attempt >= max_retries:
                raise
            last_exc = exc
            delay = min(_BASE_DELAY * (2**attempt) + random.uniform(0, _JITTER_MAX), _MAX_DELAY)
            await asyncio.sleep(delay)
            continue
        assert iterator is not None
        yield first
        async for chunk in iterator:
            yield chunk
        return
    assert last_exc is not None
    raise last_exc


__all__ = [
    "Usage",
    "StreamChunk",
    "Response",
    "ThinkingParams",
    "ThinkingPreset",
    "Provider",
    "collect_stream",
    "retry_stream",
]
