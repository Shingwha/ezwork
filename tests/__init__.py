"""Test helpers shared across test files."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Sequence

from ezwork.core.provider import StreamChunk, Usage


class MockProvider:
    """Provider that replays a canned script of responses.

    `script` items:
      - str                       plain text response (finish_reason='stop')
      - list[tool_call_dict]      tool-call response (finish_reason='tool_calls')
                                   each dict: {"id","name","arguments": dict|json_str}
      - {"content","tool_calls","usage","finish_reason","reasoning_content","error"}
                                  full response (use for edge cases)
      - ("error", "msg")          error response

    Each call to stream() pops the next scripted response. Captures the call
    args (including thinking/effort) for assertions.
    """

    def __init__(self, script: Sequence[Any], *, model: str = "mock-model"):
        self._script = list(script)
        self._model = model
        self.call_count = 0
        self.calls: list[dict[str, Any]] = []  # full arg capture per call

    @property
    def model(self) -> str:
        return self._model

    def is_retriable(self, exc: Exception) -> bool:
        return False

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
        self.call_count += 1
        self.calls.append(
            {
                "messages": list(messages),
                "system": system,
                "tools": list(tools) if tools else [],
                "max_tokens": max_tokens,
                "thinking": thinking,
                "reasoning_effort": reasoning_effort,
            }
        )

        if not self._script:
            raise AssertionError("MockProvider script exhausted")
        item = self._script.pop(0)

        if isinstance(item, str):
            yield StreamChunk.text_delta(item)
            yield StreamChunk.done("stop")
        elif isinstance(item, list):
            for i, tc in enumerate(item):
                args = tc.get("arguments", "")
                if isinstance(args, dict):
                    args = json.dumps(args)
                yield StreamChunk.tool_call_delta(
                    i, id=tc.get("id", f"c{i}"), name=tc.get("name", ""), arguments_delta=args
                )
            yield StreamChunk.done("tool_calls")
        elif isinstance(item, tuple) and item and item[0] == "error":
            yield StreamChunk.error(item[1])
        elif isinstance(item, dict):
            # full response dict
            for c in item.get("chunks", []):
                yield c
            if item.get("content"):
                yield StreamChunk.text_delta(item["content"])
            if item.get("reasoning_content"):
                yield StreamChunk.reasoning_delta(item["reasoning_content"])
            if item.get("usage"):
                yield StreamChunk.usage(item["usage"])
            yield StreamChunk.done(item.get("finish_reason", "stop"))
        else:
            yield StreamChunk.done("stop")


def usage(p: int = 10, c: int = 5) -> Usage:
    return Usage(prompt_tokens=p, completion_tokens=c)


__all__ = ["MockProvider", "usage"]
