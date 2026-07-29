"""LoopConfig — all knobs and extension points for the agent loop.

Two kinds of extension points live here, intentionally split:

  1. Context-transform callbacks (lists, run in registration order):
       transform_context   — mutate messages in place (compact lives here)
       before_tool_call    — observe/rewrite/veto a tool call
       after_tool_call     — observe/rewrite a tool result

  2. Output-broadcast callbacks (a single list, fan-out):
       emit                — observe events (UI, logging, recording)

We deliberately avoid an EventBus class. Fan-out is `for cb in emit: cb(event)`.
Subscribers that care about a specific event type filter inside their callback
(or use Agent.on(type, cb) which wraps the filter for you).

All callbacks are SYNC. The agent loop is async but invokes these synchronously
between awaits — this keeps ordering deterministic and avoids asyncio task
scheduling overhead for things that should be cheap.

Thinking control: `thinking` (on/off) and `reasoning_effort` are passed through
to provider.stream(). Whether they have any effect depends on the provider's
ThinkingPreset. Mutate these fields between turns to switch modes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

# Context-transform callback signatures.
# Messages are OpenAI-format dicts (see message.py).
TransformContext = Callable[[list], None]
# tool_call is an OpenAI-format dict. Returns a (possibly modified) tool_call
# dict to continue, or None to veto.
BeforeToolCall = Callable[[dict], Any]
# tool_result is an OpenAI-format role=tool dict. Returns a (possibly modified)
# tool_result dict.
AfterToolCall = Callable[[dict, dict], Any]
# Output callback. Sync; receives an Event.
Emit = Callable[[Any], None]


@dataclass
class LoopConfig:
    """All configuration for AgentLoop. Defaults are sensible for a generic
    chat agent; override fields or append to callback lists as needed."""

    # ---- limits ----
    max_tokens: int = 32768
    tool_result_limit: int = 25000
    tool_timeout: int = 240
    max_iterations: int = 0  # 0 = unlimited

    # ---- thinking control (passed through to provider.stream) ----
    # None = provider default (from its ThinkingPreset.default_enabled).
    thinking: bool | None = None
    # None = provider default effort. Only meaningful when thinking is on.
    reasoning_effort: str | None = None

    # ---- context transforms (run in order) ----
    transform_context: list[TransformContext] = field(default_factory=list)
    before_tool_call: list[BeforeToolCall] = field(default_factory=list)
    after_tool_call: list[AfterToolCall] = field(default_factory=list)

    # ---- output broadcast (fan-out) ----
    emit: list[Emit] = field(default_factory=list)

    # ---- retry policy (applied by AgentLoop around provider.stream) ----
    # If max_retries > 0, the loop wraps provider.stream with retry_stream.
    # Set to 0 to disable retry entirely (let errors surface immediately).
    max_retries: int = 6


__all__ = [
    "LoopConfig",
    "TransformContext",
    "BeforeToolCall",
    "AfterToolCall",
    "Emit",
]
