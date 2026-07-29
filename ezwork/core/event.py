"""Event types emitted by the agent loop.

There is intentionally NO EventBus class. Subscriptions live as plain callables
appended to LoopConfig.emit. Fan-out is just a list iteration.

Messages in events are OpenAI-format dicts (see message.py). The agent loop
treats messages as opaque dicts; only helpers in message.py and providers
care about their internal shape.

Events split into two groups (mirroring the LoopConfig callback split):
  - observation events (ResponseEvent, StreamChunkEvent, ToolStartEvent,
    ToolCompleteEvent, IterEndEvent, ErrorEvent) — read-only side-effects
  - IterStartEvent — its `messages` field is a mutable reference; that's where
    transform_context-style hooks (e.g. a compact reimplementation) operate

Every event carries a `type` string so emit callbacks can dispatch with a
simple equality / prefix check, no isinstance required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Event:
    """Base event. Subclasses set their own type string."""

    type: str = "event"


@dataclass
class IterStartEvent(Event):
    """Emitted at the top of each loop iteration, before the LLM call.

    NOTE: transform_context callbacks mutate self.messages in place between
    IterStartEvent and the actual LLM call. So `messages` here is the
    pre-transform snapshot; observers that need the post-transform view
    should subscribe to ResponseEvent instead.
    """

    messages: list = field(default_factory=list)  # list[dict] — OpenAI format
    iteration: int = 0
    type: str = "iter_start"


@dataclass
class ResponseEvent(Event):
    """Emitted once per iteration after the full Response is collected."""

    response: Any = None  # Response
    usage: Any = None  # Usage | None
    iteration: int = 0
    type: str = "response"


@dataclass
class StreamChunkEvent(Event):
    """Emitted for every StreamChunk as it arrives. Use for live token
    streaming UIs. Observers should treat the chunk as read-only."""

    chunk: Any = None  # StreamChunk
    iteration: int = 0
    type: str = "stream_chunk"


@dataclass
class ToolStartEvent(Event):
    """Emitted before a single tool call executes (after before_tool_call
    callbacks; if any callback returned None to veto, this event is NOT
    emitted for the vetoed call).

    `tool_call` is an OpenAI-format tool_call dict (with parsed arguments
    under .function.arguments — see agent.py for how it's built).
    """

    tool_call: Any = None
    iteration: int = 0
    type: str = "tool_start"


@dataclass
class ToolCompleteEvent(Event):
    """Emitted after a single tool call completes (after after_tool_call
    callbacks have potentially transformed the result).

    `tool_result` is an OpenAI-format role=tool message dict:
    {"role":"tool","tool_call_id":...,"content":...}
    """

    tool_call: Any = None
    tool_result: Any = None
    iteration: int = 0
    type: str = "tool_complete"


@dataclass
class IterEndEvent(Event):
    """Emitted at the end of each iteration. `final_content` is non-None on
    the terminal (no-tool-call) iteration, None on tool-call iterations."""

    messages: list = field(default_factory=list)
    final_content: Any = None  # str | None
    iteration: int = 0
    type: str = "iter_end"


@dataclass
class ErrorEvent(Event):
    """Emitted when the provider signals an error (finish_reason='error')."""

    error: str = ""
    finish_reason: str = "error"
    iteration: int = 0
    type: str = "error"


__all__ = [
    "Event",
    "IterStartEvent",
    "ResponseEvent",
    "StreamChunkEvent",
    "ToolStartEvent",
    "ToolCompleteEvent",
    "IterEndEvent",
    "ErrorEvent",
]
