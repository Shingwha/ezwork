"""Ezwork kernel — agent loop + supporting primitives.

Public API surface, all re-exported here so callers do a single import:

    from ezwork.core import (
        Agent, AgentLoop, LoopConfig, LoopResult,
        # messages (OpenAI-format dicts + helpers)
        user_text, assistant_text, assistant_with_tool_calls,
        tool_result, user_with_images, get_text, get_tool_calls,
        # tools
        Tool, ToolRegistry, ToolError,
        # prompts
        Prompt, Section,
        # provider protocol + stream primitives
        Provider, StreamChunk, Response, Usage,
        ThinkingPreset, ThinkingParams,
        collect_stream, retry_stream,
        # events
        Event, IterStartEvent, ResponseEvent, StreamChunkEvent,
        ToolStartEvent, ToolCompleteEvent, IterEndEvent, ErrorEvent,
    )
"""

from __future__ import annotations

# Lazy re-exports — keep top-level `import ezwork.core` cheap. Each name maps
# to (module, attr). Resolved on first attribute access.
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # agent + config + builder
    "Agent": (".builder", "Agent"),
    "AgentLoop": (".agent", "AgentLoop"),
    "LoopConfig": (".config", "LoopConfig"),
    "LoopResult": (".agent", "LoopResult"),
    # message helpers (messages are plain OpenAI dicts)
    "Message": (".message", "Message"),
    "ToolCall": (".message", "ToolCall"),
    "Messages": (".message", "Messages"),
    "user_text": (".message", "user_text"),
    "assistant_text": (".message", "assistant_text"),
    "assistant_with_tool_calls": (".message", "assistant_with_tool_calls"),
    "tool_result": (".message", "tool_result"),
    "user_with_images": (".message", "user_with_images"),
    "get_text": (".message", "get_text"),
    "get_tool_calls": (".message", "get_tool_calls"),
    # tool + prompt
    "Tool": (".tool", "Tool"),
    "ToolRegistry": (".tool", "ToolRegistry"),
    "ToolError": (".tool", "ToolError"),
    "Prompt": (".prompt", "Prompt"),
    "Section": (".prompt", "Section"),
    # provider
    "Provider": (".provider", "Provider"),
    "StreamChunk": (".provider", "StreamChunk"),
    "Response": (".provider", "Response"),
    "Usage": (".provider", "Usage"),
    "collect_stream": (".provider", "collect_stream"),
    "retry_stream": (".provider", "retry_stream"),
    "ThinkingPreset": (".provider", "ThinkingPreset"),
    "ThinkingParams": (".provider", "ThinkingParams"),
    # events
    "Event": (".event", "Event"),
    "IterStartEvent": (".event", "IterStartEvent"),
    "ResponseEvent": (".event", "ResponseEvent"),
    "StreamChunkEvent": (".event", "StreamChunkEvent"),
    "ToolStartEvent": (".event", "ToolStartEvent"),
    "ToolCompleteEvent": (".event", "ToolCompleteEvent"),
    "IterEndEvent": (".event", "IterEndEvent"),
    "ErrorEvent": (".event", "ErrorEvent"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        module_name, attr = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_name, __name__)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'ezwork.core' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY_IMPORTS.keys()))
