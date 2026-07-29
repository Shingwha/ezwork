"""Agent — fluent builder for AgentLoop.

Usage:

    from ezwork import Agent, LoopConfig, OpenAIProvider

    cfg = LoopConfig()
    agent = (Agent()
             .provider(OpenAIProvider(api_key=...))
             .prompt("You are helpful.")
             .tools([my_tool])
             .config(cfg)
             .on("tool_complete", lambda e: print("tool done"))
             .build())

    await agent.chat("hello")

`on(type, cb)` is a convenience: it appends a wrapped callback to
config.emit that fires only when event.type == type. Useful for short
event subscriptions without writing the filter yourself.
"""

from __future__ import annotations

from typing import Any, Callable, Self

from .agent import AgentLoop
from .config import LoopConfig
from .prompt import Prompt, Section
from .tool import Tool, ToolRegistry


class Agent:
    """Fluent builder for AgentLoop."""

    def __init__(self) -> None:
        self._provider: Any = None
        self._system_prompt: str | Prompt | list[Section] | None = None
        self._tools: ToolRegistry | list[Tool] | None = None
        self._config: LoopConfig | None = None
        self._prompt_context: dict[str, Any] = {}

    def provider(self, provider: Any) -> Self:
        self._provider = provider
        return self

    def prompt(self, prompt: str | Prompt | list[Section]) -> Self:
        self._system_prompt = prompt
        return self

    def tools(self, tools: ToolRegistry | list[Tool]) -> Self:
        self._tools = tools
        return self

    def config(self, config: LoopConfig) -> Self:
        self._config = config
        return self

    def prompt_context(self, **kwargs: Any) -> Self:
        self._prompt_context.update(kwargs)
        return self

    def thinking(self, enabled: bool | None) -> Self:
        """Toggle extended thinking for the next call. None = provider default."""
        self._ensure_config().thinking = enabled
        return self

    def reasoning_effort(self, effort: str | None) -> Self:
        """Set reasoning effort (e.g. 'low'/'medium'/'high'). None = provider default."""
        self._ensure_config().reasoning_effort = effort
        return self

    def on(self, event_type: str, cb: Callable[[Any], None]) -> Self:
        """Append a typed event listener to config.emit.

        The listener fires only for events whose `type` attribute matches
        `event_type`. A fresh LoopConfig is created if one wasn't set.
        """
        cfg = self._ensure_config()

        def _filtered(event: Any) -> None:
            if getattr(event, "type", None) == event_type:
                cb(event)

        cfg.emit.append(_filtered)
        return self

    def emit(self, cb: Callable[[Any], None]) -> Self:
        """Append a raw (unfiltered) emit callback."""
        self._ensure_config().emit.append(cb)
        return self

    def transform(self, cb: Callable[[list], None]) -> Self:
        """Append a transform_context callback (compact hook lives here)."""
        self._ensure_config().transform_context.append(cb)
        return self

    def before_tool(self, cb: Callable[[Any], Any]) -> Self:
        self._ensure_config().before_tool_call.append(cb)
        return self

    def after_tool(self, cb: Callable[[Any, Any], Any]) -> Self:
        self._ensure_config().after_tool_call.append(cb)
        return self

    def _ensure_config(self) -> LoopConfig:
        if self._config is None:
            self._config = LoopConfig()
        return self._config

    def build(self) -> AgentLoop:
        if self._provider is None:
            raise ValueError("Provider is required. Call .provider() first.")
        if self._system_prompt is None:
            raise ValueError("System prompt is required. Call .prompt() first.")

        # tools
        if isinstance(self._tools, list):
            registry = ToolRegistry()
            for tool in self._tools:
                registry.register(tool)
        elif isinstance(self._tools, ToolRegistry):
            registry = self._tools
        else:
            registry = ToolRegistry()

        # prompt — rendered as plain text only; apps embed any markup they
        # want directly in their section content strings.
        if isinstance(self._system_prompt, list):
            pb = Prompt()
            for s in self._system_prompt:
                pb.register(s)
            prompt_str = pb.context(**self._prompt_context).build()
        elif isinstance(self._system_prompt, Prompt):
            prompt_str = self._system_prompt.context(**self._prompt_context).build()
        else:
            prompt_str = self._system_prompt

        return AgentLoop(
            provider=self._provider,
            system_prompt=prompt_str,
            tools=registry,
            config=self._config or LoopConfig(),
        )


__all__ = ["Agent"]
