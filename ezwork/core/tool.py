"""Tool + ToolRegistry + ToolError.

A Tool is a named function with a simplified params spec (dict, not full
JSON Schema). The registry caches the generated OpenAI function-calling
schema — that schema is the neutral wire format providers accept.

Tools support both sync and async implementations. Sync tools are wrapped by
the agent loop with asyncio.to_thread so they don't block the loop.

Note: the kernel ships NO built-in tools. See examples/builtin_tools for
reference implementations (read, bash, ...).
"""

from __future__ import annotations

import inspect
from typing import Callable


class ToolError(Exception):
    """Tool execution error — caught by the agent loop and turned into a
    ToolResult with is_error=True."""

    def __init__(self, message: str, code: str = "execution_error"):
        self.message = message
        self.code = code
        super().__init__(message)


class Tool:
    """A tool — sync or async.

    run() / run_async() propagate ToolError and Exception upward. The agent
    loop decides how to render errors into the conversation.
    """

    def __init__(
        self,
        name: str,
        description: str,
        params: dict[str, dict],
        func: Callable[[dict], str],
    ):
        self.name = name
        self.description = description
        self._required: list[str] = []
        normalized: dict[str, dict] = {}
        for key, spec in params.items():
            if not isinstance(spec, dict):
                raise TypeError(
                    f"Tool '{name}': param '{key}' must be a dict with 'type' and "
                    f"'description', got {type(spec).__name__}: {spec!r}"
                )
            normalized[key] = spec
            if "default" not in spec and not spec.get("optional"):
                self._required.append(key)
        self.params = normalized
        self.func = func
        self.is_async = inspect.iscoroutinefunction(func)

    def run(self, args: dict) -> str:
        validated = self._validate_args(args)
        return self.func(validated)

    async def run_async(self, args: dict) -> str:
        validated = self._validate_args(args)
        if self.is_async:
            return await self.func(validated)
        return self.func(validated)

    def _validate_args(self, args: dict) -> dict:
        result = dict(args)
        for name, spec in self.params.items():
            if name not in result:
                if "default" in spec:
                    result[name] = spec["default"]
                elif name in self._required:
                    raise ToolError(f"Missing required parameter '{name}'", "missing_param")
        return result

    def to_schema(self) -> dict:
        """OpenAI function-calling schema. This is the neutral tool description
        format the kernel passes to providers."""
        properties: dict[str, dict] = {}
        for name, spec in self.params.items():
            prop: dict = {"type": spec.get("type", "string")}
            if "description" in spec:
                prop["description"] = spec["description"]
            if "enum" in spec:
                prop["enum"] = spec["enum"]
            if "default" in spec:
                prop["default"] = spec["default"]
            properties[name] = prop
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": list(self._required),
                },
            },
        }


class ToolRegistry:
    """Instance-scoped tool registry — storage and schema generation."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._schema_cache: list[dict] | None = None

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        self._schema_cache = None

    def unregister(self, name: str) -> Tool | None:
        result = self._tools.pop(name, None)
        if result is not None:
            self._schema_cache = None
        return result

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def all_schemas(self) -> list[dict]:
        if self._schema_cache is None:
            self._schema_cache = [t.to_schema() for t in self._tools.values()]
        return self._schema_cache

    def derived(self, exclude: set[str] | None = None) -> ToolRegistry:
        """Return a new registry sharing the same Tool instances, optionally
        excluding some names. Used to build filtered tool sets (e.g. for a
        user-implemented sub-agent)."""
        new = ToolRegistry()
        if exclude:
            new._tools = {k: v for k, v in self._tools.items() if k not in exclude}
        else:
            new._tools = dict(self._tools)
        return new

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools


__all__ = ["Tool", "ToolRegistry", "ToolError"]
