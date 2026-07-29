"""AgentLoop — the LLM chat engine, callback-driven.

Inputs (all via constructor):
  - provider        streaming LLM provider (Provider protocol)
  - system_prompt   string system prompt
  - tools           ToolRegistry
  - config          LoopConfig (limits + callbacks + thinking control)

Messages are OpenAI-format dicts (see message.py). The loop appends/reads
them as plain dicts; only the message helpers and providers care about
internal shape.

Per-turn flow (the `_loop` method):

    while True:
        run transform_context callbacks        # compact hook lives here
        emit IterStartEvent
        stream → collect_stream → Response
            (emit StreamChunkEvent per chunk)
        emit ResponseEvent
        if response.is_error: emit ErrorEvent; break
        if response.tool_calls:
            for each: before_tool_call callbacks → emit ToolStartEvent
                      → execute (timeout + truncate) → after_tool_call callbacks
                      → emit ToolCompleteEvent
            append assistant(tool_calls) + tool result messages
            emit IterEndEvent(final_content=None)
            if max_iterations reached: break
        else:
            append assistant message
            emit IterEndEvent(final_content=content)
            break

Cancellation: asyncio.Task.cancel() propagates CancelledError naturally; the
loop appends a short interrupt marker to messages so a resumed conversation
isn't confused. Tool execution errors become role=tool messages with the
error string — the model sees the error and can recover.

Tool call argument parsing: the wire format carries arguments as a JSON
string. The loop parses it once to a dict for tool execution (tools receive
dicts), but stores the original JSON string in messages (wire format).
"""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import base64
from dataclasses import dataclass, field
from pathlib import Path

from .config import LoopConfig
from .event import (
    ErrorEvent,
    IterEndEvent,
    IterStartEvent,
    ResponseEvent,
    StreamChunkEvent,
    ToolCompleteEvent,
    ToolStartEvent,
)
from .message import (
    assistant_text,
    assistant_with_tool_calls,
    tool_result,
    user_text,
    user_with_images,
)
from .provider import Response, Usage, collect_stream, retry_stream
from .tool import ToolError, ToolRegistry

_log = logging.getLogger(__name__)


@dataclass
class LoopResult:
    """Result of a full loop run."""

    content: str = ""
    tool_calls_made: int = 0
    messages: list = field(default_factory=list)
    had_error: bool = False


class AgentLoop:
    """LLM chat engine — receives all dependencies via constructor."""

    INTERRUPT_MSG = "[Response was interrupted by the user before completion.]"
    INTERRUPT_TOOL_MSG = "[Tool execution was interrupted by the user before completion.]"

    def __init__(
        self,
        provider,
        system_prompt: str,
        tools: ToolRegistry,
        config: LoopConfig | None = None,
    ):
        self.provider = provider
        self.system_prompt = system_prompt
        self._tools = tools
        self.config = config or LoopConfig()
        self.messages: list[dict] = []
        self._last_usage: Usage | None = None
        self._total_usage = Usage(0, 0)
        self._iteration_count = 0
        self._tool_call_count = 0

    def reset(self) -> None:
        """Clear conversation state for reuse. Shared deps stay."""
        self.messages = []
        self._last_usage = None
        self._total_usage = Usage(0, 0)
        self._iteration_count = 0
        self._tool_call_count = 0

    # ---- chat ----

    async def chat(self, user_input: str, images: list[str] | None = None) -> str:
        """One conversation turn. Cancel via asyncio.Task.cancel()."""
        self.messages.append(self._build_user_message(user_input, images))
        return await self._loop()

    async def run_with_messages(self, messages: list[dict]) -> LoopResult:
        """Run the loop with a pre-existing message list (shallow-copied).
        Returns a full LoopResult (used by sub-agents, replays, etc.)."""
        self.messages = list(messages)
        try:
            content = await self._loop()
            return LoopResult(
                content=content,
                tool_calls_made=self._tool_call_count,
                messages=self.messages,
            )
        except Exception as e:
            return LoopResult(
                content=str(e),
                tool_calls_made=self._tool_call_count,
                messages=self.messages,
                had_error=True,
            )

    # ---- the loop ----

    async def _loop(self) -> str:
        final_content = ""
        self._iteration_count = 0
        self._tool_call_count = 0
        self._total_usage = Usage(0, 0)

        while True:
            iteration = self._iteration_count

            # 1. transform_context callbacks (compact, message rewrites)
            self._run_transforms()

            # 2. emit IterStart (snapshot pre-call, post-transform)
            self._emit(IterStartEvent(messages=self.messages, iteration=iteration))

            # 3. call provider with retry-wrapped stream
            response = await self._call_provider(iteration)

            # 4. bookkeeping + ResponseEvent
            if response.usage:
                self._last_usage = response.usage
                self._total_usage = Usage(
                    self._total_usage.prompt_tokens + response.usage.prompt_tokens,
                    self._total_usage.completion_tokens + response.usage.completion_tokens,
                )
            if response.content is not None:
                final_content = response.content
            self._emit(ResponseEvent(response=response, usage=response.usage, iteration=iteration))

            # 5. error path
            if response.is_error:
                self._emit(
                    ErrorEvent(
                        error=response.error or "unknown error",
                        finish_reason=response.finish_reason or "error",
                        iteration=iteration,
                    )
                )
                self.messages.append(
                    assistant_text(f"[provider error: {response.error}]")
                )
                break

            # 6. tool calls vs final response
            if response.tool_calls:
                tool_result_msgs = await self._run_tool_calls_parallel(
                    response.tool_calls, iteration
                )
                self._tool_call_count += len(response.tool_calls)
                self.messages.append(
                    _assistant_msg_from_response(response, with_tool_calls=True)
                )
                self.messages.extend(tool_result_msgs)
                self._emit(
                    IterEndEvent(
                        messages=self.messages, final_content=None, iteration=iteration
                    )
                )

                self._iteration_count += 1
                if self.config.max_iterations > 0 and self._iteration_count >= self.config.max_iterations:
                    break
            else:
                self.messages.append(_assistant_msg_from_response(response, with_tool_calls=False))
                self._emit(
                    IterEndEvent(
                        messages=self.messages, final_content=response.content, iteration=iteration
                    )
                )
                break

        return final_content

    # ---- provider call (with stream + retry + chunk emit) ----

    async def _call_provider(self, iteration: int) -> Response:
        cfg = self.config

        async def _consume() -> Response:
            def _factory():
                return self.provider.stream(
                    self.messages,
                    self.system_prompt,
                    self._tools.all_schemas(),
                    cfg.max_tokens,
                    thinking=cfg.thinking,
                    reasoning_effort=cfg.reasoning_effort,
                )

            if cfg.max_retries > 0:
                chunks = retry_stream(_factory, self.provider.is_retriable, max_retries=cfg.max_retries)
            else:
                chunks = _factory()

            async def _tee():
                async for chunk in chunks:
                    self._emit(StreamChunkEvent(chunk=chunk, iteration=iteration))
                    yield chunk

            return await collect_stream(_tee())

        try:
            return await _consume()
        except asyncio.CancelledError:
            self.messages.append(assistant_text(self.INTERRUPT_MSG))
            raise
        except Exception as exc:
            _log.warning("provider stream failed: %s", exc)
            return Response(content=None, finish_reason="error", error=str(exc))

    # ---- transforms & callbacks ----

    def _run_transforms(self) -> None:
        for cb in self.config.transform_context:
            try:
                cb(self.messages)
            except Exception:
                _log.debug("transform_context %s failed", cb, exc_info=True)

    def _emit(self, event) -> None:
        for cb in self.config.emit:
            try:
                cb(event)
            except Exception:
                _log.debug("emit %s failed", cb, exc_info=True)

    # ---- tool execution ----

    async def _run_tool_calls_parallel(
        self, tool_calls: list[dict], iteration: int
    ) -> list[dict]:
        """Returns a list of role=tool result messages (OpenAI format)."""

        async def _run_one(tc: dict) -> dict:
            # tc is the wire-format tool_call dict:
            # {"id","type":"function","function":{"name","arguments": json_str}}
            tc_id = tc.get("id", "")
            fn = tc.get("function", {})
            name = fn.get("name", "")
            args_raw = fn.get("arguments", "")
            try:
                args = json.loads(args_raw) if args_raw else {}
            except Exception:
                args = {"_raw": args_raw}

            # before_tool_call callbacks receive a normalised dict and may
            # rewrite or veto. We pass {"id","name","arguments": dict}.
            normalised = {"id": tc_id, "name": name, "arguments": args}
            current = normalised
            for cb in self.config.before_tool_call:
                try:
                    ret = cb(current)
                except Exception:
                    _log.debug("before_tool_call %s failed", cb, exc_info=True)
                    continue
                if ret is None:
                    current = None
                    break
                current = ret

            if current is None:
                result_msg = tool_result(tc_id, "[vetoed by before_tool_call callback]", is_error=True)
                # emit with the original tool_call shape for observer continuity
                self._emit(
                    ToolCompleteEvent(
                        tool_call=tc,
                        tool_result=result_msg,
                        iteration=iteration,
                    )
                )
                return result_msg

            # rebuild a wire-shape tool_call for events (arguments back to JSON str)
            event_tc = {
                "id": current.get("id", tc_id),
                "type": "function",
                "function": {
                    "name": current.get("name", name),
                    "arguments": json.dumps(current.get("arguments", {}), ensure_ascii=False),
                },
            }
            self._emit(ToolStartEvent(tool_call=event_tc, iteration=iteration))

            content, is_error = await self._execute_one_tool(
                current.get("name", name), current.get("arguments", args)
            )

            result_msg = tool_result(current.get("id", tc_id), content, is_error=is_error)

            # after_tool_call callbacks may rewrite the result message
            for cb in self.config.after_tool_call:
                try:
                    ret = cb(event_tc, result_msg)
                    if ret is not None:
                        result_msg = ret
                except Exception:
                    _log.debug("after_tool_call %s failed", cb, exc_info=True)

            self._emit(
                ToolCompleteEvent(
                    tool_call=event_tc,
                    tool_result=result_msg,
                    iteration=iteration,
                )
            )
            return result_msg

        raw_results = await asyncio.gather(
            *[_run_one(tc) for tc in tool_calls], return_exceptions=True
        )
        out: list[dict] = []
        for i, raw in enumerate(raw_results):
            tc = tool_calls[i]
            if isinstance(raw, BaseException):
                out.append(
                    tool_result(tc.get("id", ""), f"error: {raw}", is_error=True)
                )
            else:
                out.append(raw)
        return out

    async def _execute_one_tool(self, name: str, args: dict) -> tuple[str, bool]:
        """Run a single tool. Returns (content, is_error)."""
        tool = self._tools.get(name)
        if tool is None:
            return f"unknown tool '{name}'", True
        try:
            if tool.is_async:
                raw = await asyncio.wait_for(
                    tool.run_async(args), timeout=self.config.tool_timeout
                )
            else:
                raw = await asyncio.wait_for(
                    asyncio.to_thread(tool.run, args), timeout=self.config.tool_timeout
                )
        except asyncio.TimeoutError:
            return f"timeout: {self.config.tool_timeout}s", True
        except ToolError as e:
            return f"{e.code}: {e.message}", True
        except Exception as e:
            return f"error: {e}", True

        return self._truncate(raw), False

    def _truncate(self, result: str) -> str:
        limit = self.config.tool_result_limit
        if limit > 0 and len(result) > limit:
            return result[:limit] + "\n... [truncated]"
        return result

    # ---- user message helpers ----

    @staticmethod
    def _build_user_message(text: str, images: list[str] | None) -> dict:
        if not images:
            return user_text(text)
        img_parts: list[dict[str, str]] = []
        for path_str in images:
            p = Path(path_str)
            if not p.exists():
                continue
            mime, _ = mimetypes.guess_type(path_str)
            if not mime or not mime.startswith("image/"):
                continue
            try:
                b64 = base64.b64encode(p.read_bytes()).decode()
            except Exception:
                continue
            img_parts.append({"url": f"data:{mime};base64,{b64}"})
        if not img_parts:
            return user_text(text)
        return user_with_images(text, img_parts)

    # ---- introspection ----

    @property
    def tool_registry(self) -> ToolRegistry:
        return self._tools

    @property
    def iteration(self) -> int:
        """Number of LLM iterations completed in the last chat() loop."""
        return self._iteration_count if self._iteration_count else 0

    @property
    def last_usage(self) -> Usage | None:
        return self._last_usage

    @property
    def total_usage(self) -> Usage:
        return self._total_usage


def _assistant_msg_from_response(response: Response, *, with_tool_calls: bool) -> dict:
    """Build the assistant message to append to history from a Response."""
    if not with_tool_calls:
        msg: dict = {"role": "assistant", "content": response.content or ""}
        if response.reasoning_content:
            msg["reasoning_content"] = response.reasoning_content
        return msg
    # tool-call turn — preserve reasoning_content and the wire tool_calls
    msg = {"role": "assistant", "content": response.content or ""}
    if response.reasoning_content:
        msg["reasoning_content"] = response.reasoning_content
    msg["tool_calls"] = response.tool_calls
    return msg


__all__ = ["AgentLoop", "LoopResult"]
