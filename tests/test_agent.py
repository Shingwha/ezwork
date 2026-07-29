"""Agent loop tests — exercises the callback-driven architecture end to end
using MockProvider. Messages are OpenAI-format dicts."""

import asyncio

import pytest

from ezwork.core.agent import AgentLoop
from ezwork.core.config import LoopConfig
from ezwork.core.event import (
    ErrorEvent,
    IterEndEvent,
    IterStartEvent,
    ResponseEvent,
    ToolCompleteEvent,
    ToolStartEvent,
)
from ezwork.core.message import get_text, get_tool_calls
from ezwork.core.tool import Tool, ToolError, ToolRegistry

from . import MockProvider, usage


def _reg(*tools: Tool) -> ToolRegistry:
    r = ToolRegistry()
    for t in tools:
        r.register(t)
    return r


def _add_tool() -> Tool:
    return Tool(
        name="add",
        description="add two numbers",
        params={
            "a": {"type": "number", "description": "x"},
            "b": {"type": "number", "description": "y"},
        },
        func=lambda args: str(args["a"] + args["b"]),
    )


def _record_events(cfg: LoopConfig) -> list:
    seen: list = []
    cfg.emit.append(lambda e: seen.append(e))
    return seen


# ---- simple text turn ----


async def test_chat_single_text_response():
    provider = MockProvider(["hello"])
    agent = AgentLoop(provider, "sys", _reg(_add_tool()), LoopConfig(max_retries=0))
    out = await agent.chat("hi")
    assert out == "hello"
    # one user + one assistant message
    assert len(agent.messages) == 2
    assert agent.messages[0]["role"] == "user"
    assert agent.messages[1]["role"] == "assistant"
    assert agent.messages[1]["content"] == "hello"


async def test_chat_emits_lifecycle_events():
    cfg = LoopConfig(max_retries=0)
    seen = _record_events(cfg)
    provider = MockProvider(["hi"])
    agent = AgentLoop(provider, "sys", _reg(), cfg)
    await agent.chat("hello")

    types = [e.type for e in seen]
    # iter_start → stream_chunk(text) → stream_chunk(done) → response → iter_end
    assert types[0] == "iter_start"
    assert types[-1] == "iter_end"
    assert "response" in types
    resp_idx = types.index("response")
    assert types[1:resp_idx] == ["stream_chunk", "stream_chunk"]
    resp_event = next(e for e in seen if isinstance(e, ResponseEvent))
    assert resp_event.response.content == "hi"
    end_event = next(e for e in seen if isinstance(e, IterEndEvent))
    assert end_event.final_content == "hi"


# ---- thinking passthrough ----


async def test_thinking_and_effort_passed_to_provider():
    provider = MockProvider(["ok"])
    cfg = LoopConfig(max_retries=0, thinking=True, reasoning_effort="high")
    agent = AgentLoop(provider, "sys", _reg(), cfg)
    await agent.chat("think hard")
    assert provider.calls[0]["thinking"] is True
    assert provider.calls[0]["reasoning_effort"] == "high"


async def test_thinking_defaults_none():
    provider = MockProvider(["ok"])
    agent = AgentLoop(provider, "sys", _reg(), LoopConfig(max_retries=0))
    await agent.chat("hi")
    assert provider.calls[0]["thinking"] is None
    assert provider.calls[0]["reasoning_effort"] is None


# ---- tool calling ----


async def test_chat_executes_tool_then_responds():
    provider = MockProvider(
        [
            [{"id": "c1", "name": "add", "arguments": {"a": 1, "b": 2}}],
            "result is 3",
        ]
    )
    cfg = LoopConfig(max_retries=0)
    seen = _record_events(cfg)
    agent = AgentLoop(provider, "sys", _reg(_add_tool()), cfg)

    out = await agent.chat("add 1 and 2")
    assert out == "result is 3"

    types = [e.type for e in seen]
    assert "tool_start" in types and "tool_complete" in types

    # messages: user, assistant(tool_calls), tool(result), assistant(text)
    assert len(agent.messages) == 4
    asst = agent.messages[1]
    assert asst["role"] == "assistant"
    assert asst["tool_calls"][0]["function"]["name"] == "add"
    # arguments preserved as a JSON string on the wire
    assert asst["tool_calls"][0]["function"]["arguments"] == '{"a": 1, "b": 2}'

    tool_msg = agent.messages[2]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "c1"
    assert tool_msg["content"] == "3"

    assert provider.call_count == 2


async def test_tool_error_becomes_error_result():
    def _boom(args):
        raise ToolError("kaboom", "boom_code")

    failing = Tool("boom", "x", {}, _boom)
    provider = MockProvider(
        [
            [{"id": "c1", "name": "boom", "arguments": {}}],
            "recovered",
        ]
    )
    agent = AgentLoop(provider, "sys", _reg(failing), LoopConfig(max_retries=0))
    out = await agent.chat("go")
    assert out == "recovered"
    tool_msg = agent.messages[2]
    assert tool_msg["role"] == "tool"
    assert "[error]" in tool_msg["content"]
    assert "boom_code" in tool_msg["content"]


async def test_unknown_tool_yields_error_result():
    provider = MockProvider(
        [
            [{"id": "c1", "name": "ghost", "arguments": {}}],
            "ok",
        ]
    )
    agent = AgentLoop(provider, "sys", _reg(), LoopConfig(max_retries=0))
    await agent.chat("call ghost")
    tool_msg = agent.messages[2]
    assert "[error]" in tool_msg["content"]
    assert "unknown tool" in tool_msg["content"]


# ---- callbacks: transform_context / before / after ----


async def test_transform_context_can_rewrite_messages():
    """compact-style hook lives here: replace messages with a summary."""
    provider = MockProvider(["final"])
    cfg = LoopConfig(max_retries=0)

    def compactor(messages: list) -> None:
        # collapse everything to the first user message
        if len(messages) > 1:
            del messages[1:]

    cfg.transform_context.append(compactor)
    agent = AgentLoop(provider, "sys", _reg(), cfg)
    # seed some history so the compactor has something to do
    from ezwork.core.message import user_text, assistant_text

    agent.messages.append(user_text("old turn 1"))
    agent.messages.append(assistant_text("old reply 1"))

    await agent.chat("new question")

    # messages start as [old1, old2, new question] = 3 entries; compactor
    # keeps messages[0] and drops the rest → provider sees 1 message.
    assert provider.calls[0]["messages"] is not None
    assert len(provider.calls[0]["messages"]) == 1


async def test_before_tool_call_can_veto():
    provider = MockProvider(
        [
            [{"id": "c1", "name": "add", "arguments": {"a": 1, "b": 2}}],
            "ok",
        ]
    )
    cfg = LoopConfig(max_retries=0)
    cfg.before_tool_call.append(lambda tc: None)  # veto everything
    agent = AgentLoop(provider, "sys", _reg(_add_tool()), cfg)
    await agent.chat("go")
    tool_msg = agent.messages[2]
    assert "[error]" in tool_msg["content"]
    assert "vetoed" in tool_msg["content"]


async def test_before_tool_call_can_rewrite_args():
    provider = MockProvider(
        [
            [{"id": "c1", "name": "add", "arguments": {"a": 1, "b": 2}}],
            "ok",
        ]
    )
    cfg = LoopConfig(max_retries=0)

    def double_a(tc):
        return {"id": tc["id"], "name": tc["name"], "arguments": {**tc["arguments"], "a": tc["arguments"]["a"] * 10}}

    cfg.before_tool_call.append(double_a)
    agent = AgentLoop(provider, "sys", _reg(_add_tool()), cfg)
    await agent.chat("go")
    # 10 + 2 = 12
    assert agent.messages[2]["content"] == "12"


async def test_after_tool_call_can_rewrite_result():
    provider = MockProvider(
        [
            [{"id": "c1", "name": "add", "arguments": {"a": 1, "b": 2}}],
            "ok",
        ]
    )
    cfg = LoopConfig(max_retries=0)
    from ezwork.core.message import tool_result

    cfg.after_tool_call.append(
        lambda tc, res: tool_result(res["tool_call_id"], "intercepted")
    )
    agent = AgentLoop(provider, "sys", _reg(_add_tool()), cfg)
    await agent.chat("go")
    assert agent.messages[2]["content"] == "intercepted"


# ---- error path ----


async def test_provider_error_emits_error_event_and_breaks():
    provider = MockProvider([("error", "rate limited")])
    cfg = LoopConfig(max_retries=0)
    seen = _record_events(cfg)
    agent = AgentLoop(provider, "sys", _reg(), cfg)
    out = await agent.chat("hi")

    assert out == ""
    err_events = [e for e in seen if isinstance(e, ErrorEvent)]
    assert len(err_events) == 1
    assert "rate limited" in err_events[0].error
    # error appended as assistant message so follow-up turns have context
    assert any(
        m.get("role") == "assistant" and "provider error" in (m.get("content") or "")
        for m in agent.messages
    )


# ---- limits ----


async def test_max_iterations_caps_loop():
    provider = MockProvider(
        [[{"id": f"c{i}", "name": "add", "arguments": {"a": 1, "b": 1}}] for i in range(100)]
    )
    cfg = LoopConfig(max_retries=0, max_iterations=2)
    agent = AgentLoop(provider, "sys", _reg(_add_tool()), cfg)
    await agent.chat("loop")
    assert agent.iteration == 2


# ---- reasoning content preserved ----


async def test_reasoning_content_preserved_on_final_turn():
    provider = MockProvider(
        [
            {"content": "answer", "reasoning_content": "step by step", "finish_reason": "stop"}
        ]
    )
    agent = AgentLoop(provider, "sys", _reg(), LoopConfig(max_retries=0))
    await agent.chat("q")
    asst = agent.messages[1]
    assert asst["reasoning_content"] == "step by step"
