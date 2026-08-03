"""Message model — OpenAI Chat Completions format.

Messages are plain dicts in the OpenAI wire format. The kernel does NOT
introduce a competing "neutral" type: every mainstream provider we target
(OpenAI, DeepSeek, GLM, MiniMax, Mimo, Step, LongCat, …) speaks this format
already, so a translation layer would only add impedance.

Canonical shapes (per OpenAI Chat Completions):

    {"role": "system",    "content": str}
    {"role": "user",      "content": str | list[content_part]}
    {"role": "assistant", "content": str | None,
                          "tool_calls": [{"id", "type": "function",
                                          "function": {"name", "arguments": json_str}}],
                          "reasoning_content": str}        # vendor extension
    {"role": "tool",      "tool_call_id": str, "content": str}

For multimodal user input, `content` is a list of parts:
    [{"type": "text", "text": ...}, {"type": "image_url", "image_url": {"url": ...}}]

This module provides thin, pure helpers for the small manipulations the
agent loop needs. Providers may pass messages straight through with at most
light sanitisation (e.g. dropping orphan tool_calls).
"""

from __future__ import annotations

from typing import Any


# Type aliases — all just dict at runtime; these are for readability/tooling.
Message = dict[str, Any]
"""An OpenAI-format message dict."""

ToolCall = dict[str, Any]
"""An OpenAI-format tool_call entry: {"id","type":"function","function":{"name","arguments"}}."""

Messages = list[Message]


# ---- pure helpers --------------------------------------------------------


def user_text(text: str) -> Message:
    """Build a plain user message."""
    return {"role": "user", "content": text}


def assistant_text(text: str) -> Message:
    """Build a plain assistant message."""
    return {"role": "assistant", "content": text}


def assistant_with_tool_calls(
    *,
    content: str | None = None,
    tool_calls: list[ToolCall],
    reasoning_content: str | None = None,
) -> Message:
    """Build an assistant message that issues tool calls.

    `tool_calls` items may be either full OpenAI tool_call dicts or the inner
    {"name","arguments"} form; both are normalised here.
    """
    msg: Message = {"role": "assistant", "content": content if content is not None else ""}
    if reasoning_content:
        msg["reasoning_content"] = reasoning_content
    msg["tool_calls"] = [_normalise_tool_call(tc) for tc in tool_calls]
    return msg


def tool_result(tool_call_id: str, content: str, *, is_error: bool = False) -> Message:
    """Build a role=tool result message. `is_error` is reflected by prefixing
    the content — OpenAI has no error flag on tool results."""
    text = content
    if is_error and not content.startswith("[error"):
        text = f"[error] {content}"
    return {"role": "tool", "tool_call_id": tool_call_id, "content": text}


def user_with_images(text: str, images: list[dict[str, str]]) -> Message:
    """Build a multimodal user message. `images` is a list of
    {"url": data_or_remote_url, "detail": "auto"|"low"|"high"}.

    Falls back to a plain text message when `images` is empty."""
    if not images:
        return user_text(text)
    parts: list[dict[str, Any]] = []
    for img in images:
        parts.append(
            {"type": "image_url", "image_url": {"url": img["url"], "detail": img.get("detail", "auto")}}
        )
    if text:
        parts.append({"type": "text", "text": text})
    return {"role": "user", "content": parts}


def get_text(msg: Message) -> str:
    """Best-effort extract of textual content from any message."""
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        return "".join(parts)
    return ""


def get_tool_calls(msg: Message) -> list[ToolCall]:
    """Return the message's tool_calls (empty list if none)."""
    tcs = msg.get("tool_calls")
    return list(tcs) if tcs else []


def _normalise_tool_call(tc: ToolCall) -> ToolCall:
    """Accept either the full {"id","type","function"} form or a compact
    {"id","name","arguments"} form and return the full OpenAI form."""
    if "function" in tc:
        return tc
    return {
        "id": tc.get("id", ""),
        "type": "function",
        "function": {
            "name": tc.get("name", ""),
            "arguments": tc.get("arguments", ""),
        },
    }


__all__ = [
    "Message",
    "ToolCall",
    "Messages",
    "user_text",
    "assistant_text",
    "assistant_with_tool_calls",
    "tool_result",
    "user_with_images",
    "get_text",
    "get_tool_calls",
]
