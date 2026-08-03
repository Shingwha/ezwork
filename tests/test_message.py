"""Message helper tests — one test per helper family (wire shapes are the
public contract; details like role/content/tool_calls are asserted together)."""

from ezwork.core.message import (
    assistant_text,
    assistant_with_tool_calls,
    get_text,
    get_tool_calls,
    tool_result,
    user_text,
    user_with_images,
)


def test_text_helpers():
    assert user_text("hello") == {"role": "user", "content": "hello"}
    assert assistant_text("hi") == {"role": "assistant", "content": "hi"}
    # get_text resolves string content, part lists, and missing content.
    assert get_text({"role": "user", "content": "hi"}) == "hi"
    parts = {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}
    assert get_text(parts) == "ab"
    assert get_text({"role": "assistant"}) == ""


def test_assistant_with_tool_calls_forms():
    # Compact form is normalised to the full wire shape.
    m = assistant_with_tool_calls(
        content="thinking",
        tool_calls=[{"id": "c1", "name": "foo", "arguments": '{"x": 1}'}],
    )
    assert m["role"] == "assistant"
    assert m["content"] == "thinking"
    assert m["tool_calls"] == [
        {"id": "c1", "type": "function", "function": {"name": "foo", "arguments": '{"x": 1}'}}
    ]
    # Full form passes through untouched; reasoning_content is preserved.
    full = {"id": "c2", "type": "function", "function": {"name": "bar", "arguments": "{}"}}
    m = assistant_with_tool_calls(
        content=None, tool_calls=[full], reasoning_content="why"
    )
    assert m["tool_calls"] == [full]
    assert m["reasoning_content"] == "why"


def test_tool_result_forms():
    assert tool_result("abc", "42") == {"role": "tool", "tool_call_id": "abc", "content": "42"}
    assert tool_result("abc", "boom", is_error=True)["content"] == "[error] boom"
    # An already-prefixed error is never double-prefixed.
    assert tool_result("abc", "[error] already", is_error=True)["content"] == "[error] already"


def test_user_with_images():
    m = user_with_images("look", [{"url": "data:image/png;base64,xxx"}])
    assert m["role"] == "user"
    assert m["content"][0]["type"] == "image_url"
    assert m["content"][1] == {"type": "text", "text": "look"}
    # No images → plain text content.
    assert user_with_images("just text", []) == {"role": "user", "content": "just text"}


def test_get_tool_calls():
    tcs = [{"id": "c1", "type": "function", "function": {"name": "x", "arguments": "{}"}}]
    assert get_tool_calls({"role": "assistant", "tool_calls": tcs}) == tcs
    assert get_tool_calls({"role": "user", "content": "x"}) == []
