from ezwork.core.message import (
    assistant_text,
    assistant_with_tool_calls,
    get_text,
    get_tool_calls,
    tool_result,
    user_text,
    user_with_images,
)


def test_user_text():
    m = user_text("hello")
    assert m == {"role": "user", "content": "hello"}


def test_assistant_text():
    assert assistant_text("hi") == {"role": "assistant", "content": "hi"}


def test_assistant_with_tool_calls_normalises_compact_form():
    m = assistant_with_tool_calls(
        content="thinking",
        tool_calls=[{"id": "c1", "name": "foo", "arguments": '{"x": 1}'}],
    )
    assert m["role"] == "assistant"
    assert m["content"] == "thinking"
    assert m["tool_calls"] == [
        {"id": "c1", "type": "function", "function": {"name": "foo", "arguments": '{"x": 1}'}}
    ]


def test_assistant_with_tool_calls_passes_full_form_through():
    full = {
        "id": "c1",
        "type": "function",
        "function": {"name": "foo", "arguments": "{}"},
    }
    m = assistant_with_tool_calls(content=None, tool_calls=[full])
    assert m["tool_calls"] == [full]


def test_assistant_with_tool_calls_reasoning():
    m = assistant_with_tool_calls(
        content="x", tool_calls=[{"id": "c1", "name": "t", "arguments": "{}"}],
        reasoning_content="why",
    )
    assert m["reasoning_content"] == "why"


def test_tool_result_plain():
    m = tool_result("abc", "42")
    assert m == {"role": "tool", "tool_call_id": "abc", "content": "42"}


def test_tool_result_error_prefix():
    m = tool_result("abc", "boom", is_error=True)
    assert m["content"] == "[error] boom"


def test_tool_result_no_double_error_prefix():
    m = tool_result("abc", "[error] already", is_error=True)
    assert m["content"] == "[error] already"


def test_user_with_images():
    m = user_with_images("look", [{"url": "data:image/png;base64,xxx"}])
    assert m["role"] == "user"
    assert isinstance(m["content"], list)
    assert m["content"][0]["type"] == "image_url"
    assert m["content"][1] == {"type": "text", "text": "look"}


def test_user_with_images_empty_falls_back_to_text():
    m = user_with_images("just text", [])
    assert m == {"role": "user", "content": "just text"}


def test_get_text_string():
    assert get_text({"role": "user", "content": "hi"}) == "hi"


def test_get_text_parts():
    m = {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}
    assert get_text(m) == "ab"


def test_get_text_missing():
    assert get_text({"role": "assistant"}) == ""


def test_get_tool_calls():
    tcs = [{"id": "c1", "type": "function", "function": {"name": "x", "arguments": "{}"}}]
    assert get_tool_calls({"role": "assistant", "tool_calls": tcs}) == tcs


def test_get_tool_calls_empty():
    assert get_tool_calls({"role": "user", "content": "x"}) == []
