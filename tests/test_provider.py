"""Provider-side tests: collect_stream accumulation + StreamChunk helpers."""



from ezwork.core.provider import StreamChunk, Usage, collect_stream


async def _aiter(chunks):
    for c in chunks:
        yield c


async def test_collect_stream_plain_text():
    chunks = _aiter([StreamChunk.text_delta("hello "), StreamChunk.text_delta("world"), StreamChunk.done()])
    r = await collect_stream(chunks)
    assert r.content == "hello world"
    assert r.tool_calls is None
    assert r.finish_reason == "stop"
    assert not r.is_error


async def test_collect_stream_reasoning_separate_from_text():
    chunks = _aiter(
        [
            StreamChunk.reasoning_delta("thinking..."),
            StreamChunk.text_delta("answer"),
            StreamChunk.done(),
        ]
    )
    r = await collect_stream(chunks)
    assert r.content == "answer"
    assert r.reasoning_content == "thinking..."


async def test_collect_stream_tool_call_assembles_arguments_as_json_string():
    chunks = _aiter(
        [
            StreamChunk.tool_call_delta(0, id="call_1", name="sum"),
            StreamChunk.tool_call_delta(0, arguments_delta='{"a": 1'),
            StreamChunk.tool_call_delta(0, arguments_delta=', "b": 2}'),
            StreamChunk.done("tool_calls"),
        ]
    )
    r = await collect_stream(chunks)
    assert r.finish_reason == "tool_calls"
    assert r.tool_calls is not None and len(r.tool_calls) == 1
    tc = r.tool_calls[0]
    assert tc["id"] == "call_1"
    assert tc["function"]["name"] == "sum"
    # arguments is a JSON string (matches OpenAI wire format)
    assert tc["function"]["arguments"] == '{"a": 1, "b": 2}'


async def test_collect_stream_multiple_tool_calls_preserve_order():
    chunks = _aiter(
        [
            StreamChunk.tool_call_delta(1, id="b", name="two", arguments_delta="{}"),
            StreamChunk.tool_call_delta(0, id="a", name="one", arguments_delta="{}"),
            StreamChunk.done("tool_calls"),
        ]
    )
    r = await collect_stream(chunks)
    assert [tc["id"] for tc in r.tool_calls] == ["b", "a"]


async def test_collect_stream_arguments_assembled_verbatim():
    """arguments is the raw concatenated string, NOT parsed — provider/agent
    parse it once when needed."""
    chunks = _aiter(
        [
            StreamChunk.tool_call_delta(0, id="x", name="t", arguments_delta="{bad"),
            StreamChunk.done("tool_calls"),
        ]
    )
    r = await collect_stream(chunks)
    tc = r.tool_calls[0]
    assert tc["function"]["arguments"] == "{bad"


async def test_collect_stream_usage_last_wins():
    chunks = _aiter(
        [
            StreamChunk.usage(Usage(1, 2)),
            StreamChunk.usage(Usage(10, 20)),
            StreamChunk.done(),
        ]
    )
    r = await collect_stream(chunks)
    assert r.usage.prompt_tokens == 10
    assert r.usage.completion_tokens == 20


async def test_collect_stream_error_sets_is_error():
    chunks = _aiter([StreamChunk.error("boom")])
    r = await collect_stream(chunks)
    assert r.is_error
    assert r.finish_reason == "error"
    assert r.error == "boom"


async def test_collect_stream_empty():
    r = await collect_stream(_aiter([]))
    assert r.content is None
    assert r.tool_calls is None
    assert r.finish_reason is None


def test_stream_chunk_factories():
    assert StreamChunk.text_delta("x").type == "text_delta"
    assert StreamChunk.done().finish_reason == "stop"
    assert StreamChunk.error("e").error == "e"
    u = Usage(1, 2)
    assert StreamChunk.usage(u).usage is u
