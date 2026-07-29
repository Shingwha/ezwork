import pytest

from ezwork.core.tool import Tool, ToolError, ToolRegistry


def _echo_tool():
    return Tool(
        name="echo",
        description="echoes input",
        params={
            "text": {"type": "string", "description": "what to echo"},
            "upper": {"type": "boolean", "default": False, "description": "uppercase"},
        },
        func=lambda a: a["text"].upper() if a.get("upper") else a["text"],
    )


def test_tool_runs_with_defaults():
    t = _echo_tool()
    assert t.run({"text": "hi"}) == "hi"
    assert t.run({"text": "hi", "upper": True}) == "HI"


def test_tool_missing_required_raises():
    t = _echo_tool()
    with pytest.raises(ToolError) as ei:
        t.run({})
    assert ei.value.code == "missing_param"


def test_tool_to_schema_shape():
    schema = _echo_tool().to_schema()
    assert schema["type"] == "function"
    fn = schema["function"]
    assert fn["name"] == "echo"
    props = fn["parameters"]["properties"]
    assert set(props) == {"text", "upper"}
    assert "text" in fn["parameters"]["required"]
    assert "upper" not in fn["parameters"]["required"]
    assert props["upper"]["default"] is False


def test_tool_async_detection():
    async def _f(a):
        return "async"

    t = Tool("a", "desc", {}, _f)
    assert t.is_async is True


def test_tool_registry_caches_schema():
    reg = ToolRegistry()
    reg.register(_echo_tool())
    s1 = reg.all_schemas()
    s2 = reg.all_schemas()
    assert s1 is s2  # cached
    reg.register(Tool("x", "x", {}, lambda a: "x"))
    s3 = reg.all_schemas()
    assert s3 is not s1  # cache invalidated


def test_tool_registry_derived_excludes():
    reg = ToolRegistry()
    reg.register(_echo_tool())
    reg.register(Tool("other", "x", {}, lambda a: "x"))
    derived = reg.derived(exclude={"other"})
    assert derived.get("echo") is not None
    assert derived.get("other") is None
    # original unaffected
    assert reg.get("other") is not None


def test_tool_param_must_be_dict():
    with pytest.raises(TypeError):
        Tool("bad", "x", {"p": "not a dict"}, lambda a: "x")
