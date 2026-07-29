from ezwork.core.prompt import Prompt, Section


def test_prompt_simple_section():
    p = Prompt([Section("a", "hello", priority=0)])
    out = p.build()
    assert out == "a: hello"


def test_prompt_no_kernel_tags():
    # the kernel renders plain text only; it never adds tags of its own
    p = Prompt([Section("rule", "be brief", priority=0)])
    out = p.build()
    assert out == "rule: be brief"
    assert "<rule>" not in out
    assert "<system-prompt>" not in out


def test_prompt_priority_ordering_stable():
    p = Prompt(
        [
            Section("late", "L", priority=100),
            Section("early", "E", priority=1),
            Section("mid", "M", priority=50),
        ]
    )
    out = p.build()
    assert out.index("E") < out.index("M") < out.index("L")


def test_prompt_disable_section():
    p = Prompt([Section("a", "x"), Section("b", "y")])
    p.disable("a")
    out = p.build()
    assert out == "b: y"


def test_prompt_dynamic_callable_content():
    p = Prompt([Section("greet", lambda ctx: f"hi {ctx['name']}")])
    out = p.context(name="world").build()
    assert out == "greet: hi world"


def test_prompt_multiline_content():
    p = Prompt([Section("note", "line1\nline2")])
    out = p.build()
    assert out == "note:\nline1\nline2"


def test_prompt_register_overrides_same_name():
    p = Prompt([Section("a", "old")])
    p.register(Section("a", "new"))
    out = p.build()
    assert out == "a: new"


def test_prompt_empty_section_skipped():
    p = Prompt([Section("empty", ""), Section("real", "x")])
    out = p.build()
    assert out == "real: x"


def test_prompt_sections_joined_by_blank_line():
    p = Prompt([Section("a", "1", priority=0), Section("b", "2", priority=1)])
    out = p.build()
    assert out == "a: 1\n\nb: 2"
