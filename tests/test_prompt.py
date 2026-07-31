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


# ── skill discovery / frontmatter ────────────────────────────

from ezwork.app.prompt import _discover_skills, _parse_frontmatter


def test_parse_frontmatter_simple():
    text = """---
name: foo
description: Does foo things.
---

# body
"""
    fm = _parse_frontmatter(text)
    assert fm == {"name": "foo", "description": "Does foo things."}


def test_parse_frontmatter_folded_block():
    text = """---
name: bar
description: >-
  Wraps any HTTP API, web service, library, database, or existing tool into a
  self-contained CLI application.
---

# body
"""
    fm = _parse_frontmatter(text)
    assert fm["name"] == "bar"
    assert fm["description"] == (
        "Wraps any HTTP API, web service, library, database, or existing tool "
        "into a self-contained CLI application."
    )


def test_parse_frontmatter_missing_or_malformed():
    assert _parse_frontmatter("# no frontmatter\nbody") == {}
    assert _parse_frontmatter("---\nunclosed") == {}


def test_parse_frontmatter_ignores_quotes_and_colon_in_value():
    fm = _parse_frontmatter('---\nname: "quoted"\ndescription: see https://x.com/y\n---\n')
    assert fm == {"name": "quoted", "description": "see https://x.com/y"}


def test_discover_skills_uses_frontmatter_name_and_description(tmp_path):
    (tmp_path / "skills").mkdir()
    skill_dir = tmp_path / "skills" / "my-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: real-name\ndescription: Does real things.\n---\n\nbody\n",
        encoding="utf-8",
    )
    skills = _discover_skills([tmp_path / "skills"])
    assert skills == [("real-name", "Does real things.", str((skill_dir / "SKILL.md").resolve()))]


def test_discover_skills_skips_dirs_without_skill_md(tmp_path):
    (tmp_path / "skills").mkdir()
    (tmp_path / "skills" / "nope").mkdir()
    assert _discover_skills([tmp_path / "skills"]) == []
