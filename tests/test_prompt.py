from ezwork.core.prompt import Prompt, Section
from ezwork.app.prompt import _discover_skills, _parse_frontmatter


def test_prompt_rendering_forms():
    # Single-line section, multi-line section, and blank-line joining.
    assert Prompt([Section("a", "hello", priority=0)]).build() == "a: hello"
    assert Prompt([Section("note", "line1\nline2")]).build() == "note:\nline1\nline2"
    p = Prompt([Section("a", "1", priority=0), Section("b", "2", priority=1)])
    assert p.build() == "a: 1\n\nb: 2"


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


def test_prompt_register_overrides_and_empty_skipped():
    p = Prompt([Section("a", "old")])
    p.register(Section("a", "new"))
    assert p.build() == "a: new"
    # Empty sections are skipped entirely.
    p = Prompt([Section("empty", ""), Section("real", "x")])
    assert p.build() == "real: x"


# ── skill discovery / frontmatter ────────────────────────────


def test_parse_frontmatter_simple():
    text = """---
name: foo
description: Does foo things.
---

# body
"""
    fm = _parse_frontmatter(text)
    assert fm == {"name": "foo", "description": "Does foo things."}
    # Quoted values and colons inside values are handled.
    fm = _parse_frontmatter('---\nname: "quoted"\ndescription: see https://x.com/y\n---\n')
    assert fm == {"name": "quoted", "description": "see https://x.com/y"}


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


def test_discover_skills_uses_frontmatter_and_skips_bare_dirs(tmp_path):
    (tmp_path / "skills").mkdir()
    skill_dir = tmp_path / "skills" / "my-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: real-name\ndescription: Does real things.\n---\n\nbody\n",
        encoding="utf-8",
    )
    (tmp_path / "skills" / "nope").mkdir()  # no SKILL.md → skipped
    skills = _discover_skills([tmp_path / "skills"])
    assert skills == [("real-name", "Does real things.", str((skill_dir / "SKILL.md").resolve()))]


def test_build_system_prompt_assembles_dynamic_sections(tmp_path):
    """Smoke test for the assembly wiring: cwd, skills and AGENTS.md are
    injected into the output. Deliberately NOT wording-coupled — the prompt
    text itself is iterated freely without breaking tests."""
    from ezwork.app.prompt import build_system_prompt

    skills_dir = tmp_path / ".ezwork" / "skills"
    (skills_dir / "demo").mkdir(parents=True)
    (skills_dir / "demo" / "SKILL.md").write_text(
        "---\nname: demo\ndescription: test skill\n---\nbody\n",
        encoding="utf-8",
    )
    (tmp_path / "AGENTS.md").write_text("## project rule\nuse tabs\n", encoding="utf-8")

    prompt = build_system_prompt(
        cwd=str(tmp_path),
        home=str(tmp_path),
        config_path=str(tmp_path / "config.json"),
        sessions_dir=str(tmp_path / "sessions"),
        skills_dirs=[skills_dir],
    )
    assert str(tmp_path) in prompt                       # environment section
    assert 'name="demo"' in prompt                       # skills block injected
    assert "## project rule" in prompt                   # AGENTS.md injected
