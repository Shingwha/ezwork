"""System-prompt assembly for the Ezwork app.

The kernel renders ONLY plain text (ezwork.core.prompt.Prompt.build() adds no
tags and does no nesting). This module composes flat sections and embeds any
markup it wants directly inside the section content strings. The kernel never
sees or wraps the content. Swap this module to change the prompt style without
touching the kernel.

Tool definitions are intentionally NOT listed here — the provider already sends
each tool's full schema (name, description, params) with every tool-calling
request, so the model already knows them. Repeating them in the system prompt
would be redundant.

Layout (priority ascending — stable, low-churn sections first to maximise
prefix-cache hits):
    identity      (0)   who the agent is
    guidelines    (10)  working rules
    shell         (12)  efficient shell usage (search, pipes, parallelism)
    environment   (20)  cwd / home / config path / sessions path / skills dirs
    subagents     (25)  how to spawn a sub-agent via `ezwork -p` from bash
    skills        (30)  skill discovery + read-based invocation
    agents        (40)  memory guide + merged global ~/.ezwork/AGENTS.md
                        and nearest project AGENTS.md (dynamic, last)
"""

from __future__ import annotations

from pathlib import Path

from ezwork.core import Prompt, Section

IDENTITY = (
    "You are Ezwork, a lean coding assistant with file and shell tools. "
    "Be concise. Use the tools to inspect and modify files; do not echo "
    "large file contents back to the user unless asked."
)

GUIDELINES = (
    "Working rules:\n"
    "- Prefer the edit tool over rewriting whole files; after modifying a "
    "file, verify the change (re-read or run a check).\n"
    "- When a tool returns an error, read it and recover; do not give up.\n"
    "- Keep answers short; do not restate the user's request.\n"
    "- Search via bash (no dedicated search tools): content with "
    "`grep -rn --exclude-dir={node_modules,.git,__pycache__,.venv,dist,build,target} 'pat' .`, "
    "files with `find . -name '*.py'`. List a directory with `ls` before "
    "assuming what exists."
)

SHELL = (
    "Efficient shell usage:\n"
    "- Batch independent tool calls into ONE message: tool calls in a single "
    "response run in parallel, so 3-5 independent commands complete in one "
    "round-trip — never run them one at a time.\n"
    "- Make one call answer a whole question: chain with pipes "
    "(`grep -rln 'TODO' src | head`), `&&`/`;` (`ls src && wc -l src/*.py`), "
    "or compound commands instead of one call per item.\n"
    "- Inspect many files in one command instead of reading them one by one: "
    "`grep -rn -A2 -B2 'pattern' src/`, `head -100 a.py b.py c.py`, "
    "`find src -name '*.py' -exec wc -l {} +`.\n"
    "- Use loops and xargs for repetitive work: "
    "`for f in src/*.py; do wc -l \"$f\"; done`, "
    "`find tests -name '*.py' | xargs grep -l skip`.\n"
    "- Trim outputs you only need a preview of (`head`, `-l`, `wc -l`) so "
    "tool results stay small.\n"
    "- Run long independent commands inside one call with `&` and `wait` "
    "when their outputs don't interleave.\n"
    "- Parallel bash calls share one session: parallelize read-only "
    "commands freely; keep `cd`/`export` sequences sequential. Bare `cd` "
    "and `export` persist across calls; `cd x && cmd` changes the "
    "directory only for that call."
)

SUBAGENTS = (
    "Sub-agents: spawn an isolated agent via `ezwork -p \"task\"` from bash "
    "(fresh session, own history, shares your tools/config). Default to "
    "delegating multi-step (5+ calls), investigative, or batch work — it "
    "keeps this conversation clean. Do not delegate 1-3 quick calls or work "
    "whose result feeds your very next decision.\n"
    "\n"
    "Prompts must be self-contained (the sub-agent cannot see this "
    "conversation): goal + output format + steps + tool constraints.\n"
    "Output rules: the output is ALL the parent sees — be complete and "
    "structured; state failures explicitly; when reporting file changes, "
    "give path + brief description.\n"
    "\n"
    "Example: ezwork -p \"Analyse src/auth/: read __init__.py, list public "
    "functions, check each has a test in tests/test_auth.py, report "
    "missing ones as [missing-test] <fn> — <reason>. Do not modify files.\"\n"
    "Piped stdin is appended as context: `git diff | ezwork -p \"write a "
    "commit message\"`. Continue with `ezwork -p \"...\" -s <session_id>`."
)


AGENTS_GUIDE = (
    "Persistent memory (survives across sessions and restarts) — the "
    "following AGENTS.md contents are injected below:\n"
    "- Global: ~/.ezwork/AGENTS.md — durable facts about the user, the "
    "machine and the environment (e.g. proxy setup, git gotchas, tool "
    "preferences). Applies in every project.\n"
    "- Project: <project root>/AGENTS.md — conventions, decisions and "
    "gotchas specific to one project. Looked up from the working "
    "directory upward; the nearest AGENTS.md wins.\n"
    "- WRITE to these files when you learn something durable: a user "
    "preference, an environment quirk, a recurring gotcha, a project "
    "convention. Update the file directly with the read + edit/write "
    "tools. Keep entries short, factual, organised with ## headers; "
    "append or edit precisely — never rewrite unrelated content.\n"
    "- Do NOT write: transient task state, code facts that are obvious "
    "from the codebase, or anything the user did not confirm matters.\n"
    "- On conflict: project memory wins for project-specific matters; "
    "global memory wins for user/environment facts. Both are advisory — "
    "explicit user instructions in the current conversation always win."
)


def _parse_frontmatter(text: str) -> dict:
    """Parse the YAML frontmatter block of a SKILL.md (zero dependencies).

    Handles the subset our skills actually use: plain `key: value` scalars
    (quoted or not) and `>-` / `|` block scalars. Anything fancier is
    ignored. Returns {} when there is no frontmatter or it is malformed.
    """
    if not text.startswith("---"):
        return {}
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}
    fm: dict[str, str] = {}
    key: str | None = None
    block: list[str] = []
    block_literal = False
    for raw in parts[1].splitlines():
        line = raw.rstrip()
        if not line.strip() or line.startswith("#") or line in ("---", "..."):
            continue
        if key is not None and line.startswith(" "):
            block.append(line.strip())
            continue
        if key is not None:  # block ended by a new key at column 0
            fm[key] = "\n".join(block) if block_literal else " ".join(block)
            key = None
        k, _, v = line.partition(":")
        key = k.strip()
        v = v.strip()
        if v in (">", ">-"):
            block, block_literal = [], False
        elif v in ("|", "|-"):
            block, block_literal = [], True
        else:
            fm[key] = v.strip("'\"")
            key = None
    if key is not None:
        fm[key] = "\n".join(block) if block_literal else " ".join(block)
    return fm


def _read_skill_md(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def _discover_skills(skills_dirs: list[Path]) -> list[tuple[str, str, str]]:
    """Find skills: any subdirectory of a skills dir that contains SKILL.md.
    Returns [(name, description, skill_md_path), ...]. The frontmatter `name`
    wins over the directory name; earlier dirs win on name conflicts (the
    caller lists dirs highest-priority first)."""
    seen: set[str] = set()
    out: list[tuple[str, str, str]] = []
    for base in skills_dirs:
        if not base.exists() or not base.is_dir():
            continue
        try:
            children = sorted(base.iterdir())
        except OSError:
            continue
        for child in children:
            if not child.is_dir():
                continue
            skill_md = child / "SKILL.md"
            if not skill_md.exists():
                continue
            fm = _parse_frontmatter(_read_skill_md(skill_md))
            name = fm.get("name") or child.name
            if name in seen:
                continue
            seen.add(name)
            out.append((name, fm.get("description", ""), str(skill_md.resolve())))
    return out


def _render_skills_block(skills_dirs: list[Path]) -> str:
    skills = _discover_skills(skills_dirs)
    if not skills:
        return "(no skills available)"
    lines = [
        "Skills are reusable capability packs — each is a directory with a "
        "SKILL.md that describes what it does and how to use it.",
        "To use a skill, read its SKILL.md with the read tool "
        '(e.g. read(path="<path>")) and then follow its instructions exactly. '
        "Never assume skill content — always read SKILL.md before acting.",
        "",
        "Available skills (name — description; SKILL.md path):",
    ]
    for name, desc, skill_md in skills:
        tag = f'  <skill name="{name}"'
        if desc:
            esc = desc.replace("&", "&amp;").replace("<", "&lt;").replace('"', "&quot;")
            tag += f' description="{esc}"'
        tag += f' path="{skill_md}"/>'
        lines.append(tag)
    return "\n".join(lines)


def _find_project_agents(cwd: str) -> Path | None:
    """Nearest AGENTS.md walking up from cwd (project memory)."""
    for d in [Path(cwd), *Path(cwd).resolve().parents]:
        candidate = d / "AGENTS.md"
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def _render_agents(home: str, cwd: str) -> str:
    """Persistent-memory guide + merged global ~/.ezwork/AGENTS.md and the
    nearest project AGENTS.md, each labelled with its source path so the
    agent knows where to write."""
    parts: list[str] = []
    global_p = Path(home) / ".ezwork" / "AGENTS.md"
    for label, p in (("Global memory", global_p), ("Project memory", _find_project_agents(cwd))):
        if p is None:
            continue
        try:
            if p.exists() and p.is_file():
                txt = p.read_text(encoding="utf-8").strip()
                if txt:
                    parts.append(f"## {label} ({p})\n\n{txt}")
        except OSError:
            continue
    if not parts:
        contents = "(no AGENTS.md files yet — create ~/.ezwork/AGENTS.md or <project>/AGENTS.md to persist facts)"
    else:
        contents = "\n\n---\n\n".join(parts)
    return AGENTS_GUIDE + "\n\n" + contents


def build_system_prompt(
    *,
    cwd: str,
    home: str,
    config_path: str,
    sessions_dir: str,
    skills_dirs: list[Path] | None = None,
) -> str:
    """Compose the full plain-text system prompt with app-embedded markup."""
    skills_dirs = skills_dirs or [
        Path(home) / ".ezwork" / "skills",
        Path(cwd) / ".ezwork" / "skills",
    ]

    environment = (
        f"Environment:\n"
        f"- working directory: {cwd}\n"
        f"- home: {home}\n"
        f"- config file: {config_path} (edit api_key/model here, then restart)\n"
        f"- sessions directory: {sessions_dir}\n"
        f"- skills directories: {', '.join(str(d) for d in skills_dirs)}"
    )

    sections = [
        Section("identity", IDENTITY, priority=0),
        Section("guidelines", GUIDELINES, priority=10),
        Section("shell", SHELL, priority=12),
        Section("environment", environment, priority=20),
        Section("subagents", SUBAGENTS, priority=25),
        Section("skills", _render_skills_block(skills_dirs), priority=30),
        Section("agents", _render_agents(home, cwd), priority=40),
    ]

    return Prompt(sections).build()


__all__ = ["build_system_prompt"]
