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
    guidelines    (10)  working rules + grep/glob via bash hint
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
    "- Prefer the edit tool over rewriting whole files.\n"
    "- After modifying a file, verify the change (re-read or run a check).\n"
    "- When a tool returns an error, read it and recover; do not give up.\n"
    "- Keep answers short; skip restating the user's request.\n"
    "- Prefer the dedicated search tools over shell commands: `glob` for file "
    "discovery (e.g. `**/*.py`) and `grep` for content search (regex, with "
    "type/context/count modes) — they filter noise automatically and give "
    "cleaner results. Bash `find`/`grep` are allowed only when you need shell "
    "piping or other bash features (e.g. `grep -r x | wc -l`)."
)

SUBAGENTS = (
    "Sub-agents (delegation): you can spawn an isolated sub-agent by running "
    "the `ezwork` command itself through the bash tool. Each run is a fresh, "
    "isolated session that shares your tools and config but has its own "
    "history — delegate to keep this conversation clean and your reasoning "
    "focused. Default to delegating: when in doubt, delegate.\n"
    "\n"
    "WHEN to delegate (delegate if ANY of these holds):\n"
    "  - The work needs many tool calls (roughly 5+ steps: reading many "
    "files, searching, retrying, batch edits) and only the final result "
    "matters to you.\n"
    "  - It is investigative: exploring an unknown codebase, digging into "
    "logs, reproducing a bug, comparing versions.\n"
    "  - It is a trial-and-error grind (build → fail → fix loops) that "
    "would flood this conversation with noise.\n"
    "  - It splits into independent chunks — fire one sub-agent per chunk "
    "in the same round and let them run in parallel.\n"
    "  - It is a long batch operation (many files, big edits, slow "
    "commands).\n"
    "\n"
    "WHEN NOT to delegate (just do it yourself):\n"
    "  - 1-3 quick tool calls.\n"
    "  - The result must feed directly into your very next decision "
    "mid-task — do the first steps yourself and delegate only the deep-dive "
    "parts.\n"
    "  - The task needs the user's live input.\n"
    "\n"
    "Start a sub-agent:\n"
    "  ezwork -p \"your task description here\"\n"
    "The answer comes back as stdout (visible in the tool result). "
    "A `session: <id>` line is also printed — just use that id to continue.\n"
    "\n"
    "Piped content is appended as context, so you can hand the sub-agent "
    "data without a file round-trip:\n"
    "  cat data.txt | ezwork -p \"summarize the key numbers\"\n"
    "  git diff | ezwork -p \"write a commit message for this diff\"\n"
    "\n"
    "Continue that same sub-agent on a follow-up:\n"
    "  ezwork -p \"follow-up task\" -s \"<session_id>\"\n"
    "\n"
    "IMPORTANT — Prompting rules for sub-agents:\n"
    "Since a sub-agent cannot see this conversation, every sub-agent prompt "
    "must be a fully self-contained specification. A good sub-agent prompt "
    "includes:\n"
    "  1. Goal — what needs to be done.\n"
    "  2. Output format — what the final answer should look like "
    "(e.g. \"list the files\", \"print the diff\", \"write a summary\").\n"
    "  3. Workflow — the steps to follow, especially order of operations.\n"
    "  4. Tool restrictions — which tools to use or avoid "
    "(e.g. \"read-only, do not modify any files\", \"use grep for search, "
    "do not read files one by one\", \"use edit tool, do not rewrite whole "
    "files\").\n"
    "\n"
    "Output rules for sub-agents (the sub-agent must follow these when "
    "producing its answer):\n"
    "  - The output is the ONLY thing the parent session sees. The parent "
    "cannot see intermediate steps, errors, or tool results. Therefore the "
    "output must be complete and self-explanatory.\n"
    "  - Structure the output clearly: use headings, lists, code blocks, "
    "or tables as appropriate. Make it easy for the parent to read.\n"
    "  - Always summarise what was done at the end, especially if the task "
    "involved multiple steps or file modifications.\n"
    "  - If a step failed or was skipped, say so explicitly — do not hide "
    "it. The parent needs to know the actual state.\n"
    "  - When reporting file changes, include the file path and a brief "
    "description of each change (e.g. \"modified src/main.py: added input "
    "validation on line 42\").\n"
    "\n"
    "Bad example (too vague):\n"
    "  ezwork -p \"check the auth module\"\n"
    "\n"
    "Good example:\n"
    "  ezwork -p \"Analyse the auth module in src/auth/.\n"
    "  1. Read src/auth/__init__.py and list all public functions.\n"
    "  2. For each function, check if it has a unit test in tests/test_auth.py.\n"
    "  3. Report any function that lacks a test, using this format:\n"
    "     [missing-test] <function_name> — <reason>\n"
    "  Do not modify any files.\"\n"
    "\n"
    "If the task is complex, break it into multiple sub-agent calls "
    "rather than one giant prompt. Use follow-up calls (-s) to build on "
    "previous results."
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


def _discover_skills(skills_dirs: list[Path]) -> list[tuple[str, str]]:
    """Find skills: any subdirectory of a skills dir that contains SKILL.md.
    Returns [(name, skill_md_path), ...]. Earlier dirs win on name conflicts
    (the caller lists dirs highest-priority first)."""
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
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
            name = child.name
            if name in seen:
                continue
            seen.add(name)
            out.append((name, str(skill_md.resolve())))
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
        "Available skills (name + SKILL.md path):",
    ]
    for name, skill_md in skills:
        lines.append(f'  <skill name="{name}" path="{skill_md}"/>')
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
        Section("environment", environment, priority=20),
        Section("subagents", SUBAGENTS, priority=25),
        Section("skills", _render_skills_block(skills_dirs), priority=30),
        Section("agents", _render_agents(home, cwd), priority=40),
    ]

    return Prompt(sections).build()


__all__ = ["build_system_prompt"]
