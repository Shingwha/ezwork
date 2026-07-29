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
    agents        (40)  merged ~/.ezwork/AGENTS.md + ./AGENTS.md
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
    "- For searching, use the bash tool: `grep -r` for content search and "
    "`find` / shell globbing (e.g. `ls **/*.py`) for file discovery. These "
    "are faster and more flexible than reading files one by one."
)

SUBAGENTS = (
    "Sub-agents (delegation): you can spawn an isolated sub-agent by running "
    "the `ezwork` command itself through the bash tool. Each run is a fresh, "
    "isolated session that shares your tools and config but has its own "
    "history. Use this to delegate a self-contained task (e.g. a focused "
    "investigation or a big edit) without polluting this conversation.\n"
    "\n"
    "Start a sub-agent and capture its session id from stderr:\n"
    "  SID=$(ezwork -p \"analyse the auth module and list its entry points\" "
    "2>&1 >/dev/null | grep '^session:' | cut -d' ' -f2)\n"
    "The sub-agent's answer is written to stdout (read it from the bash "
    "result); its `session: <id>` line goes to stderr.\n"
    "\n"
    "Continue that same sub-agent on a follow-up:\n"
    "  ezwork -p \"now write tests for what you found\" -s \"$SID\"\n"
    "\n"
    "Notes:\n"
    "- The answer text comes back as the bash command's stdout — read it from "
    "the tool result, do not expect it to appear in this chat.\n"
    "- Use this for genuinely independent subtasks; for small lookups just use "
    "read/grep directly.\n"
    "- Keep sub-agent prompts fully self-contained (they cannot see this "
    "conversation)."
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
        "To use a skill, read its SKILL.md with the read tool "
        '(e.g. read(path="<path>")) and then follow its instructions exactly. '
        "Never assume skill content — always read SKILL.md before acting.",
        "",
        "Available skills (name + SKILL.md path):",
    ]
    for name, skill_md in skills:
        lines.append(f'  <skill name="{name}" path="{skill_md}"/>')
    return "\n".join(lines)


def _render_agents(home: str, cwd: str) -> str:
    """Merge global ~/.ezwork/AGENTS.md and project ./AGENTS.md (if present)."""
    paths = [Path(home) / ".ezwork" / "AGENTS.md", Path(cwd) / "AGENTS.md"]
    parts: list[str] = []
    for p in paths:
        try:
            if p.exists() and p.is_file():
                txt = p.read_text(encoding="utf-8").strip()
                if txt:
                    parts.append(txt)
        except OSError:
            continue
    if not parts:
        return "(no AGENTS.md)"
    return "\n\n---\n\n".join(parts)


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
