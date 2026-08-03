"""Session export — Markdown and JSON dumps of a conversation.

Used by the `/export` command. JSON export is the portable format (loadable
again with SessionStore), Markdown is for humans.
"""

from __future__ import annotations

from pathlib import Path

from .utils import write_json


def _content_text(content) -> str:
    """OpenAI message content (str or list of parts) -> plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                if p.get("type") == "text":
                    parts.append(str(p.get("text", "")))
                elif p.get("type") == "image_url":
                    parts.append("[image]")
        return "\n".join(parts)
    return str(content)


def _rounds_table(session) -> str:
    """Per-round summary table (usage, tools, outcome) for the Markdown export."""
    if not session.rounds:
        return ""
    rows = ["| round | time | msgs | tokens (P/C) | tool calls | failures | outcome |",
            "| ----- | ---- | ---- | ------------ | ---------- | -------- | -------- |"]
    for r in session.rounds:
        usage = r.get("usage_total") or {}
        calls = ", ".join(f"{k}×{v}" for k, v in (r.get("tool_calls") or {}).items()) or "-"
        fails = ", ".join(f"{k}×{v}" for k, v in (r.get("tool_failures") or {}).items()) or "-"
        if r.get("error"):
            outcome = f"error: {r['error'][:60]}"
        elif r.get("interrupted"):
            outcome = "interrupted"
        else:
            outcome = "ok"
        rows.append(
            f"| {r.get('round', '?')} | {str(r.get('updated_at', ''))[:16]} "
            f"| {r.get('message_count', '?')} | {usage.get('prompt_tokens', '?')}/{usage.get('completion_tokens', '?')} "
            f"| {calls} | {fails} | {outcome} |"
        )
    return "\n".join(rows)


def render_session_md(session, system_prompt: str = "") -> str:
    """Render a session as Markdown with YAML frontmatter."""
    u = session.usage_total
    tok = f"{u.get('prompt_tokens', '?')} prompt / {u.get('completion_tokens', '?')} completion"
    tools = ", ".join(f"{k}×{v}" for k, v in session.tool_stats.items()) or "-"
    lines = [
        "---",
        f"id: {session.id}",
        f"created_at: {session.created_at}",
        f"updated_at: {session.updated_at}",
        f"workdir: {session.workdir}",
        f"model: {session.model}",
        f"provider: {session.provider}",
        f"messages: {len(session.messages)}",
        f"rounds: {len(session.rounds)}",
        f"tokens: {tok}",
        f"tool calls: {tools}",
        "---",
        "",
    ]
    if system_prompt:
        lines += ["## System prompt", "", "```text", system_prompt, "```", ""]

    table = _rounds_table(session)
    if table:
        lines += ["## Per-round summary", "", table, ""]

    for msg in session.messages:
        role = msg.get("role")
        if role == "system":
            continue
        if role == "user":
            lines += ["## User", "", _content_text(msg.get("content", "")), ""]
        elif role == "assistant":
            if msg.get("reasoning_content"):
                lines += [
                    "<details><summary>Reasoning</summary>",
                    "",
                    _content_text(msg["reasoning_content"]),
                    "",
                    "</details>",
                    "",
                ]
            content = msg.get("content")
            tcs = msg.get("tool_calls")
            if content and not tcs:
                lines += ["## Assistant", "", _content_text(content), ""]
            elif tcs:
                for tc in tcs:
                    fn = tc.get("function", {})
                    lines += [
                        f"### Tool: {fn.get('name', '?')}",
                        "",
                        "```json",
                        fn.get("arguments", ""),
                        "```",
                        "",
                    ]
        elif role == "tool":
            lines += ["### Tool result", "", "```text", _content_text(msg.get("content", "")), "```", ""]

    return "\n".join(lines).rstrip() + "\n"


def export_session(
    session,
    path: Path | str,
    *,
    fmt: str = "json",
    system_prompt: str = "",
) -> Path:
    """Export a session to a file ('json' or 'md'). Returns the written path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "md":
        p.write_text(render_session_md(session, system_prompt), encoding="utf-8")
    else:
        data = session.to_dict()
        if system_prompt:  # override the latest-round prompt for the export
            data["system_prompt"] = system_prompt
        write_json(p, data)
    return p


__all__ = ["render_session_md", "export_session"]
