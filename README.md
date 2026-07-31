# Ezwork

A lean agent kernel with a minimal CLI app on top. The kernel is dependency-free
and ships no opinionated app; the bundled `ezwork/app/` is one small, friendly way
to run it — four built-in tools (read/write/edit/bash), JSON config, file-based
sessions, and a plain-text prompt.

Designed to be the agent you can drop onto a server in one command when you
don't want to install something heavier.

> The kernel itself (`ezwork/core/`) is dependency-free and ships no tools, no
> CLI, no session storage, no built-in compaction, no sub-agents. `tools/`,
> `providers/`, and `app/` are sibling packages you can use, replace, or ignore.

## Install

Install it as a standalone tool with `uv` — `ezwork` ends up on your PATH and
runs in its own isolated environment:

```bash
# From the GitHub repo (recommended — always the latest release):
uv tool install git+https://github.com/Shingwha/ezwork.git

# Or from a local clone (for development):
uv tool install -e ".[dev]"       # editable install + pytest, for contributors
```

`openai` is a required dependency, so a plain install gives you a working agent
out of the box — no extra step to enable the providers. After install, the
`ezwork` command is available everywhere; just run it — it writes a config
template on first run.

> Upgrading later is just `uv tool upgrade ezwork` (or
> `uv tool upgrade ezwork --reinstall` to force).

## Try the CLI

```bash
ezwork                                # first run: writes ~/.ezwork/config.json, then exits
$EDITOR ~/.ezwork/config.json         # fill in provider + api_key + model
ezwork                                # interactive REPL
ezwork -p "list the python files in this dir"        # one-shot, prints answer
ezwork -p "continue the refactor" -s session_xxxxx   # continue a prior session
cat file.txt | ezwork -p "summarize"                 # piped content = context
git diff | ezwork -p "write a commit message"        # same, for diffs
git diff | ezwork -p -                               # `-p -`: stdin is the whole prompt
```

`~/.ezwork/config.json` (single provider):

```json
{
  "provider": "longcat",
  "api_key": "ak-...",
  "base_url": "",
  "model": "LongCat-2.0",
  "thinking": true,
  "reasoning_effort": "",
  "max_tokens": 32768
}
```

`provider` can be `longcat`, `deepseek`, `glm`, `mimo`, `minimax`, or `openai`.
`reasoning_effort` is a string (e.g. `"high"` / `"max"`); leave it empty for the
vendor default. Only some vendors honour it (DeepSeek, GLM). Edit the file and
restart to switch.

REPL commands: `/exit` `/clear` `/help` `/sessions [N]` `/resume [id]`. Each
`/sessions` row shows id, update time, and a short content preview (default 10
rows; `/sessions 20` shows 20). Model and thinking are configured in
`~/.ezwork/config.json` (the agent can edit it itself via the file tools, then you
restart). Streaming text prints live; tool calls show as `-> name(args)` /
`ok result` (or `x error`).

### One-shot is script-friendly

`ezwork -p` writes the answer to **stdout**; the session id + token usage go to
**stderr**, printed as a `session: <id>` line. No parsing needed — read the id
straight off the stderr and pass it to `-s`:

```bash
ezwork -p "summarize this repo"        # stderr → session: session_xxxxx
ezwork -p "now list its open todos" -s session_xxxxx
```

**Piped stdin is appended as context** (the same convention as
`cat file | claude -p "query"`), so you can hand the agent data without
writing it to a file or stuffing it into the argument:

```bash
cat logs.txt | ezwork -p "explain the errors"
git diff | ezwork -p "summarize this diff"
curl -s https://example.com/api | ezwork -p "summarize the response"
```

Use `-p -` when the piped content *is* the whole prompt (read like `cat -`,
interactively until Ctrl-D on a TTY); empty stdin exits with code 2 rather
than sending an empty prompt.

### Sub-agents are just sessions

There is no separate sub-agent subsystem. Every `-p` run is an isolated session;
any session can be continued with `-s <id>`. To run a sub-agent from a parent
workflow, spin up a one-shot session and keep its id — same config, same tools,
isolated history:

```bash
ezwork -p "analyse the auth module"    # stderr → session: session_xxxxx
ezwork -p "now write tests for it" -s session_xxxxx
```

To hand the sub-agent data without a file round-trip, pipe it — piped stdin
is appended as context:

```bash
cat src/auth/models.py | ezwork -p "list the public functions and their tests"
git diff | ezwork -p "propose a commit message for this diff"
```

**Prompting rules for sub-agents** — a sub-agent cannot see the parent
conversation, so every sub-agent prompt must be a fully self-contained
specification:

1. **Goal** — what needs to be done.
2. **Output format** — what the final answer should look like (e.g. "list the
   files", "print the diff", "write a summary").
3. **Workflow** — the steps to follow, especially order of operations.
4. **Tool restrictions** — which tools to use or avoid (e.g. "read-only, do
   not modify any files", "use grep for search, do not read files one by one",
   "use the edit tool, do not rewrite whole files").

**Output rules** — the sub-agent's output is the *only* thing the parent sees,
so it must be complete and self-explanatory: structure it with headings, lists,
code blocks or tables; always summarise what was done at the end; say so
explicitly if a step failed or was skipped; and when reporting file changes,
include the path and a brief description of each change.

Good example:

```bash
ezwork -p "Analyse the auth module in src/auth/.
1. Read src/auth/__init__.py and list all public functions.
2. For each function, check if it has a unit test in tests/test_auth.py.
3. Report any function that lacks a test, using this format:
   [missing-test] <function_name> — <reason>
Do not modify any files."
```

If the task is complex, break it into multiple sub-agent calls rather than one
giant prompt, and use follow-up calls (`-s`) to build on previous results.

## Hello agent (no CLI)

```python
import asyncio
from ezwork.core import Agent
from ezwork.providers import LongCat       # or DeepSeek / GLM / Mimo / MiniMax
from ezwork.tools import ReadTool, BashTool

async def main():
    agent = (
        Agent()
        .provider(LongCat(api_key="ak-..."))
        .prompt("You are a concise assistant.")
        .tools([ReadTool(), BashTool()])
        .thinking(True)
        .build())
    print(await agent.chat("what files are in the current directory?"))

asyncio.run(main())
```

## Thinking models

Each vendor has its own thinking API shape. Built-in providers ship the right
shape out of the box — just toggle thinking on:

```python
from ezwork.providers import DeepSeek
from ezwork.core import Agent

provider = DeepSeek(api_key="sk-...")   # base_url + model + preset pre-wired
agent = (
    Agent()
    .provider(provider)
    .prompt("...")
    .thinking(True)               # toggle extended thinking
    .reasoning_effort("max")      # vendor-specific effort (DeepSeek: high|max)
    .build()
)
```

Built-in providers: `LongCat`, `DeepSeek`, `GLM`, `Mimo`, `MiniMax`.
Writing your own? See **[docs/providers.md](docs/providers.md)**.

## Architecture in one screen

```
            ┌────────── LoopConfig ──────────┐
            │  limits + thinking control     │
AgentLoop ──┤  transform_context  (compact)  │
  messages  │  before_tool_call   (veto)     │
  in/out    │  after_tool_call    (rewrite)  │
  (OpenAI   │  emit               (UI/log)  │
   dicts)   └────────────────────────────────┘
                       │  uses
              ┌────────┴────────┐
              ▼                 ▼
        Provider         ToolRegistry
   (stream only,         (OpenAI function
    + ThinkingPreset)     schema)
              │
       messages stay OpenAI dict; providers pass through
       (or convert at the boundary for non-compatible APIs)
```

See `AGENTS.md` for the full design.

## Layout

One top-level package, four subpackages:

```
ezwork/
├── core/        # the kernel (zero deps): agent, provider protocol, message,
│                # event, config, tool base, prompt (plain-text only), builder
├── providers/   # OpenAIProvider + per-vendor modules (LongCat/DeepSeek/GLM/...)
│                # each bundles preset + base_url + default model + factory
├── tools/       # read/write/edit/bash built-in tools
└── app/         # minimal app layer: config, session, prompt assembly, CLI
                 # (the `ezwork` entry point lives at ezwork.app.cli:main)
docs/            # guides — see docs/providers.md for adding providers
tests/           # mock-provider driven
```

Import paths:

```python
from ezwork.core import Agent, LoopConfig, Tool, ToolRegistry, Prompt, Section
from ezwork.tools import ReadTool, WriteTool, EditTool, BashTool
from ezwork.providers import LongCat, DeepSeek, OpenAIProvider, ...
```

### The prompt is plain text

The kernel's `Prompt.build()` renders **plain text only** — it adds no tags of
its own. If an app wants XML or any markup, it embeds it directly in the section
content strings. The bundled `ezwork/app/prompt.py` does exactly this: its
sections are plain text, but the tools/skills sections contain `<tool>` /
`<skills>` tags written by the app. Swap `app/prompt.py` to change the style
without touching the kernel.

## License

MIT.
