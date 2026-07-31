# Ezwork — architecture guide for contributors

Ezwork is a lean agent **kernel** plus a minimal app layer. The kernel provides
the agent loop, an OpenAI-format message model, a streaming provider protocol,
a tool registry, and a plain-text prompt framework. It ships no CLI, no
session storage, no built-in tools, no compaction, no sub-agents — those live
in the bundled `ezwork/app/` (one opinionated example) or in your own app.

## 1. Layering

```
┌─────────────────────────────────────────────────────────────┐
│  ezwork/app/  ← bundled app layer (the `ezwork` command)        │
│    config.py      Config (~/.ezwork/config.json)              │
│    session.py     Session + SessionStore                    │
│    prompt.py      build_system_prompt (plain text + XML)    │
│    cli.py         entry point: REPL + oneshot (-p/-s)       │
└──────────────────────────┬──────────────────────────────────┘
                           │ uses
┌──────────────────────────┴──────────────────────────────────┐
│  ezwork/core/      THE KERNEL (zero deps)                     │
│    agent.py       AgentLoop (the engine)                     │
│    builder.py     Agent fluent builder                       │
│    provider.py    Provider protocol + StreamChunk + Response │
│    message.py     OpenAI-format message helpers              │
│    event.py       event types (no EventBus class)            │
│    config.py      LoopConfig (limits + callback lists)       │
│    tool.py        Tool + ToolRegistry                        │
│    prompt.py      Prompt + Section (plain-text only)         │
│                                                              │
│  ezwork/providers/ OpenAIProvider + per-vendor factories      │
│    openai.py + longcat.py / deepseek.py / glm.py /           │
│               mimo.py / minimax.py                           │
│                                                              │
│  ezwork/tools/     built-in tools (read/write/edit/bash)      │
└──────────────────────────────────────────────────────────────┘
```

Import paths (public contract):

    from ezwork.core import Agent, LoopConfig, Tool, ToolRegistry, Prompt, Section
    from ezwork.tools import ReadTool, WriteTool, EditTool, BashTool
    from ezwork.providers import LongCat, DeepSeek, GLM, Mimo, MiniMax, OpenAIProvider

Dependency rule: `core/` depends on nothing outside itself (and the stdlib).
`providers/` depends on `core` + the optional `openai` package. `tools/`
depends on `core`. `app/` depends on all three and is the `ezwork` entry point.

## 2. The agent loop (`core/agent.py`)

`AgentLoop(provider, system_prompt: str, tools: ToolRegistry, config: LoopConfig)`.

`chat(user_input)` appends the user message and runs the loop:

```
while True:
    run transform_context callbacks           # in-place message mutation
    emit IterStartEvent
    response = collect_stream(provider.stream(...))   # emit StreamChunkEvent per chunk
    emit ResponseEvent
    if response.is_error:
        emit ErrorEvent; append error message; break
    if response.tool_calls:
        for each call:
            before_tool_call callbacks (veto by returning None)
            emit ToolStartEvent
            execute (timeout + truncation)
            after_tool_call callbacks (may rewrite result)
            emit ToolCompleteEvent
        append assistant(tool_uses) + tool_result messages
        emit IterEndEvent(final_content=None)
        if max_iterations reached: break
    else:
        append assistant(text) message
        emit IterEndEvent(final_content=content)
        break
```

Invariants:

- **Cancellation** propagates as `CancelledError`; the loop appends an interrupt marker first.
- **Tool errors** become `role=tool` messages with the error string — the model sees them and recovers. The loop never raises for a tool failure.
- **Provider errors** (`finish_reason="error"`) are encoded into a `Response`, surfaced via `ErrorEvent`, and stop the loop. Retries happen around `provider.stream` via `retry_stream`, governed by `LoopConfig.max_retries`.

## 3. Messages (`core/message.py`)

Plain **OpenAI Chat Completions dicts** — no competing neutral type. Every targeted provider already speaks this format.

Canonical shapes:

```python
{"role": "system",    "content": str}
{"role": "user",      "content": str | list[content_part]}
{"role": "assistant", "content": str | None,
                      "tool_calls": [{"id","type":"function",
                                      "function":{"name","arguments": json_str}}],
                      "reasoning_content": str}        # vendor extension
{"role": "tool",      "tool_call_id": str, "content": str}
```

Multimodal user content is a list of parts:
`[{"type":"text","text":...}, {"type":"image_url","image_url":{"url":...}}]`

Helpers (pure, no class hierarchy): `user_text`, `assistant_text`,
`assistant_with_tool_calls`, `tool_result`, `user_with_images`, `get_text`,
`get_tool_calls`.

**Tool-call arguments are a JSON *string* on the wire.** The loop parses to a dict exactly once before invoking the tool and re-serialises for events; history always stores the wire string. OpenAI-compatible providers pass messages through with only light sanitisation (e.g. dropping orphan tool_calls).

## 4. Providers (`core/provider.py`)

A provider implements ONE method:

```python
async def stream(self, messages, system, tools, max_tokens, *,
                 thinking=None, reasoning_effort=None) -> AsyncIterator[StreamChunk]
```

`StreamChunk` types: `text_delta`, `reasoning_delta`, `tool_call_delta`, `usage`, `done`, `error`. Errors are encoded, not raised (except `CancelledError`).

- `collect_stream(chunks) -> Response` — accumulate a stream into one `Response`.
- `retry_stream(factory, is_retriable)` — retry a stream factory on retriable errors **before** the first chunk; once a chunk is emitted, the stream is committed.
- `tools` is an OpenAI function-calling schema list — `Tool.to_schema()` produces it.

### Thinking (`ThinkingPreset`)

Each vendor has its own thinking vocabulary. The provider **hardcodes no thinking field names** — it delegates to a `ThinkingPreset`:

```python
class ThinkingPreset(Protocol):
    default_enabled: bool
    effort_levels: list[str]
    default_effort: str
    def build_params(self, enabled: bool, effort: str | None) -> ThinkingParams: ...
```

`ThinkingParams(body=..., top_params=...)` is merged into the request. The kernel defines ONLY the protocol — no presets registry. Concrete presets live in `ezwork/providers/` (one module per vendor). Switch thinking at runtime via `LoopConfig.thinking` / `LoopConfig.reasoning_effort`, or `Agent.thinking(b)` / `Agent.reasoning_effort(e)`.

## 5. LoopConfig (`core/config.py`)

Extension points, deliberately split:

**Context transforms** (ordered lists, run in registration order, sync):
- `transform_context: list[(messages) -> None]` — mutate messages in place. **Compaction lives here** (a transform that rewrites/summarises messages).
- `before_tool_call: list[(ToolUse) -> ToolUse | None]` — observe, rewrite, or veto (return `None`).
- `after_tool_call: list[(ToolUse, ToolResult) -> ToolResult]` — observe or rewrite a result.

**Output broadcast** (fan-out list, sync):
- `emit: list[(Event) -> None]` — observe events. Filter by `event.type` inside the callback, or use `Agent.on(type, cb)` for the common case.

There is intentionally **no EventBus class** — fan-out is `for cb in emit: cb(event)`. If you need subscribe/unsubscribe machinery, build it in your app on top of `emit`. All callbacks are sync; if one needs I/O it should schedule its own asyncio task without blocking.

## 6. Tools (`core/tool.py`)

`Tool(name, description, params, func)`. `params` is a simplified dict spec (not full JSON Schema); sync and async funcs both work (sync is auto-wrapped with `asyncio.to_thread`). `ToolError(message, code)` signals a handled failure. `ToolRegistry` caches generated schemas and supports `derived(exclude={...})` for filtered tool sets.

The kernel ships **no** tools. `ezwork/tools/` provides reference implementations: `ReadTool`, `WriteTool`, `EditTool`, `BashTool`. The bash tool auto-detects the best available shell (bash → sh → PowerShell → cmd) at construction and bakes the active shell's syntax hint into its description.

## 7. Prompts (`core/prompt.py`)

`Prompt` holds flat `Section`s — each with `name`, `content` (string or callable taking a context dict), `priority`, `enabled`. `build()` renders **plain text only**, sorted by `(priority, name)`. The kernel adds **no markup of its own**: each section renders as `name: content` (single-line) or `name:\n<content>` (multi-line), joined by blank lines.

There is **no** built-in system-prompt assembler. Apps compose sections themselves and embed any markup (XML, delimiters) directly in section content strings. The bundled `ezwork/app/prompt.py` does this: plain-text sections, with `<skills>` tags written by the app. Convention: stable low-priority sections first (prefix-cache friendly), dynamic high-churn sections last.

## 8. The app layer (`ezwork/app/`)

One minimal, opinionated way to run the kernel — and the `ezwork` command:

- **config.py** — single-provider config at `~/.ezwork/config.json` (provider, api_key, base_url, model, thinking, reasoning_effort, max_tokens). First run writes a template and exits gracefully if not yet filled in.
- **session.py** — `Session` (data) + `SessionStore` (JSON files under `~/.ezwork/sessions/<sha256(cwd)[:16]>/`).
- **prompt.py** — `build_system_prompt()` composes identity / guidelines (incl. bash search guidance) / environment / subagents / skills / agents.
- **cli.py** — entry point: REPL + oneshot. `ezwork -p "..."` writes the answer to stdout and `session: <id>` to stderr; `ezwork -p "..." -s <id>` continues any session. Sub-agents are just sessions — no separate subsystem.

Provider errors never crash the process: the kernel emits `ErrorEvent`, the CLI's renderer prints it, and the loop keeps going.

## 9. Python 环境管理（用户约定）

本项目所有 Python 相关操作一律用 **uv** 管理，不要用 pip / 系统 Python 直接装包：

- 运行命令/脚本：`uv run <cmd>`（如 `uv run pytest`、`uv run python ...`）
- 安装/移除依赖：`uv add <pkg>` / `uv remove <pkg>`（依赖声明在 pyproject.toml）
- 安装 CLI 工具：`uv tool install <pkg>`
- Python 版本：用 `uv python` 管理，勿手动改系统解释器

## 10. Testing

`uv run pytest`. Tests are async via `pytest-asyncio` (auto mode) and use a `MockProvider` (`tests/__init__.py`) that replays a scripted stream — no real LLM calls. Prefer testing through the public API: build an `AgentLoop` with a `MockProvider`, exercise it, assert on `agent.messages` and emitted events.

## 11. Adding things

- **Tool** — instantiate/subclass `Tool`, register into a `ToolRegistry`.
- **Provider** — OpenAI-compatible vendors need only a factory + optional `ThinkingPreset` (see [docs/providers.md](docs/providers.md)); non-compatible vendors implement the `Provider.stream()` protocol and convert messages at the boundary.
- **Thinking for a new vendor** — write a `ThinkingPreset.build_params()`, pass it to `OpenAIProvider(thinking_preset=...)` or attach via a factory in `ezwork/providers/` (existing shapes for DeepSeek/GLM/LongCat/Mimo/MiniMax live there).
- **Compaction** — append a callback to `LoopConfig.transform_context`.
- **Event subscriber** — append a callable to `LoopConfig.emit`, or use `Agent.on("tool_complete", cb)`.
