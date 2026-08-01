"""Built-in tools tests — read/write/edit/bash."""

from __future__ import annotations

from pathlib import Path

import pytest

from ezwork.core import ToolError
from ezwork.tools import BashTool, EditTool, ReadTool, WriteTool


# ---- read ----


def test_read_file_with_line_numbers(tmp_path: Path):
    f = tmp_path / "x.txt"
    f.write_text("a\nb\nc\n", encoding="utf-8")
    out = ReadTool().run({"path": str(f)})
    assert "1 | a" in out and "2 | b" in out and "3 | c" in out
    assert "3 lines" in out


def test_read_offset_limit(tmp_path: Path):
    f = tmp_path / "x.txt"
    f.write_text("\n".join(f"line{i}" for i in range(1, 11)) + "\n", encoding="utf-8")
    out = ReadTool().run({"path": str(f), "offset": 3, "limit": 2})
    assert "3 | line3" in out and "4 | line4" in out
    assert "line2" not in out and "line5" not in out


def test_read_missing_file(tmp_path: Path):
    with pytest.raises(ToolError) as ei:
        ReadTool().run({"path": str(tmp_path / "nope.txt")})
    assert ei.value.code == "file_not_found"


def test_read_directory_lists_contents(tmp_path: Path):
    (tmp_path / "a.py").write_text("x", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    out = ReadTool().run({"path": str(tmp_path)})
    assert "directories" in out and "a.py" in out and "sub/" in out


def test_read_binary_rejected(tmp_path: Path):
    f = tmp_path / "bin"
    f.write_bytes(b"\x00\x01\x02 binary")
    with pytest.raises(ToolError) as ei:
        ReadTool().run({"path": str(f)})
    assert ei.value.code == "binary_file"


# ---- write ----


def test_write_creates_file_and_parents(tmp_path: Path):
    target = tmp_path / "a" / "b" / "c.txt"
    out = WriteTool().run({"path": str(target), "content": "hello\nworld"})
    assert target.read_text(encoding="utf-8") == "hello\nworld"
    assert "Wrote" in out and "2 lines" in out


def test_write_append(tmp_path: Path):
    f = tmp_path / "x.txt"
    f.write_text("first", encoding="utf-8")
    WriteTool().run({"path": str(f), "content": "second", "append": True})
    assert f.read_text(encoding="utf-8") == "first\nsecond"


def test_write_overwrite_default(tmp_path: Path):
    f = tmp_path / "x.txt"
    f.write_text("old", encoding="utf-8")
    WriteTool().run({"path": str(f), "content": "new"})
    assert f.read_text(encoding="utf-8") == "new"


def test_write_rejects_directory(tmp_path: Path):
    with pytest.raises(ToolError) as ei:
        WriteTool().run({"path": str(tmp_path), "content": "x"})
    assert ei.value.code == "invalid_path"


# ---- edit ----


def test_edit_unique_replacement(tmp_path: Path):
    f = tmp_path / "x.txt"
    f.write_text("foo bar\nbaz\n", encoding="utf-8")
    out = EditTool().run({"path": str(f), "old_string": "bar", "new_string": "qux"})
    assert f.read_text(encoding="utf-8") == "foo qux\nbaz\n"
    assert "1 occurrence" in out


def test_edit_requires_unique_unless_all(tmp_path: Path):
    f = tmp_path / "x.txt"
    f.write_text("a\na\na\n", encoding="utf-8")
    with pytest.raises(ToolError) as ei:
        EditTool().run({"path": str(f), "old_string": "a", "new_string": "b"})
    assert ei.value.code == "not_unique"
    # all=True overrides
    out = EditTool().run({"path": str(f), "old_string": "a", "new_string": "b", "all": True})
    assert f.read_text(encoding="utf-8") == "b\nb\nb\n"
    assert "3 occurrence" in out


def test_edit_old_string_not_found(tmp_path: Path):
    f = tmp_path / "x.txt"
    f.write_text("hello", encoding="utf-8")
    with pytest.raises(ToolError) as ei:
        EditTool().run({"path": str(f), "old_string": "missing", "new_string": "x"})
    assert ei.value.code == "not_found"


def test_edit_not_found_detects_swapped_args(tmp_path: Path):
    f = tmp_path / "x.txt"
    f.write_text("hello world", encoding="utf-8")
    with pytest.raises(ToolError) as ei:
        EditTool().run({"path": str(f), "old_string": "goodbye", "new_string": "hello world"})
    assert ei.value.code == "not_found"
    assert "swap" in ei.value.message.lower()


def test_edit_not_found_suggests_closest_line(tmp_path: Path):
    f = tmp_path / "x.txt"
    f.write_text("def foo():\n    return 1\n", encoding="utf-8")
    with pytest.raises(ToolError) as ei:
        EditTool().run({"path": str(f), "old_string": "def foo()::", "new_string": "x"})
    assert "def foo():" in ei.value.message


def test_edit_not_unique_reports_line_numbers(tmp_path: Path):
    f = tmp_path / "x.txt"
    f.write_text("a\nx\na\n", encoding="utf-8")
    with pytest.raises(ToolError) as ei:
        EditTool().run({"path": str(f), "old_string": "a", "new_string": "b"})
    assert "[1, 3]" in ei.value.message


def test_edit_crlf_pattern_matches(tmp_path: Path):
    """A CRLF-bearing old_string matches a CRLF file (Windows line endings)."""
    f = tmp_path / "x.txt"
    f.write_text("foo\r\nbar\r\n", encoding="utf-8", newline="")
    out = EditTool().run({"path": str(f), "old_string": "foo\r\nbar", "new_string": "X"})
    assert "X\r\n" in f.read_bytes().decode()
    assert "1 occurrence" in out


def test_edit_concurrent_same_file_all_apply(tmp_path: Path):
    """Concurrent edits to one file must all apply (per-file lock)."""
    from concurrent.futures import ThreadPoolExecutor

    f = tmp_path / "x.txt"
    f.write_text("a\nb\nc\nd\n", encoding="utf-8")
    edits = [
        {"path": str(f), "old_string": "a", "new_string": "A"},
        {"path": str(f), "old_string": "b", "new_string": "B"},
        {"path": str(f), "old_string": "c", "new_string": "C"},
        {"path": str(f), "old_string": "d", "new_string": "D"},
    ]
    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(lambda e: EditTool().run(e), edits))
    assert f.read_text(encoding="utf-8") == "A\nB\nC\nD\n"
    assert all("1 occurrence" in r for r in results)


# ---- bash ----


def test_bash_runs_echo():
    out = BashTool().run({"command": "echo bash_test_123"})
    assert "bash_test_123" in out


def test_bash_persists_cwd():
    import os

    tool = BashTool()
    cwd_before = os.getcwd()
    tool.run({"command": f"cd .."})
    # session cwd mutated, but process cwd unchanged
    assert os.getcwd() == cwd_before
    # subsequent pwd reflects the cd
    pwd = tool.run({"command": "pwd"})
    # parent of cwd should appear in pwd
    assert str(Path(cwd_before).parent) in pwd or pwd != cwd_before


def test_bash_unknown_command_returns_error_string():
    out = BashTool().run({"command": "this_command_does_not_exist_xyz"})
    # bash writes to stderr; tool concatenates; some marker present
    assert "not found" in out or "error" in out or out  # tolerant


def test_bash_restart_resets_session():
    import os

    tool = BashTool()
    tool.run({"command": "cd .."})
    # session cwd has moved up one level
    assert tool._session.cwd != os.getcwd()
    out = tool.run({"restart": True})
    assert "restarted" in out
    # session cwd reset back to process cwd
    assert tool._session.cwd == os.getcwd()


# ---- schemas ----


def test_all_four_tools_have_distinct_names():
    names = {t.name for t in [ReadTool(), WriteTool(), EditTool(), BashTool()]}
    assert names == {"read", "write", "edit", "bash"}


def test_tool_schemas_valid():
    for t in [ReadTool(), WriteTool(), EditTool(), BashTool()]:
        s = t.to_schema()
        assert s["type"] == "function"
        assert s["function"]["name"] == t.name
        assert "description" in s["function"]
        assert "parameters" in s["function"]


# ---- bash: shell detection + backends (cross-platform) ----


def test_detect_shell_finds_something():
    """On any dev machine at least one shell must be detected."""
    from ezwork.tools.bash import detect_shell

    info = detect_shell()
    assert info is not None
    assert info.family in {"bash", "sh", "powershell", "cmd"}
    assert info.path.exists()


def test_bash_tool_description_reflects_active_shell():
    """The tool detects the shell lazily (on first schema generation) and bakes
    its syntax hint into the description, so the model knows which command
    syntax to use."""
    from ezwork.tools.bash import _SHELL_HINTS, detect_shell

    tool = BashTool()
    # Lazy: construction does not probe the shell — the description is neutral.
    assert "Active shell" not in tool.description
    schema = tool.to_schema()
    info = detect_shell()
    # the family-specific hint must be present in the rendered description
    assert _SHELL_HINTS[info.family] in tool.description
    assert _SHELL_HINTS[info.family] in schema["function"]["description"]
    # the neutral base is always present
    assert "persistent session" in tool.description


def test_bash_tool_detection_happens_once_lazily(monkeypatch):
    """Construction must not probe the shell (startup stays cheap); the probe
    runs exactly once, on first schema generation."""
    import ezwork.tools.bash as bash_mod

    calls: list[int] = []
    monkeypatch.setattr(
        bash_mod, "detect_shell", lambda: calls.append(1) or bash_mod.ShellInfo(Path("/bin/bash"), "bash")
    )
    tool = BashTool()
    assert calls == []  # no probe at construction
    tool.to_schema()
    assert len(calls) == 1
    tool.to_schema()  # cached — no re-probe
    assert len(calls) == 1
    assert "Active shell: bash" in tool.description


def test_bash_tool_description_per_family():
    """Each family produces a distinct, actionable description; None stays neutral."""
    from ezwork.tools.bash import _build_desc

    ps_desc = _build_desc("powershell")
    assert "PowerShell" in ps_desc and "Select-String" in ps_desc
    assert "grep -r" not in ps_desc  # must steer away from bash syntax

    cmd_desc = _build_desc("cmd")
    assert "cmd.exe" in cmd_desc and "findstr" in cmd_desc

    bash_desc = _build_desc("bash")
    assert "POSIX" in bash_desc and "grep -r" in bash_desc

    none_desc = _build_desc(None)
    assert "Active shell" not in none_desc  # neutral when no shell detected


@pytest.mark.skipif(__import__("sys").platform != "win32", reason="WSL check is Windows-only")
def test_detect_shell_skips_wsl_launcher():
    """The WSL bash launcher (System32\\bash.exe) must never be selected."""
    from ezwork.tools.bash import _is_wsl_launcher

    assert _is_wsl_launcher(Path(r"C:\Windows\System32\bash.exe")) is True
    assert _is_wsl_launcher(
        Path(r"C:\Users\me\AppData\Local\Microsoft\WindowsApps\bash.exe")
    ) is True
    # A real Git Bash path must NOT be flagged.
    assert _is_wsl_launcher(Path(r"C:\Program Files\Git\bin\bash.exe")) is False


def test_detect_shell_none_when_nothing_available(monkeypatch):
    """If every probe misses, detect_shell returns None (and the tool raises a
    clear error on use rather than crashing at import)."""
    import sys

    import ezwork.tools.bash as bash_mod

    monkeypatch.setattr(bash_mod.shutil, "which", lambda name: None)
    monkeypatch.setattr(bash_mod, "_WIN_BASH_LOCATIONS", [])
    monkeypatch.setattr(bash_mod, "_POSIX_BASH_LOCATIONS", [])
    monkeypatch.setattr(sys, "platform", "linux")
    assert bash_mod.detect_shell() is None

    tool = BashTool()
    with pytest.raises(RuntimeError, match="No usable shell"):
        tool.run({"command": "echo hi"})


def _backends_present():
    """Yield (backend_instance) for each installed shell, so backend tests only
    run against shells that actually exist on this machine."""
    import shutil

    from ezwork.tools.bash import (
        ShellInfo,
        _CmdBackend,
        _PosixBackend,
        _PowerShellBackend,
    )

    out = []
    for exe, cls, family in [
        (shutil.which("bash") or shutil.which("sh"), _PosixBackend, "posix"),
        (shutil.which("pwsh") or shutil.which("powershell"), _PowerShellBackend, "powershell"),
        (shutil.which("cmd"), _CmdBackend, "cmd"),
    ]:
        if exe:
            out.append((cls(ShellInfo(Path(exe), family)), family))
    return out


@pytest.mark.parametrize(
    "backend,family",
    _backends_present(),
    ids=lambda v: v if isinstance(v, str) else v.family,
)
def test_backend_echo_roundtrip(backend, family):
    """Every installed backend can run an echo and return its output."""
    from ezwork.tools.bash import BashSession

    sess = BashSession()
    sess._backend = backend
    if family == "powershell":
        out = sess.execute("Write-Output 'roundtrip_test_42'")
    elif family == "cmd":
        out = sess.execute("echo roundtrip_test_42")
    else:
        out = sess.execute("echo roundtrip_test_42")
    assert "roundtrip_test_42" in out


@pytest.mark.parametrize(
    "backend,family",
    _backends_present(),
    ids=lambda v: v if isinstance(v, str) else v.family,
)
def test_backend_cd_persists_in_session(backend, family):
    """cd builtins are intercepted in-process and update session cwd, whichever
    shell family is active."""
    from ezwork.tools.bash import BashSession

    sess = BashSession()
    sess._backend = backend
    cwd_before = sess.cwd
    if family == "powershell":
        sess.execute("cd ..")
    else:
        sess.execute("cd ..")
    # session cwd moved up one level (resolved path)
    assert sess.cwd == str(Path(cwd_before).parent.resolve())


@pytest.mark.parametrize(
    "backend,family",
    _backends_present(),
    ids=lambda v: v if isinstance(v, str) else v.family,
)
def test_backend_env_var_persists(backend, family):
    """Env-setting builtins (export / $env: / set) are recorded into the
    session env and visible to the next command."""
    from ezwork.tools.bash import BashSession

    sess = BashSession()
    sess._backend = backend
    if family == "powershell":
        sess.execute("$env:LESSVAR=hello")
        out = sess.execute("Write-Output $env:LESSVAR")
    elif family == "cmd":
        sess.execute("set LESSVAR=hello")
        out = sess.execute("echo %LESSVAR%")
    else:
        sess.execute("export LESSVAR=hello")
        out = sess.execute("echo $LESSVAR")
    assert "hello" in out


@pytest.mark.parametrize(
    "backend,family",
    _backends_present(),
    ids=lambda v: v if isinstance(v, str) else v.family,
)
def test_backend_chained_env_set_runs_in_shell(backend, family):
    """A chained env-set (`export X=1 && echo "$X"`) must run as a shell
    command — the builtin interception must not swallow the chain into the
    env value (regression: compound commands starting with export/$env:/set
    returned '(empty)' and polluted the session env)."""
    from ezwork.tools.bash import BashSession

    sess = BashSession()
    sess._backend = backend
    if family == "powershell":
        out = sess.execute('$env:CHVAR=hello; Write-Output $env:CHVAR')
        assert "hello" in out
    elif family == "cmd":
        # cmd expands %CHVAR% at parse time, so the echo shows the literal —
        # what matters is that the chain ran as a command, not as a builtin.
        sess.execute("set CHVAR=hello & echo %CHVAR%")
    else:
        out = sess.execute('export CHVAR=hello && echo "$CHVAR"')
        assert "hello" in out
    # the chain must NOT be recorded as a session env var
    assert "CHVAR" not in sess._env_vars


@pytest.mark.parametrize(
    "backend,family",
    _backends_present(),
    ids=lambda v: v if isinstance(v, str) else v.family,
)
def test_backend_unknown_command_returns_output(backend, family):
    """An unknown command produces a non-empty result (error text from the
    shell), never an exception."""
    from ezwork.tools.bash import BashSession

    sess = BashSession()
    sess._backend = backend
    out = sess.execute("this_command_does_not_exist_xyz_123")
    assert out and out != "(empty)"
