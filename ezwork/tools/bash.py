"""Bash tool — persistent shell session, cross-platform.

The tool is named `bash` and the agent is encouraged to write POSIX-style
commands, but it actually runs on whatever the best available shell is:

    bash -> sh -> PowerShell (pwsh / powershell) -> cmd

Detection happens once, lazily, on first use. Where a Unix shell exists
(Linux, macOS, Git Bash on Windows, MSYS2, Cygwin) behaviour is unchanged from
a real bash session. Where none exists (a bare Windows server) it falls back to
PowerShell, then cmd, so the tool still works instead of dying.

Session state (working directory, environment variables) persists across
commands the same way on every shell family: each command is run as a fresh
subprocess with the remembered cwd/env, and bare `cd` / env-setting builtins
(`export`, `$env:X=`, `set X=`) are intercepted in-process so the state change
actually sticks. No long-lived child process — simpler and uniform across
shells.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ezwork.core import Tool, ToolError

from .utils import decode_bytes


# ─── shell detection ───────────────────────────────────────────────────────


@dataclass
class ShellInfo:
    """A detected shell: its executable path and a family tag."""

    path: Path
    family: str  # "bash" | "sh" | "powershell" | "cmd"


# Known Git Bash / MSYS2 / Cygwin install locations on Windows. Checked first
# because PATH-order `bash` on Windows can resolve to the WSL launcher.
_WIN_BASH_LOCATIONS = [
    Path(r"C:\Program Files\Git\bin\bash.exe"),
    Path(r"C:\Program Files (x86)\Git\bin\bash.exe"),
    Path(r"C:\Git\bin\bash.exe"),
    Path.home() / "AppData" / "Local" / "Programs" / "Git" / "bin" / "bash.exe",
    Path.home() / "scoop" / "apps" / "git" / "current" / "bin" / "bash.exe",
    Path(r"C:\msys64\usr\bin\bash.exe"),
    Path(r"C:\cygwin64\bin\bash.exe"),
    Path(r"C:\cygwin\bin\bash.exe"),
]

_POSIX_BASH_LOCATIONS = [
    Path("/bin/bash"),
    Path("/usr/bin/bash"),
    Path("/usr/local/bin/bash"),
    Path("/opt/homebrew/bin/bash"),
]


def _is_wsl_launcher(path: Path) -> bool:
    """True only for the WSL bash launcher — `System32\\bash.exe` or a
    `WindowsApps\\bash.exe` shim. Running it would drop us into a Linux
    filesystem view with path mismatches, so we skip it. Checked precisely via
    path parts rather than a substring so legitimate `system32`-adjacent paths
    aren't wrongly rejected."""
    parts = {p.lower() for p in path.parts}
    if "system32" in parts and path.name.lower() == "bash.exe":
        return True
    if "windowsapps" in parts and path.name.lower() == "bash.exe":
        return True
    return False


def _which(name: str) -> Optional[Path]:
    found = shutil.which(name)
    return Path(found) if found else None


def detect_shell() -> Optional[ShellInfo]:
    """Locate the best available shell for this platform.

    Order: bash -> sh -> pwsh -> powershell -> cmd (Windows only).
    Returns None if nothing usable is found; the caller raises a clear error.
    """
    if sys.platform == "win32":
        # 1. Known Git Bash / MSYS2 / Cygwin locations (most reliable on Windows).
        for p in _WIN_BASH_LOCATIONS:
            if p.exists():
                return ShellInfo(p, "bash")
        # 2. bash on PATH — but never the WSL launcher.
        b = _which("bash")
        if b and not _is_wsl_launcher(b):
            return ShellInfo(b, "bash")
        # 3. sh on PATH (Git Bash also ships sh).
        s = _which("sh")
        if s:
            return ShellInfo(s, "sh")
    else:
        b = _which("bash") or next((p for p in _POSIX_BASH_LOCATIONS if p.exists()), None)
        if b:
            return ShellInfo(b, "bash")
        s = _which("sh")
        if s:
            return ShellInfo(s, "sh")

    # 4. PowerShell Core (cross-platform), then Windows PowerShell, then cmd.
    pwsh = _which("pwsh")
    if pwsh:
        return ShellInfo(pwsh, "powershell")
    if sys.platform == "win32":
        ps = _which("powershell")
        if ps:
            return ShellInfo(ps, "powershell")
        cmd = _which("cmd")
        if cmd:
            return ShellInfo(cmd, "cmd")
    return None


# ─── shell backends ────────────────────────────────────────────────────────
#
# Each backend knows how to (a) run one command as a subprocess and (b)
# recognise its own "change directory" / "set environment variable" builtins so
# they can be applied to the persistent session state in-process. Backends
# return (stdout, stderr, returncode); the session merges and formats.
#
# `handle_builtin` returns a result string when it fully handled the command
# (cd / env-set), or None to mean "run it as a normal command".


class _ShellBackend:
    """Base shell backend. Subclasses set `family` and implement the hooks."""

    family: str = ""

    def __init__(self, info: ShellInfo) -> None:
        self.path = info.path

    def execute(
        self, command: str, *, timeout: int, cwd: str, env: dict
    ) -> tuple[bytes, bytes, int]:
        raise NotImplementedError

    def handle_builtin(self, command: str, cwd: str, env_vars: dict) -> Optional[tuple[str, str]]:
        """Return (result_string, new_cwd) if this is a cd/env builtin we handle
        in-process; None otherwise. May mutate env_vars in place."""
        raise NotImplementedError


class _PosixBackend(_ShellBackend):
    """bash / sh — POSIX syntax. Runs `<exe> -c <cmd>`."""

    family = "posix"

    def execute(self, command, *, timeout, cwd, env):
        proc = subprocess.run(
            [str(self.path), "-c", command],
            capture_output=True, timeout=timeout, cwd=cwd, env=env,
        )
        return proc.stdout, proc.stderr, proc.returncode

    def handle_builtin(self, command, cwd, env_vars):
        stripped = command.strip()
        # Bare `cd <path>` (no chaining) — handle in-process so cwd persists.
        if stripped.startswith("cd ") and "&&" not in stripped and ";" not in stripped and "|" not in stripped:
            target = stripped[3:].strip()
            new_cwd = _resolve_cd(target, cwd)
            if new_cwd is None:
                return (f"bash: cd: {target}: No such file or directory", cwd)
            return (new_cwd, new_cwd)
        # `export VAR=value` — record into the session env.
        if stripped.startswith("export ") and "=" in stripped:
            expr = stripped[7:]
            key, value = expr.split("=", 1)
            env_vars[key.strip()] = value.strip().strip("\"'")
            return ("", cwd)
        return None


class _PowerShellBackend(_ShellBackend):
    """PowerShell (pwsh or Windows PowerShell). Runs with -NoProfile -Command."""

    family = "powershell"

    def execute(self, command, *, timeout, cwd, env):
        proc = subprocess.run(
            [str(self.path), "-NoProfile", "-NoLogo", "-Command", command],
            capture_output=True, timeout=timeout, cwd=cwd, env=env,
        )
        return proc.stdout, proc.stderr, proc.returncode

    def handle_builtin(self, command, cwd, env_vars):
        stripped = command.strip()
        # `cd <path>` or `Set-Location <path>` (case-insensitive), bare form.
        m = re.match(r"(?i)^(?:cd|set-location|sl)\s+(.+)$", stripped)
        if m and ";" not in stripped and "|" not in stripped:
            target = m.group(1).strip().strip("'\"")
            new_cwd = _resolve_cd(target, cwd)
            if new_cwd is None:
                return (f"Set-Location: Cannot find path '{target}'.", cwd)
            return (new_cwd, new_cwd)
        # `$env:VAR=value` — record into the session env.
        m = re.match(r"(?i)^\$env:(\w+)\s*=\s*(.+)$", stripped)
        if m:
            env_vars[m.group(1)] = m.group(2).strip().strip("'\"")
            return ("", cwd)
        return None


class _CmdBackend(_ShellBackend):
    """cmd.exe — last resort on Windows. Limited but functional."""

    family = "cmd"

    def execute(self, command, *, timeout, cwd, env):
        proc = subprocess.run(
            [str(self.path), "/c", command],
            capture_output=True, timeout=timeout, cwd=cwd, env=env,
        )
        return proc.stdout, proc.stderr, proc.returncode

    def handle_builtin(self, command, cwd, env_vars):
        stripped = command.strip()
        # `cd <path>` (cmd accepts bare cd).
        if stripped.lower().startswith("cd ") and "&" not in stripped and "|" not in stripped:
            target = stripped[3:].strip().strip("\"")
            new_cwd = _resolve_cd(target, cwd)
            if new_cwd is None:
                return (f"The system cannot find the path specified: {target}", cwd)
            return (new_cwd, new_cwd)
        # `set VAR=value` — record into the session env.
        m = re.match(r"(?i)^set\s+(\w+)\s*=\s*(.*)$", stripped)
        if m:
            env_vars[m.group(1)] = m.group(2).strip().strip("\"")
            return ("", cwd)
        return None


_BACKENDS = {
    "bash": _PosixBackend,
    "sh": _PosixBackend,
    "powershell": _PowerShellBackend,
    "cmd": _CmdBackend,
}


def _resolve_cd(target: str, cwd: str) -> Optional[str]:
    """Resolve a cd target against cwd, expanding `~`. Returns the absolute
    path if it's an existing directory, else None."""
    if target.startswith("~"):
        target = str(Path.home()) + target[1:]
    p = Path(target)
    new_path = p if p.is_absolute() else (Path(cwd) / p)
    new_path = new_path.resolve()
    if new_path.exists() and new_path.is_dir():
        return str(new_path)
    return None


# ─── session ───────────────────────────────────────────────────────────────


class BashSession:
    """Persistent shell session — keeps cwd and env vars across commands.

    Shell-agnostic: the detected backend decides how commands run and which
    cd/env builtins are intercepted in-process. Each command is a fresh
    subprocess; state lives here, not in a long-lived child process.

    Pass `shell` to reuse an already-detected ShellInfo (e.g. one the tool
    resolved up front so its description matches the shell actually in use).
    Otherwise detection happens lazily on first use.
    """

    def __init__(self, shell: Optional[ShellInfo] = None) -> None:
        self._backend: Optional[_ShellBackend] = None
        if shell is not None:
            self._backend = _BACKENDS[shell.family](shell)
        self._cwd = str(Path(os.getcwd()).resolve())
        self._env_vars: dict[str, str] = {}

    def _ensure_backend(self) -> _ShellBackend:
        if self._backend is None:
            info = detect_shell()
            if info is None:
                raise RuntimeError(
                    "No usable shell found. Install bash (Git for Windows on "
                    "Windows, or the system bash on Unix), or ensure PowerShell "
                    "or cmd is on PATH."
                )
            self._backend = _BACKENDS[info.family](info)
        return self._backend

    @property
    def backend(self) -> _ShellBackend:
        return self._ensure_backend()

    @property
    def cwd(self) -> str:
        return self._cwd

    def execute(self, command: str, timeout: int = 240) -> str:
        backend = self._ensure_backend()

        # 1. cd / env builtins are applied to session state in-process.
        built = backend.handle_builtin(command, self._cwd, self._env_vars)
        if built is not None:
            result, new_cwd = built
            self._cwd = new_cwd
            return result.strip() or "(empty)"

        # 2. Normal command: fresh subprocess with remembered cwd/env.
        env = {**os.environ, **self._env_vars} if self._env_vars else None
        try:
            stdout, stderr, _rc = backend.execute(
                command, timeout=timeout, cwd=self._cwd, env=env
            )
        except subprocess.TimeoutExpired:
            return f"(timed out after {timeout}s)"
        except Exception as e:
            return f"error: {e}"

        out = decode_bytes(stdout)
        if stderr:
            err = decode_bytes(stderr)
            out = (out + "\n" + err) if out else err
        return out.strip() or "(empty)"

    def restart(self) -> None:
        self._cwd = str(Path(os.getcwd()).resolve())
        self._env_vars.clear()


# ─── tool ──────────────────────────────────────────────────────────────────


_BASH_PARAMS = {
    "command": {
        "type": "string",
        "optional": True,
        "description": "The command to execute, in the syntax of the active "
        "shell described below. Omit when restart=true.",
    },
    "restart": {
        "type": "boolean",
        "optional": True,
        "description": "Reset session state (working directory and environment variables)",
    },
    "timeout": {
        "type": "number",
        "optional": True,
        "description": "Max execution time in seconds (default: 240)",
    },
}

_BASE_DESC = (
    "Run a shell command in a persistent session. Working directory and "
    "environment variables persist across commands. Use restart=true (with no "
    "command) to reset session state."
)

# Per-family guidance appended to the tool description so the model writes
# commands in the right syntax for the shell that is actually in use. Detected
# once when the tool is constructed; the same shell backs the session.
_SHELL_HINTS: dict[str, str] = {
    "bash": (
        " Active shell: bash (POSIX). Write Unix-style commands: `ls`, `grep -r`, "
        "`find . -name '*.py'`, `cat`, pipes `|`, and-chains `&&`, `export VAR=val`. "
        "Bare `cd` and `export` persist across calls."
    ),
    "sh": (
        " Active shell: sh (POSIX). Write POSIX shell commands: `ls`, `grep`, `find`, "
        "pipes `|`, `&&`. Bare `cd` and `export` persist across calls. Avoid bashisms."
    ),
    "powershell": (
        " Active shell: PowerShell. Write PowerShell syntax, NOT bash: list with "
        "`Get-ChildItem` (alias `ls`/`gci`), search text with `Select-String` (alias "
        "`sls`, the grep equivalent), print with `Write-Output`. No `grep`/`find`/"
        "`export`. Set a var with `$env:VAR='val'`. Bare `cd`/`Set-Location` and "
        "`$env:` persist across calls."
    ),
    "cmd": (
        " Active shell: cmd.exe (Windows). Write cmd syntax: `dir`, `findstr` for "
        "text search (the grep equivalent), `set VAR=val` to set a variable. No "
        "`grep`/`find`/`export`/pipes-to-bash. Bare `cd` and `set` persist across calls."
    ),
}


def _build_desc(family: str | None) -> str:
    """Compose the tool description with the syntax hint for the active shell
    family. When no shell is detected the description stays neutral; the tool
    will raise a clear error on first use instead."""
    hint = _SHELL_HINTS.get(family or "", "")
    return _BASE_DESC + hint


class BashTool(Tool):
    """Run shell commands in a persistent session (cross-platform).

    The active shell is detected once at construction and baked into the tool's
    description, so the model knows which command syntax to use (POSIX vs
    PowerShell vs cmd). The same detected shell backs the session, so the
    description and runtime behaviour always agree.
    """

    def __init__(self, timeout: int = 240) -> None:
        # Detect up front so the description reflects the real shell. If nothing
        # is found, we still build the tool (neutral description); it raises a
        # clear error on first command rather than failing at import time.
        shell = detect_shell()
        self._session = BashSession(shell=shell)
        self._default_timeout = timeout
        self._family = shell.family if shell else None
        super().__init__(
            name="bash",
            description=_build_desc(self._family),
            params=_BASH_PARAMS,
            func=self._execute,
        )

    def _execute(self, args: dict) -> str:
        if args.get("restart"):
            self._session.restart()
            return "Bash session restarted"
        cmd = args.get("command")
        if not cmd:
            raise ToolError("Either 'command' or 'restart=true' is required", "missing_param")
        return self._session.execute(cmd, timeout=args.get("timeout", self._default_timeout))


__all__ = ["BashTool", "BashSession", "ShellInfo", "detect_shell"]
