"""Piped-stdin handling for `ezwork -p` / `-p -`.

Semantics follow the Claude Code convention (`cat file | claude -p "query"`):
piped stdin is appended to a `-p "query"` prompt as context, while `-p -`
treats stdin as the whole prompt.

Never blocks indefinitely: waits up to `first_byte_timeout` for the first
byte, then drains with the _STDIN_TIMEOUT window, so a never-closing pipe
(sandbox, CI) cannot hang the process.
"""

from __future__ import annotations

import sys

# How long `-p -` waits for piped stdin before giving up (see _read_piped_stdin).
_STDIN_TIMEOUT = 2.0
# How long piped-context mode waits for the FIRST byte (see _read_piped_stdin).
# Kept short so a silent inherited pipe (CI, sub-agents) returns quickly;
# once data flows, the _STDIN_TIMEOUT window applies.
_STDIN_GRACE = 0.15


def _read_piped_stdin(first_byte_timeout: float = _STDIN_GRACE) -> str | None:
    """Return piped/redirected stdin content, or None if there is none.

    Used for two purposes: `-p -` (stdin is the whole prompt) and piped
    context with `-p "query"` (stdin content is appended as context, matching
    `cat file | claude -p "query"`).

    Returns None when stdin is a TTY, has no data, or select() can't probe it
    (Windows console handles).
    """
    if sys.stdin.isatty():
        return None

    import os
    import time

    fd = sys.stdin.fileno()
    chunks: list[str] = []

    # select() only works on sockets on Windows; probe it and fall back to a
    # non-blocking drain loop there (os.set_blocking supports pipes on win32).
    use_select = True
    try:
        import select

        select.select([sys.stdin], [], [], 0)
    except (OSError, ValueError):
        use_select = False

    if use_select:
        # Grace period for the first byte; then drain with the full window.
        ready, _, _ = select.select([sys.stdin], [], [], first_byte_timeout)
        if not ready:
            return None
        deadline = time.monotonic() + _STDIN_TIMEOUT
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            ready, _, _ = select.select([sys.stdin], [], [], remaining)
            if not ready:
                break
            data = os.read(fd, 65536)
            if not data:  # EOF — the pipe writer closed
                break
            chunks.append(data.decode("utf-8", errors="replace"))
    else:
        def drain(deadline: float) -> bool:
            """Read available data until deadline/EOF; True if any was read."""
            while True:
                if time.monotonic() >= deadline:
                    return bool(chunks)
                try:
                    data = os.read(fd, 65536)
                except BlockingIOError:
                    time.sleep(0.01)
                    continue
                if not data:  # EOF — the pipe writer closed
                    return bool(chunks)
                chunks.append(data.decode("utf-8", errors="replace"))

        try:
            os.set_blocking(fd, False)
        except OSError:
            return None  # cannot probe this stdin (e.g. a console handle)
        try:
            if not drain(time.monotonic() + first_byte_timeout):
                return None
            drain(time.monotonic() + _STDIN_TIMEOUT)
        finally:
            os.set_blocking(fd, True)

    return "".join(chunks) if chunks else None


def _prompt_from_stdin() -> str:
    """Read the whole prompt from stdin for `-p -` (like `cat -`).

    On a TTY this reads interactively until EOF (Ctrl-D); piped input is
    drained via _read_piped_stdin(). Raises SystemExit(2) on empty input so
    we never send an empty prompt (or a literal '-') to the model.
    """
    if sys.stdin.isatty():
        data = sys.stdin.read()
    else:
        # The whole prompt is expected here — allow the full drain window for
        # the first byte, unlike piped-context mode.
        data = _read_piped_stdin(first_byte_timeout=_STDIN_TIMEOUT) or ""
    if not data.strip():
        print("error: -p - received no input on stdin", file=sys.stderr)
        raise SystemExit(2)
    return data


__all__ = ["_read_piped_stdin", "_prompt_from_stdin", "_STDIN_TIMEOUT", "_STDIN_GRACE"]
