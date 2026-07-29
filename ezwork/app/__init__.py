"""Ezwork app layer — minimal CLI, config, session management.

This is the application layer built on top of the `ezwork` kernel. It wires the
agent loop with a small set of built-in tools (read/write/edit/bash), a JSON
config at ~/.ezwork/config.json, file-based sessions, and a plain-text system
prompt (apps embed any XML/markup themselves — the kernel renders none).

The kernel itself (ezwork.core) stays fully reusable; this package is one
opinionated way to run it.
"""
