"""Ezwork — lean agent kernel.

Subpackages:
  - ezwork.core       the kernel: agent loop, message, provider protocol,
                      tool base, prompt, config, events, extension system
  - ezwork.tools      built-in tools (read/write/edit/bash)
  - ezwork.providers  provider implementations (OpenAIProvider + per-vendor
                      factories: LongCat, DeepSeek, GLM, Mimo, MiniMax)

Typical imports:

    from ezwork.core import Agent, LoopConfig, Tool, ToolRegistry
    from ezwork.tools import ReadTool, BashTool
    from ezwork.providers import LongCat
"""

from __future__ import annotations

__version__ = "0.0.1"
