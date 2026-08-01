"""Allow `python -m ezwork.app.cli` (same as the `ezwork` entry point)."""

from . import main

raise SystemExit(main())
