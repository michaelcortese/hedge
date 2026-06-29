#!/usr/bin/env python
"""Run the live/dry-run trading loop.

Thin wrapper around ``hedge.runner``; identical to ``python -m hedge.runner``.

    # report-only: show what it WOULD trade, no orders placed (safe default)
    .venv/bin/python scripts/run_live.py --once

    # loop in dry-run through the afternoon
    .venv/bin/python scripts/run_live.py --interval 900 --until 19:00

    # ARM real orders on demo (never prod without --allow-prod)
    .venv/bin/python scripts/run_live.py --live --once

Credentials come from config.yaml / KALSHI_* env vars (see scripts/test_auth.py).
"""

from hedge.runner import main

if __name__ == "__main__":
    main()
