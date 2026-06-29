"""Tiny on-disk JSON cache for weather/API fetches.

Two jobs:
  1. Respect free-API rate limits (Open-Meteo, api.weather.gov) by not refetching
     the same thing within a TTL.
  2. Make runs reproducible — a cached forecast replays identically, which the
     strategy contract requires ("evaluate must be reproducible").

Cache files live under ``data/cache/`` (gitignored). Keys are caller-supplied and
should encode everything that identifies the response (station + date + model +
run), so a stale key never returns the wrong city's data.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

CACHE_DIR = Path("data/cache")


def _path_for(key: str) -> Path:
    digest = hashlib.sha256(key.encode()).hexdigest()[:24]
    # Keep a readable prefix so the cache dir is browsable.
    safe = "".join(c if c.isalnum() or c in "-._" else "_" for c in key)[:60]
    return CACHE_DIR / f"{safe}__{digest}.json"


def load(key: str, ttl_seconds: float | None) -> Any | None:
    """Return a cached value for ``key`` if present and within ``ttl_seconds``.

    ``ttl_seconds=None`` means "never expire" (use for immutable historical
    fetches); a finite TTL is for live forecasts/observations.
    """
    p = _path_for(key)
    if not p.exists():
        return None
    if ttl_seconds is not None and (time.time() - p.stat().st_mtime) > ttl_seconds:
        return None
    try:
        return json.loads(p.read_text())["value"]
    except (json.JSONDecodeError, KeyError):
        return None


def store(key: str, value: Any) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = _path_for(key)
    p.write_text(json.dumps({"key": key, "stored_at": time.time(), "value": value}))
