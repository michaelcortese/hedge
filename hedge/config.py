"""Load Kalshi credentials from config.yaml and build a signed client.

Supports the nested ``environments`` layout:

    kalshi:
      env: demo                      # which environment is active
      environments:
        demo:
          base_url: https://demo-api.kalshi.co/trade-api/v2
          key_id: <uuid>
          private_key_path: secrets/kalshi_demo_private_key.pem
        prod: { ... }

Environment variables override config (so CI / one-off runs don't need a file):
``KALSHI_ENV``, ``KALSHI_KEY_ID``, ``KALSHI_PRIVATE_KEY_PATH``, ``KALSHI_BASE_URL``.

Demo is the default and the only environment any tournament code should touch until
a strategy has cleared the calibration bar (CLAUDE.md house rules).
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from hedge.kalshi import KalshiAuth, KalshiClient, load_private_key
from hedge.kalshi.client import DEMO_BASE

CONFIG_PATH = Path("config.yaml")


def _load_yaml() -> dict:
    if CONFIG_PATH.exists():
        return yaml.safe_load(CONFIG_PATH.read_text()) or {}
    return {}


def resolve_credentials(env: str | None = None) -> tuple[str, str, str]:
    """Return ``(key_id, private_key_path, base_url)`` for the active environment.

    Precedence: explicit ``env`` arg > ``KALSHI_ENV`` > ``config.kalshi.env`` >
    ``"demo"``. Individual fields can be overridden by env vars. Raises a clear
    error if the key id or PEM is missing.
    """
    cfg = _load_yaml().get("kalshi", {})
    env = env or os.environ.get("KALSHI_ENV") or cfg.get("env") or "demo"
    envs = cfg.get("environments", {}) or {}
    section = envs.get(env, {}) if isinstance(envs, dict) else {}

    # Fall back to the flat layout (key_id/private_key_path at kalshi.*) too.
    key_id = os.environ.get("KALSHI_KEY_ID") or section.get("key_id") or cfg.get("key_id")
    key_path = (os.environ.get("KALSHI_PRIVATE_KEY_PATH")
                or section.get("private_key_path") or cfg.get("private_key_path"))
    base_url = (os.environ.get("KALSHI_BASE_URL")
                or section.get("base_url")
                or (cfg.get("base_urls", {}) or {}).get(env)
                or DEMO_BASE)

    if not key_id or key_id == "REPLACE_ME":
        raise RuntimeError(f"no Kalshi key_id for env {env!r} (config.yaml or KALSHI_KEY_ID)")
    if not key_path or not Path(key_path).exists():
        raise RuntimeError(f"Kalshi private key not found for env {env!r}: {key_path!r}")
    return key_id, key_path, base_url


def build_client(env: str | None = None) -> tuple[KalshiClient, str, str]:
    """Build a signed ``KalshiClient`` for the active environment.

    Returns ``(client, env, base_url)`` so callers can log/guard which environment
    they're pointed at (never place orders against prod from tournament code).
    """
    cfg = _load_yaml().get("kalshi", {})
    env = env or os.environ.get("KALSHI_ENV") or cfg.get("env") or "demo"
    key_id, key_path, base_url = resolve_credentials(env)
    auth = KalshiAuth(key_id=key_id, private_key=load_private_key(key_path))
    return KalshiClient(auth, base_url=base_url), env, base_url
