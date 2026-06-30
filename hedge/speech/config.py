"""Load API keys + per-provider settings for the mention corpus.

One small file holds every transcript/news provider's credentials and toggles, so
adding a source is "drop a key in the file" rather than touching code. Resolution
order (first hit wins):

  1. ``$HEDGE_SPEECH_CONFIG`` (explicit path),
  2. ``secrets/speech_apis.yaml`` (the ``secrets/`` dir is gitignored),
  3. ``config.speech.yaml`` at the repo root (also keep this out of git).

Copy ``config.speech.example.yaml`` to one of those and fill in keys. Any value
may instead be supplied via environment variable (``HEDGE_SPEECH_<PROVIDER>_KEY``),
which overrides the file — handy for the Fly deploy where secrets are env vars.
A missing file is fine: providers without a key simply disable themselves.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CANDIDATES = (
    os.environ.get("HEDGE_SPEECH_CONFIG"),
    str(_REPO_ROOT / "secrets" / "speech_apis.yaml"),
    str(_REPO_ROOT / "config.speech.yaml"),
)


def _config_path() -> Path | None:
    for cand in _CANDIDATES:
        if cand and Path(cand).is_file():
            return Path(cand)
    return None


class SpeechConfig:
    """Parsed mention-corpus config: ``providers[name] -> {enabled, api_key, ...}``."""

    def __init__(self, data: dict[str, Any] | None = None):
        self._data = data or {}
        self._providers: dict[str, dict] = (self._data.get("providers") or {})

    @classmethod
    def load(cls) -> "SpeechConfig":
        path = _config_path()
        data: dict[str, Any] = {}
        if path is not None:
            with path.open() as fh:
                data = yaml.safe_load(fh) or {}
        return cls(data)

    def provider(self, name: str) -> dict[str, Any]:
        """Settings for one provider, with env-var key override applied."""
        cfg = dict(self._providers.get(name) or {})
        env_key = os.environ.get(f"HEDGE_SPEECH_{name.upper()}_KEY")
        if env_key:
            cfg["api_key"] = env_key
        return cfg

    def api_key(self, name: str) -> str | None:
        return self.provider(name).get("api_key")

    def is_enabled(self, name: str) -> bool:
        """A provider runs if explicitly enabled, or if it has a usable key.

        Keyless providers (e.g. GDELT) must be opted in with ``enabled: true``;
        keyed providers turn on automatically once a key is present, unless
        ``enabled: false`` is set.
        """
        cfg = self.provider(name)
        if "enabled" in cfg:
            return bool(cfg["enabled"])
        return bool(cfg.get("api_key"))
