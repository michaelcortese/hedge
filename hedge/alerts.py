"""Push alerts for the autonomous trader (ntfy / Pushover / Slack).

One entry point, ``notify(level, title, msg)``, sends a push notification over a
single HTTPS request. The destination is read from the ``HEDGE_ALERT_URL`` secret
(or ``alerts.url`` in config.yaml) and the channel is auto-detected from it:

  * **ntfy**  — ``HEDGE_ALERT_URL=https://ntfy.sh/<topic>`` (no account; topic name
    is the password). Level maps to ntfy priority + an emoji tag.
  * **Slack** — an incoming-webhook URL (``https://hooks.slack.com/...``).
  * **Pushover** — set ``HEDGE_PUSHOVER_TOKEN`` + ``HEDGE_PUSHOVER_USER`` instead of a
    URL; CRITICAL uses emergency priority 2 (repeats until acknowledged).

Alerts are **best-effort and never raise** into the trading loop: a failed push must
not crash or block a cycle. Switching channels is a secret change, no code.
"""

from __future__ import annotations

import os
from enum import Enum

import requests

_TIMEOUT = 5.0


class Level(str, Enum):
    INFO = "info"
    WARN = "warn"
    CRITICAL = "critical"


# Level -> (ntfy priority, emoji tag). ntfy priority 5 = max/urgent.
_NTFY = {
    Level.INFO: ("3", "information_source"),
    Level.WARN: ("4", "warning"),
    Level.CRITICAL: ("5", "rotating_light"),
}
_SLACK_EMOJI = {Level.INFO: ":information_source:", Level.WARN: ":warning:",
                Level.CRITICAL: ":rotating_light:"}


def _alert_url() -> str | None:
    url = os.environ.get("HEDGE_ALERT_URL")
    if url:
        return url
    try:
        from hedge.config import _load_yaml
        return (_load_yaml().get("alerts") or {}).get("url")
    except Exception:  # noqa: BLE001 — config is optional for alerts
        return None


def notify(level: Level | str, title: str, msg: str, *, url: str | None = None) -> bool:
    """Best-effort push notification. Returns True if the channel accepted it.

    Never raises — a broken alert path must not take down the trader.
    """
    level = level if isinstance(level, Level) else Level(level)
    url = url if url is not None else _alert_url()
    try:
        token = os.environ.get("HEDGE_PUSHOVER_TOKEN")
        user = os.environ.get("HEDGE_PUSHOVER_USER")
        if not url and token and user:
            data = {"token": token, "user": user, "title": title, "message": msg}
            if level is Level.CRITICAL:  # emergency: repeat until acknowledged
                data.update(priority=2, retry=60, expire=3600)
            return requests.post("https://api.pushover.net/1/messages.json",
                                 data=data, timeout=_TIMEOUT).ok

        if not url:
            return False

        if "ntfy" in url:
            prio, tag = _NTFY.get(level, ("3", "information_source"))
            return requests.post(url, data=msg.encode("utf-8"), timeout=_TIMEOUT,
                                 headers={"Title": title, "Priority": prio, "Tags": tag}).ok

        if "slack" in url:
            return requests.post(url, timeout=_TIMEOUT,
                                 json={"text": f"{_SLACK_EMOJI[level]} *{title}*\n{msg}"}).ok

        # Unknown URL: POST body with a Title header (ntfy-compatible default).
        return requests.post(url, data=msg.encode("utf-8"), timeout=_TIMEOUT,
                             headers={"Title": title}).ok
    except Exception:  # noqa: BLE001 — alerts are best-effort
        return False
