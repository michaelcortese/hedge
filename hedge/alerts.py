"""Push alerts for the autonomous trader.

One entry point, ``notify(level, title, msg)``, sends a best-effort push
notification. Destinations are read from:

- ``HEDGE_ALERT_URL`` (env) or ``alerts.url`` (config.yaml) — single URL or
  comma-separated list for simple multi-channel.
- ``alerts.channels`` list in config.yaml for explicit multi-channel config
  (recommended for >1 destination).
- Pushover via ``HEDGE_PUSHOVER_TOKEN`` + ``HEDGE_PUSHOVER_USER`` (env).

Supported channels (auto-detected or explicit ``type``):

* **ntfy**  — ``https://ntfy.sh/<topic>``. Level maps to priority + emoji tag.
* **Slack** — incoming webhook ``https://hooks.slack.com/...``.
* **Pushover** — CRITICAL uses emergency priority (repeats until ack).
* **Telegram** — requires bot_token + chat_id (env or config).
* **generic** — any other URL treated as ntfy-compatible (Title header + body).

The ``notify`` API is unchanged for callers. Alerts are **best-effort and never
raise** — a broken channel must not affect trading. New channels are easy to
add by subclassing ``Notifier``.

Config example (in config.yaml or deploy/config.yaml):

    alerts:
      enabled: true
      # Simple (single or comma sep):
      # url: https://ntfy.sh/hedge-alerts-xxx
      channels:
        - https://ntfy.sh/hedge-alerts-xxx
        - type: slack
          url: https://hooks.slack.com/services/...
        - type: pushover
          token: ...
          user: ...
        - type: telegram
          bot_token: ...
          chat_id: "123456789"
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
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
_SLACK_EMOJI = {
    Level.INFO: ":information_source:",
    Level.WARN: ":warning:",
    Level.CRITICAL: ":rotating_light:",
}


class Notifier(ABC):
    """Abstract notification channel. Implementations must be side-effect free
    except for the send and must swallow their own errors (return False)."""

    @abstractmethod
    def send(self, level: Level, title: str, msg: str) -> bool:
        """Send; return True if the channel accepted the message (best-effort)."""
        ...

    def __repr__(self) -> str:
        return self.__class__.__name__


class _NoopNotifier(Notifier):
    def send(self, level: Level, title: str, msg: str) -> bool:
        return False


class _NtfyNotifier(Notifier):
    def __init__(self, url: str) -> None:
        self.url = url

    def send(self, level: Level, title: str, msg: str) -> bool:
        try:
            prio, tag = _NTFY.get(level, ("3", "information_source"))
            resp = requests.post(
                self.url,
                data=msg.encode("utf-8"),
                timeout=_TIMEOUT,
                headers={"Title": title, "Priority": prio, "Tags": tag},
            )
            return resp.ok
        except Exception:  # noqa: BLE001
            return False


class _SlackNotifier(Notifier):
    def __init__(self, url: str) -> None:
        self.url = url

    def send(self, level: Level, title: str, msg: str) -> bool:
        try:
            emoji = _SLACK_EMOJI.get(level, "")
            resp = requests.post(
                self.url,
                timeout=_TIMEOUT,
                json={"text": f"{emoji} *{title}*\n{msg}"},
            )
            return resp.ok
        except Exception:  # noqa: BLE001
            return False


class _PushoverNotifier(Notifier):
    def __init__(self, token: str, user: str) -> None:
        self.token = token
        self.user = user

    def send(self, level: Level, title: str, msg: str) -> bool:
        try:
            data = {"token": self.token, "user": self.user, "title": title, "message": msg}
            if level is Level.CRITICAL:  # emergency: repeat until acknowledged
                data.update(priority="2", retry="60", expire="3600")
            resp = requests.post(
                "https://api.pushover.net/1/messages.json",
                data=data,
                timeout=_TIMEOUT,
            )
            return resp.ok
        except Exception:  # noqa: BLE001
            return False


class _TelegramNotifier(Notifier):
    """Telegram bot via sendMessage. Supports Markdown; CRITICAL gets alert emoji."""

    def __init__(self, token: str, chat_id: str | int) -> None:
        self.token = token
        self.chat_id = str(chat_id)
        self.api = f"https://api.telegram.org/bot{self.token}/sendMessage"

    def send(self, level: Level, title: str, msg: str) -> bool:
        try:
            prefix = "🚨 " if level is Level.CRITICAL else ""
            text = f"{prefix}*{title}*\n{msg}"
            resp = requests.post(
                self.api,
                json={
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": "Markdown",
                    "disable_web_page_preview": True,
                },
                timeout=_TIMEOUT,
            )
            return resp.ok
        except Exception:  # noqa: BLE001
            return False


class _GenericNotifier(Notifier):
    """Fallback: POST body with Title header (ntfy-compatible)."""

    def __init__(self, url: str) -> None:
        self.url = url

    def send(self, level: Level, title: str, msg: str) -> bool:
        try:
            resp = requests.post(
                self.url,
                data=msg.encode("utf-8"),
                timeout=_TIMEOUT,
                headers={"Title": title},
            )
            return resp.ok
        except Exception:  # noqa: BLE001
            return False


class _MultiNotifier(Notifier):
    """Dispatches to several channels. Returns True if at least one succeeded."""

    def __init__(self, notifiers: list[Notifier]) -> None:
        self.notifiers = notifiers

    def send(self, level: Level, title: str, msg: str) -> bool:
        any_ok = False
        for n in self.notifiers:
            try:
                if n.send(level, title, msg):
                    any_ok = True
            except Exception:  # noqa: BLE001
                pass
        return any_ok

    def __repr__(self) -> str:
        return f"MultiNotifier({len(self.notifiers)})"


def _load_yaml() -> dict:
    """Local copy to avoid import cycle / keep alerts self-contained for now."""
    try:
        from hedge.config import _load_yaml as _cfg_load

        return _cfg_load()
    except Exception:  # noqa: BLE001
        try:
            import yaml
            from pathlib import Path

            p = Path("config.yaml")
            if p.exists():
                return yaml.safe_load(p.read_text()) or {}
        except Exception:  # noqa: BLE001
            pass
    return {}


def _alert_url() -> str | None:
    """Legacy single-URL (or first) getter for backward compat + tests."""
    url = os.environ.get("HEDGE_ALERT_URL")
    if url:
        return url.split(",")[0].strip() or None
    try:
        cfg = _load_yaml().get("alerts") or {}
        u = cfg.get("url")
        if isinstance(u, (list, tuple)):
            return str(u[0]).strip() if u else None
        if u:
            return str(u).split(",")[0].strip() or None
    except Exception:  # noqa: BLE001
        pass
    return None


def _make_notifier_from_url(url: str | None) -> Notifier | None:
    if not url:
        return None
    u = url.strip()
    if not u:
        return None
    low = u.lower()
    if "ntfy" in low:
        return _NtfyNotifier(u)
    if "slack" in low or "hooks.slack" in low:
        return _SlackNotifier(u)
    return _GenericNotifier(u)


def _make_notifier_from_config(ch: str | dict) -> Notifier | None:
    if isinstance(ch, str):
        return _make_notifier_from_url(ch)
    if not isinstance(ch, dict):
        return None

    typ = (ch.get("type") or ch.get("kind") or "").lower().strip()

    # Explicit type handling
    if typ == "ntfy":
        u = ch.get("url")
        if u:
            return _NtfyNotifier(str(u))
    if typ == "slack":
        u = ch.get("url")
        if u:
            return _SlackNotifier(str(u))
    if typ in ("pushover", "po"):
        token = ch.get("token") or os.environ.get("HEDGE_PUSHOVER_TOKEN")
        user = ch.get("user") or os.environ.get("HEDGE_PUSHOVER_USER")
        if token and user:
            return _PushoverNotifier(str(token), str(user))
    if typ in ("telegram", "tg"):
        token = ch.get("bot_token") or ch.get("token") or os.environ.get("HEDGE_TELEGRAM_BOT_TOKEN")
        chat = ch.get("chat_id") or os.environ.get("HEDGE_TELEGRAM_CHAT_ID")
        if token and chat:
            return _TelegramNotifier(str(token), chat)

    # Fallback to url in the dict
    u = ch.get("url")
    if u:
        n = _make_notifier_from_url(str(u))
        if n:
            return n

    # Pushover shorthand in dict without type
    if "token" in ch and "user" in ch:
        return _PushoverNotifier(str(ch["token"]), str(ch["user"]))

    return None


# Cached dispatcher (reset in tests if needed via _reset_dispatcher)
_dispatcher: Notifier | None = None


def _reset_dispatcher() -> None:
    """For tests / REPL. Not part of public API."""
    global _dispatcher
    _dispatcher = None


def _build_notifiers() -> Notifier:
    notifiers: list[Notifier] = []

    # 1. HEDGE_ALERT_URL (env) — supports comma-separated for quick multi
    url_env = os.environ.get("HEDGE_ALERT_URL")
    if url_env:
        for part in [x.strip() for x in url_env.split(",") if x.strip()]:
            n = _make_notifier_from_url(part)
            if n:
                notifiers.append(n)

    # 2. alerts.url from yaml (legacy single or comma)
    if not url_env:
        try:
            cfg = _load_yaml().get("alerts") or {}
            u = cfg.get("url")
            if u:
                urls = [str(x).strip() for x in (u if isinstance(u, (list, tuple)) else str(u).split(",")) if str(x).strip()]
                for uu in urls:
                    n = _make_notifier_from_url(uu)
                    if n:
                        notifiers.append(n)
        except Exception:  # noqa: BLE001
            pass

    # 3. Pushover via dedicated env (can coexist with URL channels)
    token = os.environ.get("HEDGE_PUSHOVER_TOKEN")
    user = os.environ.get("HEDGE_PUSHOVER_USER")
    if token and user:
        notifiers.append(_PushoverNotifier(token, user))

    # 4. Explicit channels list (the "better" way)
    try:
        cfg = _load_yaml().get("alerts") or {}
        if cfg.get("enabled", True) is False:
            return _NoopNotifier()
        for ch in cfg.get("channels") or []:
            n = _make_notifier_from_config(ch)
            if n:
                notifiers.append(n)
    except Exception:  # noqa: BLE001
        pass

    # Dedup (by repr for simplicity — good enough for scaffold)
    seen: set[str] = set()
    unique: list[Notifier] = []
    for n in notifiers:
        key = repr(n)
        if key not in seen:
            seen.add(key)
            unique.append(n)

    if not unique:
        return _NoopNotifier()
    if len(unique) == 1:
        return unique[0]
    return _MultiNotifier(unique)


def _get_dispatcher() -> Notifier:
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = _build_notifiers()
    return _dispatcher


def notify(level: Level | str, title: str, msg: str, *, url: str | None = None) -> bool:
    """Best-effort push notification. Returns True if the channel(s) accepted it.

    Never raises — a broken alert path must not take down the trader.

    If ``url`` is provided (primarily for tests or ad-hoc), only that single
    destination is used (bypasses config). Supports the original single-URL
    behavior.

    Multiple channels are configured via HEDGE_ALERT_URL (comma ok), alerts.url,
    or the alerts.channels list in config.yaml.
    """
    level = level if isinstance(level, Level) else Level(level)

    if url:
        # Explicit override / test path: single destination, original behavior
        n = _make_notifier_from_url(url)
        if n is None:
            return False
        return n.send(level, title, msg)

    disp = _get_dispatcher()
    return disp.send(level, title, msg)
