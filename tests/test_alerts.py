"""Alerts: best-effort, never raises, channel auto-detected from the URL."""

from __future__ import annotations

from hedge import alerts


def test_no_channel_configured_returns_false(monkeypatch):
    monkeypatch.delenv("HEDGE_ALERT_URL", raising=False)
    monkeypatch.delenv("HEDGE_PUSHOVER_TOKEN", raising=False)
    monkeypatch.delenv("HEDGE_PUSHOVER_USER", raising=False)
    # No URL, no config -> no-op, returns False, never raises.
    assert alerts.notify(alerts.Level.CRITICAL, "t", "m", url=None) is False


def test_ntfy_post_shape(monkeypatch):
    captured = {}

    def fake_post(url, **kw):
        captured["url"] = url
        captured["headers"] = kw.get("headers")
        captured["data"] = kw.get("data")

        class R:
            ok = True
        return R()

    monkeypatch.setattr(alerts.requests, "post", fake_post)
    ok = alerts.notify(alerts.Level.CRITICAL, "boom", "kill-switch",
                       url="https://ntfy.sh/hedge-alerts-7f3k9q2x")
    assert ok is True
    assert captured["url"].endswith("hedge-alerts-7f3k9q2x")
    assert captured["headers"]["Title"] == "boom"
    assert captured["headers"]["Priority"] == "5"  # CRITICAL -> max priority


def test_notify_swallows_exceptions(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(alerts.requests, "post", boom)
    # Must not propagate — alerts are best-effort.
    assert alerts.notify("warn", "t", "m", url="https://ntfy.sh/x") is False
