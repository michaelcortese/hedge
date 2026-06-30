"""Alerts: best-effort, never raises, supports legacy single + new multi-channel scaffold."""

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


def test_multi_channel_via_comma_url(monkeypatch):
    """HEDGE_ALERT_URL can be comma-separated; both should be called (scaffold)."""
    calls = []

    def fake_post(url, **kw):
        calls.append(url)
        class R:
            ok = True
        return R()

    monkeypatch.delenv("HEDGE_PUSHOVER_TOKEN", raising=False)
    monkeypatch.delenv("HEDGE_PUSHOVER_USER", raising=False)
    monkeypatch.setenv("HEDGE_ALERT_URL", "https://ntfy.sh/a,https://hooks.slack.com/b")
    alerts._reset_dispatcher()
    monkeypatch.setattr(alerts.requests, "post", fake_post)

    ok = alerts.notify(alerts.Level.INFO, "multi", "test msg")
    assert ok is True
    assert len(calls) >= 2
    assert any("ntfy.sh/a" in c for c in calls)
    assert any("slack.com/b" in c for c in calls)
    alerts._reset_dispatcher()  # cleanup


def test_pushover_critical(monkeypatch):
    captured = {}

    def fake_post(url, **kw):
        captured["url"] = url
        captured["data"] = kw.get("data") or {}
        class R:
            ok = True
        return R()

    monkeypatch.setenv("HEDGE_PUSHOVER_TOKEN", "tok123")
    monkeypatch.setenv("HEDGE_PUSHOVER_USER", "usr456")
    monkeypatch.delenv("HEDGE_ALERT_URL", raising=False)
    alerts._reset_dispatcher()
    monkeypatch.setattr(alerts.requests, "post", fake_post)

    ok = alerts.notify(alerts.Level.CRITICAL, "crit", "important")
    assert ok is True
    assert captured["url"] == "https://api.pushover.net/1/messages.json"
    assert captured["data"].get("priority") == "2"
    alerts._reset_dispatcher()
