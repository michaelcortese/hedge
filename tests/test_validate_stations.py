"""Settlement-station validation: the CLI Daily-report truth source.

Guards the two bugs that wrongly condemned KNYC: (1) the IEM endpoint returns the
WHOLE year, so we must select the row matching the requested day (not results[0]),
and (2) the truth must be the official CLI high, with ASOS only as a labeled fallback.
"""

from __future__ import annotations

from datetime import date

import scripts.validate_stations as vs


def test_cli_high_selects_matching_day(monkeypatch):
    # The endpoint ignores month/day and returns the full year; picking results[0]
    # would return Jan 1 (the historical bug). We must match on `valid`.
    payload = {"results": [
        {"valid": "2025-06-01", "high": 70},
        {"valid": "2025-06-27", "high": 96},
        {"valid": "2025-06-28", "high": 85},
    ]}
    monkeypatch.setattr(vs, "_get_json", lambda *a, **k: payload)
    assert vs._cli_daily_high("KNYC", date(2025, 6, 27)) == 96.0
    assert vs._cli_daily_high("KNYC", date(2025, 6, 28)) == 85.0


def test_cli_high_missing_day_is_none(monkeypatch):
    monkeypatch.setattr(vs, "_get_json", lambda *a, **k: {"results": [
        {"valid": "2025-06-01", "high": 70}]})
    assert vs._cli_daily_high("KNYC", date(2025, 6, 27)) is None


def test_daily_high_prefers_cli_over_asos(monkeypatch):
    # CLI is authoritative; ASOS is only the labeled fallback when no CLI exists.
    monkeypatch.setattr(vs, "_cli_daily_high", lambda nws, d: 96.0)
    monkeypatch.setattr(vs, "_station_obs_max", lambda nws, d, tz: 94.0)
    assert vs._station_daily_high("KNYC", date(2025, 6, 27), "America/New_York") == (96.0, "CLI")

    monkeypatch.setattr(vs, "_cli_daily_high", lambda nws, d: None)
    assert vs._station_daily_high("KNYC", date(2025, 6, 27), "America/New_York") == (94.0, "ASOS")
