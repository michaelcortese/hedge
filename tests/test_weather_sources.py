"""LiveForecastSource.observed_max: blending the raw-obs and CLI floors.

The CLI product is the settlement instrument, so when present it is the
highest-trust floor — but it comes from the same sensor as the raw obs, so a CLI
value far above every observation means a parse/station mix-up and must be
distrusted rather than become a false floor. All offline via monkeypatching.
"""

from __future__ import annotations

from datetime import date

from hedge.weather import providers
from hedge.weather.sources import LiveForecastSource
from hedge.weather.stations import STATIONS

NYC = STATIONS["KXHIGHNY"]
DAY = date(2026, 7, 1)


def _patch(monkeypatch, temps: list[float], cli: float | None) -> None:
    monkeypatch.setattr(providers, "nws_recent_temps_f", lambda *a, **k: temps)
    monkeypatch.setattr(providers, "iem_cli_max_so_far_f", lambda *a, **k: cli)


def test_cli_floor_sharpens_obs_floor(monkeypatch):
    # Hourly METARs peaked at 86.2 but the CLI printed the official 87 (the
    # 1-minute peak between METARs). The sharper official floor wins.
    _patch(monkeypatch, [80.1, 86.2], cli=87.0)
    assert LiveForecastSource().observed_max(NYC, DAY) == 87.0


def test_obs_floor_stands_between_issuances(monkeypatch):
    _patch(monkeypatch, [80.1, 86.2], cli=None)
    assert LiveForecastSource().observed_max(NYC, DAY) == 86.2


def test_cli_floor_stands_alone_when_obs_feed_down(monkeypatch):
    _patch(monkeypatch, [], cli=87.0)
    assert LiveForecastSource().observed_max(NYC, DAY) == 87.0


def test_none_when_neither_source_has_data(monkeypatch):
    _patch(monkeypatch, [], cli=None)
    assert LiveForecastSource().observed_max(NYC, DAY) is None


def test_implausible_cli_gap_distrusted(monkeypatch):
    # CLI 95 vs best ob 86.2: >5F above the same sensor's raw readings is a
    # parse/station mix-up, not a real peak — keep the raw-obs floor.
    _patch(monkeypatch, [80.1, 86.2], cli=95.0)
    assert LiveForecastSource().observed_max(NYC, DAY) == 86.2


def test_plausible_cli_gap_trusted(monkeypatch):
    # 4.8F above the hourly max is within the 1-min-peak + rounding envelope.
    _patch(monkeypatch, [86.2], cli=91.0)
    assert LiveForecastSource().observed_max(NYC, DAY) == 91.0
