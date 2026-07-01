"""Observation fetcher: climate-day filtering + QC — the false-floor guards.

The nowcast's deterministic trade treats obs_max as a hard floor on the day's
final high, so these tests pin the two ways a false floor can enter and lose
money at near-full size: observations from the previous local evening (a raw
00:00-UTC fetch window) and QC-rejected sensor glitches. All offline —
``_get_json`` is monkeypatched with synthetic api.weather.gov payloads.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from hedge.weather import providers
from hedge.weather.stations import STATIONS

CHI = STATIONS["KXHIGHCHI"]  # America/Chicago: climate day = 06:00Z .. 06:00Z (CST)
NYC = STATIONS["KXHIGHNY"]   # America/New_York: climate day = 05:00Z .. 05:00Z (EST)


def _obs(ts: str, c: float | None, qc: str = "V") -> dict:
    return {"properties": {"timestamp": ts,
                           "temperature": {"value": c, "qualityControl": qc}}}


def _patch(monkeypatch, features: list[dict]) -> None:
    monkeypatch.setattr(providers, "_get_json", lambda *a, **k: {"features": features})


def test_climate_day_is_local_standard_time():
    # July (EDT): the NYC climate day still runs midnight-to-midnight EST,
    # i.e. 05:00Z..05:00Z — 1am..1am on the local (daylight) clock.
    start, end = providers._climate_day_utc("America/New_York", date(2026, 7, 1))
    assert start == datetime(2026, 7, 1, 5, tzinfo=timezone.utc)
    assert end == datetime(2026, 7, 2, 5, tzinfo=timezone.utc)
    # January (EST): same boundary — the offset is DST-invariant by construction.
    start, end = providers._climate_day_utc("America/New_York", date(2026, 1, 15))
    assert start == datetime(2026, 1, 15, 5, tzinfo=timezone.utc)


def test_previous_evening_excluded_from_climate_day(monkeypatch):
    # Overnight cold front: yesterday 8pm CDT (01:00Z on the target UTC date) hit
    # 88°F; today never leaves the 70s. The 88 must NOT become today's floor —
    # keeping it declares every bucket below 88 "impossible" and buys a
    # guaranteed-loss NO (the P0 false-floor bug).
    _patch(monkeypatch, [
        _obs("2026-07-01T01:00:00+00:00", 31.1),   # 88.0F — previous local evening
        _obs("2026-07-01T06:51:00+00:00", 20.0),   # 68.0F — 00:51 CST, today
        _obs("2026-07-01T18:51:00+00:00", 23.3),   # 73.9F — today
    ])
    temps = providers.nws_recent_temps_f(CHI, date(2026, 7, 1))
    assert len(temps) == 2
    assert max(temps) < 75.0


def test_dst_boundary_is_1am_local_clock(monkeypatch):
    # During DST the climate day starts at 1am EDT (05:00Z): a 00:30 EDT reading
    # (04:30Z) belongs to the PREVIOUS climate day.
    _patch(monkeypatch, [
        _obs("2026-07-01T04:30:00+00:00", 30.0),   # 86F, 00:30 EDT -> previous day
        _obs("2026-07-01T05:30:00+00:00", 25.0),   # 77F, 01:30 EDT -> today
    ])
    temps = providers.nws_recent_temps_f(NYC, date(2026, 7, 1))
    assert temps == [pytest.approx(77.0)]


def test_next_day_observations_excluded(monkeypatch):
    # Replay safety: an observation at/after the next climate-day boundary is out.
    _patch(monkeypatch, [
        _obs("2026-07-01T18:00:00+00:00", 25.0),   # today
        _obs("2026-07-02T05:00:00+00:00", 32.0),   # exactly the next boundary
    ])
    temps = providers.nws_recent_temps_f(NYC, date(2026, 7, 1))
    assert temps == [pytest.approx(77.0)]


def test_qc_rejected_and_questioned_dropped(monkeypatch):
    _patch(monkeypatch, [
        _obs("2026-07-01T15:00:00+00:00", 54.0, qc="X"),  # 129F glitch, rejected
        _obs("2026-07-01T16:00:00+00:00", 45.0, qc="Q"),  # questioned
        _obs("2026-07-01T17:00:00+00:00", 25.0),          # 77F, good
    ])
    temps = providers.nws_recent_temps_f(NYC, date(2026, 7, 1))
    assert temps == [pytest.approx(77.0)]


def test_missing_or_malformed_rows_skipped(monkeypatch):
    _patch(monkeypatch, [
        _obs("garbage-timestamp", 20.0),
        _obs("2026-07-01T15:00:00+00:00", None),
        {"properties": {"timestamp": "2026-07-01T15:00:00+00:00"}},  # no temperature
    ])
    assert providers.nws_recent_temps_f(NYC, date(2026, 7, 1)) == []
