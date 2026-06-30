"""nws_recent_temps_f must keep only observations on the target LOCAL day.

The NWS query starts at UTC midnight, which is the prior local *evening* in US time
zones, so the raw response pools yesterday-evening temps into today. Left unfiltered,
a warm prior evening inflates obs_max and produces a confident-but-wrong "impossible
bucket" deterministic NO — the one trade that carries size.
"""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

from hedge.weather import providers
from hedge.weather.stations import STATIONS


def _obs(ts_iso: str, temp_c: float) -> dict:
    return {"properties": {"timestamp": ts_iso, "temperature": {"value": temp_c}}}


def test_filters_prior_local_evening_out_of_obs_max(monkeypatch):
    station = STATIONS["KXHIGHNY"]            # America/New_York
    tz = ZoneInfo(station.tz)
    target = date(2026, 6, 30)

    def ts(d: date, hour: int) -> str:
        return datetime(d.year, d.month, d.day, hour, tzinfo=tz).isoformat()

    payload = {
        "features": [
            # Prior local evening — lands after UTC midnight so the query returns it,
            # but it is NOT today's reading. 35C (95F) would wrongly dominate obs_max.
            _obs(ts(date(2026, 6, 29), 22), 35.0),
            _obs(ts(target, 10), 28.0),        # today, 28C = 82.4F
            _obs(ts(target, 14), 33.0),        # today, 33C = 91.4F (the real max-so-far)
        ]
    }
    monkeypatch.setattr(providers, "_get_json", lambda *a, **k: payload)

    temps = providers.nws_recent_temps_f(station, target)

    assert len(temps) == 2                      # the prior-evening reading is dropped
    assert max(temps) == 91.4                    # not 95.0 from yesterday evening
    assert all(t < 95.0 for t in temps)


def test_drops_observations_without_a_timestamp(monkeypatch):
    station = STATIONS["KXHIGHNY"]
    target = date(2026, 6, 30)
    payload = {"features": [
        {"properties": {"temperature": {"value": 30.0}}},          # no timestamp
        _obs(datetime(2026, 6, 30, 12, tzinfo=ZoneInfo(station.tz)).isoformat(), 30.0),
    ]}
    monkeypatch.setattr(providers, "_get_json", lambda *a, **k: payload)

    temps = providers.nws_recent_temps_f(station, target)

    assert temps == [86.0]                        # only the timestamped, in-day reading
