"""Station-map sanity. The map is settlement-critical, so guard its invariants."""

from __future__ import annotations

from zoneinfo import ZoneInfo

from hedge.weather.stations import (
    STATIONS,
    station_for_series,
    station_for_ticker,
)


def test_every_series_resolves_to_itself():
    for series, st in STATIONS.items():
        assert st.series == series
        assert station_for_series(series) is st


def test_lookup_is_case_insensitive():
    assert station_for_series("kxhighny") is STATIONS["KXHIGHNY"]


def test_unknown_series_returns_none():
    assert station_for_series("KXHIGHNOPE") is None


def test_ticker_prefix_resolves_station():
    st = station_for_ticker("KXHIGHCHI-25JUN28-B72.5")
    assert st is not None and st.city == "Chicago"


def test_coords_and_tz_are_valid():
    for st in STATIONS.values():
        assert -90 <= st.lat <= 90
        assert -180 <= st.lon <= 180
        ZoneInfo(st.tz)  # raises if the zone name is bogus
        assert st.nws_station.startswith("K")
