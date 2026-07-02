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


# --------------------------------------------------------------------------- #
# CLI product (the settlement instrument) — parse + issuance selection         #
# --------------------------------------------------------------------------- #
def _cli_text(day_line: str = "JUNE 30 2026",
              max_line: str = "  MAXIMUM         87   1226 PM  99    1964  84      3       90",
              valid_line: str = "VALID TODAY AS OF 0400 PM LOCAL TIME.") -> str:
    # Abbreviated from a real CLINYC intraday product (2026-06-30 20:40Z).
    return f"""000
CDUS41 KOKX 302040
CLINYC

CLIMATE REPORT
NATIONAL WEATHER SERVICE NEW YORK, NY
440 PM EDT TUE JUN 30 2026

...THE CENTRAL PARK NY CLIMATE SUMMARY FOR {day_line}...
{valid_line}

WEATHER ITEM   OBSERVED TIME   RECORD YEAR NORMAL DEPARTURE LAST
                VALUE   (LST)  VALUE       VALUE  FROM      YEAR
...................................................................
TEMPERATURE (F)
 TODAY
{max_line}
  MINIMUM         73    522 AM  53    1919  69      4       71

PRECIPITATION (IN)
  TODAY            0.00          3.07 1984   0.13  -0.13      T
"""


def test_cli_parse_intraday_product():
    assert providers._parse_cli_product(_cli_text()) == (date(2026, 6, 30), 87.0)


def test_cli_parse_morning_variant_colon_time_negative_departure():
    # AUS 12:26Z shape: "VALID AS OF" (no TODAY), colon in the obs time, negative
    # departure later on the line — the first integer is still the observed value.
    text = _cli_text(
        max_line="  MAXIMUM         79  12:28 AM 100    1980  94    -15",
        valid_line="VALID AS OF 0700 AM LOCAL TIME.",
    )
    assert providers._parse_cli_product(text) == (date(2026, 6, 30), 79.0)


def test_cli_parse_missing_value_and_junk_rejected():
    assert providers._parse_cli_product(_cli_text(max_line="  MAXIMUM         MM")) is None
    assert providers._parse_cli_product(_cli_text(max_line="  MAXIMUM        200")) is None
    assert providers._parse_cli_product("NOT A CLI PRODUCT AT ALL") is None


def _patch_afos(monkeypatch, listing_by_date: dict, texts_by_pid: dict) -> None:
    def fake_get_json(url, params, key, ttl, **kw):
        return {"data": listing_by_date.get(params["date"], [])}

    def fake_get_text(url, key, ttl, **kw):
        return texts_by_pid.get(url.rsplit("/", 1)[-1])

    monkeypatch.setattr(providers, "_get_json", fake_get_json)
    monkeypatch.setattr(providers, "_get_text", fake_get_text)


def test_cli_max_filters_by_date_and_latest_issuance_supersedes(monkeypatch):
    # The 06:42Z product is the FINAL for the previous day (wrong climate day ->
    # ignored); the 22:43Z correction supersedes the 20:40Z intraday even though
    # its value is LOWER — the official record is replaced, not max()ed.
    listing = {"2026-06-30": [
        {"entered": "2026-06-30T06:42:00Z", "product_id": "P-final-prev"},
        {"entered": "2026-06-30T20:40:00Z", "product_id": "P-intraday"},
        {"entered": "2026-06-30T22:43:00Z", "product_id": "P-correction"},
    ], "2026-07-01": []}
    texts = {
        "P-final-prev": _cli_text(day_line="JUNE 29 2026"),
        "P-intraday": _cli_text(max_line="  MAXIMUM         88   1226 PM  99    1964"),
        "P-correction": _cli_text(max_line="  MAXIMUM         87   1226 PM  99    1964"),
    }
    _patch_afos(monkeypatch, listing, texts)
    assert providers.iem_cli_max_so_far_f(CHI, date(2026, 6, 30)) == 87.0


def test_cli_max_none_when_no_matching_product_or_no_iem_id(monkeypatch):
    _patch_afos(monkeypatch, {}, {})
    assert providers.iem_cli_max_so_far_f(NYC, date(2026, 6, 30)) is None
    from hedge.weather.stations import Station
    bare = Station("KXTEST", "Testville", "KTST", 0.0, 0.0, "America/New_York")
    assert providers.iem_cli_max_so_far_f(bare, date(2026, 6, 30)) is None
