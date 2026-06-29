"""Kalshi temperature series -> official NWS settlement station map.

**This map is the single most important correctness input in the whole system.**
A Kalshi daily-high market settles to the max temperature reported by one specific
NWS/ASOS station, rounded to a whole degree Fahrenheit. If we forecast the wrong
station (or the wrong city's airport), every strategy produces a *confident but
biased* probability — and fractional Kelly with a biased ``p`` loses money fast
(see CLAUDE.md "the load-bearing caveat"). So this table is deliberately small and
must be validated against a handful of already-resolved Kalshi markets before any
strategy is trusted with real size (see ``scripts/validate_stations.py`` / the
Phase 1 verification step).

Coordinates are the station location (used for Open-Meteo point queries); the
``nws_station`` id is used for live METAR/ASOS observations via api.weather.gov.
``tz`` is the IANA zone that defines the *local calendar day* the high is measured
over — Kalshi's "high of the day" is the max over the station-local day.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Station:
    """An official settlement station for one Kalshi temperature series."""

    series: str          # Kalshi series ticker prefix, e.g. "KXHIGHNY"
    city: str            # human label, e.g. "New York City"
    nws_station: str     # ASOS/METAR id for observations, e.g. "KNYC"
    lat: float
    lon: float
    tz: str              # IANA tz defining the local day, e.g. "America/New_York"
    #: Set False until the (series, station) pair has been checked against a
    #: resolved Kalshi market. Strategies refuse real size on unvalidated stations.
    validated: bool = False


# Start narrow: a few high-liquidity cities (per the approved plan). Expand only
# after each new row is validated against a resolved market.
#
# NOTE: the NWS official *climate* site is not always the city's main airport.
# These are the commonly-cited Kalshi settlement points.
# validated flags set by scripts/validate_stations.py against 14 days of resolved
# Kalshi settlements vs each station's official NWS Climatological Report (Daily)
# high — the value Kalshi actually settles on (exact bucket match, tol=0):
#   NY  KNYC 100% (vs KLGA 43%, KJFK 50%),  CHI KMDW 100% (vs KORD 64%),
#   MIA KMIA 100% (vs KFLL 21%),            AUS KAUS 100% (vs KATT 86%).
# NY was previously left unvalidated ONLY because the old check compared against the
# raw ASOS observed max (KNYC 57% vs KLGA 86%); that disagreed with the CLI Daily
# report by exactly the rounding/conversion nuance the market rules warn about.
# Against the correct instrument (CLI), KNYC/Central Park is unambiguous — 14/14.
_STATION_LIST: list[Station] = [
    Station("KXHIGHNY", "New York City", "KNYC", 40.78, -73.97, "America/New_York", validated=True),
    Station("KXHIGHCHI", "Chicago", "KMDW", 41.79, -87.75, "America/Chicago", validated=True),
    Station("KXHIGHMIA", "Miami", "KMIA", 25.79, -80.29, "America/New_York", validated=True),
    Station("KXHIGHAUS", "Austin", "KAUS", 30.19, -97.67, "America/Chicago", validated=True),
]

STATIONS: dict[str, Station] = {s.series: s for s in _STATION_LIST}


def station_for_series(series: str) -> Station | None:
    """Look up the settlement station for a Kalshi series ticker prefix.

    Returns None for an unknown/uncovered series so callers can abstain rather
    than guess a station (guessing would silently bias every signal).
    """
    return STATIONS.get(series.upper())


def station_for_ticker(ticker: str) -> Station | None:
    """Resolve a station from a full market/event ticker by series prefix match.

    Kalshi tickers look like ``KXHIGHNY-25JUN28-B72.5``; the series prefix is the
    part before the first ``-``.
    """
    series = ticker.split("-", 1)[0].upper()
    return station_for_series(series)
