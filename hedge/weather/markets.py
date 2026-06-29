"""Parse a Kalshi temperature market into a structured ``TempMarket``.

A Kalshi "high temp" event for one city/day is offered as a *set* of binary
bucket contracts — e.g. "70° to 71°", "72° to 73°", plus open-ended tails like
"74° or above". Each contract's ``GET /markets/{ticker}`` payload carries the
numeric bounds in ``floor_strike`` / ``cap_strike`` and a ``strike_type`` that
says whether it's a closed range or a one-sided tail. This module normalizes all
of that into a single ``TempMarket`` so the Monte Carlo core only ever deals with
``[lo, hi]`` interval bounds in °F.

Parsing is defensive: it prefers the structured strike fields and falls back to
the human ``yes_sub_title`` ("70° to 71°") only when needed, because the engine
must never silently mis-read a bucket bound.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

from hedge.weather.stations import Station, station_for_ticker

# Kalshi event tickers embed the date as e.g. "25JUN28" (YY MON DD).
_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}
_DATE_RE = re.compile(r"-(\d{2})([A-Z]{3})(\d{2})")
# Fallback: pull numbers out of "70° to 71°", "above 74", "below 31°", etc.
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass(frozen=True)
class TempMarket:
    """Normalized view of one Kalshi daily-high bucket contract.

    ``lo_f`` / ``hi_f`` are the inclusive interval the *official rounded daily
    high* (an integer °F) must fall in for YES to resolve. Open-ended tails use
    -inf / +inf. The Monte Carlo core counts the fraction of simulated highs in
    ``[lo_f, hi_f]`` to get ``P(YES)``.
    """

    ticker: str
    series: str
    station: Station
    local_date: date          # station-local calendar day the high is measured over
    lo_f: float               # inclusive lower bound (°F), -inf for a "below X" tail
    hi_f: float               # inclusive upper bound (°F), +inf for an "above X" tail
    strike_type: str          # raw Kalshi strike_type, for logging/attribution

    @property
    def is_tail(self) -> bool:
        return math.isinf(self.lo_f) or math.isinf(self.hi_f)

    def contains(self, high_f: int | float) -> bool:
        return self.lo_f <= high_f <= self.hi_f


def _parse_local_date(event_or_ticker: str, station: Station,
                      raw: dict | None) -> date:
    """Determine the station-local calendar day the high is measured over.

    Prefer the date code embedded in the ticker (unambiguous); fall back to the
    market ``close_time``/``expiration_time`` converted into the station's local
    timezone.
    """
    m = _DATE_RE.search(event_or_ticker.upper())
    if m:
        yy, mon, dd = m.group(1), m.group(2), m.group(3)
        if mon in _MONTHS:
            return date(2000 + int(yy), _MONTHS[mon], int(dd))
    # Fallback to close/expiration time, interpreted in the station's local zone.
    raw = raw or {}
    ts = raw.get("close_time") or raw.get("expiration_time")
    if ts:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(ZoneInfo(station.tz)).date()
    raise ValueError(f"cannot determine local date for market {event_or_ticker!r}")


def _bucket_bounds(raw: dict) -> tuple[float, float, str]:
    """Extract inclusive [lo, hi] °F bounds and the strike_type from a payload.

    Handles the three Kalshi shapes:
      * closed range ("between"): floor_strike <= high <= cap_strike
      * upper tail ("greater"/"greater_or_equal"): high >= floor_strike
      * lower tail ("less"/"less_or_equal"): high <= cap_strike
    Falls back to parsing the subtitle text if structured strikes are absent.
    """
    stype = str(raw.get("strike_type", "")).lower()
    floor = raw.get("floor_strike")
    cap = raw.get("cap_strike")

    if floor is not None or cap is not None:
        # Kalshi convention, VERIFIED against live markets (KXHIGHNY, 2026-06):
        #   between          floor=77 cap=78  -> "77° to 78°"  (inclusive both ends)
        #   greater (strict) floor=84         -> "85° or above" (high > floor)
        #   less    (strict) cap=77           -> "76° or below" (high < cap)
        # The daily high is an integer °F, so a STRICT tail shifts the inclusive
        # integer bound by one degree. Getting this wrong biases every tail signal
        # by ~1°F — exactly the kind of bias fractional Kelly punishes hard.
        if floor is not None and cap is not None:
            return float(floor), float(cap), stype or "between"
        if cap is None:  # upper tail
            if "greater_or_equal" in stype or "gte" in stype:
                return float(floor), math.inf, stype
            return float(floor) + 1.0, math.inf, stype or "greater"
        # lower tail (floor is None)
        if "less_or_equal" in stype or "lte" in stype:
            return -math.inf, float(cap), stype
        return -math.inf, float(cap) - 1.0, stype or "less"

    # --- subtitle fallback (e.g. "70° to 71°", "74° or above", "31° or below")
    sub = str(raw.get("yes_sub_title") or raw.get("subtitle") or "")
    nums = [float(x) for x in _NUM_RE.findall(sub)]
    low_sub = sub.lower()
    if not nums:
        raise ValueError(f"cannot parse bucket bounds from {raw.get('ticker')!r}: {sub!r}")
    if "above" in low_sub or "or more" in low_sub or "higher" in low_sub:
        return nums[0], math.inf, stype or "greater"
    if "below" in low_sub or "or less" in low_sub or "under" in low_sub:
        return -math.inf, nums[0], stype or "less"
    if len(nums) >= 2:
        return min(nums), max(nums), stype or "between"
    # single number, no tail words: treat as an exact-degree bucket [n, n].
    return nums[0], nums[0], stype or "between"


def parse_temp_market(raw: dict) -> TempMarket | None:
    """Build a ``TempMarket`` from a Kalshi ``GET /markets/{ticker}`` payload.

    Returns None if the market isn't a covered temperature market (unknown
    series/station) so callers abstain instead of guessing.
    """
    ticker = raw.get("ticker", "")
    event = raw.get("event_ticker", ticker)
    station = station_for_ticker(ticker)
    if station is None:
        return None
    local_date = _parse_local_date(event or ticker, station, raw)
    lo, hi, stype = _bucket_bounds(raw)
    return TempMarket(
        ticker=ticker,
        series=station.series,
        station=station,
        local_date=local_date,
        lo_f=lo,
        hi_f=hi,
        strike_type=stype,
    )


def discover_temp_markets(client, series: str, status: str = "open") -> list[dict]:
    """List raw market payloads for a temperature series via the Kalshi client.

    Thin wrapper over ``client.get_markets(series_ticker=...)`` (see
    ``hedge/kalshi/client.py``). Returns the raw ``markets`` list; pass each
    element through ``parse_temp_market`` to get structured buckets.
    """
    resp = client.get_markets(series_ticker=series, status=status)
    return resp.get("markets", [])
