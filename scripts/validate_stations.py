#!/usr/bin/env python
"""Validate the Kalshi settlement-station map against resolved markets.

The single biggest live-money risk (CLAUDE.md, weather/stations.py): if a series is
mapped to the wrong NWS station, every probability is confidently biased and Kelly
loses fast. The backtest CANNOT catch this — it grades against the same coordinates
it queries. The only ground truth is Kalshi's own settlements.

What this does, per city:
  1. Pull recently SETTLED Kalshi markets (public read-only, no key needed) and, for
     each (city, day) event, read the YES-resolved bucket -> the official settled
     daily-high range, e.g. "81 to 82".
  2. For each CANDIDATE NWS station, fetch that station's official **NWS
     Climatological Report (Daily)** high — the value Kalshi actually settles on,
     parsed from the CLI product (Iowa Environmental Mesonet) — for the same day and
     check whether it lands in the settled range. Falls back to the preliminary ASOS
     observed max only when no CLI has been issued; the report tags which was used.
     Validating against raw ASOS (the prior approach) wrongly condemned KNYC, because
     obs max differs from the CLI value by exactly the rounding/conversion nuances the
     market rules warn about. This is the real settlement instrument — not ERA5, which
     is too coarse to tell e.g. KMDW from KORD.
  3. Report a per-candidate match rate. The current map is only trustworthy if its
     station is the best match at a high rate. Alternatives (KORD for Chicago, KATT /
     Camp Mabry for Austin) are tested side-by-side so a wrong map is obvious.

NWS observations only reach back ~7 days, so run this routinely and accumulate the
evidence; a single run may only cover a few days. ``--write`` flips ``validated=True``
in stations.py ONLY for a current row that is the clear best match (it never silently
rewrites a station id — a suspected id error is surfaced for a human to fix).

Usage:
    .venv/bin/python scripts/validate_stations.py [--days 7] [--min-rate 0.8] [--write]
"""

from __future__ import annotations

import argparse
import re
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

from hedge.weather.providers import _get_json
from hedge.weather.stations import STATIONS, Station

PROD_BASE = "https://api.elections.kalshi.com/trade-api/v2"

# Candidate stations to test per series: the current map PLUS plausible alternates
# that Kalshi has been known to (or could) settle on. nws_station is all that matters
# for the observation pull; lat/lon are carried only for completeness.
_CANDIDATES: dict[str, list[tuple[str, float, float]]] = {
    "KXHIGHNY":  [("KNYC", 40.78, -73.97), ("KLGA", 40.78, -73.88), ("KJFK", 40.64, -73.78)],
    "KXHIGHCHI": [("KMDW", 41.79, -87.75), ("KORD", 41.98, -87.90)],
    "KXHIGHMIA": [("KMIA", 25.79, -80.29), ("KFLL", 26.07, -80.15)],
    "KXHIGHAUS": [("KAUS", 30.19, -97.67), ("KATT", 30.32, -97.76)],  # Bergstrom vs Camp Mabry
}

_DATE_RE = re.compile(r"-(\d{2})([A-Z]{3})(\d{2})")
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}


def _event_date(ticker: str) -> date | None:
    m = _DATE_RE.search(ticker.upper())
    if not m or m.group(2) not in _MONTHS:
        return None
    return date(2000 + int(m.group(1)), _MONTHS[m.group(2)], int(m.group(3)))


def _settled_high_range(series: str, since: date) -> dict[date, tuple[float, float]]:
    """Map each settled (city, day) to the official daily-high range from its YES bucket.

    Uses the tightest YES-resolved 'between' bucket [floor, cap]; falls back to a YES
    tail (>= floor or <= cap) when no closed bucket is available.
    """
    out: dict[date, tuple[float, float]] = {}
    cursor: str | None = None
    seen = 0
    while seen < 400:
        params: dict = {"series_ticker": series, "status": "settled", "limit": 100}
        if cursor:
            params["cursor"] = cursor
        resp = requests.get(PROD_BASE + "/markets", params=params, timeout=20).json()
        markets = resp.get("markets", [])
        if not markets:
            break
        for m in markets:
            seen += 1
            d = _event_date(m.get("ticker", ""))
            if d is None or d < since:
                continue
            if str(m.get("result", "")).lower() != "yes":
                continue
            floor, cap = m.get("floor_strike"), m.get("cap_strike")
            if floor is not None and cap is not None:
                out[d] = (float(floor), float(cap))             # tightest: closed bucket
            elif d not in out and floor is not None:
                out[d] = (float(floor), float("inf"))           # YES upper tail
            elif d not in out and cap is not None:
                out[d] = (float("-inf"), float(cap))
        cursor = resp.get("cursor")
        if not cursor or all((_event_date(m.get("ticker", "")) or since) < since for m in markets):
            break
    return out


def _cli_daily_high(nws_id: str, day: date) -> float | None:
    """Official daily high (°F) from the NWS **Climatological Report (Daily)**.

    THIS is what Kalshi settles on ("the highest temperature ... as reported by the
    NWS Climatological Report (Daily)"), NOT the raw ASOS observation stream. The two
    differ by exactly the rounding/conversion nuances the market rules warn about, so
    validating against the obs max can wrongly condemn the correct station (it flagged
    NYC/KNYC as a poor match when KNYC/Central Park is in fact the official site).

    Sourced from the Iowa Environmental Mesonet, which parses the CLI text products
    into JSON keyed by the issuing site id (e.g. ``KNYC`` = Central Park). Returns the
    already-rounded whole-°F high, or None if no CLI has been issued for that day yet.
    """
    # The endpoint ignores month/day and returns the WHOLE year's reports in
    # ``results``; we must select the row whose ``valid`` matches the requested day
    # (taking results[0] silently returns Jan 1). Cache per-year to avoid refetching.
    data = _get_json(
        "https://mesonet.agron.iastate.edu/json/cli.py",
        {"station": nws_id, "year": day.year},
        f"cli:{nws_id}:{day.year}", ttl=None,
    )
    if not data:
        return None
    target = day.isoformat()
    results = data.get("results") if isinstance(data.get("results"), list) else []
    row = next((r for r in results if r.get("valid") == target), None)
    if row is None:
        return None
    high = row.get("high")
    try:
        return float(high) if high is not None else None
    except (TypeError, ValueError):
        return None


def _station_obs_max(nws_id: str, day: date, tz: str) -> float | None:
    """ASOS observed daily max (°F) — the FALLBACK truth when no CLI exists yet.

    Preliminary and subject to the rounding/conversion nuances above, so it is only
    used (clearly labeled) when the official CLI Daily report is unavailable.
    """
    z = ZoneInfo(tz)
    start = datetime(day.year, day.month, day.day, tzinfo=z).astimezone(timezone.utc)
    end = start + timedelta(days=1)
    key = f"valstn:{nws_id}:{day.isoformat()}"
    data = _get_json(
        f"https://api.weather.gov/stations/{nws_id}/observations",
        {"start": start.isoformat().replace("+00:00", "Z"),
         "end": end.isoformat().replace("+00:00", "Z")},
        key, ttl=None,
    )
    if not data:
        return None
    temps = []
    for feat in data.get("features", []):
        ts = feat.get("properties", {}).get("timestamp")
        c = (feat.get("properties", {}).get("temperature") or {}).get("value")
        if ts is None or c is None:
            continue
        if start <= datetime.fromisoformat(ts.replace("Z", "+00:00")) < end:
            temps.append(c * 9 / 5 + 32)
    return max(temps) if temps else None


def _station_daily_high(nws_id: str, day: date, tz: str) -> tuple[float, str] | None:
    """The official daily high to validate against: CLI Daily report, else ASOS max.

    Returns ``(high_f, source)`` where ``source`` is ``"CLI"`` (authoritative — what
    Kalshi settles on) or ``"ASOS"`` (preliminary fallback), or None if neither is
    available for the day.
    """
    cli = _cli_daily_high(nws_id, day)
    if cli is not None:
        return cli, "CLI"
    obs = _station_obs_max(nws_id, day, tz)
    if obs is not None:
        return obs, "ASOS"
    return None


def _in_range(high: float, lo: float, hi: float, tol: float = 0.0) -> bool:
    return (lo - tol) <= round(high) <= (hi + tol)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7, help="how far back to pull settlements (NWS obs reach ~7d)")
    ap.add_argument("--min-rate", type=float, default=0.8, help="match rate required to call a station validated")
    ap.add_argument("--min-days", type=int, default=3, help="minimum matched days before trusting a verdict")
    ap.add_argument("--tol", type=float, default=0.0, help="degrees of slack when matching obs to the settled bucket")
    ap.add_argument("--write", action="store_true", help="flip validated=True in stations.py for confirmed current rows")
    args = ap.parse_args()

    since = date.today() - timedelta(days=args.days)
    confirmed: list[str] = []
    for series, station in STATIONS.items():
        print(f"\n=== {series}  ({station.city}, current map -> {station.nws_station}) ===")
        settled = _settled_high_range(series, since)
        if not settled:
            print("  no settled markets in window — cannot validate yet.")
            continue
        print(f"  {len(settled)} settled day(s): "
              + ", ".join(f"{d:%m-%d}:{lo:g}-{hi:g}" for d, (lo, hi) in sorted(settled.items())))

        best_id, best_rate = None, -1.0
        for nws_id, lat, lon in _CANDIDATES.get(series, [(station.nws_station, station.lat, station.lon)]):
            hits = n = cli_days = 0
            for d, (lo, hi) in settled.items():
                truth = _station_daily_high(nws_id, d, station.tz)
                if truth is None:
                    continue
                high, source = truth
                n += 1
                cli_days += source == "CLI"
                hits += _in_range(high, lo, hi, args.tol)
            rate = hits / n if n else 0.0
            flag = "  <- current" if nws_id == station.nws_station else ""
            cov = f"{hits}/{n}" if n else "no data"
            src = f"  [{cli_days}/{n} CLI]" if n else ""
            print(f"    {nws_id:5} match {cov:>6}  rate={rate:.0%}{src}{flag}")
            if n >= args.min_days and rate > best_rate:
                best_id, best_rate = nws_id, rate

        if best_id is None:
            print("  verdict: INSUFFICIENT obs coverage — re-run after more settlements.")
        elif best_id == station.nws_station and best_rate >= args.min_rate:
            print(f"  verdict: CONFIRMED ({best_id} @ {best_rate:.0%}).")
            confirmed.append(series)
        elif best_id != station.nws_station:
            print(f"  verdict: ⚠ LIKELY WRONG STATION — {best_id} matches better "
                  f"({best_rate:.0%}) than current {station.nws_station}. Fix stations.py by hand.")
        else:
            print(f"  verdict: current station best but only {best_rate:.0%} (< {args.min_rate:.0%}); keep collecting.")

    if args.write and confirmed:
        _flip_validated(confirmed)
        print(f"\nwrote validated=True for: {', '.join(confirmed)}")
    elif args.write:
        print("\nnothing confirmed to write.")
    else:
        print(f"\n(dry run) confirmed and ready to flip validated=True: {confirmed or 'none'}")
        if confirmed:
            print("re-run with --write to patch hedge/weather/stations.py")


def _flip_validated(series_list: list[str]) -> None:
    """Set validated=True for the named series' rows in stations.py (current id only)."""
    path = Path("hedge/weather/stations.py")
    text = path.read_text()
    for series in series_list:
        # Append validated=True to the Station(...) row for this series if not present.
        pat = re.compile(rf'(Station\("{series}".*?)(\)\,)', re.DOTALL)
        def repl(m):
            row = m.group(1)
            if "validated" in row:
                return m.group(0)
            return f"{row}, validated=True{m.group(2)}"
        text = pat.sub(repl, text, count=1)
    path.write_text(text)


if __name__ == "__main__":
    main()
