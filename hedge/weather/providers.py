"""Free forecast + observation fetchers, normalized to one shape.

Every provider function returns a list of ``ForecastRecord`` (one per model/run)
so the Monte Carlo core can treat all sources uniformly: each record carries a
predicted daily high and the hourly temperature curve for the target local day.

Sources (all free, no API key — per the approved plan):
  * Open-Meteo ``/v1/forecast`` with ``models=`` → a free multi-model ensemble
    (GFS, ECMWF, ICON, GEM ...). Each model is one independent ForecastRecord.
  * NWS ``api.weather.gov`` gridpoint forecast → the official human-tuned forecast.
  * NWS latest/recent observations → live METAR/ASOS temps, used by the nowcast
    to compute the observed max-so-far.

All responses go through the disk cache (``hedge/weather/cache.py``). Live
forecasts use a short TTL; the historical-archive fetchers (``archive.py``) use
``ttl=None`` so backtests are reproducible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests

from hedge.weather import cache
from hedge.weather.stations import Station

# api.weather.gov requires a descriptive User-Agent or it returns 403.
_UA = {"User-Agent": "hedge-weather-bot (contact: mcortese1406@gmail.com)"}

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
# Models exposed individually by Open-Meteo; each becomes one ensemble member.
DEFAULT_MODELS = ["gfs_seamless", "ecmwf_ifs025", "icon_seamless", "gem_seamless"]

_LIVE_TTL = 1800.0  # 30 min for live forecasts/obs


@dataclass(frozen=True)
class ForecastRecord:
    """One model's prediction of the daily high for a station-local day."""

    provider: str                 # "open-meteo" | "nws"
    model: str                    # "gfs_seamless", "nws-gridpoint", ...
    target_date: date
    daily_high_f: float | None    # predicted max for the local day
    hourly_f: list[float] = field(default_factory=list)  # local-day hourly temps


def _get_json(
    url: str, params: dict | None, key: str, ttl: float | None,
    timeout: float = 45.0,
) -> dict | None:
    cached = cache.load(key, ttl)
    if cached is not None:
        return cached
    # Retry transient timeouts/connection blips against the free APIs.
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, headers=_UA, timeout=timeout)
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_exc = exc
            continue
        if not resp.ok:
            return None
        data = resp.json()
        cache.store(key, data)
        return data
    if last_exc is not None:
        raise last_exc
    return None


def open_meteo_forecast(
    station: Station,
    target_date: date,
    models: list[str] | None = None,
    *,
    ttl: float | None = _LIVE_TTL,
) -> list[ForecastRecord]:
    """Fetch one ForecastRecord per model from Open-Meteo for ``target_date``."""
    models = models or DEFAULT_MODELS
    params = {
        "latitude": station.lat,
        "longitude": station.lon,
        "timezone": station.tz,
        "temperature_unit": "fahrenheit",
        "daily": "temperature_2m_max",
        "hourly": "temperature_2m",
        "models": ",".join(models),
        "start_date": target_date.isoformat(),
        "end_date": target_date.isoformat(),
    }
    key = f"openmeteo:{station.series}:{target_date}:{','.join(models)}"
    data = _get_json(OPEN_METEO_URL, params, key, ttl)
    if not data:
        return []
    return _parse_open_meteo(data, models, target_date)


def _parse_open_meteo(data: dict, models: list[str], target_date: date) -> list[ForecastRecord]:
    """Open-Meteo returns either flat keys (one model) or ``key_<model>`` suffixed
    keys (multiple models). Normalize both into per-model records."""
    out: list[ForecastRecord] = []
    daily = data.get("daily", {}) or {}
    hourly = data.get("hourly", {}) or {}
    multi = len(models) > 1

    for model in models:
        suffix = f"_{model}" if multi else ""
        highs = daily.get(f"temperature_2m_max{suffix}")
        temps = hourly.get(f"temperature_2m{suffix}")
        high = None
        if isinstance(highs, list) and highs and highs[0] is not None:
            high = float(highs[0])
        hourly_vals = [float(t) for t in (temps or []) if t is not None]
        if high is None and hourly_vals:
            high = max(hourly_vals)
        if high is None and not hourly_vals:
            continue
        out.append(ForecastRecord("open-meteo", model, target_date, high, hourly_vals))
    return out


def nws_forecast(
    station: Station,
    target_date: date,
    *,
    ttl: float | None = _LIVE_TTL,
) -> ForecastRecord | None:
    """Fetch the NWS gridpoint forecast and extract the daily high for the date.

    Two-step API: ``/points/{lat},{lon}`` returns the gridpoint forecast URL,
    then the forecast endpoint returns 12-hour periods with ``temperature`` (°F).
    """
    pkey = f"nws-points:{station.lat},{station.lon}"
    points = _get_json(
        f"https://api.weather.gov/points/{station.lat},{station.lon}",
        None, pkey, ttl=None,  # gridpoint mapping is stable -> cache forever
    )
    if not points:
        return None
    forecast_url = points.get("properties", {}).get("forecast")
    if not forecast_url:
        return None
    fkey = f"nws-forecast:{station.series}:{target_date}"
    fc = _get_json(forecast_url, None, fkey, ttl)
    if not fc:
        return None
    high = None
    for period in fc.get("properties", {}).get("periods", []):
        start = str(period.get("startTime", ""))[:10]
        if start == target_date.isoformat() and period.get("isDaytime"):
            t = period.get("temperature")
            if t is not None:
                high = float(t) if high is None else max(high, float(t))
    if high is None:
        return None
    return ForecastRecord("nws", "nws-gridpoint", target_date, high, [])


IEM_DAILY_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/daily.py"


def iem_daily_max_f(
    station: Station,
    start: date,
    end: date,
    *,
    ttl: float | None = None,
) -> dict[str, float]:
    """Realized daily-MAX (°F) per local day from the IEM ASOS archive.

    This is the *settlement-instrument* truth: Iowa Environmental Mesonet republishes
    the official NWS/ASOS station observations Kalshi settles a "high in city X" market
    on — unlike ERA5 (a grid reanalysis that runs systematically off the station, e.g.
    KMIA ~3°F on 2026-06-29). Used by :func:`hedge.weather.calibration.fit_calibration`
    when ``truth="station"`` so the fitted bias absorbs the grid→station offset.

    Returns ``{iso_date: max_f}``. Empty on any failure (missing ``iem_id``/network,
    network error, no rows) so the caller can fall back to ERA5. Cached forever
    (``ttl=None``) like the other archives, for reproducibility.
    """
    iem_id = getattr(station, "iem_id", None)
    network = getattr(station, "iem_network", None)
    if not iem_id or not network:
        return {}
    # IEM daily.py wants the date range as split year/month/day fields and returns a
    # top-level JSON array of {station, day, max_temp_f} rows (verified live 2026-06).
    params = {
        "network": network,
        "stations": iem_id,
        "var": "max_temp_f",
        "format": "json",
        "year1": start.year, "month1": start.month, "day1": start.day,
        "year2": end.year, "month2": end.month, "day2": end.day,
    }
    key = f"iem-daily:{station.series}:{iem_id}:{start}:{end}"
    data = _get_json(IEM_DAILY_URL, params, key, ttl)
    # Response is a bare list; older/alt shapes wrap it under "data".
    rows = data if isinstance(data, list) else (data or {}).get("data", [])
    out: dict[str, float] = {}
    for row in rows or []:
        day = row.get("day") or row.get("date")
        hi = row.get("max_temp_f")
        if day is None or hi is None:
            continue
        try:
            out[str(day)[:10]] = float(hi)
        except (TypeError, ValueError):
            continue
    return out


IEM_AFOS_LIST_URL = "https://mesonet.agron.iastate.edu/api/1/nws/afos/list.json"
IEM_NWSTEXT_URL = "https://mesonet.agron.iastate.edu/api/1/nwstext/"


def _get_text(url: str, key: str, ttl: float | None, timeout: float = 45.0) -> str | None:
    """Like ``_get_json`` but for text/plain responses (NWS text products)."""
    cached = cache.load(key, ttl)
    if cached is not None:
        return cached
    last_exc: Exception | None = None
    for _attempt in range(3):
        try:
            resp = requests.get(url, headers=_UA, timeout=timeout)
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_exc = exc
            continue
        if not resp.ok:
            return None
        cache.store(key, resp.text)
        return resp.text
    if last_exc is not None:
        raise last_exc
    return None


# Every CLI variant carries the climate day on one line, e.g.
# "...THE CENTRAL PARK NY CLIMATE SUMMARY FOR JUNE 30 2026..." — the morning/afternoon
# intraday issuances ("VALID [TODAY] AS OF 0400 PM LOCAL TIME.") and the
# after-midnight final (YESTERDAY sections) alike.
_CLI_DATE_RE = re.compile(r"CLIMATE SUMMARY FOR\s+([A-Z]+)\s+(\d{1,2})\s+(\d{4})")
# In the TEMPERATURE section the first integer after MAXIMUM is the observed value;
# the observation time and the record/normal columns follow it on the same line
# ("MAXIMUM  87  1226 PM  99  1964 ..."). A missing value prints "MM" -> no match.
_CLI_MAX_RE = re.compile(r"^\s*MAXIMUM\s+(-?\d+)\b")
_CLI_MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], start=1)}


def _parse_cli_product(text: str) -> tuple[date, float] | None:
    """Extract ``(climate_day, observed_max_f)`` from one CLI product text.

    Returns None when either is absent (unparseable product, or the max printed
    "MM"/missing) or the value is outside a sane °F range — a mis-parse here would
    become a false settlement floor, so refuse rather than guess.
    """
    up = text.upper()
    m = _CLI_DATE_RE.search(up)
    if not m or m.group(1)[:3] not in _CLI_MONTHS:
        return None
    try:
        day = date(int(m.group(3)), _CLI_MONTHS[m.group(1)[:3]], int(m.group(2)))
    except ValueError:
        return None
    lines = up.splitlines()
    temp_idx = next(
        (i for i, ln in enumerate(lines) if ln.lstrip().startswith("TEMPERATURE")), None)
    if temp_idx is None:
        return None
    for ln in lines[temp_idx:]:
        mm = _CLI_MAX_RE.match(ln)
        if mm:
            value = float(mm.group(1))
            return (day, value) if -60.0 <= value <= 135.0 else None
    return None


def iem_cli_max_so_far_f(
    station: Station,
    target_date: date,
    *,
    ttl: float | None = 600.0,
) -> float | None:
    """Official max-so-far (°F) for the station's climate day, from the NWS CLI.

    The CLI (Climatological Report Daily) is the *settlement instrument*: Kalshi's
    temperature markets resolve to the value this product prints. NWS issues it
    intraday — verified 2026-07-01: a ~4:40–5:45pm local issuance for all four
    cities printing the official high-so-far (AUS also got a morning one) — and a
    final after local midnight with the settled value. Reading it removes the last
    basis between a raw-obs floor and the paying number (tenths-°C conversion,
    1-minute peaks the hourly METARs miss — the 06-29 class of mismatch).

    Mechanics: list the AFOS issuances for the target UTC date *and* the next
    (a late-evening western-city issuance files under the next UTC day), fetch each
    product text (immutable by product_id → cached forever), parse, keep only
    products whose printed climate day equals ``target_date``, and return the value
    from the LATEST issuance — a correction supersedes the original, it does not
    max() with it. Best-effort: returns None on any failure or when no product
    matches yet (most mornings), and callers fall back to raw observations.
    """
    iem_id = getattr(station, "iem_id", "")
    if not iem_id:
        return None
    pil = f"CLI{iem_id.upper()}"
    best_entered, best_value = "", None
    try:
        rows: list[dict] = []
        for d in (target_date, target_date + timedelta(days=1)):
            key = f"afos-list:{pil}:{d.isoformat()}"
            data = _get_json(
                IEM_AFOS_LIST_URL, {"pil": pil, "date": d.isoformat()}, key, ttl)
            rows.extend((data or {}).get("data", []) or [])
        for row in rows:
            pid = str(row.get("product_id") or "")
            if not pid:
                continue
            text = _get_text(f"{IEM_NWSTEXT_URL}{pid}", key=f"nwstext:{pid}", ttl=None)
            parsed = _parse_cli_product(text or "")
            if parsed is None or parsed[0] != target_date:
                continue
            entered = str(row.get("entered") or pid)  # ISO ts; lexicographic == chrono
            if entered > best_entered:
                best_entered, best_value = entered, parsed[1]
    except requests.RequestException:
        return None  # IEM hiccup must never take the raw-obs floor down with it
    return best_value


def _climate_day_utc(tz_name: str, day: date) -> tuple[datetime, datetime]:
    """UTC bounds ``[start, end)`` of the NWS climatological day for a local date.

    The climate day the CLI report (and hence Kalshi settlement) summarizes runs
    midnight-to-midnight LOCAL STANDARD TIME year-round — during DST the boundary
    sits at 1am on the local clock, not midnight. Standard offset = utcoffset - dst,
    probed at local noon so the probe itself can't straddle a transition.
    """
    tz = ZoneInfo(tz_name)
    noon = datetime(day.year, day.month, day.day, 12, tzinfo=tz)
    std_offset = (noon.utcoffset() or timedelta(0)) - (noon.dst() or timedelta(0))
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc) - std_offset
    return start, start + timedelta(hours=24)


#: NWS QC codes marking a temperature as rejected ("X") or questioned ("Q").
_BAD_QC = {"X", "Q"}


def nws_recent_temps_f(
    station: Station,
    target_date: date,
    *,
    ttl: float | None = 600.0,
) -> list[float]:
    """Observed temperatures (°F) reported so far on ``target_date`` at the station.

    Used by the nowcast strategy to compute the observed max-so-far — a hard floor
    on the day's final high that the deterministic "bucket impossible" trade bets
    on at near-full size. A single wrong-day or junk reading therefore becomes a
    confident guaranteed loss, so two filters guard the floor:

      * Only observations whose timestamp falls inside the station's NWS
        climatological day (midnight-to-midnight local STANDARD time — 1am on the
        local clock during DST) are kept. A raw 00:00-UTC window would include
        ~4-8pm of the *previous* local evening for all US stations; after an
        overnight cold front, yesterday evening is warmer than today's whole day
        and would print a false floor.
      * Observations whose temperature failed NWS quality control (rejected /
        questioned) are dropped — one glitched sensor spike is a false floor too.
    """
    start_utc, end_utc = _climate_day_utc(station.tz, target_date)
    key = f"nws-obs:{station.nws_station}:{target_date}"
    data = _get_json(
        f"https://api.weather.gov/stations/{station.nws_station}/observations",
        {"start": start_utc.isoformat(timespec="seconds")}, key, ttl,
    )
    if not data:
        return []
    temps: list[float] = []
    for feat in data.get("features", []):
        props = feat.get("properties", {})
        try:
            ts = datetime.fromisoformat(
                str(props.get("timestamp", "")).replace("Z", "+00:00"))
        except ValueError:
            continue
        if ts.tzinfo is None or not (start_utc <= ts < end_utc):
            continue
        temp = props.get("temperature") or {}
        c = temp.get("value")
        if c is None or str(temp.get("qualityControl", "")).upper() in _BAD_QC:
            continue
        temps.append(c * 9 / 5 + 32)  # API returns Celsius
    return temps
