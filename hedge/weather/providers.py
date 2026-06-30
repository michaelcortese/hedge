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

from dataclasses import dataclass, field
from datetime import date

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


def nws_recent_temps_f(
    station: Station,
    target_date: date,
    *,
    ttl: float | None = 600.0,
) -> list[float]:
    """Observed temperatures (°F) reported so far on ``target_date`` at the station.

    Used by the nowcast strategy to compute the observed max-so-far (a hard floor
    on the day's final high). Pulls recent observations and keeps those whose
    timestamp falls on the target local day.
    """
    key = f"nws-obs:{station.nws_station}:{target_date}"
    data = _get_json(
        f"https://api.weather.gov/stations/{station.nws_station}/observations",
        {"start": f"{target_date.isoformat()}T00:00:00+00:00"}, key, ttl,
    )
    if not data:
        return []
    temps: list[float] = []
    for feat in data.get("features", []):
        props = feat.get("properties", {})
        c = (props.get("temperature") or {}).get("value")
        if c is None:
            continue
        temps.append(c * 9 / 5 + 32)  # API returns Celsius
    return temps


# --------------------------------------------------------------------------- #
# Raw-response accessors — capture the exact API payloads the model saw.        #
#                                                                               #
# These return the unparsed JSON the parsing functions above reduce, reusing    #
# the SAME cache keys/params so a call within the same cycle is a cache hit (no  #
# extra API request). The runner's market+weather data log emits these verbatim #
# so the research dataset is a byte-faithful record that can replay the model.  #
# --------------------------------------------------------------------------- #
def open_meteo_forecast_raw(
    station: Station, target_date: date, models: list[str] | None = None,
    *, ttl: float | None = _LIVE_TTL,
) -> dict | None:
    """Raw Open-Meteo multi-model forecast JSON (daily high + hourly per model)."""
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
    return _get_json(OPEN_METEO_URL, params, key, ttl)


def nws_forecast_raw(
    station: Station, target_date: date, *, ttl: float | None = _LIVE_TTL,
) -> dict | None:
    """Raw NWS gridpoint forecast JSON (the 12-hour ``periods`` ``nws_forecast`` reads)."""
    pkey = f"nws-points:{station.lat},{station.lon}"
    points = _get_json(
        f"https://api.weather.gov/points/{station.lat},{station.lon}",
        None, pkey, ttl=None,
    )
    if not points:
        return None
    forecast_url = points.get("properties", {}).get("forecast")
    if not forecast_url:
        return None
    fkey = f"nws-forecast:{station.series}:{target_date}"
    return _get_json(forecast_url, None, fkey, ttl)


def nws_observations_raw(
    station: Station, target_date: date, *, ttl: float | None = 600.0,
) -> dict | None:
    """Raw NWS observations feature collection (the obs ``nws_recent_temps_f`` reduces)."""
    key = f"nws-obs:{station.nws_station}:{target_date}"
    return _get_json(
        f"https://api.weather.gov/stations/{station.nws_station}/observations",
        {"start": f"{target_date.isoformat()}T00:00:00+00:00"}, key, ttl,
    )
