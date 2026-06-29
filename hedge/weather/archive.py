"""Historical weather archive — the truth source for backtests and climatology.

Two free Open-Meteo archives power the tournament:

  * **Reanalysis** (``archive-api.open-meteo.com``, ERA5): the realized daily high
    for any past day. This is "truth" — both the climatology baseline and the
    backtest's outcome labels come from here. (ERA5 lags ~5 days; fine for history.)
  * **Historical Forecast** (``historical-forecast-api.open-meteo.com``): what each
    model *predicted* for a past day at a given lead time. This is what makes a real
    backtest possible — we score yesterday's strategy against what was actually
    forecastable then, not with hindsight. (Wired into the backtest in Phase 3.)

Everything here is immutable history, so it is cached with ``ttl=None`` — fetch
once, replay forever, deterministic backtests.
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache

from hedge.weather.providers import _get_json
from hedge.weather.stations import Station

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HIST_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"


def archive_daily_highs(
    station: Station, start: date, end: date
) -> dict[str, float]:
    """Realized daily-high (°F) per local day in ``[start, end]`` from ERA5.

    Returns ``{iso_date: high_f}``. This is the outcome label for backtests.
    """
    params = {
        "latitude": station.lat,
        "longitude": station.lon,
        "timezone": station.tz,
        "temperature_unit": "fahrenheit",
        "daily": "temperature_2m_max",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
    }
    key = f"era5:{station.series}:{start}:{end}"
    data = _get_json(ARCHIVE_URL, params, key, ttl=None)
    if not data:
        return {}
    daily = data.get("daily", {}) or {}
    days = daily.get("time", []) or []
    highs = daily.get("temperature_2m_max", []) or []
    return {d: float(h) for d, h in zip(days, highs) if h is not None}


def realized_high(station: Station, target_date: date) -> float | None:
    """The single realized daily high (°F) for one local day, or None if unavailable."""
    return archive_daily_highs(station, target_date, target_date).get(
        target_date.isoformat()
    )


def _circular_yday_distance(a: int, b: int) -> int:
    """Distance in days between two day-of-year values, wrapping around New Year."""
    d = abs(a - b)
    return min(d, 365 - d)


@lru_cache(maxsize=256)
def _climatology_cached(
    series: str, lat: float, lon: float, tz: str, nws: str,
    ref_year: int, target_yday: int, years: int, window_days: int,
) -> tuple[float, ...]:
    # Pull the whole multi-year span [ref_year-years .. ref_year-1] in ONE request
    # (cached on disk), then filter to days within ±window of the target calendar
    # day. The span is identical for every target date in a backtest, so this fetch
    # is shared across all days and buckets.
    st = Station(series, "", nws, lat, lon, tz)
    span = archive_daily_highs(st, date(ref_year - years, 1, 1), date(ref_year - 1, 12, 31))
    out: list[float] = []
    for d_iso, high in span.items():
        yday = date.fromisoformat(d_iso).timetuple().tm_yday
        if _circular_yday_distance(yday, target_yday) <= window_days:
            out.append(high)
    return tuple(out)


def climatology_highs(
    station: Station,
    target_date: date,
    *,
    years: int = 20,
    window_days: int = 7,
) -> list[float]:
    """Realized highs (°F) around ``target_date``'s calendar day across past years.

    Builds the climatology predictive distribution: realized highs within
    ``±window_days`` of the same calendar day, over the ``years`` years strictly
    *before* ``target_date`` (referencing the target year, not today, so a backtest
    has no lookahead). This empirical sample is the null model every forecast
    strategy must beat. One cached multi-year fetch backs the whole thing.
    """
    return list(_climatology_cached(
        station.series, station.lat, station.lon, station.tz, station.nws_station,
        target_date.year, target_date.timetuple().tm_yday, years, window_days,
    ))


def historical_model_highs(
    station: Station,
    target_date: date,
    *,
    models: list[str] | None = None,
) -> list[float]:
    """Archived model forecasts of the daily high for a past day (for backtests).

    Uses Open-Meteo's historical-forecast archive so the backtest sees what the
    models actually predicted, not a hindsight reanalysis. One value per model.
    """
    from hedge.weather.providers import DEFAULT_MODELS, _parse_open_meteo

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
    key = f"histfc:{station.series}:{target_date}:{','.join(models)}"
    data = _get_json(HIST_FORECAST_URL, params, key, ttl=None)
    if not data:
        return []
    recs = _parse_open_meteo(data, models, target_date)
    return [r.daily_high_f for r in recs if r.daily_high_f is not None]


def archive_hourly_range(
    station: Station, start: date, end: date
) -> dict[str, dict[int, float]]:
    """Realized hourly temps (°F) per local day: ``{iso_date: {hour: temp}}``.

    Backs the nowcast backtest: replaying "what had been observed by hour H" needs
    the intraday curve. One ranged request covers the window (cached forever).
    """
    params = {
        "latitude": station.lat,
        "longitude": station.lon,
        "timezone": station.tz,
        "temperature_unit": "fahrenheit",
        "hourly": "temperature_2m",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
    }
    key = f"era5-hourly:{station.series}:{start}:{end}"
    data = _get_json(ARCHIVE_URL, params, key, ttl=None)
    if not data:
        return {}
    hourly = data.get("hourly", {}) or {}
    times = hourly.get("time", []) or []
    temps = hourly.get("temperature_2m", []) or []
    out: dict[str, dict[int, float]] = {}
    for t, temp in zip(times, temps):
        if temp is None:
            continue
        d_iso, hh = t[:10], int(t[11:13])
        out.setdefault(d_iso, {})[hh] = float(temp)
    return out


def historical_model_highs_range(
    station: Station,
    start: date,
    end: date,
    *,
    models: list[str] | None = None,
) -> dict[str, list[float]]:
    """Archived per-model daily-high forecasts for every day in ``[start, end]``.

    One request covers the whole window (vs one per day), which keeps calibration
    fitting cheap. Returns ``{iso_date: [high_per_model, ...]}``.
    """
    from hedge.weather.providers import DEFAULT_MODELS

    models = models or DEFAULT_MODELS
    params = {
        "latitude": station.lat,
        "longitude": station.lon,
        "timezone": station.tz,
        "temperature_unit": "fahrenheit",
        "daily": "temperature_2m_max",
        "models": ",".join(models),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
    }
    key = f"histfc-range:{station.series}:{start}:{end}:{','.join(models)}"
    data = _get_json(HIST_FORECAST_URL, params, key, ttl=None)
    if not data:
        return {}
    daily = data.get("daily", {}) or {}
    days = daily.get("time", []) or []
    multi = len(models) > 1
    # Collect each model's series, then transpose into per-day lists.
    per_model: list[list] = []
    for model in models:
        suffix = f"_{model}" if multi else ""
        per_model.append(daily.get(f"temperature_2m_max{suffix}", []) or [])
    out: dict[str, list[float]] = {}
    for i, d in enumerate(days):
        vals = [float(col[i]) for col in per_model
                if i < len(col) and col[i] is not None]
        if vals:
            out[d] = vals
    return out
