"""Forecast sources — the seam that lets one strategy run live *and* in backtest.

A weather strategy never calls ``requests`` directly. It asks a ``ForecastSource``
for "the model point-forecasts of the daily high for this station/date" and "the
temperatures observed so far". In production that's ``LiveForecastSource`` (hits
Open-Meteo + NWS via ``providers.py``); in the tournament the backtester injects an
archive-backed source replaying what the models said on that historical day. Same
strategy code, swappable data — which is the whole point of the Signal contract.
"""

from __future__ import annotations

from datetime import date
from typing import Protocol

from hedge.weather import providers
from hedge.weather.providers import DEFAULT_MODELS
from hedge.weather.stations import Station


class ForecastSource(Protocol):
    """What a weather strategy needs from the data layer."""

    def point_highs(self, station: Station, target_date: date) -> list[float]:
        """One forecast of the daily high (°F) per model/source. Empty -> abstain."""
        ...

    def observed_max(self, station: Station, target_date: date) -> float | None:
        """Max temperature (°F) observed so far on the local day, or None."""
        ...


class LiveForecastSource:
    """Production source: Open-Meteo multi-model + NWS gridpoint, plus live obs."""

    def __init__(self, models: list[str] | None = None):
        self.models = models or DEFAULT_MODELS

    def point_highs(self, station: Station, target_date: date) -> list[float]:
        highs: list[float] = []
        for rec in providers.open_meteo_forecast(station, target_date, self.models):
            if rec.daily_high_f is not None:
                highs.append(rec.daily_high_f)
        nws = providers.nws_forecast(station, target_date)
        if nws is not None and nws.daily_high_f is not None:
            highs.append(nws.daily_high_f)
        return highs

    def observed_max(self, station: Station, target_date: date) -> float | None:
        temps = providers.nws_recent_temps_f(station, target_date)
        return max(temps) if temps else None


class ArchiveForecastSource:
    """Backtest source: replays what the models forecast for a *past* day.

    Injected into the same strategy code the runner uses live, so a backtest scores
    the real strategy — not a reimplementation. ``observed_max`` returns None here
    (full-day forecast backtest); the nowcast's intraday backtest uses a dedicated
    source added in Phase 4.
    """

    def __init__(
        self,
        models: list[str] | None = None,
        preloaded: dict[str, list[float]] | None = None,
    ):
        # ``preloaded`` maps iso-date -> per-model highs (from
        # ``historical_model_highs_range``) so a whole backtest window costs one
        # request per city instead of one per day.
        self.models = models or DEFAULT_MODELS
        self.preloaded = preloaded

    def point_highs(self, station: Station, target_date: date) -> list[float]:
        if self.preloaded is not None:
            return self.preloaded.get(target_date.isoformat(), [])
        from hedge.weather.archive import historical_model_highs

        return historical_model_highs(station, target_date, models=self.models)

    def observed_max(self, station: Station, target_date: date) -> float | None:
        return None


class HistoricalIntradaySource:
    """Backtest source for the nowcast: replays the day *as of a given hour*.

    ``observed_max`` returns the realized max temperature up to ``as_of_hour`` from
    archived hourly data; ``point_highs`` returns the archived model forecasts. Both
    read from preloaded ranged dicts so a whole backtest costs two requests per city.
    """

    def __init__(
        self,
        forecast_highs: dict[str, list[float]],
        hourly_by_day: dict[str, dict[int, float]],
        as_of_hour: int,
    ):
        self.forecast_highs = forecast_highs
        self.hourly_by_day = hourly_by_day
        self.as_of_hour = as_of_hour

    def point_highs(self, station: Station, target_date: date) -> list[float]:
        return self.forecast_highs.get(target_date.isoformat(), [])

    def observed_max(self, station: Station, target_date: date) -> float | None:
        hours = self.hourly_by_day.get(target_date.isoformat(), {})
        seen = [t for h, t in hours.items() if h <= self.as_of_hour]
        return max(seen) if seen else None
