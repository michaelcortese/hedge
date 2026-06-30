"""Blend — ensemble early in the day, nowcast once observations bite.

The ensemble has an opinion all day but is widest in the morning; the nowcast is
sharp but only after the observed max-so-far becomes informative (afternoon). The
blend is the strategy you'd actually *run*: it defers to the nowcast when the
nowcast is willing to act, and falls back to the ensemble otherwise. Both
components share the same Monte Carlo core and calibration, so the handoff is
apples-to-apples.

It's intentionally a thin orchestrator over the other two strategies — no new
modeling, just the routing logic, which keeps the tournament's attribution clean
(the blend's score is exactly "ensemble-then-nowcast", nothing hidden).
"""

from __future__ import annotations

from datetime import date, datetime

from hedge.signal import Signal
from hedge.strategies.base import MarketView, Strategy
from hedge.strategies.weather_ensemble import WeatherEnsembleStrategy
from hedge.strategies.weather_nowcast import WeatherNowcastStrategy
from hedge.weather.calibration import CalibrationTable
from hedge.weather.sources import ForecastSource, LiveForecastSource


class WeatherBlendStrategy(Strategy):
    name = "weather_blend"

    def __init__(
        self,
        source: ForecastSource | None = None,
        calibration: CalibrationTable | None = None,
        *,
        n_draws: int = 20_000,
        as_of: date | None = None,
        now: datetime | None = None,
        min_hour: int = 14,
    ):
        source = source or LiveForecastSource()
        calibration = calibration or CalibrationTable()
        self.nowcast = WeatherNowcastStrategy(
            source, calibration, n_draws=n_draws, min_hour=min_hour, now=now,
        )
        self.ensemble = WeatherEnsembleStrategy(
            source, calibration, n_draws=n_draws, as_of=as_of,
        )

    def evaluate(self, market: MarketView) -> Signal | None:
        # Prefer the sharper nowcast whenever it is willing to act today.
        sig = self.nowcast.evaluate(market)
        if sig is not None:
            return Signal(
                ticker=sig.ticker, prob=sig.prob, n_draws=sig.n_draws,
                std_error=sig.std_error, strategy=self.name,
                meta={**sig.meta, "via": "nowcast"},
                deterministic=sig.deterministic,
            )
        sig = self.ensemble.evaluate(market)
        if sig is not None:
            return Signal(
                ticker=sig.ticker, prob=sig.prob, n_draws=sig.n_draws,
                std_error=sig.std_error, strategy=self.name,
                meta={**sig.meta, "via": "ensemble"},
            )
        return None
