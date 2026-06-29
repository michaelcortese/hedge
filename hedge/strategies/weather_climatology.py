"""Climatology baseline — the null model every real strategy must beat.

This strategy ignores the forecast entirely. It builds the predictive distribution
of today's high from *history alone*: the realized highs on the same calendar day
(±a window) across the past ~20 years. The empirical sample of those highs is the
distribution; ``P(YES)`` is the fraction that landed in the bucket.

It exists for two reasons:
  1. **Tournament control.** A forecast strategy that can't beat climatology on
     Brier/log-loss has no edge and must not be trusted with size. Climatology is
     the bar.
  2. **Sanity floor.** When forecasts are unavailable it still produces a sane,
     honestly-wide signal rather than abstaining blindly.

Its honest uncertainty is large (a ~300-sample empirical distribution), so the
sizing engine bets it small — exactly right for a no-edge baseline.
"""

from __future__ import annotations

from datetime import date

import numpy as np

from hedge.signal import Signal
from hedge.strategies.base import MarketView, Strategy
from hedge.weather.archive import climatology_highs
from hedge.weather.distribution import HighTempDistribution, clamp_prob
from hedge.weather.markets import parse_temp_market


class WeatherClimatologyStrategy(Strategy):
    name = "weather_climatology"

    def __init__(self, *, years: int = 20, window_days: int = 7):
        self.years = years
        self.window_days = window_days

    def evaluate(self, market: MarketView) -> Signal | None:
        tm = parse_temp_market(market.raw)
        if tm is None:
            return None

        highs = climatology_highs(
            tm.station, tm.local_date,
            years=self.years, window_days=self.window_days,
        )
        if len(highs) < 30:
            return None  # too little history to form a baseline

        draws = np.rint(np.asarray(highs, dtype=float)).astype(int)
        dist = HighTempDistribution(
            draws=draws, center=float(draws.mean()), sigma=float(draws.std(ddof=1))
        )
        p_raw = dist.prob_for_market(tm)
        n = draws.size
        se = float(np.sqrt(max(p_raw * (1 - p_raw), 1e-9) / n))
        p = clamp_prob(p_raw, n)

        return Signal(
            ticker=tm.ticker,
            prob=p,
            n_draws=n,
            std_error=se,
            strategy=self.name,
            meta={
                "city": tm.station.city,
                "local_date": tm.local_date.isoformat(),
                "n_years": self.years,
                "n_samples": int(n),
                "clim_mean": round(dist.center, 1),
                "clim_std": round(dist.sigma, 1),
                "bucket": [tm.lo_f, tm.hi_f],
            },
        )
