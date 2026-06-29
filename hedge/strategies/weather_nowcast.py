"""Intraday nowcast — the highest-edge temperature strategy.

The insight: once the day is underway, the observed max-so-far is a *hard floor* on
the final daily high (the high can only go up from here), and after the afternoon
peak the remaining-day max collapses to a narrow band. Kalshi's bucket prices
routinely lag the live observation, so this is where the mispricing lives.

Mechanics:
  * Pull the temperatures observed so far today (NWS METAR/ASOS) -> ``obs_max``.
  * Take the forecast distribution of the day's high (same ensemble core as
    ``weather_ensemble``), but **truncate it from below at ``obs_max``** — no draw
    may finish below what's already happened.
  * As the day matures ``obs_max`` rises toward the true high and the forecast's own
    spread shrinks, so the distribution sharpens dramatically vs the morning view.

It **abstains before local solar noon** (``min_hour``), when observations carry
little information about the eventual peak and the floor is far below the high.
"""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

from hedge.signal import Signal
from hedge.strategies.base import MarketView, Strategy
from hedge.weather.calibration import CalibrationTable
from hedge.weather.distribution import bucket_prob_and_se
from hedge.weather.markets import parse_temp_market
from hedge.weather.sources import ForecastSource, LiveForecastSource


class WeatherNowcastStrategy(Strategy):
    name = "weather_nowcast"

    def __init__(
        self,
        source: ForecastSource | None = None,
        calibration: CalibrationTable | None = None,
        *,
        n_draws: int = 20_000,
        min_hour: int = 12,
        now: datetime | None = None,
    ):
        # ``now`` is injectable so the backtest can replay a specific time of day.
        self.source = source or LiveForecastSource()
        self.calibration = calibration or CalibrationTable()
        self.n_draws = n_draws
        self.min_hour = min_hour
        self.now = now

    def evaluate(self, market: MarketView) -> Signal | None:
        tm = parse_temp_market(market.raw)
        if tm is None:
            return None

        station_tz = ZoneInfo(tm.station.tz)
        now = (self.now or datetime.now(station_tz)).astimezone(station_tz)
        # Only act on the *current* day, after the morning, when obs are informative.
        if now.date() != tm.local_date or now.hour < self.min_hour:
            return None

        obs_max = self.source.observed_max(tm.station, tm.local_date)
        if obs_max is None:
            return None  # no observations yet -> nothing to nowcast from

        highs = self.source.point_highs(tm.station, tm.local_date)
        # Forecast center must respect the floor: the final high is at least obs_max.
        if not highs:
            highs = [obs_max]
        highs = [max(h, obs_max) for h in highs]

        lead = 0  # same-day
        sigma = self.calibration.sigma_for(tm.series, lead)
        # Intraday the remaining uncertainty is below the all-day error; taper it as
        # the afternoon advances (less day left -> tighter).
        hours_left = max(0, 20 - now.hour)  # ~8pm typical peak-cooling reference
        sigma = sigma * min(1.0, 0.35 + 0.065 * hours_left)
        residuals = self.calibration.residuals_for(tm.series, lead)
        bias = self.calibration.bias_for(tm.series, lead)
        mean_disp = self.calibration.dispersion_for(tm.series, lead)

        p, se = bucket_prob_and_se(
            highs, tm,
            model_sigma=sigma,
            n_draws=self.n_draws,
            seed=abs(hash((tm.ticker, now.hour))) % (2**31),
            residuals=residuals,
            floor_high=obs_max,
            bias=bias,
            mean_dispersion=mean_disp,
        )
        return Signal(
            ticker=tm.ticker,
            prob=p,
            n_draws=self.n_draws,
            std_error=se,
            strategy=self.name,
            meta={
                "city": tm.station.city,
                "local_date": tm.local_date.isoformat(),
                "as_of_hour": now.hour,
                "obs_max": round(obs_max, 1),
                "model_sigma": round(sigma, 2),
                "n_models": len(highs),
                "bucket": [tm.lo_f, tm.hi_f],
            },
        )
