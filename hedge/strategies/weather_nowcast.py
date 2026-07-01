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

The *probabilistic* path **abstains until mid-afternoon** (``min_hour``, ~2pm local),
when observations carry little information about the eventual peak and the floor is far
below the high — this is where the durable obs-lag edge lives, so morning cycles add
only fee bleed. The *deterministic* path (a bucket the observed max has already logically
settled) runs ALL DAY: the floor is valid from the first observation, and on
frontal-passage days the high locks in before noon.
"""

from __future__ import annotations

import math
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
        min_hour: int = 14,
        now: datetime | None = None,
    ):
        # ``now`` is injectable so the backtest can replay a specific time of day.
        self.source = source or LiveForecastSource()
        self.calibration = calibration or CalibrationTable()
        self.n_draws = n_draws
        self.min_hour = min_hour
        self.now = now

    #: Probability used for a logically-settled outcome (clamped off 0/1, which the
    #: Signal contract forbids). Tiny, so the engine treats it as near-certain.
    _DET_EPS = 1e-4

    def _deterministic_signal(self, tm, obs_max: float, now) -> Signal | None:
        """Return a deterministic Signal if the observed max already settles the bucket.

        ``obs_round`` is the observed max rounded the way NWS settles (floor(x+0.5)).
        Requires a validated station — an unvalidated station map could make a wrong
        "impossible" call a confident-wrong NO. Returns None when the bucket is not yet
        logically decided (the usual case), so the probabilistic path runs.
        """
        if not tm.station.validated:
            return None
        obs_round = math.floor(obs_max + 0.5)
        prob = None
        if math.isfinite(tm.hi_f) and obs_round > tm.hi_f:
            prob = self._DET_EPS                      # YES impossible (max already above)
        elif math.isinf(tm.hi_f) and obs_round >= tm.lo_f:
            prob = 1.0 - self._DET_EPS                # YES certain ("X or above" met)
        if prob is None:
            return None
        return Signal(
            ticker=tm.ticker, prob=prob, n_draws=self.n_draws, std_error=1e-6,
            strategy=self.name, deterministic=True,
            meta={
                "city": tm.station.city,
                "local_date": tm.local_date.isoformat(),
                "as_of_hour": now.hour,
                "obs_max": round(obs_max, 1),
                "obs_round": obs_round,
                "bucket": [tm.lo_f, tm.hi_f],
                "deterministic": "impossible" if prob < 0.5 else "certain",
            },
        )

    def evaluate(self, market: MarketView) -> Signal | None:
        tm = parse_temp_market(market.raw)
        if tm is None:
            return None

        station_tz = ZoneInfo(tm.station.tz)
        now = (self.now or datetime.now(station_tz)).astimezone(station_tz)
        # Only act on the *current* local day.
        if now.date() != tm.local_date:
            return None

        obs_max = self.source.observed_max(tm.station, tm.local_date)

        # Deterministic "impossible/certain bucket": the day's max can only rise, and
        # settlement rounds the high to a whole °F (NWS: floor(x+0.5)). So once the
        # observed max-so-far ALREADY rounds above a bucket's top, YES is impossible
        # (buy NO at near-certainty); once it meets an "X or above" threshold, YES is
        # already certain. This is the cleanest who's-on-the-other-side trade — a slow
        # book still bidding a settled-impossible bucket. Gated to validated stations:
        # a wrong station map would make "impossible" a confident-wrong NO.
        #
        # Checked ALL DAY, before the min_hour gate: the floor logic is valid from the
        # first observation of the climate day, and on frontal-passage days (temps
        # falling since morning) the high locks in hours before the afternoon window —
        # buckets below it are dead all day while the book still bids them. min_hour
        # exists to keep the *probabilistic* path out of the uninformative morning; it
        # was never a correctness condition for the deterministic one. (Safe only
        # because nws_recent_temps_f filters obs to the local climate day — yesterday
        # evening's temps polluting obs_max was exactly the false-floor bug.)
        if obs_max is not None:
            det = self._deterministic_signal(tm, obs_max, now)
            if det is not None:
                return det

        # The probabilistic path needs observations and stays gated to the afternoon:
        # morning obs carry little information about the eventual peak, so morning
        # cycles would add only fee bleed.
        if obs_max is None or now.hour < self.min_hour:
            return None

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
