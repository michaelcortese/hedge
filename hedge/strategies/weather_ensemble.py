"""Multi-model forecast ensemble — the all-day workhorse temperature strategy.

Blends the daily-high forecasts from several independent models (GFS, ECMWF, ICON,
GEM via Open-Meteo, plus the NWS gridpoint) into one Monte Carlo predictive
distribution, then reads off ``P(YES)`` for each Kalshi bucket. It is the only
strategy that has an opinion early in the day (before observations constrain the
high), and it is the calibration backbone the others lean on.

This file stays thin per the repo contract: all data + math lives in
``hedge/weather/``. ``evaluate`` only wires a ``MarketView`` to that machinery and
returns a ``Signal``. Data access goes through an injectable ``ForecastSource`` so
the exact same code runs live and inside the backtest tournament.
"""

from __future__ import annotations

from datetime import date

from hedge.signal import Signal
from hedge.strategies.base import MarketView, Strategy
from hedge.weather.calibration import CalibrationTable
from hedge.weather.distribution import bucket_prob_and_se
from hedge.weather.markets import TempMarket, parse_temp_market
from hedge.weather.sources import ForecastSource, LiveForecastSource


def _seed_from(market: TempMarket) -> int:
    """Deterministic per-market seed so ``evaluate`` is reproducible/backtestable."""
    return abs(hash((market.ticker, market.local_date.isoformat()))) % (2**31)


class WeatherEnsembleStrategy(Strategy):
    name = "weather_ensemble"

    def __init__(
        self,
        source: ForecastSource | None = None,
        calibration: CalibrationTable | None = None,
        *,
        n_draws: int = 20_000,
        as_of: date | None = None,
    ):
        # ``as_of`` is the "today" used to compute lead time; injectable for backtests.
        self.source = source or LiveForecastSource()
        self.calibration = calibration or CalibrationTable()
        self.n_draws = n_draws
        self.as_of = as_of

    def evaluate(self, market: MarketView) -> Signal | None:
        tm = parse_temp_market(market.raw)
        if tm is None:
            return None  # not a covered temperature market

        highs = self.source.point_highs(tm.station, tm.local_date)
        if len(highs) < 2:
            return None  # too few models to form an ensemble -> abstain

        today = self.as_of or date.today()
        lead = (tm.local_date - today).days
        sigma = self.calibration.sigma_for(tm.series, lead)
        residuals = self.calibration.residuals_for(tm.series, lead)
        bias = self.calibration.bias_for(tm.series, lead)
        mean_disp = self.calibration.dispersion_for(tm.series, lead)

        # Fold the grid→station settlement basis into the spread: this is a pure
        # forecast-grid estimate, so the difference between the forecast point and the
        # NWS/ASOS station Kalshi settles on is real, uncaptured uncertainty. (The
        # nowcast does NOT add this — its obs floor is already read off the settlement
        # station, so it would be a double penalty there.)
        settle_sigma = self.calibration.settlement_sigma_for(tm.series)

        p, se = bucket_prob_and_se(
            highs, tm,
            model_sigma=sigma,
            n_draws=self.n_draws,
            seed=_seed_from(tm),
            residuals=residuals,
            bias=bias,
            mean_dispersion=mean_disp,
            settlement_sigma_f=settle_sigma,
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
                "lead_days": lead,
                "n_models": len(highs),
                "model_highs": [round(h, 1) for h in highs],
                "model_sigma": round(sigma, 2),
                "settlement_sigma": round(settle_sigma, 2),
                "bucket": [tm.lo_f, tm.hi_f],
            },
        )
