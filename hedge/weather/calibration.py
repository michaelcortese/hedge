"""Forecast-error calibration: how wide should the buckets be?

The Monte Carlo core needs a forecast-error spread (``model_sigma``, °F) per
station and lead time. This module provides it in two layers:

  * **Prior** (``default_sigma``): a hand-set lead-time curve so strategies run
    *before* any historical fitting exists. Same-day forecasts are tight; error
    grows with lead time.
  * **Fitted** (``CalibrationTable``): residual std fit from the Open-Meteo
    historical-forecast archive vs realized highs (wired in Phase 3 via
    ``archive.py``). Falls back to the prior wherever data is thin.

Keeping calibration separate from the strategies is deliberate: a strategy that
reports an honestly-wide ``model_sigma`` gets sized correctly, while one that
claims false precision gets over-bet (CLAUDE.md, "Why std error matters").
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Prior forecast-error std (°F) by lead time in days. Index 0 == same-day.
_PRIOR_SIGMA_BY_LEAD = [2.0, 3.0, 3.8, 4.5, 5.2, 6.0]

# Hard floor (°F) on any fitted forecast-error std. The fitted σ collapsed to ~1.0-1.4°F
# because the forecast feed and the ERA5 "truth" it was scored against are BOTH
# Open-Meteo grid products — correlated, so the residual std understates the true error
# vs the independent NWS/ASOS station Kalshi settles on. A too-tight σ makes the MC pmf
# spike on a single 1°-wide bucket, which fractional Kelly then over-bets (the 2026-06-29
# loss). A sane next-day high error is ~2-3°F; never let the fit claim more precision than
# this floor without evidence. See CLAUDE.md "Why std error matters".
SIGMA_FLOOR_F = 2.0

# Default extra std (°F) for the grid→station settlement basis when calibrating against
# ERA5 (the default): the forecast/ERA5 grid point differs from the NWS/ASOS station
# Kalshi settles on by a structured per-city amount the forecast spread doesn't capture.
# Measured 2026-06 over 15 days: |mean basis| ~1.3-1.5°F with daily gaps to ~4.6°F. Folded
# into the predictive distribution so forecast-grid strategies don't claim a precision the
# settlement instrument doesn't support. Set to 0 when truth="station" (then the basis is
# absorbed into the fitted bias/residuals directly).
SETTLEMENT_SIGMA_DEFAULT_F = 1.5


def default_sigma(lead_days: int) -> float:
    """Lead-time prior for forecast-error std (°F). Clamped to the table ends."""
    if lead_days < 0:
        lead_days = 0
    if lead_days >= len(_PRIOR_SIGMA_BY_LEAD):
        return _PRIOR_SIGMA_BY_LEAD[-1]
    return _PRIOR_SIGMA_BY_LEAD[lead_days]


@dataclass
class CalibrationTable:
    """Fitted forecast-error spread (and optional residual samples) per (series, lead).

    ``sigma`` maps ``(series, lead_days) -> std(°F)``. ``residuals`` optionally maps
    the same key to a sample of signed errors (predicted - realized) so the MC core
    can draw a non-Gaussian shape. ``bias`` is the *mean* signed error per key (the
    systematic warm/cold tendency the MC core must correct for); ``dispersion`` is the
    average inter-model spread over the fit window, so the MC core can add only the
    *excess* same-day disagreement instead of double-counting spread already baked
    into ``sigma``. All fall back sensibly when a key is missing or under-sampled.
    """

    sigma: dict[tuple[str, int], float] = field(default_factory=dict)
    residuals: dict[tuple[str, int], np.ndarray] = field(default_factory=dict)
    bias: dict[tuple[str, int], float] = field(default_factory=dict)
    dispersion: dict[tuple[str, int], float] = field(default_factory=dict)
    #: Per-series grid→station settlement basis std (°F). Missing -> the conservative
    #: default, so even an unfit/empty table gives forecast strategies a basis cushion.
    settlement_sigma: dict[str, float] = field(default_factory=dict)

    def sigma_for(self, series: str, lead_days: int) -> float:
        return self.sigma.get((series.upper(), max(lead_days, 0)),
                              default_sigma(lead_days))

    def settlement_sigma_for(self, series: str,
                             default: float = SETTLEMENT_SIGMA_DEFAULT_F) -> float:
        """Grid→station settlement-basis std (°F) for ``series`` (see
        :data:`SETTLEMENT_SIGMA_DEFAULT_F`). Defaults conservatively when unfit."""
        return self.settlement_sigma.get(series.upper(), default)

    def residuals_for(self, series: str, lead_days: int) -> np.ndarray | None:
        return self.residuals.get((series.upper(), max(lead_days, 0)))

    def bias_for(self, series: str, lead_days: int) -> float:
        """Mean signed error ``predicted - realized`` (°F), or 0.0 if unfit."""
        return self.bias.get((series.upper(), max(lead_days, 0)), 0.0)

    def dispersion_for(self, series: str, lead_days: int) -> float | None:
        """Average inter-model spread (°F) over the fit window, or None if unfit."""
        return self.dispersion.get((series.upper(), max(lead_days, 0)))


def fit_calibration(stations, start, end, *, min_samples: int = 30,
                    sigma_floor: float = SIGMA_FLOOR_F,
                    truth: str = "era5") -> CalibrationTable:
    """Fit forecast-error spread per city from the historical-forecast archive.

    For each station we pool the signed errors ``ensemble_mean_forecast - realized``
    over ``[start, end]`` and take their std as the calibrated *spread* and their mean
    as the systematic *bias*. We don't have a clean per-lead split from the archive, so
    we anchor the fitted spread at lead 1 and reshape across leads using the prior
    curve — calibrating the *magnitude* per city while keeping the sensible lead-time
    growth. The bias is a physical offset of the forecast itself (a station that reads
    warm reads warm at every lead), so it is stored lead-independent, not stretched by
    the spread curve. Cities with too little data fall back to the prior entirely.

    Args:
        sigma_floor: hard lower bound (°F) on every fitted σ (see :data:`SIGMA_FLOOR_F`).
            Prevents the correlated-source collapse from producing over-confident,
            over-bet buckets.
        truth: realized-high source to score forecasts against. ``"era5"`` (default)
            uses the ERA5 grid reanalysis; ``"station"`` uses the IEM ASOS station
            daily-max — the instrument Kalshi actually settles on — so the fitted bias
            absorbs the grid→station offset (MIA ran ~3°F cold vs settlement on
            2026-06-29). ``"station"`` falls back to ERA5 per-city if the IEM feed is
            empty, and MUST be validated against resolved settlements before driving
            real size (a subtler wrong-truth bias is still a bias).
    """
    from hedge.weather.archive import (
        archive_daily_highs,
        historical_model_highs_range,
    )
    from hedge.weather.providers import iem_daily_max_f

    def _realized(st) -> dict[str, float]:
        if truth == "station":
            obs = iem_daily_max_f(st, start, end)
            if obs:
                return obs
        return archive_daily_highs(st, start, end)

    table = CalibrationTable()
    for st in stations:
        realized = _realized(st)
        forecasts = historical_model_highs_range(st, start, end)
        residuals: list[float] = []
        dispersions: list[float] = []
        for d_iso, fc in forecasts.items():
            obs = realized.get(d_iso)
            if obs is not None and fc:
                residuals.append(float(np.mean(fc)) - obs)
                if len(fc) > 1:
                    dispersions.append(float(np.std(fc, ddof=1)))
        if len(residuals) < min_samples:
            continue
        res = np.asarray(residuals, dtype=float)
        sigma1 = float(res.std(ddof=1))
        bias1 = float(res.mean())
        mean_disp = float(np.mean(dispersions)) if dispersions else 0.0
        anchor = default_sigma(1)
        for lead in range(len(_PRIOR_SIGMA_BY_LEAD)):
            # Floor each lead's σ: the correlated-source fit understates true error vs
            # the settlement station, so never let it drop below the sane prior floor.
            scaled = sigma1 * default_sigma(lead) / anchor
            table.sigma[(st.series, lead)] = max(scaled, sigma_floor)
            table.residuals[(st.series, lead)] = res
            table.bias[(st.series, lead)] = bias1
            table.dispersion[(st.series, lead)] = mean_disp
        # When fit against the settlement station the basis is already inside the bias
        # and residual spread, so no extra settlement term; against ERA5 it is not, so
        # carry the conservative default. (Keyed per series even if equal, so callers
        # read a populated table rather than always hitting the default.)
        table.settlement_sigma[st.series] = (
            0.0 if truth == "station" else SETTLEMENT_SIGMA_DEFAULT_F)
    return table
