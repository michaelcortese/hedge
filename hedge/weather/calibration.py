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
    can draw a non-Gaussian shape. Both fall back to the prior when a key is missing
    or under-sampled.
    """

    sigma: dict[tuple[str, int], float] = field(default_factory=dict)
    residuals: dict[tuple[str, int], np.ndarray] = field(default_factory=dict)

    def sigma_for(self, series: str, lead_days: int) -> float:
        return self.sigma.get((series.upper(), max(lead_days, 0)),
                              default_sigma(lead_days))

    def residuals_for(self, series: str, lead_days: int) -> np.ndarray | None:
        return self.residuals.get((series.upper(), max(lead_days, 0)))


def fit_calibration(stations, start, end, *, min_samples: int = 30) -> CalibrationTable:
    """Fit forecast-error spread per city from the historical-forecast archive.

    For each station we pool the signed errors ``ensemble_mean_forecast - realized``
    over ``[start, end]`` and take their std as the calibrated level. We don't have
    a clean per-lead split from the archive, so we anchor the fitted level at lead 1
    and reshape across leads using the prior curve — calibrating the *magnitude* per
    city while keeping the sensible lead-time growth. Cities with too little data
    fall back to the prior entirely.
    """
    from hedge.weather.archive import (
        archive_daily_highs,
        historical_model_highs_range,
    )

    table = CalibrationTable()
    for st in stations:
        realized = archive_daily_highs(st, start, end)
        forecasts = historical_model_highs_range(st, start, end)
        residuals: list[float] = []
        for d_iso, fc in forecasts.items():
            obs = realized.get(d_iso)
            if obs is not None and fc:
                residuals.append(float(np.mean(fc)) - obs)
        if len(residuals) < min_samples:
            continue
        res = np.asarray(residuals, dtype=float)
        sigma1 = float(res.std(ddof=1))
        anchor = default_sigma(1)
        for lead in range(len(_PRIOR_SIGMA_BY_LEAD)):
            table.sigma[(st.series, lead)] = sigma1 * default_sigma(lead) / anchor
            table.residuals[(st.series, lead)] = res
    return table
