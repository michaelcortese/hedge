"""The Monte Carlo core: forecasts -> predictive distribution -> bucket P(YES).

Given a set of model point-forecasts for a station's daily high (plus a calibrated
forecast-error spread), this draws a Monte Carlo sample of the *official rounded
daily high* and counts the fraction landing in each Kalshi bucket. That fraction
is the strategy's ``P(YES)``.

Two sources of spread, kept distinct on purpose:
  * **model error** — how far the blended forecast typically lands from the realized
    high at this lead time. This dominates and comes from ``calibration.py`` (or a
    sane default before calibration is fit). It is what actually widens the buckets.
  * **inter-model dispersion** — how much the ensemble members disagree *right now*.
    A useful same-day signal of difficulty, blended in as a floor.

Two sources of *estimate uncertainty* (the ``std_error`` we report to the sizing
engine, per the Signal contract — over-confidence gets over-bet):
  * **sampling** error from finite draws: ``sqrt(p(1-p)/n_draws)``.
  * **parameter** error from not knowing the true center: propagated by re-pricing
    the bucket at center ± SE(center). Honest uncertainty > false precision.

Rounding matters: NWS reports whole °F and Kalshi settles on that, so we round each
draw to the nearest integer before bucketing. A bucket "72° to 73°" resolves YES iff
the rounded high is 72 or 73 — inclusive integer containment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hedge.weather.markets import TempMarket

#: Fallback forecast-error spread (°F) used before ``calibration.py`` has fit a
#: lead-time/city-specific value. ~3°F is a reasonable next-day high error.
DEFAULT_MODEL_SIGMA_F = 3.0


@dataclass
class HighTempDistribution:
    """A Monte Carlo sample of the official rounded daily high (integer °F)."""

    draws: np.ndarray   # shape (n_draws,), rounded integer highs
    center: float       # blended forecast center (pre-rounding), for logging
    sigma: float        # total spread used (°F), for logging

    @property
    def n_draws(self) -> int:
        return int(self.draws.size)

    def prob_in(self, lo_f: float, hi_f: float) -> float:
        """Fraction of draws with ``lo_f <= high <= hi_f`` (inclusive)."""
        mask = (self.draws >= lo_f) & (self.draws <= hi_f)
        return float(np.count_nonzero(mask) / self.draws.size)

    def prob_for_market(self, market: TempMarket) -> float:
        return self.prob_in(market.lo_f, market.hi_f)

    def mean(self) -> float:
        return float(self.draws.mean())

    def quantile(self, q: float) -> float:
        return float(np.quantile(self.draws, q))


def build_distribution(
    point_highs: list[float],
    *,
    model_sigma: float = DEFAULT_MODEL_SIGMA_F,
    n_draws: int = 20_000,
    seed: int | None = None,
    residuals: np.ndarray | None = None,
    floor_high: float | None = None,
) -> HighTempDistribution:
    """Build a predictive distribution of the rounded daily high.

    Args:
        point_highs: one forecast of the daily high per model/source.
        model_sigma: calibrated forecast-error std (°F). Combined in quadrature
            with inter-model dispersion to set the total spread.
        n_draws: Monte Carlo sample size (sets sampling std error downstream).
        seed: RNG seed for reproducibility (strategies seed from market+date).
        residuals: optional empirical forecast-error sample (°F) to draw the shape
            from instead of a Gaussian — captures skew/fat tails when available.
        floor_high: optional hard lower bound on the final high (the nowcast's
            observed max-so-far). Draws below it are lifted to it: the day's max
            cannot end up below what's already been observed.
    """
    if not point_highs:
        raise ValueError("need at least one point forecast")
    rng = np.random.default_rng(seed)
    arr = np.asarray(point_highs, dtype=float)
    center = float(arr.mean())
    dispersion = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    sigma = float(np.hypot(model_sigma, dispersion))

    if residuals is not None and residuals.size >= 30:
        # Resample empirical residuals, scaled so their std matches `sigma`.
        res = np.asarray(residuals, dtype=float)
        res = (res - res.mean()) / (res.std(ddof=1) or 1.0) * sigma
        samples = center + rng.choice(res, size=n_draws, replace=True)
    else:
        samples = rng.normal(center, sigma, size=n_draws)

    if floor_high is not None:
        samples = np.maximum(samples, floor_high)

    draws = np.rint(samples).astype(int)
    return HighTempDistribution(draws=draws, center=center, sigma=sigma)


def _center_se(point_highs: list[float], model_sigma: float) -> float:
    """Std error of the blended center: combines spread-of-the-mean across models
    with the irreducible model error. With few members this is crude but honest."""
    arr = np.asarray(point_highs, dtype=float)
    n = max(arr.size, 1)
    ens_se = (arr.std(ddof=1) / np.sqrt(n)) if arr.size > 1 else model_sigma
    # Don't let a falsely-agreeing ensemble claim near-zero center error.
    return float(max(ens_se, model_sigma / np.sqrt(n)))


def bucket_prob_and_se(
    point_highs: list[float],
    market: TempMarket,
    *,
    model_sigma: float = DEFAULT_MODEL_SIGMA_F,
    n_draws: int = 20_000,
    seed: int | None = None,
    residuals: np.ndarray | None = None,
    floor_high: float | None = None,
) -> tuple[float, float]:
    """Return ``(p, std_error)`` for a market's YES, folding both uncertainty terms.

    ``p`` is clamped into the open interval ``(0, 1)`` the Signal contract requires;
    a bucket that no draw hit is reported as ``~1/(2*n_draws)``, not 0 — an honest
    "very unlikely, not impossible".
    """
    dist = build_distribution(
        point_highs, model_sigma=model_sigma, n_draws=n_draws,
        seed=seed, residuals=residuals, floor_high=floor_high,
    )
    p = dist.prob_for_market(market)

    # Sampling component.
    sampling_se = float(np.sqrt(max(p * (1 - p), 1e-9) / n_draws))

    # Parameter component: re-price at center +/- SE(center), same shape/spread.
    se_center = _center_se(point_highs, model_sigma)
    shifted = [h + se_center for h in point_highs], [h - se_center for h in point_highs]
    p_hi = build_distribution(shifted[0], model_sigma=model_sigma, n_draws=n_draws,
                              seed=seed, residuals=residuals,
                              floor_high=floor_high).prob_for_market(market)
    p_lo = build_distribution(shifted[1], model_sigma=model_sigma, n_draws=n_draws,
                              seed=seed, residuals=residuals,
                              floor_high=floor_high).prob_for_market(market)
    param_se = abs(p_hi - p_lo) / 2.0

    se = float(np.hypot(sampling_se, param_se))
    p = clamp_prob(p, n_draws)
    return p, se


def clamp_prob(p: float, n_draws: int) -> float:
    """Keep ``p`` strictly inside ``(0, 1)`` as the Signal contract requires."""
    eps = 0.5 / n_draws
    return min(max(p, eps), 1.0 - eps)
