"""Monte Carlo core: probabilities are well-formed and behave sensibly."""

from __future__ import annotations

import math

import numpy as np

from hedge.weather.distribution import (
    bucket_prob_and_se,
    build_distribution,
    clamp_prob,
)
from hedge.weather.markets import parse_temp_market


def _market(lo=None, hi=None, **kw):
    raw = {"ticker": "KXHIGHNY-25JUN28-X", "event_ticker": "KXHIGHNY-25JUN28"}
    if lo is not None:
        raw["floor_strike"] = lo
    if hi is not None:
        raw["cap_strike"] = hi
    raw["strike_type"] = kw.get("strike_type", "between")
    return parse_temp_market(raw)


def test_bucket_probs_sum_to_one_over_complete_grid():
    # A complete partition of the real line must capture ~all probability mass.
    highs = [78, 80, 81, 79, 82]
    dist = build_distribution(highs, model_sigma=3.0, n_draws=50_000, seed=1)
    edges = list(range(60, 101))  # buckets [60,61), ... covering the mass
    total = 0.0
    total += dist.prob_in(-math.inf, 60)
    for lo in edges[:-1]:
        total += dist.prob_in(lo, lo)  # single-degree buckets
    total += dist.prob_in(101, math.inf)
    # account for the 60..100 inclusive single-degree slots
    total = (
        dist.prob_in(-math.inf, 59)
        + sum(dist.prob_in(t, t) for t in range(60, 101))
        + dist.prob_in(101, math.inf)
    )
    assert abs(total - 1.0) < 1e-9


def test_center_bucket_is_most_likely():
    highs = [80, 80, 80, 80]
    dist = build_distribution(highs, model_sigma=3.0, n_draws=50_000, seed=2)
    p_center = dist.prob_in(80, 80)
    p_tail = dist.prob_in(90, 90)
    assert p_center > p_tail


def test_tighter_sigma_concentrates_mass():
    highs = [80, 80, 80]
    tight = build_distribution(highs, model_sigma=1.0, n_draws=50_000, seed=3)
    wide = build_distribution(highs, model_sigma=5.0, n_draws=50_000, seed=3)
    assert tight.prob_in(79, 81) > wide.prob_in(79, 81)


def test_floor_high_lifts_distribution():
    highs = [80, 80, 80]
    # If we've already observed 85, no draw should land below 85.
    dist = build_distribution(highs, model_sigma=4.0, n_draws=20_000, seed=4,
                              floor_high=85)
    assert dist.draws.min() >= 85
    assert dist.prob_in(-math.inf, 84) == 0.0


def test_prob_and_se_in_range_and_clamped():
    m = _market(72, 73)
    p, se = bucket_prob_and_se([72.5, 73, 72, 71.5], m, model_sigma=2.0,
                               n_draws=20_000, seed=5)
    assert 0.0 < p < 1.0
    assert se >= 0.0


def test_empty_bucket_clamped_not_zero():
    m = _market(120, 121)  # absurd for an 80-degree forecast
    p, se = bucket_prob_and_se([80, 80, 80], m, model_sigma=2.0,
                               n_draws=20_000, seed=6)
    assert p > 0.0  # clamped to ~1/(2N), never exactly 0


def test_clamp_prob_bounds():
    assert 0 < clamp_prob(0.0, 1000) < 1
    assert 0 < clamp_prob(1.0, 1000) < 1
    assert clamp_prob(0.5, 1000) == 0.5
