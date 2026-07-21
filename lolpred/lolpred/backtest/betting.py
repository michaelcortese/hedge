"""Betting layer: odds math, synthetic bookmaker, bet selection, settlement.

Implements contract section 5 (docs/CONTRACTS.md). All probabilities are
P(blue wins) unless a function says otherwise. Decimal odds are European
style: a winning 1-unit stake returns ``odds`` units (profit ``odds - 1``).

Conventions
-----------
- "Vigged implied" probability = ``1 / decimal_odds`` (what you actually pay).
- "Fair" probability = de-vigged (booksum normalized to 1).
- Edges are computed against the *vigged* implied probability, because that is
  the price you transact at; a model that only beats vigged prices has no edge
  vs. the fair line, but the stake you risk is priced with the vig included.
- Every stochastic path takes a seed (``np.random.default_rng(seed)``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "implied_prob",
    "devig_proportional",
    "devig_shin",
    "kelly_fraction",
    "make_synthetic_odds",
    "select_bets",
    "settle_bets",
    "simulate_bankroll",
    "bootstrap_roi_ci",
]


def implied_prob(decimal_odds):
    """Vigged implied probability of decimal odds: ``1 / odds``.

    Accepts scalars or array-likes; returns the same shape.
    """
    return 1.0 / np.asarray(decimal_odds, dtype=float) if np.ndim(decimal_odds) else 1.0 / float(decimal_odds)


def devig_proportional(p_imp_a, p_imp_b):
    """De-vig two implied probabilities proportionally.

    Divides each by the booksum so the pair sums to 1 while preserving the
    ratio ``p_a / p_b``. Accepts scalars or arrays; returns a tuple
    ``(fair_a, fair_b)``.
    """
    a = np.asarray(p_imp_a, dtype=float)
    b = np.asarray(p_imp_b, dtype=float)
    book = a + b
    fair_a = a / book
    fair_b = b / book
    if np.ndim(p_imp_a) == 0 and np.ndim(p_imp_b) == 0:
        return float(fair_a), float(fair_b)
    return fair_a, fair_b


def devig_shin(p_imp_a, p_imp_b):
    """De-vig two implied probabilities with Shin's (1992/93) method.

    Shin's model attributes the bookmaker margin to a fraction ``z`` of
    insider money; solving it de-vigs longshots more aggressively than the
    proportional method. For two outcomes ``z`` has a closed form. With
    booksum ``B = pi_a + pi_b`` and ``a = pi_a^2 / B``, ``b = pi_b^2 / B``:

        1 - z = 2 * (a + b - 1) / ((a - b)^2 - 1)
        p_i   = (sqrt(z^2 + 4 * (1 - z) * pi_i^2 / B) - z) / (2 * (1 - z))

    which is the standard Shin solution (the fixed-point iteration used for
    n > 2 outcomes degenerates at n = 2, hence the closed form).

    Note: at a typical two-way esports vig of ~5% Shin differs from
    :func:`devig_proportional` by well under a percentage point of
    probability, so it is offered optionally; the default pipeline uses the
    proportional method.

    Falls back to proportional de-vigging when the booksum is <= 1 (no vig to
    explain; Shin's ``z`` would be negative). Scalars only.

    Returns ``(fair_a, fair_b)`` summing to 1.
    """
    pi_a = float(p_imp_a)
    pi_b = float(p_imp_b)
    book = pi_a + pi_b
    if book <= 1.0:
        return devig_proportional(pi_a, pi_b)
    a = pi_a * pi_a / book
    b = pi_b * pi_b / book
    d = a - b
    one_minus_z = 2.0 * (a + b - 1.0) / (d * d - 1.0)
    z = 1.0 - one_minus_z
    denom = 2.0 * one_minus_z
    fair_a = (np.sqrt(z * z + 4.0 * one_minus_z * pi_a * pi_a / book) - z) / denom
    fair_b = (np.sqrt(z * z + 4.0 * one_minus_z * pi_b * pi_b / book) - z) / denom
    return float(fair_a), float(fair_b)


def kelly_fraction(p, decimal_odds):
    """Full-Kelly stake fraction for a binary bet at decimal odds.

    ``f* = (p * d - 1) / (d - 1)``, floored at 0 (never bet a negative
    edge). Odds ``d <= 1`` return 0 (no possible profit). Accepts scalars or
    arrays; returns the same shape.
    """
    p_arr = np.asarray(p, dtype=float)
    d_arr = np.asarray(decimal_odds, dtype=float)
    safe_denom = np.where(d_arr > 1.0, d_arr - 1.0, 1.0)
    f = np.where(d_arr > 1.0, (p_arr * d_arr - 1.0) / safe_denom, 0.0)
    f = np.maximum(f, 0.0)
    if np.ndim(p) == 0 and np.ndim(decimal_odds) == 0:
        return float(f)
    return f


def make_synthetic_odds(
    ref_prob,
    shrink: float = 0.9,
    noise_sd: float = 0.4,
    margin: float = 0.05,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a SYNTHETIC bookmaker line from a reference probability.

    The bookmaker's fair probability ``q`` is the reference probability
    shrunk toward 0.5 on the logit scale by ``shrink`` and perturbed with
    ``N(0, noise_sd)`` logit noise (``np.random.default_rng(seed)``), then
    clipped to ``[0.02, 0.98]`` so odds stay sane. The margin is applied
    proportionally: ``imp_blue = q * (1 + margin)``,
    ``imp_red = (1 - q) * (1 + margin)``, so the total book is exactly
    ``1 + margin``; decimal odds are ``1 / imp``.

    Deterministic given ``seed`` and input order. If ``ref_prob`` is a
    Series its index is preserved.

    Returns a DataFrame with columns:
    ``odds_blue, odds_red`` (decimal, vigged), ``imp_blue, imp_red`` (vigged
    implied probs), ``fair_blue, fair_red`` (de-vigged via
    :func:`devig_proportional`).
    """
    index = ref_prob.index if isinstance(ref_prob, pd.Series) else None
    p = np.asarray(ref_prob, dtype=float)
    p = np.clip(p, 1e-9, 1.0 - 1e-9)  # keep logits finite
    rng = np.random.default_rng(seed)
    logit = np.log(p / (1.0 - p))
    logit_book = shrink * logit + rng.normal(0.0, noise_sd, size=p.shape)
    q = 1.0 / (1.0 + np.exp(-logit_book))
    q = np.clip(q, 0.02, 0.98)

    imp_blue = q * (1.0 + margin)
    imp_red = (1.0 - q) * (1.0 + margin)
    fair_blue, fair_red = devig_proportional(imp_blue, imp_red)
    return pd.DataFrame(
        {
            "odds_blue": 1.0 / imp_blue,
            "odds_red": 1.0 / imp_red,
            "imp_blue": imp_blue,
            "imp_red": imp_red,
            "fair_blue": fair_blue,
            "fair_red": fair_red,
        },
        index=index,
    )


_BET_COLUMNS = ["side", "model_p", "odds", "edge", "stake_frac"]


def select_bets(
    model_p,
    odds_blue,
    odds_red,
    min_edge: float = 0.04,
    kelly_mult: float = 0.25,
    max_stake: float = 0.02,
    min_hist_games: int = 10,
    hist_games_blue=None,
    hist_games_red=None,
) -> pd.DataFrame:
    """Select at most one bet per game against a two-way price (vectorized).

    Edges are against the *vigged* implied probabilities (what you pay):
    ``edge_blue = model_p - 1/odds_blue`` and
    ``edge_red = (1 - model_p) - 1/odds_red``. For each game the side with
    the larger edge is a candidate; it becomes a bet iff its edge exceeds
    ``min_edge``. Never bets both sides of one game.

    Stake fraction is capped fractional Kelly:
    ``stake_frac = min(max_stake, kelly_mult * kelly_fraction(p_side, odds_side))``;
    bets with ``stake_frac <= 0`` are dropped.

    If ``hist_games_blue`` / ``hist_games_red`` are provided, a game is only
    bettable when the provided count(s) are ``>= min_hist_games`` for BOTH
    teams (cold-start refusal, DESIGN.md item 8).

    The returned DataFrame carries the original index (game reference): it is
    taken from the first pandas input among ``model_p, odds_blue, odds_red``,
    else a RangeIndex. Columns: ``side`` ('blue'|'red'), ``model_p``
    (probability of the *chosen* side), ``odds``, ``edge``, ``stake_frac``.
    """
    index = None
    for arg in (model_p, odds_blue, odds_red):
        if isinstance(arg, (pd.Series, pd.DataFrame)):
            index = arg.index
            break
    p = np.asarray(model_p, dtype=float)
    ob = np.asarray(odds_blue, dtype=float)
    orr = np.asarray(odds_red, dtype=float)
    if index is None:
        index = pd.RangeIndex(len(p))

    imp_blue = 1.0 / ob
    imp_red = 1.0 / orr
    edge_blue = p - imp_blue
    edge_red = (1.0 - p) - imp_red

    pick_blue = edge_blue >= edge_red
    side_edge = np.where(pick_blue, edge_blue, edge_red)
    side_p = np.where(pick_blue, p, 1.0 - p)
    side_odds = np.where(pick_blue, ob, orr)

    mask = side_edge > min_edge
    if hist_games_blue is not None:
        mask &= np.asarray(hist_games_blue) >= min_hist_games
    if hist_games_red is not None:
        mask &= np.asarray(hist_games_red) >= min_hist_games

    stake = np.minimum(max_stake, kelly_mult * kelly_fraction(side_p, side_odds))
    mask &= stake > 0.0

    bets = pd.DataFrame(
        {
            "side": np.where(pick_blue, "blue", "red"),
            "model_p": side_p,
            "odds": side_odds,
            "edge": side_edge,
            "stake_frac": stake,
        },
        index=index,
    )
    return bets.loc[mask, _BET_COLUMNS]


def settle_bets(bets: pd.DataFrame, blue_win) -> pd.DataFrame:
    """Grade bets against realized outcomes.

    ``blue_win`` is the 0/1 outcome for ALL games, referenced by the bets'
    index: pass a Series (aligned by label) or an array (the bets' integer
    index values are used positionally). A convenience fallback: an array
    whose length equals ``len(bets)`` but cannot be indexed by the bets'
    index is aligned positionally to the bets.

    Adds columns:
    - ``won`` (bool): ``(side == 'blue') == bool(blue_win)``.
    - ``pnl`` (per unit bankroll, NOT compounded): won ->
      ``stake_frac * (odds - 1)``, lost -> ``-stake_frac``.
    - ``ret``: return per unit stake, ``pnl / stake_frac``.

    Bankroll curves are computed by :func:`simulate_bankroll`.
    """
    if isinstance(blue_win, pd.Series):
        outcomes = blue_win.loc[bets.index].to_numpy()
    else:
        arr = np.asarray(blue_win)
        if len(bets) and not pd.api.types.is_integer_dtype(bets.index):
            # A plain array cannot be aligned to a non-integer bets index:
            # the index values are not positions into the array, and silently
            # falling back to positional order could grade bets against the
            # wrong games.  (Previously this fell through to a confusing
            # numpy comparison error inside `idx.max() >= len(arr)`.)
            raise TypeError(
                "settle_bets: cannot align array outcomes to a non-integer "
                f"bets index (dtype {bets.index.dtype}); pass blue_win as a "
                "pandas Series indexed like the bets (label alignment), or "
                "use integer positional indices on the bets frame"
            )
        idx = np.asarray(bets.index)
        if len(arr) == len(bets) and (len(bets) == 0 or idx.max(initial=-1) >= len(arr)):
            outcomes = arr
        else:
            outcomes = arr[idx] if len(bets) else arr[:0]

    settled = bets.copy()
    won = (settled["side"].to_numpy() == "blue") == outcomes.astype(bool)
    stake = settled["stake_frac"].to_numpy(dtype=float)
    odds = settled["odds"].to_numpy(dtype=float)
    pnl = np.where(won, stake * (odds - 1.0), -stake)
    settled["won"] = won
    settled["pnl"] = pnl
    settled["ret"] = np.divide(pnl, stake, out=np.zeros_like(pnl), where=stake != 0)
    return settled


def simulate_bankroll(bets_settled: pd.DataFrame, start: float = 1.0, compound: bool = True) -> pd.Series:
    """Run a bankroll through settled bets in the given (chronological) order.

    Per bet: amount risked = ``stake_frac * current_bankroll`` when
    ``compound=True``, else ``stake_frac * start`` (flat fractions of the
    starting bankroll). Bankroll then moves by ``+amount * (odds - 1)`` on a
    win and ``-amount`` on a loss.

    Note on ``compound=False`` (flat staking): every bet risks a fixed
    fraction of the START bankroll regardless of the current balance, so a
    long losing streak can drive the bankroll below zero.  This is
    intentionally NOT floored at zero — flat mode is a *diagnostic* arm
    (linear in per-bet pnl, a miscalibration canary vs the Kelly arm), not a
    simulation of playable staking; flooring would silently censor exactly
    the tail it exists to expose.

    Returns a Series of the bankroll AFTER each bet, aligned to the bets'
    index in order.
    """
    won = bets_settled["won"].to_numpy(dtype=bool)
    stake = bets_settled["stake_frac"].to_numpy(dtype=float)
    odds = bets_settled["odds"].to_numpy(dtype=float)

    bankroll = float(start)
    curve = np.empty(len(bets_settled), dtype=float)
    for i in range(len(bets_settled)):
        base = bankroll if compound else float(start)
        amount = stake[i] * base
        bankroll += amount * (odds[i] - 1.0) if won[i] else -amount
        curve[i] = bankroll
    return pd.Series(curve, index=bets_settled.index, name="bankroll")


def bootstrap_roi_ci(
    settled_bets: pd.DataFrame,
    n: int = 10_000,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for ROI over a set of settled bets.

    ROI = total ``pnl`` / total ``stake_frac``. Resamples the bets with
    replacement ``n`` times (``np.random.default_rng(seed)``, deterministic)
    and takes the ``(1 - level) / 2`` and ``1 - (1 - level) / 2`` percentile
    ROIs.

    Returns ``(lo, hi, point)`` where ``point`` is the full-sample ROI.
    """
    pnl = settled_bets["pnl"].to_numpy(dtype=float)
    stake = settled_bets["stake_frac"].to_numpy(dtype=float)
    m = len(pnl)
    if m == 0:
        return (float("nan"), float("nan"), float("nan"))
    point = float(pnl.sum() / stake.sum())

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, m, size=(n, m))
    rois = pnl[idx].sum(axis=1) / stake[idx].sum(axis=1)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(rois, [alpha, 1.0 - alpha])
    return float(lo), float(hi), point
