"""The acceptance bar for the Kalshi esports edge search (family J, stat-protocol).

Every candidate edge in docs/EDGE_SEARCH.md must pass through
:func:`evaluate_frozen_rule` on the CONFIRMATION window (match_start >=
2026-06-20) with its rule frozen *before* the confirmation data was touched.
This module is the single implementation of that bar so every family is graded
identically; docs/EDGE_PROTOCOL.md is the plain-language companion.

Design decisions (frozen — changing them after seeing confirmation results
voids the campaign):

* **Cluster bootstrap by ``event_ticker``.** Markets from the same event
  (same series, often the same teams/day) settle together and share news;
  bets inside an event are not independent. Resampling whole events is the
  unit of independence; a naive per-bet bootstrap understates variance and
  overstates significance whenever a rule fires several times per event.
* **One-sided bootstrap p-value.** ``H0: E[pnl per bet] <= 0``. The p-value is
  the fraction of cluster-bootstrap replicates with total pnl <= 0, with the
  standard ``(1 + hits) / (1 + n_boot)`` correction so p is never exactly 0.
  This is the percentile-CI-inversion approximation; it is honest as long as
  the number of event clusters is not tiny (hence the ``MIN_EVENTS`` gate).
* **Bonferroni across everything tried.** ``p_adj = p * n_families_tested *
  n_variants_in_family`` (capped at 1). Bonferroni is conservative but it is
  the only correction that survives "we do not know how correlated the
  families are"; the price is paid in the minimum detectable edge (see
  :func:`minimum_detectable_edge`).

Verdict (frozen):

* ``INSUFFICIENT_N``  — fewer than ``MIN_BETS`` (30) bets or fewer than
  ``MIN_EVENTS`` (20) distinct event clusters. No claim either way.
* ``SIGNIFICANT``     — n gates pass AND ``p_adj < 0.05`` AND the cluster-
  bootstrap **99%** CI lower bound on ROI is > 0.
* ``NOT_SIGNIFICANT`` — everything else.

P&L convention (matches ``kalshi_eval``): a bet costs ``(entry_price + fee) *
stake`` dollars up front; a win pays ``stake`` dollars (settlement is free), so
``pnl = (1 - entry_price - fee) * stake`` on a win and ``-(entry_price + fee) *
stake`` on a loss. ``entry_price`` is the dollar cost of the side actually
bought (a NO bet at no-price 0.35 enters at 0.35, not 0.65).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "MIN_BETS",
    "MIN_EVENTS",
    "ALPHA",
    "evaluate_frozen_rule",
    "minimum_detectable_edge",
    "kelly_capped_stakes",
    "summarize_families",
]

#: Minimum executed bets on the confirmation window for any verdict.
MIN_BETS = 30
#: Minimum distinct ``event_ticker`` clusters (the bootstrap's effective n).
MIN_EVENTS = 20
#: Family-wise significance level for the adjusted p-value.
ALPHA = 0.05

_REQUIRED_COLUMNS = ("event_ticker", "entry_price", "fee", "won")


# --------------------------------------------------------------- validation


def _validate_bets(bets: pd.DataFrame) -> None:
    if not isinstance(bets, pd.DataFrame):
        raise TypeError(f"bets must be a pandas DataFrame, got {type(bets).__name__}")
    missing = [c for c in _REQUIRED_COLUMNS if c not in bets.columns]
    if missing:
        raise ValueError(f"bets is missing required columns: {missing}")
    if len(bets) == 0:
        raise ValueError("bets is empty — nothing to evaluate")
    entry = bets["entry_price"].to_numpy(dtype=float)
    fee = bets["fee"].to_numpy(dtype=float)
    if not np.all((entry > 0.0) & (entry < 1.0)):
        raise ValueError("entry_price must be strictly inside (0, 1) dollars")
    if not np.all(fee >= 0.0):
        raise ValueError("fee must be >= 0 dollars")
    if "stake" in bets.columns and not np.all(bets["stake"].to_numpy(dtype=float) > 0):
        raise ValueError("stake must be > 0 contracts")


# ------------------------------------------------------------ the main gate


def evaluate_frozen_rule(
    bets: pd.DataFrame,
    n_families_tested: int,
    n_variants_in_family: int = 1,
    seed: int = 0,
    n_boot: int = 10_000,
) -> dict:
    """Grade a frozen rule's executed bets on the confirmation window.

    Parameters
    ----------
    bets:
        One row per executed bet. Required columns: ``event_ticker`` (cluster
        key), ``entry_price`` (dollars paid per contract for the side bought,
        after YES/NO side conversion), ``fee`` (dollars per contract, taker
        unless maker fills are evidenced), ``won`` (bool). Optional ``stake``
        (contracts, default 1).
    n_families_tested:
        How many hypothesis FAMILIES the campaign has tested (see
        docs/EDGE_SEARCH.md — count them all, not just the survivors).
    n_variants_in_family:
        How many rule variants THIS family tried in discovery before freezing
        this one. Every peek counts.
    seed, n_boot:
        Bootstrap RNG seed and replicate count (``np.random.default_rng``,
        deterministic for a given seed).

    Returns
    -------
    dict with keys: ``n``, ``n_events``, ``staked``, ``pnl``, ``roi``,
    ``mean_pnl_per_bet``, ``roi_ci95`` / ``roi_ci99`` /
    ``mean_pnl_ci95`` / ``mean_pnl_ci99`` (each ``(lo, hi)``), ``p_value``,
    ``n_tests``, ``p_adj``, ``verdict``, ``es5_pnl`` (expected shortfall:
    mean total pnl over the worst 5% of bootstrap replicates — the "unlucky
    rerun" loss), ``max_drawdown`` (realized peak-to-trough of cumulative pnl
    in row order, >= 0), ``break_even_extra_cost_per_bet`` (dollars of extra
    per-bet cost — slippage, adverse fills, fee underestimate — that would
    zero the total pnl; 0 if pnl is already <= 0), ``seed``, ``n_boot``.
    """
    _validate_bets(bets)
    if n_families_tested < 1 or n_variants_in_family < 1:
        raise ValueError("n_families_tested and n_variants_in_family must be >= 1")
    if n_boot < 100:
        raise ValueError("n_boot must be >= 100 for meaningful percentiles")

    entry = bets["entry_price"].to_numpy(dtype=float)
    fee = bets["fee"].to_numpy(dtype=float)
    won = bets["won"].to_numpy(dtype=bool)
    stake = (
        bets["stake"].to_numpy(dtype=float)
        if "stake" in bets.columns
        else np.ones(len(bets))
    )

    cost = (entry + fee) * stake  # dollars at risk per bet
    pnl = np.where(won, (1.0 - entry - fee) * stake, -cost)

    n = int(len(bets))
    codes, uniques = pd.factorize(bets["event_ticker"].to_numpy())
    k = int(len(uniques))
    total_pnl = float(pnl.sum())
    total_cost = float(cost.sum())
    roi = total_pnl / total_cost
    mean_pnl = total_pnl / n

    # ---- cluster bootstrap: resample whole event_ticker clusters ----------
    cl_pnl = np.bincount(codes, weights=pnl, minlength=k)
    cl_cost = np.bincount(codes, weights=cost, minlength=k)
    cl_n = np.bincount(codes, minlength=k).astype(float)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(n_boot, k))
    b_pnl = cl_pnl[idx].sum(axis=1)
    b_cost = cl_cost[idx].sum(axis=1)
    b_n = cl_n[idx].sum(axis=1)
    b_roi = b_pnl / b_cost
    b_mean = b_pnl / b_n

    def _ci(x: np.ndarray, level: float) -> tuple[float, float]:
        half = (1.0 - level) / 2.0 * 100.0
        lo, hi = np.percentile(x, [half, 100.0 - half])
        return float(lo), float(hi)

    roi_ci95 = _ci(b_roi, 0.95)
    roi_ci99 = _ci(b_roi, 0.99)
    mean_ci95 = _ci(b_mean, 0.95)
    mean_ci99 = _ci(b_mean, 0.99)

    # One-sided p for H0: pnl <= 0, via the bootstrap distribution.
    p_value = float((1 + int(np.sum(b_pnl <= 0.0))) / (1 + n_boot))
    n_tests = int(n_families_tested) * int(n_variants_in_family)
    p_adj = float(min(1.0, p_value * n_tests))

    # Expected-shortfall-style tail: mean of the worst 5% bootstrap reruns.
    q5 = np.percentile(b_pnl, 5.0)
    es5_pnl = float(b_pnl[b_pnl <= q5].mean())

    # Realized max drawdown of cumulative pnl, in the order the bets are given
    # (pass rows in chronological order for this to mean anything).
    cum = np.concatenate([[0.0], np.cumsum(pnl)])
    max_drawdown = float(np.max(np.maximum.accumulate(cum) - cum))

    break_even = float(max(0.0, mean_pnl))

    if n < MIN_BETS or k < MIN_EVENTS:
        verdict = "INSUFFICIENT_N"
    elif p_adj < ALPHA and roi_ci99[0] > 0.0:
        verdict = "SIGNIFICANT"
    else:
        verdict = "NOT_SIGNIFICANT"

    return {
        "n": n,
        "n_events": k,
        "staked": total_cost,
        "pnl": total_pnl,
        "roi": roi,
        "mean_pnl_per_bet": mean_pnl,
        "roi_ci95": roi_ci95,
        "roi_ci99": roi_ci99,
        "mean_pnl_ci95": mean_ci95,
        "mean_pnl_ci99": mean_ci99,
        "p_value": p_value,
        "n_tests": n_tests,
        "p_adj": p_adj,
        "verdict": verdict,
        "es5_pnl": es5_pnl,
        "max_drawdown": max_drawdown,
        "break_even_extra_cost_per_bet": break_even,
        "seed": int(seed),
        "n_boot": int(n_boot),
    }


# ------------------------------------------------------------ power planning


def minimum_detectable_edge(
    n_bets: int,
    per_bet_sd: float = 0.5,
    alpha: float = 0.05,
    power: float = 0.8,
    n_tests: int = 10,
) -> float:
    """Smallest true mean edge (CENTS per contract) detectable at this bar.

    Standard one-sided two-moment power formula with a Bonferroni-adjusted
    level ``alpha / n_tests``:

        MDE = (z_{1 - alpha/n_tests} + z_{power}) * per_bet_sd / sqrt(n_bets)

    ``per_bet_sd`` is the per-bet pnl standard deviation in DOLLARS; 0.5 is
    the right order for flat one-contract bets near 50c (a coin-flip contract
    pays about +/-0.5). If bets cluster by event, use the number of EVENTS as
    ``n_bets`` (the effective sample size), not the number of bets.

    Returns the edge in cents/contract. Smaller is better; a rule whose
    plausible edge is below its MDE cannot be confirmed on this data no matter
    how real it is — that is a "collect more data" answer, not a "ship it".
    """
    if n_bets <= 0:
        raise ValueError("n_bets must be positive")
    if not (0.0 < alpha < 1.0) or not (0.0 < power < 1.0):
        raise ValueError("alpha and power must be in (0, 1)")
    if per_bet_sd <= 0:
        raise ValueError("per_bet_sd must be positive")
    if n_tests < 1:
        raise ValueError("n_tests must be >= 1")
    z_alpha = stats.norm.ppf(1.0 - alpha / n_tests)
    z_power = stats.norm.ppf(power)
    return float((z_alpha + z_power) * per_bet_sd / np.sqrt(n_bets) * 100.0)


# ----------------------------------------------------------------- staking


def kelly_capped_stakes(p_fair, entry_price, cap: float = 0.02) -> float:
    """Capped Kelly fraction of bankroll for a binary contract.

    For a contract bought at ``entry_price`` dollars (all-in cost incl. fees)
    that pays 1 on a win, full Kelly is ``(p_fair - entry_price) /
    (1 - entry_price)``. Returns that fraction clipped to ``[0, cap]`` —
    never bet a negative edge, never bet more than ``cap`` (default 2%) of
    bankroll on one contract position. The tight cap is deliberate: Kelly is
    only optimal if ``p_fair`` is unbiased, and the whole point of this
    protocol is that our probabilities are suspects, not facts.
    """
    p = float(p_fair)
    q = float(entry_price)
    if not (0.0 < p < 1.0):
        raise ValueError("p_fair must be strictly inside (0, 1)")
    if not (0.0 < q < 1.0):
        raise ValueError("entry_price must be strictly inside (0, 1)")
    if cap <= 0.0:
        raise ValueError("cap must be positive")
    f = (p - q) / (1.0 - q)
    return float(min(cap, max(0.0, f)))


# ------------------------------------------------------------- league table


def summarize_families(family_results: list[dict]) -> str:
    """Render a league table over :func:`evaluate_frozen_rule` outputs.

    Each dict should be an ``evaluate_frozen_rule`` result, optionally with a
    ``"family"`` (or ``"name"``) key for the label. Sorted by adjusted p.
    The footer states how many families would clear the RAW p < 0.05 bar by
    pure luck if every family were noise — the reason the adjusted bar exists.
    """
    if not family_results:
        return "no family results to summarize"

    def _label(r: dict, i: int) -> str:
        return str(r.get("family", r.get("name", f"family_{i}")))

    rows = sorted(
        (dict(r, _label=_label(r, i)) for i, r in enumerate(family_results)),
        key=lambda r: (r.get("p_adj", 1.0), r.get("p_value", 1.0)),
    )
    header = (
        f"{'family':<24}{'n':>5}{'events':>7}{'roi':>8}"
        f"{'roi_ci95':>18}{'p':>9}{'p_adj':>8}  verdict"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        lo, hi = r.get("roi_ci95", (float("nan"), float("nan")))
        lines.append(
            f"{r['_label']:<24}{r.get('n', 0):>5}{r.get('n_events', 0):>7}"
            f"{r.get('roi', float('nan')):>+8.3f}"
            f"{f'[{lo:+.3f},{hi:+.3f}]':>18}"
            f"{r.get('p_value', float('nan')):>9.4f}"
            f"{r.get('p_adj', float('nan')):>8.4f}"
            f"  {r.get('verdict', '?')}"
        )
    n_fam = len(rows)
    n_sig = sum(1 for r in rows if r.get("verdict") == "SIGNIFICANT")
    expected_raw = 0.05 * n_fam
    lines += [
        "-" * len(header),
        (
            f"{n_fam} families reported; {n_sig} SIGNIFICANT at the adjusted bar. "
            f"If ALL {n_fam} were pure noise, ~{expected_raw:.1f} would still clear "
            f"raw p < 0.05 by luck; the Bonferroni-adjusted bar holds the chance "
            f"of even one false SIGNIFICANT at <= 5% family-wise."
        ),
    ]
    return "\n".join(lines)
