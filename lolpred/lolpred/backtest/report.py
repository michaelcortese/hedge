"""Evaluation report over walk-forward predictions (CONTRACTS.md section 6).

Probability-quality metrics (accuracy / Brier / log-loss), equal-count-bin
calibration (ECE + reliability table), a series-momentum iid-violation
detector, and a human-readable multi-section summary including a betting
section when settled bets are supplied.

No prints — :func:`summarize` returns the report text.  All betting section
headers carry the SYNTHETIC ODDS disclaimer when ``synthetic_odds=True``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from lolpred.backtest.betting import bootstrap_roi_ci, simulate_bankroll

__all__ = [
    "probability_metrics",
    "ece",
    "reliability_table",
    "momentum_test",
    "summarize",
]

_EPS = 1e-15

_SYNTHETIC_TAG = "[SYNTHETIC ODDS — plumbing validation only, not evidence of real edge]"


def _logit(p: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def probability_metrics(y, p) -> dict:
    """Accuracy / Brier / log-loss of probabilities ``p`` against 0/1 ``y``.

    ``accuracy`` thresholds at 0.5 (``p >= 0.5`` predicts 1).  Log-loss clips
    ``p`` to ``[1e-15, 1 - 1e-15]``.  Returns
    ``{"accuracy", "brier", "logloss", "n"}``.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if y.shape != p.shape:
        raise ValueError("y and p must have the same shape")
    pc = np.clip(p, _EPS, 1.0 - _EPS)
    return {
        "accuracy": float(np.mean((p >= 0.5) == (y == 1))),
        "brier": float(np.mean((p - y) ** 2)),
        "logloss": float(-np.mean(y * np.log(pc) + (1.0 - y) * np.log1p(-pc))),
        "n": int(len(y)),
    }


def _equal_count_bins(y, p, n_bins: int) -> pd.DataFrame:
    """Split (y, p) into ~equal-count bins by sorted p; one row per bin."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    order = np.argsort(p, kind="stable")
    rows = []
    for b, idx in enumerate(np.array_split(order, n_bins)):
        if len(idx) == 0:
            continue
        rows.append(
            {
                "bin": b,
                "n": int(len(idx)),
                "p_mean": float(p[idx].mean()),
                "y_rate": float(y[idx].mean()),
            }
        )
    return pd.DataFrame(rows, columns=["bin", "n", "p_mean", "y_rate"])


def ece(y, p, n_bins: int = 10) -> float:
    """Expected calibration error over equal-count bins.

    ``sum_b (n_b / n) * |mean(p_b) - mean(y_b)|`` with bins formed by sorting
    ``p`` and splitting into ``n_bins`` near-equal-count chunks.
    """
    tab = _equal_count_bins(y, p, n_bins)
    n = tab["n"].sum()
    if n == 0:
        return float("nan")
    w = tab["n"].to_numpy(dtype=float) / float(n)
    return float(np.sum(w * np.abs(tab["p_mean"].to_numpy() - tab["y_rate"].to_numpy())))


def reliability_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Equal-count reliability table: columns ``bin, n, p_mean, y_rate``."""
    return _equal_count_bins(y, p, n_bins)


def momentum_test(
    preds: pd.DataFrame, n_resamples: int = 200, seed: int = 0
) -> dict:
    """Detect within-series momentum (an iid violation) beyond model skill.

    Expected input columns (one row per game):
      ``series_id``, ``game_in_series``, ``blue_team``, ``red_team``,
      ``blue_win`` (0/1), ``model_p`` (model's P(blue win) for that game).

    Exact regression run
    --------------------
    Within each series (sorted by ``game_in_series``), for every game k that
    has a previous game in the same series, define

      * ``won_prev`` = 1.0 if game k's BLUE team is the team that won game
        k-1, else 0.0 (team-agnostic: teams swap sides between games, so this
        is computed by winner *name*, not by side);
      * response = ``blue_win`` of game k;
      * skill control = ``logit(model_p)`` of game k.

    Fit an (effectively unpenalized, ``C=1e6``) sklearn
    ``LogisticRegression(fit_intercept=True)`` of the response on
    ``[logit(model_p), won_prev]``.  ``lag_coef`` is the coefficient on
    ``won_prev``: under no momentum, ``blue_win`` is independent of the
    series' past given the model probability, so the coefficient tends to 0;
    persistent over-performance by the previous game's winner (momentum the
    model does not know about) pushes it positive.  The framing is invariant
    to which team is labeled blue (flipping orientation flips the response,
    negates the logit, and maps ``won_prev -> 1 - won_prev``, which only
    moves the intercept).

    Significance proxy: ``n_resamples`` bootstrap row-resamples (with
    replacement, ``np.random.default_rng(seed)``) refit the regression;
    ``sign_stability`` is the fraction of successful resamples whose lag
    coefficient has the same sign as the full-sample fit.  ~0.5 means noise;
    near 1.0 means a stable directional effect.  Degenerate resamples
    (single-class response) are skipped.

    Returns ``{"lag_coef", "sign_stability", "n"}`` where ``n`` is the
    number of lag rows (games with a previous game in their series).
    """
    required = {"series_id", "game_in_series", "blue_team", "red_team", "blue_win", "model_p"}
    missing = required - set(preds.columns)
    if missing:
        raise ValueError(f"momentum_test: missing columns {sorted(missing)}")

    df = preds.sort_values(["series_id", "game_in_series"], kind="stable")
    winner = np.where(df["blue_win"].astype(int) == 1, df["blue_team"], df["red_team"])
    prev_winner = pd.Series(winner, index=df.index).groupby(df["series_id"]).shift(1)
    has_prev = prev_winner.notna().to_numpy()

    y = df["blue_win"].astype(int).to_numpy()[has_prev]
    x_skill = _logit(df["model_p"].to_numpy())[has_prev]
    won_prev = (df["blue_team"] == prev_winner).astype(float).to_numpy()[has_prev]

    n = int(len(y))
    if n < 10 or len(np.unique(y)) < 2:
        return {"lag_coef": float("nan"), "sign_stability": float("nan"), "n": n}

    X = np.column_stack([x_skill, won_prev])

    def _fit_lag_coef(Xb: np.ndarray, yb: np.ndarray) -> float:
        lr = LogisticRegression(
            C=1e6, solver="lbfgs", max_iter=1000, fit_intercept=True
        )
        lr.fit(Xb, yb)
        return float(lr.coef_[0][1])

    lag_coef = _fit_lag_coef(X, y)
    full_sign = np.sign(lag_coef)

    rng = np.random.default_rng(seed)
    same_sign = 0
    fitted = 0
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        fitted += 1
        if np.sign(_fit_lag_coef(X[idx], yb)) == full_sign:
            same_sign += 1
    stability = same_sign / fitted if fitted else float("nan")
    return {"lag_coef": lag_coef, "sign_stability": float(stability), "n": n}


def _max_drawdown(bets_settled: pd.DataFrame) -> float:
    """Max drawdown of the compounded bankroll curve (start bankroll 1.0)."""
    curve = simulate_bankroll(bets_settled, start=1.0, compound=True)
    full = np.concatenate([[1.0], curve.to_numpy(dtype=float)])
    peak = np.maximum.accumulate(full)
    return float(np.max(1.0 - full / peak))


def _betting_metrics(bets_settled: pd.DataFrame) -> dict:
    pnl = bets_settled["pnl"].to_numpy(dtype=float)
    stake = bets_settled["stake_frac"].to_numpy(dtype=float)
    lo, hi, roi = bootstrap_roi_ci(bets_settled)
    curve = simulate_bankroll(bets_settled, start=1.0, compound=True)
    return {
        "n_bets": int(len(bets_settled)),
        "hit_rate": float(bets_settled["won"].mean()),
        "total_staked": float(stake.sum()),
        "total_pnl": float(pnl.sum()),
        "roi": roi,
        "roi_ci_lo": lo,
        "roi_ci_hi": hi,
        "max_drawdown": _max_drawdown(bets_settled),
        "final_bankroll": float(curve.iloc[-1]) if len(curve) else 1.0,
        # Flat-stake comparison arm: equal stake per bet -> ROI = mean return
        # per unit stake.  Divergence from Kelly ROI is a miscalibration canary.
        "flat_roi": float(bets_settled["ret"].mean()),
    }


def summarize(
    preds: pd.DataFrame,
    bets_settled: pd.DataFrame | None = None,
    synthetic_odds: bool = True,
) -> tuple[dict, str]:
    """Full evaluation summary: (metrics dict, human-readable report text).

    ``preds`` is :func:`~lolpred.backtest.walkforward.run_walkforward` output
    (needs ``blue_win`` + ``model_p``; scores ``baseline_p``, ``fair_blue``
    and a per-fold table when the columns are present).  Baseline rows always
    include the trivial constants 0.5 and the in-sample blue rate.

    ``bets_settled`` is :func:`~lolpred.backtest.betting.settle_bets` output
    (columns ``won, pnl, ret, stake_frac, odds``); when given, a betting
    section is added.  Every betting section header carries the SYNTHETIC
    ODDS disclaimer when ``synthetic_odds=True``.
    """
    if "blue_win" not in preds.columns or "model_p" not in preds.columns:
        raise ValueError("preds must have 'blue_win' and 'model_p' columns")
    y = preds["blue_win"].to_numpy(dtype=int)
    n = len(preds)

    # ---- model vs baselines ------------------------------------------------
    models: dict[str, dict] = {"model": probability_metrics(y, preds["model_p"])}
    if "baseline_p" in preds.columns:
        models["baseline_elo_bt"] = probability_metrics(y, preds["baseline_p"])
    models["const_0.5"] = probability_metrics(y, np.full(n, 0.5))
    blue_rate = float(np.mean(y))
    models["const_bluerate(in-sample)"] = probability_metrics(y, np.full(n, blue_rate))
    if "fair_blue" in preds.columns:
        models["market_fair(devig)"] = probability_metrics(y, preds["fair_blue"])

    # ---- per-fold table ----------------------------------------------------
    per_fold = None
    if "fold_id" in preds.columns:
        rows = []
        for fid, grp in preds.groupby("fold_id", sort=True):
            yg = grp["blue_win"].to_numpy(dtype=int)
            row = {
                "fold_id": fid,
                "n": len(grp),
                "model_logloss": probability_metrics(yg, grp["model_p"])["logloss"],
            }
            if "baseline_p" in grp.columns:
                row["baseline_logloss"] = probability_metrics(yg, grp["baseline_p"])[
                    "logloss"
                ]
            if "is_holdout" in grp.columns:
                row["is_holdout"] = bool(grp["is_holdout"].any())
            rows.append(row)
        per_fold = pd.DataFrame(rows)

    # ---- calibration -------------------------------------------------------
    model_ece = ece(y, preds["model_p"])
    rel = reliability_table(y, preds["model_p"])

    betting = _betting_metrics(bets_settled) if bets_settled is not None and len(bets_settled) else None

    result = {
        "n": n,
        "models": models,
        "per_fold": per_fold,
        "ece": model_ece,
        "reliability": rel,
        "betting": betting,
    }

    # ---- text report -------------------------------------------------------
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("Walk-forward evaluation report")
    lines.append("=" * 72)
    lines.append(f"games scored: {n}   blue win rate: {blue_rate:.4f}")
    lines.append("")

    lines.append("-- Model vs baselines (out-of-sample) --")
    name_w = max(len(k) for k in models)
    lines.append(f"{'name':<{name_w}}  {'n':>6}  {'accuracy':>8}  {'brier':>7}  {'logloss':>8}")
    for name, m in models.items():
        lines.append(
            f"{name:<{name_w}}  {m['n']:>6d}  {m['accuracy']:>8.4f}  "
            f"{m['brier']:>7.4f}  {m['logloss']:>8.4f}"
        )
    lines.append("")

    if per_fold is not None and len(per_fold):
        lines.append("-- Per-fold --")
        has_base = "baseline_logloss" in per_fold.columns
        header = f"{'fold':>4}  {'n':>6}  {'model_ll':>8}"
        if has_base:
            header += f"  {'base_ll':>8}"
        if "is_holdout" in per_fold.columns:
            header += "  holdout"
        lines.append(header)
        for _, r in per_fold.iterrows():
            line = f"{int(r['fold_id']):>4d}  {int(r['n']):>6d}  {r['model_logloss']:>8.4f}"
            if has_base:
                line += f"  {r['baseline_logloss']:>8.4f}"
            if "is_holdout" in per_fold.columns:
                line += f"  {'yes' if r['is_holdout'] else 'no':>7}"
            lines.append(line)
        lines.append("")

    lines.append("-- Calibration (model) --")
    lines.append(f"ECE ({len(rel)} equal-count bins): {model_ece:.4f}")
    lines.append(f"{'bin':>3}  {'n':>6}  {'p_mean':>7}  {'y_rate':>7}")
    for _, r in rel.iterrows():
        lines.append(
            f"{int(r['bin']):>3d}  {int(r['n']):>6d}  {r['p_mean']:>7.4f}  {r['y_rate']:>7.4f}"
        )
    lines.append("")

    if betting is not None:
        tag = f" {_SYNTHETIC_TAG}" if synthetic_odds else ""
        lines.append(f"-- Betting (quarter-Kelly, compounded){tag} --")
        lines.append(f"n_bets:         {betting['n_bets']}")
        lines.append(f"hit_rate:       {betting['hit_rate']:.4f}")
        lines.append(f"total_staked:   {betting['total_staked']:.4f} (bankroll fractions)")
        lines.append(f"total_pnl:      {betting['total_pnl']:+.4f}")
        lines.append(
            f"ROI:            {betting['roi']:+.4f} "
            f"(bootstrap 95% CI [{betting['roi_ci_lo']:+.4f}, {betting['roi_ci_hi']:+.4f}])"
        )
        lines.append(f"max_drawdown:   {betting['max_drawdown']:.4f}")
        lines.append(f"final_bankroll: {betting['final_bankroll']:.4f} (start 1.0)")
        lines.append("")
        lines.append(f"-- Betting (flat-stake comparison arm){tag} --")
        lines.append(f"flat-stake ROI: {betting['flat_roi']:+.4f}")
        lines.append("")

    return result, "\n".join(lines)
