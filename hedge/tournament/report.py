"""Scoring + leaderboard for the backtest tournament.

Turns the per-day graded predictions from ``backtest.py`` into the numbers that
decide which strategy to trust:

  * **Brier score** (multi-category): mean squared error of the probability vector
    vs the one-hot realized bucket. Lower = better. Proper scoring rule.
  * **Log loss**: -log(prob assigned to the bucket that actually happened). Punishes
    confident misses hard. Lower = better.
  * **CRPS**: integrates squared CDF error over the temperature line — rewards being
    *close* even when wrong, which pure hit/miss scores miss. Lower = better.
  * **Calibration error** (ECE): are buckets given 30% actually YES ~30% of the
    time? The thing that makes Kelly safe (a biased ``p`` loses money).
  * **Skill vs climatology**: ``1 - score_strategy / score_climatology``. Positive
    means a real edge over the null model; <= 0 means don't trust it with size.

A strategy that can't beat climatology on Brier and log loss is, by construction,
not enabled for real size.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from hedge.tournament.backtest import Bucket, GradedDay

_EPS = 1e-12


def brier(probs: list[float], realized_idx: int) -> float:
    p = np.asarray(probs, dtype=float)
    y = np.zeros_like(p)
    y[realized_idx] = 1.0
    return float(np.sum((p - y) ** 2))


def log_loss(probs: list[float], realized_idx: int) -> float:
    return float(-math.log(max(probs[realized_idx], _EPS)))


def crps(grid: list[Bucket], probs: list[float], realized: float) -> float:
    """Discrete CRPS over the integer temperature support of the grid.

    Uses each finite bucket's value (tails use their finite edge as a proxy point)
    to form a step CDF, then sums squared error against the realized step.
    """
    pts: list[float] = []
    for b in grid:
        if math.isinf(b.lo_f):
            pts.append(b.hi_f)
        elif math.isinf(b.hi_f):
            pts.append(b.lo_f)
        else:
            pts.append((b.lo_f + b.hi_f) / 2)
    p = np.asarray(probs, dtype=float)
    cdf = np.cumsum(p)
    pts_arr = np.asarray(pts, dtype=float)
    # Standard discrete CRPS: sum over thresholds of (F(t) - 1{obs <= t})^2.
    step = (pts_arr >= round(realized)).astype(float)
    return float(np.sum((cdf - step) ** 2))


def _per_day_scores(records: list[GradedDay]) -> pd.DataFrame:
    rows = []
    for r in records:
        rows.append({
            "strategy": r.strategy,
            "city": r.city,
            "date": r.local_date,
            "realized": r.realized_high,
            "brier": brier(r.probs, r.realized_idx),
            "log_loss": log_loss(r.probs, r.realized_idx),
            "crps": crps(r.grid, r.probs, r.realized_high),
            "p_realized": r.probs[r.realized_idx],
        })
    return pd.DataFrame(rows)


def calibration_error(records: list[GradedDay], n_bins: int = 10) -> dict[str, float]:
    """Expected Calibration Error per strategy over all (bucket) predictions."""
    pairs: dict[str, list[tuple[float, int]]] = {}
    for r in records:
        for i, p in enumerate(r.probs):
            pairs.setdefault(r.strategy, []).append((p, 1 if i == r.realized_idx else 0))
    out: dict[str, float] = {}
    for strat, ps in pairs.items():
        arr = np.asarray(ps, dtype=float)
        probs, ys = arr[:, 0], arr[:, 1]
        bins = np.linspace(0, 1, n_bins + 1)
        idx = np.clip(np.digitize(probs, bins) - 1, 0, n_bins - 1)
        ece = 0.0
        for b in range(n_bins):
            m = idx == b
            if m.any():
                ece += (m.mean()) * abs(probs[m].mean() - ys[m].mean())
        out[strat] = float(ece)
    return out


def _matched_skill(df: pd.DataFrame, baseline: str, metric: str) -> pd.Series:
    """Skill vs baseline on the *intersection* of (city, date) both predicted.

    Strategies abstain on different days, so comparing each strategy's full-sample
    mean against the baseline's full-sample mean mixes different day sets — a
    strategy that only acts on easy days looks artificially skilful. We instead
    join each strategy to the baseline on (city, date) and take the ratio of means
    over the common days only. Ratio form (not difference) so the λ thresholds in
    ``_lambda_tier`` keep their meaning.
    """
    base = (df[df["strategy"] == baseline][["city", "date", metric]]
            .rename(columns={metric: "_base"}))
    if base.empty:
        return pd.Series(dtype=float)
    merged = df.merge(base, on=["city", "date"], how="inner")
    g = merged.groupby("strategy")
    num = g[metric].mean()
    den = g["_base"].mean()
    return 1 - num / den.where(den != 0, np.nan)


def leaderboard(records: list[GradedDay], *, baseline: str = "weather_climatology") -> pd.DataFrame:
    """Aggregate per-strategy scores + matched-pairs skill vs the climatology baseline."""
    df = _per_day_scores(records)
    agg = df.groupby("strategy").agg(
        n_days=("brier", "size"),
        brier=("brier", "mean"),
        log_loss=("log_loss", "mean"),
        crps=("crps", "mean"),
        mean_p_realized=("p_realized", "mean"),
    ).reset_index()

    ece = calibration_error(records)
    agg["calib_err"] = agg["strategy"].map(ece)

    if (df["strategy"] == baseline).any():
        # Days the strategy and baseline both scored (fair, matched comparison).
        base = df[df["strategy"] == baseline][["city", "date"]]
        n_matched = (df.merge(base, on=["city", "date"], how="inner")
                     .groupby("strategy").size())
        agg["n_matched"] = agg["strategy"].map(n_matched).fillna(0).astype(int)
        agg["skill_brier"] = agg["strategy"].map(_matched_skill(df, baseline, "brier"))
        agg["skill_log_loss"] = agg["strategy"].map(_matched_skill(df, baseline, "log_loss"))
        agg["skill_crps"] = agg["strategy"].map(_matched_skill(df, baseline, "crps"))
    return agg.sort_values("brier").reset_index(drop=True)


def render_markdown(board: pd.DataFrame, *, title: str = "Backtest tournament") -> str:
    """Human leaderboard with a recommended Kelly-λ tier per strategy."""
    lines = [f"# {title}", ""]
    lines.append("Lower Brier / log-loss / CRPS / calib-err is better. "
                 "Skill > 0 means it beats the climatology null model.")
    lines.append("")
    cols = ["strategy", "n_days", "brier", "log_loss", "crps", "calib_err"]
    skill_cols = [c for c in ("skill_brier", "skill_log_loss", "skill_crps") if c in board]
    cols += skill_cols
    header = "| " + " | ".join(cols) + " | λ tier |"
    sep = "|" + "|".join(["---"] * (len(cols) + 1)) + "|"
    lines += [header, sep]
    for _, row in board.iterrows():
        tier = _lambda_tier(row)
        cells = []
        for c in cols:
            v = row[c]
            cells.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + f" | {tier} |")
    lines.append("")
    lines.append("**λ tier** is a suggested fractional-Kelly cap: `none` = do not "
                 "trade (no skill or mis-calibrated); `0.10/0.25` scale with demonstrated "
                 "skill and calibration. Promote only after the forward paper round confirms it.")
    return "\n".join(lines)


def _lambda_tier(row) -> str:
    skill = row.get("skill_brier", float("nan"))
    calib = row.get("calib_err", 1.0)
    if not (skill > 0) or calib > 0.10:
        return "none"
    if skill > 0.15 and calib < 0.05:
        return "0.25"
    return "0.10"
