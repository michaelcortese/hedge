#!/usr/bin/env python
"""Evaluate the WinModel against REAL settled Kalshi LoL match markets.

The decisive edge test: train on everything strictly before --train-end
(same held-out-tail convention as scripts/train.py / run_walkforward: the
chronologically last 15% of training rows is held out; tails >= 200 rows are
split in half into an early-stopping validation set and a Platt calibration
slice), join each settled Kalshi market to its series in the feature table,
and score model probabilities against the market's own pre-match prices —
probability quality vs the t-5-minute mid plus an executable-touch P&L
simulation at the t-5 bid/ask with Kalshi taker fees.

Leakage guard: any joined market whose game-1 date is EARLIER than
--train-end (i.e. the model saw the game) is dropped and counted.

Artifacts written to --out-dir: report.txt, metrics.json, matched.parquet.

Usage:
  .venv/bin/python scripts/eval_vs_kalshi.py
  .venv/bin/python scripts/eval_vs_kalshi.py --train-end 2026-05-01 \\
      --markets data/odds/kalshi_lol.parquet --min-volume 10
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from lolpred.backtest.kalshi_eval import evaluate, join_markets_to_series
from lolpred.backtest.walkforward import _MIN_TAIL_FOR_CAL_SPLIT
from lolpred.models.xgb import PERSP_COL, WinModel


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Evaluate the model against real settled Kalshi LoL markets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--features", default="data/processed/features_extended.parquet",
                    help="feature parquet (must extend past --train-end)")
    ap.add_argument("--markets", default="data/odds/kalshi_lol.parquet",
                    help="Kalshi price parquet from build_market_prices")
    ap.add_argument("--train-end", default="2026-05-01",
                    help="train on games with date STRICTLY before this "
                    "(YYYY-MM-DD); markets settling on earlier games are dropped")
    ap.add_argument("--out-dir", default="artifacts/kalshi_eval",
                    help="directory for report.txt / metrics.json / matched.parquet")
    ap.add_argument("--seed", type=int, default=0, help="model + bootstrap seed")
    ap.add_argument("--min-volume", type=float, default=0.0,
                    help="drop markets with total volume below this (0 = keep all)")
    ap.add_argument("--calib-frac", type=float, default=0.15,
                    help="chronological training tail held out for early "
                    "stopping + Platt calibration")
    ap.add_argument("--model-params", default=None,
                    help="JSON dict of XGBoost param overrides for WinModel "
                    '(test hook), e.g. \'{"n_estimators": 60, "n_jobs": 2}\'')
    return ap.parse_args(argv)


def _train_model(feats: pd.DataFrame, train_end: pd.Timestamp,
                 calib_frac: float, seed: int, model_params: dict | None) -> WinModel:
    """Fit a WinModel on date < train_end with the walk-forward tail convention.

    Identical scheme to scripts/train.py / run_walkforward: the
    chronologically last ``calib_frac`` of rows is a held-out tail; tails of
    at least ``_MIN_TAIL_FOR_CAL_SPLIT`` (200) rows are split in half — early-
    stopping val first, Platt calibration slice second (most recent) —
    smaller tails do double duty as the val set.
    """
    train = feats[feats["date"] < train_end].sort_values("date", kind="stable")
    if train.empty:
        raise SystemExit(f"no training games with date < {train_end.date()}")

    feature_cols = [c for c in train.columns if c.startswith("f_") and c != PERSP_COL]
    X = train[feature_cols].reset_index(drop=True)
    y = train["blue_win"].to_numpy(dtype=int)

    n = len(X)
    n_tail = min(int(round(calib_frac * n)), n - 1)
    X_val = y_val = X_cal = y_cal = None
    if n_tail >= 2:
        X_fit, y_fit = X.iloc[:-n_tail], y[:-n_tail]
        X_tail, y_tail = X.iloc[-n_tail:], y[-n_tail:]
        if n_tail >= _MIN_TAIL_FOR_CAL_SPLIT:
            half = n_tail // 2
            X_val, y_val = X_tail.iloc[:half], y_tail[:half]
            X_cal, y_cal = X_tail.iloc[half:], y_tail[half:]
        else:
            X_val, y_val = X_tail, y_tail
    else:
        X_fit, y_fit = X, y

    print(f"training on {len(X_fit)} games "
          f"(val tail: {0 if X_val is None else len(X_val)}, "
          f"cal tail: {0 if X_cal is None else len(X_cal)}) "
          f"strictly before {train_end.date()}")
    model = WinModel(params=model_params, seed=seed)
    model.fit(X_fit, y_fit, X_val, y_val, X_cal=X_cal, y_cal=y_cal)
    print(f"best_iteration: {model.best_iteration_}  calibrated: {model.calibrated_}")
    return model


def _jsonify(o):
    """Recursively coerce metrics to strict-JSON-safe types (NaN -> null)."""
    if isinstance(o, dict):
        return {str(k): _jsonify(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonify(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, pd.Timestamp):
        return str(o)
    return o


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    t0 = time.monotonic()

    markets_path = Path(args.markets)
    if not markets_path.is_file():
        raise SystemExit(f"markets parquet not found: {markets_path} "
                         "(run scripts/fetch_kalshi_odds.py first)")

    feats = pd.read_parquet(args.features)
    feats["date"] = pd.to_datetime(feats["date"])
    train_end = pd.Timestamp(args.train_end)

    model_params = json.loads(args.model_params) if args.model_params else None
    model = _train_model(feats, train_end, args.calib_frac, args.seed, model_params)

    markets = pd.read_parquet(markets_path)
    joined = join_markets_to_series(markets, feats)
    join_attrs = dict(joined.attrs)

    # Leakage guard: every evaluated market's game-1 must postdate the
    # training window; drop (and count) any earlier ones.
    early = pd.to_datetime(joined["game1_date"]) < train_end
    n_pre_train = int(early.sum())
    if n_pre_train:
        print(f"WARNING: dropping {n_pre_train} matched markets with game-1 "
              f"date < {train_end.date()} (inside the training window)")
    joined = joined[~early].copy()
    joined.attrs.update(join_attrs)
    assert (pd.to_datetime(joined["game1_date"]) >= train_end).all()

    metrics, report = evaluate(
        joined, feats, model,
        min_volume=args.min_volume, seed=args.seed,
    )
    metrics["train_end"] = str(train_end.date())
    metrics["n_dropped_pre_train_end"] = n_pre_train
    metrics["model"] = {
        "best_iteration": (int(model.best_iteration_)
                           if model.best_iteration_ is not None else None),
        "calibrated": bool(model.calibrated_),
        "seed": args.seed,
        "model_params": model_params,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_market = metrics.pop("per_market")
    per_market.to_parquet(out_dir / "matched.parquet")
    (out_dir / "report.txt").write_text(report + "\n")
    (out_dir / "metrics.json").write_text(
        json.dumps(_jsonify(metrics), indent=2, allow_nan=False)
    )

    print(report)
    print(f"wrote {out_dir / 'report.txt'}, {out_dir / 'metrics.json'}, "
          f"{out_dir / 'matched.parquet'}")
    print(f"elapsed: {time.monotonic() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
