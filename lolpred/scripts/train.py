#!/usr/bin/env python
"""Train the final WinModel on all data up to a cutoff and save artifacts.

The chronologically LAST ``--calib-frac`` of the training window is held out
as a tail, exactly the scheme run_walkforward uses per fold: when the tail
has at least ``_MIN_TAIL_FOR_CAL_SPLIT`` (200) rows it is split
chronologically in half — the FIRST half is the early-stopping validation
set, the SECOND (most recent) half is the Platt calibration slice
(``X_cal``/``y_cal``) — so the calibrator is not fit on the same rows the
stopping point was chosen on.  Smaller tails do double duty as the val set
(no separate cal slice); everything before the tail is fit data.

Artifacts written to --out-dir:
  * ``model.joblib``       — the fitted WinModel (via WinModel.save)
  * ``training_meta.json`` — cutoff, row counts, feature columns,
                             best_iteration, calibrated flag, val logloss
                             (early-stop half only) + cal logloss (when a
                             calibration slice exists)

Note: no rolling feature state is persisted. scripts/predict.py recomputes
features from the canonical games table at prediction time (fast with the
games.parquet cache written by build_features.py), which reuses the one
tested feature-builder code path instead of a second stale-state one. Ergo:
prediction requires the raw CSVs (or the games cache) to be present.

Usage:
  .venv/bin/python scripts/train.py --features data/processed/features.parquet
  .venv/bin/python scripts/train.py --train-end 2025-12-31 --out-dir artifacts/model
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from lolpred.backtest.report import probability_metrics
from lolpred.backtest.walkforward import _MIN_TAIL_FOR_CAL_SPLIT
from lolpred.models.xgb import PERSP_COL, WinModel


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Train the final WinModel and save model artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--features", default="data/processed/features.parquet",
                    help="feature parquet from scripts/build_features.py")
    ap.add_argument("--train-end", default=None,
                    help="train on games with date <= this (YYYY-MM-DD); "
                    "default: the max date in the feature table")
    ap.add_argument("--calib-frac", type=float, default=0.15,
                    help="chronological tail fraction held out for early "
                    "stopping + Platt calibration")
    ap.add_argument("--out-dir", default="artifacts/model",
                    help="directory for model.joblib + training_meta.json")
    ap.add_argument("--seed", type=int, default=0, help="model seed")
    ap.add_argument("--model-params", default=None,
                    help="JSON dict of XGBoost param overrides for WinModel, "
                    'e.g. \'{"n_estimators": 200, "n_jobs": 2}\' '
                    "(mainly for tests / quick runs)")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    t0 = time.monotonic()

    feats = pd.read_parquet(args.features)
    feats["date"] = pd.to_datetime(feats["date"])
    train_end = (
        pd.Timestamp(args.train_end) if args.train_end else feats["date"].max()
    )
    feats = feats[feats["date"] <= train_end].sort_values("date", kind="stable")
    if feats.empty:
        print(f"no games with date <= {train_end.date()}; nothing to train on")
        return 1

    feature_cols = [c for c in feats.columns
                    if c.startswith("f_") and c != PERSP_COL]
    X = feats[feature_cols].reset_index(drop=True)
    y = feats["blue_win"].to_numpy(dtype=int)

    # Tail convention (identical to run_walkforward): chronologically last
    # calib_frac of rows; tails >= _MIN_TAIL_FOR_CAL_SPLIT are split in half
    # into early-stopping val (first) / Platt calibration (second, most
    # recent); smaller tails do double duty as the val set.
    n = len(X)
    n_tail = min(int(round(args.calib_frac * n)), n - 1)
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
          f"through {train_end.date()}")

    model_params = json.loads(args.model_params) if args.model_params else None
    model = WinModel(params=model_params, seed=args.seed)
    model.fit(X_fit, y_fit, X_val, y_val, X_cal=X_cal, y_cal=y_cal)

    val_logloss = None  # on the early-stop half only
    if X_val is not None:
        val_logloss = probability_metrics(
            y_val, model.predict_proba(X_val)
        )["logloss"]
    cal_logloss = None  # on the calibration half, when present
    if X_cal is not None:
        cal_logloss = probability_metrics(
            y_cal, model.predict_proba(X_cal)
        )["logloss"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.joblib"
    model.save(model_path)

    best_it = model.best_iteration_
    meta = {
        "train_end": str(train_end.date()),
        "n_games": int(n),
        "n_fit": int(len(X_fit)),
        "n_val": 0 if X_val is None else int(len(X_val)),
        "n_cal": 0 if X_cal is None else int(len(X_cal)),
        "feature_columns": feature_cols,
        "best_iteration": int(best_it) if best_it is not None else None,
        "calibrated": bool(model.calibrated_),
        "val_logloss": val_logloss,
        "cal_logloss": cal_logloss,
        "seed": args.seed,
        "model_params": model_params,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "features_path": str(args.features),
    }
    (out_dir / "training_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"saved {model_path}")
    print(f"saved {out_dir / 'training_meta.json'}")
    print(f"best_iteration: {meta['best_iteration']}  "
          f"calibrated: {meta['calibrated']}  "
          f"val logloss: {val_logloss if val_logloss is None else round(val_logloss, 4)}  "
          f"cal logloss: {cal_logloss if cal_logloss is None else round(cal_logloss, 4)}")
    print(f"elapsed: {time.monotonic() - t0:.1f}s")
    print("note: predict.py rebuilds features from the games table at "
          "prediction time — keep data/raw (or the games.parquet cache) "
          "around.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
