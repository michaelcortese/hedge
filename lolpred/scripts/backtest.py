#!/usr/bin/env python
"""Walk-forward backtest: model quality + a SYNTHETIC-odds betting simulation.

Pipeline: load the feature parquet -> make_folds -> run_walkforward (WinModel
vs the EloLogisticBaseline, everything strictly out of sample) -> build a
synthetic bookmaker line -> select/settle quarter-Kelly bets -> summarize.

SYNTHETIC ODDS — the load-bearing caveat
----------------------------------------
No historical esports odds ship with this repo, so the bookmaker line is
synthesized with ``make_synthetic_odds``. To avoid grading the model against
odds derived from itself, the reference probability is the OUT-OF-SAMPLE
BASELINE prediction (``baseline_p``, the Elo+BT logistic), shrunk + noised +
vigged — NOT the model being evaluated. This makes the "bookmaker" an
independent-ish reference, but only independent-ish: the baseline is trained
on the same games as the model, so shared regime shifts hit both. Every
betting number in the report is therefore labeled SYNTHETIC ODDS and is
plumbing validation, not evidence of real edge.

Artifacts written to --out-dir: preds.parquet (out-of-sample predictions),
bets.parquet (settled bets), report.txt, metrics.json.

Exit code is 0 even when the strategy loses money — this is a report, not a
gate.

Usage:
  .venv/bin/python scripts/backtest.py --features data/processed/features.parquet
  .venv/bin/python scripts/backtest.py --holdout-start 2025-01-01 --min-edge 0.06
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from lolpred.backtest.betting import make_synthetic_odds, select_bets, settle_bets
from lolpred.backtest.report import momentum_test, summarize
from lolpred.backtest.walkforward import make_folds, run_walkforward
from lolpred.models.xgb import EloLogisticBaseline, WinModel


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Walk-forward backtest with a synthetic-odds betting sim.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--features", default="data/processed/features.parquet",
                    help="feature parquet from scripts/build_features.py")
    ap.add_argument("--out-dir", default="artifacts/backtest",
                    help="directory for preds/bets/report/metrics artifacts")
    ap.add_argument("--burn-in-end", default="2018-12-31",
                    help="last day of the never-tested burn-in period; if the "
                    "data starts later, make_folds falls back to the first "
                    "30%% of the date span")
    ap.add_argument("--fold-months", type=int, default=6,
                    help="calendar months per walk-forward test fold")
    ap.add_argument("--gap-days", type=int, default=7,
                    help="embargo days between a fold's training data and its "
                    "test start")
    ap.add_argument("--holdout-start", default=None,
                    help="start date of a single final holdout fold (e.g. "
                    "2023-01-01). Default None = no holdout: all post-burn-in "
                    "data is tiled into folds. Set it once you are done "
                    "tuning, and look at the holdout row exactly once.")
    ap.add_argument("--min-edge", type=float, default=0.05,
                    help="minimum edge vs the vigged implied prob to bet")
    ap.add_argument("--kelly-mult", type=float, default=0.25,
                    help="fraction of full Kelly to stake (0.25 = quarter)")
    ap.add_argument("--max-stake", type=float, default=0.02,
                    help="cap on stake as a fraction of bankroll per bet")
    ap.add_argument("--min-hist-games", type=int, default=10,
                    help="refuse to bet unless BOTH teams have at least this "
                    "many prior games (cold-start gate)")
    ap.add_argument("--odds", default="synthetic", choices=["synthetic"],
                    help="odds source. Only 'synthetic' is supported today; "
                    "the flag is reserved for real historical odds files "
                    "later. Synthetic odds are generated from the "
                    "out-of-sample BASELINE predictions (baseline_p), not "
                    "from the evaluated model — see the module docstring "
                    "caveat: the baseline is independent-ish, not independent.")
    ap.add_argument("--synthetic-shrink", type=float, default=0.9,
                    help="logit shrink of the synthetic bookmaker toward 0.5")
    ap.add_argument("--synthetic-noise", type=float, default=0.4,
                    help="sd of the synthetic bookmaker's logit noise")
    ap.add_argument("--synthetic-margin", type=float, default=0.05,
                    help="synthetic bookmaker overround (booksum = 1 + margin)")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for model fits and synthetic-odds noise")
    ap.add_argument("--min-league-games", type=int, default=0,
                    help="restrict to leagues with at least this many games "
                    "in the feature frame (0 = keep all)")
    ap.add_argument("--model-params", default=None,
                    help="JSON dict of XGBoost param overrides for WinModel, "
                    'e.g. \'{"n_estimators": 200, "n_jobs": 2}\' '
                    "(mainly for tests / quick runs)")
    return ap.parse_args(argv)


def _jsonify(obj):
    """Recursively convert numpy/pandas objects into JSON-serializable ones."""
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return [_jsonify(r) for r in obj.to_dict(orient="records")]
    if isinstance(obj, pd.Series):
        return _jsonify(obj.to_dict())
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    t0 = time.monotonic()

    feats = pd.read_parquet(args.features)
    feats["date"] = pd.to_datetime(feats["date"])
    print(f"loaded {len(feats)} games from {args.features} "
          f"({feats['date'].min().date()} .. {feats['date'].max().date()})")

    if args.min_league_games > 0:
        counts = feats["league"].value_counts()
        keep = counts.index[counts >= args.min_league_games]
        before = len(feats)
        feats = feats[feats["league"].isin(keep)]
        print(f"--min-league-games {args.min_league_games}: "
              f"{before} -> {len(feats)} games in {len(keep)} leagues")

    model_params = json.loads(args.model_params) if args.model_params else None

    # ---- walk-forward out-of-sample predictions ---------------------------
    folds = make_folds(
        feats["date"],
        burn_in_end=args.burn_in_end,
        fold_months=args.fold_months,
        gap_days=args.gap_days,
        holdout_start=args.holdout_start,
    )
    if not folds:
        print("no viable folds — not enough post-burn-in data")
        return 0
    print(f"{len(folds)} folds "
          f"({folds[0].test_start.date()} .. {folds[-1].test_end.date()})")

    preds = run_walkforward(
        feats,
        folds,
        model_factory=lambda: WinModel(params=model_params, seed=args.seed),
        baseline_factory=lambda: EloLogisticBaseline(seed=args.seed),
        seed=args.seed,
    )
    if preds.empty:
        print("walk-forward produced no out-of-sample predictions")
        return 0

    # ---- synthetic bookmaker from the BASELINE (not the model) ------------
    odds = make_synthetic_odds(
        preds["baseline_p"],  # Series -> index preserved for alignment
        shrink=args.synthetic_shrink,
        noise_sd=args.synthetic_noise,
        margin=args.synthetic_margin,
        seed=args.seed,
    )
    preds = preds.copy()
    preds["fair_blue"] = odds["fair_blue"]  # scored as market baseline

    hist_blue = feats.loc[preds.index, "f_hist_games_blue"].to_numpy()
    hist_red = feats.loc[preds.index, "f_hist_games_red"].to_numpy()

    bets = select_bets(
        preds["model_p"],
        odds["odds_blue"],
        odds["odds_red"],
        min_edge=args.min_edge,
        kelly_mult=args.kelly_mult,
        max_stake=args.max_stake,
        min_hist_games=args.min_hist_games,
        hist_games_blue=hist_blue,
        hist_games_red=hist_red,
    )
    settled = settle_bets(bets, preds["blue_win"])

    # ---- report ------------------------------------------------------------
    metrics, text = summarize(preds, settled, synthetic_odds=True)

    momentum = None
    mom_cols = {"series_id", "game_in_series", "blue_team", "red_team"}
    if mom_cols.issubset(preds.columns):
        momentum = momentum_test(preds, seed=args.seed)
        text += (
            "\n-- Momentum test (within-series iid check) --\n"
            f"lag_coef:       {momentum['lag_coef']:+.4f} "
            "(0 under no momentum; positive = prev-game winner over-performs)\n"
            f"sign_stability: {momentum['sign_stability']:.3f} "
            "(~0.5 noise, ~1.0 stable effect)\n"
            f"n lag games:    {momentum['n']}\n"
        )

    text += (
        "\nNOTE: synthetic odds were generated from the out-of-sample "
        "BASELINE\npredictions (Elo+BT logistic), not from the evaluated "
        "model. The baseline is\nan independent-ish reference only — it is "
        "trained on the same games — so\nbetting numbers validate plumbing, "
        "not real market edge.\n"
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(out_dir / "preds.parquet")
    bets_out = settled.join(
        preds[["gameid", "date", "league", "blue_team", "red_team", "blue_win"]],
        how="left",
    )
    bets_out.to_parquet(out_dir / "bets.parquet")
    (out_dir / "report.txt").write_text(text)

    metrics_json = _jsonify(
        {
            **metrics,
            "momentum": momentum,
            "n_folds": len(folds),
            "elapsed_s": round(time.monotonic() - t0, 1),
            "args": vars(args),
        }
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics_json, indent=2))

    print()
    print(text)
    print(f"artifacts -> {out_dir}/ (preds.parquet, bets.parquet, "
          f"report.txt, metrics.json) in {time.monotonic() - t0:.1f}s")
    return 0  # a losing model is still a successful report


if __name__ == "__main__":
    raise SystemExit(main())
