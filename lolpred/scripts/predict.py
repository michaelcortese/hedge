#!/usr/bin/env python
"""Predict a hypothetical matchup with a trained model: game/series probs, odds.

How the feature row is built (v1, deliberately simple and honest)
-----------------------------------------------------------------
There is no persisted per-team feature state. Instead a synthetic "upcoming
game" row (blue vs red on --date, stats NaN) is appended to the canonical
games table restricted to games STRICTLY BEFORE --date, and the tested
chronological feature builder is run over the whole thing. The builder's
strict-date visibility guarantees the fake row's own (empty) stats never
matter: its features are computed purely from prior state. The last row's
``f_*`` columns are the feature vector.

Cost: one full feature build. Seconds when the canonical games cache
(``games.parquet``, written by scripts/build_features.py next to the feature
parquet) is available; ~a minute-plus when re-parsing raw CSVs via --raw.

Team names are matched case-insensitively (exact) against the games table;
on a miss, the closest names are suggested and the script exits with code 2.

Usage:
  .venv/bin/python scripts/predict.py --blue T1 --red "Gen.G" --best-of 5
  .venv/bin/python scripts/predict.py --blue T1 --red "Gen.G" --best-of 5 \\
      --odds-blue 1.65 --odds-red 2.30 --score 1-1 --json
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from lolpred.backtest.betting import devig_proportional, kelly_fraction
from lolpred.data.loader import load_games
from lolpred.features.build import build_matchup_features
from lolpred.models.xgb import WinModel
from lolpred.series import exact_score_probs, series_win_prob, wins_needed

DEFAULT_GAMES_CACHE = "data/processed/games.parquet"
MIN_HIST_GAMES = 10  # warning threshold (matches the backtest bet gate)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Predict game/series win probabilities for a matchup.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--blue", required=True, help="blue-side team name")
    ap.add_argument("--red", required=True, help="red-side team name")
    ap.add_argument("--best-of", type=int, choices=(1, 3, 5), default=3,
                    help="series format")
    ap.add_argument("--date", default=None,
                    help="prediction date YYYY-MM-DD (default: today); only "
                    "games strictly before it feed the features")
    ap.add_argument("--league-hint", default=None,
                    help="league label for the hypothetical game (default: "
                    "the blue team's most recent league)")
    ap.add_argument("--model-dir", default="artifacts/model",
                    help="directory containing model.joblib from train.py")
    ap.add_argument("--raw", default="data/raw",
                    help="raw OE CSVs (dir/glob) — slow fallback when no "
                    "games cache is available")
    ap.add_argument("--features", default=None,
                    help="feature parquet path; fast path — its sibling "
                    "games.parquet (written by build_features.py) is used "
                    "as the canonical games table")
    ap.add_argument("--games-cache", default=None,
                    help="canonical games parquet (fastest path; default: "
                    f"{DEFAULT_GAMES_CACHE} when it exists, else --raw)")
    ap.add_argument("--odds-blue", type=float, default=None,
                    help="bookmaker decimal odds on blue")
    ap.add_argument("--odds-red", type=float, default=None,
                    help="bookmaker decimal odds on red")
    ap.add_argument("--kelly-mult", type=float, default=0.25,
                    help="fraction of full Kelly for suggested stakes")
    ap.add_argument("--score", default=None,
                    help="mid-series state 'a-b' (blue wins-red wins), e.g. 1-1")
    ap.add_argument("--json", action="store_true",
                    help="machine-readable JSON output on stdout")
    return ap.parse_args(argv)


def _load_games_table(args: argparse.Namespace) -> pd.DataFrame:
    """Resolve the canonical games table: cache > features sibling > raw."""
    candidates: list[Path] = []
    if args.games_cache:
        candidates.append(Path(args.games_cache))
    if args.features:
        candidates.append(Path(args.features).parent / "games.parquet")
    if not candidates:
        default = Path(DEFAULT_GAMES_CACHE)
        if default.is_file():
            candidates.append(default)
    for path in candidates:
        if not path.is_file():
            raise FileNotFoundError(
                f"games cache not found: {path} "
                "(run scripts/build_features.py to create it)"
            )
        games = pd.read_parquet(path)
        games["date"] = pd.to_datetime(games["date"])
        return games
    print(f"no games cache; loading raw CSVs from {args.raw} (slow)...",
          file=sys.stderr)
    return load_games(args.raw)


def _resolve_team(name: str, games: pd.DataFrame) -> str | None:
    """Case-insensitive exact match against every team name in the table."""
    all_names = pd.unique(
        pd.concat([games["blue_team"], games["red_team"]], ignore_index=True)
    )
    by_lower = {str(n).lower(): str(n) for n in sorted(map(str, all_names))}
    hit = by_lower.get(name.lower())
    if hit is not None:
        return hit
    close = difflib.get_close_matches(name.lower(), by_lower, n=5, cutoff=0.5)
    print(f"unknown team {name!r}.", file=sys.stderr)
    if close:
        print("closest matches: "
              + ", ".join(by_lower[c] for c in close), file=sys.stderr)
    return None


def _feature_row(
    games: pd.DataFrame, blue: str, red: str, date: pd.Timestamp, league: str
) -> pd.DataFrame:
    """Feature vector for the hypothetical game via the tested builder path.

    Appends a stats-NaN "upcoming game" row dated ``date`` to the games
    strictly before ``date`` and rebuilds features; strict-date visibility
    means the fake row's features come purely from prior state.
    """
    hist = games[games["date"] < date]
    if hist.empty:
        raise SystemExit(f"no historical games before {date.date()}")
    fake = {c: np.nan for c in games.columns}
    fake.update(
        gameid="___PREDICT___",
        date=date,
        league=league,
        year=int(date.year),
        split=None,
        playoffs=0,
        patch=None,
        game_in_series=1,
        series_id=f"{date.date()}|{league}|{'|'.join(sorted((blue, red)))}",
        datacompleteness="synthetic",
        blue_team=blue,
        red_team=red,
        blue_win=0,  # placeholder; never visible to its own features
        blue_players="",
        red_players="",
    )
    table = pd.concat([hist, pd.DataFrame([fake])], ignore_index=True)
    feats = build_matchup_features(table)
    last = feats.iloc[[-1]]
    assert last["gameid"].iloc[0] == "___PREDICT___"
    return last


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    date = (pd.Timestamp(args.date) if args.date
            else pd.Timestamp.today().normalize())

    model_path = Path(args.model_dir) / "model.joblib"
    if not model_path.is_file():
        print(f"model not found: {model_path} (run scripts/train.py first)",
              file=sys.stderr)
        return 1
    model = WinModel.load(model_path)

    games = _load_games_table(args)

    blue = _resolve_team(args.blue, games)
    red = _resolve_team(args.red, games)
    if blue is None or red is None:
        return 2
    if blue == red:
        print(f"--blue and --red are the same team ({blue})", file=sys.stderr)
        return 2

    hist = games[games["date"] < date]
    league = args.league_hint
    if league is None:
        for team in (blue, red):
            mask = (hist["blue_team"] == team) | (hist["red_team"] == team)
            if mask.any():
                league = str(hist.loc[mask, "league"].iloc[-1])
                break
        else:
            league = "UNKNOWN"

    print(f"building features over {int((games['date'] < date).sum())} "
          f"historical games (< {date.date()})...", file=sys.stderr)
    row = _feature_row(games, blue, red, date, league)
    missing = [c for c in model.feature_columns_ if c not in row.columns]
    if missing:
        raise SystemExit(
            f"model expects feature columns absent from the builder output: "
            f"{missing} — model and code base out of sync; retrain."
        )
    X = row[model.feature_columns_]
    p = float(model.predict_proba(X)[0])

    hist_b = float(row["f_hist_games_blue"].iloc[0])
    hist_r = float(row["f_hist_games_red"].iloc[0])
    warnings: list[str] = []
    for team, n_hist in ((blue, hist_b), (red, hist_r)):
        if n_hist == 0:
            warnings.append(f"{team} has NO games before {date.date()} — "
                            "prediction is essentially a prior")
        elif n_hist < MIN_HIST_GAMES:
            warnings.append(f"{team} has only {int(n_hist)} prior games "
                            f"(< {MIN_HIST_GAMES}) — the backtest bet gate "
                            "would refuse this market")

    bo = args.best_of
    series_p = series_win_prob(p, bo)
    score_state = None
    if args.score:
        try:
            a_str, b_str = args.score.split("-", 1)
            a, b = int(a_str), int(b_str)
        except ValueError:
            print(f"--score must look like '1-1', got {args.score!r}",
                  file=sys.stderr)
            return 2
        w = wins_needed(bo)
        if not (0 <= a <= w and 0 <= b <= w) or (a == w and b == w):
            print(f"--score {args.score} impossible in a best-of-{bo}",
                  file=sys.stderr)
            return 2
        score_state = {
            "score": f"{a}-{b}",
            "series_p_blue": series_win_prob(p, bo, a, b),
        }
    scores = {f"{k[0]}-{k[1]}": v for k, v in exact_score_probs(p, bo).items()}

    betting = None
    if (args.odds_blue is None) != (args.odds_red is None):
        print("provide both --odds-blue and --odds-red (or neither)",
              file=sys.stderr)
        return 2
    if args.odds_blue is not None:
        imp_b, imp_r = 1.0 / args.odds_blue, 1.0 / args.odds_red
        fair_b, fair_r = devig_proportional(imp_b, imp_r)
        betting = {
            "odds_blue": args.odds_blue,
            "odds_red": args.odds_red,
            "implied_blue": imp_b,
            "implied_red": imp_r,
            "fair_blue": fair_b,
            "fair_red": fair_r,
            "edge_blue": p - imp_b,      # vs vigged price (what you pay)
            "edge_red": (1.0 - p) - imp_r,
            "kelly_mult": args.kelly_mult,
            "stake_frac_blue": args.kelly_mult * kelly_fraction(p, args.odds_blue),
            "stake_frac_red": args.kelly_mult
            * kelly_fraction(1.0 - p, args.odds_red),
        }

    result = {
        "blue": blue,
        "red": red,
        "date": str(date.date()),
        "league": league,
        "best_of": bo,
        "p_blue": p,
        "p_red": 1.0 - p,
        "fair_odds_blue": 1.0 / p,
        "fair_odds_red": 1.0 / (1.0 - p),
        "series_p_blue": series_p,
        "series_p_red": 1.0 - series_p,
        "series_from_score": score_state,
        "exact_score_probs": scores,
        "hist_games_blue": hist_b,
        "hist_games_red": hist_r,
        "warnings": warnings,
        "betting": betting,
    }

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print(f"\n{blue} (blue) vs {red} (red)  —  {league}, {date.date()}, "
          f"best-of-{bo}")
    print("-" * 64)
    print(f"game win prob:    {blue}: {p:.3f}   {red}: {1 - p:.3f}")
    print(f"fair decimal odds: {blue}: {1 / p:.2f}   {red}: {1 / (1 - p):.2f}")
    if bo > 1:
        print(f"series win prob:  {blue}: {series_p:.3f}   "
              f"{red}: {1 - series_p:.3f}")
    if score_state:
        sp = score_state["series_p_blue"]
        print(f"from score {score_state['score']}:  {blue}: {sp:.3f}   "
              f"{red}: {1 - sp:.3f}")
    if bo > 1:
        print("exact score distribution (blue-red):")
        for k, v in sorted(scores.items(), key=lambda kv: -kv[1]):
            print(f"    {k}: {v:.3f}")
    if betting:
        print(f"\nvs offered odds (blue {betting['odds_blue']:.2f} / "
              f"red {betting['odds_red']:.2f}, "
              f"book {betting['implied_blue'] + betting['implied_red']:.3f}):")
        print(f"  implied (vigged): blue {betting['implied_blue']:.3f}  "
              f"red {betting['implied_red']:.3f}")
        print(f"  fair (de-vigged): blue {betting['fair_blue']:.3f}  "
              f"red {betting['fair_red']:.3f}")
        print(f"  edge vs price:    blue {betting['edge_blue']:+.3f}  "
              f"red {betting['edge_red']:+.3f}")
        print(f"  {betting['kelly_mult']:.2f}-Kelly stake:  "
              f"blue {betting['stake_frac_blue']:.4f}  "
              f"red {betting['stake_frac_red']:.4f}  (bankroll fraction)")
    for w in warnings:
        print(f"\nWARNING: {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
