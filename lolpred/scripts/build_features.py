#!/usr/bin/env python
"""Build the matchup feature table from raw Oracle's Elixir CSVs.

Loads the raw CSVs into the canonical game table (lolpred.data.loader),
optionally filters by year/league, runs the chronological feature builder
(lolpred.features.build) and writes:

  * ``--out`` (default data/processed/features.parquet): one row per game,
    meta columns + ``f_``-prefixed features — the input to backtest.py and
    train.py;
  * ``games.parquet`` next to ``--out``: the canonical game table itself —
    the fast-path cache consumed by predict.py (so prediction doesn't have
    to re-parse the raw CSVs).

Usage:
  .venv/bin/python scripts/build_features.py --raw data/raw --out data/processed/features.parquet
  .venv/bin/python scripts/build_features.py --raw data/raw/oe_2024.csv data/raw/oe_2025.csv \\
      --min-year 2019 --leagues LCK LPL LEC
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from lolpred.data.loader import filter_games, load_games
from lolpred.features.build import FEATURE_COLUMNS, build_matchup_features


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--raw",
        nargs="+",
        default=["data/raw"],
        help="raw OE CSV path(s), a directory (all *.csv under it) or globs "
        "(default: data/raw)",
    )
    ap.add_argument(
        "--out",
        default="data/processed/features.parquet",
        help="output parquet for the feature table (default: "
        "data/processed/features.parquet); games.parquet is written next to it",
    )
    ap.add_argument(
        "--min-year",
        type=int,
        default=None,
        help="keep only games with season year >= this (applied after load)",
    )
    ap.add_argument(
        "--leagues",
        nargs="+",
        default=None,
        help="keep only these leagues (exact OE league labels, e.g. LCK LPL)",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    t0 = time.monotonic()
    games = load_games(args.raw, verbose=True)
    if args.min_year is not None or args.leagues is not None:
        before = len(games)
        games = filter_games(games, leagues=args.leagues, min_year=args.min_year)
        print(f"filtered {before} -> {len(games)} games "
              f"(min_year={args.min_year}, leagues={args.leagues})")
    if games.empty:
        print("no games left after loading/filtering; nothing to do", file=sys.stderr)
        return 1

    t_load = time.monotonic()
    feats = build_matchup_features(games)
    t_build = time.monotonic()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out, index=False)
    games_out = out.parent / "games.parquet"
    games.to_parquet(games_out, index=False)
    elapsed = time.monotonic() - t0

    n_feat = len([c for c in feats.columns if c.startswith("f_")])
    print()
    print(f"games:      {len(feats)}")
    print(f"date range: {feats['date'].min().date()} .. {feats['date'].max().date()}")
    print(f"leagues:    {feats['league'].nunique()}")
    print(f"features:   {n_feat} (contract lists {len(FEATURE_COLUMNS)})")
    print(f"wrote       {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"wrote       {games_out} ({games_out.stat().st_size / 1e6:.1f} MB) "
          "[games cache for predict.py]")
    print(f"elapsed:    {elapsed:.1f}s (load {t_load - t0:.1f}s, "
          f"features {t_build - t_load:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
