#!/usr/bin/env python
"""Fetch Leaguepedia ScoreboardGames -> canonical rows -> merged game table.

Bridges the gap left by the unavailable Oracle's Elixir 2026 file: fetches
pro games from Leaguepedia's Cargo API for a date window, converts them to
the canonical game-table schema (docs/CONTRACTS.md section 1), and appends
them to the OE table (rows strictly after OE's max date only).

Does NOT build features — the orchestrator does that on the merged table.

Usage:
    .venv/bin/python scripts/fetch_leaguepedia.py \
        --from 2025-10-06 --to 2026-07-15 \
        --out data/raw/leaguepedia_2026.parquet \
        --merged-out data/processed/games_extended.parquet \
        --oe-games data/processed/games.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lolpred.data.leaguepedia import (  # noqa: E402
    fetch_scoreboard_games,
    merge_with_canonical,
    to_canonical,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--from", dest="date_from", default="2025-10-06",
                   help="start date (UTC, inclusive) [%(default)s]")
    p.add_argument("--to", dest="date_to", default=str(date.today()),
                   help="end date (UTC, inclusive) [today]")
    p.add_argument("--out", default="data/raw/leaguepedia_2026.parquet",
                   help="canonical LP rows parquet [%(default)s]")
    p.add_argument("--merged-out",
                   default="data/processed/games_extended.parquet",
                   help="merged OE+LP game table parquet [%(default)s]")
    p.add_argument("--oe-games", default="data/processed/games.parquet",
                   help="existing OE canonical game table [%(default)s]")
    p.add_argument("--pace", type=float, default=2.0,
                   help="seconds between API calls [%(default)s]")
    p.add_argument("--max-pages", type=int, default=100,
                   help="max 500-row pages to fetch [%(default)s]")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    print(f"fetching Leaguepedia ScoreboardGames "
          f"{args.date_from} .. {args.date_to} (pace {args.pace}s)")
    raw = fetch_scoreboard_games(args.date_from, args.date_to,
                                 pace_s=args.pace, max_pages=args.max_pages)
    print(f"raw rows: {len(raw)}")

    lp = to_canonical(raw)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lp.to_parquet(out_path, index=False)
    print(f"canonical LP games: {len(lp)} -> {out_path}")
    if lp.empty:
        print("no games fetched; skipping merge")
        return 0

    # --- summary: games by league by month ---
    by = (lp.assign(month=lp["date"].dt.to_period("M").astype(str))
            .groupby(["league", "month"]).size().unstack(fill_value=0))
    by = by.loc[by.sum(axis=1).sort_values(ascending=False).index]
    print("\ngames by league by month (top 20 leagues):")
    print(by.head(20).to_string())
    print(f"\ndate range: {lp['date'].min()} .. {lp['date'].max()}")

    oe_path = Path(args.oe_games)
    if not oe_path.exists():
        print(f"\nOE games table {oe_path} not found; skipping merge")
        return 0

    oe = pd.read_parquet(oe_path)
    merged = merge_with_canonical(oe, lp)
    merged_path = Path(args.merged_out)
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(merged_path, index=False)

    renames = merged.attrs.get("lp_renames", {})
    new_teams = merged.attrs.get("lp_new_teams", [])
    print(f"\nmerged: {len(oe)} OE + {merged.attrs.get('lp_appended', 0)} LP "
          f"appended -> {len(merged)} games -> {merged_path}")
    print(f"renamed teams (LP -> OE): {len(renames)}")
    for lp_name, oe_name in sorted(renames.items()):
        print(f"  {lp_name!r} -> {oe_name!r}")
    print(f"new teams (kept as-is): {len(new_teams)}")
    for t in sorted(new_teams):
        print(f"  {t!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
