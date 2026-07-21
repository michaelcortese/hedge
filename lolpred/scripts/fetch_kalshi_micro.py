#!/usr/bin/env python
"""Fetch Kalshi LoL microstructure data: all public trades + minute candles.

For every ticker in the odds parquet (default data/odds/kalshi_lol.parquet):
  1. All public trades (paginated /markets/trades)
       -> data/odds/kalshi_lol_trades.parquet
  2. Minute candles (period_interval=1) over the market's active life,
     (match_start - 12h, fallback open_time) .. close_time, chunked <= 24h
     per call -> data/odds/kalshi_lol_candles_1m.parquet

Incremental: tickers already complete in the candles parquet are skipped and
outputs are flushed every ~50 tickers, so a crash/rerun resumes. Public market
data — no Kalshi credentials needed.

Usage:
  .venv/bin/python scripts/fetch_kalshi_micro.py
  .venv/bin/python scripts/fetch_kalshi_micro.py --limit 5            # debug
  .venv/bin/python scripts/fetch_kalshi_micro.py --tickers-from data/odds/other.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lolpred.data.kalshi_micro import build_fetch_plan, fetch_micro  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fetch_kalshi_micro")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--tickers-from", default="data/odds/kalshi_lol.parquet",
        help="parquet with the tickers to fetch (needs ticker; match_start optional)",
    )
    ap.add_argument(
        "--markets", default="data/odds/kalshi_lol_markets.parquet",
        help="raw settled-markets parquet supplying open_time/close_time",
    )
    ap.add_argument("--trades-out", default="data/odds/kalshi_lol_trades.parquet")
    ap.add_argument("--candles-out", default="data/odds/kalshi_lol_candles_1m.parquet")
    ap.add_argument("--limit", type=int, default=None, help="debug: only first N tickers")
    ap.add_argument("--pace", type=float, default=0.2, help="seconds between HTTP calls")
    ap.add_argument("--flush-every", type=int, default=50,
                    help="write outputs every N tickers (crash-resumable)")
    args = ap.parse_args()

    tickers_df = pd.read_parquet(args.tickers_from)
    markets_df = None
    if args.markets and Path(args.markets).exists():
        markets_df = pd.read_parquet(args.markets)

    plan = build_fetch_plan(tickers_df, markets_df=markets_df)
    log.info("fetch plan: %d tickers from %s", len(plan), args.tickers_from)
    if args.limit:
        plan = plan.head(args.limit)

    t0 = time.monotonic()
    summary = fetch_micro(
        plan,
        trades_path=args.trades_out,
        candles_path=args.candles_out,
        session=requests.Session(),
        pace_s=args.pace,
        flush_every=args.flush_every,
        progress_every=50,
    )
    runtime = time.monotonic() - t0

    n_trades_total = n_candles_total = 0
    if Path(args.trades_out).exists():
        n_trades_total = len(pd.read_parquet(args.trades_out, columns=["ticker"]))
    if Path(args.candles_out).exists():
        n_candles_total = len(pd.read_parquet(args.candles_out, columns=["ticker"]))

    print()
    print(f"tickers in plan:     {len(plan)}")
    print(f"fetched this run:    {summary['tickers_done']}")
    print(f"skipped (cached):    {summary['tickers_skipped_cached']}")
    print(f"failed:              {summary['tickers_failed']}")
    print(f"new trade rows:      {summary['n_trades']}  (file total {n_trades_total}) -> {args.trades_out}")
    print(f"new candle rows:     {summary['n_candles']}  (file total {n_candles_total}) -> {args.candles_out}")
    print(f"runtime:             {runtime / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
