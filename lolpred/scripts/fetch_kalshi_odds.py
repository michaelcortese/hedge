#!/usr/bin/env python
"""Fetch Kalshi LoL match-market odds into parquet for backtesting.

Pulls all settled markets for a series (default KXLOLGAME), then for each
market fetches hourly candlesticks and records pre-match price snapshots
(t-5min and t-60min before match start). Incremental: reruns only fetch
markets newer than what's already in the output parquets.

Public market data — no Kalshi credentials needed.

Usage:
  .venv/bin/python scripts/fetch_kalshi_odds.py
  .venv/bin/python scripts/fetch_kalshi_odds.py --limit 20            # debug
  .venv/bin/python scripts/fetch_kalshi_odds.py --minute-candles      # refine t5
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

from lolpred.data.kalshi_market import (  # noqa: E402
    build_market_prices,
    fetch_market_candles,
    fetch_settled_lol_markets,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fetch_kalshi_odds")


def refine_t5_with_minute_candles(
    prices: pd.DataFrame,
    tickers: set[str],
    session: requests.Session,
    pace_s: float = 0.15,
) -> pd.DataFrame:
    """Replace the t5 snapshot with minute-resolution candles over the final hour."""
    prices = prices.copy()
    n = 0
    for idx, row in prices.iterrows():
        if row["ticker"] not in tickers:
            continue
        start = row["match_start"]
        if pd.isna(start):
            continue
        start_unix = int(pd.Timestamp(start).timestamp())
        series_ticker = str(row["ticker"]).split("-")[0]
        try:
            candles = fetch_market_candles(
                row["ticker"], series_ticker,
                start_unix - 3900, start_unix,  # final 65 min
                period_interval=1, session=session,
            )
        except Exception as exc:
            log.warning("minute candles failed for %s: %s", row["ticker"], exc)
            time.sleep(pace_s)
            continue
        time.sleep(pace_s)
        sel = candles[candles["end_period_ts"] <= start_unix - 5 * 60]
        if sel.empty:
            continue
        last = sel.iloc[-1]
        bid, ask = last["yes_bid_close"], last["yes_ask_close"]
        prices.loc[idx, ["t5_bid", "t5_ask", "t5_mid", "t5_ts"]] = [
            bid, ask, (bid + ask) / 2.0, last["ts"],
        ]
        n += 1
    log.info("minute-candle refinement applied to %d markets", n)
    return prices


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--series", default="KXLOLGAME")
    ap.add_argument("--out", default="data/odds/kalshi_lol.parquet")
    ap.add_argument("--raw-markets-out", default="data/odds/kalshi_lol_markets.parquet")
    ap.add_argument(
        "--minute-candles", action="store_true",
        help="also fetch period_interval=1 for the final hour and refine the t5 snapshot",
    )
    ap.add_argument("--limit", type=int, default=None, help="debug: only first N markets")
    args = ap.parse_args()

    out = Path(args.out)
    raw_out = Path(args.raw_markets_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw_out.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    markets = fetch_settled_lol_markets(
        series_ticker=args.series, session=session, cache_path=raw_out,
    )
    markets.to_parquet(raw_out, index=False)
    log.info("settled markets: %d (saved %s)", len(markets), raw_out)

    if args.limit:
        markets = markets.head(args.limit)

    already_priced: set[str] = set()
    if out.exists():
        already_priced = set(pd.read_parquet(out, columns=["ticker"])["ticker"])

    prices = build_market_prices(markets, cache_path=out, session=session)
    skips = prices.attrs.get("skip_reasons", {})

    if args.minute_candles and len(prices):
        new_tickers = set(prices["ticker"]) - already_priced
        prices = refine_t5_with_minute_candles(prices, new_tickers, session)

    prices.to_parquet(out, index=False)

    # ---- summary ------------------------------------------------------------
    spread = (prices["t5_ask"] - prices["t5_bid"]).dropna()
    print()
    print(f"series:            {args.series}")
    print(f"settled markets:   {len(markets)}")
    print(f"priced markets:    {len(prices)}  -> {out}")
    print(f"skip reasons:      {skips or 'none'}")
    if len(prices):
        print(f"date window:       {prices['match_start'].min()} .. {prices['match_start'].max()}")
        print(f"mean t5 spread:    {spread.mean():.4f}" if len(spread) else "mean t5 spread:    n/a")
        print(f"share volume>0:    {(prices['volume'] > 0).mean():.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
