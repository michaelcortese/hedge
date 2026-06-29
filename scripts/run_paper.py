#!/usr/bin/env python
"""Forward paper tournament against Kalshi (demo by default).

Two modes:

  # log one snapshot cycle: every strategy's signal + live prices for open markets
  .venv/bin/python scripts/run_paper.py snapshot

  # after markets settle, score the logged signals into a P&L leaderboard
  .venv/bin/python scripts/run_paper.py score

Run ``snapshot`` on a schedule (e.g. hourly via cron or the /loop skill) across a
trading day; the nowcast/blend only act in the afternoon, so multiple cycles per
day matter. ``score`` reads back the settled results and ranks strategies by
realized per-contract P&L and fractional-Kelly bankroll growth.

Credentials (env vars, same as scripts/test_auth.py):
    KALSHI_KEY_ID, KALSHI_PRIVATE_KEY_PATH, optional KALSHI_BASE_URL (defaults demo).
Read-only market data is enough for paper trading — no orders are ever placed.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from hedge.config import build_client
from hedge.kalshi import KalshiClient
from hedge.strategies.base import MarketView
from hedge.strategies.weather_blend import WeatherBlendStrategy
from hedge.strategies.weather_climatology import WeatherClimatologyStrategy
from hedge.strategies.weather_ensemble import WeatherEnsembleStrategy
from hedge.strategies.weather_nowcast import WeatherNowcastStrategy
from hedge.tournament import paper
from hedge.weather.calibration import CalibrationTable
from hedge.weather.markets import discover_temp_markets, parse_temp_market
from hedge.weather.sources import LiveForecastSource
from hedge.weather.stations import STATIONS


def _client() -> KalshiClient:
    try:
        client, env, base = build_client()
    except RuntimeError as e:
        sys.exit(f"{e}\n(configure config.yaml or KALSHI_* env vars; see scripts/test_auth.py)")
    print(f"[kalshi] env={env} base={base}")
    return client


def _build_strategies(calib: CalibrationTable):
    src = LiveForecastSource()
    return [
        WeatherEnsembleStrategy(src, calib),
        WeatherNowcastStrategy(src, calib),
        WeatherBlendStrategy(src, calib),
        WeatherClimatologyStrategy(),
    ]


def _open_market_views(client) -> list[MarketView]:
    views: list[MarketView] = []
    for series in STATIONS:
        for raw in discover_temp_markets(client, series, status="open"):
            views.append(MarketView(raw.get("ticker", ""), raw))
    return views


def cmd_snapshot() -> None:
    client = _client()
    strategies = _build_strategies(CalibrationTable())
    views = _open_market_views(client)
    rows = paper.snapshot(strategies, views)
    print(f"logged {len(rows)} signal rows across {len(views)} open markets "
          f"at {datetime.now(ZoneInfo('UTC')):%Y-%m-%d %H:%M}Z")


def cmd_score(days: int) -> None:
    from pathlib import Path

    paths = sorted(paper.PAPER_DIR.glob("signals_*.jsonl"))[-days:]
    if not paths:
        sys.exit("no logged signals yet — run `snapshot` first.")
    rows = paper.load_rows(paths)

    # Pull settlement results for every logged ticker.
    client = _client()
    outcomes: dict[str, bool] = {}
    for ticker in rows["ticker"].unique():
        try:
            m = client.get_market(ticker).get("market", {})
        except Exception:  # noqa: BLE001
            continue
        result = str(m.get("result", "")).lower()
        if result in ("yes", "no"):
            outcomes[ticker] = result == "yes"

    board = paper.score(rows, outcomes)
    if board.empty:
        print(f"{len(rows)} signals logged; {len(outcomes)} markets settled so far. "
              "No scored trades yet — wait for more settlements.")
        return
    print(board.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["snapshot", "score"])
    ap.add_argument("--days", type=int, default=14, help="score: how many recent log days to include")
    args = ap.parse_args()
    if args.mode == "snapshot":
        cmd_snapshot()
    else:
        cmd_score(args.days)


if __name__ == "__main__":
    main()
