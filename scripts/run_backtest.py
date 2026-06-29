#!/usr/bin/env python
"""Run the historical tournament: fit calibration, backtest strategies, score.

Usage:
    .venv/bin/python scripts/run_backtest.py [--days 60] [--lead 1] [--cities NY,CHI]

Pipeline:
  1. Split history into a TRAIN window (fit forecast-error calibration per city)
     and a disjoint TEST window (grade strategies) — no calibration leakage.
  2. Prefetch archived model forecasts once per city (cheap, cached).
  3. Run the ensemble + climatology strategies over every (city, test-day) via the
     backtest harness, scoring against ERA5 realized highs.
  4. Print + save the leaderboard (Brier / log-loss / CRPS / calibration / skill).

All fetches are cached under data/cache/, so re-runs are instant and reproducible.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

from datetime import datetime
from zoneinfo import ZoneInfo

from hedge.strategies.weather_blend import WeatherBlendStrategy
from hedge.strategies.weather_climatology import WeatherClimatologyStrategy
from hedge.strategies.weather_ensemble import WeatherEnsembleStrategy
from hedge.strategies.weather_nowcast import WeatherNowcastStrategy
from hedge.tournament.backtest import run_backtest
from hedge.tournament.report import leaderboard, render_markdown
from hedge.weather.archive import archive_hourly_range, historical_model_highs_range
from hedge.weather.calibration import fit_calibration
from hedge.weather.sources import ArchiveForecastSource, HistoricalIntradaySource
from hedge.weather.stations import STATIONS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=60, help="length of the test window")
    ap.add_argument("--lead", type=int, default=1, help="forecast lead time in days")
    ap.add_argument("--cities", type=str, default="", help="comma list of series suffixes, e.g. NY,CHI")
    ap.add_argument("--lag", type=int, default=7, help="skip the most recent N days (ERA5 latency)")
    ap.add_argument("--nowcast-hour", type=int, default=15, help="local hour the nowcast/blend evaluate at")
    args = ap.parse_args()

    stations = list(STATIONS.values())
    if args.cities:
        wanted = {f"KXHIGH{c.strip().upper()}" for c in args.cities.split(",")}
        stations = [s for s in stations if s.series in wanted]

    # Windows: [test_start, test_end] for grading; the equal window before it trains.
    test_end = date.today() - timedelta(days=args.lag)
    test_start = test_end - timedelta(days=args.days)
    train_end = test_start - timedelta(days=1)
    train_start = train_end - timedelta(days=args.days)

    print(f"train: {train_start} .. {train_end}   test: {test_start} .. {test_end}")
    print(f"cities: {[s.city for s in stations]}  lead={args.lead}d")

    print("fitting calibration on train window ...")
    calib = fit_calibration(stations, train_start, train_end)
    for st in stations:
        print(f"  {st.city:14} sigma(lead{args.lead})="
              f"{calib.sigma_for(st.series, args.lead):.2f} F")

    # Prefetch archived forecasts + hourly obs once per city for the test window.
    preloaded = {
        st.series: historical_model_highs_range(st, test_start, test_end)
        for st in stations
    }
    hourly = {
        st.series: archive_hourly_range(st, test_start, test_end)
        for st in stations
    }

    def strategy_factory(station, target_day, as_of):
        fc = preloaded[station.series]
        archive_src = ArchiveForecastSource(preloaded=fc)
        intraday_src = HistoricalIntradaySource(fc, hourly[station.series], args.nowcast_hour)
        now = datetime(
            target_day.year, target_day.month, target_day.day,
            args.nowcast_hour, tzinfo=ZoneInfo(station.tz),
        )
        return [
            WeatherEnsembleStrategy(source=archive_src, calibration=calib, as_of=as_of),
            WeatherNowcastStrategy(source=intraday_src, calibration=calib, now=now),
            WeatherBlendStrategy(source=intraday_src, calibration=calib, as_of=as_of, now=now),
            WeatherClimatologyStrategy(years=20, window_days=7),
        ]

    print("grading strategies over test window ...")
    records = run_backtest(
        stations, test_start, test_end, strategy_factory, lead_days=args.lead,
    )
    if not records:
        print("no graded days — check archive availability for this window.")
        return

    board = leaderboard(records)
    print("\n" + board.to_string(index=False))

    out_dir = Path("data/runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"backtest_{test_start}_{test_end}.md"
    out_path.write_text(render_markdown(board, title=f"Backtest {test_start}..{test_end}"))
    print(f"\nleaderboard written to {out_path}")


if __name__ == "__main__":
    main()
