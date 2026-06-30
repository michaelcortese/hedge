#!/usr/bin/env python
"""Forward paper tournament against Kalshi (demo by default).

Three modes:

  # log one snapshot cycle: every strategy's signal + live prices for open markets
  .venv/bin/python scripts/run_paper.py snapshot

  # snapshot continuously through the trading day (the steady-state paper bot)
  .venv/bin/python scripts/run_paper.py loop --interval 900

  # after markets settle, score the logged signals into a P&L leaderboard
  .venv/bin/python scripts/run_paper.py score

``loop`` is the continuous paper trader: it builds the client and strategies once
and re-snapshots every ``--interval`` seconds, isolating per-cycle errors so a
single transient API hiccup never kills the run, until ``--until`` (HH:MM local),
``--cycles``, or Ctrl-C. The nowcast/blend strategies only act in the afternoon,
so accumulating many cycles per day is what actually builds the track record. If
you'd rather drive cadence externally, run ``snapshot`` from cron or the /loop
skill instead. ``score`` reads back the settled results and ranks strategies by
realized per-contract P&L and fractional-Kelly bankroll growth.

Credentials (env vars, same as scripts/test_auth.py):
    KALSHI_KEY_ID, KALSHI_PRIVATE_KEY_PATH, optional KALSHI_BASE_URL (defaults demo).
Read-only market data is enough for paper trading — no orders are ever placed.
"""

from __future__ import annotations

import argparse
import signal as _signal
import sys
import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from hedge.config import build_client
from hedge.kalshi import KalshiClient
from hedge.strategies.base import MarketView
from hedge.strategies.weather_blend import WeatherBlendStrategy
from hedge.strategies.weather_climatology import WeatherClimatologyStrategy
from hedge.strategies.weather_ensemble import WeatherEnsembleStrategy
from hedge.strategies.weather_nowcast import WeatherNowcastStrategy
from hedge.tournament import paper
from hedge.weather.calibration import CalibrationTable, fit_calibration
from hedge.weather.markets import discover_temp_markets, parse_temp_market
from hedge.weather.sources import LiveForecastSource
from hedge.weather.stations import STATIONS


def _fit_calibration(days: int = 45, lag: int = 7) -> CalibrationTable:
    """Fit forecast-error spread+bias per city over a recent archived window.

    Mirrors scripts/run_backtest.py so the paper bot trades the SAME calibrated
    sigma/bias the backtest graded — not an empty prior. Cached fetches, no creds.
    Falls back to an empty table (priors) if the archive is unreachable.
    """
    end = date.today() - timedelta(days=lag)
    start = end - timedelta(days=days)
    try:
        calib = fit_calibration(list(STATIONS.values()), start, end)
        fitted = sorted({s for (s, _lead) in calib.sigma})
        print(f"[calib] fit {start}..{end}; calibrated series: {fitted or 'none (using priors)'}")
        return calib
    except Exception as e:  # noqa: BLE001 — never let calibration block paper logging
        print(f"[calib] fit failed ({type(e).__name__}: {e}); using priors.")
        return CalibrationTable()


def _client(prod_readonly: bool = False) -> KalshiClient:
    if prod_readonly:
        # Keyless prod market-data feed: the forward "edge-vs-real-price" track runs
        # against prod liquidity without credentials and can never place an order.
        from hedge.kalshi.client import PROD_BASE
        print(f"[kalshi] env=prod-readonly base={PROD_BASE} (public data, no orders)")
        return KalshiClient.read_only(PROD_BASE)
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


def _run_cycle(client, strategies) -> int:
    """One snapshot cycle: fetch open markets, log signals+prices. Returns rows."""
    views = _open_market_views(client)
    rows = paper.snapshot(strategies, views)
    print(f"logged {len(rows)} signal rows across {len(views)} open markets "
          f"at {datetime.now(ZoneInfo('UTC')):%Y-%m-%d %H:%M}Z", flush=True)
    return len(rows)


def cmd_snapshot(prod: bool = False) -> None:
    client = _client(prod)
    strategies = _build_strategies(_fit_calibration())
    _run_cycle(client, strategies)


def cmd_loop(interval: float, until: str | None, cycles: int | None, prod: bool = False) -> None:
    """Snapshot repeatedly until --until, --cycles, or Ctrl-C.

    Builds the client and strategies ONCE and reuses them; each cycle's errors are
    caught so a transient API failure logs a warning and the loop continues.
    """
    client = _client(prod)
    strategies = _build_strategies(_fit_calibration())

    stop_at = None
    if until is not None:
        try:
            hh, mm = (int(x) for x in until.split(":"))
        except ValueError:
            sys.exit(f"--until must be HH:MM (24h local time), got {until!r}")
        now = datetime.now().astimezone()
        stop_at = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if stop_at <= now:  # already past today -> nothing to do, don't loop overnight
            sys.exit(f"--until {until} is already in the past (local now {now:%H:%M}); nothing to do.")

    # Graceful shutdown: flip a flag on SIGINT/SIGTERM so we finish the current
    # sleep rather than dying mid-snapshot.
    stopping = {"flag": False}

    def _handle(signum, _frame):
        stopping["flag"] = True
        print(f"\n[loop] signal {signum} received — stopping after this cycle.", flush=True)

    _signal.signal(_signal.SIGINT, _handle)
    _signal.signal(_signal.SIGTERM, _handle)

    horizon = f"{stop_at:%Y-%m-%d %H:%M %Z}" if stop_at else "Ctrl-C"
    print(f"[loop] interval={interval:g}s until={horizon}", flush=True)

    n = 0
    while not stopping["flag"]:
        try:
            _run_cycle(client, strategies)
        except Exception as e:  # noqa: BLE001 — one bad cycle must not kill the loop
            print(f"[loop] cycle error ({type(e).__name__}): {e}", flush=True)
        n += 1
        if cycles is not None and n >= cycles:
            print(f"[loop] reached --cycles={cycles}; done.", flush=True)
            break
        if stop_at is not None and datetime.now().astimezone() >= stop_at:
            print(f"[loop] reached --until; done.", flush=True)
            break
        # Sleep in short slices so a stop signal is honored promptly.
        slept = 0.0
        while slept < interval and not stopping["flag"]:
            if stop_at is not None and datetime.now().astimezone() >= stop_at:
                break
            time.sleep(min(1.0, interval - slept))
            slept += 1.0
    print(f"[loop] stopped after {n} cycle(s).", flush=True)


def cmd_edge(prod: bool = False) -> None:
    """Print the live edge table: calibrated prob vs real quote, and decide()'s call.

    This is the "am I beating the market right now" view — it runs the same
    edge/Kelly/fee math the scorer uses (``paper.decide``), but against the current
    order book instead of settled outcomes, so it needs no waiting. Markets with no
    two-sided quote (common on demo) surface as wide/abstain — that's honest: there
    is no market to beat.
    """
    client = _client(prod)
    strategies = _build_strategies(_fit_calibration())
    risk = paper.RiskParams()
    views = _open_market_views(client)
    rows = []
    for mv in views:
        yb, ya = mv.yes_bid, mv.yes_ask
        if yb is None or ya is None:
            continue
        for strat in strategies:
            sig = strat.evaluate(mv)
            if sig is None:
                continue
            dec = paper.decide(sig.prob, sig.sigma, yb, ya, risk)
            rows.append({
                "ticker": sig.ticker, "strategy": sig.strategy,
                "prob": round(sig.prob, 3), "yes_bid": yb, "yes_ask": ya,
                "side": dec.side, "edge_$": round(dec.edge, 3),
                "kelly_frac": round(dec.kelly_frac, 4),
            })
    if not rows:
        print("no open markets with a two-sided quote — nothing to price.")
        return
    import pandas as pd
    df = pd.DataFrame(rows)
    actionable = df[df["side"] != "none"]
    print(df.sort_values(["ticker", "strategy"]).to_string(index=False))
    print(f"\n{len(df)} (strategy,market) prices; {len(actionable)} clear the edge gate "
          f"(tau={risk.tau_min_cents}c, k_sigma={risk.k_sigma}).")
    if not actionable.empty:
        print("\nActionable edges:")
        print(actionable.sort_values("edge_$", ascending=False).to_string(index=False))


def cmd_score(days: int, prod: bool = False) -> None:
    from pathlib import Path

    paths = sorted(paper.PAPER_DIR.glob("signals_*.jsonl"))[-days:]
    if not paths:
        sys.exit("no logged signals yet — run `snapshot` first.")
    rows = paper.load_rows(paths)

    # Pull settlement results for every logged ticker.
    client = _client(prod)
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


def cmd_skill(days: int, prod: bool = False) -> None:
    """Score model-vs-market skill (Brier) by strategy and hour — the #3 go/no-go view.

    The live skill gate ramps λ on exactly this signal (model Brier beating the market
    mid). This is its offline counterpart: prove the nowcast beats the mid in the
    afternoon window BEFORE arming real size.
    """
    paths = sorted(paper.PAPER_DIR.glob("signals_*.jsonl"))[-days:]
    if not paths:
        sys.exit("no logged signals yet — run `snapshot`/`loop` first.")
    rows = paper.load_rows(paths)
    client = _client(prod)
    outcomes: dict[str, bool] = {}
    for ticker in rows["ticker"].unique():
        try:
            m = client.get_market(ticker).get("market", {})
        except Exception:  # noqa: BLE001
            continue
        result = str(m.get("result", "")).lower()
        if result in ("yes", "no"):
            outcomes[ticker] = result == "yes"

    overall = paper.score_skill_vs_market(rows, outcomes, by_hour=False)
    if overall.empty:
        print(f"{len(rows)} signals logged; {len(outcomes)} settled. "
              "No scored signals yet — wait for more settlements.")
        return
    print("SKILL vs MARKET MID — overall (skill>0 => model beats the market):")
    print(overall.to_string(index=False))
    print("\nby strategy × UTC hour (confirm the nowcast edge concentrates afternoon):")
    print(paper.score_skill_vs_market(rows, outcomes, by_hour=True).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["snapshot", "loop", "score", "edge", "skill"])
    ap.add_argument("--prod", action="store_true",
                    help="read live PROD market data (keyless, read-only, never places orders)")
    ap.add_argument("--days", type=int, default=14, help="score: how many recent log days to include")
    ap.add_argument("--interval", type=float, default=900.0,
                    help="loop: seconds between snapshots (default 900 = 15 min)")
    ap.add_argument("--until", type=str, default=None,
                    help="loop: stop at this local time, HH:MM 24h (e.g. 19:00)")
    ap.add_argument("--cycles", type=int, default=None,
                    help="loop: stop after this many snapshot cycles")
    args = ap.parse_args()
    if args.mode == "snapshot":
        cmd_snapshot(args.prod)
    elif args.mode == "edge":
        cmd_edge(args.prod)
    elif args.mode == "skill":
        cmd_skill(args.days, args.prod)
    elif args.mode == "loop":
        if args.interval <= 0:
            sys.exit("--interval must be > 0")
        cmd_loop(args.interval, args.until, args.cycles, args.prod)
    else:
        cmd_score(args.days, args.prod)


if __name__ == "__main__":
    main()
