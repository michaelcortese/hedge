#!/usr/bin/env python
"""One-shot multi-series Kalshi esports data pull.

Fetches, for 12 esports series (LoL/CS2/Valorant/Dota2/CoD/R6/OW derivatives):
  1. settled market lists      -> data/odds/multi/{series}_markets.parquet
  2. t5/t60/t120 pre-match price snapshots via ONE hourly-candles call per
     market                    -> data/odds/multi/{series}_prices.parquet
  3. trades + minute candles for KXLOLMAP / KXLOLTOTALMAPS only
                               -> {series}_trades.parquet, {series}_candles_1m.parquet

Reuses lolpred.data.kalshi_market / kalshi_micro; everything new (generalized
title parsing, pacing session, wall-clock budget) lives here, not the module.

Title formats observed live 2026-07-15:
  MAP series:   "Will {X} win map {N} in the {A} vs. {B} match?"
  GAME series:  "Will {X} win the {A} vs. {B} {Game} match?"
  TOTALMAPS:    "Will over {T} maps be played in the {A} vs. {B} {Game} match?"

Run:  .venv/bin/python scripts/fetch_kalshi_multi.py --budget-min 135
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import requests

from lolpred.data import kalshi_micro
from lolpred.data.kalshi_market import (
    _snapshot,
    fetch_market_candles,
    fetch_settled_lol_markets,
    normalize_team,
    parse_event_start,
)

logger = logging.getLogger("fetch_multi")

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "odds" / "multi"

# Phase order per the research plan (coordinator revision 3, 2026-07-15):
# LoL derivative prices, then LoL microstructure (trades + minute candles —
# a frozen rule needs KXLOLMAP micro urgently), then micro for the sibling
# GAME series paired with their MAP series (game-end clocks for a
# confirmatory test): CS2 pair first, then Valorant, then Dota 2. The
# remaining price-snapshot series (KXCS2TOTALMAPS, KXCODMAP, KXR6GAME,
# KXOWGAME, and all non-LoL prices) are DROPPED — not needed.
PHASE_PRICES_LOL = ["KXLOLMAP", "KXLOLTOTALMAPS"]
PHASE_MICRO_LOL = ["KXLOLMAP", "KXLOLTOTALMAPS"]
PHASE_MICRO_GAMES = [
    "KXCS2GAME",
    "KXCS2MAP",
    "KXVALORANTGAME",
    "KXVALORANTMAP",
    "KXDOTA2GAME",
    "KXDOTA2MAP",
]
PHASE_PRICES_REST: list[str] = []

SNAPSHOT_MINUTES = (5, 60, 120)

# Hourly-candle request cap is 5000 candles -> clamp window to < 5000 h.
MAX_HOURLY_WINDOW_H = 4900

# Trailing game names to strip off team B (title = "... vs. {B} {Game} match?").
# Longest-first so "overwatch 2" wins over "overwatch", etc.
_GAME_SUFFIXES = [
    "league of legends",
    "counter-strike 2",
    "rainbow six siege",
    "call of duty",
    "overwatch 2",
    "rainbow six",
    "overwatch",
    "valorant",
    "dota 2",
    "cs2",
    "cod",
    "r6",
]

_RE_TOTAL = re.compile(
    r"^will\s+(over|under|more\s+than|fewer\s+than|at\s+least)\s+([\d.]+)\s+maps?"
    r"\s+be\s+played\s+in\s+the\s+(.+?)\s+vs\.?\s+(.+)\s+match\??\s*$",
    re.IGNORECASE,
)
_RE_MAP = re.compile(
    r"^will\s+(.+?)\s+win\s+map\s+(\d+)\s+in\s+the\s+(.+?)\s+vs\.?\s+(.+)\s+match\??\s*$",
    re.IGNORECASE,
)
_RE_GAME = re.compile(
    r"^will\s+(.+?)\s+win\s+the\s+(.+?)\s+vs\.?\s+(.+)\s+match\??\s*$",
    re.IGNORECASE,
)


# ------------------------------------------------------------------------ pacing


class PacedSession(requests.Session):
    """requests.Session that enforces a minimum gap between GETs.

    All lolpred fetchers take a ``session`` and route every HTTP call through
    ``session.get``, so this gives one global >=pace_s pacing across market
    lists, candles, and trades without touching the module code. Another job
    may be hitting the same shared Kalshi rate limit concurrently.
    """

    def __init__(self, pace_s: float = 0.28):
        super().__init__()
        self.pace_s = float(pace_s)
        self._next_ok = 0.0

    def get(self, *args, **kwargs):  # noqa: A003 - requests API
        wait = self._next_ok - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        try:
            return super().get(*args, **kwargs)
        finally:
            self._next_ok = time.monotonic() + self.pace_s


# ----------------------------------------------------------------------- parsing


def _clean(name: str) -> str:
    return name.strip().rstrip(".").strip()


def strip_game_suffix(team: str) -> str:
    """Drop a trailing game name ('Golden Lions League of Legends' -> 'Golden Lions')."""
    low = team.lower()
    for suffix in _GAME_SUFFIXES:
        if low.endswith(" " + suffix):
            candidate = team[: -(len(suffix) + 1)].rstrip()
            if candidate:
                return candidate
    return team


def parse_title(title: str | None, yes_sub_title: str | None) -> dict | None:
    """Generalized esports title parse -> dict, or None if unparseable.

    Returns keys: yes_team, opp_team (None for TOTALMAPS), team_a, team_b,
    map_number (MAP series only), threshold (TOTALMAPS only).
    """
    if not isinstance(title, str) or not title.strip():
        return None
    t = title.strip()

    m = _RE_TOTAL.match(t)
    if m:
        _comp, thr, team_a, team_b = m.groups()
        try:
            threshold = float(thr)
        except ValueError:
            return None
        return {
            "yes_team": None,
            "opp_team": None,
            "team_a": _clean(team_a),
            "team_b": strip_game_suffix(_clean(team_b)),
            "map_number": None,
            "threshold": threshold,
        }

    map_number: int | None = None
    m = _RE_MAP.match(t)
    if m:
        yes_raw, map_no, team_a, team_b = m.groups()
        map_number = int(map_no)
    else:
        m = _RE_GAME.match(t)
        if not m:
            return None
        yes_raw, team_a, team_b = m.groups()

    team_a = _clean(team_a)
    team_b = strip_game_suffix(_clean(team_b))

    candidates = [_clean(yes_raw)]
    if isinstance(yes_sub_title, str) and yes_sub_title.strip():
        candidates.insert(0, yes_sub_title.strip())

    for yes in candidates:
        for fold in (str.lower, normalize_team):
            if fold(yes) == fold(team_a):
                return {
                    "yes_team": team_a, "opp_team": team_b,
                    "team_a": team_a, "team_b": team_b,
                    "map_number": map_number, "threshold": None,
                }
            if fold(yes) == fold(team_b):
                return {
                    "yes_team": team_b, "opp_team": team_a,
                    "team_a": team_a, "team_b": team_b,
                    "map_number": map_number, "threshold": None,
                }
    return None


# ------------------------------------------------------------------ price builder

PRICE_COLS = (
    ["ticker", "event_ticker", "yes_team", "opp_team", "match_start", "result"]
    + [f"t{m}_{f}" for m in SNAPSHOT_MINUTES for f in ("bid", "ask", "mid", "ts")]
    + ["oi", "volume", "n_candles", "team_a", "team_b", "map_number", "threshold"]
)


def build_prices(
    series: str,
    markets_df: pd.DataFrame,
    cache_path: Path,
    session: requests.Session,
    deadline: float,
    flush_every: int = 200,
    progress_every: int = 100,
) -> tuple[int, Counter, list[str], bool]:
    """Adapted from kalshi_market.build_market_prices with the generalized
    title parse, a t120 snapshot, per-200-market flushes, and a deadline.

    Returns (n_priced_total, skip_counter, example_bad_titles, completed).
    """
    cached: pd.DataFrame | None = None
    done: set[str] = set()
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        done = set(cached["ticker"])
        logger.info("[%s] price cache: %d rows already built", series, len(cached))

    todo = markets_df[~markets_df["ticker"].isin(done)]
    skips: Counter = Counter()
    bad_titles: list[str] = []
    out_rows: list[dict] = []
    completed = True

    def _flush() -> int:
        frames = [pd.DataFrame(out_rows, columns=PRICE_COLS)]
        if cached is not None and len(cached):
            frames.insert(0, cached.reindex(columns=PRICE_COLS))
        df = pd.concat(frames, ignore_index=True)
        df = df.drop_duplicates(subset="ticker", keep="last").reset_index(drop=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        return len(df)

    for i, mkt in enumerate(todo.itertuples(index=False), 1):
        if time.monotonic() > deadline:
            logger.warning("[%s] deadline hit at %d/%d — flushing partial", series, i, len(todo))
            completed = False
            break
        if progress_every and i % progress_every == 0:
            logger.info("[%s] prices: %d/%d (built %d, skipped %d)",
                        series, i, len(todo), len(out_rows), sum(skips.values()))
        if flush_every and out_rows and i % flush_every == 0:
            n = _flush()
            logger.info("[%s] flushed -> %s (%d rows)", series, cache_path, n)

        result_raw = str(getattr(mkt, "result", "") or "").lower()
        if result_raw not in ("yes", "no"):
            skips["unknown_result"] += 1
            continue

        parsed = parse_title(getattr(mkt, "title", None), getattr(mkt, "yes_sub_title", None))
        if parsed is None:
            skips["unparseable_title"] += 1
            if len(bad_titles) < 5:
                bad_titles.append(str(getattr(mkt, "title", None)))
            continue

        match_start = parse_event_start(mkt.event_ticker) or parse_event_start(mkt.ticker)
        if match_start is None:
            for fallback in (mkt.expected_expiration_time, mkt.close_time):
                if pd.notna(fallback):
                    match_start = fallback
                    break
        if match_start is None or pd.isna(match_start):
            skips["no_match_start"] += 1
            continue

        open_time = mkt.open_time if pd.notna(mkt.open_time) else match_start - pd.Timedelta(days=7)
        close_time = mkt.close_time if pd.notna(mkt.close_time) else match_start + pd.Timedelta(hours=6)
        # Hourly-candle requests are capped at 5000 candles per call.
        open_time = max(open_time, close_time - pd.Timedelta(hours=MAX_HOURLY_WINDOW_H))
        series_ticker = str(mkt.ticker).split("-")[0]

        try:
            candles = fetch_market_candles(
                mkt.ticker, series_ticker,
                int(open_time.timestamp()), int(close_time.timestamp()),
                period_interval=60, session=session,
            )
        except Exception as exc:  # retries already exhausted inside _get_json
            logger.warning("[%s] candle fetch failed for %s: %s", series, mkt.ticker, exc)
            skips["candle_fetch_error"] += 1
            continue

        if candles is None or candles.empty:
            skips["no_candles"] += 1
            continue

        start_unix = int(match_start.timestamp())
        row: dict = {
            "ticker": mkt.ticker,
            "event_ticker": mkt.event_ticker,
            "yes_team": parsed["yes_team"],
            "opp_team": parsed["opp_team"],
            "match_start": match_start,
            "result": 1 if result_raw == "yes" else 0,
            "team_a": parsed["team_a"],
            "team_b": parsed["team_b"],
            "map_number": parsed["map_number"],
            "threshold": parsed["threshold"],
        }
        for m in SNAPSHOT_MINUTES:
            snap = _snapshot(candles, start_unix - m * 60)
            row[f"t{m}_bid"] = snap["bid"]
            row[f"t{m}_ask"] = snap["ask"]
            row[f"t{m}_mid"] = snap["mid"]
            row[f"t{m}_ts"] = snap["ts"]
        row["oi"] = candles["open_interest"].iloc[-1]
        row["volume"] = candles["volume"].sum()
        row["n_candles"] = len(candles)
        out_rows.append(row)

    n_total = _flush() if (out_rows or cached is not None) else 0
    if skips:
        logger.info("[%s] price skips: %s", series, dict(skips))
    return n_total, skips, bad_titles, completed


# ------------------------------------------------------------------- micro phase


def build_micro_plan(markets_df: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker minute-candle windows: match_start - 2h .. close_time."""
    rows: list[dict] = []
    dropped = 0
    for mkt in markets_df.drop_duplicates(subset="ticker").itertuples(index=False):
        match_start = parse_event_start(mkt.event_ticker) or parse_event_start(mkt.ticker)
        if match_start is None or pd.isna(match_start):
            for fallback in (mkt.expected_expiration_time, mkt.close_time):
                if pd.notna(fallback):
                    match_start = fallback
                    break
        if match_start is None or pd.isna(match_start):
            dropped += 1
            continue
        start = pd.Timestamp(match_start) - pd.Timedelta(hours=2)
        if pd.notna(mkt.close_time):
            end = pd.Timestamp(mkt.close_time)
        else:
            end = pd.Timestamp(match_start) + pd.Timedelta(hours=6)
        # Market closed before its window opens (rescheduled/cancelled match).
        if end <= start and pd.notna(mkt.open_time):
            start = pd.Timestamp(mkt.open_time)
        start_ts, end_ts = int(start.timestamp()), int(end.timestamp())
        if end_ts <= start_ts:
            dropped += 1
            continue
        rows.append({"ticker": mkt.ticker, "start_ts": start_ts, "end_ts": end_ts})
    if dropped:
        logger.warning("build_micro_plan: dropped %d tickers with no usable window", dropped)
    return pd.DataFrame(rows, columns=["ticker", "start_ts", "end_ts"])


def run_micro(
    series: str,
    markets_df: pd.DataFrame,
    session: requests.Session,
    deadline: float,
    slice_size: int = 100,
) -> tuple[dict, bool]:
    """Trades + minute candles for one series, in deadline-checked slices."""
    trades_path = OUT_DIR / f"{series}_trades.parquet"
    candles_path = OUT_DIR / f"{series}_candles_1m.parquet"
    plan = build_micro_plan(markets_df)
    logger.info("[%s] micro plan: %d tickers", series, len(plan))

    totals: Counter = Counter()
    completed = True
    for lo in range(0, len(plan), slice_size):
        if time.monotonic() > deadline:
            logger.warning("[%s] deadline hit before micro slice %d — stopping", series, lo)
            completed = False
            break
        summary = kalshi_micro.fetch_micro(
            plan.iloc[lo:lo + slice_size],
            trades_path, candles_path,
            session=session, pace_s=0.0,  # PacedSession enforces global pacing
            flush_every=50, progress_every=50,
        )
        totals.update({k: v for k, v in summary.items() if k != "tickers_skipped_cached"})
        logger.info("[%s] micro slice %d..%d: %s", series, lo, lo + slice_size, summary)
    return dict(totals), completed


# -------------------------------------------------------------------------- main


def ensure_markets(series: str, session: requests.Session) -> pd.DataFrame:
    """Fetch (incrementally, cached) and persist the settled-market list."""
    markets_path = OUT_DIR / f"{series}_markets.parquet"
    markets_df = fetch_settled_lol_markets(
        series_ticker=series, session=session, cache_path=markets_path,
    )
    markets_path.parent.mkdir(parents=True, exist_ok=True)
    markets_df.to_parquet(markets_path, index=False)
    logger.info("[%s] %d settled markets -> %s", series, len(markets_df), markets_path)
    return markets_df


def do_prices(series: str, session: requests.Session, deadline: float, report: dict) -> None:
    if time.monotonic() > deadline:
        logger.warning("[%s] prices skipped entirely — budget exhausted", series)
        report[series] = {"status": "unfetched"}
        return
    try:
        markets_df = ensure_markets(series, session)
    except Exception as exc:
        logger.error("[%s] market list fetch failed: %s", series, exc)
        report[series] = {"status": "markets_failed", "error": str(exc)}
        return
    n_priced, skips, bad_titles, done = build_prices(
        series, markets_df, OUT_DIR / f"{series}_prices.parquet", session, deadline,
    )
    report[series] = {
        "status": "complete" if done else "partial",
        "markets": len(markets_df),
        "priced": n_priced,
        "skips": dict(skips),
        "example_bad_titles": bad_titles,
    }
    logger.info("[%s] done: %s", series, report[series])


def do_micro(series: str, session: requests.Session, deadline: float, report: dict) -> None:
    key = f"{series}_micro"
    if time.monotonic() > deadline:
        logger.warning("[%s] micro skipped — budget exhausted", series)
        report[key] = {"status": "unfetched"}
        return
    try:
        markets_df = ensure_markets(series, session)
    except Exception as exc:
        logger.error("[%s] market list fetch failed: %s", series, exc)
        report[key] = {"status": "markets_failed", "error": str(exc)}
        return
    totals, done = run_micro(series, markets_df, session, deadline)
    report[key] = {"status": "complete" if done else "partial", **totals}
    logger.info("[%s] micro done: %s", series, report[key])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget-min", type=float, default=135.0,
                    help="wall-clock budget in minutes; flush and stop when hit")
    ap.add_argument("--pace", type=float, default=0.28,
                    help="minimum seconds between HTTP GETs (shared rate limit)")
    ap.add_argument("--skip-micro", action="store_true",
                    help="skip the trades/minute-candles phases")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    session = PacedSession(pace_s=args.pace)
    deadline = time.monotonic() + args.budget_min * 60.0

    report: dict = {}

    for series in PHASE_PRICES_LOL:
        do_prices(series, session, deadline, report)
    if not args.skip_micro:
        for series in PHASE_MICRO_LOL:
            do_micro(series, session, deadline, report)
        for series in PHASE_MICRO_GAMES:
            do_micro(series, session, deadline, report)
    for series in PHASE_PRICES_REST:
        do_prices(series, session, deadline, report)

    logger.info("FINAL REPORT:\n%s", json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
