"""Kalshi market-data ingestion for LoL match markets (series KXLOLGAME).

Read-only public market data — no auth, no signing. Plain ``requests`` GETs on
the prod trade API (same host the hedge repo's ``KalshiClient.read_only`` uses,
reimplemented here so lolpred stays self-contained).

Endpoints used (verified live, 2026-07):
  GET /markets?series_ticker=KXLOLGAME&status=settled&limit=200&cursor=...
      -> paginated list; last_price/volume/liquidity often null here, real
         price data comes from candlesticks.
  GET /series/{series}/markets/{ticker}/candlesticks
      ?start_ts=&end_ts=&period_interval=60   (unix seconds; 60=hourly, 1=minute)
      -> {"candlesticks": [{end_period_ts, price:{..._dollars}, yes_bid:{...},
          yes_ask:{...}, volume_fp:"0.00", open_interest_fp:"38678.18"}]}
         Dollar fields are strings.
"""

from __future__ import annotations

import logging
import re
import time
from collections import Counter
from pathlib import Path
from typing import Callable

import pandas as pd
import requests

BASE = "https://api.elections.kalshi.com/trade-api/v2"

logger = logging.getLogger(__name__)

# Backoff sleeps for 429/5xx (Kalshi sends no Retry-After): 3 retries, then give up.
_RETRY_SLEEPS = (2.0, 4.0, 8.0)

_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

# Only ever strip these trailing words; anything else ('Lions', 'Dragons', ...)
# is part of the name. Conservative on purpose: 'Golden Lions' != 'Lions'.
_TEAM_SUFFIXES = {"esports", "esport", "gaming", "team"}


# --------------------------------------------------------------------------- HTTP


def _get_json(
    session: requests.Session,
    url: str,
    params: dict | None = None,
    timeout: float = 15.0,
) -> dict:
    """GET with retry/backoff on 429 and 5xx (sleep 2s/4s/8s, give up after 3)."""
    for sleep_s in (*_RETRY_SLEEPS, None):
        resp = session.get(url, params=params, timeout=timeout)
        if resp.status_code == 429 or resp.status_code >= 500:
            if sleep_s is None:
                resp.raise_for_status()
            logger.warning(
                "HTTP %s from %s — retrying in %.0fs", resp.status_code, url, sleep_s
            )
            time.sleep(sleep_s)
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError("unreachable")  # pragma: no cover


# ------------------------------------------------------------------------- parsing


def parse_event_start(event_ticker_or_ticker: str) -> pd.Timestamp | None:
    """Match start encoded in the event ticker, as a UTC Timestamp.

    "KXLOLGAME-26JUL141600MAZGL" -> 2026-07-14 16:00 UTC
    (2-digit year, 3-letter month, 2-digit day, HHMM). Also accepts the full
    market ticker ("...-MAZ"). Returns None if unparseable.
    """
    if not isinstance(event_ticker_or_ticker, str):
        return None
    parts = event_ticker_or_ticker.split("-")
    if len(parts) < 2:
        return None
    m = re.match(r"^(\d{2})([A-Z]{3})(\d{2})(\d{2})(\d{2})", parts[1].upper())
    if not m:
        return None
    yy, mon, dd, hh, mm = m.groups()
    month = _MONTHS.get(mon)
    if month is None:
        return None
    try:
        return pd.Timestamp(
            year=2000 + int(yy), month=month, day=int(dd),
            hour=int(hh), minute=int(mm), tz="UTC",
        )
    except ValueError:
        return None


def parse_teams(title: str, yes_sub_title: str | None) -> tuple[str, str] | None:
    """(yes_team, other_team) from the market title.

    Title pattern: "Will {A} win the {A} vs. {B} League of Legends match?"
    ``yes_sub_title`` (when present) names the YES team; otherwise the leading
    "Will {X} win" clause does. Handles ' vs. ' / ' vs ' and casing variants.
    Returns None if unparseable.
    """
    if not isinstance(title, str) or not title.strip():
        return None
    m = re.search(
        r"will\s+(.+?)\s+win\s+the\s+(.+?)\s+vs\.?\s+(.+?)\s+league\s+of\s+legends",
        title,
        re.IGNORECASE,
    )
    if not m:
        return None
    yes_raw, team_a, team_b = (g.strip().rstrip(".") for g in m.groups())
    yes = yes_raw
    if isinstance(yes_sub_title, str) and yes_sub_title.strip():
        yes = yes_sub_title.strip()

    for exact in (str.lower, normalize_team):
        if exact(yes) == exact(team_a):
            return (team_a, team_b)
        if exact(yes) == exact(team_b):
            return (team_b, team_a)
    return None


def normalize_team(name: str) -> str:
    """Lowercase, strip punctuation and trailing 'esports'/'gaming'/'team'.

    Used later to join against Oracle's Elixir team names. Deliberately
    conservative: only known org-suffix words are stripped, and never the
    whole name ('T1 Esports' -> 't1', but 'Golden Lions' -> 'golden lions').
    """
    if not isinstance(name, str):
        return ""
    words = re.sub(r"[^\w\s]", " ", name.lower()).split()
    while len(words) > 1 and words[-1] in _TEAM_SUFFIXES:
        words.pop()
    return " ".join(words)


# ------------------------------------------------------------------------ fetchers


def fetch_settled_lol_markets(
    series_ticker: str = "KXLOLGAME",
    session: requests.Session | None = None,
    base: str = BASE,
    cache_path: str | Path | None = None,
) -> pd.DataFrame:
    """All settled markets for a series, paginated via cursor.

    If ``cache_path`` points at an existing parquet of a previous run, only
    markets with close_time strictly newer than the cached max are fetched
    (via min_close_ts) and merged with the cache (dedup on ticker).
    """
    session = session or requests.Session()

    cached: pd.DataFrame | None = None
    params: dict = {"series_ticker": series_ticker, "status": "settled", "limit": 200}
    if cache_path is not None and Path(cache_path).exists():
        cached = pd.read_parquet(cache_path)
        if len(cached) and cached["close_time"].notna().any():
            max_close = cached["close_time"].max()
            params["min_close_ts"] = int(max_close.timestamp()) + 1
            logger.info(
                "market cache %s: %d rows, fetching close_time > %s",
                cache_path, len(cached), max_close,
            )

    rows: list[dict] = []
    cursor: str | None = None
    while True:
        page_params = dict(params)
        if cursor:
            page_params["cursor"] = cursor
        payload = _get_json(session, f"{base}/markets", params=page_params)
        markets = payload.get("markets") or []
        rows.extend(markets)
        cursor = payload.get("cursor")
        if not cursor or not markets:
            break

    cols = [
        "ticker", "event_ticker", "title", "yes_sub_title",
        "open_time", "close_time", "expected_expiration_time",
        "result", "last_price", "volume", "liquidity",
    ]
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols]
    for c in ("open_time", "close_time", "expected_expiration_time"):
        df[c] = pd.to_datetime(df[c], utc=True, errors="coerce", format="ISO8601")

    if cached is not None:
        df = pd.concat([cached, df], ignore_index=True)
        df = df.drop_duplicates(subset="ticker", keep="last")
    return df.sort_values("close_time").reset_index(drop=True)


def _dollars(block: dict | None, key: str) -> float:
    if not isinstance(block, dict):
        return float("nan")
    v = block.get(key)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def fetch_market_candles(
    ticker: str,
    series_ticker: str,
    start_ts: int,
    end_ts: int,
    period_interval: int = 60,
    session: requests.Session | None = None,
    base: str = BASE,
) -> pd.DataFrame:
    """Candlesticks for one market, flattened to floats (dollars are strings upstream).

    Columns: end_period_ts, ts (UTC), yes_bid_{open,high,low,close},
    yes_ask_{open,high,low,close}, price_open, price_close, volume, open_interest.
    Empty (but correctly-columned) frame if the API returns no candles.
    """
    session = session or requests.Session()
    url = f"{base}/series/{series_ticker}/markets/{ticker}/candlesticks"
    payload = _get_json(
        session, url,
        params={
            "start_ts": int(start_ts),
            "end_ts": int(end_ts),
            "period_interval": int(period_interval),
        },
    )
    rows = []
    for c in payload.get("candlesticks") or []:
        row = {"end_period_ts": int(c["end_period_ts"])}
        for side in ("yes_bid", "yes_ask"):
            for ohlc in ("open", "high", "low", "close"):
                row[f"{side}_{ohlc}"] = _dollars(c.get(side), f"{ohlc}_dollars")
        row["price_open"] = _dollars(c.get("price"), "open_dollars")
        row["price_close"] = _dollars(c.get("price"), "close_dollars")
        row["volume"] = _dollars(c, "volume_fp")
        row["open_interest"] = _dollars(c, "open_interest_fp")
        rows.append(row)

    cols = (
        ["end_period_ts"]
        + [f"{s}_{o}" for s in ("yes_bid", "yes_ask") for o in ("open", "high", "low", "close")]
        + ["price_open", "price_close", "volume", "open_interest"]
    )
    df = pd.DataFrame(rows, columns=cols)
    df = df.sort_values("end_period_ts").reset_index(drop=True)
    df.insert(1, "ts", pd.to_datetime(df["end_period_ts"], unit="s", utc=True))
    return df


# --------------------------------------------------------------------- price table


def _snapshot(candles: pd.DataFrame, cutoff_ts: int) -> dict:
    """Last candle at or before ``cutoff_ts`` -> bid/ask/mid/ts (NaN/NaT if none)."""
    sel = candles[candles["end_period_ts"] <= cutoff_ts]
    if sel.empty:
        return {"bid": float("nan"), "ask": float("nan"), "mid": float("nan"), "ts": pd.NaT}
    row = sel.iloc[-1]
    bid, ask = row["yes_bid_close"], row["yes_ask_close"]
    return {"bid": bid, "ask": ask, "mid": (bid + ask) / 2.0, "ts": row["ts"]}


def build_market_prices(
    markets_df: pd.DataFrame,
    candle_fetcher: Callable[..., pd.DataFrame] = fetch_market_candles,
    snapshot_minutes_before: tuple[int, ...] = (5, 60),
    pace_s: float = 0.15,
    progress_every: int = 100,
    cache_path: str | Path | None = None,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Pre-match price snapshots for each settled market.

    For every market: match_start = parse_event_start(event_ticker) (fallback
    expected_expiration_time, then close_time); fetch hourly candles over
    [open_time, close_time]; for each m in ``snapshot_minutes_before`` take the
    LAST candle with end_period_ts <= match_start - m*60 -> t{m}_bid/ask/mid/ts.
    Also: final open_interest, total volume, n_candles, result (1=yes, 0=no).

    Incremental: if ``cache_path`` exists, tickers already priced are skipped
    and cached rows are kept. Skip reasons are counted, logged, and stashed in
    ``df.attrs['skip_reasons']``.
    """
    session = session or requests.Session()

    cached: pd.DataFrame | None = None
    done: set[str] = set()
    if cache_path is not None and Path(cache_path).exists():
        cached = pd.read_parquet(cache_path)
        done = set(cached["ticker"])
        logger.info("price cache %s: %d rows already built", cache_path, len(cached))

    todo = markets_df[~markets_df["ticker"].isin(done)]
    skips: Counter = Counter()
    out_rows: list[dict] = []

    for i, mkt in enumerate(todo.itertuples(index=False), 1):
        if progress_every and i % progress_every == 0:
            logger.info("build_market_prices: %d/%d (built %d, skipped %d)",
                        i, len(todo), len(out_rows), sum(skips.values()))

        result_raw = str(getattr(mkt, "result", "") or "").lower()
        if result_raw not in ("yes", "no"):
            skips["unknown_result"] += 1
            continue

        teams = parse_teams(getattr(mkt, "title", None), getattr(mkt, "yes_sub_title", None))
        if teams is None:
            skips["unparseable_teams"] += 1
            continue
        yes_team, opp_team = teams

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
        series_ticker = str(mkt.ticker).split("-")[0]

        try:
            candles = candle_fetcher(
                mkt.ticker, series_ticker,
                int(open_time.timestamp()), int(close_time.timestamp()),
            )
        except Exception as exc:  # retries already exhausted inside the fetcher
            logger.warning("candle fetch failed for %s: %s", mkt.ticker, exc)
            skips["candle_fetch_error"] += 1
            time.sleep(pace_s)
            continue
        time.sleep(pace_s)  # ~20 reads/s shared budget — stay well under

        if candles is None or candles.empty:
            skips["no_candles"] += 1
            continue

        start_unix = int(match_start.timestamp())
        row: dict = {
            "ticker": mkt.ticker,
            "event_ticker": mkt.event_ticker,
            "yes_team": yes_team,
            "opp_team": opp_team,
            "match_start": match_start,
            "result": 1 if result_raw == "yes" else 0,
        }
        for m in snapshot_minutes_before:
            snap = _snapshot(candles, start_unix - m * 60)
            row[f"t{m}_bid"] = snap["bid"]
            row[f"t{m}_ask"] = snap["ask"]
            row[f"t{m}_mid"] = snap["mid"]
            row[f"t{m}_ts"] = snap["ts"]
        row["oi"] = candles["open_interest"].iloc[-1]
        row["volume"] = candles["volume"].sum()
        row["n_candles"] = len(candles)
        out_rows.append(row)

    cols = (
        ["ticker", "event_ticker", "yes_team", "opp_team", "match_start", "result"]
        + [f"t{m}_{f}" for m in snapshot_minutes_before for f in ("bid", "ask", "mid", "ts")]
        + ["oi", "volume", "n_candles"]
    )
    df = pd.DataFrame(out_rows, columns=cols)
    if cached is not None and len(cached):
        df = pd.concat([cached, df], ignore_index=True)
        df = df.drop_duplicates(subset="ticker", keep="last").reset_index(drop=True)

    if skips:
        logger.info("build_market_prices skips: %s", dict(skips))
    df.attrs["skip_reasons"] = dict(skips)
    return df
