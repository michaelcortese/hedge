"""Kalshi microstructure data for LoL markets: public trades + minute candles.

Read-only public market data — no auth. Complements ``kalshi_market`` (which
builds pre-match snapshots) with the raw tick/minute record of each market's
whole life, for research on intra-match price dynamics.

Endpoints used (verified live, 2026-07-15):
  GET /markets/trades?ticker=&limit=1000&cursor=...
      -> {"trades": [{trade_id, ticker, count_fp:"9.35",
          yes_price_dollars:"0.0100", no_price_dollars:"0.9900",
          taker_side:"yes"|"no", taker_book_side, taker_outcome_side,
          created_time:"2026-07-11T12:40:12.37995Z", is_block_trade}],
          "cursor": "..."}   (dollar/count fields are strings; older payload
          variants use integer-cent ``yes_price``/``no_price`` and int ``count``
          — both are handled)
  GET /series/{series}/markets/{ticker}/candlesticks
      ?start_ts=&end_ts=&period_interval=1
      -> minute candles; the REQUESTED window is capped at 5000 candles
         (HTTP 400 "max candlesticks: 5000" beyond ~83h at 1-minute interval),
         so windows are chunked to <= 24h (1440 minutes) per call.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable

import pandas as pd
import requests

from lolpred.data.kalshi_market import BASE, _dollars, _get_json, parse_event_start

logger = logging.getLogger(__name__)

# Empirical per-request cap on GET .../candlesticks (verified live 2026-07-15:
# a window spanning >5000 periods returns HTTP 400 "max candlesticks: 5000").
CANDLE_REQUEST_CAP = 5000
# Chunk minute-candle windows to 24h (1440 candles) — well under the cap.
DEFAULT_CHUNK_S = 24 * 3600

TRADE_COLS = ["ticker", "ts", "price_dollars", "count", "taker_side"]
CANDLE_COLS = [
    "ticker", "end_period_ts", "yes_bid_close", "yes_ask_close",
    "price_close", "volume", "oi",
]


# -------------------------------------------------------------------- normalization


def normalize_trade(raw: dict) -> dict:
    """One raw trade -> {ticker, ts, price_dollars, count, taker_side}.

    ``price_dollars`` is always the YES price. Handles both live schema
    variants: dollars-as-strings (``yes_price_dollars``/``count_fp``) and
    integer cents (``yes_price``/``count``).
    """
    price = raw.get("yes_price_dollars")
    if price is not None:
        try:
            price_f = float(price)
        except (TypeError, ValueError):
            price_f = float("nan")
    else:
        cents = raw.get("yes_price")
        try:
            price_f = float(cents) / 100.0
        except (TypeError, ValueError):
            price_f = float("nan")

    count = raw.get("count_fp")
    if count is None:
        count = raw.get("count")
    try:
        count_f = float(count)
    except (TypeError, ValueError):
        count_f = float("nan")

    return {
        "ticker": raw.get("ticker"),
        "ts": pd.to_datetime(raw.get("created_time"), utc=True, errors="coerce"),
        "price_dollars": price_f,
        "count": count_f,
        "taker_side": raw.get("taker_side"),
    }


def _flatten_candle(raw: dict) -> dict:
    """One raw candlestick -> flat floats (dollar fields are strings upstream)."""
    return {
        "end_period_ts": int(raw["end_period_ts"]),
        "yes_bid_close": _dollars(raw.get("yes_bid"), "close_dollars"),
        "yes_ask_close": _dollars(raw.get("yes_ask"), "close_dollars"),
        "price_close": _dollars(raw.get("price"), "close_dollars"),
        "volume": _dollars(raw, "volume_fp"),
        "oi": _dollars(raw, "open_interest_fp"),
    }


# ------------------------------------------------------------------------- chunking


def chunk_windows(
    start_ts: int, end_ts: int, chunk_s: int = DEFAULT_CHUNK_S
) -> list[tuple[int, int]]:
    """Split [start_ts, end_ts] into contiguous windows of at most ``chunk_s``."""
    start_ts, end_ts = int(start_ts), int(end_ts)
    if end_ts <= start_ts:
        return []
    out: list[tuple[int, int]] = []
    s = start_ts
    while s < end_ts:
        e = min(s + int(chunk_s), end_ts)
        out.append((s, e))
        s = e
    return out


# ------------------------------------------------------------------------ fetchers


def fetch_market_trades(
    ticker: str,
    session: requests.Session | None = None,
    base: str = BASE,
    limit: int = 1000,
    pace_s: float = 0.2,
) -> pd.DataFrame:
    """All public trades for one market, paginated via cursor.

    Columns: ticker, ts (UTC), price_dollars (YES price), count, taker_side.
    """
    session = session or requests.Session()
    rows: list[dict] = []
    cursor: str | None = None
    while True:
        params: dict = {"ticker": ticker, "limit": int(limit)}
        if cursor:
            params["cursor"] = cursor
        payload = _get_json(session, f"{base}/markets/trades", params=params)
        trades = payload.get("trades") or []
        rows.extend(normalize_trade(t) for t in trades)
        cursor = payload.get("cursor")
        if not cursor or not trades:
            break
        if pace_s:
            time.sleep(pace_s)

    df = pd.DataFrame(rows, columns=TRADE_COLS)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.sort_values("ts").reset_index(drop=True)


def fetch_minute_candles(
    ticker: str,
    series_ticker: str,
    start_ts: int,
    end_ts: int,
    session: requests.Session | None = None,
    base: str = BASE,
    chunk_s: int = DEFAULT_CHUNK_S,
    pace_s: float = 0.2,
) -> pd.DataFrame:
    """Minute candles (period_interval=1) over [start_ts, end_ts], chunked.

    Requested windows are chunked to <= ``chunk_s`` seconds per call (the API
    caps a request at ``CANDLE_REQUEST_CAP`` candles). Fully-empty candles —
    no volume, no price, no bid/ask close — are dropped.

    Columns: ticker, end_period_ts (UTC), yes_bid_close, yes_ask_close,
    price_close (nullable), volume, oi.
    """
    session = session or requests.Session()
    url = f"{base}/series/{series_ticker}/markets/{ticker}/candlesticks"

    rows: list[dict] = []
    windows = chunk_windows(start_ts, end_ts, chunk_s=chunk_s)
    for i, (s, e) in enumerate(windows):
        payload = _get_json(
            session, url,
            params={"start_ts": s, "end_ts": e, "period_interval": 1},
        )
        rows.extend(_flatten_candle(c) for c in payload.get("candlesticks") or [])
        if pace_s and i < len(windows) - 1:
            time.sleep(pace_s)

    df = pd.DataFrame(
        rows,
        columns=["end_period_ts", "yes_bid_close", "yes_ask_close",
                 "price_close", "volume", "oi"],
    )
    # Chunk boundaries can double-report the boundary candle.
    df = df.drop_duplicates(subset="end_period_ts", keep="last")
    df = df.sort_values("end_period_ts").reset_index(drop=True)

    keep = (
        (df["volume"] > 0)
        | df["price_close"].notna()
        | df["yes_bid_close"].notna()
        | df["yes_ask_close"].notna()
    )
    df = df[keep].reset_index(drop=True)

    df.insert(0, "ticker", ticker)
    df["end_period_ts"] = pd.to_datetime(df["end_period_ts"], unit="s", utc=True)
    return df[CANDLE_COLS]


# ----------------------------------------------------------------------- fetch plan


def build_fetch_plan(
    tickers_df: pd.DataFrame, markets_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Per-ticker candle windows: DataFrame[ticker, start_ts, end_ts] (unix s).

    Window = (match_start - 12h, fallback open_time) .. close_time. Sources,
    in order: columns already on ``tickers_df``; a merge from ``markets_df``
    (open_time/close_time); match_start parsed from the event/market ticker.
    Missing close_time falls back to match_start + 6h. Tickers with no usable
    start or end are dropped (logged).
    """
    df = tickers_df.drop_duplicates(subset="ticker").copy()
    if markets_df is not None:
        have = [c for c in ("open_time", "close_time") if c not in df.columns]
        cols = [c for c in ("ticker", *have) if c in markets_df.columns]
        if len(cols) > 1:
            df = df.merge(
                markets_df.drop_duplicates(subset="ticker")[cols],
                on="ticker", how="left",
            )
    for c in ("match_start", "open_time", "close_time"):
        if c not in df.columns:
            df[c] = pd.NaT

    rows: list[dict] = []
    dropped = 0
    for mkt in df.itertuples(index=False):
        match_start = mkt.match_start
        if pd.isna(match_start):
            match_start = (
                parse_event_start(getattr(mkt, "event_ticker", None))
                or parse_event_start(mkt.ticker)
            )

        if match_start is not None and pd.notna(match_start):
            start = pd.Timestamp(match_start) - pd.Timedelta(hours=12)
        elif pd.notna(mkt.open_time):
            start = pd.Timestamp(mkt.open_time)
        else:
            dropped += 1
            continue

        if pd.notna(mkt.close_time):
            end = pd.Timestamp(mkt.close_time)
        elif match_start is not None and pd.notna(match_start):
            end = pd.Timestamp(match_start) + pd.Timedelta(hours=6)
        else:
            dropped += 1
            continue

        # Markets closed early (rescheduled/cancelled match) can have their
        # whole life BEFORE match_start - 12h -> fall back to open_time.
        if start >= end and pd.notna(mkt.open_time):
            start = pd.Timestamp(mkt.open_time)

        start_ts, end_ts = int(start.timestamp()), int(end.timestamp())
        if end_ts <= start_ts:
            dropped += 1
            continue
        rows.append({"ticker": mkt.ticker, "start_ts": start_ts, "end_ts": end_ts})

    if dropped:
        logger.warning("build_fetch_plan: dropped %d tickers with no usable window", dropped)
    return pd.DataFrame(rows, columns=["ticker", "start_ts", "end_ts"])


# ---------------------------------------------------------------------- incremental


def completed_tickers(candles_path: str | Path) -> set[str]:
    """Tickers already present in the candles output parquet (== complete).

    Candles are always written after (and together with) that ticker's trades,
    so presence in the candles file implies the trades are on disk too.
    """
    path = Path(candles_path)
    if not path.exists():
        return set()
    return set(pd.read_parquet(path, columns=["ticker"])["ticker"])


def _merge_write(path: Path, new: pd.DataFrame, cols: list[str]) -> int:
    """existing-minus-refetched + new -> parquet. Returns total rows written."""
    frames = []
    if path.exists() and len(new):
        old = pd.read_parquet(path)
        frames.append(old[~old["ticker"].isin(set(new["ticker"]))])
    elif path.exists():
        frames.append(pd.read_parquet(path))
    frames.append(new)
    out = pd.concat(frames, ignore_index=True) if frames else new
    out = out[cols]
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return len(out)


def fetch_micro(
    plan: pd.DataFrame,
    trades_path: str | Path,
    candles_path: str | Path,
    session: requests.Session | None = None,
    pace_s: float = 0.2,
    flush_every: int = 50,
    progress_every: int = 50,
    trades_fetcher: Callable[..., pd.DataFrame] = fetch_market_trades,
    candles_fetcher: Callable[..., pd.DataFrame] = fetch_minute_candles,
) -> dict:
    """Fetch trades + minute candles for every ticker in ``plan``; write parquets.

    Incremental: tickers already in ``candles_path`` are skipped. Both outputs
    are flushed to disk (trades first, then candles — the completeness marker)
    every ``flush_every`` tickers so a crash resumes where it left off. A
    ticker's rows are added only when BOTH fetches succeed; per-ticker
    failures (retries already exhausted inside ``_get_json``) are logged,
    counted, and skipped.

    Returns a summary dict: tickers done/failed/skipped, new trade/candle rows.
    """
    session = session or requests.Session()
    trades_path, candles_path = Path(trades_path), Path(candles_path)

    done = completed_tickers(candles_path)
    todo = plan[~plan["ticker"].isin(done)]
    logger.info(
        "fetch_micro: %d tickers todo (%d already complete in %s)",
        len(todo), len(plan) - len(todo), candles_path,
    )

    trade_frames: list[pd.DataFrame] = []
    candle_frames: list[pd.DataFrame] = []
    n_done = n_failed = n_trades = n_candles = 0
    unflushed = 0

    def _flush() -> None:
        nonlocal trade_frames, candle_frames, unflushed
        if not unflushed:
            return
        new_trades = (
            pd.concat(trade_frames, ignore_index=True)
            if trade_frames else pd.DataFrame(columns=TRADE_COLS)
        )
        new_candles = (
            pd.concat(candle_frames, ignore_index=True)
            if candle_frames else pd.DataFrame(columns=CANDLE_COLS)
        )
        # Trades first: candles presence is the "ticker complete" marker.
        t_total = _merge_write(trades_path, new_trades, TRADE_COLS)
        c_total = _merge_write(candles_path, new_candles, CANDLE_COLS)
        logger.info(
            "flushed %d tickers -> %s (%d rows), %s (%d rows)",
            unflushed, trades_path, t_total, candles_path, c_total,
        )
        trade_frames, candle_frames, unflushed = [], [], 0

    for i, mkt in enumerate(todo.itertuples(index=False), 1):
        if progress_every and i % progress_every == 0:
            logger.info(
                "fetch_micro: %d/%d tickers (done %d, failed %d, %d trades, %d candles)",
                i, len(todo), n_done, n_failed, n_trades, n_candles,
            )
        series_ticker = str(mkt.ticker).split("-")[0]
        try:
            trades = trades_fetcher(mkt.ticker, session=session, pace_s=pace_s)
            if pace_s:
                time.sleep(pace_s)
            candles = candles_fetcher(
                mkt.ticker, series_ticker, mkt.start_ts, mkt.end_ts,
                session=session, pace_s=pace_s,
            )
        except Exception as exc:  # retries already exhausted inside _get_json
            logger.warning("fetch failed for %s: %s", mkt.ticker, exc)
            n_failed += 1
            if pace_s:
                time.sleep(pace_s)
            continue
        if pace_s:
            time.sleep(pace_s)

        if len(trades):
            trade_frames.append(trades)
            n_trades += len(trades)
        if len(candles):
            candle_frames.append(candles)
            n_candles += len(candles)
        elif not len(trades):
            logger.info("no trades and no candles for %s", mkt.ticker)
        n_done += 1
        unflushed += 1
        if flush_every and unflushed >= flush_every:
            _flush()

    _flush()
    return {
        "tickers_done": n_done,
        "tickers_failed": n_failed,
        "tickers_skipped_cached": len(plan) - len(todo),
        "n_trades": n_trades,
        "n_candles": n_candles,
    }
