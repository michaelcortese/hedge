"""Tests for lolpred.data.kalshi_micro — no network (fixtures mimic live schema)."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from lolpred.data.kalshi_micro import (
    CANDLE_COLS,
    TRADE_COLS,
    build_fetch_plan,
    chunk_windows,
    completed_tickers,
    fetch_market_trades,
    fetch_micro,
    fetch_minute_candles,
    normalize_trade,
)

TICKER = "KXLOLGAME-26JUL141600MAZGL-MAZ"


# ------------------------------------------------------------------- normalize_trade


def test_normalize_trade_dollars_variant():
    """Live schema (2026-07): string dollars + count_fp string."""
    raw = {
        "trade_id": "de2916a8-a5c5-6872-4484-3aaa9b0ade3a",
        "ticker": TICKER,
        "count_fp": "9.35",
        "yes_price_dollars": "0.0100",
        "no_price_dollars": "0.9900",
        "taker_side": "yes",
        "taker_book_side": "bid",
        "taker_outcome_side": "yes",
        "created_time": "2026-07-11T12:40:12.37995Z",
        "is_block_trade": False,
    }
    row = normalize_trade(raw)
    assert row["ticker"] == TICKER
    assert row["price_dollars"] == pytest.approx(0.01)
    assert row["count"] == pytest.approx(9.35)
    assert row["taker_side"] == "yes"
    assert row["ts"] == pd.Timestamp("2026-07-11 12:40:12.379950", tz="UTC")


def test_normalize_trade_cents_variant():
    """Older/other payload variant: integer cents + int count."""
    raw = {
        "trade_id": "t1",
        "ticker": TICKER,
        "count": 3,
        "yes_price": 55,
        "no_price": 45,
        "taker_side": "no",
        "created_time": "2026-07-11T12:00:00Z",
    }
    row = normalize_trade(raw)
    assert row["price_dollars"] == pytest.approx(0.55)
    assert row["count"] == pytest.approx(3.0)
    assert row["taker_side"] == "no"


def test_normalize_trade_missing_fields_are_nan():
    row = normalize_trade({"ticker": TICKER, "created_time": None})
    assert math.isnan(row["price_dollars"])
    assert math.isnan(row["count"])
    assert pd.isna(row["ts"])


# ------------------------------------------------------------------- pagination mock


class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def _trade(trade_id, ts, price="0.4200", count="2.00", side="yes"):
    return {
        "trade_id": trade_id,
        "ticker": TICKER,
        "count_fp": count,
        "yes_price_dollars": price,
        "no_price_dollars": f"{1 - float(price):.4f}",
        "taker_side": side,
        "created_time": ts,
        "is_block_trade": False,
    }


class _FakeTradesSession:
    """Two pages of /markets/trades linked by cursor."""

    def __init__(self):
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        if not (params or {}).get("cursor"):
            page = {
                "trades": [
                    _trade("t2", "2026-07-11T12:00:02Z", price="0.4400"),
                    _trade("t1", "2026-07-11T12:00:01Z"),
                ],
                "cursor": "page2",
            }
        else:
            assert params["cursor"] == "page2"
            page = {
                "trades": [_trade("t0", "2026-07-11T11:59:00Z", side="no")],
                "cursor": "",
            }
        return _FakeResponse(page)


def test_fetch_market_trades_paginates_and_normalizes():
    session = _FakeTradesSession()
    df = fetch_market_trades(TICKER, session=session, pace_s=0)

    assert len(session.calls) == 2
    url, first_params = session.calls[0]
    assert url.endswith("/markets/trades")
    assert first_params == {"ticker": TICKER, "limit": 1000}
    assert session.calls[1][1]["cursor"] == "page2"

    assert list(df.columns) == TRADE_COLS
    assert len(df) == 3
    assert df["ts"].is_monotonic_increasing  # sorted oldest-first
    assert df["ts"].dt.tz is not None
    assert df["price_dollars"].iloc[-1] == pytest.approx(0.44)
    assert set(df["taker_side"]) == {"yes", "no"}


def test_fetch_market_trades_empty_market():
    class _Empty:
        def get(self, url, params=None, timeout=None):
            return _FakeResponse({"trades": [], "cursor": ""})

    df = fetch_market_trades(TICKER, session=_Empty(), pace_s=0)
    assert list(df.columns) == TRADE_COLS
    assert len(df) == 0


# ---------------------------------------------------------------------- chunking


def test_chunk_windows_36h_gives_two_chunks():
    start, end = 1_000_000, 1_000_000 + 36 * 3600
    chunks = chunk_windows(start, end)
    assert len(chunks) == 2
    assert chunks[0] == (start, start + 24 * 3600)
    assert chunks[1] == (start + 24 * 3600, end)
    assert all(e - s <= 24 * 3600 for s, e in chunks)


def test_chunk_windows_exact_and_sub_chunk():
    assert chunk_windows(0, 24 * 3600) == [(0, 24 * 3600)]
    assert chunk_windows(0, 3600) == [(0, 3600)]
    assert chunk_windows(100, 100) == []
    assert chunk_windows(100, 50) == []


def _candle(ts, bid="0.4000", ask="0.4400", close=None, vol="0.00", oi="10.00"):
    price = {"previous_dollars": "0.4200"}
    if close is not None:
        price["close_dollars"] = close
    return {
        "end_period_ts": ts,
        "yes_bid": {"close_dollars": bid, "open_dollars": bid,
                    "high_dollars": bid, "low_dollars": bid},
        "yes_ask": {"close_dollars": ask, "open_dollars": ask,
                    "high_dollars": ask, "low_dollars": ask},
        "price": price,
        "volume_fp": vol,
        "open_interest_fp": oi,
    }


class _FakeCandlesSession:
    """Serves candles per requested chunk; records the windows asked for."""

    def __init__(self):
        self.windows = []

    def get(self, url, params=None, timeout=None):
        p = dict(params or {})
        assert p["period_interval"] == 1
        self.windows.append((p["start_ts"], p["end_ts"]))
        candles = [
            _candle(p["start_ts"] + 60, close="0.4200", vol="5.00"),
            # fully empty candle: no volume, no price, no bid/ask -> dropped
            {"end_period_ts": p["start_ts"] + 120, "volume_fp": "0.00",
             "open_interest_fp": "0.00"},
        ]
        return _FakeResponse({"candlesticks": candles})


def test_fetch_minute_candles_chunks_and_drops_empty_rows():
    session = _FakeCandlesSession()
    start = 1_800_000_000
    end = start + 36 * 3600  # 36h life -> 2 chunked calls
    df = fetch_minute_candles(TICKER, "KXLOLGAME", start, end, session=session, pace_s=0)

    assert session.windows == [(start, start + 24 * 3600), (start + 24 * 3600, end)]
    assert list(df.columns) == CANDLE_COLS
    assert len(df) == 2  # one kept candle per chunk; empty rows dropped
    assert (df["ticker"] == TICKER).all()
    assert df["end_period_ts"].dt.tz is not None
    assert df["yes_bid_close"].iloc[0] == pytest.approx(0.40)
    assert df["price_close"].iloc[0] == pytest.approx(0.42)
    assert df["volume"].iloc[0] == pytest.approx(5.0)
    assert df["oi"].iloc[0] == pytest.approx(10.0)


def test_fetch_minute_candles_keeps_quote_only_rows():
    """A candle with bid/ask but zero volume and no trade price is kept."""

    class _QuoteOnly:
        def get(self, url, params=None, timeout=None):
            return _FakeResponse({"candlesticks": [_candle(params["start_ts"] + 60)]})

    df = fetch_minute_candles(TICKER, "KXLOLGAME", 0, 3600, session=_QuoteOnly(), pace_s=0)
    assert len(df) == 1
    assert math.isnan(df["price_close"].iloc[0])


# ------------------------------------------------------------------ build_fetch_plan


def test_build_fetch_plan_window_and_fallbacks():
    tickers = pd.DataFrame({
        "ticker": [TICKER, "KXLOLGAME-26JUL151800AAABB-AAA"],
        "match_start": [pd.Timestamp("2026-07-14 16:00", tz="UTC"), pd.NaT],
    })
    markets = pd.DataFrame({
        "ticker": [TICKER, "KXLOLGAME-26JUL151800AAABB-AAA"],
        "open_time": [pd.Timestamp("2026-07-10 00:00", tz="UTC")] * 2,
        "close_time": [pd.Timestamp("2026-07-14 18:00", tz="UTC"),
                       pd.Timestamp("2026-07-15 20:00", tz="UTC")],
    })
    plan = build_fetch_plan(tickers, markets_df=markets)
    assert list(plan.columns) == ["ticker", "start_ts", "end_ts"]
    assert len(plan) == 2

    row = plan[plan["ticker"] == TICKER].iloc[0]
    # start = match_start - 12h, end = close_time
    assert row["start_ts"] == int(pd.Timestamp("2026-07-14 04:00", tz="UTC").timestamp())
    assert row["end_ts"] == int(pd.Timestamp("2026-07-14 18:00", tz="UTC").timestamp())

    # NaT match_start -> parsed from the ticker (26JUL151800 -> 2026-07-15 18:00)
    row2 = plan[plan["ticker"] != TICKER].iloc[0]
    assert row2["start_ts"] == int(pd.Timestamp("2026-07-15 06:00", tz="UTC").timestamp())
    assert row2["end_ts"] == int(pd.Timestamp("2026-07-15 20:00", tz="UTC").timestamp())


def test_build_fetch_plan_early_closed_market_falls_back_to_open_time():
    """Market closed days BEFORE match_start - 12h (rescheduled match)."""
    tickers = pd.DataFrame({
        "ticker": [TICKER],
        "match_start": [pd.Timestamp("2026-05-10 12:00", tz="UTC")],
    })
    markets = pd.DataFrame({
        "ticker": [TICKER],
        "open_time": [pd.Timestamp("2026-05-07 00:17", tz="UTC")],
        "close_time": [pd.Timestamp("2026-05-09 20:09", tz="UTC")],
    })
    plan = build_fetch_plan(tickers, markets_df=markets)
    assert len(plan) == 1
    row = plan.iloc[0]
    assert row["start_ts"] == int(pd.Timestamp("2026-05-07 00:17", tz="UTC").timestamp())
    assert row["end_ts"] == int(pd.Timestamp("2026-05-09 20:09", tz="UTC").timestamp())


# ------------------------------------------------------------- incremental skip logic


def _plan(*tickers):
    rows = [{"ticker": t, "start_ts": 0, "end_ts": 3600} for t in tickers]
    return pd.DataFrame(rows, columns=["ticker", "start_ts", "end_ts"])


def _stub_trades_df(ticker, n=2):
    return pd.DataFrame({
        "ticker": [ticker] * n,
        "ts": pd.to_datetime(["2026-07-11T12:00:00Z"] * n, utc=True),
        "price_dollars": [0.5] * n,
        "count": [1.0] * n,
        "taker_side": ["yes"] * n,
    })


def _stub_candles_df(ticker, n=3):
    return pd.DataFrame({
        "ticker": [ticker] * n,
        "end_period_ts": pd.to_datetime([60 * (i + 1) for i in range(n)], unit="s", utc=True),
        "yes_bid_close": [0.4] * n,
        "yes_ask_close": [0.44] * n,
        "price_close": [0.42] * n,
        "volume": [1.0] * n,
        "oi": [10.0] * n,
    })


def test_completed_tickers_missing_file(tmp_path):
    assert completed_tickers(tmp_path / "nope.parquet") == set()


def test_fetch_micro_skips_completed_and_resumes(tmp_path):
    trades_path = tmp_path / "trades.parquet"
    candles_path = tmp_path / "candles.parquet"

    fetched = []

    def stub_trades(ticker, session=None, pace_s=0.2, **kw):
        fetched.append(("trades", ticker))
        return _stub_trades_df(ticker)

    def stub_candles(ticker, series_ticker, start_ts, end_ts, session=None, pace_s=0.2, **kw):
        fetched.append(("candles", ticker))
        return _stub_candles_df(ticker)

    # First run: A and B fetched and written.
    summary = fetch_micro(
        _plan("A", "B"), trades_path, candles_path,
        session=object(), pace_s=0, flush_every=1,
        trades_fetcher=stub_trades, candles_fetcher=stub_candles,
    )
    assert summary["tickers_done"] == 2
    assert summary["tickers_skipped_cached"] == 0
    assert summary["n_trades"] == 4 and summary["n_candles"] == 6
    assert completed_tickers(candles_path) == {"A", "B"}

    # Second run over A, B, C: only C is fetched.
    fetched.clear()
    summary = fetch_micro(
        _plan("A", "B", "C"), trades_path, candles_path,
        session=object(), pace_s=0, flush_every=1,
        trades_fetcher=stub_trades, candles_fetcher=stub_candles,
    )
    assert [t for _, t in fetched] == ["C", "C"]
    assert summary["tickers_done"] == 1
    assert summary["tickers_skipped_cached"] == 2

    trades = pd.read_parquet(trades_path)
    candles = pd.read_parquet(candles_path)
    assert list(trades.columns) == TRADE_COLS
    assert list(candles.columns) == CANDLE_COLS
    assert sorted(set(trades["ticker"])) == ["A", "B", "C"]
    assert len(trades) == 6  # 2 per ticker, no duplicates from the rerun
    assert len(candles) == 9


def test_fetch_micro_failed_ticker_not_marked_complete(tmp_path):
    trades_path = tmp_path / "trades.parquet"
    candles_path = tmp_path / "candles.parquet"

    def stub_trades(ticker, session=None, pace_s=0.2, **kw):
        return _stub_trades_df(ticker)

    def boom(ticker, *a, **kw):
        raise RuntimeError("HTTP 500")

    summary = fetch_micro(
        _plan("A"), trades_path, candles_path,
        session=object(), pace_s=0, flush_every=1,
        trades_fetcher=stub_trades, candles_fetcher=boom,
    )
    assert summary["tickers_failed"] == 1
    assert summary["tickers_done"] == 0
    # Failed ticker leaves NO partial rows: neither trades nor candles written.
    assert completed_tickers(candles_path) == set()
    assert not trades_path.exists() or "A" not in set(pd.read_parquet(trades_path)["ticker"])
