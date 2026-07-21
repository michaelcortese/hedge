"""Tests for lolpred.data.kalshi_market — no network (fixtures mimic live schema)."""

from __future__ import annotations

import os

import pandas as pd
import pytest

from lolpred.data.kalshi_market import (
    build_market_prices,
    fetch_settled_lol_markets,
    normalize_team,
    parse_event_start,
    parse_teams,
)

SAMPLE_TITLE = (
    "Will Maze Gaming win the Maze Gaming vs. Golden Lions "
    "League of Legends match?"
)


# ------------------------------------------------------------------ parse_event_start


def test_parse_event_start_event_ticker():
    ts = parse_event_start("KXLOLGAME-26JUL141600MAZGL")
    assert ts == pd.Timestamp("2026-07-14 16:00", tz="UTC")


def test_parse_event_start_full_market_ticker():
    ts = parse_event_start("KXLOLGAME-26JUL141600MAZGL-MAZ")
    assert ts == pd.Timestamp("2026-07-14 16:00", tz="UTC")


@pytest.mark.parametrize(
    "garbage",
    ["", "KXLOLGAME", "KXLOLGAME-garbage", "KXLOLGAME-26XXX141600ABC", None, 42,
     "KXLOLGAME-26JUL341600MAZGL"],  # day 34 does not exist
)
def test_parse_event_start_garbage(garbage):
    assert parse_event_start(garbage) is None


# ------------------------------------------------------------------------ parse_teams


def test_parse_teams_yes_is_first_team():
    assert parse_teams(SAMPLE_TITLE, "Maze Gaming") == ("Maze Gaming", "Golden Lions")


def test_parse_teams_yes_is_second_team():
    title = (
        "Will Golden Lions win the Maze Gaming vs. Golden Lions "
        "League of Legends match?"
    )
    assert parse_teams(title, "Golden Lions") == ("Golden Lions", "Maze Gaming")


def test_parse_teams_null_sub_title_falls_back_to_will_clause():
    assert parse_teams(SAMPLE_TITLE, None) == ("Maze Gaming", "Golden Lions")


def test_parse_teams_vs_without_dot_and_casing():
    title = "will MAZE GAMING win the Maze Gaming vs Golden Lions league of legends match?"
    assert parse_teams(title, "maze gaming") == ("Maze Gaming", "Golden Lions")


@pytest.mark.parametrize("title", [None, "", "Will it rain tomorrow?"])
def test_parse_teams_unparseable(title):
    assert parse_teams(title, "Whatever") is None


# --------------------------------------------------------------------- normalize_team


def test_normalize_team():
    assert normalize_team("T1 Esports") == "t1"
    assert normalize_team("Golden Lions") == "golden lions"
    assert normalize_team("Maze Gaming") == "maze"
    assert normalize_team("Team") == "team"  # never strip down to nothing
    assert normalize_team(None) == ""


# ------------------------------------------------------------------ pagination (mock)


class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class _FakeSession:
    """Two pages of /markets, linked by cursor."""

    def __init__(self):
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        market = {
            "event_ticker": "KXLOLGAME-26JUL141600MAZGL",
            "title": SAMPLE_TITLE,
            "yes_sub_title": "Maze Gaming",
            "open_time": "2026-07-10T00:00:00Z",
            "close_time": "2026-07-14T18:00:00Z",
            "expected_expiration_time": "2026-07-14T19:00:00Z",
            "result": "yes",
            "last_price": None,
            "volume": None,
            "liquidity": None,
        }
        if not (params or {}).get("cursor"):
            page = {"markets": [dict(market, ticker="TICK-1")], "cursor": "page2"}
        else:
            assert params["cursor"] == "page2"
            page = {"markets": [dict(market, ticker="TICK-2", result="no")], "cursor": ""}
        return _FakeResponse(page)


def test_fetch_settled_lol_markets_paginates():
    session = _FakeSession()
    df = fetch_settled_lol_markets(session=session)
    assert len(session.calls) == 2
    url, first_params = session.calls[0]
    assert url.endswith("/markets")
    assert first_params["series_ticker"] == "KXLOLGAME"
    assert first_params["status"] == "settled"
    assert first_params["limit"] == 200
    assert "cursor" not in first_params
    assert session.calls[1][1]["cursor"] == "page2"

    assert sorted(df["ticker"]) == ["TICK-1", "TICK-2"]
    assert df["close_time"].dt.tz is not None
    assert set(df["result"]) == {"yes", "no"}


# ------------------------------------------------------------- build_market_prices


def _markets_df(tickers=("KXLOLGAME-26JUL141600MAZGL-MAZ",)):
    rows = []
    for t in tickers:
        rows.append(
            {
                "ticker": t,
                "event_ticker": "KXLOLGAME-26JUL141600MAZGL",
                "title": SAMPLE_TITLE,
                "yes_sub_title": "Maze Gaming",
                "open_time": pd.Timestamp("2026-07-10 00:00", tz="UTC"),
                "close_time": pd.Timestamp("2026-07-14 18:00", tz="UTC"),
                "expected_expiration_time": pd.Timestamp("2026-07-14 19:00", tz="UTC"),
                "result": "yes",
                "last_price": None,
                "volume": None,
                "liquidity": None,
            }
        )
    return pd.DataFrame(rows)


# Match start is 2026-07-14 16:00 UTC = 1784044800.
MATCH_START_UNIX = int(pd.Timestamp("2026-07-14 16:00", tz="UTC").timestamp())


def _candles_df():
    """Hand-built hourly candles around the t-60 and t-5 cutoffs.

    t60 cutoff = start - 3600; t5 cutoff = start - 300. Candles straddle each
    cutoff so the test proves the candle BEFORE the cutoff is chosen, not after.
    """
    rows = [
        # 2h before start — the only candle <= t60 cutoff
        {"end_period_ts": MATCH_START_UNIX - 7200, "yes_bid_close": 0.40,
         "yes_ask_close": 0.44, "volume": 10.0, "open_interest": 100.0},
        # 30min before start — after t60 cutoff, before t5 cutoff -> t5 pick
        {"end_period_ts": MATCH_START_UNIX - 1800, "yes_bid_close": 0.50,
         "yes_ask_close": 0.54, "volume": 5.0, "open_interest": 120.0},
        # 1min before start — after the t5 cutoff, must NOT be picked
        {"end_period_ts": MATCH_START_UNIX - 60, "yes_bid_close": 0.90,
         "yes_ask_close": 0.94, "volume": 2.0, "open_interest": 130.0},
    ]
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["end_period_ts"], unit="s", utc=True)
    for col in ("yes_bid_open", "yes_bid_high", "yes_bid_low",
                "yes_ask_open", "yes_ask_high", "yes_ask_low",
                "price_open", "price_close"):
        df[col] = 0.5
    return df


def test_build_market_prices_snapshots_and_result():
    calls = []

    def stub_fetcher(ticker, series_ticker, start_ts, end_ts, **kw):
        calls.append((ticker, series_ticker, start_ts, end_ts))
        return _candles_df()

    df = build_market_prices(_markets_df(), candle_fetcher=stub_fetcher, pace_s=0)

    assert len(df) == 1
    row = df.iloc[0]
    assert calls[0][1] == "KXLOLGAME"  # series derived from ticker

    # t60: candle at -7200 chosen (the -1800 candle is AFTER the cutoff)
    assert row["t60_bid"] == pytest.approx(0.40)
    assert row["t60_ask"] == pytest.approx(0.44)
    assert row["t60_mid"] == pytest.approx(0.42)
    assert row["t60_ts"] == pd.Timestamp(MATCH_START_UNIX - 7200, unit="s", tz="UTC")

    # t5: candle at -1800 chosen (the -60 candle is AFTER the cutoff)
    assert row["t5_bid"] == pytest.approx(0.50)
    assert row["t5_ask"] == pytest.approx(0.54)
    assert row["t5_mid"] == pytest.approx(0.52)
    assert row["t5_ts"] == pd.Timestamp(MATCH_START_UNIX - 1800, unit="s", tz="UTC")

    assert row["result"] == 1
    assert row["yes_team"] == "Maze Gaming"
    assert row["opp_team"] == "Golden Lions"
    assert row["match_start"] == pd.Timestamp("2026-07-14 16:00", tz="UTC")
    assert row["oi"] == pytest.approx(130.0)
    assert row["volume"] == pytest.approx(17.0)
    assert row["n_candles"] == 3

    expected_cols = [
        "ticker", "event_ticker", "yes_team", "opp_team", "match_start", "result",
        "t5_bid", "t5_ask", "t5_mid", "t5_ts",
        "t60_bid", "t60_ask", "t60_mid", "t60_ts",
        "oi", "volume", "n_candles",
    ]
    assert list(df.columns) == expected_cols


def test_build_market_prices_result_no_maps_to_zero():
    markets = _markets_df()
    markets.loc[0, "result"] = "no"
    df = build_market_prices(markets, candle_fetcher=lambda *a, **k: _candles_df(), pace_s=0)
    assert df.iloc[0]["result"] == 0


def test_build_market_prices_skips_no_candles():
    empty = _candles_df().iloc[0:0]
    df = build_market_prices(
        _markets_df(), candle_fetcher=lambda *a, **k: empty, pace_s=0
    )
    assert len(df) == 0
    assert df.attrs["skip_reasons"] == {"no_candles": 1}


def test_build_market_prices_skips_unparseable_teams():
    markets = _markets_df()
    markets.loc[0, "title"] = "Will it rain tomorrow?"
    df = build_market_prices(
        markets, candle_fetcher=lambda *a, **k: _candles_df(), pace_s=0
    )
    assert len(df) == 0
    assert df.attrs["skip_reasons"] == {"unparseable_teams": 1}


# -------------------------------------------------------------- live smoke (opt-in)


@pytest.mark.network
@pytest.mark.skipif(
    os.environ.get("KALSHI_NET") != "1",
    reason="live Kalshi smoke test; set KALSHI_NET=1 to run",
)
def test_live_first_page_has_markets():
    import requests

    from lolpred.data.kalshi_market import BASE

    resp = requests.get(
        f"{BASE}/markets",
        params={"series_ticker": "KXLOLGAME", "status": "settled", "limit": 5},
        timeout=15,
    )
    resp.raise_for_status()
    assert len(resp.json()["markets"]) > 0
