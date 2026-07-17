"""Shared research library for Kalshi mention-market analysis.

Loads the dataset produced by research_collect_mentions.py and provides the
single scoring convention all candidate strategies are graded with:

- Fills are EXECUTABLE prices, not mids: buy YES at the candle's yes_ask (or
  last trade + half-spread fallback), buy NO at 100 - yes_bid.
- Taker fee: ceil(0.07 * C * P * (1-P)) cents per contract batch (P in dollars).
- Significance: cluster bootstrap by EVENT (phrase markets within one call are
  strongly correlated) — never by market.

Everything in integer cents internally; convert at the edges.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data" / "research" / "mentions"


def _fnum(v) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- loading

def load_markets() -> pd.DataFrame:
    """One row per settled mention market, with event metadata + phrase."""
    rows = []
    for line in (DATA / "events.jsonl").open():
        ev = json.loads(line)
        for m in ev.get("markets") or []:
            if m.get("result") not in ("yes", "no"):
                continue
            rows.append({
                "ticker": m["ticker"],
                "event_ticker": ev["event_ticker"],
                "series": ev.get("series_ticker") or m["ticker"].split("-")[0],
                "event_title": ev.get("title", ""),
                "phrase": (m.get("yes_sub_title") or m.get("subtitle")
                           or m.get("title", "")),
                "market_title": m.get("title", ""),
                "rules": m.get("rules_primary", ""),
                "result": 1 if m["result"] == "yes" else 0,
                "open_time": pd.Timestamp(m.get("open_time")),
                "close_time": pd.Timestamp(m.get("close_time")),
                "expiration_time": pd.Timestamp(
                    m.get("expected_expiration_time")
                    or m.get("expiration_time")),
                "volume": _fnum(m.get("volume_fp") or m.get("volume")),
                "liquidity": _fnum(m.get("liquidity_dollars") or m.get("liquidity")),
                "last_price": _fnum(m.get("last_price_dollars") or m.get("last_price")),
                "settlement_ts": m.get("settlement_ts"),
                "occurrence": pd.Timestamp(m.get("occurrence_datetime")),
            })
    df = pd.DataFrame(rows)
    if len(df):
        df["family"] = df["series"].map(classify_family)
        df["company"] = df["series"].str.extract(
            r"^KXEARNINGSMENTION(.+)$", expand=False)
    return df


def classify_family(series: str) -> str:
    if series.startswith("KXEARNINGSMENTION"):
        return "earnings"
    if "PRESSMENTION" in series or "SECPRESS" in series:
        return "briefing"
    if "TRUMP" in series or "SAY" in series:
        return "speech"
    return "other"


def load_candles() -> pd.DataFrame:
    """Long frame: one row per (ticker, hour-candle)."""
    rows = []
    f = DATA / "candles.jsonl"
    if not f.exists():
        return pd.DataFrame()
    def cents(d: dict, key: str):
        """Read `key` from a candle sub-dict in either format, in CENTS."""
        v = d.get(key + "_dollars")
        if v is not None:
            try:
                return float(v) * 100.0
            except (TypeError, ValueError):
                return None
        v = d.get(key)
        return float(v) if v is not None else None

    for line in f.open():
        rec = json.loads(line)
        for c in rec.get("candles") or []:
            price = c.get("price") or {}
            ybid = c.get("yes_bid") or {}
            yask = c.get("yes_ask") or {}
            rows.append({
                "ticker": rec["ticker"],
                "ts": pd.Timestamp(c.get("end_period_ts"), unit="s", tz="UTC"),
                "open": cents(price, "open"), "close": cents(price, "close"),
                "high": cents(price, "high"), "low": cents(price, "low"),
                "prev_price": cents(price, "previous"),
                "yes_bid_close": cents(ybid, "close"),
                "yes_ask_close": cents(yask, "close"),
                "volume": _fnum(c.get("volume_fp") or c.get("volume")) or 0,
                "open_interest": _fnum(c.get("open_interest_fp")
                                       or c.get("open_interest")) or 0,
            })
    return pd.DataFrame(rows)


def load_trades() -> pd.DataFrame:
    rows = []
    f = DATA / "trades.jsonl"
    if not f.exists():
        return pd.DataFrame()
    for line in f.open():
        rec = json.loads(line)
        for t in rec.get("trades") or []:
            yp = t.get("yes_price_dollars")
            rows.append({
                "ticker": rec["ticker"],
                "ts": pd.Timestamp(t.get("created_time")),
                "yes_price": (float(yp) * 100.0) if yp is not None
                             else t.get("yes_price"),
                "count": _fnum(t.get("count_fp") or t.get("count")) or 0,
                "taker_side": t.get("taker_side"),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- economics

def taker_fee_cents(price_cents: float, contracts: int = 1) -> float:
    """Kalshi taker fee in cents for `contracts` at price (cents)."""
    p = price_cents / 100.0
    return math.ceil(7 * contracts * p * (1 - p)) if 0 < p < 1 else 0.0


def pnl_yes_cents(entry_yes_cents: float, result: int, fee: bool = True) -> float:
    """P&L in cents/contract of buying 1 YES at entry, held to settlement."""
    gross = (100.0 if result == 1 else 0.0) - entry_yes_cents
    return gross - (taker_fee_cents(entry_yes_cents) if fee else 0.0)


def pnl_no_cents(entry_yes_cents: float, result: int, fee: bool = True) -> float:
    """P&L in cents/contract of buying 1 NO at (100 - yes_bid)."""
    no_price = 100.0 - entry_yes_cents
    gross = (100.0 if result == 0 else 0.0) - no_price
    return gross - (taker_fee_cents(no_price) if fee else 0.0)


# ---------------------------------------------------------------- statistics

def cluster_bootstrap(values: np.ndarray, clusters: np.ndarray,
                      n_boot: int = 10_000, seed: int = 7) -> dict:
    """Mean + bootstrap CI resampling whole clusters (events)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(clusters)
    idx_by_cluster = {c: np.flatnonzero(clusters == c) for c in uniq}
    means = np.empty(n_boot)
    for b in range(n_boot):
        cs = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx_by_cluster[c] for c in cs])
        means[b] = values[sel].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    m = values.mean()
    se = means.std(ddof=1)
    # p-value for H0: mean <= 0 (one-sided) via bootstrap distribution
    p_one_sided = float((means <= 0).mean())
    return {"mean": float(m), "ci_lo": float(lo), "ci_hi": float(hi),
            "se": float(se), "n": int(len(values)),
            "n_clusters": int(len(uniq)), "p_le_0": p_one_sided}


def brier(probs: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean((probs - outcomes) ** 2))


def market_calibration_table(prices_cents: np.ndarray, outcomes: np.ndarray,
                             clusters: np.ndarray,
                             bins=(1, 5, 10, 20, 35, 50, 65, 80, 90, 95, 99)) -> pd.DataFrame:
    """Realized YES frequency by price bucket, with cluster-aware SEs."""
    df = pd.DataFrame({"p": prices_cents, "y": outcomes, "c": clusters})
    df["bin"] = pd.cut(df["p"], bins=list(bins), include_lowest=True)
    out = []
    for b, g in df.groupby("bin", observed=True):
        if not len(g):
            continue
        ev = g.groupby("c")["y"].mean()
        out.append({"bin": str(b), "n": len(g), "n_events": g["c"].nunique(),
                    "mean_price": g["p"].mean() / 100.0,
                    "realized": g["y"].mean(),
                    "se_cluster": ev.std(ddof=1) / max(1, np.sqrt(len(ev)))})
    return pd.DataFrame(out)
