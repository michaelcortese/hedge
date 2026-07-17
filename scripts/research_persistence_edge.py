#!/usr/bin/env python3
"""Test: does phrase resolution-history carry edge beyond the market price?

Signal: recency-weighted Beta-Binomial over the SAME (series, normalized
phrase) resolution history, using only markets whose close_time predates the
entry snapshot (no lookahead).

Tests:
  1. Encompassing logistic regression: y ~ logit(market price) + logit(p_hat).
     If p_hat's coefficient > 0 with cluster-robust p < .01, history adds info.
  2. Economic: threshold rule vs executable quotes (buy YES at ask / NO at
     100-bid) with Kalshi taker fees; cluster bootstrap by event AND by phrase.

Usage: research_persistence_edge.py [--entry-hours 24] [--thresh 0.10]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from research_mentions_lib import (  # noqa: E402
    cluster_bootstrap, load_candles, load_markets, pnl_no_cents, pnl_yes_cents,
    taker_fee_cents)


def norm_phrase(p: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", str(p).lower()).strip()


def build_signal(df: pd.DataFrame, half_life: float = 8.0) -> pd.DataFrame:
    """Recency-weighted Beta-Binomial p_hat per market from prior resolutions."""
    df = df.sort_values("close_time").reset_index(drop=True)
    df["nphrase"] = df["phrase"].map(norm_phrase)
    lam = 0.5 ** (1.0 / half_life)
    out_n, out_k, out_w, out_kw = [], [], [], []
    hist: dict[tuple, list] = {}
    for _, r in df.iterrows():
        key = (r["series"], r["nphrase"])
        h = hist.setdefault(key, [])
        # only settled strictly before this market OPENS (conservative)
        usable = [(t, y) for t, y in h if t < r["open_time"]]
        n = len(usable)
        w = np.array([lam ** (n - 1 - i) for i in range(n)]) if n else np.array([])
        ys = np.array([y for _, y in usable]) if n else np.array([])
        out_n.append(n)
        out_k.append(int(ys.sum()) if n else 0)
        out_w.append(float(w.sum()) if n else 0.0)
        out_kw.append(float((w * ys).sum()) if n else 0.0)
        h.append((r["close_time"], int(r["result"])))
    df["n_hist"], df["k_hist"] = out_n, out_k
    df["w_hist"], df["kw_hist"] = out_w, out_kw
    # Beta(1,1) prior over weighted counts
    df["p_hat"] = (df["kw_hist"] + 1.0) / (df["w_hist"] + 2.0)
    df["p_hat_sd"] = np.sqrt(df["p_hat"] * (1 - df["p_hat"]) / (df["w_hist"] + 2.0))
    return df


def entry_quotes(markets: pd.DataFrame, candles: pd.DataFrame,
                 entry_hours: float) -> pd.DataFrame:
    """Join each market to its candle nearest (close_time - entry_hours),
    requiring the candle to be at least entry_hours*0.5 before close."""
    if candles.empty:
        return pd.DataFrame()
    c = candles.dropna(subset=["close"]).copy()
    m = markets.set_index("ticker")
    c = c[c.ticker.isin(m.index)]
    c["close_time"] = c.ticker.map(m.close_time)
    c["target"] = c.close_time - pd.Timedelta(hours=entry_hours)
    c["dist"] = (c.ts - c.target).abs()
    # candle must be BEFORE close - half the horizon (stay away from resolution)
    c = c[c.ts <= c.close_time - pd.Timedelta(hours=entry_hours * 0.5)]
    pick = c.sort_values("dist").groupby("ticker").first().reset_index()
    pick = pick.rename(columns={"ts": "entry_ts"})
    keep = ["ticker", "entry_ts", "close", "yes_bid_close", "yes_ask_close",
            "volume", "open_interest"]
    return pick[keep]


def run(entry_hours: float, thresh: float, half_life: float,
        min_hist: float = 3.0, fee: bool = True) -> None:
    mk = load_markets()
    mk = mk.dropna(subset=["close_time", "open_time"])
    sig = build_signal(mk, half_life=half_life)
    ca = load_candles()
    q = entry_quotes(sig, ca, entry_hours)
    if q.empty:
        print("no candle data yet — rerun when phase 2 lands")
        return
    d = sig.merge(q, on="ticker", how="inner")
    d = d[d.n_hist >= min_hist].copy()
    # executable prices in cents; require a real two-sided quote
    d = d.dropna(subset=["yes_bid_close", "yes_ask_close"])
    d = d[(d.yes_ask_close >= 1) & (d.yes_ask_close <= 99)
          & (d.yes_bid_close >= 1) & (d.yes_bid_close <= 99)
          & (d.yes_ask_close > d.yes_bid_close - 1)]
    print(f"testable markets: {len(d)} (entry T-{entry_hours}h, "
          f"hist>={min_hist}, half_life={half_life})")
    if len(d) < 50:
        print("not enough priced markets yet — rerun when candles accumulate")
        return

    # ---- Test 1: encompassing regression with cluster-robust SEs
    import statsmodels.api as sm  # lazy
    mid = (d.yes_bid_close + d.yes_ask_close) / 200.0
    eps = 1e-3
    X = pd.DataFrame({
        "logit_price": np.log(mid.clip(eps, 1 - eps) / (1 - mid.clip(eps, 1 - eps))),
        "logit_phat": np.log(d.p_hat.clip(eps, 1 - eps) / (1 - d.p_hat.clip(eps, 1 - eps))),
    })
    X = sm.add_constant(X)
    fit = sm.Logit(d.result.values.astype(float), X).fit(
        disp=0, cov_type="cluster",
        cov_kwds={"groups": d.event_ticker.astype("category").cat.codes.values})
    print("\n== Encompassing logistic (y ~ logit price + logit p_hat), "
          "event-clustered SE ==")
    for name, b, p in zip(X.columns, fit.params, fit.pvalues):
        print(f"  {name:12s} beta={b:+.3f}  p={p:.2e}")

    # ---- Test 2: economic threshold rule
    d["edge_yes"] = d.p_hat - d.yes_ask_close / 100.0
    d["edge_no"] = d.yes_bid_close / 100.0 - d.p_hat
    trades = []
    for _, r in d.iterrows():
        if r.edge_yes > thresh:
            trades.append((r.event_ticker, r.series, r.nphrase,
                           pnl_yes_cents(r.yes_ask_close, r.result, fee), "yes", r.ticker))
        elif r.edge_no > thresh:
            trades.append((r.event_ticker, r.series, r.nphrase,
                           pnl_no_cents(r.yes_bid_close, r.result, fee), "no", r.ticker))
    t = pd.DataFrame(trades, columns=["event", "series", "phrase", "pnl", "side", "ticker"])
    print(f"\n== Threshold rule |edge|>{thresh:.2f}: {len(t)} trades ==")
    if len(t) < 20:
        print("too few trades")
        return
    print(t.side.value_counts().to_dict())
    for label, cl in (("event", t.event), ("phrase", t.series + "|" + t.phrase)):
        bs = cluster_bootstrap(t.pnl.values, cl.values)
        print(f"  clustered by {label:6s}: mean={bs['mean']:+.2f}c/contract "
              f"CI95=[{bs['ci_lo']:+.2f},{bs['ci_hi']:+.2f}] "
              f"p(<=0)={bs['p_le_0']:.4f} n={bs['n']} clusters={bs['n_clusters']}")
    print("\nby series:")
    print(t.groupby("series").agg(n=("pnl", "size"), mean_pnl=("pnl", "mean"))
          .sort_values("n", ascending=False).head(12))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry-hours", type=float, default=24)
    ap.add_argument("--thresh", type=float, default=0.10)
    ap.add_argument("--half-life", type=float, default=8.0)
    ap.add_argument("--min-hist", type=float, default=3)
    ap.add_argument("--no-fee", action="store_true")
    a = ap.parse_args()
    run(a.entry_hours, a.thresh, a.half_life, a.min_hist, fee=not a.no_fee)
