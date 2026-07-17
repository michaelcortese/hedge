#!/usr/bin/env python3
"""In-event hazard test: does the market under-collapse P(YES) as the event
runs out of road?

Structure exploited: a mention market resolves YES the instant the phrase is
said (close_time == mention moment, confirmed by early closes); NO markets
close at event end. So within an event:

  event_end   = max close_time over the event's markets (the NO close)
  event_start = event_end - W (per-series window) — refined by tape activity
  tau         = fraction of the event window elapsed

Truth curve: among markets in an initial-price bucket, P(eventual YES | still
unresolved at tau). Market curve: mean tape price of those unresolved markets
in the tau bucket. If market >> truth at high tau, buying NO in-event is edge.

Economic test: at tau* in a grid, for every unresolved market whose last tape
price within [tau*-w, tau*] is p, buy NO at (100 - p) + slip; settle at
result. Cluster bootstrap by event.
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
    cluster_bootstrap, load_markets, load_trades, taker_fee_cents)

# event window (hours before event_end) per series prefix; conservative
WINDOWS = {
    "KXWCMENTION": 2.5, "KXMLBMENTION": 3.5, "KXNBAMENTION": 3.0,
    "KXNHLMENTION": 3.0, "KXTRUMPMENTION": 1.5, "KXTRUMPMENTIONB": 1.5,
    "KXHEARINGMENTION": 3.0, "KXVANCEMENTION": 1.5, "KXMAMDANIMENTION": 1.5,
    "KXLOVEISLMENTION": 1.5, "KXFIGHTMENTION": 1.0, "KXLATENIGHTMENTION": 1.5,
    "KXSECPRESSMENTION": 1.0, "KXPOLITICSMENTION": 1.5,
}


def onset_windows(mk: pd.DataFrame, trades: pd.DataFrame,
                  rate_threshold: int = 10) -> pd.Series:
    """Causal event-start detector: first moment (any time in the market's
    life) where the pooled 15-min trade count across the event's markets
    reaches `rate_threshold`. Live-implementable: a monitor sees the same
    trades in real time. Returns event_ticker -> onset Timestamp."""
    t = trades.merge(mk[["ticker", "event_ticker"]], on="ticker", how="inner")
    out = {}
    date_re = re.compile(r"-(\d{2})([A-Z]{3})(\d{2})")
    months = {m: i + 1 for i, m in enumerate(
        ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
         "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}
    for et, g in t.groupby("event_ticker"):
        ts = g.ts.sort_values().reset_index(drop=True)
        # ex-ante date gate: the ticker embeds the scheduled event date
        m = date_re.search(et)
        if m:
            yy, mon, dd = int(m.group(1)), months.get(m.group(2)), int(m.group(3))
            if mon:
                day = pd.Timestamp(2000 + yy, mon, dd, tz="UTC")
                ts = ts[(ts >= day - pd.Timedelta(hours=12))
                        & (ts <= day + pd.Timedelta(hours=42))]
                ts = ts.reset_index(drop=True)
        if len(ts) < rate_threshold:
            continue
        # rolling count: time between trade i and i+threshold-1 <= 15 min
        gap = ts.shift(-(rate_threshold - 1)) - ts
        hit = ts[gap <= pd.Timedelta(minutes=15)]
        if len(hit):
            out[et] = hit.iloc[0]
    return pd.Series(out, name="event_start")


def build_event_frame(mk: pd.DataFrame, mode: str = "expost",
                      trades: pd.DataFrame | None = None) -> pd.DataFrame:
    """mode='expost': window ends at realized max close (needs hindsight).
    mode='exante': window = [median occurrence_datetime, +W hours].
    mode='onset': window = [tape-burst onset, +W hours] — fully causal."""
    if mode == "onset":
        assert trades is not None
        mk = mk.dropna(subset=["close_time"]).copy()
        mk["W_h"] = mk.series.map(WINDOWS)
        mk = mk.dropna(subset=["W_h"])
        ev_start = onset_windows(mk, trades)
        mk = mk.join(ev_start, on="event_ticker")
        mk = mk.dropna(subset=["event_start"])
        mk["event_end"] = mk.event_start + pd.to_timedelta(mk.W_h, unit="h")
        mk["mention_tau"] = np.where(
            (mk.result == 1) & (mk.close_time >= mk.event_start),
            (mk.close_time - mk.event_start).dt.total_seconds()
            / (mk.event_end - mk.event_start).dt.total_seconds().replace(0, np.nan),
            np.nan)
        return mk
    mk = mk.dropna(subset=["close_time"]).copy()
    mk["W_h"] = mk.series.map(WINDOWS)
    mk = mk.dropna(subset=["W_h"])
    if mode == "exante":
        ev_start = (mk.dropna(subset=["occurrence"])
                    .groupby("event_ticker").occurrence.median()
                    .rename("event_start"))
        mk = mk.join(ev_start, on="event_ticker")
        mk = mk.dropna(subset=["event_start"])
        mk["event_end"] = mk.event_start + pd.to_timedelta(mk.W_h, unit="h")
    else:
        ev_end = mk.groupby("event_ticker").close_time.max().rename("event_end")
        mk = mk.join(ev_end, on="event_ticker")
        mk["event_start"] = mk.event_end - pd.to_timedelta(mk.W_h, unit="h")
    # YES markets that closed inside the window: mention time = close_time
    mk["mention_tau"] = np.where(
        (mk.result == 1) & (mk.close_time >= mk.event_start),
        (mk.close_time - mk.event_start).dt.total_seconds()
        / (mk.event_end - mk.event_start).dt.total_seconds().replace(0, np.nan),
        np.nan)
    return mk


def run(taus: list[float], slip_cents: float, min_trades_window: int) -> None:
    mk = build_event_frame(load_markets())
    tr = load_trades()
    if tr.empty:
        print("no tapes yet")
        return
    tr = tr.merge(mk[["ticker", "event_ticker", "series", "result",
                      "event_start", "event_end"]], on="ticker", how="inner")
    tr["tau"] = ((tr.ts - tr.event_start).dt.total_seconds()
                 / (tr.event_end - tr.event_start).dt.total_seconds())
    tr = tr.dropna(subset=["tau", "yes_price"])
    in_ev = tr[(tr.tau >= 0) & (tr.tau <= 1)]
    print(f"tapes: {tr.ticker.nunique()} markets | in-event trades: {len(in_ev)} "
          f"on {in_ev.ticker.nunique()} markets, {in_ev.event_ticker.nunique()} events")

    # mention hazard sanity: what fraction of eventual-YES resolve inside window
    ym = mk[(mk.result == 1) & mk.ticker.isin(tr.ticker.unique())]
    inw = ym.mention_tau.notna().mean() if len(ym) else float("nan")
    print(f"eventual-YES with mention inside assumed window: {inw:.2%} (n={len(ym)})")

    print("\n== tau-grid: market price vs conditional truth (unresolved at tau) ==")
    rows = []
    trades_all = []
    for tau in taus:
        # markets unresolved at tau: NO markets, or YES with mention_tau > tau
        cand = mk[mk.ticker.isin(tr.ticker.unique())].copy()
        cand["unres"] = (cand.result == 0) | (cand.mention_tau > tau)
        cand = cand[cand.unres]
        # last trade price in [tau-0.15, tau]
        w = in_ev[(in_ev.tau <= tau) & (in_ev.tau > tau - 0.15)]
        last = (w.sort_values("ts").groupby("ticker").yes_price.last()
                .rename("p_tau"))
        d = cand.join(last, on="ticker", how="inner")
        # eventual YES among these
        if len(d) < 30:
            rows.append((tau, len(d), np.nan, np.nan))
            continue
        truth = d.result.mean()
        price = d.p_tau.mean() / 100.0
        rows.append((tau, len(d), price, truth))
        # economic: buy NO on everything unresolved priced in [10,60]
        g = d[(d.p_tau >= 10) & (d.p_tau <= 60)]
        for r in g.itertuples():
            no_cost = 100.0 - r.p_tau + slip_cents
            pnl = (100.0 if r.result == 0 else 0.0) - no_cost \
                - taker_fee_cents(no_cost)
            trades_all.append((tau, r.event_ticker, r.series, pnl))
    t = pd.DataFrame(rows, columns=["tau", "n_unres", "mean_price", "cond_truth"])
    t["gap"] = t.mean_price - t.cond_truth
    print(t.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

    # price-bucket-controlled gap at each tau (composition-safe)
    print("\n== gap by (tau, price bucket): realized - price, cluster SE ==")
    from research_mentions_lib import market_calibration_table
    for tau in taus:
        cand = mk[mk.ticker.isin(tr.ticker.unique())].copy()
        cand["unres"] = (cand.result == 0) | (cand.mention_tau > tau)
        cand = cand[cand.unres]
        w = in_ev[(in_ev.tau <= tau) & (in_ev.tau > tau - 0.15)]
        last = (w.sort_values("ts").groupby("ticker").yes_price.last()
                .rename("p_tau"))
        d = cand.join(last, on="ticker", how="inner")
        if len(d) < 60:
            continue
        tab = market_calibration_table(d.p_tau.values, d.result.values,
                                       d.event_ticker.values,
                                       bins=(1, 15, 30, 50, 70, 99))
        tab["gap"] = tab["realized"] - tab["mean_price"]
        tab["z"] = tab["gap"] / tab["se_cluster"].replace(0, np.nan)
        tab.insert(0, "tau", tau)
        print(tab[["tau", "bin", "n", "n_events", "mean_price", "realized",
                   "gap", "z"]].to_string(index=False,
                  float_format=lambda x: f"{x:+.3f}" if abs(x) < 10 else f"{x:.0f}"))

    ta = pd.DataFrame(trades_all, columns=["tau", "event", "series", "pnl"])
    print(f"\n== buy-NO in-event (price 10-60c, slip {slip_cents}c, fee) ==")
    for tau, g in ta.groupby("tau"):
        if len(g) < 25:
            continue
        bs = cluster_bootstrap(g.pnl.values, g.event.values)
        print(f"  tau={tau:.2f}: n={bs['n']:5d} ev={bs['n_clusters']:4d} "
              f"mean={bs['mean']:+6.2f}c CI=[{bs['ci_lo']:+6.2f},{bs['ci_hi']:+6.2f}] "
              f"p(<=0)={bs['p_le_0']:.4f}")
    if len(ta):
        print("\nby series (all taus pooled):")
        print(ta.groupby("series").agg(n=("pnl", "size"), mean=("pnl", "mean"))
              .sort_values("n", ascending=False).head(10).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slip", type=float, default=1.0)
    ap.add_argument("--min-trades", type=int, default=25)
    a = ap.parse_args()
    run([0.3, 0.5, 0.7, 0.85, 0.95], a.slip, a.min_trades)
