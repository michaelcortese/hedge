#!/usr/bin/env python3
"""Market-side tests needing no external corpus: calibration + basket overround.

Test A (calibration grid): realized YES frequency vs executable price bucket
at entry horizon T-Xh, event-clustered SEs. Miscalibration > fee floor at any
bucket = tradable price-only edge (favorite-longshot family).

Test B (event-basket overround): per event, sum of YES mids vs realized YES
count. If events are systematically overpriced in aggregate, buying the NO
basket pools phrase-level noise; events are the natural iid unit.

Usage: research_market_calibration.py [--entry-hours 24]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from research_mentions_lib import (  # noqa: E402
    cluster_bootstrap, load_candles, load_markets, market_calibration_table,
    pnl_no_cents, pnl_yes_cents)
from research_persistence_edge import entry_quotes  # noqa: E402


def run(entry_hours: float) -> None:
    mk = load_markets().dropna(subset=["close_time", "open_time"])
    ca = load_candles()
    q = entry_quotes(mk, ca, entry_hours)
    if q.empty:
        print("no candles yet")
        return
    d = mk.merge(q, on="ticker", how="inner")
    d = d.dropna(subset=["yes_bid_close", "yes_ask_close"])
    d = d[(d.yes_ask_close.between(1, 99)) & (d.yes_bid_close.between(1, 99))]
    d["mid"] = (d.yes_bid_close + d.yes_ask_close) / 2.0
    print(f"priced markets at T-{entry_hours}h: {len(d)} "
          f"({d.event_ticker.nunique()} events)\n")

    # ---------- Test A: calibration grid
    tab = market_calibration_table(d["mid"].values, d["result"].values,
                                   d["event_ticker"].values)
    tab["gap"] = tab["realized"] - tab["mean_price"]
    tab["z"] = tab["gap"] / tab["se_cluster"].replace(0, np.nan)
    print("== Calibration by mid-price bucket (gap = realized - price) ==")
    print(tab.to_string(index=False,
                        float_format=lambda x: f"{x:+.3f}" if abs(x) < 10 else f"{x:.0f}"))

    # simple fade rules, executable prices + fees
    print("\n== Fade rules (executable, fee-adjusted, event-clustered) ==")
    for name, mask, side in [
        ("buy NO on mid>=80", d.mid >= 80, "no"),
        ("buy NO on mid in [50,80)", (d.mid >= 50) & (d.mid < 80), "no"),
        ("buy YES on mid>=80", d.mid >= 80, "yes"),
        ("buy NO on mid<=20", d.mid <= 20, "no"),
        ("buy YES on mid<=20", d.mid <= 20, "yes"),
        ("buy YES on mid in (20,50)", (d.mid > 20) & (d.mid < 50), "yes"),
    ]:
        g = d[mask]
        if len(g) < 30:
            continue
        pnl = np.array([
            pnl_yes_cents(r.yes_ask_close, r.result) if side == "yes"
            else pnl_no_cents(r.yes_bid_close, r.result)
            for r in g.itertuples()])
        bs = cluster_bootstrap(pnl, g.event_ticker.values)
        print(f"  {name:28s} n={bs['n']:5d} ev={bs['n_clusters']:4d} "
              f"mean={bs['mean']:+6.2f}c CI=[{bs['ci_lo']:+6.2f},{bs['ci_hi']:+6.2f}] "
              f"p(<=0)={bs['p_le_0']:.4f}")

    # ---------- Test B: basket overround
    ev = d.groupby("event_ticker").agg(
        n=("result", "size"), sum_mid=("mid", "mean"),  # placeholder, fixed below
        yes_frac=("result", "mean"), series=("series", "first"))
    ev["sum_mid"] = d.groupby("event_ticker")["mid"].sum() / 100.0
    ev["sum_yes"] = d.groupby("event_ticker")["result"].sum()
    ev["overround"] = ev["sum_mid"] - ev["sum_yes"]
    ev = ev[ev.n >= 5]
    print(f"\n== Basket overround (events with >=5 priced markets): {len(ev)} events ==")
    print(f"  mean sum(mid)-sum(YES) per event: {ev.overround.mean():+.3f} "
          f"(se {ev.overround.std(ddof=1)/np.sqrt(len(ev)):.3f}) "
          f"| mean n/event {ev.n.mean():.1f}")
    print(ev.groupby(ev.series.str.slice(0, 14)).overround.agg(["count", "mean"])
          .sort_values("count", ascending=False).head(10).to_string())

    # Economic: buy the full NO basket per event, executable, fees
    pnl_basket = []
    for et, g in d.groupby("event_ticker"):
        if len(g) < 5:
            continue
        pnl = sum(pnl_no_cents(r.yes_bid_close, r.result) for r in g.itertuples())
        pnl_basket.append((et, pnl / len(g), g.series.iloc[0]))
    b = pd.DataFrame(pnl_basket, columns=["event", "pnl_pc", "series"])
    bs = cluster_bootstrap(b.pnl_pc.values, b.event.values)
    print(f"\n  NO-basket everything: mean={bs['mean']:+.2f}c/contract "
          f"CI=[{bs['ci_lo']:+.2f},{bs['ci_hi']:+.2f}] p(<=0)={bs['p_le_0']:.4f} "
          f"events={bs['n_clusters']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry-hours", type=float, default=24)
    a = ap.parse_args()
    run(a.entry_hours)
