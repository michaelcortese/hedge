#!/usr/bin/env python3
"""Book-executable evaluation of the pre-registered in-event NO-carry rule.

Same signal moments as research_prereg_eval.py (last tape trade with
tau in (0.25, 0.40], unresolved, yes-trade-price in [30,70]) but P&L is
computed against the REAL minute-book:

  taker: buy NO at no_ask = 100 - yes_bid(minute at/just before moment)
  maker: post NO bid at 100 - yes_ask(minute); count filled iff some later
         in-window tape trade prints at yes_price >= that yes_ask (a YES
         taker lifted through our level'ish proxy); else no trade.

Cluster bootstrap by event. Scope oos/formation/all as before.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from research_mentions_lib import (  # noqa: E402
    DATA, cluster_bootstrap, load_markets, load_trades, taker_fee_cents)
from research_inevent_hazard import build_event_frame  # noqa: E402
from research_prereg_eval import TAU, TAU_LO, P_LO, P_HI  # noqa: E402


def load_minute() -> pd.DataFrame:
    rows = []
    f = DATA / "minute_candles.jsonl"
    for line in f.open():
        rec = json.loads(line)
        for c in rec.get("candles") or []:
            ybid = c.get("yes_bid") or {}
            yask = c.get("yes_ask") or {}

            def cents(d, k):
                v = d.get(k + "_dollars")
                return float(v) * 100.0 if v is not None else None

            rows.append({
                "ticker": rec["ticker"],
                "ts": pd.Timestamp(c.get("end_period_ts"), unit="s", tz="UTC"),
                "yes_bid": cents(ybid, "close"),
                "yes_ask": cents(yask, "close"),
            })
    return pd.DataFrame(rows)


def signal_moments(mode: str) -> pd.DataFrame:
    tr = load_trades()
    mk = build_event_frame(load_markets(), mode=mode, trades=tr)
    tr = tr.merge(mk[["ticker", "event_ticker", "series", "result",
                      "event_start", "event_end", "mention_tau"]],
                  on="ticker", how="inner")
    tr["tau"] = ((tr.ts - tr.event_start).dt.total_seconds()
                 / (tr.event_end - tr.event_start).dt.total_seconds())
    w = tr[(tr.tau > TAU_LO) & (tr.tau <= TAU)].dropna(subset=["yes_price"])
    last = (w.sort_values("ts").groupby("ticker").agg(
        p_tau=("yes_price", "last"), moment=("ts", "last"),
        event=("event_ticker", "first"), series=("series", "first"),
        result=("result", "first"), mtau=("mention_tau", "first"),
        event_end=("event_end", "first")).reset_index())
    last = last[(last.result == 0) | (last.mtau > TAU)]
    return last[(last.p_tau >= P_LO) & (last.p_tau <= P_HI)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=["oos", "formation", "all"], default="oos")
    ap.add_argument("--mode", choices=["expost", "onset"], default="expost")
    ap.add_argument("--max-no-cost", type=float, default=75.0,
                    help="skip if crossing costs more than this (spread blowout)")
    args = ap.parse_args()
    formation = set(json.load((DATA / "formation_events.json").open()))
    sig = signal_moments(args.mode)
    if args.scope == "oos":
        sig = sig[~sig.event.isin(formation)]
    elif args.scope == "formation":
        sig = sig[sig.event.isin(formation)]
    mc = load_minute()
    tr_all = load_trades()

    rows_t, rows_m = [], []
    n_nobook = n_wide = 0
    for r in sig.itertuples():
        g = mc[(mc.ticker == r.ticker) & (mc.ts <= r.moment)]
        g = g.dropna(subset=["yes_bid", "yes_ask"])
        g = g[g.ts >= r.moment - pd.Timedelta(minutes=10)]
        if not len(g):
            n_nobook += 1
            continue
        book = g.sort_values("ts").iloc[-1]
        # ---- taker: cross to NO ask
        no_cost = 100.0 - book.yes_bid
        if book.yes_bid >= 1 and no_cost <= args.max_no_cost:
            pnl = (100.0 if r.result == 0 else 0.0) - no_cost \
                - taker_fee_cents(no_cost)
            rows_t.append((r.event, r.series, r.ticker, no_cost, pnl))
        else:
            n_wide += 1
        # ---- maker: post NO bid at 100 - yes_ask; filled iff a later
        # in-window trade prints at yes_price >= yes_ask (proxy)
        if book.yes_ask <= 99:
            later = tr_all[(tr_all.ticker == r.ticker)
                           & (tr_all.ts > r.moment)
                           & (tr_all.ts <= r.event_end)]
            filled = bool(len(later[later.yes_price >= book.yes_ask]))
            if filled:
                no_cost_m = 100.0 - book.yes_ask
                pnl_m = (100.0 if r.result == 0 else 0.0) - no_cost_m  # maker free
                rows_m.append((r.event, r.series, r.ticker, no_cost_m, pnl_m))

    for name, rows in (("TAKER (cross NO ask)", rows_t),
                       ("MAKER (post at 100-yes_ask, fill-proxy)", rows_m)):
        t = pd.DataFrame(rows, columns=["event", "series", "ticker", "cost", "pnl"])
        print(f"\n== {name} | scope={args.scope} mode={args.mode} ==")
        print(f"  signals={len(sig)} nobook={n_nobook} spread-skip={n_wide} "
              f"trades={len(t)}")
        if len(t) < 10:
            print("  too few")
            continue
        bs = cluster_bootstrap(t.pnl.values, t.event.values)
        print(f"  mean={bs['mean']:+.2f}c CI95=[{bs['ci_lo']:+.2f},{bs['ci_hi']:+.2f}] "
              f"p(<=0)={bs['p_le_0']:.4f} n={bs['n']} events={bs['n_clusters']} "
              f"avg_cost={t.cost.mean():.1f}c")
        print(t.groupby("series").agg(n=("pnl", "size"), mean=("pnl", "mean"))
              .sort_values("n", ascending=False).head(10).to_string())


if __name__ == "__main__":
    main()
