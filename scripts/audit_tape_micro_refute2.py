"""Generalist-refuter round-2 supplement (fresh eyes).

G. Reconcile headline stats with the INDEPENDENT bootstrap in
   research_mentions_lib.cluster_bootstrap (different implementation).
H. ev_end anchor construction: (a) computed from taped markets only vs all
   settled markets in events.jsonl; (b) all-YES events where max close_time
   is the last mention, not a schedule -> quantify share + P&L contribution.
I. Anchor-error sensitivity: shift assumed event end by +/-30/60 min.
J. Live-superset variant: at T=ev_end-60min take last print of ANY age
   (drop the implicit freshness filter) -> does the edge survive?
K. Dataset hygiene: dup tickers across events; trades with null trade_id.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_mentions_lib as lib  # noqa: E402
from research_tape_micro import build_dataset, cluster_bootstrap  # noqa: E402
from research_mentions_lib import taker_fee_cents  # noqa: E402


def net_no(g, entry_no):
    entry = np.clip(entry_no, 1, 99)
    fee = np.array([taker_fee_cents(e) for e in entry])
    return np.where(g.result.to_numpy() == 0, 100.0, 0.0) - entry - fee


def line(lbl, g, entry_no, bs=cluster_bootstrap):
    if len(g) < 5:
        print(f"  {lbl}: n={len(g)} too few")
        return
    net = net_no(g, entry_no)
    st = bs(net, g.event_ticker.to_numpy())
    print(f"  {lbl}: n={st['n']} ({st['n_clusters']} ev) YES={g.result.mean():.3f} "
          f"net={st['mean']:+.2f} CI=[{st['ci_lo']:+.2f},{st['ci_hi']:+.2f}] "
          f"p={st['p_le_0']:.4f}")


def select(jj, endcol="ev_end", lo=60, hi=180):
    mte = (jj[endcol] - jj.ts).dt.total_seconds() / 60.0
    w = jj[(mte >= lo) & (mte < hi)]
    last = w.sort_values("ts").groupby("ticker").tail(1)
    return last[last.yes_price.between(20, 80)].copy()


def main():
    mk, tr, j = build_dataset()
    del tr

    print("K. dataset hygiene")
    raw = lib.load_markets()
    print(f"  events.jsonl settled rows: {len(raw)}; unique tickers: "
          f"{raw.ticker.nunique()}; dup ticker rows: {len(raw)-raw.ticker.nunique()}")
    dup = raw[raw.ticker.duplicated(keep=False)]
    if len(dup):
        both = dup.groupby("ticker").agg(n_ev=("event_ticker", "nunique"),
                                         res=("result", "nunique"))
        print(f"  dup tickers spanning >1 event: {(both.n_ev > 1).sum()}, "
              f"conflicting results: {(both.res > 1).sum()}")

    jj = j.join(j.groupby("event_ticker").close_time.max().rename("ev_end"),
                on="event_ticker")

    g0 = select(jj)
    entry0 = (100.0 - g0.yes_price + 4.0).to_numpy()

    print("\nG. reconcile with research_mentions_lib.cluster_bootstrap "
          "(independent implementation, 10k reps)")
    line("headline via lib bootstrap", g0, entry0, bs=lib.cluster_bootstrap)
    # and lib's pnl_no_cents convention on the same entries
    pl = np.array([lib.pnl_no_cents(100.0 - e, r) for e, r in
                   zip(np.clip(entry0, 1, 99), g0.result)])
    print(f"  lib.pnl_no_cents mean on same entries: {pl.mean():+.2f} "
          f"(script net mean {net_no(g0, entry0).mean():+.2f})")

    print("\nH. ev_end anchor construction")
    # (a) recompute ev_end from ALL settled markets (events.jsonl), not taped
    mk_end = raw.groupby("event_ticker").close_time.max().rename("ev_end_mk")
    jj2 = jj.join(mk_end, on="event_ticker")
    diff = (jj2.groupby("event_ticker")
            .agg(a=("ev_end", "first"), b=("ev_end_mk", "first")))
    dmin = (diff.b - diff.a).dt.total_seconds() / 60.0
    print(f"  events where mk-based end differs from tape-based: "
          f"{(dmin.abs() > 1).sum()} / {len(diff)}; "
          f"max shift {dmin.abs().max():.0f} min")
    ga = select(jj2, endcol="ev_end_mk")
    line("headline with mk-based ev_end", ga,
         (100.0 - ga.yes_price + 4.0).to_numpy())
    # (b) all-YES events: anchor is last mention, not a schedule
    evres = mk.groupby("event_ticker").result.agg(["mean", "count"])
    all_yes = set(evres[evres["mean"] == 1.0].index)
    g0["all_yes_ev"] = g0.event_ticker.isin(all_yes)
    ay = g0[g0.all_yes_ev]
    ny = g0[~g0.all_yes_ev]
    tot = net_no(g0, entry0).sum()
    print(f"  all-YES events among signals: {ay.event_ticker.nunique()} ev, "
          f"n={len(ay)}, pnl share={net_no(ay, (100.0-ay.yes_price+4.0).to_numpy()).sum()/tot:+.1%}")
    line("signals in NOT-all-YES events (anchor = a real scheduled halt)",
         ny, (100.0 - ny.yes_price + 4.0).to_numpy())

    print("\nI. anchor-error sensitivity (shift assumed end, window follows)")
    for shift in (-60, -30, 30, 60):
        jj3 = jj.copy()
        jj3["ev_end_s"] = jj3.ev_end + pd.Timedelta(minutes=shift)
        gs = select(jj3, endcol="ev_end_s")
        line(f"assumed end {shift:+d} min", gs,
             (100.0 - gs.yes_price + 4.0).to_numpy())

    print("\nJ. live-superset at T=ev_end-60: last print of ANY age, 20-80c,"
          " market open at T")
    T = jj.ev_end - pd.Timedelta(minutes=60)
    before = jj[jj.ts <= T]
    last = before.sort_values("ts").groupby("ticker").tail(1)
    last["T"] = last.ev_end - pd.Timedelta(minutes=60)
    gl = last[(last.yes_price.between(20, 80))
              & (last.close_time > last["T"])].copy()
    age = (gl["T"] - gl.ts).dt.total_seconds() / 60.0
    line("ALL ages", gl, (100.0 - gl.yes_price + 4.0).to_numpy())
    line("age > 120 min (the prints the backtest excluded)", gl[age > 120],
         (100.0 - gl[age > 120].yes_price + 4.0).to_numpy())
    for amax in (360, 720):
        sub = gl[age <= amax]
        line(f"age <= {amax} min", sub,
             (100.0 - sub.yes_price + 4.0).to_numpy())


if __name__ == "__main__":
    main()
