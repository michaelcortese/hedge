"""Adversarial statistics audit of the family-2b taker-NO rule.

Attacks: multiplicity, temporal stability (month/week), drop-top-events,
day-clustered bootstrap, last-print-in-window selection (fixed-entry-time
causal variant), staleness, composition (World Cup share).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from research_tape_micro import build_dataset, cluster_bootstrap  # noqa: E402
from research_mentions_lib import taker_fee_cents  # noqa: E402


def net_pnl(g: pd.DataFrame) -> np.ndarray:
    entry = (100.0 - g.yes_price + 4.0).clip(1, 99)
    fee = entry.map(lambda p: taker_fee_cents(p))
    return (np.where(g.result == 0, 100.0, 0.0) - entry - fee).astype(float)


def report(tag: str, g: pd.DataFrame, cluster_col: str = "event_ticker"):
    if len(g) < 5:
        print(f"  {tag}: n={len(g)} (too small)")
        return None
    st = cluster_bootstrap(net_pnl(g), g[cluster_col].to_numpy())
    print(f"  {tag}: n={st['n']} ({st['n_clusters']} cl) "
          f"YES={g.result.mean():.3f} px={g.yes_price.mean():.1f}c "
          f"net={st['mean']:+.2f} CI=[{st['ci_lo']:+.2f},{st['ci_hi']:+.2f}] "
          f"p={st['p_le_0']:.4f}")
    return st


def main():
    mk, tr, j = build_dataset()
    print(f"dataset: {len(mk)} settled markets / {mk.event_ticker.nunique()} events; "
          f"{len(j)} joined trades")

    jj = j.join(j.groupby("event_ticker").close_time.max().rename("ev_end"),
                on="event_ticker")
    jj["mte"] = (jj.ev_end - jj.ts).dt.total_seconds() / 60.0

    # ---- reproduce headline cell
    w = jj[(jj.mte >= 60) & (jj.mte < 180)]
    last = w.sort_values("ts").groupby("ticker").tail(1)
    g0 = last[last.yes_price.between(20, 80)].copy()
    print("\n=== REPRODUCE headline [60,180) 20-80c ===")
    report("headline", g0)

    # ---- composition
    g0["ser"] = g0["ticker"].str.split("-").str[0]
    print("\n=== composition by series (n, events, mean net) ===")
    comp = g0.assign(pnl=net_pnl(g0)).groupby("ser").agg(
        n=("ticker", "size"), ev=("event_ticker", "nunique"),
        yes=("result", "mean"), mean_net=("pnl", "mean")).sort_values(
        "n", ascending=False)
    print(comp.round(2).to_string())
    tot = net_pnl(g0).sum()
    comp2 = g0.assign(pnl=net_pnl(g0)).groupby("ser")["pnl"].sum() / tot
    print("share of total net pnl:\n", comp2.round(3).sort_values(
        ascending=False).to_string())

    # exclude World Cup + speech
    for drop in (["KXWCMENTION"], ["KXWCMENTION", "KXWCGAMEMENTION"],):
        gg = g0[~g0.ser.isin(drop)]
        report(f"ex-{drop}", gg)
    gsp = g0[g0.family != "speech"]
    report("ex-speech (the claim)", gsp)
    gsp_wc = gsp[~gsp.ser.str.contains("WC")]
    report("ex-speech ex-WC*", gsp_wc)

    # ---- temporal stability: by settlement (ev_end) month and week
    print("\n=== by calendar month of ev_end ===")
    g0["month"] = g0.ev_end.dt.to_period("M").astype(str)
    for m_, gg in g0.groupby("month"):
        report(f"month {m_}", gg)
    print("\n=== by ISO week of ev_end ===")
    g0["week"] = g0.ev_end.dt.strftime("%G-W%V")
    for wk, gg in g0.groupby("week"):
        report(f"week {wk}", gg)
    # last-30-days vs before
    cut = g0.ev_end.max() - pd.Timedelta(days=30)
    report("last 30 days", g0[g0.ev_end > cut])
    report("before that", g0[g0.ev_end <= cut])

    # ---- drop-top-events
    print("\n=== drop-top-events (by event total pnl contribution) ===")
    g0["pnl"] = net_pnl(g0)
    ev_pnl = g0.groupby("event_ticker")["pnl"].sum().sort_values(ascending=False)
    print("top 8 events by pnl contribution:")
    print(ev_pnl.head(8).round(1).to_string())
    for k in (1, 3, 5, 10):
        keep = g0[~g0.event_ticker.isin(ev_pnl.head(k).index)]
        report(f"drop top {k} events", keep)
    # drop top DAYS
    g0["day"] = g0.ev_end.dt.date
    day_pnl = g0.groupby("day")["pnl"].sum().sort_values(ascending=False)
    print("top 5 days:", {str(k): round(v, 1) for k, v in
                          day_pnl.head(5).items()})
    for k in (1, 3, 5):
        keep = g0[~g0.day.isin(day_pnl.head(k).index)]
        report(f"drop top {k} days", keep)

    # ---- day-clustered bootstrap
    print("\n=== cluster by DAY instead of event ===")
    report("day-clustered", g0, cluster_col="day")
    g0["ser_day"] = g0.ser + "_" + g0.day.astype(str)
    report("series-x-day clustered", g0, cluster_col="ser_day")

    # ---- staleness of the signal print
    print("\n=== staleness: age of last print (mte - 60 = how far into window) ===")
    g0["print_age_at_wend"] = g0.mte - 60.0  # minutes before window end
    for lo, hi in [(0, 30), (30, 60), (60, 120)]:
        gg = g0[g0.print_age_at_wend.between(lo, hi)]
        report(f"last print {lo}-{hi}min before window-end", gg)

    # markets that CLOSED during/before the window (resolved during window)
    closed_early = g0[g0.close_time < g0.ev_end - pd.Timedelta(minutes=60)]
    print(f"\nmarkets in cell whose close_time < ev_end-60min: "
          f"{len(closed_early)} (YES rate {closed_early.result.mean() if len(closed_early) else float('nan'):.3f})")
    open_at_entry = g0[g0.close_time >= g0.ev_end - pd.Timedelta(minutes=60)]
    report("only markets still open at ev_end-60 (live-enterable)", open_at_entry)

    # ---- fixed-entry-time causal variants: at T min before ev_end,
    # use most recent print (any age <= max_age), require 20-80c, market open.
    print("\n=== fixed-entry-time variants (enter exactly T min before ev_end) ===")
    ev_end_map = jj.groupby("ticker").ev_end.first()
    close_map = jj.groupby("ticker").close_time.first()
    res_map = jj.groupby("ticker").result.first()
    evt_map = jj.groupby("ticker").event_ticker.first()
    fam_map = jj.groupby("ticker").family.first()
    for T in (180, 120, 60):
        for max_age in (120, 1e9):
            entry_ts = jj.ev_end - pd.Timedelta(minutes=T)
            pre = jj[jj.ts <= entry_ts]
            lastp = pre.sort_values("ts").groupby("ticker").tail(1).set_index("ticker")
            lastp["age_min"] = (
                (lastp.ev_end - pd.Timedelta(minutes=T)) - lastp.ts
            ).dt.total_seconds() / 60.0
            sel = lastp[(lastp.age_min <= max_age)
                        & lastp.yes_price.between(20, 80)]
            # market must still be open at entry time
            sel = sel[sel.close_time > sel.ev_end - pd.Timedelta(minutes=T)]
            sel = sel.reset_index()
            tag = f"T={T}min max_print_age={'inf' if max_age > 1e6 else int(max_age)}m"
            report(tag, sel)
            if T == 60 and max_age > 1e6:
                sel["fam"] = sel.ticker.map(fam_map)
                report("  ..ex-speech", sel[sel.fam != "speech"])

    # ---- multiplicity accounting: count of tested cells in the study
    print("\n=== multiplicity ===")
    print("see log: 64 CI cells in tape_micro_full.log + ~12 family2b cells")


if __name__ == "__main__":
    main()
