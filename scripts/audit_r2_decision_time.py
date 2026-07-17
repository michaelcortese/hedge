"""Round-2 audit (own test): decompose the family-2b edge.

Question: with a PERFECT schedule anchor (ex-post ev_end, the best case any
live broadcast schedule could achieve), does the rule survive when entries
happen at a fixed decision clock time T instead of retro-selecting "the last
print in [60,180) before end"?

Published rule = last print per market in the window. Live, at time T you see
the current book/last print; you cannot know it will remain the last print.
If decision-time entry kills the edge, the published number is an artifact of
conditioning on trading having stopped (dead quotes), not tradable theta.

Grid: T = ev_end - {60, 90, 120, 150} min; lookback L = {15, 30, 60, 120} min
for the most recent print before T; 20-80c; market open at T; entry
(100-p)+4c, taker fee; hold to settlement. Cluster bootstrap by event.
Also: staleness of the quote at T for winners vs losers, and a variant
entering ONCE per market at the FIRST qualifying (T,L=30) among the T grid.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from research_mentions_lib import load_markets, taker_fee_cents  # noqa: E402
from research_tape_micro import cluster_bootstrap, load_trades_full  # noqa: E402

FEE = np.vectorize(lambda p: taker_fee_cents(min(max(p, 1), 99)))


def stat_line(g, label):
    if len(g) < 15:
        print(f"  {label}: n={len(g)} insufficient")
        return
    entry = (100.0 - g.yes_price + 4.0).clip(1, 99)
    net = (np.where(g.result == 0, 100.0, 0.0) - entry - FEE(entry)).astype(float)
    st = cluster_bootstrap(net, g.event_ticker.to_numpy())
    print(f"  {label}: n={st['n']} ({st['n_clusters']} ev) "
          f"YES={g.result.mean():.3f} px={g.yes_price.mean():.1f} "
          f"net={st['mean']:+.2f} CI=[{st['ci_lo']:+.2f},{st['ci_hi']:+.2f}] "
          f"p={st['p_le_0']:.4f}")
    return net


def main():
    mk = load_markets().drop_duplicates(subset=["ticker"])
    tr = load_trades_full()
    j = tr.merge(mk[["ticker", "result", "event_ticker", "series",
                     "close_time"]], on="ticker", how="inner")
    ev_end = j.groupby("event_ticker").close_time.max().rename("ev_end")
    j = j.join(ev_end, on="event_ticker")
    j = j.sort_values(["ticker", "ts"], kind="stable")

    print("== decision-time entries, PERFECT (ex-post) schedule anchor ==")
    for tmin in (60, 90, 120, 150):
        for L in (15, 30, 60, 120):
            T = j.ev_end - pd.Timedelta(minutes=tmin)
            w = j[(j.ts < T) & (j.ts >= T - pd.Timedelta(minutes=L))]
            last = w.groupby("ticker").tail(1)
            # market must still be open (tradable) at decision time
            last = last[last.close_time > last.ev_end
                        - pd.Timedelta(minutes=tmin)]
            g = last[last.yes_price.between(20, 80)]
            stat_line(g, f"T=end-{tmin:>3}m, lookback {L:>3}m")
        print()

    # FIRST qualifying print in the window: fully causal at entry given a
    # perfect schedule (when a 20-80c print appears in [end-180,end-60),
    # buy NO immediately). Compare vs the published LAST-print selection.
    print("== first vs last print in [60,180) window, ex-post anchor ==")
    mte0 = (j.ev_end - j.ts).dt.total_seconds() / 60.0
    w0 = j[(mte0 >= 60) & (mte0 < 180) & (j.ts <= j.close_time)]
    band = w0[w0.yes_price.between(20, 80)]
    stat_line(band.groupby("ticker").head(1),
              "FIRST 20-80c print in window (implementable)")
    stat_line(band.groupby("ticker").tail(1),
              "LAST 20-80c print in window")
    lastw0 = w0.groupby("ticker").tail(1)
    stat_line(lastw0[lastw0.yes_price.between(20, 80)],
              "published: last print in window, then 20-80c filter")

    # staleness structure at T = end-90, L=120
    tmin, L = 90, 120
    T = j.ev_end - pd.Timedelta(minutes=tmin)
    w = j[(j.ts < T) & (j.ts >= T - pd.Timedelta(minutes=L))]
    last = w.groupby("ticker").tail(1)
    last = last[last.close_time > last.ev_end - pd.Timedelta(minutes=tmin)]
    g = last[last.yes_price.between(20, 80)].copy()
    g["stale_at_T_min"] = (
        (g.ev_end - pd.Timedelta(minutes=tmin)) - g.ts
    ).dt.total_seconds() / 60.0
    print("== staleness of quote at decision time (T=end-90m, L=120m) ==")
    for lab, mask in [("winners (result=NO)", g.result == 0),
                      ("losers (result=YES)", g.result == 1)]:
        gg = g[mask]
        print(f"  {lab}: n={len(gg)} stale-at-T median="
              f"{gg.stale_at_T_min.median():.1f}m "
              f"q75={gg.stale_at_T_min.quantile(.75):.1f}m")
    stat_line(g[g.stale_at_T_min <= 10], "fresh (<=10m) quotes at T")
    stat_line(g[g.stale_at_T_min > 10], "stale (>10m) quotes at T")

    # published-rule entries: how many had ANY later print in the window
    # (i.e., could you have known it was the 'last' print?)
    mte = (j.ev_end - j.ts).dt.total_seconds() / 60.0
    w2 = j[(mte >= 60) & (mte < 180)]
    lastw = w2.groupby("ticker").tail(1)
    gpub = lastw[lastw.yes_price.between(20, 80)].copy()
    gpub["mins_before_end"] = (
        (gpub.ev_end - gpub.ts).dt.total_seconds() / 60.0)
    print("\n== published-rule entry timing (mins before TRUE end) ==")
    print(gpub.mins_before_end.describe(
        percentiles=[.1, .25, .5, .75, .9]).round(1).to_string())
    # any subsequent print before own close (tape resumed after entry)?
    nxt = j.groupby("ticker").ts.max().rename("last_tape_ts")
    gpub = gpub.join(nxt, on="ticker")
    resumed = (gpub.last_tape_ts > gpub.ts)
    print(f"  entries where tape RESUMED after the entry print: "
          f"{resumed.mean():.1%}")
    for lab, mask in [("tape resumed later", resumed),
                      ("entry print was final print ever", ~resumed)]:
        stat_line(gpub[mask], lab)


if __name__ == "__main__":
    main()
