"""Adversarial statistics audit of the family-2b taker-NO theta-carry rule.

Attacks: reproduction, calendar-month/week stability, drop-top-events,
day-clustered bootstrap, event-weighted means, series composition,
multiplicity discount, last-print staleness / selection diagnostics.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from research_mentions_lib import taker_fee_cents  # noqa: E402
from research_tape_micro import build_dataset, cluster_bootstrap  # noqa: E402


def boot(net, cl, seed=7):
    st = cluster_bootstrap(np.asarray(net, float), np.asarray(cl), seed=seed)
    return st


def fmt(st, extra=""):
    return (f"n={st['n']:>5} ({st['n_clusters']:>3} cl) "
            f"mean={st['mean']:+6.2f} CI=[{st['ci_lo']:+6.2f},{st['ci_hi']:+6.2f}] "
            f"p(<=0)={st['p_le_0']:.4f} {extra}")


def main():
    mk, tr, j = build_dataset()
    jj = j.join(j.groupby("event_ticker").close_time.max().rename("ev_end"),
                on="event_ticker")
    jj["mte"] = (jj.ev_end - jj.ts).dt.total_seconds() / 60.0

    # ---- entries: last print in [60,180) pre event-end, 20-80c
    w = jj[(jj.mte >= 60) & (jj.mte < 180)]
    last = w.sort_values("ts").groupby("ticker").tail(1)
    g = last[last.yes_price.between(20, 80)].copy()
    entry = (100.0 - g.yes_price + 4.0).clip(1, 99)
    fee = entry.map(lambda p: taker_fee_cents(p))
    g["net"] = (np.where(g.result == 0, 100.0, 0.0) - entry - fee).astype(float)
    g["ev_end_date"] = g.ev_end.dt.date
    g["month"] = g.ev_end.dt.to_period("M").astype(str)
    g["week"] = g.ev_end.dt.to_period("W").astype(str)

    print("=" * 78)
    print("A. REPRODUCTION (claim: n=1111, 185 ev, +9.63 [6.47,13.00])")
    st = boot(g.net, g.event_ticker)
    print("  ", fmt(st, f"YES rate={g.result.mean():.3f} "
                        f"mean px={g.yes_price.mean():.1f}c"))

    print("=" * 78)
    print("B. STABILITY — by calendar month of event end")
    for m, gg in g.groupby("month"):
        if len(gg) < 5:
            print(f"  {m}: n={len(gg)} (too few)")
            continue
        print(f"  {m}: ", fmt(boot(gg.net, gg.event_ticker),
                              f"YES={gg.result.mean():.3f} px={gg.yes_price.mean():.1f}"))
    print("\n  by ISO week:")
    for wk, gg in sorted(g.groupby("week")):
        st2 = (boot(gg.net, gg.event_ticker) if len(gg) >= 5 else None)
        if st2:
            print(f"  {wk}: ", fmt(st2))
        else:
            print(f"  {wk}: n={len(gg)} mean={gg.net.mean():+.2f}")
    # split halves by event-end date
    med = g.ev_end.median()
    for lbl, gg in [("first half", g[g.ev_end <= med]),
                    ("second half", g[g.ev_end > med])]:
        print(f"  {lbl} (split {med.date()}): ",
              fmt(boot(gg.net, gg.event_ticker)))
    # last N days
    for days in (14, 21, 30):
        cut = g.ev_end.max() - pd.Timedelta(days=days)
        gg = g[g.ev_end > cut]
        print(f"  last {days}d: ", fmt(boot(gg.net, gg.event_ticker)))

    print("=" * 78)
    print("C. DROP-TOP-EVENTS (by total positive contribution)")
    contrib = g.groupby("event_ticker").net.sum().sort_values(ascending=False)
    print("  top 10 events by P&L contribution:")
    for evt, c in contrib.head(10).items():
        n_ev = (g.event_ticker == evt).sum()
        print(f"    {evt}: sum={c:+8.1f}c over {n_ev} mkts")
    for k in (1, 3, 5, 10, 20):
        keep = g[~g.event_ticker.isin(contrib.head(k).index)]
        print(f"  drop top {k:>2}: ", fmt(boot(keep.net, keep.event_ticker)))

    print("=" * 78)
    print("D. CLUSTER BY DAY instead of event")
    st = boot(g.net, g.ev_end_date.astype(str))
    print("  day-clustered:  ", fmt(st))
    st = boot(g.net, g.week)
    print("  week-clustered: ", fmt(st))

    print("=" * 78)
    print("E. EVENT-WEIGHTED (mean of event means; equal weight per event)")
    evm = g.groupby("event_ticker").net.mean()
    # simple iid bootstrap over events
    rng = np.random.default_rng(7)
    draws = rng.choice(evm.to_numpy(), size=(4000, len(evm)))
    means = draws.mean(axis=1)
    print(f"  n_events={len(evm)} mean-of-means={evm.mean():+.2f} "
          f"CI=[{np.percentile(means, 2.5):+.2f},{np.percentile(means, 97.5):+.2f}] "
          f"p(<=0)={(means <= 0).mean():.4f}")
    print(f"  median event mean={evm.median():+.2f}; "
          f"share of events with mean>0: {(evm > 0).mean():.2f}")

    print("=" * 78)
    print("F. COMPOSITION — by series")
    g["series2"] = g.ticker.str.split("-").str[0]
    for s, gg in sorted(g.groupby("series2"), key=lambda kv: -len(kv[1])):
        if len(gg) < 10:
            continue
        print(f"  {s:<20}", fmt(boot(gg.net, gg.event_ticker)))
    wc = g[g.series2 == "KXWCMENTION"]
    nonwc = g[g.series2 != "KXWCMENTION"]
    print(f"  WC share of markets: {len(wc)/len(g):.2f}, "
          f"of total P&L: {wc.net.sum()/g.net.sum():.2f}")
    print("  ex-WorldCup:        ", fmt(boot(nonwc.net, nonwc.event_ticker)))
    nonwc_mlb = g[~g.series2.isin(["KXWCMENTION", "KXMLBMENTION"])]
    print("  ex-WC & ex-MLB:     ",
          fmt(boot(nonwc_mlb.net, nonwc_mlb.event_ticker)))
    print("\n  by month x (WC vs rest):")
    for m, gm in g.groupby("month"):
        for lbl, gg in [("WC ", gm[gm.series2 == "KXWCMENTION"]),
                        ("rest", gm[gm.series2 != "KXWCMENTION"])]:
            if len(gg) < 5:
                continue
            print(f"    {m} {lbl}: ", fmt(boot(gg.net, gg.event_ticker)))

    print("=" * 78)
    print("G. SELECTION / STALENESS diagnostics")
    g["stale_min"] = g.mte - 60.0  # staleness at a decision made at end-60
    print("  entry-print mte distribution (min before event end):")
    print(g.mte.describe(percentiles=[.1, .25, .5, .75, .9]).round(1).to_string())
    # any later prints on the same market after the entry print?
    later = jj[(jj.mte < 60) & (jj.mte >= 0)]
    has_later = g.ticker.isin(later.ticker.unique())
    print(f"  entries with ANY print in final 60min: {has_later.mean():.2f}")
    ge, gl = g[~has_later], g[has_later]
    print("   no-later-print subset: ", fmt(boot(ge.net, ge.event_ticker)))
    print("   has-later-print subset:", fmt(boot(gl.net, gl.event_ticker)))
    # for markets with later prints: what does the FIRST post-window print say?
    fp = later.sort_values("ts").groupby("ticker").head(1)[
        ["ticker", "yes_price"]].rename(columns={"yes_price": "next_px"})
    gm = g.merge(fp, on="ticker", how="left")
    moved = gm.dropna(subset=["next_px"])
    print(f"  first post-window print vs entry print: mean diff="
          f"{(moved.next_px - moved.yes_price).mean():+.2f}c, "
          f"median {(moved.next_px - moved.yes_price).median():+.2f}c")
    # re-price entries at the first post-window print where available (proxy
    # for the true price at decision time end-60): stale-quote sensitivity
    gm["px2"] = gm.next_px.fillna(gm.yes_price)
    e2 = (100.0 - gm.px2 + 4.0).clip(1, 99)
    f2 = e2.map(lambda p: taker_fee_cents(p))
    n2 = (np.where(gm.result == 0, 100.0, 0.0) - e2 - f2).astype(float)
    print("  repriced at first print AFTER end-60 (still 20-80 filter on old):")
    print("   ", fmt(boot(n2, gm.event_ticker)))
    # drop entries where the first post-window print exited 20-80 (mention
    # spike or collapse already visible at decision time)
    inband = gm[(gm.px2.between(20, 80))]
    e3 = (100.0 - inband.px2 + 4.0).clip(1, 99)
    f3 = e3.map(lambda p: taker_fee_cents(p))
    n3 = (np.where(inband.result == 0, 100.0, 0.0) - e3 - f3).astype(float)
    print("  repriced AND re-filtered 20-80 at decision time: ")
    print("   ", fmt(boot(n3, inband.event_ticker)))
    # markets already closed before decision time end-60 (cannot enter live)
    closed_before = g[g.close_time < (g.ev_end - pd.Timedelta(minutes=60))]
    print(f"  entries whose own market closed before end-60: "
          f"{len(closed_before)} (YES rate {closed_before.result.mean() if len(closed_before) else float('nan'):.2f})")
    open_only = g[g.close_time >= (g.ev_end - pd.Timedelta(minutes=60))]
    print("   excluding them:      ", fmt(boot(open_only.net,
                                               open_only.event_ticker)))

    print("=" * 78)
    print("H. MULTIPLICITY — normal-approx z and Bonferroni")
    st = boot(g.net, g.event_ticker)
    z = st["mean"] / st["se"]
    from scipy import stats as sps
    p1 = 1 - sps.norm.cdf(z)
    print(f"  headline z={z:.2f} one-sided normal p={p1:.2e}")
    for ncells in (9, 60, 500, 5000):
        print(f"  Bonferroni x{ncells}: adj p={min(1.0, p1*ncells):.2e}")


if __name__ == "__main__":
    main()
