"""Part 2: spread-aware entry repricing + decision-time causal variant.

Is the +9.63 an artifact of the flat +4c allowance in wide-spread series?
Reprice NO entries using per-series effective spreads and taker-side logic:
  - last print taker-NO  -> print IS the NO ask -> NO cost = 100 - p
  - last print taker-YES -> print at YES ask -> NO ask ~ (100 - p) + spread
Also: series-level correlation between measured edge and effective spread,
and a fully decision-time variant (decide at end-60, last print <=120min old).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from research_mentions_lib import taker_fee_cents  # noqa: E402
from research_tape_micro import build_dataset, cluster_bootstrap, family4  # noqa: E402


def boot(net, cl, seed=7):
    return cluster_bootstrap(np.asarray(net, float), np.asarray(cl), seed=seed)


def fmt(st):
    return (f"n={st['n']:>5} ({st['n_clusters']:>3} cl) "
            f"mean={st['mean']:+6.2f} CI=[{st['ci_lo']:+6.2f},{st['ci_hi']:+6.2f}] "
            f"p(<=0)={st['p_le_0']:.4f}")


def main():
    mk, tr, j = build_dataset()
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        m4, agg4, eff = family4(j)
    jj = j.join(j.groupby("event_ticker").close_time.max().rename("ev_end"),
                on="event_ticker")
    jj["mte"] = (jj.ev_end - jj.ts).dt.total_seconds() / 60.0

    w = jj[(jj.mte >= 60) & (jj.mte < 180)]
    last = w.sort_values("ts").groupby("ticker").tail(1)
    g = last[last.yes_price.between(20, 80)].copy()
    g["series2"] = g.ticker.str.split("-").str[0]
    g["spr"] = g.series2.map(lambda s: eff.get(s, 4.0))

    print("=" * 78)
    print("SPREAD-AWARE ENTRY REPRICING (per-series effective spread)")
    print(f"  spread among entries: mean={g.spr.mean():.2f}c "
          f"median={g.spr.median():.2f}c p90={g.spr.quantile(.9):.2f}c")
    variants = {
        "flat +4c (headline)": 4.0 * np.ones(len(g)),
        "full eff spread always": g.spr.to_numpy(),
        "side-aware (taker-NO print pays 0, taker-YES pays full spread)":
            np.where(g.taker_side == "yes", g.spr, 0.0),
        "side-aware half (NO pays half, YES pays full)":
            np.where(g.taker_side == "yes", g.spr, g.spr / 2),
        "flat +6c": 6.0 * np.ones(len(g)),
        "flat +8c": 8.0 * np.ones(len(g)),
    }
    for lbl, mark in variants.items():
        e = (100.0 - g.yes_price.to_numpy() + mark).clip(1, 99)
        fee = np.array([taker_fee_cents(x) for x in e])
        net = np.where(g.result == 0, 100.0, 0.0) - e - fee
        print(f"  {lbl:<58}", fmt(boot(net, g.event_ticker)))
    print(f"  taker-side of entry print: YES {(g.taker_side=='yes').mean():.2f}")

    print("=" * 78)
    print("SERIES EDGE vs SPREAD correlation (n>=20 series)")
    rows = []
    for s, gg in g.groupby("series2"):
        if len(gg) < 20:
            continue
        e = (100.0 - gg.yes_price + 4.0).clip(1, 99)
        fee = e.map(lambda p: taker_fee_cents(p))
        net = np.where(gg.result == 0, 100.0, 0.0) - e - fee
        rows.append({"series": s, "n": len(gg), "spr": gg.spr.iloc[0],
                     "edge4c": float(np.mean(net))})
    sdf = pd.DataFrame(rows)
    print(sdf.sort_values("spr").to_string(index=False))
    print(f"  corr(edge, spread) weighted by n: "
          f"{np.corrcoef(sdf.spr, sdf.edge4c)[0,1]:.2f} (unweighted)")

    print("=" * 78)
    print("DECISION-TIME CAUSAL VARIANT: at t*=end-60, use last print <=120min old,")
    print("require 20-80c at that print, side-aware spread entry, fee.")
    # last print at or before end-60, within lookback 120 min
    w2 = jj[(jj.mte >= 60) & (jj.mte < 180)]
    last2 = w2.sort_values("ts").groupby("ticker").tail(1)
    g2 = last2[last2.yes_price.between(20, 80)].copy()
    g2["series2"] = g2.ticker.str.split("-").str[0]
    g2["spr"] = g2.series2.map(lambda s: eff.get(s, 4.0))
    mark = np.where(g2.taker_side == "yes", g2.spr, 0.0)
    e = (100.0 - g2.yes_price.to_numpy() + mark).clip(1, 99)
    fee = np.array([taker_fee_cents(x) for x in e])
    net = np.where(g2.result == 0, 100.0, 0.0) - e - fee
    st = boot(net, g2.event_ticker)
    print("  side-aware:", fmt(st))
    # month stability of the side-aware variant
    g2 = g2.assign(net=net)
    g2["month"] = g2.ev_end.dt.tz_localize(None).dt.to_period("M").astype(str)
    for m, gg in g2.groupby("month"):
        print(f"    {m}: ", fmt(boot(gg.net, gg.event_ticker)))
    # drop top events under side-aware pricing
    contrib = g2.groupby("event_ticker").net.sum().sort_values(ascending=False)
    for k in (5, 10, 20):
        keep = g2[~g2.event_ticker.isin(contrib.head(k).index)]
        print(f"    drop top {k:>2} ev:", fmt(boot(keep.net, keep.event_ticker)))
    # ex the two possibly-dying supplies
    core = g2[~g2.series2.isin(["KXWCMENTION", "KXMLBMENTION"])]
    print("    ex-WC/MLB:    ", fmt(boot(core.net, core.event_ticker)))


if __name__ == "__main__":
    main()
