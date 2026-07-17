"""Audit follow-up: (a) rule entries split by last-print taker side,
(b) entry realism vs subsequent REAL taker-NO prints, (c) rescue attempts
for a causal anchor: stricter onset thresholds + walk-forward schedule-prior
(per-series median end time-of-day)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from research_mentions_lib import load_markets, taker_fee_cents  # noqa: E402
from research_tape_micro import cluster_bootstrap, load_trades_full  # noqa: E402
from research_inevent_hazard import onset_windows  # noqa: E402
from audit_causal_anchor import (  # noqa: E402
    FEE, WINDOWS_EXT, last_print_in_window, rule_pnl, series_duration)


def main():
    mk = load_markets().drop_duplicates(subset=["ticker"])
    tr = load_trades_full()
    j = tr.merge(mk[["ticker", "result", "event_ticker", "series", "family",
                     "close_time"]], on="ticker", how="inner")
    ev_end = j.groupby("event_ticker").close_time.max().rename("ev_end")
    j = j.join(ev_end, on="event_ticker")

    last = last_print_in_window(j, "ev_end", 60, 180)
    g = last[last.yes_price.between(20, 80)].copy()
    entry = (100.0 - g.yes_price + 4.0).clip(1, 99)
    g["net"] = (np.where(g.result == 0, 100.0, 0.0) - entry
                - FEE(entry)).astype(float)

    print("== (a) rule entries by taker side of the LAST print ==")
    for side, gg in g.groupby("taker_side"):
        st = cluster_bootstrap(gg.net.to_numpy(), gg.event_ticker.to_numpy())
        print(f"  last print taker-{side}: n={st['n']} ({st['n_clusters']} ev)"
              f" YES={gg.result.mean():.3f} px={gg.yes_price.mean():.1f}"
              f" net={st['mean']:+.2f} CI=[{st['ci_lo']:+.2f},{st['ci_hi']:+.2f}]")

    print("\n== (b) entry realism: is there a REAL NO fill at/near the modeled"
          " price after the entry moment? ==")
    # For each rule entry, find taker-NO prints in the same market with
    # ts in (entry_ts, entry_ts+60min] — the NO ask actually traded there.
    ent = g[["ticker", "ts", "yes_price", "result", "event_ticker",
             "net"]].rename(columns={"ts": "ets", "yes_price": "p_last"})
    tno = j[~j.taker_yes][["ticker", "ts", "yes_price", "count"]]
    m = ent.merge(tno, on="ticker", how="left")
    m = m[(m.ts > m.ets) & (m.ts <= m.ets + pd.Timedelta(minutes=60))]
    # best (lowest NO cost = highest yes_price) real NO fill within 60 min
    best = (m.assign(no_cost=100.0 - m.yes_price)
             .groupby(["ticker", "ets"]).no_cost.min().rename("real_no_cost"))
    ent = ent.join(best, on=["ticker", "ets"])
    ent["model_no_cost"] = (100.0 - ent.p_last + 4.0).clip(1, 99)
    has = ent.real_no_cost.notna()
    print(f"  entries with ANY taker-NO print within 60min after entry: "
          f"{has.mean():.1%} ({has.sum()}/{len(ent)})")
    d = ent[has]
    diff = d.real_no_cost - d.model_no_cost
    print(f"  real NO cost - modeled cost (c): mean={diff.mean():+.2f} "
          f"median={diff.median():+.2f} "
          f"P(real>model)={(diff > 0).mean():.1%}")
    # re-price: fill only when a real NO print happened, at that real level
    net_real = (np.where(d.result == 0, 100.0, 0.0) - d.real_no_cost
                - FEE(d.real_no_cost)).astype(float)
    st = cluster_bootstrap(net_real, d.event_ticker.to_numpy())
    print(f"  repriced at real NO fills (filled subset only): n={st['n']} "
          f"({st['n_clusters']} ev) net={st['mean']:+.2f} "
          f"CI=[{st['ci_lo']:+.2f},{st['ci_hi']:+.2f}] p={st['p_le_0']:.4f}")
    # unfilled subset: what did the model claim for those?
    u = ent[~has]
    stu = cluster_bootstrap(u.net.to_numpy(), u.event_ticker.to_numpy())
    print(f"  UNFILLED subset modeled net (fantasy fills): n={stu['n']} "
          f"({stu['n_clusters']} ev) net={stu['mean']:+.2f} "
          f"CI=[{stu['ci_lo']:+.2f},{stu['ci_hi']:+.2f}]")
    stf = cluster_bootstrap(d.net.to_numpy(), d.event_ticker.to_numpy())
    print(f"  FILLED subset modeled net: n={stf['n']} net={stf['mean']:+.2f} "
          f"CI=[{stf['ci_lo']:+.2f},{stf['ci_hi']:+.2f}]")

    # same but 180-min fill horizon (up to event end)
    m2 = ent.drop(columns=["real_no_cost"], errors="ignore")
    m2 = m2.merge(tno, on="ticker", how="left")
    m2 = m2[(m2.ts > m2.ets)]
    best2 = (m2.assign(no_cost=100.0 - m2.yes_price)
              .groupby(["ticker", "ets"]).no_cost.min()
              .rename("real_no_cost2"))
    ent = ent.join(best2, on=["ticker", "ets"])
    has2 = ent.real_no_cost2.notna()
    d2 = ent[has2]
    net2 = (np.where(d2.result == 0, 100.0, 0.0) - d2.real_no_cost2
            - FEE(d2.real_no_cost2)).astype(float)
    st2 = cluster_bootstrap(net2, d2.event_ticker.to_numpy())
    print(f"  any-time-to-close horizon: filled {has2.mean():.1%}; repriced "
          f"net={st2['mean']:+.2f} CI=[{st2['ci_lo']:+.2f},{st2['ci_hi']:+.2f}]"
          f" (best-fill = look-ahead optimistic)")

    print("\n== (c) causal-anchor rescue attempts ==")
    # c1: stricter onset thresholds
    for thr in (50, 150):
        onset = onset_windows(mk, tr, rate_threshold=thr).rename("onset")
        evs = (j.groupby("event_ticker")
                 .agg(series=("series", "first"), ev_end=("ev_end", "first"))
                 .join(onset, how="inner"))
        D = evs.series.map(lambda s: series_duration(s, WINDOWS_EXT))
        pe = (evs.onset + pd.to_timedelta(D, unit="h")).rename("pred_end")
        err = (pe - evs.ev_end).dt.total_seconds() / 60.0
        ok = err.dropna()
        print(f"  onset thr={thr}: events={len(ok)} err median="
              f"{ok.median():+.0f}m |err|<=60m: {(ok.abs()<=60).mean():.1%}")
        jj = j.join(pe, on="event_ticker").dropna(subset=["pred_end"])
        lastc = last_print_in_window(jj, "pred_end", 60, 180)
        rule_pnl(lastc, f"onset thr={thr} V1")

    # c2: schedule-prior anchor — walk-forward per-series median end
    # time-of-day (ET). pred_end = event date (from ticker) + that TOD.
    date_re = re.compile(r"-(\d{2})([A-Z]{3})(\d{2})")
    months = {m: i + 1 for i, m in enumerate(
        ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
         "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}
    evs = (j.groupby("event_ticker")
             .agg(series=("series", "first"), ev_end=("ev_end", "first"))
             .reset_index())
    def tick_day(et):
        mm = date_re.search(et)
        if not mm:
            return pd.NaT
        mo = months.get(mm.group(2))
        if not mo:
            return pd.NaT
        try:
            return pd.Timestamp(2000 + int(mm.group(1)), mo,
                                int(mm.group(3)), tz="America/New_York")
        except ValueError:
            return pd.NaT
    evs["day"] = evs.event_ticker.map(tick_day)
    evs = evs.dropna(subset=["day"])
    end_et = evs.ev_end.dt.tz_convert("America/New_York")
    # hours since midnight of ticker day (can exceed 24 for post-midnight ends)
    evs["end_h"] = (end_et - evs.day).dt.total_seconds() / 3600.0
    evs = evs[(evs.end_h > -6) & (evs.end_h < 42)]
    evs = evs.sort_values("ev_end").reset_index(drop=True)
    pred = np.full(len(evs), np.nan)
    hist: dict[str, list] = {}
    for i, r in evs.iterrows():
        h = hist.setdefault(r.series, [])
        if len(h) >= 5:
            pred[i] = np.median(h)
        h.append(r.end_h)
    evs["pred_end_h"] = pred
    evs = evs.dropna(subset=["pred_end_h"])
    evs["pred_end"] = evs.day + pd.to_timedelta(evs.pred_end_h, unit="h")
    evs["pred_end"] = evs.pred_end.dt.tz_convert("UTC")
    err = (evs.pred_end - evs.ev_end).dt.total_seconds() / 60.0
    print(f"\n  schedule-prior (walk-fwd median end TOD, >=5 prior events): "
          f"events={len(evs)} err median={err.median():+.0f}m "
          f"IQR=[{err.quantile(.25):+.0f},{err.quantile(.75):+.0f}] "
          f"|err|<=30m: {(err.abs()<=30).mean():.1%} "
          f"|err|<=60m: {(err.abs()<=60).mean():.1%}")
    jj = j.merge(evs[["event_ticker", "pred_end"]], on="event_ticker",
                 how="inner")
    lastc = last_print_in_window(jj, "pred_end", 60, 180)
    gs = rule_pnl(lastc, "schedule-prior V1 [60,180)")
    if gs is not None:
        gs2 = gs.copy()
        for s in ("KXWCMENTION",):
            rule_pnl(gs2[gs2.series != s].drop(columns="net"),
                     f"schedule-prior V1 ex-{s}")
        gs2["month"] = gs2.ev_end.dt.to_period("M")
        for mth, gm in gs2.groupby("month"):
            rule_pnl(gm.drop(columns="net"), f"schedule-prior {mth}")
        bys = (gs2.groupby("series")
                  .agg(n=("net", "size"), mean=("net", "mean"),
                       total=("net", "sum")).sort_values("total",
                                                         ascending=False))
        print(bys.round(2).to_string())
    # per-series anchor quality for the survivors
    q = evs.assign(err=err.abs()).groupby("series").err.median()
    print("\n  per-series median |anchor err| (min):")
    print(q.sort_values().round(0).to_string())


if __name__ == "__main__":
    main()
