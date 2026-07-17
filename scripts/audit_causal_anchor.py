"""ADVERSARIAL AUDIT (flank 1): causal event-end anchor for the family-2b
taker-NO rule, plus counterparty/composition checks.

Steps:
  0. Reproduce the ex-post headline ([60,180) pre event-end, 20-80c, NO
     at 100-last+4c, taker fee).
  1. Anchor-error sensitivity on the ex-post anchor: shift assumed end
     +/-30/60 min.
  2. Fully causal anchor: onset detector (tape burst, date-gated) +
     per-series duration D. Variants:
       (a) D = hand WINDOWS (extended with public-knowledge durations)
       (b) D = walk-forward per-series median(actual_end - onset)
     Rule V1: last print with ts in [pred_end-180m, pred_end-60m), 20-80c,
     ts <= close_time, buy NO at (100-p)+4c + fee, settle.
     Rule V2: decision at T = pred_end - 90m; most recent print in
     [T-120m, T), 20-80c, market open at T (close_time > T).
  3. Counterparty: PnL of ACTUAL taker-NO prints in the ex-post window/band
     (executable-reality check) and markout of the resting YES bids that
     fill them (are they informed?).
  4. Composition: per-series contribution; exclude WC; exclude WC+NBA+NCAA;
     month split.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from research_mentions_lib import load_markets, taker_fee_cents  # noqa: E402
from research_tape_micro import cluster_bootstrap, load_trades_full  # noqa: E402
from research_inevent_hazard import WINDOWS, onset_windows  # noqa: E402

FEE = np.vectorize(lambda p: taker_fee_cents(min(max(p, 1), 99)))

# public-knowledge typical durations (hours) for series absent from WINDOWS
WINDOWS_EXT = dict(WINDOWS)
WINDOWS_EXT.update({
    "KXNFLMENTION": 3.25, "KXNCAAMENTION": 3.0, "KXNCAABMENTION": 3.0,
    "KXFOXNEWSMENTION": 1.0, "KXLASTWORDMENTION": 1.0, "KXMADDOWMENTION": 1.0,
    "KXSNLMENTION": 1.5, "KXSURVIVORMENTION": 1.5, "KXMTPMENTION": 1.0,
    "KXFTNMENTION": 1.0, "KXBESSENTMTPMENTION": 1.0, "KXPSAKIMENTION": 1.0,
    "KXTHEWEEKNIGHTMENTION": 1.0, "KXMENTION": 1.0, "KXPERSONMENTION": 1.0,
    "KXHOCHULMENTION": 1.0, "KXBERNIEMENTION": 1.0, "KXNEWSOMMENTION": 1.0,
    "KXCARNEYMENTION": 1.0, "KXCONGRESSMENTION": 2.0, "KXGOVERNORMENTION": 1.0,
    "KXHEGSETHMENTION": 1.0, "KXMRBEASTMENTION": 0.5, "KXATHLETEMENTION": 1.0,
    "KXLASTWORDCOUNT": 1.0,
})
for s in ("KXEARNINGSMENTION",):
    pass  # earnings handled by prefix below


def series_duration(series: str, table: dict) -> float | None:
    if series in table:
        return table[series]
    if series.startswith("KXEARNINGSMENTION"):
        return 1.25  # typical earnings call ~60-90 min
    return None


def rule_pnl(last: pd.DataFrame, label: str, spread: float = 4.0):
    g = last[last.yes_price.between(20, 80)]
    if len(g) < 15:
        print(f"  {label}: n={len(g)} insufficient")
        return None
    entry = (100.0 - g.yes_price + spread).clip(1, 99)
    fee = FEE(entry)
    net = (np.where(g.result == 0, 100.0, 0.0) - entry - fee).astype(float)
    st = cluster_bootstrap(net, g.event_ticker.to_numpy())
    print(f"  {label}: n={st['n']} ({st['n_clusters']} ev) "
          f"YES={g.result.mean():.3f} px={g.yes_price.mean():.1f}c "
          f"net={st['mean']:+.2f} CI=[{st['ci_lo']:+.2f},{st['ci_hi']:+.2f}] "
          f"p={st['p_le_0']:.4f}")
    return g.assign(net=net)


def last_print_in_window(j, end_col, lo_min, hi_min, feasible=True):
    mte = (j[end_col] - j.ts).dt.total_seconds() / 60.0
    w = j[(mte >= lo_min) & (mte < hi_min)]
    if feasible:
        w = w[w.ts <= w.close_time]  # print after halt is untradeable
    return w.sort_values("ts").groupby("ticker").tail(1)


def main():
    mk = load_markets().drop_duplicates(subset=["ticker"])
    tr = load_trades_full()
    j = tr.merge(mk[["ticker", "result", "event_ticker", "series", "family",
                     "close_time"]], on="ticker", how="inner")
    print(f"dataset: {j.ticker.nunique()} tapes / "
          f"{j.event_ticker.nunique()} events / {len(j)} trades")

    # ---------------- 0. reproduce ex-post headline
    ev_end = j.groupby("event_ticker").close_time.max().rename("ev_end")
    j = j.join(ev_end, on="event_ticker")
    print("\n== 0. ex-post anchor reproduction ==")
    last0 = last_print_in_window(j, "ev_end", 60, 180, feasible=False)
    rule_pnl(last0, "[60,180) as published (incl. post-close prints)")
    last0f = last_print_in_window(j, "ev_end", 60, 180, feasible=True)
    g0 = rule_pnl(last0f, "[60,180) feasible prints only")

    # ---------------- 1. anchor-error sensitivity (ex-post anchor shifted)
    print("\n== 1. anchor error: assumed end = true end + delta ==")
    for dmin in (-60, -30, 0, 30, 60):
        j["end_shift"] = j.ev_end + pd.Timedelta(minutes=dmin)
        last = last_print_in_window(j, "end_shift", 60, 180)
        # feasibility at entry moment is automatic (print time); but with a
        # late-shifted anchor the window may extend past true end - entries
        # whose print is after true end-0 are still real prints (tradable).
        rule_pnl(last, f"delta={dmin:+4d} min")
    j.drop(columns=["end_shift"], inplace=True)

    # ---------------- 2. causal anchor
    print("\n== 2. causal anchor: onset + per-series duration ==")
    onset = onset_windows(mk, tr).rename("onset")
    evs = (j.groupby("event_ticker")
             .agg(series=("series", "first"), ev_end=("ev_end", "first"))
             .join(onset, how="inner"))
    print(f"  onset detected for {len(evs)} / {j.event_ticker.nunique()} events")

    # anchor quality vs truth
    for name, table in (("WINDOWS_EXT", WINDOWS_EXT),):
        D = evs.series.map(lambda s: series_duration(s, table))
        pe = evs.onset + pd.to_timedelta(D, unit="h")
        err = (pe - evs.ev_end).dt.total_seconds() / 60.0
        ok = err.dropna()
        print(f"  [{name}] pred_end - true_end (min): n={len(ok)} "
              f"median={ok.median():+.1f} IQR=[{ok.quantile(.25):+.1f},"
              f"{ok.quantile(.75):+.1f}] |err|<=30m: {(ok.abs()<=30).mean():.1%}"
              f" |err|<=60m: {(ok.abs()<=60).mean():.1%}")

    # walk-forward duration: per-series expanding median of (ev_end - onset)
    evs2 = evs.copy()
    evs2["span_h"] = (evs2.ev_end - evs2.onset).dt.total_seconds() / 3600.0
    evs2 = evs2.sort_values("ev_end")
    wf = []
    for s, g in evs2.groupby("series"):
        sp = g.span_h.to_numpy()
        med = [np.median(sp[:i]) if i >= 5 else np.nan for i in range(len(g))]
        wf.append(pd.Series(med, index=g.index))
    evs2["D_wf"] = pd.concat(wf)
    err_wf = ((evs2.onset + pd.to_timedelta(evs2.D_wf, unit="h") - evs2.ev_end)
              .dt.total_seconds() / 60.0).dropna()
    print(f"  [walk-fwd median span] err: n={len(err_wf)} "
          f"median={err_wf.median():+.1f} "
          f"IQR=[{err_wf.quantile(.25):+.1f},{err_wf.quantile(.75):+.1f}] "
          f"|err|<=30m: {(err_wf.abs()<=30).mean():.1%} "
          f"|err|<=60m: {(err_wf.abs()<=60).mean():.1%}")

    for name, D_h in (
            ("hand durations (WINDOWS_EXT)",
             evs.series.map(lambda s: series_duration(s, WINDOWS_EXT))),
            ("walk-forward per-series median", evs2.D_wf)):
        pe = (evs.onset + pd.to_timedelta(D_h, unit="h")).rename("pred_end")
        jj = j.join(pe, on="event_ticker").dropna(subset=["pred_end"])
        print(f"\n  --- causal variant: {name} "
              f"({jj.event_ticker.nunique()} events with anchor) ---")
        lastc = last_print_in_window(jj, "pred_end", 60, 180)
        gc = rule_pnl(lastc, "V1 last print in [60,180) pre pred_end")
        # V2: decision at fixed clock time T = pred_end - 90 min
        Tcol = jj.pred_end - pd.Timedelta(minutes=90)
        w = jj[(jj.ts < Tcol) & (jj.ts >= Tcol - pd.Timedelta(minutes=120))
               & (jj.ts <= jj.close_time)]
        lastT = w.sort_values("ts").groupby("ticker").tail(1)
        # market must still be open at decision time T
        lastT = lastT[lastT.close_time
                      > lastT.pred_end - pd.Timedelta(minutes=90)]
        rule_pnl(lastT, "V2 decision at T=pred_end-90m (open-at-T only)")
        if gc is not None and name.startswith("walk"):
            print("\n  causal V1 by series (top contributors):")
            bys = (gc.groupby("series")
                     .agg(n=("net", "size"), mean=("net", "mean"),
                          total=("net", "sum"))
                     .sort_values("total", ascending=False))
            print((bys.round(2)).to_string())
            print("\n  causal V1 excluding KXWCMENTION:")
            rule_pnl(gc[gc.series != "KXWCMENTION"].drop(columns="net"),
                     "V1 ex-WC")
            dead = ["KXWCMENTION", "KXNBAMENTION", "KXNCAAMENTION",
                    "KXNCAABMENTION", "KXNHLMENTION"]
            rule_pnl(gc[~gc.series.isin(dead)].drop(columns="net"),
                     "V1 ex-{WC,NBA,NCAA,NHL} (post-Jul-19 supply)")
            gc["month"] = gc.ev_end.dt.to_period("M")
            print("\n  causal V1 by month:")
            for m, gm in gc.groupby("month"):
                rule_pnl(gm.drop(columns="net"), f"month {m}")

    # ---------------- 3. counterparty reality check
    print("\n== 3. actual taker-NO prints in [60,180) pre true event end, "
          "20-80c: did real NO takers make money at REAL fills? ==")
    mte = (j.ev_end - j.ts).dt.total_seconds() / 60.0
    w = j[(mte >= 60) & (mte < 180) & j.yes_price.between(20, 80)
          & (j.ts <= j.close_time)]
    for side, lbl in ((False, "taker-NO (the rule's trade, real fills)"),
                      (True, "taker-YES (the other side)")):
        g = w[w.taker_yes == side]
        if side:  # taker buys YES at print price
            entry = g.yes_price
            gross = np.where(g.result == 1, 100.0, 0.0) - entry
        else:     # taker buys NO at 100 - print price (real executed level)
            entry = 100.0 - g.yes_price
            gross = np.where(g.result == 0, 100.0, 0.0) - entry
        net = gross - FEE(entry)
        st = cluster_bootstrap(net, g.event_ticker.to_numpy(),
                               weights=g["count"].to_numpy())
        stu = cluster_bootstrap(net, g.event_ticker.to_numpy())
        print(f"  {lbl}: n={len(g)} prints ({g.event_ticker.nunique()} ev, "
              f"{g.ticker.nunique()} mkts) vol-wtd net={st['mean']:+.2f} "
              f"CI=[{st['ci_lo']:+.2f},{st['ci_hi']:+.2f}] p={st['p_le_0']:.4f}"
              f" | unwtd {stu['mean']:+.2f}")
    vol_no = w.loc[~w.taker_yes, "count"].sum()
    vol_yes = w.loc[w.taker_yes, "count"].sum()
    print(f"  volume split in window/band: taker-NO {vol_no:.0f} "
          f"({vol_no/(vol_no+vol_yes):.1%}) vs taker-YES {vol_yes:.0f}")
    w2 = w.copy()
    w2["month"] = w2.ev_end.dt.to_period("M")
    print("  taker-NO volume share by month (crowding?):")
    for m, gm in w2.groupby("month"):
        vn = gm.loc[~gm.taker_yes, "count"].sum()
        vy = gm.loc[gm.taker_yes, "count"].sum()
        print(f"    {m}: NO share {vn/(vn+vy):.1%}  (vol {vn+vy:,.0f})")

    # ---------------- 4. composition of the PUBLISHED rule
    if g0 is not None:
        print("\n== 4. composition of the ex-post rule ==")
        bys = (g0.groupby("series")
                 .agg(n=("net", "size"), mean=("net", "mean"),
                      total=("net", "sum"))
                 .sort_values("total", ascending=False))
        print(bys.round(2).to_string())
        tot = g0.net.sum()
        wc = g0.loc[g0.series == "KXWCMENTION", "net"].sum()
        print(f"  WC share of total net: {wc/tot:.1%} "
              f"({g0[g0.series=='KXWCMENTION'].shape[0]}/{len(g0)} entries)")
        rule_pnl(g0[g0.series != "KXWCMENTION"].drop(columns="net"),
                 "ex-post rule ex-WC")
        dead = ["KXWCMENTION", "KXNBAMENTION", "KXNCAAMENTION",
                "KXNCAABMENTION", "KXNHLMENTION"]
        rule_pnl(g0[~g0.series.isin(dead)].drop(columns="net"),
                 "ex-post rule ex-{WC,NBA,NCAA,NHL}")
        g0m = g0.copy()
        g0m["month"] = g0m.ev_end.dt.to_period("M")
        for m, gm in g0m.groupby("month"):
            rule_pnl(gm.drop(columns="net"), f"ex-post rule month {m}")


if __name__ == "__main__":
    main()
