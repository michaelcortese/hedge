"""Audit part 3: is the measurement anchor (max close_time) even a
schedulable quantity? Within-series dispersion of event-end time-of-day,
occurrence_datetime-based anchor, and entry staleness/post-event share."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from research_mentions_lib import load_markets  # noqa: E402
from research_tape_micro import cluster_bootstrap, load_trades_full  # noqa: E402
from audit_causal_anchor import FEE, last_print_in_window, rule_pnl  # noqa: E402

TZ = "America/New_York"


def main():
    mk = load_markets().drop_duplicates(subset=["ticker"])
    tr = load_trades_full()
    j = tr.merge(mk[["ticker", "result", "event_ticker", "series",
                     "close_time"]], on="ticker", how="inner")
    ev_end = j.groupby("event_ticker").close_time.max().rename("ev_end")
    j = j.join(ev_end, on="event_ticker")

    evs = (j.groupby("event_ticker")
             .agg(series=("series", "first"), ev_end=("ev_end", "first"))
             .reset_index())
    occ = (mk.dropna(subset=["occurrence"])
             .groupby("event_ticker").occurrence.median().rename("occ"))
    evs = evs.join(occ, on="event_ticker")
    print(f"events: {len(evs)}; with occurrence_datetime: "
          f"{evs.occ.notna().sum()} ({evs.occ.notna().mean():.1%})")

    # 1. within-series dispersion of event-end time-of-day (ET)
    et = evs.ev_end.dt.tz_convert(TZ)
    evs["end_tod_h"] = et.dt.hour + et.dt.minute / 60.0
    print("\n== within-series dispersion of ev_end (max close_time) "
          "time-of-day, ET ==")
    def circ_iqr(h):
        # naive IQR is fine unless the series straddles midnight; also report
        # share within +/-30m of the series median
        med = h.median()
        dev = (h - med).abs()
        dev = np.minimum(dev, 24 - dev)  # circular
        return pd.Series({"n": len(h), "med_tod": med,
                          "within30m": (dev <= 0.5).mean(),
                          "within60m": (dev <= 1.0).mean(),
                          "iqr_h": h.quantile(.75) - h.quantile(.25)})
    disp = (evs.groupby("series").end_tod_h.apply(circ_iqr).unstack()
              .sort_values("n", ascending=False))
    print(disp[disp.n >= 10].round(2).to_string())

    # 2. ev_end - occurrence (is Kalshi's scheduled datetime + fixed span
    #    a viable anchor?)
    ok = evs.dropna(subset=["occ"]).copy()
    if len(ok):
        ok["span_h"] = (ok.ev_end - ok.occ).dt.total_seconds() / 3600.0
        print("\n== ev_end - occurrence_datetime (h), by series (n>=10) ==")
        sp = (ok.groupby("series").span_h
                .agg(n="size", med="median",
                     q25=lambda s: s.quantile(.25),
                     q75=lambda s: s.quantile(.75)))
        sp["iqr_min"] = (sp.q75 - sp.q25) * 60
        print(sp[sp.n >= 10].round(2).to_string())
        # walk-forward anchor: occ + per-series expanding median span
        ok = ok.sort_values("ev_end").reset_index(drop=True)
        pred = np.full(len(ok), np.nan)
        hist: dict[str, list] = {}
        for i, r in ok.iterrows():
            h = hist.setdefault(r.series, [])
            if len(h) >= 5:
                pred[i] = np.median(h)
            h.append(r.span_h)
        ok["pred_end"] = ok.occ + pd.to_timedelta(pred, unit="h")
        ok = ok.dropna(subset=["pred_end"])
        err = (ok.pred_end - ok.ev_end).dt.total_seconds() / 60.0
        print(f"\n  occurrence+walk-fwd-span anchor: events={len(ok)} "
              f"err median={err.median():+.0f}m "
              f"IQR=[{err.quantile(.25):+.0f},{err.quantile(.75):+.0f}] "
              f"|err|<=30m: {(err.abs()<=30).mean():.1%} "
              f"|err|<=60m: {(err.abs()<=60).mean():.1%}")
        jj = j.merge(ok[["event_ticker", "pred_end"]], on="event_ticker",
                     how="inner")
        lastc = last_print_in_window(jj, "pred_end", 60, 180)
        rule_pnl(lastc, "occurrence-anchor V1 [60,180)")

    # 3. staleness + post-mention-activity structure of the winning entries
    last = last_print_in_window(j, "ev_end", 60, 180)
    g = last[last.yes_price.between(20, 80)].copy()
    entry = (100.0 - g.yes_price + 4.0).clip(1, 99)
    g["net"] = (np.where(g.result == 0, 100.0, 0.0) - entry
                - FEE(entry)).astype(float)
    # how stale was the entry print (time since PREVIOUS print)?
    j_sorted = j.sort_values(["ticker", "ts"])
    prev_ts = j_sorted.groupby("ticker")["ts"].shift(1)
    stale_min = ((j_sorted.ts - prev_ts).dt.total_seconds() / 60.0)
    g = g.join(stale_min.rename("stale_prev_min"), how="left")
    # time left to (own) close after entry, and to ev_end
    g["min_to_own_close"] = (g.close_time - g.ts).dt.total_seconds() / 60.0
    g["min_to_ev_end"] = (g.ev_end - g.ts).dt.total_seconds() / 60.0
    print("\n== entry staleness / deadness structure ==")
    print(f"  mins from entry to EVENT end: median="
          f"{g.min_to_ev_end.median():.0f}")
    print(f"  mins from entry to OWN close: median="
          f"{g.min_to_own_close.median():.0f} "
          f"(YES mkts close early on mention)")
    for lab, mask in [("winners (NO)", g.result == 0),
                      ("losers (YES)", g.result == 1)]:
        gg = g[mask]
        print(f"  {lab}: n={len(gg)} entry-print staleness median="
              f"{gg.stale_prev_min.median():.1f}m; any print in final 60m "
              f"before ev_end: "
              f"{(gg.min_to_own_close > gg.min_to_ev_end - 60).mean():.1%}")
    # does the edge survive requiring a FRESH print (<=10 min stale)?
    fresh = g[g.stale_prev_min <= 10]
    rule_pnl(fresh.drop(columns="net"), "entries with fresh (<=10m) prints")
    stale = g[g.stale_prev_min > 60]
    rule_pnl(stale.drop(columns="net"), "entries with stale (>60m) prints")


if __name__ == "__main__":
    main()
