"""Addendum to audit_adverse_selection.py: outcome-clean fill evidence.

The full-hour 'fillable' proxy conditions on the price path (losers keep NO
cheap all hour, winners decay away from our limit), so its -5.3c is possibly
a selection artifact. Since ZERO selected markets halt within 15 min of
entry, taker-NO flow within our limit in the FIRST 5/10 min post-entry is a
(nearly) outcome-clean fill-evidence window: it is what an aggressive limit
posted at entry could plausibly capture.

Run:  .venv/bin/python scripts/audit_adverse_selection_fill.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from research_mentions_lib import DATA, taker_fee_cents  # noqa: E402
from audit_adverse_selection import boot, fmt, net_pnl, load_markets  # noqa: E402

MIN_NS = 60_000_000_000


def main():
    mk = load_markets().drop_duplicates(subset=["ticker"])
    j = pd.read_parquet(DATA / "_audit_advsel_tape.parquet")
    jj = j.join(j.groupby("event_ticker").close_time.max().rename("ev_end"),
                on="event_ticker")
    jj["mte"] = (jj.ev_end - jj.ts).dt.total_seconds() / 60.0
    w = jj[(jj.mte >= 60) & (jj.mte < 180)]
    last = w.sort_values("ts").groupby("ticker").tail(1)
    g = last[last.yes_price.between(20, 80)].copy()
    g = g.rename(columns={"ts": "sig_ts", "yes_price": "p_sig"})
    print(f"signals: n={len(g)}, events={g.event_ticker.nunique()}")

    tapes = {}
    for tk, gg in j[j.ticker.isin(g.ticker)].groupby("ticker", sort=False):
        tapes[tk] = (gg["ts"].dt.as_unit("ns").astype("int64").to_numpy(),
                     gg["yes_price"].to_numpy(),
                     gg["count"].to_numpy(), gg["taker_yes"].to_numpy())

    rows = []
    for row in g.itertuples():
        ts, px, ct, ty = tapes[row.ticker]
        entry = row.ev_end.value - 60 * MIN_NS
        rec = {"ticker": row.ticker, "event_ticker": row.event_ticker,
               "result": row.result, "p_sig": row.p_sig}
        for mins in (5, 10, 30):
            m = (ts > entry) & (ts <= entry + mins * MIN_NS)
            lim = m & (~ty) & (px >= row.p_sig - 4.0)
            rec[f"fill{mins}"] = float(ct[lim].sum())
            # broader: ANY print at yes >= p-4 (market still near our level)
            near = m & (px >= row.p_sig - 4.0)
            rec[f"near{mins}"] = float(ct[near].sum())
        rows.append(rec)
    s = pd.DataFrame(rows)
    s["net"] = net_pnl(s.p_sig, s.result)

    for mins in (5, 10, 30):
        col = f"fill{mins}"
        v = s[col]
        print(f"\n--- fill-evidence window: first {mins} min post-entry, "
              f"taker-NO prints at NO <= (100-p)+4c ---")
        print(f"  share zero: {(v == 0).mean():.1%}; "
              f"med={v.median():.0f} q75={v.quantile(.75):.0f} "
              f"(among >0: med={v[v > 0].median():.0f})")
        for cond, lbl in [(v > 0, "FILLABLE (evidence of NO buys at limit)"),
                          (v == 0, "NOT fillable (no NO prints at limit)")]:
            gg = s[cond]
            if len(gg) < 20:
                continue
            st = boot(gg.net.to_numpy(), gg.event_ticker.to_numpy())
            print("  " + fmt(st, f"{lbl} (YES rate {gg.result.mean():.3f})"))
        gg = s[v > 0]
        if len(gg) >= 20:
            wgt = gg[col].clip(upper=100).to_numpy()
            st = boot(gg.net.to_numpy(), gg.event_ticker.to_numpy(),
                      weights=wgt)
            print("  " + fmt(st, "  fillable, size-weighted (cap 100)"))
        # split fillability by outcome to expose residual path conditioning
        fy = s[s.result == 1][col]
        fn = s[s.result == 0][col]
        print(f"  fillable share | losers(YES): {(fy > 0).mean():.1%}  "
              f"winners(NO): {(fn > 0).mean():.1%}")

    # realistic per-signal size from the clean window
    v10 = s["fill10"]
    days = 65.4  # ev_end span from main audit (17.0 signals/day over data)
    per_day = len(s) / days
    for share in (0.5, 1.0):
        cap = (v10 * share).clip(upper=100)
        print(f"\n  capacity from 10-min window at {share:.0%} participation "
              f"(cap 100): mean={cap.mean():.1f} ct/signal, med={cap.median():.1f}; "
              f"at +5.9c/ct (repriced edge) -> "
              f"${(cap.mean() * per_day * 0.059):.0f}/day")


if __name__ == "__main__":
    main()
