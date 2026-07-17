#!/usr/bin/env python3
"""Evaluate the pre-registered in-event NO-carry rule (PREREGISTERED_RULE.md).

Exact rule: at tau=0.40 (last trade in tau (0.25, 0.40]), unresolved market,
price p in [30,70] cents -> buy NO at (100-p)+slip, hold to settlement.

--oos       : only events NOT in formation_events.json (default)
--formation : only formation events
--all       : everything
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

TAU = 0.40
TAU_LO = 0.25
P_LO, P_HI = 30.0, 70.0


def trades_for_rule(slip: float, require_no_taker: bool = False,
                    mode: str = "expost") -> pd.DataFrame:
    tr = load_trades()
    mk = build_event_frame(load_markets(), mode=mode, trades=tr)
    tr = tr.merge(mk[["ticker", "event_ticker", "series", "result",
                      "event_start", "event_end", "mention_tau"]],
                  on="ticker", how="inner")
    tr["tau"] = ((tr.ts - tr.event_start).dt.total_seconds()
                 / (tr.event_end - tr.event_start).dt.total_seconds())
    w = tr[(tr.tau > TAU_LO) & (tr.tau <= TAU)].dropna(subset=["yes_price"])
    if require_no_taker:
        # only reference trades where the taker actually BOUGHT NO at this
        # price — the NO fill provably existed at (100 - yes_price)
        w = w[w.taker_side == "no"]
    last = (w.sort_values("ts").groupby("ticker").agg(
        p_tau=("yes_price", "last"), event=("event_ticker", "first"),
        series=("series", "first"), result=("result", "first"),
        mtau=("mention_tau", "first")).reset_index())
    # unresolved at TAU: NO result, or mention after TAU
    last = last[(last.result == 0) | (last.mtau > TAU)]
    last = last[(last.p_tau >= P_LO) & (last.p_tau <= P_HI)]
    no_cost = 100.0 - last.p_tau + slip
    fee = no_cost.map(taker_fee_cents)
    last["pnl"] = np.where(last.result == 0, 100.0, 0.0) - no_cost - fee
    return last


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=["oos", "formation", "all"], default="oos")
    ap.add_argument("--slip", type=float, default=2.0)
    ap.add_argument("--require-no-taker", action="store_true")
    ap.add_argument("--mode", choices=["expost", "exante", "onset"],
                    default="onset")
    args = ap.parse_args()
    formation = set(json.load((DATA / "formation_events.json").open()))
    t = trades_for_rule(args.slip, args.require_no_taker, args.mode)
    if args.scope == "oos":
        t = t[~t.event.isin(formation)]
    elif args.scope == "formation":
        t = t[t.event.isin(formation)]
    print(f"scope={args.scope} slip={args.slip}c: {len(t)} trades, "
          f"{t.event.nunique()} events")
    if len(t) < 10:
        print("not enough data yet")
        return
    bs = cluster_bootstrap(t.pnl.values, t.event.values)
    print(f"mean={bs['mean']:+.2f}c/contract CI95=[{bs['ci_lo']:+.2f},"
          f"{bs['ci_hi']:+.2f}] p(<=0)={bs['p_le_0']:.4f} "
          f"n={bs['n']} events={bs['n_clusters']}")
    print("\nby series:")
    print(t.groupby("series").agg(n=("pnl", "size"), mean=("pnl", "mean"),
                                  ev=("event", "nunique"))
          .sort_values("n", ascending=False).to_string())
    # leave-one-series-out
    print("\nleave-one-series-out means:")
    for s in t.series.unique():
        g = t[t.series != s]
        if len(g) < 10 or g.event.nunique() < 5:
            continue
        bs = cluster_bootstrap(g.pnl.values, g.event.values)
        print(f"  -{s:20s} mean={bs['mean']:+.2f}c CI=[{bs['ci_lo']:+.2f},"
              f"{bs['ci_hi']:+.2f}] p={bs['p_le_0']:.4f} ev={bs['n_clusters']}")


if __name__ == "__main__":
    main()
