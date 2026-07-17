"""Adversarial audit of the family2b LATE-EVENT TAKER-NO rule.

Attacks:
  A. Reproduce the headline [60,180)min 20-80c number.
  B. Tradeability at the only implementable entry moment (window end,
     T = ev_end - 60min): markets already halted at T (close_time < T)
     cannot be traded live -> split the sample and P&L.
  C. Last-print staleness at T and mid-window entry variants.
  D. Reprice entries at the real NO ask (100 - yes_bid) from minute candles
     at the signal moment, where coverage exists.
  E. Temporal stability by ev_end month; composition by series (World Cup).
  F. Post-close prints inside the window / resolved-during-window handling.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from research_mentions_lib import DATA, taker_fee_cents  # noqa: E402
from research_tape_micro import build_dataset, cluster_bootstrap  # noqa: E402


def net_no(g: pd.DataFrame, entry_no: np.ndarray) -> np.ndarray:
    entry = np.clip(entry_no, 1, 99)
    fee = np.array([taker_fee_cents(e) for e in entry])
    return (np.where(g.result.to_numpy() == 0, 100.0, 0.0) - entry - fee)


def stat_line(lbl, g, entry_no):
    if len(g) < 5:
        print(f"  {lbl}: n={len(g)} (too few)")
        return None
    net = net_no(g, entry_no)
    st = cluster_bootstrap(net, g.event_ticker.to_numpy())
    print(f"  {lbl}: n={st['n']} ({st['n_clusters']} ev) YES={g.result.mean():.3f} "
          f"net={st['mean']:+.2f} CI=[{st['ci_lo']:+.2f},{st['ci_hi']:+.2f}] "
          f"p={st['p_le_0']:.4f}")
    return st


def main():
    mk, tr, j = build_dataset()
    del tr
    jj = j.join(j.groupby("event_ticker").close_time.max().rename("ev_end"),
                on="event_ticker")
    jj["mte"] = (jj.ev_end - jj.ts).dt.total_seconds() / 60.0

    lo, hi = 60, 180
    w = jj[(jj.mte >= lo) & (jj.mte < hi)]
    last = w.sort_values("ts").groupby("ticker").tail(1)
    g = last[last.yes_price.between(20, 80)].copy()

    print("=" * 72)
    print("A. REPRODUCE headline [60,180) 20-80c, entry=(100-p)+4c")
    entry0 = (100.0 - g.yes_price + 4.0).to_numpy()
    stat_line("headline", g, entry0)

    print("\nB. TRADEABILITY at T = ev_end - 60min (the only implementable")
    print("   entry moment matching this selection)")
    g["T"] = g.ev_end - pd.Timedelta(minutes=60)
    g["open_at_T"] = g.close_time > g["T"]
    print(g.groupby(["open_at_T", "result"]).size().rename("n").to_string())
    got = g[g.open_at_T]
    gnot = g[~g.open_at_T]
    stat_line("tradeable at T (close_time > T)", got,
              (100.0 - got.yes_price + 4.0).to_numpy())
    if len(gnot):
        stat_line("UNTRADEABLE at T (halted before entry)", gnot,
                  (100.0 - gnot.yes_price + 4.0).to_numpy())
        print(f"   untradeable YES rate={gnot.result.mean():.3f}  "
              f"(result=0 here are free wins the live trader cannot get)")

    print("\nC. STALENESS of the last print at T (minutes)")
    stale = (g["T"] - g.ts).dt.total_seconds() / 60.0
    print(stale.describe(percentiles=[.1, .25, .5, .75, .9]).round(1).to_string())
    # also: prints after own close inside window?
    n_after = (g.ts > g.close_time).sum()
    print(f"  signal prints after own close_time: {n_after}")
    # resolved during window handling: YES markets halted inside window
    halted_in_win = g[(g.close_time >= g.ev_end - pd.Timedelta(minutes=180))
                      & (g.close_time < g["T"])]
    print(f"  markets halted INSIDE the window before T: {len(halted_in_win)} "
          f"(result mix: {halted_in_win.result.value_counts().to_dict()})")

    print("\nD. REPRICE at real NO ask from minute candles at T (100 - yes_bid)")
    tick_set = set(g.ticker)
    cov = {}
    with (DATA / "minute_candles.jsonl").open() as f:
        for line in f:
            rec = json.loads(line)
            tk = rec.get("ticker")
            if tk not in tick_set:
                continue
            rows = []
            for c in rec.get("candles") or []:
                yb = (c.get("yes_bid") or {}).get("close_dollars")
                ya = (c.get("yes_ask") or {}).get("close_dollars")
                if yb is None:
                    continue
                rows.append((int(c["end_period_ts"]), float(yb) * 100,
                             float(ya) * 100 if ya is not None else np.nan))
            if rows:
                cov.setdefault(tk, []).extend(rows)
    print(f"  minute-candle coverage: {len(cov)} of {len(g)} signal markets")
    rp = []
    for _, r in g.iterrows():
        rows = cov.get(r.ticker)
        if not rows:
            continue
        arr = np.array(sorted(rows))
        t_entry = r["T"].timestamp()
        # last candle at/just before T with a bid
        k = np.searchsorted(arr[:, 0], t_entry, side="right") - 1
        if k < 0 or t_entry - arr[k, 0] > 600:  # need a quote within 10min of T
            continue
        yes_bid = arr[k, 1]
        no_ask = 100.0 - yes_bid
        rp.append({"ticker": r.ticker, "event_ticker": r.event_ticker,
                   "result": r.result, "yes_price": r.yes_price,
                   "no_ask_real": no_ask,
                   "assumed": 100.0 - r.yes_price + 4.0,
                   "yes_ask_c": arr[k, 2]})
    rp = pd.DataFrame(rp)
    if len(rp):
        rp["slip"] = rp.no_ask_real - rp.assumed
        print(f"  matched {len(rp)} signals with a live quote within 10min of T")
        print("  slip = real NO ask - assumed entry (positive = backtest too "
              "optimistic):")
        print(rp.slip.describe(percentiles=[.1, .25, .5, .75, .9]).round(2).to_string())
        stat_line("repriced at REAL NO ask", rp, rp.no_ask_real.to_numpy())
        stat_line("same subset, assumed entry", rp, rp.assumed.to_numpy())
        # sanity: quoted spread at T
        spr = (rp.yes_ask_c - (100 - rp.no_ask_real)).dropna()
        if len(spr):
            print(f"  quoted YES spread at T: median={spr.median():.1f}c "
                  f"mean={spr.mean():.1f}c")

    print("\nE. TEMPORAL + COMPOSITION")
    g["month"] = g.ev_end.dt.to_period("M").astype(str)
    for mth, gg in g.groupby("month"):
        stat_line(f"month {mth}", gg, (100.0 - gg.yes_price + 4.0).to_numpy())
    # last 30 days vs before
    cutoff = g.ev_end.max() - pd.Timedelta(days=30)
    for lbl, gg in [("first part", g[g.ev_end <= cutoff]),
                    ("last 30 days", g[g.ev_end > cutoff])]:
        stat_line(lbl, gg, (100.0 - gg.yes_price + 4.0).to_numpy())
    g["series"] = g.ticker.str.split("-").str[0]
    print("\n  by series (n>=30):")
    tot_net = net_no(g, entry0).sum()
    for s, gg in sorted(g.groupby("series"), key=lambda kv: -len(kv[1])):
        if len(gg) < 30:
            continue
        net = net_no(gg, (100.0 - gg.yes_price + 4.0).to_numpy())
        print(f"    {s:<24} n={len(gg):>4} ev={gg.event_ticker.nunique():>3} "
              f"net={net.mean():+6.2f}  share_of_total_pnl="
              f"{net.sum()/tot_net:+.1%}")
    wc = g[g.series == "KXWCMENTION"]
    print(f"  World Cup share: {len(wc)/len(g):.1%} of signals, "
          f"{net_no(wc, (100.0 - wc.yes_price + 4.0).to_numpy()).sum()/tot_net:+.1%} of P&L")

    print("\nF. speech excluded check + ex-speech headline")
    from research_mentions_lib import classify_family
    g["fam2"] = g.series.map(classify_family)
    gs = g[g.fam2 != "speech"]
    stat_line("ex-speech headline", gs, (100.0 - gs.yes_price + 4.0).to_numpy())


if __name__ == "__main__":
    main()
