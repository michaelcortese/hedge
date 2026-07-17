#!/usr/bin/env python
"""Independent replication of the claimed taker theta-carry rule on mention markets.

Written blind: does NOT import or read the original research code.

Rule under audit:
  For each settled mention market, take the last trade print strictly BEFORE
  (event_end - 60min), require it to be >= (event_end - 180min).
  event_end = max close_time over all markets in the same event_ticker (ex-post anchor).
  If that print's yes_price q is in [20, 80] cents inclusive:
    buy 1 NO at cost = (100 - q) + 4c spread allowance + taker fee at NO price.
    fee = ceil(0.07 * P * (1-P) * 100) cents, P = NO price in dollars, C = 1.
  Hold to settlement: pnl = (100 if result=="no" else 0) - cost.
  Exclusion: market must not have resolved before entry (close_time >= entry time,
  where entry time = the qualifying print's timestamp).
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np

BASE = "/home/mcortese/fun/hedge/.claude/worktrees/mentions-ml/data/research/mentions"


def parse_ts(s: str) -> datetime:
    # ISO 8601, sometimes fractional seconds, always Z
    s = s.replace("Z", "+00:00")
    return datetime.fromisoformat(s)


def cents_price(rec: dict, base: str):
    """Extract a price in integer-ish cents from a record, defensively."""
    v = rec.get(base)
    if v is not None:
        return float(v)  # legacy integer cents
    v = rec.get(base + "_dollars")
    if v is not None:
        return float(v) * 100.0
    v = rec.get(base + "_fp")
    if v is not None:
        return float(v)
    return None


def main():
    # ---------- load markets ----------
    markets = {}  # ticker -> dict
    n_market_rows = 0
    n_unsettled = 0
    n_missing_close = 0
    dup_market_tickers = 0
    with open(f"{BASE}/events.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ev = json.loads(line)
            for m in ev.get("markets", []):
                n_market_rows += 1
                t = m.get("ticker")
                res = m.get("result")
                if res not in ("yes", "no"):
                    n_unsettled += 1
                    continue
                ct = m.get("close_time")
                if not ct:
                    n_missing_close += 1
                    continue
                if t in markets:
                    dup_market_tickers += 1
                markets[t] = {
                    "ticker": t,
                    "event_ticker": m.get("event_ticker") or ev.get("event_ticker"),
                    "result": res,
                    "close_time": parse_ts(ct),
                    "open_time": parse_ts(m["open_time"]) if m.get("open_time") else None,
                }
    print(f"[load] market rows={n_market_rows} settled+usable={len(markets)} "
          f"unsettled={n_unsettled} missing_close={n_missing_close} dup_tickers={dup_market_tickers}")

    # event end = max close_time per event_ticker (ex-post anchor, as specified)
    event_end = {}
    for m in markets.values():
        e = m["event_ticker"]
        if e not in event_end or m["close_time"] > event_end[e]:
            event_end[e] = m["close_time"]
    print(f"[load] events={len(event_end)}")

    # ---------- load trades ----------
    trades = {}  # ticker -> sorted list of (dt, yes_cents)
    n_trades = 0
    n_dup_trade_id = 0
    n_bad_price = 0
    n_trade_lines = 0
    seen_ids_global = set()
    with open(f"{BASE}/trades.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n_trade_lines += 1
            tk = rec.get("ticker")
            lst = trades.setdefault(tk, [])
            for tr in rec.get("trades", []):
                n_trades += 1
                tid = tr.get("trade_id")
                if tid is not None:
                    if tid in seen_ids_global:
                        n_dup_trade_id += 1
                        continue
                    seen_ids_global.add(tid)
                ts = tr.get("created_time") or tr.get("ts")
                q = cents_price(tr, "yes_price")
                if ts is None or q is None or not (0 < q < 100):
                    n_bad_price += 1
                    continue
                lst.append((parse_ts(ts), q))
    for tk in trades:
        trades[tk].sort(key=lambda x: x[0])
    print(f"[load] trade lines={n_trade_lines} trades={n_trades} "
          f"dup_trade_id={n_dup_trade_id} bad_price_or_ts={n_bad_price}")

    # ---------- apply the rule ----------
    rows = []
    excl_resolved_before_entry = 0
    n_no_trades = 0
    n_no_print_in_window = 0
    n_price_out_of_band = 0
    rows_no_exclusion = 0  # sensitivity: without the resolved-before-entry filter

    for tk, m in markets.items():
        end = event_end[m["event_ticker"]]
        deadline = end - timedelta(minutes=60)
        window_start = end - timedelta(minutes=180)
        tape = trades.get(tk)
        if not tape:
            n_no_trades += 1
            continue
        # last print strictly before deadline
        last = None
        for dt, q in tape:
            if dt < deadline:
                last = (dt, q)
            else:
                break
        if last is None or last[0] < window_start:
            n_no_print_in_window += 1
            continue
        entry_time, q = last
        if not (20.0 <= q <= 80.0):
            n_price_out_of_band += 1
            continue
        rows_no_exclusion += 1
        if m["close_time"] < entry_time:
            excl_resolved_before_entry += 1
            continue
        p_no = (100.0 - q) / 100.0
        fee = math.ceil(0.07 * p_no * (1.0 - p_no) * 100.0)  # cents, C=1
        cost = (100.0 - q) + 4.0 + fee
        payout = 100.0 if m["result"] == "no" else 0.0
        rows.append({
            "ticker": tk,
            "event": m["event_ticker"],
            "series": tk.split("-")[0],
            "q": q,
            "entry_time": entry_time,
            "result": m["result"],
            "pnl": payout - cost,
        })

    print(f"\n[funnel] no_tape={n_no_trades} no_qualifying_print={n_no_print_in_window} "
          f"price_out_of_band={n_price_out_of_band} "
          f"resolved_before_entry_excluded={excl_resolved_before_entry}")
    print(f"[funnel] n_without_exclusion={rows_no_exclusion} n_final={len(rows)}")

    if not rows:
        print("NO TRADES SELECTED"); sys.exit(1)

    pnl = np.array([r["pnl"] for r in rows])
    qs = np.array([r["q"] for r in rows])
    yes = np.array([r["result"] == "yes" for r in rows])
    events = np.array([r["event"] for r in rows])
    n_events = len(set(events))

    print(f"\n=== HEADLINE ===")
    print(f"n = {len(rows)} markets across {n_events} events")
    print(f"mean net pnl = {pnl.mean():+.2f} c/contract  (total {pnl.sum():+.0f}c)")
    print(f"median = {np.median(pnl):+.1f}  std = {pnl.std(ddof=1):.1f}")
    print(f"YES resolution rate = {yes.mean()*100:.1f}%  vs mean entry yes_price = {qs.mean():.1f}c")

    # ---------- event-clustered bootstrap ----------
    rng = np.random.default_rng(42)
    ev_list = sorted(set(events))
    ev_idx = {e: np.where(events == e)[0] for e in ev_list}
    reps = 10_000
    means = np.empty(reps)
    ne = len(ev_list)
    for i in range(reps):
        pick = rng.integers(0, ne, ne)
        idx = np.concatenate([ev_idx[ev_list[j]] for j in pick])
        means[i] = pnl[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    p_le0 = (means <= 0).mean()
    p_two = 2 * min(p_le0, 1 - p_le0)
    print(f"\nevent-clustered bootstrap ({reps} reps): "
          f"CI95 [{lo:+.2f}, {hi:+.2f}]  P(mean<=0)={p_le0:.5f}  two-sided p~{p_two:.5f}"
          f"{'  (<1e-4)' if p_le0 < 1e-4 else ''}")

    # ---------- price buckets ----------
    print(f"\n=== PRICE BUCKETS (entry yes_price) ===")
    for lo_b, hi_b, label in [(20, 40, "20-40"), (40, 60, "40-60"), (60, 80, "60-80")]:
        # half-open [lo,hi) except last bucket closed at 80
        mask = (qs >= lo_b) & ((qs < hi_b) if hi_b != 80 else (qs <= 80))
        if mask.sum() == 0:
            continue
        print(f"  {label}c: n={mask.sum():5d}  mean={pnl[mask].mean():+6.2f}c  "
              f"yes_rate={yes[mask].mean()*100:5.1f}%  mean_q={qs[mask].mean():5.1f}c")

    # ---------- series breakdown ----------
    print(f"\n=== SERIES (n>=10) ===")
    by_series = defaultdict(list)
    for r in rows:
        by_series[r["series"]].append(r)
    for s, rs in sorted(by_series.items(), key=lambda kv: -len(kv[1])):
        if len(rs) < 10:
            continue
        p = np.array([r["pnl"] for r in rs])
        y = np.mean([r["result"] == "yes" for r in rs])
        qq = np.mean([r["q"] for r in rs])
        nev = len(set(r["event"] for r in rs))
        print(f"  {s:32s} n={len(p):5d} ev={nev:4d} mean={p.mean():+7.2f}c "
              f"yes={y*100:5.1f}% q={qq:5.1f}c")
    earn = np.array([r["pnl"] for r in rows if r["series"].startswith("KXEARNINGSMENTION")])
    if len(earn):
        print(f"  [agg] KXEARNINGSMENTION*             n={len(earn):5d} mean={earn.mean():+7.2f}c")
    non_earn = np.array([r["pnl"] for r in rows if not r["series"].startswith("KXEARNINGSMENTION")])
    if len(non_earn):
        print(f"  [agg] all other series               n={len(non_earn):5d} mean={non_earn.mean():+7.2f}c")

    # ---------- sensitivity ----------
    print(f"\n=== SENSITIVITY ===")
    # (a) include markets that resolved before entry (should be worse / lookahead)
    # recompute quickly
    all_pnl = list(pnl)
    for tk, m in markets.items():
        end = event_end[m["event_ticker"]]
        deadline = end - timedelta(minutes=60)
        window_start = end - timedelta(minutes=180)
        tape = trades.get(tk)
        if not tape:
            continue
        last = None
        for dt, q in tape:
            if dt < deadline:
                last = (dt, q)
            else:
                break
        if last is None or last[0] < window_start:
            continue
        entry_time, q = last
        if not (20.0 <= q <= 80.0):
            continue
        if m["close_time"] >= entry_time:
            continue  # already counted
        p_no = (100.0 - q) / 100.0
        fee = math.ceil(0.07 * p_no * (1.0 - p_no) * 100.0)
        cost = (100.0 - q) + 4.0 + fee
        payout = 100.0 if m["result"] == "no" else 0.0
        all_pnl.append(payout - cost)
    all_pnl = np.array(all_pnl)
    print(f"  without resolved-before-entry exclusion: n={len(all_pnl)} mean={all_pnl.mean():+.2f}c")
    # (b) window edge variants
    for pre_lo, pre_hi in [(60, 180), (60, 120), (90, 180), (30, 180), (60, 240)]:
        sel = []
        for tk, m in markets.items():
            end = event_end[m["event_ticker"]]
            deadline = end - timedelta(minutes=pre_lo)
            window_start = end - timedelta(minutes=pre_hi)
            tape = trades.get(tk)
            if not tape:
                continue
            last = None
            for dt, q in tape:
                if dt < deadline:
                    last = (dt, q)
                else:
                    break
            if last is None or last[0] < window_start:
                continue
            entry_time, q = last
            if not (20.0 <= q <= 80.0):
                continue
            if m["close_time"] < entry_time:
                continue
            p_no = (100.0 - q) / 100.0
            fee = math.ceil(0.07 * p_no * (1.0 - p_no) * 100.0)
            cost = (100.0 - q) + 4.0 + fee
            payout = 100.0 if m["result"] == "no" else 0.0
            sel.append(payout - cost)
        sel = np.array(sel)
        print(f"  window {pre_lo}-{pre_hi}min pre-end: n={len(sel)} mean={sel.mean():+.2f}c")


if __name__ == "__main__":
    main()
