#!/usr/bin/env python3
"""Sibling-fade validation: after a mid-event YES resolution in a mention event,
buy 1 NO on each still-open sibling (pre-registered rule, frozen 2026-07-17).

Subcommands:
  plan   -- enumerate triggers, check existing coverage, write fetch plan
  fetch  -- fetch minute candles for planned siblings (<=600 requests, polite)
  eval   -- run the pre-registered test on all available data

Data:
  in : data/research/mentions/events.jsonl, minute_candles*.jsonl
  out: data/research/mentions/minute_candles_siblingfade.jsonl
       data/research/mentions/siblingfade_plan.json
"""
from __future__ import annotations

import datetime as dt
import json
import math
import random
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

BASE = "https://api.elections.kalshi.com/trade-api/v2"
OUT = Path(__file__).resolve().parents[1] / "data" / "research" / "mentions"
PLAN = OUT / "siblingfade_plan.json"
NEWC = OUT / "minute_candles_siblingfade.jsonl"
SEED = 20260717
BUDGET = 580  # request budget (<=600 total incl. retries slack)
_last = [0.0]


def ts(s: str) -> int:
    return int(dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())


def load_events():
    """-> list of (event_ticker, series_ticker, [(ticker, result, close_ts)])"""
    evs = []
    for line in (OUT / "events.jsonl").open():
        e = json.loads(line)
        mm = e.get("markets")
        if not mm:
            continue
        ms = [(m["ticker"], m["result"], ts(m["close_time"]))
              for m in mm if m.get("result") in ("yes", "no")]
        if ms:
            evs.append((e["event_ticker"], e.get("series_ticker", ""), ms))
    return evs


def enumerate_triggers(evs):
    """All (evt, ser, t0, sib, sib_close, sib_result, end, trig_idx) pairs.

    Trigger: market resolves YES at t0, end - t0 >= 1800.
    Sibling: still open at t0+120.
    trig_idx: 0 for the sibling's FIRST trigger (primary), 1+ for later ones.
    """
    rows = []
    for evt, ser, ms in evs:
        end = max(c for _, _, c in ms)
        trigs = sorted([(c, t) for t, r, c in ms if r == "yes" and end - c >= 1800])
        if not trigs:
            continue
        seen = defaultdict(int)
        for t0, trig in trigs:
            for sib, sres, sct in ms:
                if sib == trig or sct <= t0 + 120:
                    continue
                rows.append(dict(event=evt, series=ser, trigger=trig, t0=t0,
                                 sib=sib, sib_close=sct, sib_result=sres,
                                 end=end, trig_idx=seen[sib]))
                seen[sib] += 1
    return rows


def scan_candle_file(path: Path, want: set[str], store: dict):
    if not path.exists():
        return
    for line in path.open():
        try:
            d = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        tk = d.get("ticker")
        if tk not in want:
            continue
        store.setdefault(tk, []).extend(d.get("candles") or [])


def existing_files():
    return [OUT / "minute_candles.jsonl", OUT / "minute_candles_r3oos.jsonl"]


def cmd_plan():
    evs = load_events()
    rows = enumerate_triggers(evs)
    prim = [r for r in rows if r["trig_idx"] == 0]
    n_trig = len({(r["event"], r["trigger"], r["t0"]) for r in rows})
    print(f"population: {len(rows)} trigger-sib pairs, {len(prim)} primary "
          f"(first-trigger) pairs, {n_trig} triggers, "
          f"{len({r['event'] for r in rows})} events")

    # coverage of primary pairs by existing minute files (range check only)
    want = {r["sib"] for r in prim}
    rng = {}
    for p in existing_files():
        for line in p.open():
            try:
                d = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            tk = d.get("ticker")
            if tk not in want:
                continue
            cs = d.get("candles") or []
            if not cs:
                continue
            tt = [c["end_period_ts"] for c in cs]
            lo, hi = min(tt), max(tt)
            if tk in rng:
                rng[tk] = (min(rng[tk][0], lo), max(rng[tk][1], hi))
            else:
                rng[tk] = (lo, hi)

    def covered(r):
        v = rng.get(r["sib"])
        return v and v[0] <= r["t0"] + 120 and v[1] >= r["t0"] + 600

    cov = [r for r in prim if covered(r)]
    uncov = [r for r in prim if not covered(r)]
    print(f"already covered: {len(cov)} pairs in "
          f"{len({r['event'] for r in cov})} events; "
          f"uncovered: {len(uncov)} pairs in "
          f"{len({r['event'] for r in uncov})} events")

    # sibling fetch windows: first-trigger t0 minus 2h (control pre-window)
    # through last trigger + 1h, capped at sib close and 4900 minutes.
    lastt0 = defaultdict(int)
    for r in rows:
        k = (r["event"], r["sib"])
        lastt0[k] = max(lastt0[k], r["t0"])
    win = {}
    for r in uncov:
        k = (r["event"], r["sib"])
        lo = r["t0"] - 7200
        hi = min(lastt0[k] + 3600, r["sib_close"] + 60)
        hi = min(hi, lo + 4900 * 60)
        win[r["sib"]] = (lo, hi, r["event"])

    # sample at EVENT level, seed frozen, until budget
    ev_sibs = defaultdict(list)
    for sib, (lo, hi, evt) in win.items():
        ev_sibs[evt].append(sib)
    ev_order = sorted(ev_sibs)
    random.Random(SEED).shuffle(ev_order)
    chosen, n_req = [], 0
    for evt in ev_order:
        sibs = ev_sibs[evt]
        if n_req + len(sibs) > BUDGET:
            continue
        n_req += len(sibs)
        chosen.append(evt)
    plan = {s: win[s][:2] for e in chosen for s in ev_sibs[e]}
    print(f"fetch plan: {len(plan)} requests over {len(chosen)} events "
          f"(of {len(ev_order)} uncovered events -> "
          f"{len(chosen)/len(ev_order):.1%} event coverage)")
    PLAN.write_text(json.dumps({"windows": plan, "events": chosen,
                                "n_uncovered_events": len(ev_order)}))


def get(path: str, params: dict, tries: int = 8):
    url = BASE + path + "?" + urllib.parse.urlencode(params)
    for i in range(tries):
        wait = 0.85 - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "hedge-research/0.1"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            if getattr(e, "code", None) == 404:
                return None
            time.sleep(min(90, 2 ** i * 3))
    return None


def cmd_fetch():
    plan = json.loads(PLAN.read_text())["windows"]
    done = set()
    if NEWC.exists():
        for line in NEWC.open():
            try:
                done.add(json.loads(line)["ticker"])
            except Exception:  # noqa: BLE001
                pass
    todo = sorted((t, w) for t, w in plan.items() if t not in done)
    print(f"{len(todo)} fetches ({len(done)} done)", flush=True)
    with NEWC.open("a") as fh:
        for i, (tk, (lo, hi)) in enumerate(todo):
            series = tk.split("-")[0]
            d = get(f"/series/{series}/markets/{tk}/candlesticks",
                    {"start_ts": lo, "end_ts": hi, "period_interval": 1})
            fh.write(json.dumps({"ticker": tk,
                                 "candles": (d or {}).get("candlesticks", []),
                                 "ok": d is not None}) + "\n")
            fh.flush()
            if i % 50 == 0:
                print(f"{i}/{len(todo)}", flush=True)
    print("FETCH DONE", flush=True)


# ---------------------------------------------------------------- evaluation

def cents(x):
    return None if x is None else round(float(x) * 100, 2)


def bar_quotes(c):
    """-> (yes_bid_close, yes_ask_close, price_close) in cents (None-able)."""
    yb = (c.get("yes_bid") or {}).get("close_dollars")
    ya = (c.get("yes_ask") or {}).get("close_dollars")
    pc = (c.get("price") or {}).get("close_dollars")
    return cents(yb), cents(ya), cents(pc)


def quote_price(c):
    """Universe price: bid/ask mid, else last close. None if neither."""
    yb, ya, pc = bar_quotes(c)
    if yb is not None and ya is not None:
        return (yb + ya) / 2
    return pc


def fee_cents(price_cents):
    return math.ceil(0.07 * price_cents * (100 - price_cents) / 100)


def family(series):
    s = series.upper()
    if any(k in s for k in ("WCMENTION", "MLB", "NBA", "NHL", "FIGHT", "NFL",
                            "SOCCER", "TENNIS", "GOLF", "F1", "UFC")):
        return "sports"
    if "EARNINGS" in s:
        return "earnings"
    if any(k in s for k in ("TRUMP", "MAMDANI", "HEARING", "POLITIC", "VANCE",
                            "FOXNEWS", "LASTWORD", "WHITEHOUSE", "PRESSER",
                            "SENATE", "GOV", "MAYOR", "DEBATE", "POTUS", "JD",
                            "NEWSOM", "AOC", "CUOMO", "SLIWA", "BONDI",
                            "HEGSETH", "RUBIO", "LEAVITT", "MTG", "KAMALA")):
        return "politics"
    return "entertainment"


def load_all_candles(want):
    """ticker -> sorted [(end_ts, bar)] merged across files (new file last=wins)."""
    store = {}
    for p in existing_files() + [NEWC]:
        scan_candle_file(p, want, store)
    out = {}
    for tk, cs in store.items():
        by = {}
        for c in cs:
            by[c["end_period_ts"]] = c
        out[tk] = sorted(by.items())
    return out


def find_entry(bars, t0):
    """First bar with end_ts in (t0+120, t0+600] and non-null yes_bid."""
    for et, c in bars:
        if et <= t0 + 120:
            continue
        if et > t0 + 600:
            break
        yb, ya, _ = bar_quotes(c)
        if yb is not None:
            return et, yb, ya
    return None


def universe_quote(bars, t0):
    """Most recent bar at/just after t0+120 (within [t0-600, t0+180])."""
    best = None
    for et, c in bars:
        if t0 - 600 <= et <= t0 + 180:
            q = quote_price(c)
            if q is not None:
                best = (et, q)
        elif et > t0 + 180:
            break
    return best


def bootstrap_p(pnls, events, reps=10000, seed=SEED):
    """Event-clustered bootstrap of the mean. Returns (mean, lo, hi, p_onesided)."""
    import numpy as np
    ev = sorted(set(events))
    by = {e: [] for e in ev}
    for p, e in zip(pnls, events):
        by[e].append(p)
    arrs = [np.array(by[e], dtype=float) for e in ev]
    rng = np.random.default_rng(seed)
    n = len(arrs)
    means = np.empty(reps)
    for i in range(reps):
        idx = rng.integers(0, n, n)
        cat = np.concatenate([arrs[j] for j in idx])
        means[i] = cat.mean()
    mean = float(np.mean(np.concatenate(arrs)))
    lo, hi = np.percentile(means, [0.5, 99.5])
    p = float((means <= 0).mean())
    return mean, float(lo), float(hi), p


def cmd_eval():
    import numpy as np
    evs = load_events()
    rows = enumerate_triggers(evs)
    want = {r["sib"] for r in rows}
    candles = load_all_candles(want)
    print(f"candle data for {len(candles)}/{len(want)} sibling tickers")

    entries = []          # primary (trig_idx==0)
    later_entries = []    # trig_idx>0
    n_open_noquote = 0
    n_band_reject = 0
    n_no_entrybar = 0
    for r in rows:
        bars = candles.get(r["sib"])
        if not bars:
            continue
        uq = universe_quote(bars, r["t0"])
        if uq is None:
            n_open_noquote += 1
            continue
        if not (15 <= uq[1] <= 85):
            n_band_reject += 1
            continue
        ent = find_entry(bars, r["t0"])
        if ent is None:
            n_no_entrybar += 1
            continue
        et, yb, ya = ent
        no_ask = 100 - yb
        cost = no_ask + fee_cents(no_ask)
        pnl = (100 if r["sib_result"] == "no" else 0) - cost
        rec = dict(r, entry_ts=et, yes_bid=yb, yes_ask=ya, no_ask=no_ask,
                   cost=cost, pnl=pnl, uprice=uq[1], fam=family(r["series"]))
        # YES-side mirror (only if ask quoted)
        if ya is not None:
            ycost = ya + fee_cents(ya)
            rec["pnl_yes"] = (100 if r["sib_result"] == "yes" else 0) - ycost
        (entries if r["trig_idx"] == 0 else later_entries).append(rec)

    print(f"universe checks: no-quote={n_open_noquote} band-reject={n_band_reject} "
          f"no-entry-bar={n_no_entrybar}")
    print(f"primary entries: {len(entries)} in "
          f"{len({e['event'] for e in entries})} events; "
          f"later-trigger entries: {len(later_entries)}")
    if not entries:
        return

    pnls = [e["pnl"] for e in entries]
    evts = [e["event"] for e in entries]
    mean, lo, hi, p = bootstrap_p(pnls, evts)
    n_ev = len(set(evts))
    print(f"\nPRIMARY: mean={mean:+.2f}c/contract 99%CI=[{lo:+.2f},{hi:+.2f}] "
          f"p(one-sided,<=0)={p:.5f} n={len(pnls)} markets, {n_ev} events")
    wr = np.mean([e["sib_result"] == "no" for e in entries])
    print(f"  NO win rate={wr:.3f}, mean cost={np.mean([e['cost'] for e in entries]):.1f}c")
    bar = "VALIDATES" if (p < 0.01 and n_ev >= 25 and mean > 0) else (
        "UNDERPOWERED" if n_ev < 25 else "FAILS")
    print(f"  pre-registered bar (p<0.01, n>=25 events): {bar}")

    # YES-side mirror
    ye = [e for e in entries if "pnl_yes" in e]
    if ye:
        m2, l2, h2, p2 = bootstrap_p([e["pnl_yes"] for e in ye],
                                     [e["event"] for e in ye], seed=SEED + 1)
        print(f"\nYES-side entry (buy YES at ask): mean={m2:+.2f}c "
              f"99%CI=[{l2:+.2f},{h2:+.2f}] p={p2:.5f} n={len(ye)}")

    # later-trigger entries
    if later_entries:
        m3, l3, h3, p3 = bootstrap_p([e["pnl"] for e in later_entries],
                                     [e["event"] for e in later_entries],
                                     seed=SEED + 2)
        print(f"later-trigger entries: mean={m3:+.2f}c 99%CI=[{l3:+.2f},{h3:+.2f}] "
              f"p={p3:.5f} n={len(later_entries)}")

    # splits
    def split(keyf, label):
        groups = defaultdict(list)
        for e in entries:
            groups[keyf(e)].append(e)
        print(f"\nby {label}:")
        for k in sorted(groups):
            g = groups[k]
            gm = np.mean([e["pnl"] for e in g])
            print(f"  {k:>16}: mean={gm:+6.2f}c n={len(g):4d} "
                  f"events={len({e['event'] for e in g}):3d} "
                  f"winrate={np.mean([e['sib_result']=='no' for e in g]):.3f}")

    split(lambda e: e["fam"], "family")
    split(lambda e: ("<1h" if e["end"] - e["t0"] < 3600 else
                     "1-3h" if e["end"] - e["t0"] < 10800 else ">3h"),
          "time remaining at trigger")
    split(lambda e: ("15-35" if e["uprice"] < 35 else
                     "35-65" if e["uprice"] < 65 else "65-85"),
          "sibling YES price bucket")

    # ------------------------------------------------------------- control
    # matched non-triggered moments: same sibling, random minute with quote in
    # [15,85], >=30min from sibling close, no trigger in the event within the
    # prior 30 min or next 10 min. Entry mechanics identical.
    trig_times = defaultdict(list)
    for r in rows:
        trig_times[r["event"]].append(r["t0"])
    rngc = random.Random(SEED + 3)
    controls = []
    sibs_entered = {(e["event"], e["sib"]) for e in entries}
    sib_meta = {(r["event"], r["sib"]): r for r in rows if r["trig_idx"] == 0}
    for (evt, sib) in sorted(sibs_entered):
        bars = candles.get(sib)
        r = sib_meta[(evt, sib)]
        tt = trig_times[evt]
        cand = []
        for i, (et, c) in enumerate(bars):
            if et > r["sib_close"] - 1800:
                break
            if any(-600 <= et - t0 <= 1800 for t0 in tt):
                continue
            q = quote_price(c)
            if q is None or not (15 <= q <= 85):
                continue
            ent = find_entry(bars, et - 120)  # same (t+2,t+10] machinery, t=et-120
            if ent is None:
                continue
            cand.append((et, ent))
        if not cand:
            continue
        et, (ets, yb, ya) = rngc.choice(cand)
        no_ask = 100 - yb
        cost = no_ask + fee_cents(no_ask)
        pnl = (100 if r["sib_result"] == "no" else 0) - cost
        controls.append(dict(event=evt, sib=sib, pnl=pnl, cost=cost))
    if controls:
        m4, l4, h4, p4 = bootstrap_p([c["pnl"] for c in controls],
                                     [c["event"] for c in controls],
                                     seed=SEED + 4)
        print(f"\nCONTROL (matched non-triggered NO buys): mean={m4:+.2f}c "
              f"99%CI=[{l4:+.2f},{h4:+.2f}] p={p4:.5f} n={len(controls)} "
              f"({len({c['event'] for c in controls})} events)")
        # paired diff on siblings present in both
        pk = {(c["event"], c["sib"]): c["pnl"] for c in controls}
        diffs, devt = [], []
        for e in entries:
            k = (e["event"], e["sib"])
            if k in pk:
                diffs.append(e["pnl"] - pk[k])
                devt.append(e["event"])
        if diffs:
            m5, l5, h5, p5 = bootstrap_p(diffs, devt, seed=SEED + 5)
            print(f"paired (trigger - control) diff: mean={m5:+.2f}c "
                  f"99%CI=[{l5:+.2f},{h5:+.2f}] p={p5:.5f} n={len(diffs)}")

    json.dump(dict(entries=entries, later=later_entries, controls=controls),
              (OUT / "siblingfade_entries.json").open("w"), default=str)
    print("\nwrote siblingfade_entries.json")


if __name__ == "__main__":
    {"plan": cmd_plan, "fetch": cmd_fetch, "eval": cmd_eval}[sys.argv[1]]()
