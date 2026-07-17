#!/usr/bin/env python3
"""Round-3 PRE-REGISTERED out-of-sample test of the book-filtered
family-2b taker-NO rule on Kalshi mention markets.

FROZEN RULE (registered 2026-07-17 before fetching any new data):
  Signal: last tape print per market with ts in [60,180) min before the
  ex-post event end (max close_time across the event's markets, computed
  on the tape-merged frame exactly as research_tape_micro.family2b does),
  print at/before the market's own close_time (market still open),
  20 <= yes_price <= 80, speech family EXCLUDED (open-ended-speech series;
  broadcast/sports/TV/hearings/earnings INCLUDED).
  Book: last minute candle with a yes_bid close at or before the signal
  moment, within 10 min; else no quote -> skip (counted separately).
  NO ask = 100 - yes_bid.
  FILTER: trade only if NO ask <= (100 - last_print) + 4.
  Entry cost = NO ask + ceil(0.07*P*(1-P)) cents (P = NO price, dollars).
  P&L = (100 if result=="no" else 0) - cost, held to settlement.
  Success bar: filtered mean > 0, event-clustered bootstrap (10k reps)
  p(<=0) < 0.01, n >= 30 filtered trades.

SAMPLING: from the full signal population, exclude every ticker whose
signal-minute book already exists on disk (minute_candles.jsonl /
minute_targets.json — the round-2 audit sample), then draw 250 uniformly
at random with seed 20260717.

Usage:
  research_r3_oos_bookfilter.py signals   # build population + sample
  research_r3_oos_bookfilter.py fetch     # fetch minute candles (resumable)
  research_r3_oos_bookfilter.py eval      # apply frozen rule, report
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from research_mentions_lib import DATA, load_markets, taker_fee_cents  # noqa: E402
from research_tape_micro import cluster_bootstrap, load_trades_full  # noqa: E402

BASE = "https://api.elections.kalshi.com/trade-api/v2"
SAMPLE_F = DATA / "r3oos_sample.json"
CANDLES_F = DATA / "minute_candles_r3oos.jsonl"
SEED_SAMPLE = 20260717
N_SAMPLE = 250
N_BOOT = 10_000

# Frozen speech/open-ended-speech exclusion (series-level).
EXCLUDE_SERIES = {
    "KXTRUMPMENTION", "KXTRUMPMENTIONB", "KXVANCEMENTION",
    "KXMAMDANIMENTION", "KXSECPRESSMENTION", "KXPOLITICSMENTION",
}
# pattern catch-all for any other open-ended-speech series
EXCLUDE_PAT = ("TRUMP", "VANCE", "MAMDANI", "PRESSMENTION",
               "POLITICSMENTION", "SAY")


def is_speech(series: str) -> bool:
    return series in EXCLUDE_SERIES or any(p in series for p in EXCLUDE_PAT)


def fam(series: str) -> str:
    if series.startswith("KXEARNINGSMENTION"):
        return "earnings"
    if series.startswith("KXHEARINGMENTION"):
        return "hearings"
    return "broadcast/sports/TV"


# ------------------------------------------------------------- signals

def build_signals() -> None:
    mk = load_markets().drop_duplicates(subset=["ticker"])
    tr = load_trades_full()
    j = tr.merge(mk[["ticker", "result", "event_ticker", "series",
                     "close_time"]], on="ticker", how="inner")
    ev_end = j.groupby("event_ticker").close_time.max().rename("ev_end")
    j = j.join(ev_end, on="event_ticker")
    j["mte"] = (j.ev_end - j.ts).dt.total_seconds() / 60.0
    w = j[(j.mte >= 60) & (j.mte < 180) & (j.ts <= j.close_time)]
    last = w.sort_values("ts", kind="stable").groupby("ticker").tail(1)
    g = last[last.yes_price.between(20, 80)].copy()
    print(f"signal population (all series, pre-exclusion): {len(g)} markets, "
          f"{g.event_ticker.nunique()} events")
    n_speech = int(g.series.map(is_speech).sum())
    g = g[~g.series.map(is_speech)]
    print(f"  speech-family excluded: {n_speech} -> non-speech population "
          f"{len(g)} markets, {g.event_ticker.nunique()} events")
    print("  series counts:")
    print(g.series.value_counts().to_string())

    covered = set(json.load((DATA / "minute_targets.json").open()))
    for line in (DATA / "minute_candles.jsonl").open():
        try:
            covered.add(json.loads(line)["ticker"])
        except Exception:  # noqa: BLE001
            pass
    n_cov = int(g.ticker.isin(covered).sum())
    g = g[~g.ticker.isin(covered)]
    print(f"  already-covered (r2 minute books) excluded: {n_cov} -> "
          f"never-covered population {len(g)} markets, "
          f"{g.event_ticker.nunique()} events")

    tks = sorted(g.ticker)
    rng = np.random.default_rng(SEED_SAMPLE)
    take = (list(rng.choice(tks, size=N_SAMPLE, replace=False))
            if len(tks) > N_SAMPLE else tks)
    gs = g[g.ticker.isin(take)]
    out = {
        "population_all": int(len(last[last.yes_price.between(20, 80)])),
        "n_speech_excluded": n_speech,
        "n_covered_excluded": n_cov,
        "population_final": len(tks),
        "sample": {
            r.ticker: {
                "signal_ts": int(r.ts.timestamp()),
                "yes_price": float(r.yes_price),
                "event_ticker": r.event_ticker,
                "series": r.series,
                "result": int(r.result),
                "close_time": int(r.close_time.timestamp()),
                "ev_end": int(r.ev_end.timestamp()),
                "mte": float(r.mte),
            } for r in gs.itertuples()
        },
    }
    SAMPLE_F.write_text(json.dumps(out))
    print(f"sample: {len(out['sample'])} markets, "
          f"{gs.event_ticker.nunique()} events -> {SAMPLE_F}")


# ------------------------------------------------------------- fetch

_last = [0.0]


def get(path: str, params: dict, tries: int = 8):
    url = BASE + path + "?" + urllib.parse.urlencode(params)
    for i in range(tries):
        wait = 0.80 - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "hedge-research/0.1"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            code = getattr(e, "code", None)
            if code == 404:
                return None
            back = min(90, 2 ** i * (8 if code == 429 else 2))
            time.sleep(back)
    return None


def fetch() -> None:
    sample = json.load(SAMPLE_F.open())["sample"]
    done = set()
    if CANDLES_F.exists():
        for line in CANDLES_F.open():
            try:
                done.add(json.loads(line)["ticker"])
            except Exception:  # noqa: BLE001
                pass
    todo = [(t, v) for t, v in sample.items() if t not in done]
    print(f"{len(todo)} fetches ({len(done)} done)", flush=True)
    with CANDLES_F.open("a") as fh:
        for i, (tk, v) in enumerate(todo):
            series = tk.split("-")[0]
            d = get(f"/series/{series}/markets/{tk}/candlesticks",
                    {"start_ts": v["signal_ts"] - 1500,
                     "end_ts": v["signal_ts"] + 120,
                     "period_interval": 1})
            fh.write(json.dumps({"ticker": tk,
                                 "candles": (d or {}).get("candlesticks", []),
                                 "ok": d is not None}) + "\n")
            fh.flush()
            if i % 25 == 0:
                print(f"{i}/{len(todo)}", flush=True)
    print("FETCH DONE", flush=True)


# ------------------------------------------------------------- eval

def stat(net, ev, label):
    st = cluster_bootstrap(np.asarray(net, dtype=float), np.asarray(ev),
                           n_boot=N_BOOT, seed=7)
    print(f"  {label}: n={st['n']} ({st['n_clusters']} ev) "
          f"mean={st['mean']:+.2f}c CI95=[{st['ci_lo']:+.2f},"
          f"{st['ci_hi']:+.2f}] p(<=0)={st['p_le_0']:.4f}")
    return st


def evaluate() -> None:
    meta = json.load(SAMPLE_F.open())
    sample = meta["sample"]
    books: dict[str, np.ndarray] = {}
    n_fail = 0
    for line in CANDLES_F.open():
        rec = json.loads(line)
        if not rec.get("ok"):
            n_fail += 1
        rows = []
        for c in rec.get("candles") or []:
            yb = (c.get("yes_bid") or {}).get("close_dollars")
            if yb is None:
                continue
            ya = (c.get("yes_ask") or {}).get("close_dollars")
            rows.append((int(c["end_period_ts"]), float(yb) * 100.0,
                         float(ya) * 100.0 if ya is not None else np.nan))
        if rows:
            books[rec["ticker"]] = np.array(sorted(rows))

    rows = []
    n_noquote = 0
    for tk, v in sample.items():
        arr = books.get(tk)
        sig = v["signal_ts"]
        if arr is None:
            n_noquote += 1
            continue
        k = np.searchsorted(arr[:, 0], sig, side="right") - 1
        if k < 0 or sig - arr[k, 0] > 600:
            n_noquote += 1
            continue
        yes_bid = arr[k, 1]
        no_ask = 100.0 - yes_bid
        rows.append({**v, "ticker": tk, "yes_bid": yes_bid, "no_ask": no_ask,
                     "yes_ask": arr[k, 2],
                     "quote_age_s": sig - arr[k, 0]})
    d = pd.DataFrame(rows)
    n = len(sample)
    print("=" * 72)
    print("R3 OOS book-filtered family-2b rule — FROZEN, seed", SEED_SAMPLE)
    print(f"population(all-series 20-80c last-print) = {meta['population_all']}"
          f"; speech excluded = {meta['n_speech_excluded']}; "
          f"r2-covered excluded = {meta['n_covered_excluded']}; "
          f"final population = {meta['population_final']}")
    print(f"sample = {n}; fetch-failures = {n_fail}; "
          f"no quote within 10 min = {n_noquote}; "
          f"with quote = {len(d)} ({len(d)/n:.1%} coverage)")
    if not len(d):
        return
    zero_bid = d[d.yes_bid < 1]
    print(f"  zero/sub-1c yes_bid at signal: {len(zero_bid)}")

    d["assumed"] = (100.0 - d.yes_price + 4.0).clip(1, 99)
    d["slip"] = d.no_ask - (100.0 - d.yes_price)
    d["fee"] = d.no_ask.clip(1, 99).map(taker_fee_cents)
    d["pnl"] = np.where(d.result == 0, 100.0, 0.0) - d.no_ask - d.fee
    afee = d.assumed.map(taker_fee_cents)
    d["pnl_assumed"] = np.where(d.result == 0, 100.0, 0.0) - d.assumed - afee

    keep = d[d.no_ask <= d.assumed]  # NO ask <= (100-p)+4
    print(f"\nFILTER keep-rate: {len(keep)}/{len(d)} = {len(keep)/len(d):.1%}")
    print("\n-- pre-registered headline --")
    st = stat(keep.pnl, keep.event_ticker, "FILTERED, real NO ask + fee")
    ok = (st["mean"] > 0 and st["p_le_0"] < 0.01 and st["n"] >= 30)
    unf = d[d.yes_bid >= 1]
    stat(unf.pnl, unf.event_ticker,
         "UNFILTERED repriced (real ask, yes_bid>=1)")
    stat(d.pnl_assumed, d.event_ticker,
         "same sample, assumed (100-p)+4c cost   ")
    stat(keep.pnl_assumed, keep.event_ticker,
         "filtered subset, assumed cost          ")

    print("\n-- kept trades: YES-resolution rate vs last-print price --")
    for plo, phi in [(20, 40), (40, 60), (60, 80)]:
        gg = keep[keep.yes_price.between(plo, phi)]
        if not len(gg):
            continue
        print(f"  [{plo}-{phi}c] n={len(gg)} YES rate={gg.result.mean():.3f} "
              f"mean px={gg.yes_price.mean():.1f}c mean cost="
              f"{(gg.no_ask+gg.fee).mean():.1f}c mean pnl={gg.pnl.mean():+.2f}")

    print("\n-- per-family (kept) --")
    keep = keep.assign(family=keep.series.map(fam))
    for f, gg in keep.groupby("family"):
        if len(gg) >= 5:
            stat(gg.pnl, gg.event_ticker, f"[{f}] ")
        else:
            print(f"  [{f}] n={len(gg)} pnl mean="
                  f"{gg.pnl.mean():+.2f} (too few)")
    print("\n-- per-series (kept, n>=5) --")
    print(keep.groupby("series").agg(
        n=("pnl", "size"), ev=("event_ticker", "nunique"),
        yes_rate=("result", "mean"), mean_pnl=("pnl", "mean"))
        .sort_values("n", ascending=False).round(2).to_string())

    print("\n-- assumed vs real cost (all quoted) --")
    print("  slip = real NO ask - (100 - last_print), cents:")
    print(d.slip.describe(percentiles=[.1, .25, .5, .75, .9])
          .round(2).to_string())
    print(f"  quote age at signal: median={d.quote_age_s.median():.0f}s "
          f"p90={d.quote_age_s.quantile(.9):.0f}s")
    spr = (d.yes_ask - d.yes_bid).dropna()
    if len(spr):
        print(f"  quoted YES spread at signal: median={spr.median():.1f}c "
              f"p90={spr.quantile(.9):.1f}c")

    print("\n" + "=" * 72)
    print("VERDICT:", "OOS-PASS" if ok else "OOS-FAIL",
          f"(bar: filtered mean>0, p<0.01, n>=30; got mean={st['mean']:+.2f}, "
          f"p={st['p_le_0']:.4f}, n={st['n']})")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "eval"
    {"signals": build_signals, "fetch": fetch, "eval": evaluate}[cmd]()
