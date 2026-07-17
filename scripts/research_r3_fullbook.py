#!/usr/bin/env python3
"""R3 FULL-BOOK: execution-realism test of the re-anchored taker-NO rule.

Rule (pre-specified): at T = true_event_end - 60 min, for still-open mention
markets whose last tape print (within prior 120 min) is 20-80c YES, buy 1 NO
taker, hold to settlement.  This script fetches minute books for ALL
re-anchored signals (offsets -60 and -90) and prices every trade at the REAL
NO ask (= 100 - yes_bid of the last minute bar at or before T, <=10 min old).

Phases:
  plan     build signal sets, decide fetch targets (budget 2,200 requests)
  fetch    pull minute candles from the public Kalshi API (anon, >=0.85s/req)
  analyze  price everything; variants B (book gate), F (flow gate), B&F;
           slippage distributions; latency sensitivity; per-family verdicts

Run: .venv/bin/python scripts/research_r3_fullbook.py {plan|fetch|analyze}
"""
from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from research_mentions_lib import DATA, load_markets, taker_fee_cents  # noqa: E402
from research_tape_micro import cluster_bootstrap  # noqa: E402
from research_r3_reanchor import load_true_ends, entries_at  # noqa: E402

R3 = DATA / "r3"
OUT_JSONL = DATA / "minute_candles_r3full.jsonl"
PLAN_JSON = R3 / "fullbook_plan.json"
LEGACY = ["minute_candles.jsonl", "minute_candles_r3oos.jsonl",
          "minute_candles_siblingfade.jsonl"]
N_BOOT = 10_000
SPREAD_C = 4.0
BUDGET = 2_200
SEED = 20260717
BASE = "https://api.elections.kalshi.com/trade-api/v2"
PRIORITY_FAMS = ("hearings", "NHL", "NBA", "earnings")  # full coverage first
SAMPLE_FAMS = ("WC", "MLB")                             # sampled if over budget


# ------------------------------------------------------------ signal sets

def build_signals():
    te = load_true_ends()
    j = pd.read_parquet(R3 / "_j_cache.parquet")
    in_ev = te.event_ticker.isin(j.event_ticker.unique())
    te = te[in_ev].copy()
    sigs = {}
    for off in (60, 90):
        g = entries_at(j, te, off)
        g = g[["ticker", "event_ticker", "fam", "yes_price", "result",
               "taker_side", "ts", "t_entry", "close_time"]].copy()
        sigs[off] = g
        print(f"offset -{off}: {len(g)} signals, "
              f"{g.event_ticker.nunique()} events "
              f"({g.fam.value_counts().to_dict()})")
    return sigs


def needed_windows(sigs) -> dict[str, tuple[int, int]]:
    """ticker -> (lo_ts, hi_ts) unix seconds covering [T-120min, T+10min]
    for every offset the ticker signals at."""
    win: dict[str, tuple[int, int]] = {}
    for off, g in sigs.items():
        for tk, T in zip(g.ticker, g.t_entry):
            t = int(T.timestamp())
            lo, hi = t - 120 * 60, t + 10 * 60
            if tk in win:
                lo = min(lo, win[tk][0])
                hi = max(hi, win[tk][1])
            win[tk] = (lo, hi)
    return win


# ------------------------------------------------------------ minute books

def load_books(tickers: set[str], files=None) -> dict[str, tuple]:
    """ticker -> (sorted end_ts[], yes_bid[], yes_ask[]) cents."""
    acc: dict[str, dict[int, tuple]] = {}
    for fn in (files or LEGACY + [OUT_JSONL.name]):
        p = DATA / fn
        if not p.exists():
            continue
        for line in p.open():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            tk = rec.get("ticker")
            if tk not in tickers:
                continue
            d = acc.setdefault(tk, {})
            for c in rec.get("candles") or []:
                ts = c.get("end_period_ts")
                if ts is None:
                    continue
                yb = (c.get("yes_bid") or {}).get("close_dollars")
                ya = (c.get("yes_ask") or {}).get("close_dollars")
                d[ts] = (float(yb) * 100 if yb is not None else np.nan,
                         float(ya) * 100 if ya is not None else np.nan)
    out = {}
    for tk, d in acc.items():
        ts = np.array(sorted(d), dtype=np.int64)
        out[tk] = (ts, np.array([d[t][0] for t in ts]),
                   np.array([d[t][1] for t in ts]))
    return out


def covers(books, tk, T: pd.Timestamp) -> bool:
    """Existing data has a bar usable at T (end_ts in [T-600, T])."""
    if tk not in books:
        return False
    ts = books[tk][0]
    t = int(T.timestamp())
    i = np.searchsorted(ts, t, side="right") - 1
    return i >= 0 and ts[i] >= t - 600


def bar_at(books, tk, T_ts: int, stale_s: int = 600):
    """(yes_bid, yes_ask, age_s) of last bar ending at or before T."""
    if tk not in books:
        return None
    ts, yb, ya = books[tk]
    i = np.searchsorted(ts, T_ts, side="right") - 1
    if i < 0 or ts[i] < T_ts - stale_s:
        return None
    return yb[i], ya[i], T_ts - ts[i]


# ------------------------------------------------------------ plan phase

def phase_plan():
    sigs = build_signals()
    win = needed_windows(sigs)
    all_tk = set(win)
    books = load_books(all_tk, files=LEGACY + [OUT_JSONL.name])

    # a ticker is already covered if every offset-T it signals at has a bar
    need_T: dict[str, list[pd.Timestamp]] = {}
    fam_of: dict[str, str] = {}
    for off, g in sigs.items():
        for tk, T, fam in zip(g.ticker, g.t_entry, g.fam):
            need_T.setdefault(tk, []).append(T)
            fam_of[tk] = fam
    todo = [tk for tk, Ts in need_T.items()
            if not all(covers(books, tk, T) for T in Ts)]
    print(f"\npopulation: {len(all_tk)} tickers; already covered: "
          f"{len(all_tk) - len(todo)}; to fetch: {len(todo)}")
    fam_ct = pd.Series([fam_of[t] for t in todo]).value_counts()
    print(f"todo by family: {fam_ct.to_dict()}")

    sampled_note = ""
    if len(todo) > BUDGET:
        pri = [t for t in todo if fam_of[t] in PRIORITY_FAMS]
        rest = sorted(t for t in todo if fam_of[t] not in PRIORITY_FAMS)
        k = BUDGET - len(pri)
        rng = np.random.default_rng(SEED)
        keep = list(rng.choice(rest, size=k, replace=False)) if k > 0 else []
        todo = pri + keep
        sampled_note = (f"over budget: kept all {len(pri)} priority "
                        f"({'/'.join(PRIORITY_FAMS)}), sampled {len(keep)}/"
                        f"{len(rest)} of {'/'.join(SAMPLE_FAMS)} seed {SEED}")
        print(sampled_note)

    plan = {tk: [int(win[tk][0]), int(win[tk][1]), fam_of[tk]] for tk in todo}
    PLAN_JSON.write_text(json.dumps(
        {"targets": plan, "sampled_note": sampled_note}))
    print(f"wrote {len(plan)} targets -> {PLAN_JSON}")


# ------------------------------------------------------------ fetch phase

_last = [0.0]


def get(path: str, params: dict, tries: int = 8):
    url = BASE + path + "?" + urllib.parse.urlencode(params)
    for i in range(tries):
        wait = 0.85 - (time.time() - _last[0])
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
            time.sleep(min(90, (2 ** i) * 2))
    return None


def phase_fetch():
    plan = json.loads(PLAN_JSON.read_text())["targets"]
    done = set()
    if OUT_JSONL.exists():
        for line in OUT_JSONL.open():
            try:
                done.add(json.loads(line)["ticker"])
            except Exception:  # noqa: BLE001
                pass
    todo = [(tk, v) for tk, v in plan.items() if tk not in done]
    print(f"{len(todo)} fetches to go ({len(done)} already in file)",
          flush=True)
    t0 = time.time()
    with OUT_JSONL.open("a") as fh:
        for i, (tk, (lo, hi, fam)) in enumerate(todo):
            series = tk.split("-")[0]
            d = get(f"/series/{series}/markets/{tk}/candlesticks",
                    {"start_ts": lo, "end_ts": hi, "period_interval": 1})
            fh.write(json.dumps(
                {"ticker": tk, "win": [lo, hi], "fam": fam,
                 "candles": (d or {}).get("candlesticks", []),
                 "ok": d is not None}) + "\n")
            fh.flush()
            if i % 100 == 0:
                el = time.time() - t0
                print(f"{i}/{len(todo)} ({el:.0f}s)", flush=True)
    print("FETCH DONE", flush=True)


# ------------------------------------------------------------ analysis

def boot(vals, clus):
    return cluster_bootstrap(np.asarray(vals, float), np.asarray(clus),
                             n_boot=N_BOOT)


def fmt(st, extra=""):
    return (f"n={st['n']:>4} ({st['n_clusters']:>3} ev) "
            f"mean={st['mean']:+7.2f}c CI=[{st['ci_lo']:+7.2f},"
            f"{st['ci_hi']:+7.2f}] p(<=0)={st['p_le_0']:.4f}{extra}")


def report(label, df, col="pnl", extra=""):
    if not len(df):
        print(f"  {label:<40} n=0")
        return None
    st = boot(df[col], df.event_ticker)
    print(f"  {label:<40} {fmt(st, extra)}")
    return st


def price_signals(g: pd.DataFrame, books) -> pd.DataFrame:
    """Attach real-book entries at T, T+1min, T+5min to signal frame."""
    rows = []
    for r in g.itertuples():
        t = int(r.t_entry.timestamp())
        rec = dict(ticker=r.ticker, event_ticker=r.event_ticker, fam=r.fam,
                   yes_price=r.yes_price, result=int(r.result),
                   taker_side=r.taker_side, has_book=r.ticker in books,
                   yes_bid=np.nan, yes_ask=np.nan, bar_age=np.nan,
                   yb1=np.nan, yb5=np.nan)
        b = bar_at(books, r.ticker, t)
        if b is not None:
            rec["yes_bid"], rec["yes_ask"], rec["bar_age"] = b
        for lag, key in ((60, "yb1"), (300, "yb5")):
            bl = bar_at(books, r.ticker, t + lag)
            if bl is not None:
                rec[key] = bl[0]
        rows.append(rec)
    df = pd.DataFrame(rows)

    # modeled entry/pnl (exactly the re-anchor spec)
    ent_m = np.clip(100.0 - df.yes_price + SPREAD_C, 1, 99)
    df["pnl_model"] = (np.where(df.result == 0, 100.0, 0.0) - ent_m
                       - np.array([taker_fee_cents(e) for e in ent_m]))

    # real entry: NO ask = 100 - yes_bid; needs finite yes_bid > 0
    def real_pnl(yb):
        ok = np.isfinite(yb) & (yb > 0) & (yb < 100)
        ent = np.where(ok, 100.0 - yb, np.nan)
        fee = np.array([taker_fee_cents(e) if np.isfinite(e) else np.nan
                        for e in ent])
        return ok, (np.where(df.result == 0, 100.0, 0.0) - ent - fee), ent

    ok0, pnl0, ent0 = real_pnl(df.yes_bid.to_numpy(float))
    df["quote_ok"], df["pnl"], df["entry"] = ok0, pnl0, ent0
    for src, pcol in (("yb1", "pnl_t1"), ("yb5", "pnl_t5")):
        okl, pnll, _ = real_pnl(df[src].to_numpy(float))
        df[pcol] = np.where(okl, pnll, np.nan)

    df["no_ask"] = 100.0 - df.yes_bid          # real NO ask (cents)
    df["slip"] = df.no_ask - (100.0 - df.yes_price)  # vs last-print-implied
    df["spread"] = df.yes_ask - df.yes_bid
    df["gate_B"] = df.no_ask <= (100.0 - df.yes_price) + SPREAD_C
    df["gate_F"] = df.taker_side == "yes"
    return df


def analyze_offset(off, g, books):
    print("\n" + "=" * 78)
    print(f"OFFSET -{off} MIN — {len(g)} signals, "
          f"{g.event_ticker.nunique()} events")
    df = price_signals(g, books)

    # ---- coverage
    nb = df.has_book.sum()
    bar = df.yes_bid.notna().sum()  # bar existed at T (bid may be 0/NaN)
    q = df.quote_ok.sum()
    print(f"coverage: candle-file {nb}/{len(df)} "
          f"({nb / len(df):.1%}); bar at T (<=10min old) "
          f"{bar}/{len(df)} ({bar / len(df):.1%}); usable YES-bid quote "
          f"{q}/{len(df)} ({q / len(df):.1%})")
    nofill = df[~df.quote_ok & df.has_book]
    print(f"no-fill (fetched but empty/absent book at T): {len(nofill)} "
          f"({nofill.fam.value_counts().to_dict()})  <- no taker entry "
          f"exists for these")
    for fam, gg in df.groupby("fam"):
        print(f"  [{fam:<9}] fetched {gg.has_book.mean():.1%} "
              f"quote@T {gg.quote_ok.mean():.1%} (n={len(gg)})")

    fill = df[df.quote_ok].copy()
    if not len(fill):
        print("no priced signals")
        return df

    # ---- primary
    print(f"\nPRIMARY — real NO-ask entry at T (mean entry "
          f"{fill.entry.mean():.1f}c vs modeled "
          f"{(100 - fill.yes_price + SPREAD_C).mean():.1f}c):")
    report("ALL real fill", fill, extra=f"  YES={fill.result.mean():.3f}")
    report("ALL modeled, same covered set", fill, col="pnl_model")
    for fam, gg in fill.groupby("fam"):
        report(f"[{fam:<9}] real fill", gg,
               extra=f"  YES={gg.result.mean():.3f}")
        report(f"[{fam:<9}] modeled (same set)", gg, col="pnl_model")

    # ---- variants
    for name, mask in (("B (NO ask <= implied+4c)", fill.gate_B),
                       ("F (last print taker-YES)", fill.gate_F),
                       ("B&F", fill.gate_B & fill.gate_F)):
        kept, skip = fill[mask], fill[~mask]
        print(f"\nVARIANT {name}: keep {len(kept)}/{len(fill)} "
              f"({len(kept) / len(fill):.1%})")
        report("kept", kept, extra=f"  YES={kept.result.mean():.3f}"
               if len(kept) else "")
        report("skipped", skip, extra=f"  YES={skip.result.mean():.3f}"
               if len(skip) else "")
        for fam, gg in kept.groupby("fam"):
            if len(gg) >= 5:
                report(f"  kept [{fam}]", gg)

    # ---- slippage & spread
    print("\nSLIPPAGE — (real NO ask) - (100 - last_print), cents "
          "(modeled assumed +4):")
    for fam, gg in fill.groupby("fam"):
        s = gg.slip.dropna()
        if not len(s):
            continue
        print(f"  [{fam:<9}] n={len(s):>4} mean={s.mean():+6.2f} "
              f"p25={s.quantile(.25):+6.1f} med={s.median():+6.1f} "
              f"p75={s.quantile(.75):+6.1f} p95={s.quantile(.95):+6.1f}")
    s = fill.slip.dropna()
    print(f"  [ALL      ] n={len(s):>4} mean={s.mean():+6.2f} "
          f"med={s.median():+6.1f} p95={s.quantile(.95):+6.1f}")
    sp = fill.spread.dropna()
    print(f"median YES bid-ask spread at T: {sp.median():.1f}c "
          f"(mean {sp.mean():.1f}c, n={len(sp)}); per family: "
          + ", ".join(f"{f}={gg.spread.median():.0f}c"
                      for f, gg in fill.groupby('fam')))

    # ---- latency
    print("\nLATENCY — same signals repriced at later bars:")
    for col, lab in (("pnl", "entry at T"), ("pnl_t1", "entry at T+1min"),
                     ("pnl_t5", "entry at T+5min")):
        gg = fill[fill[col].notna()]
        report(lab, gg, col=col)

    # ---- verdicts
    print("\nPER-FAMILY VERDICT (real asks, p(<=0)<0.01, n>=15 events):")
    for fam, gg in fill.groupby("fam"):
        st = boot(gg.pnl, gg.event_ticker)
        ok = st["p_le_0"] < 0.01 and st["n_clusters"] >= 15
        print(f"  [{fam:<9}] {'PASS' if ok else 'fail'}  {fmt(st)}")
    return df


def phase_analyze():
    sigs = build_signals()
    all_tk = set()
    for g in sigs.values():
        all_tk |= set(g.ticker)
    books = load_books(all_tk)
    print(f"minute books loaded for {len(books)}/{len(all_tk)} tickers")
    if PLAN_JSON.exists():
        note = json.loads(PLAN_JSON.read_text()).get("sampled_note")
        if note:
            print(f"NOTE: {note}")
    for off in (60, 90):
        analyze_offset(off, sigs[off], books)


if __name__ == "__main__":
    {"plan": phase_plan, "fetch": phase_fetch,
     "analyze": phase_analyze}[sys.argv[1]]()
