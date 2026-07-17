"""R3: re-anchor the late-event taker-NO theta-carry rule on TRUE event ends.

The killed maker rule and the taker theta-carry candidate were anchored on
event_end = max close_time across an event's markets, which the r3
feasibility study proved is administratively late (MLB median +776 min).
This script re-runs the economics against ground-truth end times.

Tests (event-clustered bootstrap, 10k reps, throughout):
  1. DIAGNOSIS  — where did the original rule's signals fall vs TRUE end,
                  and how does its P&L split by that?
  2. RE-ANCHOR  — entry at T = true_end - {60,90,120} min, last print in
                  prior 120 min in 20-80c, market open at T, buy NO at
                  (100-p)+4c + fee, hold to settlement.  Also repriced at
                  real minute-book NO asks where candles cover T.
  3. THETA CAL  — YES resolution rate vs price for still-open markets at
                  true-elapsed-fraction tau = 0.5/0.7/0.9.
  4. STRESS     — shift assumed true end +/-30/60 min, re-report test 2.

Run:  .venv/bin/python scripts/research_r3_reanchor.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from research_mentions_lib import DATA, load_markets, taker_fee_cents  # noqa: E402
from research_tape_micro import cluster_bootstrap, load_trades_full  # noqa: E402

R3 = DATA / "r3"
N_BOOT = 10_000
SPREAD_C = 4.0  # modeled spread cross, as in the original rule


def boot(vals, clus, **kw):
    return cluster_bootstrap(np.asarray(vals, float), np.asarray(clus),
                             n_boot=N_BOOT, **kw)


def fmt(st, extra=""):
    return (f"n={st['n']:>4} ({st['n_clusters']:>3} ev) "
            f"mean={st['mean']:+7.2f}c CI=[{st['ci_lo']:+7.2f},"
            f"{st['ci_hi']:+7.2f}] p(<=0)={st['p_le_0']:.4f}{extra}")


# ------------------------------------------------------------ true ends

def load_true_ends() -> pd.DataFrame:
    rows = []

    wc = pd.read_csv(R3 / "wc_matches.csv")
    for _, r in wc.iterrows():
        if pd.isna(r.est_end_utc):
            continue
        rows.append(dict(event_ticker=r.event_ticker, fam="WC",
                         t_start=pd.Timestamp(r.kickoff_utc),
                         t_end=pd.Timestamp(r.est_end_utc),
                         conf=r.match_confidence))

    mlb = pd.read_csv(R3 / "mlb_eval.csv")
    for _, r in mlb.iterrows():
        if pd.isna(r.g_end) or pd.isna(r.g_first_pitch):
            continue
        rows.append(dict(event_ticker=r.event_ticker, fam="MLB",
                         t_start=pd.Timestamp(r.g_first_pitch),
                         t_end=pd.Timestamp(r.g_end), conf="high"))

    nn = pd.read_csv(R3 / "nba_nhl_games.csv")
    for _, r in nn.iterrows():
        if pd.isna(r.actual_end_utc):
            continue
        rows.append(dict(event_ticker=r.event_ticker, fam=r.league,
                         t_start=pd.Timestamp(r.actual_first_play_utc),
                         t_end=pd.Timestamp(r.actual_end_utc),
                         conf=r.match_confidence))

    ec = pd.read_csv(R3 / "earnings_calls.csv")
    n_bad_so = 0
    for _, r in ec.iterrows():
        if pd.isna(r.so) or pd.isna(r.est_dur):
            continue
        so = pd.Timestamp(r.so)
        td = pd.Timestamp(r.tdate, tz="UTC")
        if abs((so - td).total_seconds()) > 2.5 * 86400:  # bogus sched date
            n_bad_so += 1
            continue
        rows.append(dict(event_ticker=r.event_ticker, fam="earnings",
                         t_start=so,
                         t_end=so + pd.Timedelta(minutes=float(r.est_dur)),
                         conf="est"))

    h = pd.read_csv(R3 / "hearings_findings.csv")
    for _, r in h.iterrows():
        if pd.isna(r.actual_end_et) or pd.isna(r.actual_start_et):
            continue
        m = re.match(r"KXHEARINGMENTION-(\d{2})([A-Z]{3})(\d{2})",
                     r.event_ticker)
        if not m:
            continue
        yy, mon, dd = m.groups()
        date = pd.Timestamp(f"20{yy}-{mon}-{dd}")

        def et(s):
            t = pd.Timestamp(f"{date.date()} {s.strip()}")
            return t.tz_localize("America/New_York").tz_convert("UTC")

        rows.append(dict(event_ticker=r.event_ticker, fam="hearings",
                         t_start=et(r.actual_start_et),
                         t_end=et(r.actual_end_et), conf="high"))

    te = pd.DataFrame(rows).drop_duplicates("event_ticker")
    te["t_start"] = pd.to_datetime(te.t_start, utc=True)
    te["t_end"] = pd.to_datetime(te.t_end, utc=True)
    print(f"true-end table: {len(te)} events "
          f"({te.fam.value_counts().to_dict()}); "
          f"{n_bad_so} earnings rows dropped for bogus sched_occ")
    return te


# ------------------------------------------------------------ minute books

def load_minute_books(tickers: set[str]) -> dict[str, tuple]:
    """ticker -> (sorted end_ts[], yes_bid[], yes_ask[]) in cents."""
    acc: dict[str, dict[int, tuple]] = {}
    for fn in ("minute_candles.jsonl", "minute_candles_r3oos.jsonl",
               "minute_candles_siblingfade.jsonl"):
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
                yb = (c.get("yes_bid") or {}).get("close_dollars")
                ya = (c.get("yes_ask") or {}).get("close_dollars")
                if ts is None:
                    continue
                d[ts] = (float(yb) * 100 if yb is not None else np.nan,
                         float(ya) * 100 if ya is not None else np.nan)
    out = {}
    for tk, d in acc.items():
        ts = np.array(sorted(d), dtype=np.int64)
        yb = np.array([d[t][0] for t in ts])
        ya = np.array([d[t][1] for t in ts])
        out[tk] = (ts, yb, ya)
    return out


def book_at(books, tk, T: pd.Timestamp, fresh_s: int = 300):
    """(yes_bid, yes_ask) from last candle ending in [T-fresh, T+60]."""
    if tk not in books:
        return None
    ts, yb, ya = books[tk]
    t = int(T.timestamp())
    i = np.searchsorted(ts, t + 60, side="right") - 1
    if i < 0 or ts[i] < t - fresh_s:
        return None
    return yb[i], ya[i]


# ------------------------------------------------------------ P&L helpers

def no_pnl_modeled(yes_px, result):
    entry = np.clip(100.0 - np.asarray(yes_px, float) + SPREAD_C, 1, 99)
    fee = np.array([taker_fee_cents(e) for e in entry])
    return np.where(np.asarray(result) == 0, 100.0, 0.0) - entry - fee


def report(label, pnl, clus, extra=""):
    st = boot(pnl, clus)
    print(f"  {label:<38} {fmt(st, extra)}")
    return st


# ================================================================ test 1

def test1_diagnosis(j, te):
    print("\n" + "=" * 78)
    print("TEST 1 — DIAGNOSIS: original rule's signals vs TRUE end")
    jj = j.join(j.groupby("event_ticker").close_time.max().rename("ev_end"),
                on="event_ticker")
    jj["mte"] = (jj.ev_end - jj.ts).dt.total_seconds() / 60.0
    w = jj[(jj.mte >= 60) & (jj.mte < 180)]
    last = w.sort_values("ts").groupby("ticker").tail(1)
    sig = last[last.yes_price.between(20, 80)].copy()
    print(f"original signal set (60-180min pre max_close, 20-80c): "
          f"n={len(sig)} markets, {sig.event_ticker.nunique()} events")

    m = sig.merge(te, on="event_ticker", how="left")
    matched = m[m.t_end.notna()].copy()
    print(f"matched to true ends: n={len(matched)} markets, "
          f"{matched.event_ticker.nunique()} events "
          f"({matched.fam.value_counts().to_dict()})")
    unm = m[m.t_end.isna()]
    print(f"unmatched (no true end): n={len(unm)} markets in "
          f"{unm.event_ticker.nunique()} events "
          f"(top prefixes: "
          f"{unm.ticker.str.split('-').str[0].value_counts().head(5).to_dict()})")

    matched["dt_end"] = (matched.ts - matched.t_end).dt.total_seconds() / 60.0
    matched["dt_start"] = (matched.ts
                           - matched.t_start).dt.total_seconds() / 60.0
    conds = [matched.dt_start < 0, matched.dt_end < 0,
             matched.dt_end < 60]
    matched["grp"] = np.select(
        conds, ["pre-start", "in-event", "0-60min post"], ">60min post")
    matched["pnl"] = no_pnl_modeled(matched.yes_price, matched.result)

    frac = matched.grp.value_counts(normalize=True)
    print(f"\nsignal-moment placement vs TRUE end "
          f"(n={len(matched)} signals):")
    for g in ["pre-start", "in-event", "0-60min post", ">60min post"]:
        gg = matched[matched.grp == g]
        if not len(gg):
            print(f"  {g:<14} n=0")
            continue
        st = boot(gg.pnl, gg.event_ticker)
        print(f"  {g:<14} frac={frac.get(g, 0):.3f} "
              f"YES rate={gg.result.mean():.3f} "
              f"mean px={gg.yes_price.mean():4.1f}c  P&L {fmt(st)}")
    st_all = boot(matched.pnl, matched.event_ticker)
    print(f"  {'ALL matched':<14} {'':<11}"
          f"YES rate={matched.result.mean():.3f} "
          f"mean px={matched.yes_price.mean():4.1f}c  P&L {fmt(st_all)}")
    # unmatched-set P&L for reference (part of original headline)
    if len(unm):
        st_u = boot(no_pnl_modeled(unm.yes_price, unm.result),
                    unm.event_ticker)
        print(f"  {'unmatched set':<14} {'':<11}"
              f"YES rate={unm.result.mean():.3f}"
              f"{'':<15}P&L {fmt(st_u)}")
    return matched


# ================================================================ test 2

def entries_at(j, te, offset_min: float, end_col: str = "t_end"):
    """Signal set for entry at T = true_end - offset: last print in
    (T-120min, T], 20-80c, market close_time > T."""
    jt = j.merge(te, on="event_ticker", how="inner")
    T = jt[end_col] - pd.Timedelta(minutes=offset_min)
    m = ((jt.ts <= T) & (jt.ts > T - pd.Timedelta(minutes=120))
         & (jt.close_time > T))
    w = jt[m]
    last = w.sort_values("ts").groupby("ticker").tail(1).copy()
    last["t_entry"] = last[end_col] - pd.Timedelta(minutes=offset_min)
    return last[last.yes_price.between(20, 80)].copy()


def test2_reanchored(j, te, books):
    print("\n" + "=" * 78)
    print("TEST 2 — RE-ANCHORED RULE: entry at T = true_end - offset, "
          "last print in prior 120min in 20-80c, market open at T, "
          "buy NO (modeled: (100-p)+4c + fee)")
    keep = {}
    for off in (60, 90, 120):
        g = entries_at(j, te, off)
        print(f"\n-- offset {off} min: {len(g)} signals, "
              f"{g.event_ticker.nunique()} events, "
              f"YES rate={g.result.mean():.3f}, "
              f"mean px={g.yes_price.mean():.1f}c")
        if not len(g):
            continue
        g["pnl"] = no_pnl_modeled(g.yes_price, g.result)
        report("ALL (modeled fill)", g.pnl, g.event_ticker)
        for fam, gg in g.groupby("fam"):
            if len(gg) < 5:
                print(f"  [{fam:<9}] n={len(gg)} (too few)")
                continue
            report(f"[{fam:<9}] modeled", gg.pnl, gg.event_ticker,
                   extra=f"  YES={gg.result.mean():.3f}")
        for plo, phi in [(20, 40), (40, 60), (60, 80)]:
            gg = g[g.yes_price.between(plo, phi)]
            if len(gg) < 10:
                continue
            report(f"px {plo}-{phi}c modeled", gg.pnl, gg.event_ticker,
                   extra=f"  YES={gg.result.mean():.3f}")
        keep[off] = g

        # ---- real minute-book NO asks at T
        rows = []
        for _, r in g.iterrows():
            b = book_at(books, r.ticker, r.t_entry)
            if b is None:
                continue
            yb, ya = b
            if not np.isfinite(yb) or yb <= 0:
                continue
            entry = 100.0 - yb  # NO ask
            if entry >= 100 or entry <= 0:
                continue
            fee = taker_fee_cents(min(max(entry, 1), 99))
            pnl = (100.0 if r.result == 0 else 0.0) - entry - fee
            rows.append(dict(event_ticker=r.event_ticker, fam=r.fam,
                             pnl=pnl, pnl_model=r.pnl, entry=entry,
                             yes_bid=yb, yes_ask=ya, result=r.result))
        rb = pd.DataFrame(rows)
        if len(rb):
            print(f"  real-book subsample: {len(rb)} signals, "
                  f"{rb.event_ticker.nunique()} events, "
                  f"mean NO-ask entry={rb.entry.mean():.1f}c "
                  f"(vs modeled {100 - g.yes_price.mean() + SPREAD_C:.1f}c)")
            report("REAL NO-ask fill", rb.pnl, rb.event_ticker,
                   extra=f"  YES={rb.result.mean():.3f}")
            report("modeled, same subsample", rb.pnl_model, rb.event_ticker)
            for fam, gg in rb.groupby("fam"):
                if len(gg) < 5:
                    continue
                report(f"[{fam:<9}] real fill", gg.pnl, gg.event_ticker)
        else:
            print("  real-book subsample: no coverage")
        keep[(off, "real")] = rb
    return keep


# ================================================================ test 3

def test3_theta(j, te):
    print("\n" + "=" * 78)
    print("TEST 3 — THETA-DECAY CALIBRATION vs TRUE elapsed fraction "
          "(still-open markets, last print <= T_tau)")
    jt = j.merge(te, on="event_ticker", how="inner")
    for tau in (0.5, 0.7, 0.9):
        Tt = jt.t_start + tau * (jt.t_end - jt.t_start)
        m = (jt.ts <= Tt) & (jt.close_time > Tt)
        w = jt[m]
        last = w.sort_values("ts").groupby("ticker").tail(1).copy()
        Tmap = dict(zip(jt.ticker, Tt))
        last["stale_min"] = [
            (Tmap[t] - ts).total_seconds() / 60.0
            for t, ts in zip(last.ticker, last.ts)]
        print(f"\n-- tau={tau}: {len(last)} open markets w/ prints, "
              f"{last.event_ticker.nunique()} events, median print "
              f"staleness={last.stale_min.median():.0f} min")
        for plo, phi in [(1, 20), (20, 40), (40, 60), (60, 80), (80, 99)]:
            gg = last[last.yes_price.between(plo, phi)]
            if len(gg) < 10:
                continue
            edge = gg.yes_price - 100.0 * gg.result  # gross NO edge c/ct
            st = boot(edge, gg.event_ticker)
            print(f"  px {plo:>2}-{phi:>2}c: n={st['n']:>4} "
                  f"({st['n_clusters']:>3} ev) mean px="
                  f"{gg.yes_price.mean():4.1f}c YES rate="
                  f"{gg.result.mean():.3f}  gross NO edge="
                  f"{st['mean']:+6.2f}c CI=[{st['ci_lo']:+6.2f},"
                  f"{st['ci_hi']:+6.2f}] p(<=0)={st['p_le_0']:.4f}")
        g = last[last.yes_price.between(20, 80)]
        if len(g) >= 10:
            edge = g.yes_price - 100.0 * g.result
            st = boot(edge, g.event_ticker)
            print(f"  20-80c pooled: {fmt(st)}")
            fr = g[g.stale_min <= 30]
            if len(fr) >= 10:
                st2 = boot(fr.yes_price - 100.0 * fr.result, fr.event_ticker)
                print(f"  20-80c fresh prints (<=30min): {fmt(st2, extra=f'  YES={fr.result.mean():.3f}')}")
            for fam, gg in g.groupby("fam"):
                if len(gg) < 8:
                    continue
                st2 = boot(gg.yes_price - 100.0 * gg.result, gg.event_ticker)
                print(f"    [{fam:<9}] {fmt(st2, extra=f'  YES={gg.result.mean():.3f}')}")


# ================================================================ test 4

def test4_stress(j, te):
    print("\n" + "=" * 78)
    print("TEST 4 — ANCHOR-ERROR STRESS: shift assumed true end "
          "(systematic), rule = entry at shifted_end - 60min")
    for shift in (-60, -30, 0, 30, 60):
        te2 = te.copy()
        te2["t_end"] = te2.t_end + pd.Timedelta(minutes=shift)
        g = entries_at(j, te2, 60)
        if not len(g):
            print(f"  shift {shift:+4d} min: no signals")
            continue
        pnl = no_pnl_modeled(g.yes_price, g.result)
        st = boot(pnl, g.event_ticker)
        print(f"  shift {shift:+4d} min: {fmt(st, extra=f'  YES={g.result.mean():.3f}')}")
        for fam, gg in g.groupby("fam"):
            if len(gg) < 20:
                continue
            st2 = boot(no_pnl_modeled(gg.yes_price, gg.result),
                       gg.event_ticker)
            print(f"      [{fam:<9}] {fmt(st2)}")


# ================================================================ main

def main():
    te = load_true_ends()
    cache = R3 / "_j_cache.parquet"
    if cache.exists():
        j = pd.read_parquet(cache)
        mk = load_markets().drop_duplicates(subset=["ticker"])
    else:
        mk = load_markets().drop_duplicates(subset=["ticker"])
        tr = load_trades_full()
        j = tr.merge(mk[["ticker", "result", "event_ticker", "series",
                         "family", "open_time", "close_time"]],
                     on="ticker", how="inner")
        j.to_parquet(cache)
    print(f"dataset: {len(mk)} settled markets, tapes for "
          f"{j.ticker.nunique()} markets ({j.event_ticker.nunique()} ev), "
          f"{len(j)} trades")
    in_ev = te.event_ticker.isin(j.event_ticker.unique())
    print(f"true-end events present in settled dataset: {in_ev.sum()}"
          f"/{len(te)} "
          f"({te[~in_ev].fam.value_counts().to_dict()} missing)")
    te = te[in_ev].copy()

    mt_tickers = set(
        j[j.event_ticker.isin(te.event_ticker)].ticker.unique())
    books = load_minute_books(mt_tickers)
    print(f"minute-book coverage: {len(books)}/{len(mt_tickers)} matched "
          f"tickers have candles")

    test1_diagnosis(j, te)
    test2_reanchored(j, te, books)
    test3_theta(j, te)
    test4_stress(j, te)


if __name__ == "__main__":
    main()
