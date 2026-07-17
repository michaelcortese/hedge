"""ADVERSE-SELECTION / COUNTERPARTY audit of the family-2b mention rule.

RULE UNDER AUDIT: 60-180 min before event end, if last print is 20-80c YES,
buy NO taker at (100-last_print)+4c, taker fee, hold to settlement.
Claimed +9.63 c/ct net, n=1,111 / 185 events.

Structural fact this audit leans on: the "last print in [60,180)" is only
knowable at ENTRY TIME = event_end - 60 min, and by construction there are
ZERO prints between the signal print and entry time. All post-entry tape is
at mte < 60. YES markets halt at the mention, so a market can be halted
(already resolved) BEFORE entry time and still be selected by the backtest.

Attacks:
  1. POST-ENTRY DRIFT   - tape drift after entry, split by outcome; reprice
                          entries at first post-entry print (staleness stress).
  2. TIME-TO-DOOM       - (resolution - entry) for YES resolvers; halted-
                          pre-entry lookahead; burst filter on signal print.
  3. WHO SELLS NO       - taker side / size of the signal print and prior
                          flow; P&L conditional on counterparty type.
  4. SIZE REALISM       - post-entry taker-NO volume within 4c of the print.

All significance: cluster bootstrap over event_ticker, 10,000 reps.

Run:  .venv/bin/python scripts/audit_adverse_selection.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from research_mentions_lib import DATA, load_markets, taker_fee_cents  # noqa: E402

RNG_SEED = 7
N_BOOT = 10_000


def boot(values, clusters, n_boot: int = N_BOOT, seed: int = RNG_SEED,
         weights=None) -> dict:
    """Fast cluster bootstrap (whole-cluster resample, pooled weighted mean)."""
    values = np.asarray(values, dtype=float)
    w = np.ones_like(values) if weights is None else np.asarray(weights, float)
    codes, uniq = pd.factorize(np.asarray(clusters))
    sums = np.bincount(codes, weights=values * w)
    wsum = np.bincount(codes, weights=w)
    rng = np.random.default_rng(seed)
    k = len(uniq)
    draws = rng.integers(0, k, size=(n_boot, k))
    means = sums[draws].sum(axis=1) / wsum[draws].sum(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": float(sums.sum() / wsum.sum()),
            "ci_lo": float(lo), "ci_hi": float(hi),
            "n": int(len(values)), "n_ev": int(k),
            "p_le_0": float((means <= 0).mean()),
            "p_ge_0": float((means >= 0).mean())}


def fmt(st: dict, label: str) -> str:
    return (f"{label}: n={st['n']} ({st['n_ev']} ev) mean={st['mean']:+.2f} "
            f"CI95=[{st['ci_lo']:+.2f},{st['ci_hi']:+.2f}] "
            f"p(<=0)={st['p_le_0']:.4f} p(>=0)={st['p_ge_0']:.4f}")


# ---------------------------------------------------------------- loading
# (copied from research_tape_micro.py load_trades_full -- identical filter:
#  drop trades without yes_price_dollars or taker_side, dedupe trade_id,
#  last re-appended market line wins)

def load_trades_full() -> pd.DataFrame:
    per_mkt: dict[str, tuple] = {}
    with (DATA / "trades.jsonl").open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            tk = rec["ticker"]
            ts, px, ct, side = [], [], [], []
            seen = set()
            for t in rec.get("trades") or []:
                tid = t.get("trade_id")
                if tid in seen:
                    continue
                seen.add(tid)
                yp = t.get("yes_price_dollars")
                s = t.get("taker_side")
                if yp is None or s not in ("yes", "no"):
                    continue
                ts.append(t.get("created_time"))
                px.append(float(yp) * 100.0)
                ct.append(float(t.get("count_fp") or t.get("count") or 0))
                side.append(s == "yes")
            per_mkt[tk] = (ts, px, ct, side)
    frames = {
        "ticker": np.concatenate([np.repeat(tk, len(v[0]))
                                  for tk, v in per_mkt.items()]),
        "ts": np.concatenate([np.array(v[0], dtype=object)
                              for v in per_mkt.values()]),
        "yes_price": np.concatenate([np.array(v[1], dtype=float)
                                     for v in per_mkt.values()]),
        "count": np.concatenate([np.array(v[2], dtype=float)
                                 for v in per_mkt.values()]),
        "taker_yes": np.concatenate([np.array(v[3], dtype=bool)
                                     for v in per_mkt.values()]),
    }
    per_mkt.clear()
    df = pd.DataFrame(frames)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, format="ISO8601")
    df = df.dropna(subset=["yes_price", "ts"])
    return df.sort_values(["ticker", "ts"], kind="stable").reset_index(drop=True)


def build():
    mk = load_markets().drop_duplicates(subset=["ticker"])
    cache = DATA / "_audit_advsel_tape.parquet"
    if cache.exists():
        j = pd.read_parquet(cache)
        return mk, j
    tr = load_trades_full()
    j = tr.merge(mk[["ticker", "result", "event_ticker", "series", "family",
                     "close_time"]], on="ticker", how="inner")
    try:
        j.to_parquet(cache)
    except Exception as e:
        print(f"  (cache write skipped: {e})")
    return mk, j


# ------------------------------------------------- signal construction

def select_signals(j: pd.DataFrame) -> pd.DataFrame:
    """Exact family-2b selection + per-signal tape features."""
    jj = j.join(j.groupby("event_ticker").close_time.max().rename("ev_end"),
                on="event_ticker")
    jj["mte"] = (jj.ev_end - jj.ts).dt.total_seconds() / 60.0
    w = jj[(jj.mte >= 60) & (jj.mte < 180)]
    last = w.sort_values("ts").groupby("ticker").tail(1)
    g = last[last.yes_price.between(20, 80)].copy()
    g = g.rename(columns={"ts": "sig_ts", "yes_price": "p_sig",
                          "mte": "mte_sig"})
    print(f"selected signals: n={len(g)}, events={g.event_ticker.nunique()}")

    # per-ticker tape arrays for feature computation (int64 ns timestamps
    # to sidestep tz-aware/naive comparison issues)
    tapes = {}
    for tk, gg in j[j.ticker.isin(g.ticker)].groupby("ticker", sort=False):
        tapes[tk] = (gg["ts"].dt.as_unit("ns").astype("int64").to_numpy(),
                     gg["yes_price"].to_numpy(),
                     gg["count"].to_numpy(), gg["taker_yes"].to_numpy())

    MIN_NS = 60_000_000_000
    feats = []
    for row in g.itertuples():
        ts, px, ct, ty = tapes[row.ticker]
        sig_ts = row.sig_ts.value
        entry_ts = row.ev_end.value - 60 * MIN_NS

        # prior flow around the signal print (inclusive of signal stamp)
        m5 = (ts > sig_ts - 5 * MIN_NS) & (ts <= sig_ts)
        m15 = (ts > sig_ts - 15 * MIN_NS) & (ts <= sig_ts)
        vol5 = float(ct[m5].sum())
        signed15 = float(np.where(ty[m15], ct[m15], -ct[m15]).sum())
        # the full final stamp (sweep) the signal print belongs to
        stamp = ts == sig_ts
        stamp_vol = float(ct[stamp].sum())
        stamp_signed = float(np.where(ty[stamp], ct[stamp], -ct[stamp]).sum())

        # post-entry tape (strictly after entry time = ev_end - 60min)
        post = ts > entry_ts
        pts, ppx, pct, pty = ts[post], px[post], ct[post], ty[post]
        first_px = float(ppx[0]) if len(ppx) else np.nan
        first_delay = (float((pts[0] - entry_ts) / MIN_NS)
                       if len(pts) else np.nan)

        def last_px_within(minutes):
            m = pts <= entry_ts + minutes * MIN_NS
            return float(ppx[m][-1]) if m.any() else np.nan

        # liquidity proxies: post-entry taker-NO volume at yes>=p-4 means
        # other takers actually bought NO at NO-price <= (100-p)+4
        no_ok = pct[(~pty) & (ppx >= row.p_sig - 4.0)].sum()
        feats.append({
            "ticker": row.ticker, "event_ticker": row.event_ticker,
            "family": row.family, "result": row.result,
            "p_sig": row.p_sig, "mte_sig": row.mte_sig,
            "sig_taker_yes": bool(row.taker_yes), "sig_count": row.count,
            "stamp_vol": stamp_vol, "stamp_signed": stamp_signed,
            "vol5_pre": vol5, "signed15_pre": signed15,
            "close_time": row.close_time, "ev_end": row.ev_end,
            "mins_close_after_entry":
                (row.close_time - row.ev_end).total_seconds() / 60.0 + 60.0,
            "mins_close_after_sig":
                (row.close_time - row.sig_ts).total_seconds() / 60.0,
            "n_post": int(post.sum()),
            "first_post_px": first_px, "first_post_delay": first_delay,
            "px_post_5": last_px_within(5), "px_post_15": last_px_within(15),
            "px_post_30": last_px_within(30),
            "post_vol_total": float(pct.sum()),
            "post_vol_takerno": float(pct[~pty].sum()),
            "post_vol_takerno_within4c": float(no_ok),
        })
    return pd.DataFrame(feats)


def net_pnl(p_sig, result, extra_spread=4.0):
    entry = np.clip(100.0 - np.asarray(p_sig, float) + extra_spread, 1, 99)
    fee = np.array([taker_fee_cents(e) for e in entry])
    return np.where(np.asarray(result) == 0, 100.0, 0.0) - entry - fee


# ---------------------------------------------------------------- attacks

def attack0_replicate(s: pd.DataFrame):
    print("\n" + "=" * 72)
    print("ATTACK 0 - replicate headline")
    s["net"] = net_pnl(s.p_sig, s.result)
    print(fmt(boot(s.net, s.event_ticker), "  headline (as claimed)"))
    print(f"  YES rate={s.result.mean():.3f}  mean yes px={s.p_sig.mean():.1f}c")


def attack1_drift(s: pd.DataFrame):
    print("\n" + "=" * 72)
    print("ATTACK 1 - POST-ENTRY DRIFT (entry anchored at ev_end-60min)")
    print(f"  signal-print staleness at entry (mte_sig-60, min): "
          f"median={np.median(s.mte_sig - 60):.1f}  "
          f"q25={np.percentile(s.mte_sig - 60, 25):.1f}  "
          f"q75={np.percentile(s.mte_sig - 60, 75):.1f}")
    has_post = s.n_post > 0
    print(f"  signals with ANY post-entry print: {has_post.mean():.1%} "
          f"({has_post.sum()}/{len(s)})")
    for res, lbl in [(1, "LOSERS (resolve YES)"), (0, "WINNERS (resolve NO)")]:
        gg = s[s.result == res]
        print(f"\n  {lbl}: n={len(gg)}, with post-entry print: "
              f"{(gg.n_post > 0).mean():.1%}")
        for col, lab in [("px_post_5", "drift @+5m"),
                         ("px_post_15", "drift @+15m"),
                         ("px_post_30", "drift @+30m"),
                         ("first_post_px", "first post print")]:
            d = (gg[col] - gg.p_sig).dropna()
            if len(d) < 5:
                print(f"    {lab}: n={len(d)} (insufficient)")
                continue
            st = boot(d.to_numpy(), gg.loc[d.index, "event_ticker"].to_numpy())
            print("    " + fmt(st, f"{lab} (yes c, cond. on print)"))
        fp = gg.first_post_delay.dropna()
        if len(fp):
            print(f"    first post-entry print delay (min): "
                  f"median={fp.median():.1f} q25={fp.quantile(.25):.1f} "
                  f"q75={fp.quantile(.75):.1f}")

    # zero-post-print group: what's its YES rate / P&L?
    z = s[~has_post]
    if len(z) > 10:
        st = boot(net_pnl(z.p_sig, z.result), z.event_ticker.to_numpy())
        print("\n  " + fmt(st, f"NO-POST-TAPE signals (YES rate "
                               f"{z.result.mean():.3f})"))
    nz = s[has_post]
    st = boot(net_pnl(nz.p_sig, nz.result), nz.event_ticker.to_numpy())
    print("  " + fmt(st, f"signals WITH post-tape (YES rate "
                         f"{nz.result.mean():.3f})"))

    # staleness stress: reprice entry at the first post-entry print + 4c.
    print("\n  REPRICING STRESS - entry at first post-entry print +4c:")
    r = s.copy()
    r["p_fill"] = r.first_post_px.fillna(r.p_sig)
    st = boot(net_pnl(r.p_fill, r.result), r.event_ticker.to_numpy())
    print("  " + fmt(st, "  all (no-post-print keeps signal px)"))
    rr = r[r.n_post > 0]
    st = boot(net_pnl(rr.p_fill, rr.result), rr.event_ticker.to_numpy())
    print("  " + fmt(st, "  only signals with post-tape, repriced"))
    # milder: reprice at last print within 15 min of entry when available
    r15 = s.copy()
    r15["p_fill"] = r15.px_post_15.fillna(r15.p_sig)
    st = boot(net_pnl(r15.p_fill, r15.result), r15.event_ticker.to_numpy())
    print("  " + fmt(st, "  repriced at last print <=15min after entry"))


def attack2_doom(s: pd.DataFrame):
    print("\n" + "=" * 72)
    print("ATTACK 2 - TIME-TO-DOOM + lookahead + burst filter")
    y = s[s.result == 1]
    print(f"  YES resolvers: n={len(y)} ({y.event_ticker.nunique()} ev)")
    t = y.mins_close_after_entry  # close_time - entry_time, minutes
    print("  (close_time - entry_time) min: "
          f"q10={t.quantile(.1):+.0f} q25={t.quantile(.25):+.0f} "
          f"med={t.median():+.0f} q75={t.quantile(.75):+.0f} "
          f"q90={t.quantile(.9):+.0f}")
    for thr in (0, 5, 15, 30, 60):
        print(f"    halted within {thr:>2} min of entry: "
              f"{(t <= thr).mean():.1%} ({(t <= thr).sum()})")
    # lookahead check: markets HALTED BEFORE entry cannot be traded at all
    dead = s[s.mins_close_after_entry <= 0]
    print(f"\n  markets already CLOSED at entry time (untradeable): "
          f"n={len(dead)} of {len(s)}; of these resolve YES: "
          f"{dead.result.mean():.1%}" if len(dead) else
          "\n  no markets closed before entry")
    alive = s[s.mins_close_after_entry > 0]
    st = boot(net_pnl(alive.p_sig, alive.result), alive.event_ticker.to_numpy())
    print("  " + fmt(st, f"  V1 tradable-only (close>entry), YES rate "
                         f"{alive.result.mean():.3f}"))

    # burst filter on the signal print
    print("\n  prior-5-min volume at signal print (incl. its stamp):")
    v = s.vol5_pre
    print(f"    med={v.median():.0f} q75={v.quantile(.75):.0f} "
          f"q90={v.quantile(.9):.0f} q99={v.quantile(.99):.0f}")
    for thr in (25, 50, 100):
        keep = s[(s.vol5_pre < thr)]
        st = boot(net_pnl(keep.p_sig, keep.result),
                  keep.event_ticker.to_numpy())
        print("    " + fmt(st, f"exclude vol5>={thr:>3} (YES rate "
                               f"{keep.result.mean():.3f})"))
        drop = s[s.vol5_pre >= thr]
        if len(drop) > 20:
            st = boot(net_pnl(drop.p_sig, drop.result),
                      drop.event_ticker.to_numpy())
            print("    " + fmt(st, f"  the EXCLUDED (vol5>={thr}) "
                                   f"(YES rate {drop.result.mean():.3f})"))

    # staleness split of the signal print (age at entry = mte_sig - 60)
    print("\n  edge by signal-print age at entry (mte_sig - 60):")
    for lo, hi in [(0, 10), (10, 30), (30, 60), (60, 120)]:
        gg = s[(s.mte_sig - 60 >= lo) & (s.mte_sig - 60 < hi)]
        if len(gg) < 20:
            continue
        st = boot(net_pnl(gg.p_sig, gg.result), gg.event_ticker.to_numpy())
        print("    " + fmt(st, f"age [{lo:>3},{hi:>3})min (YES rate "
                               f"{gg.result.mean():.3f})"))


def attack3_whosells(s: pd.DataFrame):
    print("\n" + "=" * 72)
    print("ATTACK 3 - WHO IS ON THE OTHER SIDE (signal print + prior flow)")
    print(f"  signal print taker side: YES {s.sig_taker_yes.mean():.1%}, "
          f"NO {1 - s.sig_taker_yes.mean():.1%}")
    for isyes, lbl in [(True, "last print = taker-YES (retail lotto buys)"),
                       (False, "last print = taker-NO  (someone dumping YES)")]:
        gg = s[s.sig_taker_yes == isyes]
        st = boot(net_pnl(gg.p_sig, gg.result), gg.event_ticker.to_numpy())
        print("  " + fmt(st, f"{lbl} (YES rate {gg.result.mean():.3f})"))
    print("\n  by signal-stamp sweep size (contracts in final stamp):")
    for lo, hi in [(0, 5), (5, 20), (20, 100), (100, 1e12)]:
        gg = s[(s.stamp_vol >= lo) & (s.stamp_vol < hi)]
        if len(gg) < 20:
            continue
        st = boot(net_pnl(gg.p_sig, gg.result), gg.event_ticker.to_numpy())
        print("  " + fmt(st, f"stamp vol [{lo:>3},{hi if hi < 1e11 else 'inf'})"
                             f" (YES rate {gg.result.mean():.3f})"))
    print("\n  by prior-15-min net signed taker flow (+ = YES-taker wave):")
    for cond, lbl in [(s.signed15_pre > 10, "net > +10 (YES wave into signal)"),
                      (s.signed15_pre.between(-10, 10), "net in [-10,+10]"),
                      (s.signed15_pre < -10, "net < -10 (NO wave into signal)")]:
        gg = s[cond]
        if len(gg) < 20:
            continue
        st = boot(net_pnl(gg.p_sig, gg.result), gg.event_ticker.to_numpy())
        print("  " + fmt(st, f"{lbl} (n={len(gg)}, YES rate "
                             f"{gg.result.mean():.3f})"))


def attack4_size(s: pd.DataFrame):
    print("\n" + "=" * 72)
    print("ATTACK 4 - SIZE REALISM (post-entry, i.e., final 60 min)")
    for col, lbl in [
            ("post_vol_total", "total tape volume after entry"),
            ("post_vol_takerno", "taker-NO volume after entry"),
            ("post_vol_takerno_within4c",
             "taker-NO volume at NO price <= (100-p)+4c  <-- our limit")]:
        v = s[col]
        print(f"  {lbl}:\n    med={v.median():.0f} mean={v.mean():.0f} "
              f"q25={v.quantile(.25):.0f} q75={v.quantile(.75):.0f} "
              f"share zero={(v == 0).mean():.1%}")
    v = s.post_vol_takerno_within4c
    days = (s.ev_end.max() - s.ev_end.min()).total_seconds() / 86400
    per_day = len(s) / days
    # capacity if we take X% of the observed taker-NO flow within our limit
    for share in (0.25, 0.5):
        cap = (v * share).clip(upper=100)
        print(f"  taking {share:.0%} of within-limit taker-NO flow "
              f"(cap 100/mkt): med={cap.median():.1f} ct/signal, "
              f"mean={cap.mean():.1f}; at +9.63c/ct -> "
              f"${(cap.mean() * per_day * 0.0963):.0f}/day"
              f" ({per_day:.1f} signals/day)")
    # P&L weighted by realistically fillable size (does edge survive
    # volume-weighting? if edge lives in illiquid signals it dies here)
    wgt = (v * 0.5).clip(upper=100)
    ok = wgt > 0
    st = boot(net_pnl(s.p_sig[ok], s.result[ok]),
              s.event_ticker[ok].to_numpy(), weights=wgt[ok].to_numpy())
    print("  " + fmt(st, "size-weighted net (wgt = fillable ct/signal)"))
    st = boot(net_pnl(s.p_sig[ok], s.result[ok]), s.event_ticker[ok].to_numpy())
    print("  " + fmt(st, "fillable-only, unweighted"))


def surviving_variant(s: pd.DataFrame):
    print("\n" + "=" * 72)
    print("SURVIVING VARIANT CHECK - stack the implementable filters")
    v = s[(s.mins_close_after_entry > 0)]
    v2 = v[v.vol5_pre < 50]
    for name, gg in [("V1: not halted at entry", v),
                     ("V2: V1 + prior-5-min vol < 50", v2),
                     ("V3: V2 + last print taker-NO or stamp<20",
                      v2[(~v2.sig_taker_yes) | (v2.stamp_vol < 20)])]:
        if len(gg) < 20:
            continue
        st = boot(net_pnl(gg.p_sig, gg.result), gg.event_ticker.to_numpy())
        print("  " + fmt(st, f"{name} (YES rate {gg.result.mean():.3f})"))


def main():
    print("loading markets + trades...")
    mk, j = build()
    print(f"dataset: {len(mk)} markets, {len(j)} trades, "
          f"{j.event_ticker.nunique()} events on tape")
    s = select_signals(j)
    attack0_replicate(s)
    attack1_drift(s)
    attack2_doom(s)
    attack3_whosells(s)
    attack4_size(s)
    surviving_variant(s)


if __name__ == "__main__":
    main()
