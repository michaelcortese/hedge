"""Tape-microstructure research on Kalshi word-mention markets.

Families tested (see data/research/mentions/TAPE_MICRO.md for writeup):
  1. FADE vs FOLLOW taker bursts (post-burst markout vs settlement + tape)
  2. Last-trade-price calibration in the final hours before close
  3. Taker-side imbalance as an information signal (logit, clustered SE)
  4. Effective spread + maker capacity per series
  5. Tape anomalies

Conventions: integer-ish cents internally; fee = ceil(0.07*C*P*(1-P)) cents
(taker only, maker free); significance always cluster-bootstrapped by
EVENT ticker (markets within one event are correlated).

Run:  .venv/bin/python scripts/research_tape_micro.py [--min-markets N]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from research_mentions_lib import (  # noqa: E402
    DATA, load_markets, market_calibration_table, taker_fee_cents,
)

RNG_SEED = 7


def cluster_bootstrap(values, clusters, n_boot: int = 4000,
                      seed: int = RNG_SEED, weights=None) -> dict:
    """Fast cluster bootstrap: resample whole clusters, pooled (weighted)
    mean per replicate. O(n_boot * n_clusters) after one bincount pass —
    equivalent to the lib version but usable on millions of rows."""
    values = np.asarray(values, dtype=float)
    w = np.ones_like(values) if weights is None else np.asarray(
        weights, dtype=float)
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
            "se": float(means.std(ddof=1)), "n": int(len(values)),
            "n_clusters": int(k), "p_le_0": float((means <= 0).mean())}


# ---------------------------------------------------------------- loading

def load_trades_full() -> pd.DataFrame:
    """Full tape, memory-lean. If the collector re-appended a ticker, the
    LAST line for that ticker wins (whole-market replacement, no row dups)."""
    per_mkt: dict[str, tuple] = {}
    n_dup_lines = 0
    with (DATA / "trades.jsonl").open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # collector may be mid-write on the last line
            tk = rec["ticker"]
            if tk in per_mkt:
                n_dup_lines += 1
            ts, px, ct, side, blk = [], [], [], [], []
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
                blk.append(bool(t.get("is_block_trade")))
            per_mkt[tk] = (ts, px, ct, side, blk)
    if n_dup_lines:
        print(f"  note: {n_dup_lines} re-appended market lines (last wins)")
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
        "block": np.concatenate([np.array(v[4], dtype=bool)
                                 for v in per_mkt.values()]),
    }
    per_mkt.clear()
    df = pd.DataFrame(frames)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, format="ISO8601")
    df["taker_side"] = np.where(df.taker_yes, "yes", "no")
    df = df.dropna(subset=["yes_price", "ts"])
    return df.sort_values(["ticker", "ts"], kind="stable").reset_index(drop=True)


def build_dataset():
    mk = load_markets()
    tr = load_trades_full()
    mk = mk.drop_duplicates(subset=["ticker"])
    j = tr.merge(
        mk[["ticker", "result", "event_ticker", "series", "family",
            "open_time", "close_time"]],
        on="ticker", how="inner")
    j["signed"] = np.where(j.taker_side == "yes", j["count"], -j["count"])
    return mk, tr, j


# ------------------------------------------------- family 1: bursts

def find_bursts(j: pd.DataFrame, K: float, window_min: float = 5.0,
                cooldown_min: float = 15.0) -> pd.DataFrame:
    """Non-overlapping bursts: |net signed taker flow| >= K within window."""
    out = []
    win = pd.Timedelta(minutes=window_min)
    cool = pd.Timedelta(minutes=cooldown_min)
    for tk, g in j.groupby("ticker", sort=False):
        ts = g["ts"].to_numpy()
        sg = g["signed"].to_numpy()
        px = g["yes_price"].to_numpy()
        res = int(g["result"].iloc[0])
        ev = g["event_ticker"].iloc[0]
        fam = g["family"].iloc[0]
        close = g["close_time"].iloc[0]
        n = len(g)
        i0 = 0
        next_ok = ts[0]
        cum = np.concatenate([[0.0], np.cumsum(sg)])
        for i in range(n):
            if ts[i] < next_ok:
                continue
            while ts[i] - ts[i0] > win:
                i0 += 1
            net = cum[i + 1] - cum[i0]
            if abs(net) < K:
                continue
            d = 1 if net > 0 else -1
            trig_ts, trig_px = ts[i], px[i]
            # tape markouts: first trade >= +30min / +120min after trigger
            mo30 = mo120 = np.nan
            k30 = np.searchsorted(ts, trig_ts + pd.Timedelta(minutes=30))
            if k30 < n:
                mo30 = d * (px[k30] - trig_px)
            k120 = np.searchsorted(ts, trig_ts + pd.Timedelta(minutes=120))
            if k120 < n:
                mo120 = d * (px[k120] - trig_px)
            out.append({
                "ticker": tk, "event_ticker": ev, "family": fam,
                "ts": trig_ts, "price": trig_px, "dir": d, "net": net,
                "result": res,
                "mins_to_close": (close - trig_ts).total_seconds() / 60.0,
                "mo_settle": d * (100.0 * res - trig_px),
                "mo_30m": mo30, "mo_120m": mo120,
            })
            next_ok = trig_ts + cool
            i0 = i  # restart window after trigger
    return pd.DataFrame(out)


def family1(j: pd.DataFrame, eff_spread_by_series: dict) -> dict:
    print("\n" + "=" * 72)
    print("FAMILY 1 — fade vs follow taker bursts")
    results = {}
    for K in (25, 50, 100):
        b = find_bursts(j, K=K)
        if not len(b):
            continue
        # drop bursts inside the last 10 min (no time to react; and price
        # near settlement mechanically continues)
        b = b[b.mins_to_close > 10]
        if len(b) < 10:
            continue
        stat = cluster_bootstrap(b["mo_settle"].to_numpy(),
                                 b["event_ticker"].to_numpy(), seed=RNG_SEED)
        # maker who fades earns -mo_settle, fee-free
        fade = cluster_bootstrap(-b["mo_settle"].to_numpy(),
                                 b["event_ticker"].to_numpy(), seed=RNG_SEED)
        # taker fade: pay fee at ~price plus cross the effective spread
        spr = b["ticker"].str.split("-").str[0].map(
            lambda s: eff_spread_by_series.get(s, 3.0))
        fee = b["price"].map(lambda p: taker_fee_cents(min(max(p, 1), 99)))
        taker_fade_net = (-b["mo_settle"] - fee - spr).to_numpy()
        tf = cluster_bootstrap(taker_fade_net, b["event_ticker"].to_numpy(),
                               seed=RNG_SEED)
        mo30 = b["mo_30m"].dropna()
        s30 = (cluster_bootstrap(mo30.to_numpy(),
                                 b.loc[mo30.index, "event_ticker"].to_numpy(),
                                 seed=RNG_SEED) if len(mo30) > 10 else None)
        print(f"\nK={K} net contracts / 5min: {len(b)} bursts, "
              f"{b.event_ticker.nunique()} events, "
              f"{b.ticker.nunique()} markets")
        print(f"  follow markout->settle (c/ct): mean={stat['mean']:+.2f} "
              f"CI95=[{stat['ci_lo']:+.2f},{stat['ci_hi']:+.2f}] "
              f"p(<=0)={stat['p_le_0']:.3f}")
        if s30:
            print(f"  follow markout->30m tape:      mean={s30['mean']:+.2f} "
                  f"CI95=[{s30['ci_lo']:+.2f},{s30['ci_hi']:+.2f}] n={s30['n']}")
        print(f"  MAKER fade  (net, no fee):     mean={fade['mean']:+.2f} "
              f"CI95=[{fade['ci_lo']:+.2f},{fade['ci_hi']:+.2f}] "
              f"p(<=0)={fade['p_le_0']:.3f}")
        print(f"  TAKER fade  (net fee+spread):  mean={tf['mean']:+.2f} "
              f"CI95=[{tf['ci_lo']:+.2f},{tf['ci_hi']:+.2f}] "
              f"p(<=0)={tf['p_le_0']:.3f}")
        for d, lbl in [(1, "taker-YES bursts"), (-1, "taker-NO bursts")]:
            bd = b[b["dir"] == d]
            if len(bd) < 10:
                continue
            sd = cluster_bootstrap(bd["mo_settle"].to_numpy(),
                                   bd["event_ticker"].to_numpy(), seed=RNG_SEED)
            print(f"    {lbl}: n={sd['n']} follow->settle mean={sd['mean']:+.2f}"
                  f" CI=[{sd['ci_lo']:+.2f},{sd['ci_hi']:+.2f}]"
                  f" p(<=0)={sd['p_le_0']:.3f}")
        # the tradable asymmetry: FADE taker-YES bursts by buying NO.
        by = b[(b["dir"] == 1) & (b.price.between(5, 95))]
        if len(by) > 20:
            entry = (100.0 - by.price).clip(1, 99)
            fee = entry.map(lambda p: taker_fee_cents(p))
            spr = by["ticker"].str.split("-").str[0].map(
                lambda s: eff_spread_by_series.get(s, 3.0))
            gross = np.where(by.result == 0, 100.0, 0.0) - entry
            mk_net = cluster_bootstrap(gross.to_numpy(),
                                       by.event_ticker.to_numpy(), seed=RNG_SEED)
            tk_net = cluster_bootstrap((gross - fee - spr).to_numpy(),
                                       by.event_ticker.to_numpy(), seed=RNG_SEED)
            print(f"    BUY-NO fade of taker-YES bursts (5-95c, n={len(by)}, "
                  f"{by.event_ticker.nunique()} ev):")
            print(f"      as maker (no fee): {mk_net['mean']:+.2f} "
                  f"CI=[{mk_net['ci_lo']:+.2f},{mk_net['ci_hi']:+.2f}] "
                  f"p(<=0)={mk_net['p_le_0']:.3f}")
            print(f"      as taker (fee+spread): {tk_net['mean']:+.2f} "
                  f"CI=[{tk_net['ci_lo']:+.2f},{tk_net['ci_hi']:+.2f}] "
                  f"p(<=0)={tk_net['p_le_0']:.3f}")
        results[K] = {"bursts": b, "follow": stat, "maker_fade": fade,
                      "taker_fade": tf}
    return results


def family1b(j: pd.DataFrame):
    """Markout of the MAKER who is filled by taker flow, by time-to-close.

    A taker-YES print at p means some maker sold YES (accumulated NO) at p:
    maker P&L/contract = p - 100*result, fee-free. Volume-weighted across
    all prints, cluster-bootstrapped by event.
    """
    print("\n" + "=" * 72)
    print("FAMILY 1b — maker markout when filled by taker flow "
          "(sell YES into taker-YES prints), to settlement, fee-free")
    jj = j.copy()
    jj["mins_to_close"] = (jj.close_time - jj.ts).dt.total_seconds() / 60.0
    jj["mo_maker_yesfill"] = jj.yes_price - 100.0 * jj.result  # maker sold YES
    tbuckets = [(-1e9, 10, "<10min"), (10, 60, "10-60min"),
                (60, 360, "1-6h"), (360, 1440, "6-24h"), (1440, 1e9, ">24h")]
    for side_name, mask in [("taker-YES prints (maker sells YES)",
                             jj.taker_yes),
                            ("taker-NO prints (maker buys YES)",
                             ~jj.taker_yes)]:
        print(f"\n  {side_name}:")
        sgn = 1.0 if side_name.startswith("taker-YES") else -1.0
        for lo, hi, lbl in tbuckets:
            g = jj[mask & jj.mins_to_close.between(lo, hi)]
            g = g[g.yes_price.between(3, 97)]  # drop mechanical 1-2c/98-99c
            if len(g) < 200:
                continue
            st = cluster_bootstrap(sgn * g.mo_maker_yesfill,
                                   g.event_ticker, weights=g["count"])
            print(f"    {lbl:>8}: n={st['n']:>7} prints "
                  f"({st['n_clusters']} ev) maker c/ct={st['mean']:+.2f} "
                  f"CI=[{st['ci_lo']:+.2f},{st['ci_hi']:+.2f}] "
                  f"p(<=0)={st['p_le_0']:.3f}")
    # price-bucket view of the headline case: maker selling YES mid-event
    print("\n  maker sells YES into taker-YES prints, 10min-6h before close,"
          " by price bucket:")
    g = jj[jj.taker_yes & jj.mins_to_close.between(10, 360)]
    for plo, phi in [(3, 20), (20, 40), (40, 60), (60, 80), (80, 97)]:
        gg = g[g.yes_price.between(plo, phi)]
        if len(gg) < 200:
            continue
        st = cluster_bootstrap(gg.mo_maker_yesfill, gg.event_ticker,
                               weights=gg["count"])
        print(f"    {plo:>2}-{phi}c: n={st['n']:>7} ({st['n_clusters']} ev) "
              f"maker c/ct={st['mean']:+.2f} "
              f"CI=[{st['ci_lo']:+.2f},{st['ci_hi']:+.2f}] "
              f"p(<=0)={st['p_le_0']:.3f}")


# ------------------------------------------------- family 2: calibration

def family2(j: pd.DataFrame, eff_spread_by_series: dict | None = None) -> pd.DataFrame:
    eff_spread_by_series = eff_spread_by_series or {}
    print("\n" + "=" * 72)
    print("FAMILY 2 — last-trade calibration near close")
    j2 = j.copy()
    j2["mins_to_close"] = (j2.close_time - j2.ts).dt.total_seconds() / 60.0
    out = {}
    for label, lo, hi in [("last trade 10-180min before close", 10, 180),
                          ("last trade >6h before close", 360, 1e9)]:
        w = j2[(j2.mins_to_close >= lo) & (j2.mins_to_close < hi)]
        last = w.sort_values("ts").groupby("ticker").tail(1)
        if len(last) < 30:
            print(f"  [{label}] insufficient n={len(last)}")
            continue
        tab = market_calibration_table(
            last["yes_price"].to_numpy(), last["result"].to_numpy(),
            last["event_ticker"].to_numpy())
        print(f"\n  [{label}] n={len(last)} markets, "
              f"{last.event_ticker.nunique()} events")
        print(tab.round(3).to_string(index=False))
        out[label] = (last, tab)

        # economics: fee-aware, fill assumed at last trade price
        for name, mask, side in [
                ("buy NO vs yes<=15c", last.yes_price <= 15, "no"),
                ("buy YES vs yes>=85c", last.yes_price >= 85, "yes"),
                ("buy NO vs 20c<=yes<=80c",
                 last.yes_price.between(20, 80), "no")]:
            g = last[mask]
            if len(g) < 20:
                continue
            if side == "no":
                entry = 100.0 - g.yes_price  # assume fill at ~last price
                gross = np.where(g.result == 0, 100.0, 0.0) - entry
            else:
                entry = g.yes_price
                gross = np.where(g.result == 1, 100.0, 0.0) - entry
            fee = entry.map(lambda p: taker_fee_cents(min(max(p, 1), 99)))
            net = (gross - fee).to_numpy()
            st = cluster_bootstrap(net, g.event_ticker.to_numpy(),
                                   seed=RNG_SEED)
            print(f"    {name}: n={st['n']} ({st['n_clusters']} ev) "
                  f"net c/ct={st['mean']:+.2f} "
                  f"CI=[{st['ci_lo']:+.2f},{st['ci_hi']:+.2f}] "
                  f"p(<=0)={st['p_le_0']:.3f}")
            if name.startswith("buy NO vs 20c"):
                # conservative fill: if the last print was taker-YES it sat
                # at the ask, so the NO ask is ~(100-p)+effective spread;
                # if taker-NO, NO ask ~= 100-p. Fee re-computed at entry.
                spr = g["ticker"].str.split("-").str[0].map(
                    lambda s: eff_spread_by_series.get(s, 4.0))
                pay_spread = (g["taker_side"] == "yes").astype(float)
                e3 = (100.0 - g.yes_price + spr * pay_spread).clip(1, 99)
                gr3 = np.where(g.result == 0, 100.0, 0.0) - e3
                f3 = e3.map(lambda p: taker_fee_cents(p))
                s3 = cluster_bootstrap((gr3 - f3).to_numpy(),
                                       g.event_ticker.to_numpy(),
                                       seed=RNG_SEED)
                print(f"      spread-aware taker fill: net={s3['mean']:+.2f}"
                      f" CI=[{s3['ci_lo']:+.2f},{s3['ci_hi']:+.2f}] "
                      f"p(<=0)={s3['p_le_0']:.3f}")
                for fam, gg in g.groupby("family"):
                    if len(gg) < 15:
                        continue
                    e2 = 100.0 - gg.yes_price
                    gr2 = np.where(gg.result == 0, 100.0, 0.0) - e2
                    f2 = e2.map(lambda p: taker_fee_cents(min(max(p, 1), 99)))
                    s2 = cluster_bootstrap((gr2 - f2).to_numpy(),
                                           gg.event_ticker.to_numpy(),
                                           seed=RNG_SEED)
                    print(f"      [{fam}] n={s2['n']} ({s2['n_clusters']} ev)"
                          f" net={s2['mean']:+.2f} "
                          f"CI=[{s2['ci_lo']:+.2f},{s2['ci_hi']:+.2f}] "
                          f"p(<=0)={s2['p_le_0']:.3f}")
    return out


def family2b(j: pd.DataFrame):
    """The implementable version of the near-close NO edge.

    market close_time is EVENT-DRIVEN (trading halts when the word is said),
    so anchoring windows on a market's own close_time peeks at the future.
    Anchor instead on the EVENT end (max close_time across the event's
    markets ~= scheduled broadcast/game end, knowable in advance):
    RULE: 60-180 min before event end, if YES still trades 20-80c, buy NO.
    """
    print("\n" + "=" * 72)
    print("FAMILY 2b — event-end-anchored NO rule (implementable)")
    jj = j.join(j.groupby("event_ticker").close_time.max().rename("ev_end"),
                on="event_ticker")
    jj["mte"] = (jj.ev_end - jj.ts).dt.total_seconds() / 60.0
    for lo, hi in [(10, 60), (60, 180), (180, 360)]:
        w = jj[(jj.mte >= lo) & (jj.mte < hi)]
        last = w.sort_values("ts").groupby("ticker").tail(1)
        g = last[last.yes_price.between(20, 80)]
        if len(g) < 20:
            continue
        entry = (100.0 - g.yes_price + 4.0).clip(1, 99)  # +4c spread cross
        fee = entry.map(lambda p: taker_fee_cents(p))
        net = (np.where(g.result == 0, 100.0, 0.0) - entry - fee).astype(float)
        st = cluster_bootstrap(net, g.event_ticker)
        print(f"  [{lo:>3},{hi:>3})min pre event-end, 20-80c: n={st['n']} "
              f"({st['n_clusters']} ev) YES rate={g.result.mean():.3f} "
              f"mean yes px={g.yes_price.mean():.1f}c  "
              f"net(fee+4c spread)={st['mean']:+.2f} "
              f"CI=[{st['ci_lo']:+.2f},{st['ci_hi']:+.2f}] "
              f"p(<=0)={st['p_le_0']:.4f}")
        if lo == 60:
            for fam, gg in g.groupby("family"):
                if len(gg) < 30:
                    continue
                e2 = (100.0 - gg.yes_price + 4.0).clip(1, 99)
                f2 = e2.map(lambda p: taker_fee_cents(p))
                n2 = (np.where(gg.result == 0, 100.0, 0.0) - e2 - f2)
                s2 = cluster_bootstrap(n2.astype(float), gg.event_ticker)
                print(f"      [{fam}] n={s2['n']} ({s2['n_clusters']} ev) "
                      f"net={s2['mean']:+.2f} "
                      f"CI=[{s2['ci_lo']:+.2f},{s2['ci_hi']:+.2f}] "
                      f"p={s2['p_le_0']:.4f}")
            for plo, phi in [(20, 40), (40, 60), (60, 80)]:
                gg = g[g.yes_price.between(plo, phi)]
                if len(gg) < 30:
                    continue
                e2 = (100.0 - gg.yes_price + 4.0).clip(1, 99)
                f2 = e2.map(lambda p: taker_fee_cents(p))
                n2 = (np.where(gg.result == 0, 100.0, 0.0) - e2 - f2)
                s2 = cluster_bootstrap(n2.astype(float), gg.event_ticker)
                print(f"      [{plo}-{phi}c] n={s2['n']} YES={gg.result.mean():.3f}"
                      f" net={s2['mean']:+.2f} "
                      f"CI=[{s2['ci_lo']:+.2f},{s2['ci_hi']:+.2f}] "
                      f"p={s2['p_le_0']:.4f}")
            days = (jj.ev_end.max() - jj.ev_end.min()).total_seconds() / 86400
            print(f"      opportunities: {len(g)/days:.1f} markets/day over "
                  f"{days:.0f} days; median tape volume in final 3h = "
                  f"{w[w.ticker.isin(g.ticker)].groupby('ticker')['count'].sum().median():.0f} contracts")
    return jj


# ------------------------------------------------- family 3: imbalance

def family3(j: pd.DataFrame):
    print("\n" + "=" * 72)
    print("FAMILY 3 — taker-side imbalance vs resolution")
    import statsmodels.api as sm

    j3 = j.copy()
    j3["mins_to_close"] = (j3.close_time - j3.ts).dt.total_seconds() / 60.0
    fits = {}
    # These markets pin to 1c/99c hours before close, so evaluate the signal
    # at several horizons; both signal and price control use only trades up
    # to the cutoff (aligned information sets).
    for cutoff in (60, 180, 360):
        w = j3[j3.mins_to_close >= cutoff]
        if not len(w):
            continue
        vol = w.groupby("ticker").apply(
            lambda g: pd.Series({
                "v_yes": g.loc[g.taker_side == "yes", "count"].sum(),
                "v_no": g.loc[g.taker_side == "no", "count"].sum(),
                "p_last": g.sort_values("ts")["yes_price"].iloc[-1],
            }), include_groups=False)
        meta = w.groupby("ticker").agg(result=("result", "first"),
                                       event_ticker=("event_ticker", "first"),
                                       family=("family", "first"))
        d = vol.join(meta).reset_index()
        d = d[(d.v_yes + d.v_no) > 0]
        d["imb"] = (d.v_yes - d.v_no) / (d.v_yes + d.v_no)
        # extreme prices separate perfectly and blow up the logit; the
        # signal only matters where the market is undecided anyway.
        dd = d[d.p_last.between(3, 97)].copy()
        p = dd.p_last / 100.0
        dd["logit_p"] = np.log(p / (1 - p))
        print(f"\n  --- horizon: >= {cutoff} min before close ---")
        if len(dd) < 50:
            print(f"  insufficient non-extreme markets: n={len(dd)}")
            continue
        X = sm.add_constant(dd[["logit_p", "imb"]])
        try:
            fit = sm.Logit(dd["result"], X).fit(
                disp=0, cov_type="cluster",
                cov_kwds={"groups": dd.event_ticker})
        except Exception as e:  # separation etc.
            print(f"  logit failed ({e}); GLM fallback")
            fit = sm.GLM(dd["result"], X,
                         family=sm.families.Binomial()).fit(
                cov_type="cluster", cov_kwds={"groups": dd.event_ticker})
        print(f"  n={len(dd)} markets (3-97c), "
              f"{dd.event_ticker.nunique()} events")
        print(fit.summary2().tables[1].round(4).to_string())
        fits[cutoff] = (dd, fit)

        # economics: follow flow when |imb| extreme, fill at last price on
        # the flow side, taker fee charged.
        for thr in (0.6, 0.9):
            g = dd[dd.imb.abs() >= thr]
            if len(g) < 20:
                continue
            entry = np.where(g.imb > 0, g.p_last, 100 - g.p_last).clip(1, 99)
            win = np.where(g.imb > 0, g.result == 1, g.result == 0)
            fee = np.array([taker_fee_cents(e) for e in entry])
            net = np.where(win, 100.0, 0.0) - entry - fee
            st = cluster_bootstrap(net, g.event_ticker.to_numpy(),
                                   seed=RNG_SEED)
            print(f"  FOLLOW flow |imb|>={thr}: n={st['n']} "
                  f"({st['n_clusters']} ev) net c/ct={st['mean']:+.2f} "
                  f"CI=[{st['ci_lo']:+.2f},{st['ci_hi']:+.2f}] "
                  f"p(<=0)={st['p_le_0']:.3f}")
            fade = cluster_bootstrap(-(np.where(win, 100.0, 0.0) - entry)
                                     - fee, g.event_ticker.to_numpy(),
                                     seed=RNG_SEED)
            print(f"  FADE   flow |imb|>={thr}: net c/ct={fade['mean']:+.2f} "
                  f"CI=[{fade['ci_lo']:+.2f},{fade['ci_hi']:+.2f}] "
                  f"p(<=0)={fade['p_le_0']:.3f}")
    return fits


# ------------------------------------------------- family 4: spread/capacity

def family4(j: pd.DataFrame):
    print("\n" + "=" * 72)
    print("FAMILY 4 — effective spread + maker capacity by series")
    rows = []
    for tk, g in j.groupby("ticker", sort=False):
        px = g.yes_price.to_numpy()
        side = (g.taker_side == "yes").to_numpy().astype(int)
        if len(px) < 5:
            continue
        flips = side[1:] != side[:-1]
        bounce = np.abs(np.diff(px))[flips]
        dp = np.diff(px)
        roll = np.nan
        if len(dp) > 4:
            cov = np.cov(dp[1:], dp[:-1])[0, 1]
            roll = 2 * np.sqrt(-cov) if cov < 0 else 0.0
        days = max((g.ts.max() - g.ts.min()).total_seconds() / 86400, 1 / 24)
        rows.append({
            "ticker": tk, "series": g.series.iloc[0], "family": g.family.iloc[0],
            "eff_spread": np.mean(bounce) if len(bounce) else np.nan,
            "roll": roll, "med_size": g["count"].median(),
            "vol_per_day": g["count"].sum() / days,
            "trades": len(g),
        })
    m = pd.DataFrame(rows)
    agg = m.groupby("series").agg(
        n_mkts=("ticker", "size"),
        eff_spread_c=("eff_spread", "median"),
        roll_c=("roll", "median"),
        med_trade_size=("med_size", "median"),
        vol_per_mkt_day=("vol_per_day", "median"),
    ).sort_values("n_mkts", ascending=False)
    # maker capacity: capture half the effective spread on ~25% of volume
    agg["maker_$_per_mkt_day"] = (agg.vol_per_mkt_day * 0.25
                                  * (agg.eff_spread_c / 2) / 100)
    print(agg.round(2).head(15).to_string())
    eff_by_series = (m.groupby("series")["eff_spread"].median()
                     .fillna(4.0).to_dict())
    return m, agg, eff_by_series


# ------------------------------------------------- family 5: anomalies

def family5(tr: pd.DataFrame, j: pd.DataFrame):
    print("\n" + "=" * 72)
    print("FAMILY 5 — anomalies")
    frac = (tr["count"] % 1 != 0).mean()
    print(f"  fractional-contract trades: {frac:.1%}")
    blk = tr.block.mean()
    print(f"  block trades: {blk:.2%}")
    # trades printed AFTER market close
    after = j[j.ts > j.close_time]
    print(f"  trades after close_time: {len(after)} "
          f"({after.ticker.nunique()} markets)")
    # same-timestamp sweep clusters (one taker sweeping levels)
    g = j.groupby(["ticker", "ts"]).size()
    print(f"  multi-print timestamps (sweeps): {(g > 1).mean():.1%} of stamps;"
          f" max prints in one stamp: {g.max()}")
    # extreme prices
    ext = j[(j.yes_price <= 2) | (j.yes_price >= 98)]
    print(f"  trades at <=2c or >=98c: {len(ext)} ({len(ext)/len(j):.1%})")
    if len(ext):
        won = np.where(ext.yes_price >= 98, ext.result == 1, ext.result == 0)
        entry = np.where(ext.yes_price >= 98, ext.yes_price,
                         100 - ext.yes_price)
        fee = np.array([taker_fee_cents(e) for e in entry])
        net = np.where(won, 100.0, 0.0) - entry - fee
        st = cluster_bootstrap(net, ext.event_ticker.to_numpy(),
                               seed=RNG_SEED)
        print(f"    taker buying the favorite side at the extreme won "
              f"{won.mean():.1%}; net EV c/ct={st['mean']:+.2f} "
              f"CI=[{st['ci_lo']:+.2f},{st['ci_hi']:+.2f}] "
              f"p(<=0)={st['p_le_0']:.3f}")
    return after


# ---------------------------------------------------------------- main

def main():
    mk, tr, j = build_dataset()
    print(f"dataset: {len(mk)} settled markets / {mk.event_ticker.nunique()} "
          f"events; tapes for {j.ticker.nunique()} markets "
          f"({j.event_ticker.nunique()} events), {len(j)} trades")
    print("family coverage:",
          j.groupby("family").ticker.nunique().to_dict())

    m4, agg4, eff = family4(j)
    family1(j, eff)
    family1b(j)
    family2(j, eff)
    family2b(j)
    family3(j)
    family5(tr, j)


if __name__ == "__main__":
    main()
