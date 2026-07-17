#!/usr/bin/env python
"""Hearing-mention (KXHEARINGMENTION) research model.

Walk-forward P(word said in hearing) from:
  - prior yes-rate of same strike word across earlier hearing events (smoothed)
  - word presence in the event title (hearing-topic proxy)
  - strike type flags (count strikes, 'Event does not qualify')
  - optional: document frequency of the word in a govinfo CHRG transcript corpus

Then the encompassing test vs the market price at hearing-day 12:00Z (08:00 ET,
pre-hearing), logistic outcome ~ {model logit, market logit}, event-clustered SEs.

Read-only research; no orders, no Kalshi credentials.
"""
from __future__ import annotations

import datetime as dt
import glob
import html as htmllib
import json
import os
import re
import sys
import collections

import numpy as np

DATA = "data/research/mentions"
CHRG_DIR = os.environ.get("CHRG_DIR", "/tmp/chrg")


def parse_ts(t):
    return dt.datetime.fromisoformat(t.replace("Z", "+00:00"))


def load_markets():
    mkts = {}
    with open(f"{DATA}/events.jsonl") as f:
        for line in f:
            e = json.loads(line)
            if "KXHEARINGMENTION" not in e.get("event_ticker", ""):
                continue
            for m in e.get("markets", []):
                if m.get("result") not in ("yes", "no"):
                    continue
                word = (m.get("custom_strike") or {}).get("Word") or m.get("no_sub_title")
                mkts[m["ticker"]] = dict(
                    event=e["event_ticker"],
                    word=word,
                    result=m["result"],
                    close=m["close_time"],
                    title=e.get("title", ""),
                )
    # hearing day = earliest close date within event
    ev_close = collections.defaultdict(list)
    for v in mkts.values():
        ev_close[v["event"]].append(parse_ts(v["close"]))
    ev_day = {ev: min(cs).date() for ev, cs in ev_close.items()}
    for v in mkts.values():
        v["day"] = ev_day[v["event"]]
    return mkts


def snapshot_prices(mkts, hour_utc=12):
    """Last trade strictly before hearing-day {hour_utc}:00Z per ticker."""
    out = {}
    with open(f"{DATA}/trades.jsonl") as f:
        for line in f:
            r = json.loads(line)
            t = r["ticker"]
            if t not in mkts:
                continue
            d = mkts[t]["day"]
            cut = dt.datetime(d.year, d.month, d.day, hour_utc, tzinfo=dt.timezone.utc)
            pre = [tr for tr in r["trades"] if parse_ts(tr["created_time"]) < cut]
            if pre:
                pre.sort(key=lambda tr: tr["created_time"])
                out[t] = float(pre[-1]["yes_price_dollars"])
    return out


COUNT_RE = re.compile(r"\((\d+)\+\s*times\)", re.I)


def word_key(word):
    """Normalize a strike to (variants tuple, count threshold, ndq flag)."""
    w = word.strip()
    if w.lower().startswith("event does not qualify"):
        return ("__NDQ__",), 1, True
    m = COUNT_RE.search(w)
    k = 1
    if m:
        k = int(m.group(1))
        w = COUNT_RE.sub("", w).strip()
    variants = tuple(v.strip().lower() for v in w.split("/") if v.strip())
    return variants, k, False


def build_corpus_df():
    """Document frequency of each variant in govinfo CHRG transcripts."""
    files = sorted(glob.glob(f"{CHRG_DIR}/CHRG-*.htm"))
    docs = []
    for fp in files:
        try:
            txt = open(fp, errors="ignore").read()
        except OSError:
            continue
        txt = htmllib.unescape(re.sub(r"<[^>]+>", " ", txt)).lower()
        if len(txt) < 20000:
            continue
        docs.append(txt)
    return docs


def variant_df(docs, variants, k=1):
    """Fraction of transcripts containing any variant >= k times (word-boundary)."""
    if not docs:
        return None
    pats = [re.compile(r"\b" + re.escape(v) + r"(s|'s|es)?\b") for v in variants]
    hit = 0
    for d in docs:
        n = 0
        for p in pats:
            n += len(p.findall(d))
            if n >= k:
                break
        if n >= k:
            hit += 1
    return hit / len(docs)


def logit(p, lo=0.01, hi=0.99):
    p = np.clip(p, lo, hi)
    return np.log(p / (1 - p))


def main():
    mkts = load_markets()
    prices = snapshot_prices(mkts, hour_utc=12)
    docs = build_corpus_df()
    print(f"markets={len(mkts)} events={len({v['event'] for v in mkts.values()})} "
          f"priced@12Z={len(prices)} corpus_docs={len(docs)}", flush=True)

    rows = []
    for t, v in mkts.items():
        variants, k, ndq = word_key(v["word"])
        rows.append(dict(ticker=t, event=v["event"], day=v["day"],
                         y=1.0 if v["result"] == "yes" else 0.0,
                         variants=variants, k=k, ndq=ndq,
                         title=v["title"].lower(),
                         q=prices.get(t)))

    # corpus DF feature (computed once per unique (variants,k))
    df_cache = {}
    for r in rows:
        key = (r["variants"], r["k"])
        if key not in df_cache:
            df_cache[key] = None if r["ndq"] else variant_df(docs, r["variants"], r["k"])
        r["corpus_df"] = df_cache[key]

    # walk-forward features by hearing day
    days = sorted({r["day"] for r in rows})
    rows.sort(key=lambda r: r["day"])
    hist = collections.defaultdict(lambda: [0, 0])  # variants-key -> [yes, n]
    ndq_hist = [0, 0]
    feats = []
    for day in days:
        todays = [r for r in rows if r["day"] == day]
        for r in todays:
            hk = r["variants"]
            ys, n = (ndq_hist if r["ndq"] else hist[hk])
            r["prior_rate"] = (ys + 2.0) / (n + 4.0)  # Beta(2,2) shrink to 0.5
            r["prior_n"] = n
            r["title_hit"] = float(any(re.search(r"\b" + re.escape(v), r["title"]) for v in r["variants"])) if not r["ndq"] else 0.0
        for r in todays:  # update after the whole day (no intra-day leakage)
            tgt = ndq_hist if r["ndq"] else hist[r["variants"]]
            tgt[0] += int(r["y"]); tgt[1] += 1
    feats = rows

    # ---- walk-forward logistic model: expanding window by event day ----
    import statsmodels.api as sm

    def design(rs, use_corpus):
        cols = []
        for r in rs:
            cdf = r["corpus_df"]
            c_logit = logit(cdf if cdf is not None else 0.5) if use_corpus else 0.0
            cols.append([1.0, logit(r["prior_rate"]), np.log1p(r["prior_n"]),
                         r["title_hit"], float(r["ndq"]), float(r["k"] > 1), c_logit])
        X = np.array(cols)
        if not use_corpus:
            X = X[:, :6]
        return X

    use_corpus = len(docs) >= 30
    from sklearn.linear_model import LogisticRegression
    oos = []
    min_train_events = 8
    for i, day in enumerate(days):
        train = [r for r in feats if r["day"] < day]
        test = [r for r in feats if r["day"] == day]
        if len({r["event"] for r in train}) < min_train_events:
            continue
        Xtr, ytr = design(train, use_corpus)[:, 1:], np.array([r["y"] for r in train])
        Xte = design(test, use_corpus)[:, 1:]
        clf = LogisticRegression(C=1.0, max_iter=1000)
        clf.fit(Xtr, ytr)
        p = clf.predict_proba(Xte)[:, 1]
        for r, pi in zip(test, p):
            r["model_p"] = float(np.clip(pi, 0.01, 0.99))
            oos.append(r)

    ys = np.array([r["y"] for r in oos])
    mp = np.array([r["model_p"] for r in oos])
    print(f"\nOOS walk-forward: n={len(oos)} events={len({r['event'] for r in oos})}")
    print(f"model Brier={np.mean((mp-ys)**2):.4f}  base(0.5)=0.2500")
    try:
        from sklearn.metrics import roc_auc_score
        print(f"model AUC={roc_auc_score(ys, mp):.3f}")
    except Exception:
        pass

    both = [r for r in oos if r["q"] is not None]
    ys2 = np.array([r["y"] for r in both])
    mp2 = np.array([r["model_p"] for r in both])
    q2 = np.array([r["q"] for r in both])
    ev2 = np.array([r["event"] for r in both])
    print(f"\nwith market price @12Z: n={len(both)} events={len(set(ev2))}")
    print(f"market Brier={np.mean((np.clip(q2,0.01,0.99)-ys2)**2):.4f}  model Brier={np.mean((mp2-ys2)**2):.4f}")
    try:
        from sklearn.metrics import roc_auc_score
        print(f"market AUC={roc_auc_score(ys2, q2):.3f}  model AUC={roc_auc_score(ys2, mp2):.3f}")
    except Exception:
        pass

    # ---- encompassing regression, event-clustered ----
    X = sm.add_constant(np.column_stack([logit(mp2), logit(q2)]))
    fit = sm.Logit(ys2, X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": ev2})
    print("\nEncompassing: y ~ const + model_logit + market_logit (cluster=event)")
    for name, b, se, pv in zip(["const", "model", "market"], fit.params, fit.bse, fit.pvalues):
        print(f"  {name:7s} beta={b:+.3f} se={se:.3f} p={pv:.4f}")

    # ---- threshold taker rule after fees ----
    print("\nThreshold taker rule (buy side where model_p - price > thr), fees=ceil(7*P*(1-P)) cents/contract:")
    for thr in (0.05, 0.10, 0.15, 0.20):
        pnl = []
        for r in both:
            p, q, y = r["model_p"], r["q"], r["y"]
            if q <= 0.01 or q >= 0.99:
                continue
            edge_yes, edge_no = p - q, q - p
            if edge_yes > thr:
                fee = np.ceil(7.0 * q * (1 - q)) / 100.0
                pnl.append((y - q) - fee)
            elif edge_no > thr:
                qn = 1 - q
                fee = np.ceil(7.0 * qn * (1 - qn)) / 100.0
                pnl.append(((1 - y) - qn) - fee)
        if pnl:
            a = np.array(pnl)
            print(f"  thr={thr:.2f}: n={len(a)} mean={a.mean():+.4f} total={a.sum():+.2f} hit={np.mean(a>0):.2f}")
        else:
            print(f"  thr={thr:.2f}: no trades")

    # ---- market-only vs market+corpus-DF sanity: does corpus_df alone add? ----
    cd = np.array([logit(r["corpus_df"]) if r["corpus_df"] is not None else 0.0 for r in both])
    if use_corpus:
        X2 = sm.add_constant(np.column_stack([cd, logit(q2)]))
        f2 = sm.Logit(ys2, X2).fit(disp=0, cov_type="cluster", cov_kwds={"groups": ev2})
        print("\nEncompassing (corpus DF only vs market), OOS subset:")
        for name, b, se, pv in zip(["const", "corpusDF", "market"], f2.params, f2.bse, f2.pvalues):
            print(f"  {name:8s} beta={b:+.3f} se={se:.3f} p={pv:.4f}")

    # ---- feature-level encompassing on ALL priced markets (features are
    # walk-forward-safe by construction; more power than the OOS subset) ----
    allp = [r for r in feats if r["q"] is not None and "prior_rate" in r]
    ya = np.array([r["y"] for r in allp])
    qa = np.array([r["q"] for r in allp])
    eva = np.array([r["event"] for r in allp])
    cda = np.array([logit(r["corpus_df"]) if r["corpus_df"] is not None else 0.0 for r in allp])
    pra = np.array([logit(r["prior_rate"]) for r in allp])
    tha = np.array([r["title_hit"] for r in allp])
    nda = np.array([float(r["ndq"]) for r in allp])
    print(f"\nFull-sample feature encompassing: n={len(allp)} events={len(set(eva))}")
    print(f"market Brier(all)={np.mean((np.clip(qa,0.01,0.99)-ya)**2):.4f}")
    X3 = sm.add_constant(np.column_stack([cda, pra, tha, nda, logit(qa)]))
    f3 = sm.Logit(ya, X3).fit(disp=0, cov_type="cluster", cov_kwds={"groups": eva}, maxiter=200)
    for name, b, se, pv in zip(["const", "corpusDF", "priorRate", "titleHit", "NDQ", "market"],
                               f3.params, f3.bse, f3.pvalues):
        print(f"  {name:9s} beta={b:+.3f} se={se:.3f} p={pv:.4f}")
    # corpus DF alone vs market, full sample
    X4 = sm.add_constant(np.column_stack([cda, logit(qa)]))
    f4 = sm.Logit(ya, X4).fit(disp=0, cov_type="cluster", cov_kwds={"groups": eva})
    print("corpusDF alone vs market (full sample):")
    for name, b, se, pv in zip(["const", "corpusDF", "market"], f4.params, f4.bse, f4.pvalues):
        print(f"  {name:9s} beta={b:+.3f} se={se:.3f} p={pv:.4f}")

    json.dump([{k: (str(v) if isinstance(v, (dt.date, tuple)) else v) for k, v in r.items()}
               for r in oos], open("/tmp/hearing_oos.json", "w"))


if __name__ == "__main__":
    main()
