#!/usr/bin/env python
"""GDELT news-flow nowcast pilot for Kalshi mention markets.

Route: does a pre-event surge in news coverage of a phrase predict
(a) mention-market resolution and (b) subsequent Kalshi repricing?

Subcommands:
  prep     build the political mention-market table from events.jsonl
  fetch    pull one GDELT timelinevol per phrase (daily res, whole range)
  analyze  surge features -> outcome tests + encompassing vs candle price

GDELT DOC 2.0 API: plain-HTTP works from this box (HTTPS gets TLS-reset);
hard rate limit one request / 5 s -> we clock at 7 s with backoff.
"""
import argparse
import collections
import datetime as dt
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "research", "mentions")
OUT = os.path.join(DATA, "gdelt")
GDELT = "http://api.gdeltproject.org/api/v2/doc/doc"
UA = "hedge-research/0.1"

POL = {
    "KXTRUMPMENTION", "KXTRUMPMENTIONB", "KXVANCEMENTION", "KXMAMDANIMENTION",
    "KXHEARINGMENTION", "KXPOLITICSMENTION", "KXPSAKIMENTION", "KXHOCHULMENTION",
    "KXBESSENTMTPMENTION", "KXBERNIEMENTION", "KXRUBIOMENTION", "KXCARNEYMENTION",
    "KXNEWSOMMENTION", "KXHEGSETHMENTION", "KXMTPMENTION", "KXFOXNEWSMENTION",
    "KXMADDOWMENTION", "KXLASTWORDMENTION", "KXTRUMPSAY", "KXFTNMENTION",
    "KXSTARMERMENTIONB", "KXWORLDNEWSMENTION",
}
MO = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}


def event_day(event_ticker):
    m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})", event_ticker)
    if not m:
        return None
    return dt.date(2000 + int(m.group(1)), MO[m.group(2)], int(m.group(3)))


def base_word(word):
    """Strip count annotations: 'Trump (3+ times)' -> 'Trump'."""
    return re.sub(r"\s*\(.*?\)\s*", "", word or "").strip()


def gdelt_query(word):
    """Map a Kalshi strike word to a GDELT DOC query, or None if unusable."""
    parts = [p.strip() for p in base_word(word).split("/") if p.strip()]
    terms = []
    for p in parts:
        if p.lower() in {"event does not qualify", "other"}:
            return None
        if re.fullmatch(r"\d+", p):  # bare numbers unusable
            continue
        if len(p) < 3:  # GDELT rejects short keywords (e.g. 'AI')
            continue
        terms.append('"%s"' % p if " " in p else p)
    if not terms:
        return None
    core = terms[0] if len(terms) == 1 else "(" + " OR ".join(terms) + ")"
    return core + " sourcelang:english"


def cmd_prep(args):
    rows = []
    with open(os.path.join(DATA, "events.jsonl")) as f:
        for line in f:
            e = json.loads(line)
            s = e["event_ticker"].split("-")[0]
            if s not in POL:
                continue
            for m in e.get("markets", []):
                if m.get("result") not in ("yes", "no"):
                    continue
                w = (m.get("custom_strike") or {}).get("Word") or m.get("yes_sub_title")
                d = event_day(e["event_ticker"])
                if not w or not d:
                    continue
                rows.append(dict(
                    ticker=m["ticker"], event=e["event_ticker"], series=s,
                    word=w, bword=base_word(w), day=d.isoformat(),
                    result=1 if m["result"] == "yes" else 0,
                    close=m.get("close_time"), open=m.get("open_time"),
                    vol=float(m.get("volume_fp") or 0)))
    os.makedirs(OUT, exist_ok=True)
    json.dump(rows, open(os.path.join(OUT, "markets.json"), "w"))
    # pick words to fetch: enough within-word event-day variation + both outcomes
    byw = collections.defaultdict(list)
    for r in rows:
        byw[r["bword"]].append(r)
    sel = []
    for w, rs in byw.items():
        days = {r["day"] for r in rs}
        ys = {r["result"] for r in rs}
        if len(days) >= args.min_days and len(ys) == 2 and gdelt_query(w):
            sel.append((w, len(days), len(rs)))
    sel.sort(key=lambda x: -x[1])
    sel = sel[: args.max_words]
    json.dump([w for w, _, _ in sel], open(os.path.join(OUT, "words.json"), "w"))
    print("markets:", len(rows), "words selected:", len(sel))
    for w, nd, nm in sel:
        print(f"  {w!r}: {nd} days, {nm} markets -> {gdelt_query(w)}")


def fetch_one(query, start, end):
    params = dict(query=query, mode="timelinevol", format="json",
                  startdatetime=start, enddatetime=end)
    url = GDELT + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=40) as r:
                body = r.read().decode("utf-8", "replace")
            if body.lstrip().startswith("{"):
                return json.loads(body)
            print("    non-json response:", body[:120].replace("\n", " "))
            return None
        except Exception as ex:  # 429s raise HTTPError; resets raise URLError
            # each 429 restarts/extends the penalty window — never retry eagerly
            wait = 320
            print(f"    attempt {attempt}: {ex}; sleep {wait}s", flush=True)
            time.sleep(wait)
    return None


def cmd_fetch(args):
    words = json.load(open(os.path.join(OUT, "words.json")))
    tldir = os.path.join(OUT, "timelines")
    os.makedirs(tldir, exist_ok=True)
    for i, w in enumerate(words):
        fn = os.path.join(tldir, re.sub(r"[^A-Za-z0-9]+", "_", w) + ".json")
        if os.path.exists(fn):
            continue
        q = gdelt_query(w)
        print(f"[{i+1}/{len(words)}] {w!r} -> {q}")
        d = fetch_one(q, args.start, args.end)
        if d and d.get("timeline"):
            json.dump({"word": w, "query": q, "timeline": d["timeline"][0]["data"]},
                      open(fn, "w"))
            print("    saved", len(d["timeline"][0]["data"]), "points")
        else:
            print("    FAILED / empty")
        time.sleep(args.pause)
    print("done")


def load_timelines():
    tldir = os.path.join(OUT, "timelines")
    out = {}
    for fn in os.listdir(tldir):
        d = json.load(open(os.path.join(tldir, fn)))
        series = {}
        for p in d["timeline"]:
            day = p["date"][:8]
            series[f"{day[:4]}-{day[4:6]}-{day[6:8]}"] = p["value"]
        out[d["word"]] = series
    return out


def surge_features(series, day_iso):
    """News surge strictly BEFORE event day: recent (D-2,D-1) vs base (D-16..D-3)."""
    d0 = dt.date.fromisoformat(day_iso)
    recent = [series.get((d0 - dt.timedelta(days=k)).isoformat()) for k in (1, 2)]
    base = [series.get((d0 - dt.timedelta(days=k)).isoformat()) for k in range(3, 17)]
    recent = [v for v in recent if v is not None]
    base = [v for v in base if v is not None]
    if len(recent) < 2 or len(base) < 10:
        return None
    import statistics
    mu = statistics.mean(base)
    sd = statistics.pstdev(base)
    r = statistics.mean(recent)
    z = (r - mu) / sd if sd > 1e-12 else 0.0
    lr = (r + 1e-4) / (mu + 1e-4)
    return dict(surge_z=max(min(z, 5.0), -5.0), log_ratio=min(max(lr, 0.05), 20.0),
                recent=r, base_mu=mu)


def entry_cutoff_ts(r):
    """Signal-known-at cutoff: 12:00 UTC on event day, capped at close-2h."""
    d = dt.datetime.fromisoformat(r["day"] + "T12:00:00+00:00").timestamp()
    c = dt.datetime.fromisoformat(r["close"].replace("Z", "+00:00")).timestamp()
    return min(d, c - 2 * 3600)


def cmd_prices(args):
    rows = json.load(open(os.path.join(OUT, "markets.json")))
    byt = {r["ticker"]: r for r in rows}
    prices = {}
    for fn in ("candles.jsonl", os.path.join("gdelt", "candles_extra.jsonl")):
        path = os.path.join(DATA, fn)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                c = json.loads(line)
                r = byt.get(c["ticker"])
                if not r:
                    continue
                et = entry_cutoff_ts(r)
                best = None
                for cd in c.get("candles") or []:
                    if cd["end_period_ts"] <= et:
                        try:
                            b = float(cd["yes_bid"]["close_dollars"])
                            a = float(cd["yes_ask"]["close_dollars"])
                        except Exception:
                            continue
                        if 0.0 < b < 1.0 and 0.0 < a < 1.0 and a >= b:
                            if best is None or cd["end_period_ts"] > best[0]:
                                best = (cd["end_period_ts"], b, a)
                if best:
                    prices[c["ticker"]] = {"q_ts": best[0], "bid": best[1], "ask": best[2]}
    with open(os.path.join(DATA, "trades.jsonl")) as f:
        for line in f:
            c = json.loads(line)
            r = byt.get(c["ticker"])
            if not r:
                continue
            et = entry_cutoff_ts(r)
            best = None
            for t in c.get("trades") or []:
                tts = dt.datetime.fromisoformat(
                    t["created_time"].replace("Z", "+00:00")).timestamp()
                if tts <= et and (best is None or tts > best[0]):
                    best = (tts, float(t["yes_price_dollars"]))
            if best:
                prices.setdefault(c["ticker"], {})
                prices[c["ticker"]].update({"t_ts": best[0], "last": best[1]})
    nb = sum(1 for p in prices.values() if "ask" in p)
    nt = sum(1 for p in prices.values() if "last" in p)
    print(f"prices.json: {len(prices)} tickers ({nb} with book quote, {nt} with last trade)")
    json.dump(prices, open(os.path.join(OUT, "prices.json"), "w"))


def _logit(p):
    import numpy as np
    p = np.clip(p, 0.01, 0.99)
    return np.log(p / (1 - p))


def _fit_logit(X, y):
    """Plain MLE logistic via statsmodels; returns coefs or None."""
    import numpy as np
    import statsmodels.api as sm
    try:
        m = sm.Logit(y, sm.add_constant(X, has_constant="add")).fit(disp=0, maxiter=100)
        return np.asarray(m.params)
    except Exception:
        return None


def _cluster_boot(X, y, days, B=2000, seed=7):
    """Bootstrap coef distribution resampling whole event-days."""
    import numpy as np
    rng = np.random.default_rng(seed)
    udays = np.unique(days)
    idx_by_day = {d: np.where(days == d)[0] for d in udays}
    full = _fit_logit(X, y)
    if full is None:
        return None, None
    boots = []
    for _ in range(B):
        pick = rng.choice(udays, size=len(udays), replace=True)
        idx = np.concatenate([idx_by_day[d] for d in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        b = _fit_logit(X[idx], y[idx])
        if b is not None and np.all(np.isfinite(b)):
            boots.append(b)
    boots = np.array(boots)
    return full, boots


def _report_coef(name, full, boots, j):
    import numpy as np
    if full is None or boots is None or len(boots) < 100:
        print(f"  {name}: FIT FAILED")
        return
    bj = boots[:, j]
    se = bj.std()
    # two-sided bootstrap p: fraction of boot draws crossing zero (symmetrized)
    p = 2 * min((bj <= 0).mean(), (bj >= 0).mean())
    p = max(p, 1.0 / len(bj))
    print(f"  {name}: beta={full[j]:+.3f}  boot_se={se:.3f}  boot_p={p:.4f}  (B={len(bj)})")


def _auc(y, x):
    import numpy as np
    from scipy.stats import rankdata
    y = np.asarray(y); x = np.asarray(x)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(x)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def cmd_analyze(args):
    import numpy as np
    rows = json.load(open(os.path.join(OUT, "markets.json")))
    prices = json.load(open(os.path.join(OUT, "prices.json")))
    tls = load_timelines()
    feats = []
    for r in rows:
        # anchor sanity: event must close within [-1, +2] days of ticker day,
        # else the "pre-event" GDELT window may overlap the event itself
        dd = (dt.datetime.fromisoformat(r["close"].replace("Z", "+00:00")).date()
              - dt.date.fromisoformat(r["day"])).days
        if not (-1 <= dd <= 2):
            continue
        s = tls.get(r["bword"])
        if not s:
            continue
        f = surge_features(s, r["day"])
        if not f:
            continue
        pr = prices.get(r["ticker"], {})
        feats.append({**r, **f, **pr})
    json.dump(feats, open(os.path.join(OUT, "features.json"), "w"))
    words = sorted({f["bword"] for f in feats})
    print(f"markets with GDELT feature: {len(feats)}  words: {len(words)}  "
          f"days: {len({f['day'] for f in feats})}")
    print(f"  with last-trade price: {sum(1 for f in feats if 'last' in f)}  "
          f"with candle bid/ask: {sum(1 for f in feats if 'ask' in f)}")

    y = np.array([f["result"] for f in feats], float)
    z = np.array([f["surge_z"] for f in feats])
    lr = np.log(np.array([f["log_ratio"] for f in feats]))
    days = np.array([f["day"] for f in feats])
    wIdx = {w: i for i, w in enumerate(words)}
    wcol = np.array([wIdx[f["bword"]] for f in feats])
    # within-word demeaning of signal AND outcome-base-rate via word dummies
    zc = z.copy(); lrc = lr.copy()
    for i in range(len(words)):
        m = wcol == i
        zc[m] -= zc[m].mean()
        lrc[m] -= lrc[m].mean()

    print("\n[T1] Pooled: result ~ surge_z (no controls, day-clustered)")
    full, boots = _cluster_boot(z[:, None], y, days, B=args.B)
    _report_coef("surge_z", full, boots, 1)
    print("      pooled AUC(surge_z):", round(_auc(y, z), 3))

    print("\n[T2] Within-word: result ~ surge_z_centered + word dummies")
    D = np.zeros((len(feats), len(words) - 1))
    for i in range(1, len(words)):
        D[wcol == i, i - 1] = 1.0
    X2 = np.column_stack([zc, D])
    full, boots = _cluster_boot(X2, y, days, B=args.B)
    _report_coef("surge_z(within)", full, boots, 1)
    # within-word AUC: average per-word AUC weighted by pairs
    aucs = []
    for i in range(len(words)):
        m = wcol == i
        a = _auc(y[m], z[m])
        if not np.isnan(a):
            n1 = y[m].sum(); n0 = m.sum() - n1
            aucs.append((a, n1 * n0))
    wa = sum(a * w for a, w in aucs) / sum(w for _, w in aucs)
    print("      within-word AUC (pair-weighted):", round(wa, 3))

    print("\n[T3] Encompassing vs LAST-TRADE price at cutoff")
    m3 = np.array(["last" in f for f in feats])
    if m3.sum() > 50:
        p_mkt = np.array([f.get("last", np.nan) for f in feats])[m3]
        X3 = np.column_stack([_logit(p_mkt), zc[m3]])
        full, boots = _cluster_boot(X3, y[m3], days[m3], B=args.B)
        print(f"  n={m3.sum()}")
        _report_coef("logit(p_mkt)", full, boots, 1)
        _report_coef("surge_z(within)", full, boots, 2)
        print("      AUC(p_mkt):", round(_auc(y[m3], p_mkt), 3),
              " AUC(surge within):", round(_auc(y[m3], zc[m3]), 3))

    print("\n[T4] Encompassing vs CANDLE MID at cutoff (executable book exists)")
    m4 = np.array(["ask" in f for f in feats])
    if m4.sum() > 50:
        mid = np.array([(f.get("bid", 0) + f.get("ask", 0)) / 2 if "ask" in f else np.nan
                        for f in feats])[m4]
        X4 = np.column_stack([_logit(mid), zc[m4]])
        full, boots = _cluster_boot(X4, y[m4], days[m4], B=args.B)
        print(f"  n={m4.sum()}")
        _report_coef("logit(mid)", full, boots, 1)
        _report_coef("surge_z(within)", full, boots, 2)

    print("\n[T5] Does surge predict the REPRICE (settle - price at cutoff)?")
    if m3.sum() > 50:
        dp = y[m3] - np.array([f["last"] for f in feats])[m3]
        # OLS with day-cluster bootstrap
        rng = np.random.default_rng(11)
        ud = np.unique(days[m3]); ibd = {d: np.where(days[m3] == d)[0] for d in ud}
        Z = zc[m3]
        bfull = np.polyfit(Z, dp, 1)[0]
        bs = []
        for _ in range(args.B):
            pick = rng.choice(ud, size=len(ud), replace=True)
            idx = np.concatenate([ibd[d] for d in pick])
            if Z[idx].std() > 1e-9:
                bs.append(np.polyfit(Z[idx], dp[idx], 1)[0])
        bs = np.array(bs)
        p = 2 * min((bs <= 0).mean(), (bs >= 0).mean()); p = max(p, 1 / len(bs))
        print(f"  d(price) per 1sd surge: {bfull:+.4f}  boot_se={bs.std():.4f}  boot_p={p:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--min-days", type=int, default=8)
    p.add_argument("--max-words", type=int, default=45)
    p = sub.add_parser("fetch")
    p.add_argument("--start", default="20260420000000")
    p.add_argument("--end", default="20260721000000")
    p.add_argument("--pause", type=float, default=7.0)
    p = sub.add_parser("prices")
    p = sub.add_parser("analyze")
    p.add_argument("--B", type=int, default=2000)
    args = ap.parse_args()
    {"prep": cmd_prep, "fetch": cmd_fetch, "prices": cmd_prices,
     "analyze": cmd_analyze}[args.cmd](args)
