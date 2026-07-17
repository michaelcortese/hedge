#!/usr/bin/env python
"""Targeted hourly-candle backfill for the GDELT-nowcast pilot universe.

Fetches Kalshi hourly candlesticks ONLY for political mention markets in the
45-word pilot (see research_gdelt_nowcast.py prep) that are missing from
data/research/mentions/candles.jsonl. Appends to a SEPARATE file
(gdelt/candles_extra.jsonl) so the main collector's resume logic is untouched.

Anonymous public API, self-clocked at 1 req/s, resumable.
"""
import datetime as dt
import json
import os
import time
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "research", "mentions")
OUT = os.path.join(DATA, "gdelt", "candles_extra.jsonl")
BASE = "https://api.elections.kalshi.com/trade-api/v2"
_last = [0.0]


def get(path, params, tries=6):
    url = BASE + path + "?" + urllib.parse.urlencode(params)
    for i in range(tries):
        wait = 1.0 - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "hedge-research/0.1"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception as e:
            if getattr(e, "code", None) == 404:
                return None
            time.sleep(min(120, 5 * 2 ** i))
    return None


def main():
    rows = json.load(open(os.path.join(DATA, "gdelt", "markets.json")))
    words = set(json.load(open(os.path.join(DATA, "gdelt", "words.json"))))
    have = set()
    for fn in (os.path.join(DATA, "candles.jsonl"), OUT):
        if os.path.exists(fn):
            for line in open(fn):
                try:
                    have.add(json.loads(line)["ticker"])
                except Exception:
                    pass

    def anchor_ok(r):
        dd = (dt.datetime.fromisoformat(r["close"].replace("Z", "+00:00")).date()
              - dt.date.fromisoformat(r["day"])).days
        return -1 <= dd <= 2

    todo = [r for r in rows if r["bword"] in words and anchor_ok(r)
            and r["ticker"] not in have]
    # newest first so partial data stays analysis-ready on the OOS (newer) side
    todo.sort(key=lambda r: r["close"], reverse=True)
    print(f"todo: {len(todo)} tickers", flush=True)
    def ts(s):
        return int(dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())
    with open(OUT, "a") as fh:
        for i, r in enumerate(todo):
            series = r["ticker"].split("-")[0]
            d = get(f"/series/{series}/markets/{r['ticker']}/candlesticks",
                    {"start_ts": ts(r["open"]) - 3600 if r.get("open") else 0,
                     "end_ts": ts(r["close"]) + 3600, "period_interval": 60})
            fh.write(json.dumps({"ticker": r["ticker"], "series": series,
                                 "candles": (d or {}).get("candlesticks", []),
                                 "ok": d is not None}) + "\n")
            fh.flush()
            if i % 100 == 0:
                print(f"{i}/{len(todo)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
