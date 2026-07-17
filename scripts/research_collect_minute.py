#!/usr/bin/env python3
"""Fetch 1-minute candles (with bid/ask OHLC) for targeted market windows.

Input: data/research/mentions/minute_targets.json {ticker: [start_ts, end_ts]}
Output: data/research/mentions/minute_candles.jsonl (append-only, resumable)
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://api.elections.kalshi.com/trade-api/v2"
OUT = Path(__file__).resolve().parents[1] / "data" / "research" / "mentions"
_last = [0.0]


def get(path: str, params: dict, tries: int = 8):
    url = BASE + path + "?" + urllib.parse.urlencode(params)
    for i in range(tries):
        wait = 0.75 - (time.time() - _last[0])
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
            time.sleep(min(60, 2 ** i * 2))
    return None


def main() -> None:
    targets = json.load((OUT / "minute_targets.json").open())
    out_f = OUT / "minute_candles.jsonl"
    done = set()
    if out_f.exists():
        for line in out_f.open():
            try:
                done.add(json.loads(line)["ticker"])
            except Exception:  # noqa: BLE001
                pass
    todo = [(t, w) for t, w in targets.items() if t not in done]
    print(f"{len(todo)} minute-candle fetches ({len(done)} done)", flush=True)
    with out_f.open("a") as fh:
        for i, (tk, (lo, hi)) in enumerate(todo):
            series = tk.split("-")[0]
            d = get(f"/series/{series}/markets/{tk}/candlesticks",
                    {"start_ts": lo - 600, "end_ts": hi + 600,
                     "period_interval": 1})
            fh.write(json.dumps({"ticker": tk,
                                 "candles": (d or {}).get("candlesticks", []),
                                 "ok": d is not None}) + "\n")
            fh.flush()
            if i % 100 == 0:
                print(f"{i}/{len(todo)}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
