#!/usr/bin/env python3
"""Collect the settled Kalshi mention-market dataset (public API, no auth).

Phase 1: paginate ALL settled events (nested markets) and keep the mention
family: earnings-call mentions, press-briefing mentions, "will X say Y" etc.
Phase 2: per settled mention market, pull hourly candlesticks over its life.
Phase 3: per market with volume, pull the trade tape.

Self-clocked (~1 req/s, exponential backoff on 429). Idempotent: JSONL
outputs are append-only with a done-set, safe to re-run.

Output dir: data/research/mentions/
  events.jsonl        one line per mention event (with nested markets)
  candles.jsonl       one line per market: {ticker, series, candles: [...]}
  trades.jsonl        one line per market: {ticker, trades: [...]}
  progress.txt        heartbeat
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://api.elections.kalshi.com/trade-api/v2"
OUT = Path(__file__).resolve().parents[1] / "data" / "research" / "mentions"
OUT.mkdir(parents=True, exist_ok=True)

# Anonymous public-data access only; self-clocked under the public rate limit.
_MIN_INTERVAL = 0.75

MENTION_PAT = re.compile(
    r"(mention|what will .* say|say during|says? .* during|how many times will .* say"
    r"|press briefing\?|nickname)", re.I)
SERIES_PAT = re.compile(
    r"^KX(EARNINGSMENTION|SECPRESSMENTION|TRUMPSAY|MENTION|SAY)", re.I)

_last_req = [0.0]


def get(path: str, params: dict | None = None, tries: int = 8):
    qs = ("?" + urllib.parse.urlencode(params)) if params else ""
    url = BASE + path + qs
    for i in range(tries):
        wait = _MIN_INTERVAL - (time.time() - _last_req[0])
        if wait > 0:
            time.sleep(wait)
        _last_req[0] = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "hedge-research/0.1"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            code = getattr(e, "code", None)
            if code == 404:
                return None
            time.sleep(min(60, 2 ** i * 2))
    print(f"GIVING UP on {url}", flush=True)
    return None


def heartbeat(msg: str) -> None:
    (OUT / "progress.txt").write_text(f"{time.strftime('%H:%M:%S')} {msg}\n")
    print(msg, flush=True)


def is_mention_event(ev: dict) -> bool:
    title = ev.get("title", "") or ""
    et = ev.get("event_ticker", "") or ""
    return bool(SERIES_PAT.match(et) or MENTION_PAT.search(title))


def phase1_events() -> list[dict]:
    out_f = OUT / "events.jsonl"
    done_cursor_f = OUT / "events.cursor"
    seen: set[str] = set()
    events: list[dict] = []
    if out_f.exists():
        for line in out_f.open():
            ev = json.loads(line)
            seen.add(ev["event_ticker"])
            events.append(ev)
    cursor = done_cursor_f.read_text().strip() if done_cursor_f.exists() else ""
    if cursor == "DONE":
        heartbeat(f"phase1 already complete: {len(events)} mention events")
        return events
    page = 0
    with out_f.open("a") as fh:
        while True:
            params = {"limit": 200, "status": "settled", "with_nested_markets": "true"}
            if cursor:
                params["cursor"] = cursor
            d = get("/events", params)
            if d is None:
                break
            evs = d.get("events", [])
            for ev in evs:
                if is_mention_event(ev) and ev["event_ticker"] not in seen:
                    seen.add(ev["event_ticker"])
                    events.append(ev)
                    fh.write(json.dumps(ev) + "\n")
            fh.flush()
            cursor = d.get("cursor", "")
            page += 1
            if page % 10 == 0:
                heartbeat(f"phase1 page {page}, mention events so far: {len(events)}")
            done_cursor_f.write_text(cursor or "DONE")
            if not cursor or not evs:
                break
    done_cursor_f.write_text("DONE")
    heartbeat(f"phase1 complete: {len(events)} mention events")
    return events


def iter_markets(events: list[dict]):
    for ev in events:
        for m in ev.get("markets") or []:
            yield ev, m


def phase2_candles(events: list[dict]) -> None:
    out_f = OUT / "candles.jsonl"
    done: set[str] = set()
    if out_f.exists():
        for line in out_f.open():
            try:
                done.add(json.loads(line)["ticker"])
            except Exception:  # noqa: BLE001
                pass
    todo = [(ev, m) for ev, m in iter_markets(events) if m["ticker"] not in done]

    def prio(item) -> tuple:
        ev, m = item
        st = ev.get("series_ticker") or m["ticker"].split("-")[0]
        fam = 0 if ("EARNINGSMENTION" in st or "PRESSMENTION" in st) else 1
        return (fam, m.get("close_time") or "")

    # Earnings/briefing families first (small, model-rich), then everything
    # else newest-first so partial data stays analysis-ready.
    todo.sort(key=lambda it: (prio(it)[0], ), )
    todo = (sorted([t for t in todo if prio(t)[0] == 0],
                   key=lambda it: it[1].get("close_time") or "", reverse=True)
            + sorted([t for t in todo if prio(t)[0] == 1],
                     key=lambda it: it[1].get("close_time") or "", reverse=True))
    heartbeat(f"phase2: {len(todo)} markets to fetch candles for ({len(done)} done)")
    with out_f.open("a") as fh:
        for i, (ev, m) in enumerate(todo):
            series = ev.get("series_ticker") or m["ticker"].split("-")[0]
            # open_time/close_time are RFC3339
            def ts(key: str, default: int) -> int:
                v = m.get(key)
                if not v:
                    return default
                try:
                    return int(time.mktime(time.strptime(v[:19], "%Y-%m-%dT%H:%M:%S")))
                except Exception:  # noqa: BLE001
                    return default
            start = ts("open_time", 0)
            end = ts("close_time", int(time.time()))
            d = get(f"/series/{series}/markets/{m['ticker']}/candlesticks",
                    {"start_ts": max(0, start - 3600), "end_ts": end + 3600,
                     "period_interval": 60})
            rec = {"ticker": m["ticker"], "series": series,
                   "candles": (d or {}).get("candlesticks", []),
                   "ok": d is not None}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if i % 50 == 0:
                heartbeat(f"phase2: {i}/{len(todo)} candle fetches")
    heartbeat("phase2 complete")


def phase3_trades(events: list[dict]) -> None:
    out_f = OUT / "trades.jsonl"
    done: set[str] = set()
    if out_f.exists():
        for line in out_f.open():
            try:
                done.add(json.loads(line)["ticker"])
            except Exception:  # noqa: BLE001
                pass
    def vol(m: dict) -> float:
        v = m.get("volume_fp") or m.get("volume") or 0
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0
    todo = [m for _, m in iter_markets(events)
            if m["ticker"] not in done and vol(m) > 0]
    todo.sort(key=lambda m: m.get("close_time") or "", reverse=True)
    heartbeat(f"phase3: {len(todo)} traded markets to fetch tapes for")
    with out_f.open("a") as fh:
        for i, m in enumerate(todo):
            trades: list[dict] = []
            cursor = ""
            while True:
                params = {"ticker": m["ticker"], "limit": 1000}
                if cursor:
                    params["cursor"] = cursor
                d = get("/markets/trades", params)
                if d is None:
                    break
                trades.extend(d.get("trades", []))
                cursor = d.get("cursor", "")
                if not cursor:
                    break
            fh.write(json.dumps({"ticker": m["ticker"], "trades": trades}) + "\n")
            fh.flush()
            if i % 50 == 0:
                heartbeat(f"phase3: {i}/{len(todo)} tapes")
    heartbeat("ALL PHASES COMPLETE")


if __name__ == "__main__":
    evs = phase1_events()
    only = sys.argv[1] if len(sys.argv) > 1 else "all"
    if only in ("all", "candles"):
        phase2_candles(evs)
    if only in ("all", "trades"):
        phase3_trades(evs)
