#!/usr/bin/env python
"""Collect ground-truth game timing for NBA/NHL mention-market events (r3).

For each Kalshi mention event, find the real game on ESPN, get scheduled start,
actual first-play and last-play wallclock (game end), and OT info.
"""
from __future__ import annotations

import csv
import json
import re
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path("/home/mcortese/fun/hedge/.claude/worktrees/mentions-ml")
R3 = ROOT / "data/research/mentions/r3"
CACHE = R3 / "espn_cache"
CACHE.mkdir(parents=True, exist_ok=True)

ET = ZoneInfo("America/New_York")
UA = {"User-Agent": "Mozilla/5.0 (research; game-timing)"}
_last_req = [0.0]


def fetch_json(url: str, cache_name: str) -> dict:
    path = CACHE / cache_name
    if path.exists():
        return json.loads(path.read_text())
    wait = 0.65 - (time.time() - _last_req[0])
    if wait > 0:
        time.sleep(wait)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read().decode())
    _last_req[0] = time.time()
    path.write_text(json.dumps(data))
    return data


# Kalshi 3-letter -> ESPN abbreviation (only where they differ / to be safe)
NBA_MAP = {"SAS": "SA", "NYK": "NY", "PHI": "PHI", "MIN": "MIN", "OKC": "OKC",
           "LAL": "LAL", "DET": "DET", "CLE": "CLE"}
NHL_MAP = {"COL": "COL", "MIN": "MIN", "BUF": "BUF", "MTL": "MTL", "VGK": "VGK",
           "ANA": "ANA", "CAR": "CAR"}


def parse_iso(s: str) -> datetime | None:
    if not s or not s.strip():
        return None
    s = s.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def teams_from_ticker(ticker: str) -> tuple[str, str]:
    # e.g. KXNBAMENTION-26JUN13NYKSAS or KXNBAMENTION-26CLEDETR2
    tail = ticker.split("-")[1]
    m = re.match(r"^26[A-Z]{3}\d{2}([A-Z]{3})([A-Z]{3})$", tail)
    if m:
        return m.group(1), m.group(2)
    m = re.match(r"^26([A-Z]{3})([A-Z]{3})R\d$", tail)
    if m:
        return m.group(1), m.group(2)
    raise ValueError(f"cannot parse teams from {ticker}")


def scoreboard(league: str, yyyymmdd: str) -> dict:
    sport = "basketball/nba" if league == "NBA" else "hockey/nhl"
    url = f"http://site.api.espn.com/apis/site/v2/sports/{sport}/scoreboard?dates={yyyymmdd}"
    return fetch_json(url, f"sb_{league}_{yyyymmdd}.json")


def plays_page(league: str, eid: str, page: int, limit: int = 1000) -> dict:
    sport = "basketball/leagues/nba" if league == "NBA" else "hockey/leagues/nhl"
    url = (f"http://sports.core.api.espn.com/v2/sports/{sport}/events/{eid}"
           f"/competitions/{eid}/plays?limit={limit}&page={page}")
    return fetch_json(url, f"plays_{league}_{eid}_p{page}_l{limit}.json")


def find_game(league: str, game_date: str, t1: str, t2: str):
    """game_date: YYYY-MM-DD US date. Returns (espn_event, note) or (None, note)."""
    amap = NBA_MAP if league == "NBA" else NHL_MAP
    want = {amap.get(t1, t1), amap.get(t2, t2)}
    d = datetime.strptime(game_date, "%Y-%m-%d")
    tried = []
    for off in (0, -1, 1):
        ymd = (d + timedelta(days=off)).strftime("%Y%m%d")
        sb = scoreboard(league, ymd)
        for ev in sb.get("events", []):
            comp = ev["competitions"][0]
            abbrs = {c["team"]["abbreviation"] for c in comp["competitors"]}
            if abbrs == want:
                note = "" if off == 0 else f"matched at date offset {off:+d}d"
                return ev, note
        tried.append(ymd)
    return None, f"no match on dates {tried}"


def game_timing(league: str, ev: dict):
    """Return (sched_start_utc, first_play_utc, last_play_utc, n_ot, status_note)."""
    comp = ev["competitions"][0]
    sched = parse_iso(comp.get("date") or ev.get("date"))
    status = ev.get("status", {})
    period = status.get("period", 0)
    stype = status.get("type", {}).get("name", "")
    reg = 4 if league == "NBA" else 3
    n_ot = max(0, period - reg)
    eid = ev["id"]

    first_pg = plays_page(league, eid, 1)
    n_pages = first_pg.get("pageCount", 1)
    items = first_pg.get("items", [])
    first_wc = None
    for it in items:
        wc = it.get("wallclock")
        if wc:
            first_wc = parse_iso(wc)
            break
    last_pg = first_pg if n_pages <= 1 else plays_page(league, eid, n_pages)
    last_wc = None
    for it in reversed(last_pg.get("items", [])):
        wc = it.get("wallclock")
        if wc:
            last_wc = parse_iso(wc)
            break
    return sched, first_wc, last_wc, n_ot, stype


def load_input(path: Path, league: str):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            r["league"] = league
            rows.append(r)
    return rows


def main():
    events = load_input(R3 / "nba_input.csv", "NBA") + load_input(R3 / "nhl_input.csv", "NHL")
    out_rows = []
    for r in events:
        tk = r["event_ticker"]
        league = r["league"]
        t1, t2 = teams_from_ticker(tk)
        notes = []
        game_date = (r.get("tdate") or "").strip()
        if not game_date:
            so = parse_iso(r.get("sched_occ", ""))
            if so:
                game_date = (so - timedelta(days=14)).astimezone(ET).strftime("%Y-%m-%d")
                notes.append(f"tdate missing; derived {game_date} from sched_occ-14d")
            else:
                bc = parse_iso(r["bulk_close"])
                game_date = (bc - timedelta(hours=30)).astimezone(ET).strftime("%Y-%m-%d")
                notes.append(f"tdate+sched_occ missing; guessed {game_date} from bulk_close")
        ev, mnote = find_game(league, game_date, t1, t2)
        if mnote:
            notes.append(mnote)
        if ev is None:
            out_rows.append(dict(event_ticker=tk, league=league, game_name="",
                                 sched_start_utc="", actual_first_play_utc="",
                                 actual_end_utc="", duration_min="", n_ot="",
                                 match_confidence="none", notes="; ".join(notes)))
            print(f"[MISS] {tk}: {notes}")
            continue
        try:
            sched, first_wc, last_wc, n_ot, stype = game_timing(league, ev)
        except Exception as e:  # noqa: BLE001
            sched = parse_iso(ev["competitions"][0].get("date"))
            first_wc = last_wc = None
            n_ot = ""
            notes.append(f"plays fetch failed: {e}")
            stype = ""
        name = ev.get("name") or ev.get("shortName", "")
        if stype and stype != "STATUS_FINAL":
            notes.append(f"status={stype}")
        conf = "high"
        if not last_wc:
            conf = "low"
            notes.append("no wallclock on plays")
        elif mnote:
            conf = "medium"
        dur = ""
        if sched and last_wc:
            dur = round((last_wc - sched).total_seconds() / 60.0, 1)
            if not (100 <= float(dur) <= 400):
                notes.append(f"duration {dur}min out of sane range")
                conf = "medium" if conf == "high" else conf
        # sanity: first play should be within ~45min after sched
        if sched and first_wc:
            lag = (first_wc - sched).total_seconds() / 60.0
            if lag < -30 or lag > 90:
                notes.append(f"first play {lag:.0f}min vs sched — suspect wallclock")
                conf = "low"
        out_rows.append(dict(
            event_ticker=tk, league=league, game_name=name,
            sched_start_utc=sched.strftime("%Y-%m-%d %H:%M:%S+00:00") if sched else "",
            actual_first_play_utc=first_wc.strftime("%Y-%m-%d %H:%M:%S+00:00") if first_wc else "",
            actual_end_utc=last_wc.strftime("%Y-%m-%d %H:%M:%S+00:00") if last_wc else "",
            duration_min=dur, n_ot=n_ot, match_confidence=conf,
            notes="; ".join(notes)))
        print(f"[ok] {tk} -> {name} eid={ev['id']} sched={sched} end={last_wc} ot={n_ot} {notes}")

    out = R3 / "nba_nhl_games.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nwrote {out} ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()
