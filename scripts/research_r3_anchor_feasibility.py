#!/usr/bin/env python3
"""Round-3 audit: live-anchor feasibility for the late-event taker-NO rule.

The rule triggers 60-180 min before EVENT END. Round-2 killed it because the
anchor used the ex-post end (max close_time). This script measures, per event
family, how accurately the true end could have been estimated at decision
time T = true_end - 60 min using only live-knowable external information.

Subcommands:
  parse     events.jsonl -> per-event table with close-cluster structure
            (data/research/mentions/r3/events_table.csv)
  mlb       match KXMLBMENTION events to MLB StatsAPI games, pull start /
            end / per-inning wallclock timeline (cached JSON per game)
  mlb-eval  validation of Kalshi end vs real end + inning-conditional
            remaining-time model + live-rule accuracy at T
  tv        per-series duration / end-TOD dispersion for scheduled TV
            families from Kalshi data alone (with honest proxy caveats)
  earnings  KXEARNINGSMENTION*: call start proxy (settlement-backfilled
            occurrence) + duration distribution + schedule-anchor accuracy

Ground-truth hierarchy (be honest about it):
  MLB: StatsAPI actual game end (authoritative).
  Everything else: Kalshi "bulk close" = largest same-second close batch of
  the event -- validated against MLB truth first; treated as noisy.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/research/mentions"
R3 = DATA / "r3"
R3.mkdir(exist_ok=True)

MONTHS = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}
DATE_RE = re.compile(r"-(\d{2})([A-Z]{3})(\d{2})")

FAMILIES = ("KXMLBMENTION", "KXWCMENTION", "KXNBAMENTION", "KXNHLMENTION",
            "KXHEARINGMENTION", "KXEARNINGSMENTION", "KXLOVEISLMENTION",
            "KXLATENIGHTMENTION", "KXFIGHTMENTION")


def fam_of(event_ticker: str) -> str | None:
    pre = event_ticker.split("-")[0]
    if pre.startswith("KXEARNINGSMENTION"):
        return "KXEARNINGSMENTION"
    return pre if pre in FAMILIES else None


def ticker_date(event_ticker: str):
    m = DATE_RE.search(event_ticker)
    if not m:
        return None, ""
    yy, mon, dd = int(m.group(1)), MONTHS.get(m.group(2)), int(m.group(3))
    if not mon:
        return None, ""
    suffix = event_ticker[m.end():]
    return pd.Timestamp(2000 + yy, mon, dd), suffix


def parse_events():
    """events.jsonl -> per-event summary. Dedupe by event_ticker (keep the
    newest last_updated_ts). Close clusters: group close_times within 120 s."""
    best = {}
    with open(DATA / "events.jsonl") as f:
        for line in f:
            e = json.loads(line)
            et = e.get("event_ticker")
            if not et or fam_of(et) is None or not e.get("markets"):
                continue
            lu = e.get("last_updated_ts", "")
            if et not in best or lu > best[et][0]:
                best[et] = (lu, e)
    rows = []
    for et, (_, e) in best.items():
        mk = e["markets"]
        ct = sorted(pd.Timestamp(m["close_time"]) for m in mk
                    if m.get("close_time"))
        if not ct:
            continue
        # cluster close times: new cluster when gap > 120 s
        clusters = []  # (start_ts, count)
        cur = [ct[0]]
        for t in ct[1:]:
            if (t - cur[-1]).total_seconds() > 120:
                clusters.append(cur)
                cur = []
            cur.append(t)
        clusters.append(cur)
        bulk = max(clusters, key=len)  # largest batch
        occ = [pd.Timestamp(m["occurrence_datetime"]) for m in mk
               if m.get("occurrence_datetime")]
        # "scheduled-looking" occurrence: :00 seconds, minute % 5 == 0, and
        # NOT within 90 s after any close batch (settlement echo is close+59s)
        sched_occ = []
        for o in occ:
            if o.second != 0 or o.minute % 5 != 0:
                continue
            if any(0 <= (o - c[0]).total_seconds() <= 180 for c in clusters):
                continue
            sched_occ.append(o)
        d, suffix = ticker_date(et)
        rows.append(dict(
            event_ticker=et, family=fam_of(et),
            series=et.split("-")[0],
            tdate=d, suffix=suffix,
            title=e.get("title", ""), sub_title=e.get("sub_title", ""),
            n_markets=len(mk),
            n_yes=sum(m.get("result") == "yes" for m in mk),
            n_clusters=len(clusters),
            min_close=ct[0], max_close=ct[-1],
            bulk_close=bulk[0], bulk_n=len(bulk),
            bulk_is_first=bulk[0] == clusters[0][0],
            first_cluster_n=len(clusters[0]),
            first_cluster_close=clusters[0][0],
            min_open=min(pd.Timestamp(m["open_time"]) for m in mk
                         if m.get("open_time")),
            sched_occ=min(sched_occ).isoformat() if sched_occ else "",
            n_sched_occ=len(sched_occ),
        ))
    df = pd.DataFrame(rows).sort_values(["family", "tdate"])
    df.to_csv(R3 / "events_table.csv", index=False)
    print(df.groupby("family").agg(
        n=("event_ticker", "count"),
        with_sched_occ=("sched_occ", lambda s: (s != "").sum()),
        med_markets=("n_markets", "median")))
    return df


# ---------------------------------------------------------------- MLB
TEAM_ABBR = {}  # filled from schedule responses


def http_json(url, tries=3):
    import urllib.request
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "hedge-research/0.1 (contact: research)"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode())
        except Exception as ex:  # noqa: BLE001
            if i == tries - 1:
                print("FAIL", url, ex)
                return None
            time.sleep(2.0 * (i + 1))


def mlb_collect():
    ev = pd.read_csv(R3 / "events_table.csv", parse_dates=["tdate"])
    ev = ev[ev.family == "KXMLBMENTION"].copy()
    cache = R3 / "mlb"
    cache.mkdir(exist_ok=True)
    # 1. schedules per unique date (ticker date is the US game date)
    sched = {}
    for d in sorted(ev.tdate.dropna().unique()):
        d = pd.Timestamp(d)
        fp = cache / f"sched_{d.date()}.json"
        if fp.exists():
            sched[d] = json.loads(fp.read_text())
            continue
        js = http_json(
            "https://statsapi.mlb.com/api/v1/schedule?sportId=1"
            f"&date={d.date()}&hydrate=team")
        time.sleep(0.6)
        if js is None:
            continue
        fp.write_text(json.dumps(js))
        sched[d] = js
    # 2. match by team abbreviations embedded in ticker suffix (AWAYHOME)
    matches = []
    for _, r in ev.iterrows():
        d = r.tdate
        if pd.isna(d) or d not in sched:
            matches.append(None)
            continue
        suf = r.suffix
        games = [g for day in sched[d].get("dates", [])
                 for g in day.get("games", [])]
        hit = None
        for g in games:
            away = g["teams"]["away"]["team"].get("abbreviation", "")
            home = g["teams"]["home"]["team"].get("abbreviation", "")
            if suf == f"{away}{home}" or suf == f"{home}{away}":
                hit = g
                break
        if hit is None and suf == "ALNL":  # All-Star game
            for g in games:
                if g.get("gameType") == "A":
                    hit = g
                    break
        matches.append(hit["gamePk"] if hit else None)
    ev["gamePk"] = matches
    print("matched", ev.gamePk.notna().sum(), "of", len(ev))
    # 3. per-game timeline via feed/live (trimmed by fields param)
    for pk in ev.gamePk.dropna().astype(int).unique():
        fp = cache / f"game_{pk}.json"
        if fp.exists():
            continue
        js = http_json(
            f"https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live?"
            "fields=gameData,datetime,dateTime,officialDate,gameInfo,"
            "firstPitch,gameDurationMinutes,delayDurationMinutes,status,"
            "detailedState,liveData,plays,allPlays,about,startTime,endTime,"
            "inning,halfInning,atBatIndex,linescore,currentInning")
        time.sleep(0.7)
        if js is None:
            continue
        fp.write_text(json.dumps(js))
    ev.to_csv(R3 / "mlb_matched.csv", index=False)
    print("game files:", len(list(cache.glob("game_*.json"))))


def mlb_load_games():
    ev = pd.read_csv(R3 / "mlb_matched.csv", parse_dates=[
        "tdate", "min_close", "max_close", "bulk_close",
        "first_cluster_close", "min_open"])
    ev = ev.dropna(subset=["gamePk"]).copy()
    ev["gamePk"] = ev.gamePk.astype(int)
    games = {}
    for pk in ev.gamePk.unique():
        fp = R3 / "mlb" / f"game_{pk}.json"
        if not fp.exists():
            continue
        js = json.loads(fp.read_text())
        gd = js.get("gameData", {})
        dt = gd.get("datetime", {})
        gi = gd.get("gameInfo", {})
        plays = js.get("liveData", {}).get("plays", {}).get("allPlays", [])
        # wallclock timeline: (inning, half, startTime, endTime)
        tl = []
        for p in plays:
            ab = p.get("about", {})
            st, en = ab.get("startTime"), ab.get("endTime")
            if st and en:
                tl.append((ab.get("inning"), ab.get("halfInning"),
                           pd.Timestamp(st), pd.Timestamp(en)))
        start = pd.Timestamp(dt["dateTime"]) if dt.get("dateTime") else None
        first_pitch = (pd.Timestamp(gi["firstPitch"])
                       if gi.get("firstPitch") else None)
        dur = gi.get("gameDurationMinutes")
        delay = gi.get("delayDurationMinutes", 0) or 0
        end = None
        if tl:
            end = max(t[3] for t in tl)
        elif first_pitch is not None and dur:
            end = first_pitch + pd.Timedelta(minutes=dur + delay)
        games[pk] = dict(start=start, first_pitch=first_pitch,
                         dur_min=dur, delay_min=delay, end=end, timeline=tl,
                         state=gd.get("status", {}).get("detailedState"))
    return ev, games


def mlb_eval():
    ev, games = mlb_load_games()
    ev = ev[ev.gamePk.isin(games)].copy()
    g = ev.gamePk.map(games)
    ev["g_start"] = [games[pk]["start"] for pk in ev.gamePk]
    ev["g_first_pitch"] = [games[pk]["first_pitch"] for pk in ev.gamePk]
    ev["g_end"] = [games[pk]["end"] for pk in ev.gamePk]
    ev["g_dur"] = [games[pk]["dur_min"] for pk in ev.gamePk]
    ev = ev.dropna(subset=["g_start", "g_end"]).copy()
    ev["g_start"] = pd.to_datetime(ev.g_start, utc=True)
    ev["g_end"] = pd.to_datetime(ev.g_end, utc=True)

    def mins(a, b):
        return (a - b).dt.total_seconds() / 60

    print(f"\n== MLB ground-truth validation (n={len(ev)} matched games) ==")
    for col in ("max_close", "bulk_close", "first_cluster_close",
                "min_close"):
        off = mins(pd.to_datetime(ev[col], utc=True), ev.g_end)
        print(f"  {col:20s} - real_end  min: "
              f"p10={off.quantile(.1):7.0f} p25={off.quantile(.25):7.0f} "
              f"med={off.median():7.0f} p75={off.quantile(.75):7.0f} "
              f"p90={off.quantile(.9):7.0f}  |off|<=30m: "
              f"{(off.abs() <= 30).mean():.2f}  <=60m: "
              f"{(off.abs() <= 60).mean():.2f}")

    # duration distribution
    dur = ev.g_dur.dropna()
    print(f"\n  game duration (first pitch->last out), n={len(dur)}: "
          f"med={dur.median():.0f}m IQR=[{dur.quantile(.25):.0f},"
          f"{dur.quantile(.75):.0f}] sd={dur.std():.0f}")

    # -------- inning-conditional remaining time (live observer's info) ----
    # state at time t: (inning, half) of the play in progress
    recs = []
    for pk, gm in games.items():
        tl = gm["timeline"]
        if not tl or gm["end"] is None:
            continue
        end = gm["end"]
        for inn, half, st, en in tl:
            recs.append(dict(pk=pk, inning=inn, half=half, t=st,
                             remain=(end - st).total_seconds() / 60))
    tlf = pd.DataFrame(recs)
    tlf["state"] = tlf.inning.astype(str) + "-" + tlf.half.str[0]
    print(f"\n  play-level timeline rows: {len(tlf)} over "
          f"{tlf.pk.nunique()} games")
    med_remain = tlf.groupby("state").remain.agg(["median", "count",
                                                  lambda s: s.quantile(.25),
                                                  lambda s: s.quantile(.75)])
    med_remain.columns = ["med", "n", "q25", "q75"]
    print("\n  remaining minutes by (inning-half) at play start:")
    print(med_remain.sort_values("med", ascending=False).to_string())

    # ------- live rule: est_end = now + med_remain(state), leave-one-out --
    # evaluated at T = true_end - 60 and on a 30-min grid through the game
    def state_at(pk, t):
        tl = games[pk]["timeline"]
        prev = None
        for inn, half, st, en in tl:
            if st <= t:
                prev = (inn, half)
            else:
                break
        return prev

    # global median remaining by state (pooled; n large enough that LOO
    # changes nothing material -- noted in report)
    med_map = tlf.groupby("state").remain.median().to_dict()

    rows = []
    for pk, gm in games.items():
        if not gm["timeline"] or gm["end"] is None:
            continue
        end, start = gm["end"], gm["timeline"][0][2]
        for label, t in [("T-60", end - pd.Timedelta(minutes=60)),
                         ("T-90", end - pd.Timedelta(minutes=90)),
                         ("T-120", end - pd.Timedelta(minutes=120))]:
            if t < start:
                rows.append(dict(pk=pk, at=label, err=np.nan,
                                 state="pregame"))
                continue
            stt = state_at(pk, t)
            if stt is None:
                continue
            key = f"{stt[0]}-{stt[1][0]}"
            if key not in med_map:
                continue
            est = t + pd.Timedelta(minutes=med_map[key])
            rows.append(dict(pk=pk, at=label,
                             err=(est - end).total_seconds() / 60,
                             state=key))
    lr = pd.DataFrame(rows)
    print("\n  live rule est_end = now + median_remaining(inning-half):")
    for label, gpd in lr.groupby("at"):
        e = gpd.err.dropna()
        print(f"  {label}: n={len(e)} med|err|={e.abs().median():5.1f}m "
              f"within±30={(e.abs() <= 30).mean():.2f} "
              f"within±60={(e.abs() <= 60).mean():.2f} "
              f"(pregame at T: {(gpd.state == 'pregame').mean():.2f})")

    # ------- schedule-only rule: est_end = sched_start + median duration --
    med_dur = dur.median()
    est = ev.g_start + pd.Timedelta(minutes=med_dur)
    err = mins(est, ev.g_end)
    print(f"\n  schedule rule est_end = sched_start + {med_dur:.0f}m: "
          f"med|err|={err.abs().median():.1f}m "
          f"within±30={(err.abs() <= 30).mean():.2f} "
          f"within±60={(err.abs() <= 60).mean():.2f}")
    ev.to_csv(R3 / "mlb_eval.csv", index=False)
    tlf.to_csv(R3 / "mlb_timeline.csv", index=False)


def mlb_trigger():
    """Task 1c: 30-min-grid live trigger. Enter at the first grid time where
    est_remaining (inning-half median) <= 90 min; report actual remaining."""
    _, games = mlb_load_games()
    recs = []
    for pk, gm in games.items():
        tl = gm["timeline"]
        if not tl or gm["end"] is None:
            continue
        end = gm["end"]
        for inn, half, st, en in tl:
            recs.append((f"{inn}-{half[0]}", (end - st).total_seconds() / 60))
    med_map = (pd.DataFrame(recs, columns=["state", "remain"])
               .groupby("state").remain.median().to_dict())

    def state_at(pk, t):
        prev = None
        for inn, half, st, en in games[pk]["timeline"]:
            if st <= t:
                prev = f"{inn}-{half[0]}"
            else:
                break
        return prev

    rows = []
    for pk, gm in games.items():
        tl = gm["timeline"]
        if not tl or gm["end"] is None:
            continue
        t, end = tl[0][2], gm["end"]
        while t < end + pd.Timedelta(hours=2):
            st = state_at(pk, t)
            if st and med_map.get(st, 999) <= 90:
                rows.append((end - t).total_seconds() / 60)
                break
            t += pd.Timedelta(minutes=30)
    tr = pd.Series(rows)
    print(f"MLB grid trigger (est_remaining<=90m): n={len(tr)} "
          f"actual remaining med={tr.median():.0f} "
          f"IQR=[{tr.quantile(.25):.0f},{tr.quantile(.75):.0f}] "
          f"in [60,180]: {tr.between(60, 180).mean():.2f} "
          f"90±60: {((tr - 90).abs() <= 60).mean():.2f}")


def wc_eval():
    """WC: end-estimate accuracy from r3/wc_matches.csv (ESPN wallclock
    final-whistle times collected by the research agent)."""
    d = pd.read_csv(R3 / "wc_matches.csv")
    d["dur"] = (pd.to_datetime(d.est_end_utc, utc=True)
                - pd.to_datetime(d.kickoff_utc, utc=True)
                ).dt.total_seconds() / 60
    err1 = d.dur - d.dur.median()
    print(f"WC n={len(d)} dur med={d.dur.median():.0f} "
          f"IQR=[{d.dur.quantile(.25):.0f},{d.dur.quantile(.75):.0f}] "
          f"ET={int(d.went_et.sum())} pens={int(d.went_pens.sum())}")
    print(f"  sched-only (kickoff+med): med|err|={err1.abs().median():.1f} "
          f"±30={(err1.abs() <= 30).mean():.2f} "
          f"±60={(err1.abs() <= 60).mean():.2f}")
    durs = d.dur.to_numpy()
    errs = []
    for i, dur in enumerate(durs):
        e = dur - 60
        others = np.delete(durs, i)
        cond = others[others > e] - e
        errs.append((np.median(cond) if len(cond) else 60) - 60)
    e2 = pd.Series(errs)
    print(f"  live elapsed-cond at T-60 (LOO): med|err|={e2.abs().median():.1f} "
          f"±30={(e2.abs() <= 30).mean():.2f} ±60={(e2.abs() <= 60).mean():.2f}")


def nbanhl_eval():
    """NBA/NHL: period-conditional live rule from cached ESPN play-by-play."""
    import glob
    games = {}
    for fp in glob.glob(str(R3 / "espn_cache/plays_*_p1_l1000.json")):
        m = re.search(r"plays_(NBA|NHL)_(\d+)_", fp)
        js = json.loads(Path(fp).read_text())
        tl = [((p.get("period") or {}).get("number"),
               pd.Timestamp(p["wallclock"]))
              for p in js.get("items", [])
              if p.get("wallclock") and (p.get("period") or {}).get("number")]
        if tl:
            games[(m.group(1), m.group(2))] = sorted(tl, key=lambda x: x[1])
    recs = [dict(lg=lg, per=per, remain=(tl[-1][1] - t).total_seconds() / 60)
            for (lg, _), tl in games.items() for per, t in tl]
    med_map = (pd.DataFrame(recs).groupby(["lg", "per"]).remain.median()
               .to_dict())
    rows = []
    for (lg, gid), tl in games.items():
        end, t0 = tl[-1][1], tl[0][1]
        for lab, T in [("T-60", end - pd.Timedelta(minutes=60)),
                       ("T-90", end - pd.Timedelta(minutes=90))]:
            if T < t0:
                continue
            per = max(p for p, t in tl if t <= T)
            est = T + pd.Timedelta(minutes=med_map[(lg, per)])
            rows.append(dict(lg=lg, at=lab,
                             err=(est - end).total_seconds() / 60))
    lr = pd.DataFrame(rows)
    for (lg, lab), g in lr.groupby(["lg", "at"]):
        e = g.err
        print(f"{lg} {lab}: n={len(e)} med|err|={e.abs().median():.1f} "
              f"±30={(e.abs() <= 30).mean():.2f} "
              f"±60={(e.abs() <= 60).mean():.2f}")


def earnings_eval():
    """Earnings: duration distribution from matched transcripts
    (words / 150 wpm) + schedule-anchor accuracy."""
    ee = pd.read_csv(R3 / "events_table.csv", parse_dates=["tdate"])
    ee = ee[ee.family == "KXEARNINGSMENTION"].copy()
    ee["symbol"] = ee.series.str.replace("KXEARNINGSMENTION", "", regex=False)
    tr = pd.read_parquet(DATA / "transcripts/earnings.parquet")
    tr = tr[~tr.is_stub].copy()
    tr["tdate"] = pd.to_datetime(tr.transcript_date)
    durs = []
    for _, r in ee.iterrows():
        cand = tr[(tr.symbol == r.symbol) & tr.tdate.notna()
                  & ((tr.tdate - r.tdate).dt.days.abs() <= 3)]
        if len(cand):
            durs.append(len(cand.iloc[0].full_text.split()) / 150.0)
    x = pd.Series(durs)
    err = x - x.median()
    print(f"Earnings n={len(x)} est dur med={x.median():.0f}m "
          f"IQR=[{x.quantile(.25):.0f},{x.quantile(.75):.0f}] "
          f"p10/p90=[{x.quantile(.1):.0f},{x.quantile(.9):.0f}]")
    print(f"  sched anchor (start+med): med|err|={err.abs().median():.1f} "
          f"±30={(err.abs() <= 30).mean():.2f} "
          f"±60={(err.abs() <= 60).mean():.2f}")


# ---------------------------------------------------------------- TV etc.
def tv_report():
    ev = pd.read_csv(R3 / "events_table.csv", parse_dates=[
        "min_close", "max_close", "bulk_close", "first_cluster_close",
        "min_open", "tdate"])
    for fam in ("KXLOVEISLMENTION", "KXLATENIGHTMENTION", "KXFIGHTMENTION",
                "KXWCMENTION", "KXNBAMENTION", "KXNHLMENTION",
                "KXHEARINGMENTION"):
        f = ev[ev.family == fam].copy()
        if not len(f):
            continue
        # fixed EDT offset (all events May-Jul 2026); host lacks tzdata
        f["bulk_et"] = (pd.to_datetime(f.bulk_close, utc=True)
                        + pd.Timedelta(hours=-4))
        f["tod_h"] = f.bulk_et.dt.hour + f.bulk_et.dt.minute / 60
        # circular-safe: shift day boundary to 6am ET
        f["tod_s"] = (f.tod_h - 6) % 24
        print(f"\n== {fam} (n={len(f)}) bulk-close ET time-of-day ==")
        q = f.tod_s.quantile([.1, .25, .5, .75, .9]) + 6
        print("  TOD(ET,h) p10/p25/med/p75/p90:",
              " ".join(f"{v % 24:.2f}" for v in q))
        print(f"  IQR = {(f.tod_s.quantile(.75) - f.tod_s.quantile(.25)) * 60:.0f} min")
        if fam == "KXLATENIGHTMENTION":
            f["show"] = f.title.str.extract(r"What will (.*?) say")[0]
            for show, g in f.groupby("show"):
                gq = (g.tod_s.quantile(.75) - g.tod_s.quantile(.25)) * 60
                print(f"    {show}: n={len(g)} med TOD "
                      f"{(g.tod_s.median() + 6) % 24:.2f} ET IQR={gq:.0f}m")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["parse", "mlb", "mlb-eval", "mlb-trigger",
                                    "wc-eval", "nbanhl-eval", "earnings",
                                    "tv"])
    a = ap.parse_args()
    fn = {"parse": parse_events, "mlb": mlb_collect, "mlb-eval": mlb_eval,
          "mlb-trigger": mlb_trigger, "wc-eval": wc_eval,
          "nbanhl-eval": nbanhl_eval, "earnings": earnings_eval,
          "tv": tv_report}
    fn[a.cmd]()


if __name__ == "__main__":
    main()
