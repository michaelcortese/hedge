"""M3 pre-registered confirmatory test: cross-market clinch clock (CS2).

Implements docs/EDGE_SEARCH.md 'M3 pre-registration' EXACTLY. No tuning.

Interpretations fixed BEFORE seeing any result (documented in report):
- Candle gaps = no book updates that minute -> forward-fill close bid/ask to a
  1-minute grid (book-state persistence), per ticker, from first candle to
  market close_time.
- Side-level bid on a map book at minute t = max(own yes_bid_close,
  1 - mirror yes_ask_close) on the ffilled grid (a bid for S is a bid on S's
  own book or an offer on the mirror book).
- Genuine snap (pre-reg: first minute bid >= 0.97; gate >= 2 consecutive
  minutes bid >= 0.95): first minute t with bid_S(t) >= 0.97 and
  bid_S(t+1min) >= 0.95. Clock = t. If t+1 is beyond the grid, no snap.
- Map decided by the EARLIEST genuine snap on that map (either side) — the
  ex-ante walk; a later opposite snap cannot undo it (no hindsight).
- Format N = max listed map_number rounded UP to the nearest odd (CS2 is
  Bo1/Bo3/Bo5; Kalshi mostly lists maps 1-2 for a Bo3). need = ceil(N/2).
- Fixture excluded (counted) if: no game-book pairing, any listed map event
  has zero candles on BOTH mirrors, or the match market is unsettled.
"""
from __future__ import annotations

import json
import math
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/mcortese/fun/hedge/.claude/worktrees/lol-predictor/lolpred")
from lolpred.backtest.edge_protocol import evaluate_frozen_rule
from lolpred.backtest.kalshi_eval import kalshi_taker_fee

DATA = "/home/mcortese/fun/hedge/.claude/worktrees/lol-predictor/lolpred/data/odds/multi"
SNAP_HI = 0.97
SNAP_SUSTAIN = 0.95
ENTRY_DELAY_MIN = 2
ASK_MAX = 0.97
BID_MIN = 0.85
LIVENESS_WIN_MIN = 10
CONTRA_PX = 0.80
DOUBT_PX = 0.90

norm = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())


def load_series(prefix: str):
    m = pd.read_parquet(f"{DATA}/{prefix}_markets.parquet")
    t = pd.read_parquet(f"{DATA}/{prefix}_trades.parquet")
    c = pd.read_parquet(f"{DATA}/{prefix}_candles_1m.parquet")
    return m, t, c


def ffill_grid(cd: pd.DataFrame, close_time) -> pd.DataFrame:
    """Per-ticker candles -> 1-min grid [first candle, close_time], ffilled."""
    cd = cd.sort_values("end_period_ts").drop_duplicates("end_period_ts", keep="last")
    start = cd.end_period_ts.iloc[0].ceil("min")
    end = min(pd.Timestamp(close_time).floor("min"), cd.end_period_ts.iloc[-1].ceil("min") + pd.Timedelta(minutes=0))
    end = max(end, cd.end_period_ts.iloc[-1].floor("min"))
    end = pd.Timestamp(close_time).floor("min")
    if end < start:
        end = cd.end_period_ts.iloc[-1].floor("min")
    idx = pd.date_range(start, end, freq="1min")
    g = cd.set_index("end_period_ts")[["yes_bid_close", "yes_ask_close"]].reindex(
        idx.union(cd.end_period_ts)
    ).ffill().reindex(idx)
    return g


def run(series_game: str, series_map: str):
    gm, gt, gc = load_series(series_game)
    mm, mt, mc = load_series(series_map)

    gm["slug"] = gm.event_ticker.str.replace(f"{series_game}-", "", regex=False)
    mm["slug"] = mm.event_ticker.str.replace(f"{series_map}-", "", regex=False).str.rsplit("-", n=1).str[0]
    mm["map_number"] = mm.event_ticker.str.rsplit("-", n=1).str[1].astype(int)

    counts = {}
    counts["game_fixtures_total"] = int(gm.slug.nunique())
    counts["map_fixtures_total"] = int(mm.slug.nunique())

    paired = sorted(set(gm.slug) & set(mm.slug))
    counts["paired_fixtures"] = len(paired)
    counts["game_fixtures_no_maps"] = counts["game_fixtures_total"] - len(paired)
    counts["map_fixtures_no_game"] = counts["map_fixtures_total"] - len(paired)

    # team-pair agreement audit
    mp = mm.groupby("slug").yes_sub_title.apply(lambda x: frozenset(norm(v) for v in x))
    gp = gm.groupby("slug").yes_sub_title.apply(lambda x: frozenset(norm(v) for v in x))
    agree = (mp.reindex(paired) == gp.reindex(paired))
    counts["team_pair_mismatch"] = int((~agree).sum())
    paired = [s for s in paired if agree.get(s, False)]

    map_cand_tickers = set(mc.ticker.unique())
    counts["map_candle_ticker_coverage"] = round(len(map_cand_tickers & set(mm.ticker)) / len(set(mm.ticker)), 4)

    gc_by = {k: v for k, v in gc.groupby("ticker")}
    mc_by = {k: v for k, v in mc.groupby("ticker")}
    gt_by = {k: v for k, v in gt.groupby("ticker")}

    def _excl(slug, reason, ask=None, bid=None):
        excl.append((slug, reason, ask, bid))

    def _fmt_misinferred(mmx, need):
        """Diagnostic ONLY (uses outcomes): did a side reach `need` settled map
        wins while a HIGHER-numbered map was still played? -> real format > N."""
        w = {}
        settled = mmx[mmx.result.isin(["yes", "no"])]
        if len(settled) == 0:
            return False
        by_map = settled.sort_values("map_number").groupby("map_number")
        clinch_map = None
        for mapno, grp in by_map:
            for r in grp.itertuples(index=False):
                if r.result == "yes":
                    s = norm(r.yes_sub_title)
                    w[s] = w.get(s, 0) + 1
                    if w[s] >= need and clinch_map is None:
                        clinch_map = mapno
        if clinch_map is None:
            return False
        return bool((settled.map_number > clinch_map).any())

    bets, excl, snaps_all, unclocked = [], [], [], []
    for slug in paired:
        gmx = gm[gm.slug == slug]
        mmx = mm[mm.slug == slug]
        if not gmx.result.isin(["yes", "no"]).all():
            _excl(slug, "match_unsettled"); continue
        N_listed = int(mmx.map_number.max())
        N = N_listed if N_listed % 2 == 1 else N_listed + 1
        need = math.ceil(N / 2)
        fmt_bad = _fmt_misinferred(mmx, need)

        teams = {norm(t): t for t in gmx.yes_sub_title}
        # per map event: build side-level bid series
        decided = []  # (time, side_norm, map_number, was_correct)
        missing_map = False
        fixture_snaps = []
        for ev, grp in mmx.groupby("event_ticker"):
            grids = {}
            for r in grp.itertuples(index=False):
                cd = mc_by.get(r.ticker)
                if cd is None or len(cd) == 0:
                    continue
                grids[norm(r.yes_sub_title)] = (ffill_grid(cd, r.close_time), r)
            if not grids:
                missing_map = True
                continue
            side_rows = list(grp.itertuples(index=False))
            sides = [norm(r.yes_sub_title) for r in side_rows]
            # union grid
            idx = None
            for g, _ in grids.values():
                idx = g.index if idx is None else idx.union(g.index)
            bid = {}
            for s in sides:
                own = grids.get(s)
                b = pd.Series(-1.0, index=idx)
                if own is not None:
                    b = np.maximum(b, own[0]["yes_bid_close"].reindex(idx).fillna(-1))
                other = [x for x in sides if x != s]
                if other and grids.get(other[0]) is not None:
                    b = np.maximum(b, (1.0 - grids[other[0]][0]["yes_ask_close"]).reindex(idx).fillna(-1))
                bid[s] = pd.Series(b, index=idx)
            mapno = int(grp.map_number.iloc[0])
            # map result per side
            res = {norm(r.yes_sub_title): (r.result == "yes") for r in side_rows}
            ev_snaps = []
            for s in sides:
                b = bid[s]
                hi = (b >= SNAP_HI)
                nxt = b.shift(-1)
                contig = (pd.Series(b.index, index=b.index).diff(-1).dt.total_seconds() == -60)
                genuine = hi & (nxt >= SNAP_SUSTAIN) & contig
                if genuine.any():
                    t0 = genuine.idxmax()
                    ev_snaps.append((t0, s))
                    match_row = gmx[gmx.yes_sub_title.map(norm) == s]
                    snaps_all.append({"slug": slug, "event": ev, "map_number": mapno,
                                      "side": s, "t": t0,
                                      "map_won": bool(res.get(s, False)),
                                      "match_won": bool(len(match_row) and (match_row.iloc[0].result == "yes")),
                                      "settled": grp.result.isin(["yes", "no"]).all()})
            if ev_snaps:
                ev_snaps.sort()
                t0, s = ev_snaps[0]
                decided.append((t0, s, mapno, res.get(s)))
                fixture_snaps.extend(ev_snaps)
        if missing_map:
            _excl(slug, "map_book_candles_missing"); continue
        # chronological walk to clinch
        decided.sort()
        score = {}
        clock, clinch_side = None, None
        seen_maps = set()
        walk_used_misdecided_map = False
        for t0, s, mapno, map_res in decided:
            if mapno in seen_maps:
                continue
            seen_maps.add(mapno)
            score[s] = score.get(s, 0) + 1
            if map_res is False:
                walk_used_misdecided_map = True
            if score[s] >= need:
                clock, clinch_side = t0, s
                break
        if clock is None:
            unclocked.append(slug)
            _excl(slug, "no_clinch_observed"); continue

        # ---- entry on the match book
        row = gmx[gmx.yes_sub_title.map(norm) == clinch_side]
        if len(row) != 1:
            _excl(slug, "no_match_ticker_for_side"); continue
        row = row.iloc[0]
        cd = gc_by.get(row.ticker)
        if cd is None or len(cd) == 0:
            _excl(slug, "match_candles_missing"); continue
        g = ffill_grid(cd, row.close_time)
        t_entry = (clock + pd.Timedelta(minutes=ENTRY_DELAY_MIN)).ceil("min")
        if t_entry not in g.index or pd.isna(g.loc[t_entry, "yes_ask_close"]):
            _excl(slug, "no_match_book_state_at_entry"); continue
        ask = float(g.loc[t_entry, "yes_ask_close"]); bidq = float(g.loc[t_entry, "yes_bid_close"])
        # liveness: >=1 match-book print (either mirror) within +-10 min of clock
        prints = []
        for tk in gmx.ticker:
            tr = gt_by.get(tk)
            if tr is not None:
                prints.append(tr.assign(side_px=tr.price_dollars if tk == row.ticker else 1 - tr.price_dollars))
        prints = pd.concat(prints) if prints else pd.DataFrame(columns=["ts", "side_px", "count"])
        live = ((prints.ts >= clock - pd.Timedelta(minutes=LIVENESS_WIN_MIN)) &
                (prints.ts <= clock + pd.Timedelta(minutes=LIVENESS_WIN_MIN))).any()
        gate_fail = []
        if not (ask <= ASK_MAX): gate_fail.append("ask>0.97")
        if not (bidq >= BID_MIN): gate_fail.append("bid<0.85")
        if not live: gate_fail.append("liveness")
        if gate_fail:
            _excl(slug, "gate:" + "+".join(gate_fail), ask, bidq); continue

        fee = kalshi_taker_fee(ask)
        won = row.result == "yes"
        # anti-hindsight: entry bar prints
        bar = prints[(prints.ts > t_entry - pd.Timedelta(minutes=1)) & (prints.ts <= t_entry)]
        contra = bool((bar.side_px < CONTRA_PX).any())
        doubt = prints[prints.side_px < DOUBT_PX]
        last_doubt = doubt.ts.max() if len(doubt) else pd.NaT
        doubt_after_entry = bool(len(doubt) and last_doubt > t_entry)
        # capacity: clinch-side prints <= entry ask within +-5 min of entry
        cap = prints[(prints.ts >= t_entry - pd.Timedelta(minutes=5)) &
                     (prints.ts <= t_entry + pd.Timedelta(minutes=5)) &
                     (prints.side_px <= ask)]["count"].sum()
        bets.append({
            "slug": slug, "event_ticker": row.event_ticker, "ticker": row.ticker,
            "clock": clock, "t_entry": t_entry, "N": N, "need": need,
            "entry_price": ask, "bid_at_entry": bidq, "fee": fee, "won": bool(won),
            "contra_entry_bar": contra,
            "last_doubt_rel_clock_min": (last_doubt - clock).total_seconds() / 60 if pd.notna(last_doubt) else None,
            "doubt_after_entry": doubt_after_entry,
            "capacity_contracts_5min": float(cap),
            "format_misinferred_diag": fmt_bad,
            "walk_used_misdecided_map": walk_used_misdecided_map,
            "day": str(clock.date()), "month": clock.strftime("%Y-%m"),
        })

    bets = pd.DataFrame(bets)
    snaps = pd.DataFrame(snaps_all)
    excl = pd.DataFrame(excl, columns=["slug", "reason", "ask", "bid"])
    return bets, snaps, excl, counts


def verdicts(bets: pd.DataFrame, label: str):
    out = {}
    if len(bets) == 0:
        return {"label": label, "n": 0}
    b = bets.sort_values("t_entry").copy()
    out["fixture_cluster"] = evaluate_frozen_rule(
        b.rename(columns={"event_ticker": "event_ticker"}), n_families_tested=15, n_variants_in_family=24, seed=0)
    day = b.copy(); day["event_ticker"] = day["day"]
    out["day_cluster_stress"] = evaluate_frozen_rule(day, n_families_tested=15, n_variants_in_family=24, seed=0)
    return out


UNIVERSE = [  # pre-registered: KXCS2GAME/KXVALORANTGAME/KXDOTA2GAME with same-fixture map books
    ("KXCS2GAME", "KXCS2MAP"),
    ("KXVALORANTGAME", "KXVALORANTMAP"),
    ("KXDOTA2GAME", "KXDOTA2MAP"),
]

if __name__ == "__main__":
    prelim = "--prelim" in sys.argv
    all_bets, all_snaps, all_excl, all_counts = [], [], [], {}
    for sg, sm in UNIVERSE:
        try:
            b, s, e, c = run(sg, sm)
        except FileNotFoundError as exc:
            print(f"!! {sg}/{sm}: data missing ({exc}) — excluded by availability, not selection")
            all_counts[sg] = {"data_missing": True}
            continue
        for df in (b, s, e):
            if len(df):
                df["series"] = sg
        all_bets.append(b); all_snaps.append(s); all_excl.append(e)
        all_counts[sg] = c
    bets = pd.concat([x for x in all_bets if len(x)], ignore_index=True) if any(len(x) for x in all_bets) else pd.DataFrame()
    snaps = pd.concat([x for x in all_snaps if len(x)], ignore_index=True) if any(len(x) for x in all_snaps) else pd.DataFrame()
    excl = pd.concat([x for x in all_excl if len(x)], ignore_index=True) if any(len(x) for x in all_excl) else pd.DataFrame(columns=["slug", "reason", "ask", "bid"])
    tag = "PRELIMINARY (partial map-candle coverage)" if prelim else "FINAL"
    print(f"==== M3 {tag} (pooled universe: CS2 + VALORANT + DOTA2) ====")
    print(json.dumps(all_counts, indent=1))
    print("\n-- exclusions --"); print(excl.reason.value_counts().to_string() if len(excl) else "none")
    ga = excl[excl.reason.str.startswith("gate:")]
    if len(ga):
        print("\ngate-failed entry ask distribution:"); print(ga.ask.describe().to_string())
        print(ga.ask.value_counts().sort_index().to_string())
    if len(snaps):
        st = snaps[snaps.settled]
        print(f"\n-- snaps: {len(snaps)} side-snaps; settled {len(st)}; "
              f"false-snap rate (lost MAP) {(~st.map_won).mean():.4f} ({(~st.map_won).sum()}/{len(st)}); "
              f"snap-side lost MATCH {(~st.match_won).mean():.4f} ({(~st.match_won).sum()}/{len(st)})")
    print(f"\n-- bets fired: {len(bets)}")
    if len(bets):
        print("per-series bet counts:", bets.series.value_counts().to_dict())
        print(bets[["series", "slug", "clock", "entry_price", "bid_at_entry", "fee", "won",
                    "contra_entry_bar", "doubt_after_entry", "capacity_contracts_5min", "month"]].to_string())
        print("\ncontra_entry_bar violations:", int(bets.contra_entry_bar.sum()))
        print("format_misinferred_diag (outcome-based, diagnostic only):", int(bets.format_misinferred_diag.sum()), "of", len(bets))
        print("bets whose clinch walk used a mis-decided map:", int(bets.walk_used_misdecided_map.sum()))
        print("doubt_after_entry flagged:", int(bets.doubt_after_entry.sum()))
        print("doubt rel clock (min) distribution:")
        print(bets.last_doubt_rel_clock_min.describe().to_string())
        bets["pnl"] = np.where(bets.won, 1 - bets.entry_price - bets.fee, -bets.entry_price - bets.fee)
        print("\nmonth splits:")
        print(bets.groupby("month").agg(n=("won", "size"), wins=("won", "sum"),
              avg_entry=("entry_price", "mean"), pnl=("pnl", "sum"),
              staked=("entry_price", lambda x: float((x + bets.loc[x.index, "fee"]).sum()))).to_string())
        print("\ncapacity: median", bets.capacity_contracts_5min.median(), "mean", bets.capacity_contracts_5min.mean())
        v = verdicts(bets, "all")
        print("\n==== VERDICT (clusters=fixture event_ticker) ====")
        print(json.dumps(v["fixture_cluster"], indent=1, default=str))
        print("\n==== STRESS (clusters=day) ====")
        print(json.dumps(v["day_cluster_stress"], indent=1, default=str))
        bx = bets[~bets.doubt_after_entry]
        if len(bx) and len(bx) < len(bets):
            print("\n==== VERDICT excluding doubt-after-entry flags ====")
            print(json.dumps(verdicts(bx, "nodoubt")["fixture_cluster"], indent=1, default=str))
        bets.to_parquet("/home/mcortese/.claude/jobs/efe2b3d4/tmp/m3_bets.parquet")
