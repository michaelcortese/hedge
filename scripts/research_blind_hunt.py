#!/usr/bin/env python
"""Blind edge hunt over Kalshi mention markets — flatten events.jsonl to a market table."""
import json, re, sys
from pathlib import Path
import pandas as pd

DATA = Path("/home/mcortese/fun/hedge/.claude/worktrees/mentions-ml/data/research/mentions")
OUT = DATA / "blind"
OUT.mkdir(exist_ok=True)

def f(x, default=None):
    if x is None: return default
    try: return float(x)
    except (TypeError, ValueError): return default

rows = []
with open(DATA / "events.jsonl") as fh:
    for line in fh:
        ev = json.loads(line)
        et = ev.get("event_ticker")
        for m in ev.get("markets", []):
            cs = m.get("custom_strike") or {}
            rows.append(dict(
                ticker=m.get("ticker"),
                event_ticker=et,
                series=(et or "").split("-")[0],
                word=cs.get("Word") or m.get("yes_sub_title") or m.get("no_sub_title"),
                custom_strike=json.dumps(cs) if cs else None,
                strike_type=m.get("strike_type"),
                result=m.get("result"),
                status=m.get("status"),
                open_time=m.get("open_time"),
                close_time=m.get("close_time"),
                created_time=m.get("created_time"),
                occurrence=m.get("occurrence_datetime"),
                settlement_ts=m.get("settlement_ts"),
                expiration_value=m.get("expiration_value"),
                volume=f(m.get("volume_fp"), 0.0),
                open_interest=f(m.get("open_interest_fp"), 0.0),
                last_price=f(m.get("last_price_dollars")),
                title=m.get("title"),
                rules=m.get("rules_primary"),
            ))

df = pd.DataFrame(rows)
for c in ("open_time","close_time","created_time","occurrence","settlement_ts"):
    df[c] = pd.to_datetime(df[c], errors="coerce", utc=True, format="ISO8601")
df.to_parquet(OUT / "markets.parquet")
print(f"{len(df)} markets, {df.event_ticker.nunique()} events, {df.series.nunique()} series")
print(df.result.value_counts(dropna=False))
print("\nTop series by market count:")
s = df.groupby("series").agg(n=("ticker","size"), yes_rate=("result", lambda r: (r=="yes").mean()),
                             vol=("volume","sum"), events=("event_ticker","nunique"))
print(s.sort_values("n", ascending=False).head(40).to_string())
