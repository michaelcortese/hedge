#!/usr/bin/env python
"""Blind-hunt candidate: NO-flow-persistence taker carry (k-th NO print rule).

Rule (single entry per market):
  When a mention market prints its 10th NO-taker trade with YES price in [15,50)
  cents, buy 1 NO contract at that print's price (taker; fill proven by the print),
  hold to settlement. Exclude sports-broadcast series (MLB/NBA/NHL/WC/WNBA/ATP).

Prereqs: run scripts/research_blind_hunt.py first (builds blind/markets.parquet),
then the duckdb flatten of trades.jsonl into blind/trades_joined.parquet
(see git history / this file's build_trades()).
"""
import duckdb
import numpy as np
import pandas as pd

BLIND = "data/research/mentions/blind"
SPORTS = {"KXMLBMENTION", "KXNBAMENTION", "KXNHLMENTION", "KXWCMENTION",
          "KXWNBAMENTION", "KXATPMENTION"}


def build_trades(con):
    con.execute(f"""
    COPY (
      SELECT ticker, t.unnest.created_time::TIMESTAMPTZ AS created_time,
             CAST(t.unnest.yes_price_dollars AS DOUBLE)*100 AS yes_cents,
             CAST(t.unnest.count_fp AS DOUBLE) AS cnt,
             t.unnest.taker_side AS taker_side
      FROM (SELECT ticker, unnest(trades) AS unnest
            FROM read_json('data/research/mentions/trades.jsonl',
              columns={{ticker:'VARCHAR', trades:'STRUCT(created_time VARCHAR,
                yes_price_dollars VARCHAR, count_fp VARCHAR, taker_side VARCHAR)[]'}},
              maximum_object_size=200000000)) t
    ) TO '{BLIND}/trades.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)""")
    con.execute(f"""
    COPY (
      SELECT t.*, m.event_ticker, m.series, m.result, m.open_time, m.close_time
      FROM '{BLIND}/trades.parquet' t
      JOIN (SELECT ticker, event_ticker, series, result, open_time, close_time
            FROM '{BLIND}/markets.parquet' WHERE result IN ('yes','no')) m
      USING (ticker)
    ) TO '{BLIND}/trades_joined.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)""")


def entries(con, k=10, lo=15, hi=50):
    df = con.execute(f"""
    WITH t AS (
      SELECT ticker, event_ticker ev, series, result, yes_cents, cnt, created_time,
        row_number() OVER (PARTITION BY ticker ORDER BY created_time) rn
      FROM '{BLIND}/trades_joined.parquet'
      WHERE taker_side='no' AND yes_cents>={lo} AND yes_cents<{hi}
    ) SELECT * FROM t WHERE rn={k}""").df()
    df["y"] = (df.result == "yes").astype(int)
    fee = np.ceil(0.07 * df.yes_cents * (100 - df.yes_cents) / 100)
    df["pnl"] = 100 * (1 - df.y) - (100 - df.yes_cents) - fee  # cents/contract, net
    return df


def event_boot(d, reps=10000, seed=0):
    rng = np.random.default_rng(seed)
    arrs = [g.pnl.values for _, g in d.groupby("ev")]
    means = np.empty(reps)
    for i in range(reps):
        idx = rng.integers(0, len(arrs), len(arrs))
        means[i] = np.concatenate([arrs[j] for j in idx]).mean()
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return d.pnl.mean(), np.percentile(means, [0.5, 99.5]), p


def main():
    con = duckdb.connect()
    con.execute("SET threads=4;")
    df = entries(con)
    ex = df[~df.series.isin(SPORTS)]
    jul = ex.created_time >= pd.Timestamp("2026-07-01", tz="UTC")
    for label, d in [("ex-sports ALL", ex), ("ex-sports May-Jun", ex[~jul]),
                     ("ex-sports Jul (OOS)", ex[jul]), ("all-series ALL", df)]:
        pt, ci, p = event_boot(d)
        print(f"{label:20s} n={len(d):5d} ev={d.ev.nunique():4d} "
              f"mean={pt:+.2f}c CI99=[{ci[0]:+.2f},{ci[1]:+.2f}] p={p:.4f}")


if __name__ == "__main__":
    main()
