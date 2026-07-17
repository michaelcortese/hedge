"""Composition of the minute-candle-covered subset used for repricing."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from research_mentions_lib import DATA, taker_fee_cents  # noqa: E402
from research_tape_micro import build_dataset, cluster_bootstrap  # noqa: E402


def main():
    mk, tr, j = build_dataset()
    del tr
    jj = j.join(j.groupby("event_ticker").close_time.max().rename("ev_end"),
                on="event_ticker")
    jj["mte"] = (jj.ev_end - jj.ts).dt.total_seconds() / 60.0
    w = jj[(jj.mte >= 60) & (jj.mte < 180)]
    last = w.sort_values("ts").groupby("ticker").tail(1)
    g = last[last.yes_price.between(20, 80)].copy()
    g["series"] = g.ticker.str.split("-").str[0]

    covered = set()
    with (DATA / "minute_candles.jsonl").open() as f:
        for line in f:
            covered.add(json.loads(line).get("ticker"))
    g["cov"] = g.ticker.isin(covered)

    entry = np.clip((100.0 - g.yes_price + 4.0).to_numpy(), 1, 99)
    fee = np.array([taker_fee_cents(e) for e in entry])
    g["net"] = np.where(g.result.to_numpy() == 0, 100.0, 0.0) - entry - fee

    print("covered vs not, by series:")
    print(g.groupby(["cov", "series"]).agg(n=("net", "size"),
                                           net=("net", "mean"))
          .round(2).to_string())
    for lbl, gg in [("covered", g[g.cov]), ("uncovered", g[~g.cov])]:
        st = cluster_bootstrap(gg.net.to_numpy(), gg.event_ticker.to_numpy())
        print(f"{lbl}: n={st['n']} ({st['n_clusters']} ev) "
              f"net(assumed)={st['mean']:+.2f} "
              f"CI=[{st['ci_lo']:+.2f},{st['ci_hi']:+.2f}]")


if __name__ == "__main__":
    main()
