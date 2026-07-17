"""Anchor-error sensitivity for the family2b rule.

Live trading cannot know ev_end = max close_time (ex-post). Simulate a
scheduled-end estimate that is off by +/-30 and +/-60 minutes: anchor the
window on ev_end + shift, keep everything else identical (entry at last
print in the shifted window +4c, taker fee, hold to settlement). A real
scheduled anchor errs both ways (overtime, early finishes); a rule that
dies under modest anchor error is not implementable.

Also: variant that additionally requires the market to still be OPEN at the
shifted T (live tradeability), the honest live version.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from research_mentions_lib import taker_fee_cents  # noqa: E402
from research_tape_micro import build_dataset, cluster_bootstrap  # noqa: E402


def run(jj, shift_min: float, require_open: bool):
    ev_end = jj.ev_end + pd.Timedelta(minutes=shift_min)
    mte = (ev_end - jj.ts).dt.total_seconds() / 60.0
    w = jj[(mte >= 60) & (mte < 180)]
    last = w.sort_values("ts").groupby("ticker").tail(1)
    g = last[last.yes_price.between(20, 80)].copy()
    if require_open:
        T = g.ev_end + pd.Timedelta(minutes=shift_min - 60)
        g = g[g.close_time > T]
    if len(g) < 5:
        print(f"  shift={shift_min:+4.0f} open_req={require_open}: n={len(g)}")
        return
    entry = np.clip((100.0 - g.yes_price + 4.0).to_numpy(), 1, 99)
    fee = np.array([taker_fee_cents(e) for e in entry])
    net = np.where(g.result.to_numpy() == 0, 100.0, 0.0) - entry - fee
    st = cluster_bootstrap(net, g.event_ticker.to_numpy())
    print(f"  shift={shift_min:+4.0f}min open_req={int(require_open)}: "
          f"n={st['n']} ({st['n_clusters']} ev) YES={g.result.mean():.3f} "
          f"net={st['mean']:+.2f} CI=[{st['ci_lo']:+.2f},{st['ci_hi']:+.2f}] "
          f"p={st['p_le_0']:.4f}")


def main():
    mk, tr, j = build_dataset()
    del tr
    jj = j.join(j.groupby("event_ticker").close_time.max().rename("ev_end"),
                on="event_ticker")
    print("anchor-shift sensitivity, [60,180) window, 20-80c, entry +4c:")
    for req in (False, True):
        for s in (-60, -30, 0, 30, 60):
            run(jj, s, req)


if __name__ == "__main__":
    main()
