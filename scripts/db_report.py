#!/usr/bin/env python3
"""Read-only views over the live trading DB — the learning loop's window.

Everything the bot believes, decides, does, and realizes is persisted to the durable
SQLite store (``$HEDGE_STATE_DIR/hedge.db`` on the Fly volume). This CLI turns that
into the handful of views you actually review, without hand-writing SQL or SSH-ing
into sqlite:

    db_report.py decisions [--day YYYY-MM-DD]   every decision that day (incl. HOLDs)
    db_report.py trades                         placed orders -> fills -> settlement P&L
    db_report.py calibration [--by city|strategy]
                                                predicted prob vs realized outcome
    db_report.py pnl                            realized P&L by UTC day

Add ``--csv`` to emit machine-readable rows for offline analysis, ``--db PATH`` to
point at a copied-down database.

It opens the DB read-only and never writes — safe to run against a live bot.
"""

from __future__ import annotations

import argparse
import csv
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path


def _default_db() -> Path:
    base = Path(os.environ.get("HEDGE_STATE_DIR", "data/runs/live"))
    return base / "hedge.db"


def _connect(path: Path) -> sqlite3.Connection:
    if not path.exists():
        sys.exit(f"no database at {path} (set HEDGE_STATE_DIR or pass --db)")
    # Read-only URI so we can safely inspect a live bot's DB.
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _city(ticker: str) -> str:
    # KXHIGHCHI-26JUN29-T80 -> KXHIGHCHI
    return ticker.split("-", 1)[0]


def _emit(rows: list[dict], cols: list[str], *, csv_out: bool, title: str) -> None:
    if csv_out:
        w = csv.DictWriter(sys.stdout, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
        return
    print(f"\n{title}  ({len(rows)} row{'s' if len(rows) != 1 else ''})")
    if not rows:
        print("  (none)")
        return
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    print("  " + "  ".join(c.ljust(widths[c]) for c in cols))
    print("  " + "  ".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  " + "  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


# --------------------------------------------------------------------------- #
# Views                                                                         #
# --------------------------------------------------------------------------- #
def cmd_decisions(conn: sqlite3.Connection, args) -> None:
    day = args.day or _today()
    rows = conn.execute(
        "SELECT ticker, strategy, action, side, count, price_cents, prob, edge, "
        "kelly_fraction, placed, error, reason FROM decisions "
        "WHERE utc_date=? ORDER BY id", (day,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["prob"] = None if d["prob"] is None else round(d["prob"], 3)
        d["edge"] = None if d["edge"] is None else round(d["edge"], 4)
        d["kelly_fraction"] = None if d["kelly_fraction"] is None else round(d["kelly_fraction"], 4)
        out.append(d)
    _emit(out, ["ticker", "strategy", "action", "side", "count", "price_cents",
               "prob", "edge", "kelly_fraction", "placed", "reason"],
          csv_out=args.csv, title=f"DECISIONS {day}")


def cmd_trades(conn: sqlite3.Connection, args) -> None:
    # Fills are the truth; left-join settlement so still-open trades show too.
    rows = conn.execute(
        "SELECT f.ticker, f.side, f.action, f.fill_count, f.avg_price_cents, f.fee_cents, "
        "       s.outcome, s.pnl, s.utc_date "
        "FROM fills f LEFT JOIN settled s ON s.ticker = f.ticker "
        "WHERE f.fill_count > 0 ORDER BY f.ts").fetchall()
    out = []
    realized = 0.0
    for r in rows:
        d = dict(r)
        d["avg_price_cents"] = None if d["avg_price_cents"] is None else round(d["avg_price_cents"], 1)
        d["fee_cents"] = None if d["fee_cents"] is None else round(d["fee_cents"], 1)
        if d["pnl"] is not None:
            d["pnl"] = round(d["pnl"], 2)
            realized += float(d["pnl"])
        d["status"] = "settled" if d["outcome"] else "open"
        out.append(d)
    _emit(out, ["ticker", "side", "action", "fill_count", "avg_price_cents",
               "fee_cents", "outcome", "pnl", "status"],
          csv_out=args.csv, title="TRADES (fills -> settlement)")
    if not args.csv:
        print(f"\n  realized P&L (settled only): ${realized:,.2f}")


def cmd_calibration(conn: sqlite3.Connection, args) -> None:
    # One sample per (ticker) we have a settled outcome for: the model's mean P(YES)
    # vs the realized YES outcome. De-dup by ticker so a market traded many cycles
    # doesn't dominate (mirrors the guard's sampling).
    rows = conn.execute(
        "SELECT d.ticker, AVG(d.prob) AS prob, d.strategy, s.outcome "
        "FROM decisions d JOIN settled s ON s.ticker = d.ticker "
        "WHERE d.prob IS NOT NULL GROUP BY d.ticker").fetchall()
    samples = [(float(r["prob"]), r["outcome"] == "yes",
                _city(r["ticker"]) if args.by == "city" else (r["strategy"] or "?"))
               for r in rows if r["outcome"] in ("yes", "no")]
    if not samples:
        print("\nCALIBRATION: no settled decisions yet (need resolved markets).")
        return

    groups: dict[str, list[tuple[float, bool]]] = {}
    for p, y, g in samples:
        groups.setdefault("ALL", []).append((p, y))
        if args.by:
            groups.setdefault(g, []).append((p, y))

    out = []
    for name, gs in sorted(groups.items()):
        n = len(gs)
        brier = sum((p - (1.0 if y else 0.0)) ** 2 for p, y in gs) / n
        pred = sum(p for p, _ in gs) / n
        realized = sum(1 for _, y in gs if y) / n
        out.append({"group": name, "n": n, "pred_yes": round(pred, 3),
                    "realized_yes": round(realized, 3), "brier": round(brier, 4)})
    _emit(out, ["group", "n", "pred_yes", "realized_yes", "brier"],
          csv_out=args.csv, title="CALIBRATION (predicted vs realized P(YES))")

    if not args.csv:
        # Reliability bins on the ALL set — the core "are we right?" picture.
        allgs = groups["ALL"]
        print("\n  reliability (10% bins):")
        for lo in [i / 10 for i in range(10)]:
            hi = lo + 0.1
            b = [(p, y) for p, y in allgs if lo <= p < hi or (hi == 1.0 and p == 1.0)]
            if not b:
                continue
            pred = sum(p for p, _ in b) / len(b)
            real = sum(1 for _, y in b if y) / len(b)
            bar = "#" * round(real * 20)
            print(f"    [{lo:.1f},{hi:.1f})  n={len(b):3d}  pred={pred:.2f}  "
                  f"real={real:.2f}  {bar}")
        brier = sum((p - (1.0 if y else 0.0)) ** 2 for p, y in allgs) / len(allgs)
        # Brier skill vs the base-rate-only forecast (predict the overall YES rate).
        base = sum(1 for _, y in allgs if y) / len(allgs)
        ref = sum((base - (1.0 if y else 0.0)) ** 2 for _, y in allgs) / len(allgs)
        skill = f"{1.0 - brier / ref:+.3f}" if ref > 0 else "n/a (need both outcomes)"
        print(f"\n  Brier={brier:.4f}  base-rate Brier={ref:.4f}  "
              f"skill={skill}  (>0 beats base rate)")


def cmd_pnl(conn: sqlite3.Connection, args) -> None:
    rows = conn.execute(
        "SELECT utc_date, realized, fees FROM daily_pnl ORDER BY utc_date").fetchall()
    out = [{"utc_date": r["utc_date"], "realized": round(float(r["realized"]), 2)}
           for r in rows]
    _emit(out, ["utc_date", "realized"], csv_out=args.csv, title="REALIZED P&L BY DAY")
    if not args.csv and out:
        total = sum(r["realized"] for r in out)
        print(f"\n  cumulative realized: ${total:,.2f}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="read-only views over the trading DB")
    ap.add_argument("--db", type=Path, default=None, help="path to hedge.db")
    ap.add_argument("--csv", action="store_true", help="emit CSV instead of a table")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_dec = sub.add_parser("decisions", help="every decision on a day (incl. HOLDs)")
    p_dec.add_argument("--day", type=str, default=None, help="UTC date YYYY-MM-DD (default today)")
    sub.add_parser("trades", help="placed orders -> fills -> settlement P&L")
    p_cal = sub.add_parser("calibration", help="predicted prob vs realized outcome")
    p_cal.add_argument("--by", choices=["city", "strategy"], default=None,
                       help="break the table out by city or strategy")
    sub.add_parser("pnl", help="realized P&L by UTC day")

    args = ap.parse_args(argv)
    conn = _connect(args.db or _default_db())
    {"decisions": cmd_decisions, "trades": cmd_trades,
     "calibration": cmd_calibration, "pnl": cmd_pnl}[args.cmd](conn, args)


if __name__ == "__main__":
    main()
