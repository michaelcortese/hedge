#!/usr/bin/env python3
"""Live scanner: Kalshi short-horizon BTC binaries vs perp-implied fair value.

Read-only research tool behind the Wing Maker strategy (docs/PERP_STRATEGY.md).
For every open KXBTC15M / KXBTCD market it prints the LIVE order book (never the
stale ``/markets`` list quotes) against a fair value computed from:

  - S     = KXBTCPERP mid (0.3bp-wide book, same CF Benchmarks index family
            as binary settlement)
  - sigma = EWMA of 1-min perp log-returns (fast 30m / slow 6h blend), scaled
            by the measured VARIANCE RATIO (horizon variance ~0.70x of
            sqrt-t-scaled 1-min variance — BTC microstructure mean-reverts;
            without this correction wings look ~40% too cheap)

Output columns: eYES/eNO = expected taker edge after ceil(7% P(1-P)) fees
(negative almost always — that is the point), and the wing-maker candidate
quotes (fair +/- margin) for strikes in the wing bands.

Usage:
    .venv/bin/python scripts/crypto_binary_scan.py            # one snapshot
    .venv/bin/python scripts/crypto_binary_scan.py --loop 60  # every 60s
"""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

from hedge.kalshi.perps import PerpsClient

EVT_BASE = "https://api.elections.kalshi.com/trade-api/v2"
_UA = {"User-Agent": "hedge-scan/0.1"}

VARIANCE_RATIO = 0.70  # measured 2026-07-15 over 6d; recalibrate weekly
WING_LO, WING_HI = 0.20, 0.80  # fair outside [lo,hi] = wing band
MAKER_EDGE = 0.010  # min expected edge per contract for a candidate quote


def _get(path: str) -> dict:
    req = urllib.request.Request(EVT_BASE + path, headers=_UA)
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)


def norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def taker_fee(p: float) -> float:
    return math.ceil(7 * p * (1 - p)) / 100.0 if 0 < p < 1 else 0.0


def perp_state(client: PerpsClient, ticker: str = "KXBTCPERP", hours: int = 8) -> dict:
    m = client.market(ticker)
    size = float(m["contract_size"])
    now = int(time.time())
    candles = client.candlesticks(ticker, now - hours * 3600, now, 1)
    closes: list[float] = []
    for c in candles:
        p = c.get("price", {})
        v = p.get("close") or p.get("close_dollars") or p.get("mean_dollars")
        if v:
            closes.append(float(v) / size)
    rets = [math.log(closes[i + 1] / closes[i]) for i in range(len(closes) - 1)]

    def ewma_sig(halflife: float) -> float:
        lam = 0.5 ** (1 / halflife)
        v = rets[0] ** 2
        for r in rets[1:]:
            v = lam * v + (1 - lam) * r * r
        return math.sqrt(v)

    fast, slow = ewma_sig(30), ewma_sig(360)
    return {
        "mid": (float(m["bid"]) + float(m["ask"])) / 2 / size,
        "ref": float(m["reference_price"]["price"]) / size,
        "sig_1m": max(fast, 0.7 * slow),
        "sig_fast": fast,
        "sig_slow": slow,
    }


def live_book(ticker: str) -> tuple[float, float] | None:
    """(yes_bid, yes_ask) from the LIVE orderbook endpoint."""
    try:
        ob = _get(f"/markets/{ticker}/orderbook?depth=1").get("orderbook_fp", {})
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return None
    yes, no = ob.get("yes_dollars") or [], ob.get("no_dollars") or []
    if not yes or not no:
        return None
    return float(yes[-1][0]), 1 - float(no[-1][0])


def scan_once(client: PerpsClient) -> None:
    snap = perp_state(client)
    S, sig = snap["mid"], snap["sig_1m"]
    regime = snap["sig_fast"] / snap["sig_slow"] if snap["sig_slow"] > 0 else 1.0
    print(
        f"\n[{datetime.now(timezone.utc):%H:%M:%SZ}] S={S:,.0f} ref={snap['ref']:,.0f} "
        f"sig1m={sig * 1e4:.2f}bps fast/slow={regime:.2f}"
        + ("  ** VOL REGIME BREAK — pull quotes **" if regime > 1.6 else "")
    )
    hdr = f"{'ticker':32} {'K':>9} {'min':>4} {'bid':>6} {'ask':>6} {'fair':>6} {'eYES':>7} {'eNO':>7}  maker-candidate"
    print(hdr)
    for series in ("KXBTC15M", "KXBTCD"):
        d = _get(f"/markets?series_ticker={series}&status=open&limit=100")
        for m in d.get("markets", []):
            ct = datetime.fromisoformat(m["close_time"].replace("Z", "+00:00"))
            mins = (ct - datetime.now(timezone.utc)).total_seconds() / 60
            if not (5 < mins < 60):
                continue
            st, flr = m.get("strike_type", ""), m.get("floor_strike")
            if not (st.startswith("greater") and flr):
                continue
            book = live_book(m["ticker"])
            if book is None:
                continue
            bid, ask = book
            sd = sig * math.sqrt(mins * VARIANCE_RATIO)
            fair = norm_cdf(math.log(S / flr) / sd) if sd > 0 else float(S >= flr)
            e_yes = fair - ask - taker_fee(ask)
            e_no = bid - fair - taker_fee(1 - bid)
            candidate = ""
            if fair < WING_LO or fair > WING_HI:
                # wing: propose resting quotes at fair +/- required edge,
                # never improving inside fair +/- half the current spread
                sell_at = max(fair + MAKER_EDGE, bid + 0.01)
                buy_at = min(fair - MAKER_EDGE, ask - 0.01)
                if sell_at < ask:
                    candidate += f" sellYES@{sell_at:.2f}"
                if buy_at > bid and fair > WING_HI:
                    candidate += f" buyYES@{buy_at:.2f}"
            print(
                f"{m['ticker']:32} {flr:9,.0f} {mins:4.0f} {bid:6.3f} {ask:6.3f} "
                f"{fair:6.3f} {e_yes:+7.3f} {e_no:+7.3f} {candidate}"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--loop", type=int, metavar="SECONDS", help="rescan every N seconds")
    args = ap.parse_args()
    client = PerpsClient()
    while True:
        scan_once(client)
        if not args.loop:
            break
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
