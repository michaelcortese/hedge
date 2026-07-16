#!/usr/bin/env python3
"""Paper-markout harness for the Wing Maker (docs/PERP_STRATEGY.md §8.1).

Simulates resting maker quotes on the wings of Kalshi BTC short-horizon
binaries and replays the PUBLIC trade tape against them — no credentials, no
orders, no risk. Produces the one number that gates the build: net ¢/contract
and fills/day a slow (REST-speed) maker would actually have earned.

Fill model (deliberately conservative):
  - We are LAST in queue: when we "post" at price P we snapshot all displayed
    size at P and better as ``size_ahead``; only tape volume at P after our
    post time, in excess of that, fills us.
  - A print STRICTLY better than our price (through our level) means the level
    was swept by price priority -> immediate fill.
  - Any cancel/replace resets queue position (size_ahead re-snapshotted).

Quoting rules mirror the strategy: wings only (fair < 0.20 or > 0.80), tau in
[5, 55] min, pull on vol-regime break, reprice only when the target moves more
than a tick. Fair value = perp mid + EWMA(1m) vol scaled by VARIANCE_RATIO.

Usage:
    .venv/bin/python scripts/paper_wingmaker.py run [--minutes 480]
    .venv/bin/python scripts/paper_wingmaker.py score
    .venv/bin/python scripts/paper_wingmaker.py run --once      # smoke test

Run it for ~2 weeks (e.g. under tmux / on the fly box alongside the weather
loop). Verdict gate: >= 200 settled fills and positive net ¢/contract.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

from hedge.kalshi.perps import PerpsClient

EVT_BASE = "https://api.elections.kalshi.com/trade-api/v2"
_UA = {"User-Agent": "hedge-paper-wingmaker/0.1"}

# --- strategy constants (keep in sync with docs/PERP_STRATEGY.md §6) ---
VARIANCE_RATIO = 0.70
WING_LO, WING_HI = 0.20, 0.80     # quote only when fair is outside this band
MIN_EDGE = 0.010                  # required expected edge vs fair, per contract
TAU_MIN, TAU_MAX = 5.0, 55.0      # minutes to close
REGIME_BREAK = 1.6                # fast/slow EWMA ratio -> pull quotes
SIM_SIZE = 100.0                  # contracts we pretend to rest per quote
TICK = 0.001                      # quote granularity (deci-cent bands allow 0.001)
CYCLE_S = 15.0
MAKER_FEE = 0.0                   # set to 0.25*taker if fee PDF says crypto charges makers
# Risk caps (docs/PERP_STRATEGY.md §6) — without these the paper P&L is dominated
# by averaging into a single blow-through, which the real strategy forbids.
CAP_STRIKE = 100.0                # max filled contracts per (market, side, policy)
CAP_EVENT = 300.0                 # max filled contracts per (event, policy)

# On Fly the durable volume is /data (HEDGE_STATE_DIR) — set WINGMAKER_DATA_DIR
# to /data/wingmaker there so logs survive redeploys; locally defaults to repo data/.
DATA_DIR = os.environ.get("WINGMAKER_DATA_DIR") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "runs", "wingmaker")


def _get(path: str) -> dict:
    req = urllib.request.Request(EVT_BASE + path, headers=_UA)
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)


def norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


class Log:
    def __init__(self, name: str):
        os.makedirs(DATA_DIR, exist_ok=True)
        self.f = open(os.path.join(DATA_DIR, name), "a")

    def write(self, rec: dict) -> None:
        rec["ts"] = round(time.time(), 3)
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()


# ---------------------------------------------------------------- fair value
def perp_state(client: PerpsClient, hours: int = 8) -> dict | None:
    try:
        m = client.market("KXBTCPERP")
        size = float(m["contract_size"])
        now = int(time.time())
        candles = client.candlesticks("KXBTCPERP", now - hours * 3600, now, 1)
    except Exception:
        return None
    closes = []
    for c in candles:
        p = c.get("price", {})
        v = p.get("close") or p.get("close_dollars") or p.get("mean_dollars")
        if v:
            closes.append(float(v) / size)
    if len(closes) < 60:
        return None
    rets = [math.log(closes[i + 1] / closes[i]) for i in range(len(closes) - 1)]

    def ewma(hl: float) -> float:
        lam = 0.5 ** (1 / hl)
        v = rets[0] ** 2
        for r in rets[1:]:
            v = lam * v + (1 - lam) * r * r
        return math.sqrt(v)

    fast, slow = ewma(30), ewma(360)
    return dict(
        mid=(float(m["bid"]) + float(m["ask"])) / 2 / size,
        sig_1m=max(fast, 0.7 * slow),
        regime=fast / slow if slow > 0 else 1.0,
    )


def fair_prob(S: float, K: float, sig_1m: float, mins: float) -> float:
    sd = sig_1m * math.sqrt(max(mins, 0.01) * VARIANCE_RATIO)
    if sd <= 0:
        return 1.0 if S >= K else 0.0
    return norm_cdf(math.log(S / K) / sd)


# ---------------------------------------------------------------- market I/O
def open_targets() -> list[dict]:
    """Open >=-strike markets on the target series inside the tau window."""
    out = []
    now = datetime.now(timezone.utc)
    for series in ("KXBTC15M", "KXBTCD"):
        try:
            d = _get(f"/markets?series_ticker={series}&status=open&limit=100")
        except Exception:
            continue
        for m in d.get("markets", []):
            ct = datetime.fromisoformat(m["close_time"].replace("Z", "+00:00"))
            mins = (ct - now).total_seconds() / 60
            st, flr = m.get("strike_type", ""), m.get("floor_strike")
            if TAU_MIN < mins < TAU_MAX and st.startswith("greater") and flr:
                out.append(dict(ticker=m["ticker"], K=flr, mins=mins, series=series,
                                close_ts=int(ct.timestamp())))
    return out


def full_book(ticker: str) -> dict | None:
    """Return {'bid','ask','ask_depth':{px:sz asks},'bid_depth':{px:sz bids}}."""
    try:
        ob = _get(f"/markets/{ticker}/orderbook?depth=10").get("orderbook_fp", {})
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return None
    yes = ob.get("yes_dollars") or []     # resting YES *bids*
    no = ob.get("no_dollars") or []       # resting NO bids -> YES asks at 1-px
    if not yes or not no:
        return None
    bid_depth = {float(p): float(s) for p, s in yes}
    ask_depth = {round(1 - float(p), 4): float(s) for p, s in no}
    return dict(bid=max(bid_depth), ask=min(ask_depth),
                bid_depth=bid_depth, ask_depth=ask_depth)


def new_trades(ticker: str, seen: set[str]) -> list[dict]:
    try:
        d = _get(f"/markets/trades?ticker={ticker}&limit=100")
    except Exception:
        return []
    fresh = [t for t in d.get("trades", []) if t["trade_id"] not in seen]
    for t in fresh:
        seen.add(t["trade_id"])
    return fresh


# ---------------------------------------------------------------- simulation
class Quote:
    """One simulated resting order (side: 'sell_yes' | 'buy_yes').

    policy 'model': price anchored to model fair (requires >= MIN_EDGE vs fair).
    policy 'join':  rest at the existing best same-side wing quote — the exact
                    seat the trade-tape analysis showed being paid; no model
                    opinion beyond the wing filter.
    """

    def __init__(self, ticker: str, side: str, price: float, fair: float,
                 book: dict, close_ts: int, policy: str):
        self.ticker, self.side, self.price, self.fair = ticker, side, price, fair
        self.close_ts, self.policy = close_ts, policy
        self.posted = time.time()
        self.remaining = SIM_SIZE
        if side == "sell_yes":
            self.size_ahead = sum(s for p, s in book["ask_depth"].items()
                                  if p <= price + 1e-9)
        else:
            self.size_ahead = sum(s for p, s in book["bid_depth"].items()
                                  if p >= price - 1e-9)

    def match(self, trades: list[dict]) -> list[dict]:
        """Consume tape trades; return fill records."""
        fills = []
        for t in trades:
            ts = datetime.fromisoformat(
                t["created_time"].replace("Z", "+00:00")).timestamp()
            if ts < self.posted or self.remaining <= 0:
                continue
            px = float(t["yes_price_dollars"])
            n = float(t["count_fp"])
            if self.side == "sell_yes" and t["taker_side"] == "yes":
                if px > self.price + 1e-9:          # swept through our level
                    take = self.remaining
                elif abs(px - self.price) <= 1e-9:  # trade at our level: queue
                    self.size_ahead -= n
                    take = min(self.remaining, -self.size_ahead) if self.size_ahead < 0 else 0
                    self.size_ahead = max(self.size_ahead, 0)
                else:
                    continue
            elif self.side == "buy_yes" and t["taker_side"] == "no":
                if px < self.price - 1e-9:
                    take = self.remaining
                elif abs(px - self.price) <= 1e-9:
                    self.size_ahead -= n
                    take = min(self.remaining, -self.size_ahead) if self.size_ahead < 0 else 0
                    self.size_ahead = max(self.size_ahead, 0)
                else:
                    continue
            else:
                continue
            if take > 0:
                self.remaining -= take
                fills.append(dict(kind="fill", ticker=self.ticker, side=self.side,
                                  policy=self.policy, price=self.price,
                                  fair_at_post=self.fair, size=take,
                                  trade_id=t["trade_id"], trade_px=px,
                                  trade_ts=ts, close_ts=self.close_ts))
        return fills


def target_quote(side: str, fair: float, book: dict) -> float | None:
    """Price we would rest at, or None if no viable quote."""
    if side == "sell_yes":
        px = max(round(fair + MIN_EDGE, 3), round(book["bid"] + TICK, 3))
        if px >= book["ask"] - 1e-9:               # can't improve: join the ask
            px = book["ask"]
        return px if px - fair >= MIN_EDGE - 1e-9 else None
    px = min(round(fair - MIN_EDGE, 3), round(book["ask"] - TICK, 3))
    if px <= book["bid"] + 1e-9:
        px = book["bid"]
    return px if fair - px >= MIN_EDGE - 1e-9 else None


def run(minutes: float, once: bool = False) -> None:
    client = PerpsClient()
    qlog, flog, mlog = Log("quotes.jsonl"), Log("fills.jsonl"), Log("marks.jsonl")
    quotes: dict[tuple[str, str], Quote] = {}
    seen: dict[str, set] = {}
    filled_strike: dict[tuple, float] = {}   # (ticker, side, policy) -> contracts
    filled_event: dict[tuple, float] = {}    # (event, policy) -> contracts
    end = time.time() + minutes * 60
    n_fills = 0

    def event_of(ticker: str) -> str:
        return ticker.split("-T")[0]
    while time.time() < end:
        t0 = time.time()
        ps = perp_state(client)
        if ps is None:
            time.sleep(5)
            continue
        mlog.write(dict(kind="mark", S=round(ps["mid"], 2),
                        sig1m=round(ps["sig_1m"] * 1e4, 3),
                        regime=round(ps["regime"], 3)))
        pull_all = ps["regime"] > REGIME_BREAK
        targets = open_targets()
        live = set()
        # rank candidates by proximity to the wing sweet spot (~8c / ~92c) so
        # the book-fetch budget is spent on strikes that can actually quote,
        # not the dozens of dead far-OTM strikes in each ladder
        for tgt in targets:
            tgt["fair"] = fair_prob(ps["mid"], tgt["K"], ps["sig_1m"], tgt["mins"])
            tgt["rank"] = min(abs(tgt["fair"] - 0.08), abs(tgt["fair"] - 0.92))
            tgt["has_quotes"] = any(k[0] == tgt["ticker"] for k in quotes)
        cands = sorted([t for t in targets
                        if t["has_quotes"] or 0.002 < t["fair"] < 0.35
                        or 0.65 < t["fair"] < 0.998],
                       key=lambda t: (not t["has_quotes"], t["rank"]))[:16]
        for tgt in cands:
            fair = tgt["fair"]
            book = full_book(tgt["ticker"])
            time.sleep(0.08)
            # match tape once per market against all its live quotes
            mkt_keys = [k for k in quotes if k[0] == tgt["ticker"]]
            if mkt_keys:
                tape = new_trades(tgt["ticker"], seen.setdefault(tgt["ticker"], set()))
                for k in mkt_keys:
                    for f in quotes[k].match(tape):
                        n_fills += 1
                        sk = (f["ticker"], f["side"], f["policy"])
                        ek = (event_of(f["ticker"]), f["policy"])
                        filled_strike[sk] = filled_strike.get(sk, 0) + f["size"]
                        filled_event[ek] = filled_event.get(ek, 0) + f["size"]
                        f["S"], f["sig1m"] = round(ps["mid"], 2), round(ps["sig_1m"] * 1e4, 3)
                        flog.write(f)
                        print(f"FILL [{f['policy']}] {f['side']} {f['ticker']} "
                              f"@{f['price']:.3f} x{f['size']:.0f} "
                              f"(fair was {f['fair_at_post']:.3f})", flush=True)
            if book is None or pull_all:
                for k in mkt_keys:
                    quotes.pop(k, None)
                continue
            # desired quotes per (side, policy)
            wanted: dict[tuple[str, str, str], float] = {}
            if 0.015 < fair < WING_LO:
                px = target_quote("sell_yes", fair, book)
                if px is not None:
                    wanted[(tgt["ticker"], "sell_yes", "model")] = px
            if WING_HI < fair < 0.985:
                px = target_quote("buy_yes", fair, book)
                if px is not None:
                    wanted[(tgt["ticker"], "buy_yes", "model")] = px
            if 0.01 <= book["ask"] <= WING_LO:      # join the displayed wing ask
                wanted[(tgt["ticker"], "sell_yes", "join")] = book["ask"]
            if WING_HI <= book["bid"] <= 0.99:      # join the displayed wing bid
                wanted[(tgt["ticker"], "buy_yes", "join")] = book["bid"]
            for key, px in wanted.items():
                # risk caps: once a strike/event has absorbed its budget of
                # fills, stop re-arming — no averaging into a blow-through
                if (filled_strike.get((key[0], key[1], key[2]), 0) >= CAP_STRIKE
                        or filled_event.get((event_of(key[0]), key[2]), 0) >= CAP_EVENT):
                    quotes.pop(key, None)
                    continue
                cur = quotes.get(key)
                if cur is None or abs(cur.price - px) > TICK + 1e-9:
                    quotes[key] = Quote(tgt["ticker"], key[1], px, fair, book,
                                        tgt["close_ts"], key[2])
                    qlog.write(dict(kind="quote", ticker=tgt["ticker"],
                                    side=key[1], policy=key[2], price=px,
                                    fair=round(fair, 4),
                                    size_ahead=round(quotes[key].size_ahead, 1),
                                    bid=book["bid"], ask=book["ask"],
                                    mins=round(tgt["mins"], 1)))
                live.add(key)
            for k in [k for k in mkt_keys if k not in live]:
                quotes.pop(k, None)
        for key in [k for k in quotes if k not in live]:
            quotes.pop(key)
        if once:
            print(f"smoke ok: {len(targets)} targets, {len(quotes)} live quotes, "
                  f"S={ps['mid']:,.0f} sig={ps['sig_1m']*1e4:.2f}bps")
            return
        time.sleep(max(1.0, CYCLE_S - (time.time() - t0)))
    print(f"run done, {n_fills} simulated fills", flush=True)


# -------------------------------------------------------------------- score
def score() -> None:
    path = os.path.join(DATA_DIR, "fills.jsonl")
    if not os.path.exists(path):
        print("no fills logged yet")
        return
    fills = [json.loads(l) for l in open(path)]
    marks = [json.loads(l) for l in open(os.path.join(DATA_DIR, "marks.jsonl"))]
    results: dict[str, int | None] = {}
    for tk in {f["ticker"] for f in fills}:
        try:
            m = _get(f"/markets/{tk}")["market"]
            results[tk] = (1 if m["result"] == "yes" else 0) \
                if m.get("result") in ("yes", "no") else None
        except Exception:
            results[tk] = None
        time.sleep(0.05)

    def mark_at(ts: float) -> float | None:
        best, bd = None, 1e9
        for m in marks:
            d = abs(m["ts"] - ts)
            if d < bd:
                best, bd = m["S"], d
        return best if bd <= 120 else None

    import statistics
    days = max((max(f["trade_ts"] for f in fills)
                - min(f["trade_ts"] for f in fills)) / 86400, 1 / 24)
    for policy in ("join", "model"):
        pf = [f for f in fills if f.get("policy", "model") == policy]
        if not pf:
            print(f"[{policy}] no fills")
            continue
        settled, pnls, markouts = 0, [], []
        for f in pf:
            y = results.get(f["ticker"])
            s0, s2 = mark_at(f["trade_ts"]), mark_at(f["trade_ts"] + 120)
            if s0 and s2:
                drift = (s2 / s0 - 1) * 1e4
                markouts.append(-drift if f["side"] == "sell_yes" else drift)
            if y is None:
                continue
            settled += 1
            if f["side"] == "sell_yes":
                pnl = f["price"] - y - MAKER_FEE
            else:
                pnl = y - f["price"] - MAKER_FEE
            pnls.append((pnl, f["size"]))
        print(f"\n[{policy}] fills: {len(pf)}, settled: {settled}, "
              f"{len(pf)/days:.1f} fills/day over {days:.2f} days")
        if pnls:
            tot_ct = sum(s for _, s in pnls)
            net = sum(p * s for p, s in pnls) / tot_ct
            wsum = sum(p * s for p, s in pnls)
            print(f"  net P&L: {net:+.4f}/contract on {tot_ct:,.0f} contracts "
                  f"(total {wsum:+.2f})  gate: positive over >=200 settled fills")
        if markouts:
            print(f"  2-min perp markout: {statistics.mean(markouts):+.2f}bps "
                  f"(negative = adverse selection)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("run")
    pr.add_argument("--minutes", type=float, default=480)
    pr.add_argument("--once", action="store_true")
    sub.add_parser("score")
    args = ap.parse_args()
    if args.cmd == "run":
        run(args.minutes, once=args.once)
    else:
        score()


if __name__ == "__main__":
    main()
