"""Forward paper tournament: log live signals + prices, score realized P&L.

The backtest proves *calibration* (are the probabilities right?). This proves
*profitability against real prices* — the thing calibration can't tell you, because
edge only exists where the strategy disagrees with the market.

Workflow:
  * ``snapshot`` — each cycle, for every open temperature market: build the
    ``MarketView``, run every strategy, and append one JSONL row per (strategy,
    market) capturing the signal **and** the live YES bid/ask. This is also how we
    record Kalshi prices over time, which defeats the historical-price cold-start.
  * ``score`` — after markets settle, join the logged rows to outcomes and compute
    realized P&L through the framework's documented edge/Kelly/fee math (CLAUDE.md
    "How sizing uses your signal"). Reused here, to be unified with the real
    decision engine once it's wired.

P&L is reported two ways: **per-contract** (clean skill→money signal, fee- and
spread-aware) and **fractional-Kelly bankroll growth** (what you'd actually compound).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Where paper signal logs live. On Fly, HEDGE_STATE_DIR=/data (the durable volume),
# so logs survive restarts/redeploys; locally it falls back to the repo path. This
# is what lets the forward edge-evidence track accumulate across deploys.
_STATE_DIR = os.environ.get("HEDGE_STATE_DIR")
PAPER_DIR = Path(_STATE_DIR) / "paper" if _STATE_DIR else Path("data/runs/paper")


# --------------------------------------------------------------------------- #
# Risk knobs (mirror config.yaml `risk:`; defaults match config.example.yaml). #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RiskParams:
    lambda_kelly: float = 0.25
    k_sigma: float = 2.0
    tau_min_cents: float = 2.0
    fee_coeff: float = 0.07


def taker_fee(price: float, coeff: float = 0.07) -> float:
    """Kalshi taker fee per contract in dollars: ceil(coeff * P*(1-P) * 100)c."""
    cents = math.ceil(coeff * price * (1 - price) * 100)
    return cents / 100.0


@dataclass
class Decision:
    side: str            # "yes" | "no" | "none"
    exec_price: float    # price paid per contract (dollars)
    edge: float          # net edge after fee/spread (dollars/contract)
    kelly_frac: float    # fractional-Kelly bankroll share


def _tradeable(price: float | None) -> bool:
    """A quote is fillable only if it sits strictly inside (0, 1) dollars.

    A YES ask of 0.00/1.00 (or a NO ask implied by a yes_bid of 0.00/1.00) is the
    API's way of saying "no resting size on that side" — NOT a free or certain fill.
    Taking those literally is how a one-sided/locked book becomes a phantom edge that
    Kelly bets into. Demo books are routinely one-sided; prod has locked markets too.
    """
    return price is not None and 0.0 < price < 1.0


def decide(prob: float, sigma: float, yes_bid: float, yes_ask: float,
          risk: RiskParams) -> Decision:
    """Pick a side and size it, or abstain. Mirrors the framework's sizing rules.

    Buys cross the spread (YES at ask, NO at ``1 - yes_bid``); edge is net of the
    taker fee. A side is only considered when its quote is genuinely tradeable
    (strictly inside the dollar band) — a missing/degenerate quote is not a fill.
    Trades only when the edge clears both ``tau_min_cents`` and ``k_sigma * sigma``
    (the noise gate that keeps Kelly safe).
    """
    tau = risk.tau_min_cents / 100.0
    gate = risk.k_sigma * sigma

    # Each candidate: (side, exec_price, win_prob, net_edge_after_fee, raw_edge).
    candidates: list[tuple[str, float, float, float, float]] = []
    if _tradeable(yes_ask):  # YES side: buy at the ask.
        candidates.append(("yes", yes_ask, prob,
                           (prob - yes_ask) - taker_fee(yes_ask, risk.fee_coeff),
                           prob - yes_ask))
    if _tradeable(yes_bid):  # NO side: buy at (1 - yes_bid), which is then in (0, 1).
        no_price = 1 - yes_bid
        candidates.append(("no", no_price, 1 - prob,
                           ((1 - prob) - no_price) - taker_fee(no_price, risk.fee_coeff),
                           (1 - prob) - no_price))

    if not candidates:
        return Decision("none", 0.0, 0.0, 0.0)  # no tradeable side — no market to beat

    side, price, win_prob, net_edge, raw_edge = max(candidates, key=lambda c: c[3])
    if net_edge <= 0 or net_edge < tau or raw_edge < gate:
        return Decision("none", price, net_edge, 0.0)
    # Fractional Kelly on a binary contract: f* = edge / (1 - price).
    kelly = max(0.0, (win_prob - price) / (1 - price)) * risk.lambda_kelly
    return Decision(side, price, net_edge, kelly)


def realized_pnl(decision: Decision, outcome_yes: bool, risk: RiskParams) -> float:
    """Per-contract realized P&L (dollars), net of the entry taker fee."""
    if decision.side == "none":
        return 0.0
    won = outcome_yes if decision.side == "yes" else (not outcome_yes)
    fee = taker_fee(decision.exec_price, risk.fee_coeff)
    return (1 - decision.exec_price - fee) if won else (-decision.exec_price - fee)


# --------------------------------------------------------------------------- #
# Logging                                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class PaperRow:
    ts: str
    strategy: str
    ticker: str
    prob: float
    sigma: float
    yes_bid: float
    yes_ask: float
    n_draws: int


def _log_path(day: str) -> Path:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    return PAPER_DIR / f"signals_{day}.jsonl"


def snapshot(strategies, market_views) -> list[PaperRow]:
    """Run every strategy over every market view and append signal+price rows.

    ``market_views`` is an iterable of ``MarketView`` (the caller fetches them from
    Kalshi). Returns the rows logged this cycle.
    """
    rows: list[PaperRow] = []
    ts = datetime.now(timezone.utc).isoformat()
    day = ts[:10]
    for mv in market_views:
        yb, ya = mv.yes_bid, mv.yes_ask
        if yb is None or ya is None:
            continue
        for strat in strategies:
            sig = strat.evaluate(mv)
            if sig is None:
                continue
            rows.append(PaperRow(ts, sig.strategy, sig.ticker, sig.prob,
                                 sig.sigma, yb, ya, sig.n_draws))
    if rows:
        with _log_path(day).open("a") as f:
            for r in rows:
                f.write(json.dumps(asdict(r)) + "\n")
    return rows


def load_rows(paths) -> pd.DataFrame:
    recs = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                recs.append(json.loads(line))
    return pd.DataFrame(recs)


def score(rows: pd.DataFrame, outcomes: dict[str, bool],
          risk: RiskParams | None = None) -> pd.DataFrame:
    """Score logged signals against settled outcomes -> per-strategy P&L board.

    ``outcomes`` maps ticker -> True if the market resolved YES. Each row becomes a
    (possibly abstained) decision; we aggregate per-contract P&L, hit rate, and
    fractional-Kelly bankroll growth per strategy.
    """
    risk = risk or RiskParams()
    out = []
    for _, r in rows.iterrows():
        if r["ticker"] not in outcomes:
            continue
        dec = decide(r["prob"], r["sigma"], r["yes_bid"], r["yes_ask"], risk)
        if dec.side == "none":
            continue
        pnl = realized_pnl(dec, outcomes[r["ticker"]], risk)
        out.append({
            "strategy": r["strategy"],
            "ticker": r["ticker"],
            "side": dec.side,
            "kelly_frac": dec.kelly_frac,
            "pnl_per_contract": pnl,
            "won": pnl > 0,
            "kelly_pnl": dec.kelly_frac * pnl / max(dec.exec_price, 1e-6),
        })
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out)
    return df.groupby("strategy").agg(
        n_trades=("pnl_per_contract", "size"),
        hit_rate=("won", "mean"),
        pnl_per_contract=("pnl_per_contract", "mean"),
        total_pnl=("pnl_per_contract", "sum"),
        kelly_bankroll_growth=("kelly_pnl", "sum"),
    ).reset_index().sort_values("pnl_per_contract", ascending=False)
