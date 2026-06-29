"""The decision engine: Signal + market quote -> a concrete order (or nothing).

This is the single place that turns a strategy's probability into an action. It is
deliberately pure (no I/O, no Kalshi calls) so the same code grades a backtest,
scores paper trades, and drives live execution. The pipeline mirrors the steps
documented in ``decision/__init__.py``:

    1. Total uncertainty: sigma = sqrt(signal.sigma**2 + sigma_model**2).
    2. Optional Bayesian shrinkage of the probability toward the market mid.
    3. Side selection: take whichever side (YES/NO) the probability says is
       underpriced, by net (fee-aware) taker edge.
    4. Significance gate: abstain unless |p - mid| > k_sigma * sigma.
    5. Execution price: prefer maker (post at the bid), fall back to taker (cross
       at the ask) — pick the cheapest fill that still clears tau_min net edge.
    6. Size: fractional Kelly on a CONSERVATIVE (CI-lower-bounded, fee-netted)
       edge, so a noisy signal is bet smaller.
    7. Caps: per-market, whole-portfolio, and order-book-depth limits.
    8. Reconcile against any existing position (add / reduce / flip / hold).

Prices are in DOLLARS (0.01-0.99) throughout; the integer-cent value Kalshi wants
is precomputed onto the Decision for the execution layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from hedge.decision.config import RiskConfig
from hedge.decision.edge import net_edge
from hedge.decision.sizing import kelly_fraction_from_edge
from hedge.signal import Signal


class Side(str, Enum):
    """Which side of the binary market we hold/trade exposure to."""

    YES = "yes"
    NO = "no"

    @property
    def other(self) -> "Side":
        return Side.NO if self is Side.YES else Side.YES


class Action(str, Enum):
    """What the engine wants to do this cycle."""

    BUY = "buy"      # open or add exposure to Decision.side
    SELL = "sell"    # reduce/close existing exposure (Kalshi sell closes only)
    HOLD = "hold"    # do nothing


# --------------------------------------------------------------------------- #
# Inputs                                                                       #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MarketQuote:
    """Top-of-book for one market, in dollars.

    Only the YES bid/ask are required; NO prices are derived (a NO contract is
    the complement of YES, so ``no_price = 1 - yes_price``). Optional ``*_depth``
    fields are the size resting at the best price, used for the depth cap; leave
    them None to disable that cap.
    """

    yes_bid: float
    yes_ask: float
    yes_bid_depth: int | None = None   # contracts available to buy YES as maker join
    yes_ask_depth: int | None = None   # contracts available to take (buy YES) now
    no_bid_depth: int | None = None
    no_ask_depth: int | None = None

    @property
    def mid(self) -> float:
        return (self.yes_bid + self.yes_ask) / 2.0

    # NO book is the mirror of the YES book (see CLAUDE.md orderbook note):
    #   buy NO as taker  -> no_ask = 1 - yes_bid
    #   buy NO as maker   -> no_bid = 1 - yes_ask
    @property
    def no_ask(self) -> float:
        return 1.0 - self.yes_bid

    @property
    def no_bid(self) -> float:
        return 1.0 - self.yes_ask

    @classmethod
    def from_view(cls, view: Any) -> "MarketQuote | None":
        """Build from a ``MarketView`` (or anything exposing yes_bid/yes_ask).

        Returns None if either side of the book is missing — you can't price a
        trade without a two-sided market.
        """
        yb, ya = view.yes_bid, view.yes_ask
        if yb is None or ya is None:
            return None
        return cls(yes_bid=float(yb), yes_ask=float(ya))


@dataclass(frozen=True)
class Position:
    """An existing holding in one market. ``count`` contracts of ``side``."""

    side: Side
    count: int
    avg_price: float = 0.0


# --------------------------------------------------------------------------- #
# Output                                                                        #
# --------------------------------------------------------------------------- #
@dataclass
class Decision:
    """The engine's verdict for one market. ``action == HOLD`` means do nothing."""

    ticker: str
    action: Action
    side: Side | None = None
    price: float = 0.0            # exec price for ``side`` in dollars
    price_cents: int = 0          # same price as integer cents (for the order body)
    count: int = 0                # contracts to trade
    maker: bool = False           # True = post at the bid; False = cross the spread
    edge: float = 0.0             # net edge per contract used for sizing (dollars)
    kelly_fraction: float = 0.0   # fractional-Kelly bankroll share applied
    prob: float = 0.0             # uncertainty/shrinkage-adjusted P(YES) used
    sigma: float = 0.0            # total sigma used (sampling + model, post-shrink)
    reason: str = ""              # human-readable why (esp. for HOLD)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def is_trade(self) -> bool:
        return self.action is not Action.HOLD and self.count > 0


def _hold(ticker: str, reason: str, **extra: Any) -> Decision:
    return Decision(ticker=ticker, action=Action.HOLD, reason=reason, **extra)


def _to_cents(price: float) -> int:
    """Dollars -> integer cents, clamped to the tradable 1..99 range."""
    return max(1, min(99, round(price * 100)))


def _tradeable(price: float) -> bool:
    """A quote is fillable only strictly inside (0, 1) dollars.

    A price of 0.00/1.00 is the API saying "no resting size on that side" — NOT a
    free or certain fill. Taking it literally turns a one-sided/locked book into a
    phantom edge that sizing bets into (or crashes Kelly at price 0). Mirrors
    ``hedge.tournament.paper._tradeable`` so the live path and the paper scorer agree.
    """
    return 0.0 < price < 1.0


# --------------------------------------------------------------------------- #
# The pipeline                                                                  #
# --------------------------------------------------------------------------- #
def _shrink(p: float, sigma: float, mid: float, cfg: RiskConfig) -> tuple[float, float]:
    """Optional precision-weighted shrinkage of (p, sigma) toward the market mid.

    Treats the market mid as a prior with std ``sigma_market``; combines it with
    the model estimate by inverse-variance weighting. Smaller ``sigma_market`` =>
    trust the market more => shrink harder. Returns the posterior (mean, std).
    """
    if not cfg.shrink_to_market or not cfg.sigma_market:
        return p, sigma
    if sigma <= 0:  # a zero-sigma model is already infinitely confident; leave it
        return p, sigma
    w_model = 1.0 / (sigma * sigma)
    w_mkt = 1.0 / (cfg.sigma_market * cfg.sigma_market)
    post_var = 1.0 / (w_model + w_mkt)
    post_mean = post_var * (p * w_model + mid * w_mkt)
    return post_mean, math.sqrt(post_var)


def _net_edge_at(win_prob: float, price: float, *, maker: bool, cfg: RiskConfig) -> float:
    coef = cfg.maker_fee_coef if maker else cfg.taker_fee_coef
    return net_edge(win_prob, price, maker=maker, coef=coef)


def decide(
    signal: Signal,
    quote: MarketQuote,
    bankroll: float,
    cfg: RiskConfig | None = None,
    *,
    position: Position | None = None,
    portfolio_at_risk: float = 0.0,
) -> Decision:
    """Turn one ``Signal`` + ``MarketQuote`` into an order decision.

    Args:
        signal: the strategy's opinion (gives P(YES) and its std error).
        quote: current top-of-book for the same market.
        bankroll: total bankroll in dollars (sizing is a fraction of this).
        cfg: risk knobs; defaults to a conservative ``RiskConfig()``.
        position: existing holding in this market, if any (for reconciliation).
        portfolio_at_risk: dollars already at risk across all markets, for the
            portfolio cap. The order is shrunk so the total stays within
            ``portfolio_cap * bankroll``.

    Returns a ``Decision``. ``action == HOLD`` (with a ``reason``) when the engine
    declines to trade.
    """
    cfg = cfg or RiskConfig()
    ticker = signal.ticker

    if bankroll <= 0:
        return _hold(ticker, "non-positive bankroll")

    # (1) Total uncertainty: fold model error into the signal's sampling error.
    sigma = math.hypot(signal.sigma, cfg.sigma_model)
    p = signal.prob
    mid = quote.mid

    # (2) Optional shrinkage toward the market.
    p, sigma = _shrink(p, sigma, mid, cfg)

    base = dict(prob=p, sigma=sigma)

    # (3) Side selection by net taker edge (the realistically executable edge).
    # A side is only considered when its taker exec price is genuinely tradeable;
    # a degenerate 0.00/1.00 quote is not a fill (see _tradeable).
    yes_taker = (_net_edge_at(p, quote.yes_ask, maker=False, cfg=cfg)
                 if _tradeable(quote.yes_ask) else float("-inf"))
    no_taker = (_net_edge_at(1.0 - p, quote.no_ask, maker=False, cfg=cfg)
                if _tradeable(quote.no_ask) else float("-inf"))
    if max(yes_taker, no_taker) <= 0:
        return _hold(ticker, "no tradeable side with positive net edge", **base)
    side = Side.YES if yes_taker >= no_taker else Side.NO
    win_prob = p if side is Side.YES else 1.0 - p

    # (4) Significance gate: the disagreement with the market must beat the noise.
    if abs(p - mid) <= cfg.k_sigma * sigma:
        return _hold(
            ticker,
            f"edge within noise: |p-mid|={abs(p - mid):.3f} <= "
            f"{cfg.k_sigma}*sigma={cfg.k_sigma * sigma:.3f}",
            **base,
        )

    # (5) Execution price: prefer maker (cheaper, lower fee), fall back to taker.
    maker_price = quote.yes_bid if side is Side.YES else quote.no_bid
    taker_price = quote.yes_ask if side is Side.YES else quote.no_ask
    # Only consider a fill price that is actually tradeable — a missing bid must not
    # become a phantom maker buy at ~$0.00 (which _to_cents would clamp to 1¢).
    maker_edge = (_net_edge_at(win_prob, maker_price, maker=True, cfg=cfg)
                  if _tradeable(maker_price) else float("-inf"))
    taker_edge = (_net_edge_at(win_prob, taker_price, maker=False, cfg=cfg)
                  if _tradeable(taker_price) else float("-inf"))
    if maker_edge >= cfg.tau_min:
        maker, exec_price, gross_edge = True, maker_price, maker_edge
    elif taker_edge >= cfg.tau_min:
        maker, exec_price, gross_edge = False, taker_price, taker_edge
    else:
        return _hold(
            ticker,
            f"net edge below tau_min: maker={maker_edge:.4f} taker={taker_edge:.4f} "
            f"< {cfg.tau_min:.4f}",
            **base,
        )

    # (6) Size on a CONSERVATIVE edge: shade the win prob down by z_ci*sigma, then
    # net fees at the chosen price. This stacks a CI haircut on top of fractional
    # Kelly so a noisy signal is bet smaller (or not at all).
    cons_win = win_prob - cfg.z_ci * sigma
    cons_edge = _net_edge_at(cons_win, exec_price, maker=maker, cfg=cfg)
    if cons_edge <= 0:
        return _hold(
            ticker,
            f"conservative edge non-positive after {cfg.z_ci}-sigma haircut",
            **base,
        )
    f = kelly_fraction_from_edge(cons_edge, exec_price) * cfg.lambda_kelly
    target_capital = f * bankroll

    # (7) Caps. Each is a ceiling on dollars-at-risk; take the binding one.
    market_capital = cfg.market_cap_frac * bankroll
    portfolio_room = max(0.0, cfg.portfolio_cap * bankroll - portfolio_at_risk)
    abs_cap = cfg.max_order_dollars if cfg.max_order_dollars is not None else math.inf
    capital = min(target_capital, market_capital, portfolio_room, abs_cap)
    if capital <= 0:
        return _hold(ticker, "portfolio cap leaves no room", edge=cons_edge, **base)

    count = int(math.floor(capital / exec_price))

    # Order-book-depth cap: never size past what's resting at our price.
    depth = (
        (quote.yes_bid_depth if maker else quote.yes_ask_depth)
        if side is Side.YES
        else (quote.no_bid_depth if maker else quote.no_ask_depth)
    )
    if depth is not None:
        count = min(count, int(depth))

    if count < 1:
        return _hold(ticker, "sized below one contract", edge=cons_edge, **base)

    want = Decision(
        ticker=ticker,
        action=Action.BUY,
        side=side,
        price=exec_price,
        price_cents=_to_cents(exec_price),
        count=count,
        maker=maker,
        edge=cons_edge,
        kelly_fraction=f,
        reason="open" if position is None else "add",
        **base,
    )

    # (8) Reconcile against any existing position.
    return _reconcile(want, position, quote, cfg)


def _exit_price(side: Side, quote: MarketQuote) -> float:
    """Price you receive selling (closing) a ``side`` position: cross to the bid."""
    return quote.yes_bid if side is Side.YES else quote.no_bid


def _reconcile(
    want: Decision, position: Position | None, quote: MarketQuote, cfg: RiskConfig
) -> Decision:
    """Fold the desired target into the current holding.

    - Flat, or holding the same side: buy the shortfall to the target, but only if
      it drifts more than ``rebalance_band`` (avoids churn/fees on tiny deltas);
      if we already hold MORE than the target, sell the excess.
    - Holding the opposite side: close it (SELL to flat). We don't open the new
      side in the same Decision — the next cycle, now flat, opens cleanly. This
      keeps each Decision a single atomic order. Sells cross to the bid.
    """
    if position is None or position.count == 0:
        return want

    if position.side is want.side:
        target = want.count
        drift = (target - position.count) / max(position.count, 1)
        if abs(drift) <= cfg.rebalance_band:
            return _hold(
                want.ticker,
                f"within rebalance band ({drift:+.2f}); holding {position.count}",
                side=want.side, prob=want.prob, sigma=want.sigma, edge=want.edge,
            )
        if target > position.count:
            want.count = target - position.count
            want.reason = f"add {want.count} toward target {target}"
            return want
        # target < current: trim the excess (sell reduces on Kalshi).
        exit_px = _exit_price(want.side, quote)
        return Decision(
            ticker=want.ticker, action=Action.SELL, side=want.side,
            price=exit_px, price_cents=_to_cents(exit_px),
            count=position.count - target, maker=False,
            prob=want.prob, sigma=want.sigma, edge=want.edge,
            reason=f"trim to target {target}",
        )

    # Opposite side held -> close it first (sell the held side at its bid).
    exit_px = _exit_price(position.side, quote)
    return Decision(
        ticker=want.ticker, action=Action.SELL, side=position.side,
        price=exit_px, price_cents=_to_cents(exit_px),
        count=position.count, maker=False,
        prob=want.prob, sigma=want.sigma, edge=want.edge,
        reason=f"flip: close {position.count} {position.side.value} before opening {want.side.value}",
    )
