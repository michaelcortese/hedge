"""Risk / sizing configuration for the decision engine."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskConfig:
    """Knobs governing how aggressively the engine trades.

    Defaults are deliberately conservative — quarter-Kelly, a 2-sigma
    significance gate, and a 2-cent minimum net edge.
    """

    lambda_kelly: float = 0.25      # fraction of full Kelly to bet (0.25-0.5)
    k_sigma: float = 2.0            # require |p - mid| > k_sigma * sigma to act
    z_ci: float = 1.0              # conservative-edge CI haircut: use p ∓ z*sigma
    tau_min_cents: float = 2.0      # minimum net edge (cents) to trade
    market_cap_frac: float = 0.03   # max bankroll fraction at risk per market (one bucket)
    portfolio_cap: float = 0.30     # max total bankroll fraction at risk
    rebalance_band: float = 0.25    # only rebalance if target drifts > this frac

    # Per-EVENT (city-day) concentration cap. All buckets of one "high in city X on
    # day D" event are mutually exclusive outcomes of ONE synoptic draw, so they are
    # highly correlated: a biased forecast center misses them together (the 2026-06-29
    # failure — ~11 correlated markets, 10 losses). market_cap_frac alone lets each
    # bucket take its full share, so a whole event could consume many market-caps of
    # correlated risk. This caps total dollars-at-risk across ALL buckets of one event.
    event_cap_frac: float = 0.06    # max bankroll fraction at risk per (series, day)

    # Order-book participation cap: never take more than this fraction of the size
    # resting at our price in a single order (None = take the whole resting level, the
    # legacy behavior). Caps market impact / book-walking and avoids signalling size on
    # thin retail books. Requires the order book to be fetched (depth fields populated).
    participation_frac: float | None = None

    # Tradeable-price band: refuse to OPEN a position whose execution price falls
    # outside [min_price, max_price] (dollars). Extreme contracts are where the
    # model is least calibrated (tail estimation), where fees are the largest
    # fraction of price, and — critically — where there is no bid to exit into, so
    # a losing penny contract can only ride to 0. Keeps the engine out of 1-9c
    # long-shot junk (and the rich-side mirror). Gates opens only; an existing
    # holding can still be trimmed/closed outside the band.
    min_price: float = 0.10
    max_price: float = 0.90

    # Intraday position management (trim / add / flip-to-exit existing holdings).
    # OFF by default. When True the engine reconciles live positions (trim/add/flip);
    # realized P&L from intraday closes is booked into the daily-loss stop and an
    # anti-churn cooldown (below) damps flip-flop. Enable only once the strategy's edge
    # is proven on realized, market-priced, fee-net P&L. When False the runner only
    # OPENS new positions (and reads holdings for the portfolio cap); never trims/flips.
    manage_positions: bool = False

    # Anti-churn: after a management trade (trim/add/flip-to-exit) on a ticker, do not
    # act on that ticker again for this many cycles. Prevents a noisy signal from
    # sell->rebuy->sell flip-flopping around a boundary and bleeding taker fees, and
    # stops a duplicate exit being re-sent before the close shows up in positions().
    # Only applies to managed (existing-position) tickers, never to a fresh open.
    manage_cooldown_cycles: int = 2

    # Hard absolute ceiling on dollars-at-risk per order (None = no absolute cap,
    # fractions still apply). A backstop for early live trading so a mis-read
    # bankroll can't size large — caps in dollars, not just bankroll fraction.
    max_order_dollars: float | None = None

    # Hard stop on realized losses within a single UTC day (None = no daily stop).
    # When the day's realized P&L drops below -daily_loss_stop_dollars the runner
    # latches a halt for the rest of the UTC day. Backstop against a bad run.
    daily_loss_stop_dollars: float | None = None

    # Extra model/structural uncertainty added in quadrature to the signal's
    # sampling std error. 0 = trust the signal's own sigma.
    sigma_model: float = 0.0

    # Optional Bayesian shrinkage of the model prob toward the market mid.
    # If sigma_market is None, shrinkage is disabled. Smaller sigma_market =>
    # trust the market more => shrink harder toward it.
    shrink_to_market: bool = False
    sigma_market: float | None = None

    # Maker-fee coefficient override (None = library default). Set per-market in
    # production from the official Fee Schedule.
    maker_fee_coef: float | None = None
    taker_fee_coef: float | None = None

    @property
    def tau_min(self) -> float:
        """Minimum net edge in dollars."""
        return self.tau_min_cents / 100.0

    @classmethod
    def from_dict(cls, d: dict) -> RiskConfig:
        """Build from a config.yaml `risk:` mapping, ignoring unknown keys."""
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in fields})
