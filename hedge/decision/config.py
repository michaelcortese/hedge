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
    market_cap_frac: float = 0.03   # max bankroll fraction at risk per market
    portfolio_cap: float = 0.30     # max total bankroll fraction at risk
    rebalance_band: float = 0.25    # only rebalance if target drifts > this frac

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
