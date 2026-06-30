"""The Signal contract — the ONLY thing a Monte Carlo strategy must produce.

A strategy looks at a Kalshi market and emits its estimated probability that the
market resolves YES. Everything downstream (edge calc, Kelly sizing, order
placement) is generic and lives outside the strategy. Keep your strategy code
focused on producing a well-calibrated `prob`; the bot handles the rest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Signal:
    """A single strategy's opinion about one Kalshi market.

    Attributes:
        ticker:   The Kalshi market ticker this opinion is about, e.g.
                  "KXFED-26MAR19-T3.00". Must match a real, tradable market.
        prob:     Your model's probability that the market resolves YES.
                  Must be in the open interval (0, 1).
        n_draws:  Number of independent Monte Carlo draws behind `prob`. Used to
                  derive sampling standard error. If your method is not sampling
                  based (closed form, etc.), pass a large number (e.g. 1_000_000)
                  to signal high confidence, OR set `std_error` directly.
        std_error: Optional explicit standard error of `prob`. If None, it is
                  derived from `n_draws` as sqrt(p*(1-p)/n_draws). Set this
                  directly if you have a better estimate of your own uncertainty
                  (e.g. you've folded in model/structural error, not just
                  sampling noise).
        strategy: Name of the strategy that produced this signal (for logging /
                  attribution). Defaults to "unknown".
        meta:     Free-form dict for anything you want to log alongside the
                  signal (intermediate values, scenario breakdowns, etc.). Never
                  read by the decision engine.
        deterministic: The outcome is LOGICALLY settled, not a probabilistic
                  estimate — e.g. the observed max-so-far already exceeds a
                  bucket's upper bound, so YES is impossible (``prob≈0``), or an
                  "X or above" threshold is already met (``prob≈1``). The decision
                  engine may bypass the tradeable-price *band* for such a signal
                  (a near-certain NO sits in the rich tail the band normally
                  blocks), while ALL other guards — significance, fees, depth,
                  participation, per-event/portfolio caps, station validation —
                  still apply. Only set this when the outcome is genuinely
                  determined by observation, never for a merely-confident estimate.
    """

    ticker: str
    prob: float
    n_draws: int = 1_000_000
    std_error: float | None = None
    strategy: str = "unknown"
    meta: dict[str, Any] = field(default_factory=dict)
    deterministic: bool = False

    def __post_init__(self) -> None:
        if not (0.0 < self.prob < 1.0):
            raise ValueError(
                f"prob must be in (0, 1), got {self.prob!r} for {self.ticker!r}"
            )
        if self.n_draws < 1:
            raise ValueError(f"n_draws must be >= 1, got {self.n_draws!r}")
        if self.std_error is not None and self.std_error < 0:
            raise ValueError(f"std_error must be >= 0, got {self.std_error!r}")

    @property
    def sigma(self) -> float:
        """Standard error of `prob`.

        Uses the explicit `std_error` if provided, otherwise the Monte Carlo
        sampling error sqrt(p*(1-p)/n_draws).
        """
        if self.std_error is not None:
            return self.std_error
        return math.sqrt(self.prob * (1.0 - self.prob) / self.n_draws)
