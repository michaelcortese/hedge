"""Decision engine: Signal + market quote -> {action, side, price, count}.

Pure functions, no I/O. Given a strategy's `Signal`, the current market quote,
the bankroll, and risk config, decide what order (if any) to place. The pipeline:

    1. Add model uncertainty in quadrature to the signal's sampling error.
    2. (Optional) shrink the probability toward the market mid (Bayesian prior).
    3. Pick the side whose price the probability says is underpriced.
    4. Significance gate: skip unless |p - mid| > k_sigma * sigma.
    5. Choose execution price: prefer maker (post at bid), fall back to taker
       (cross at ask) — whichever clears the minimum net edge after fees.
    6. Size with fractional Kelly on the conservative (CI-bounded) net edge.
    7. Apply per-market, portfolio, and order-book-depth caps.
    8. Reconcile against any existing position (rebalance band / flip / hold).

See the formulas in edge.py, sizing.py, fees.py.
"""

from hedge.decision.config import RiskConfig
from hedge.decision.engine import (
    Action,
    Decision,
    MarketQuote,
    Position,
    Side,
    decide,
)

__all__ = [
    "Action",
    "Decision",
    "MarketQuote",
    "Position",
    "RiskConfig",
    "Side",
    "decide",
]
