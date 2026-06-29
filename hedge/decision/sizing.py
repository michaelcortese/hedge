"""Position sizing: the Kelly criterion for binary contracts.

Buying one contract of a side at price `c` risks `c` and pays `(1 - c)` profit on
a win (win prob `w`). The Kelly fraction of bankroll to put at risk is:

    f* = (w - c) / (1 - c)

Derivation: maximize E[log wealth] = w*ln(1 + f*b) + (1-w)*ln(1 - f) with odds
b = (1-c)/c. Setting g'(f)=0 gives f* = (w*b - (1-w))/b = (w - c)/(1 - c).

Checks: f*=0 when w=c (no edge); f*=1 when w=1 (certain win). For a YES buy
(c=q, w=p) this is the familiar (p - q)/(1 - q); for a NO buy (c=1-q, w=1-p) it
reduces to (q - p)/q.

Practitioners bet a FRACTION lambda of full Kelly (0.25-0.5) because Kelly is
extremely sensitive to overestimating edge: betting above full Kelly cuts growth
and sharply raises ruin risk, and `w` is always estimated, never known. We also
feed a CONSERVATIVE (CI-lower-bounded, fee-netted) edge into the numerator, so
two safety margins stack — intentional for an automated bot.
"""

from __future__ import annotations

import math


def kelly_fraction(win_prob: float, price: float) -> float:
    """Full-Kelly fraction of bankroll to risk on a side at `price`.

    f* = (win_prob - price) / (1 - price). Clamped at 0 (never bet a negative
    edge). `price` must be < 1.
    """
    if not (0.0 < price < 1.0):
        raise ValueError(f"price must be in (0, 1), got {price!r}")
    f = (win_prob - price) / (1.0 - price)
    return max(f, 0.0)


def kelly_fraction_from_edge(net_edge_value: float, price: float) -> float:
    """Kelly fraction using an already-computed (conservative, net) edge.

    Equivalent to kelly_fraction but lets the caller pass a fee-/uncertainty-
    adjusted edge as the numerator: f = net_edge / (1 - price).
    """
    if not (0.0 < price < 1.0):
        raise ValueError(f"price must be in (0, 1), got {price!r}")
    return max(net_edge_value / (1.0 - price), 0.0)


def contracts_for_capital(capital: float, price: float) -> int:
    """How many whole contracts a capital budget buys at `price` (per-contract
    cost = price dollars). Floors to a whole contract."""
    if price <= 0:
        return 0
    return int(math.floor(capital / price))
