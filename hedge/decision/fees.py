"""Kalshi trading fees.

General taker formula (per order, in dollars), rounded UP to the next cent:

    fee = ceil( 0.07 * C * P * (1 - P) )        # C = contracts, P = price in $

Per contract this is ~0.07 * P * (1-P), maximized at P = 0.50 -> ~1.75 cents.
Maker (resting limit) orders are usually free on standard markets, but some
high-volume / special-event series apply a maker fee; we model it at 1/4 of the
taker coefficient by default and let callers override.

IMPORTANT: the 0.07 coefficient is NOT universal — some series use different
coefficients. For production edge math, key the coefficient per market from the
official Fee Schedule PDF rather than trusting this default everywhere.
"""

from __future__ import annotations

import math

TAKER_COEF = 0.07
MAKER_COEF = 0.0175  # 1/4 of taker; many markets are actually 0 (free maker)


def fee_per_contract(price: float, *, maker: bool = False, coef: float | None = None) -> float:
    """Continuous per-contract fee in dollars (no integer-cent rounding).

    Use this for edge thresholds and sizing, where the smooth value is what you
    want. `price` is in dollars (0.01-0.99).
    """
    c = coef if coef is not None else (MAKER_COEF if maker else TAKER_COEF)
    return c * price * (1.0 - price)


def order_fee(price: float, count: int, *, maker: bool = False, coef: float | None = None) -> float:
    """Actual order-level fee in dollars, rounded UP to the next cent.

    This matches how Kalshi bills a fill. Use for P&L accounting; use
    `fee_per_contract` for the decision math.
    """
    c = coef if coef is not None else (MAKER_COEF if maker else TAKER_COEF)
    raw = c * count * price * (1.0 - price)
    return math.ceil(raw * 100) / 100.0
