"""Edge: expected value per contract on a binary $1/$0 contract.

For a YES contract bought at price q (dollars) when the true P(YES) is p:
    win  (prob p):     contract -> $1, you paid q  => profit (1 - q)
    lose (prob 1-p):   contract -> $0              => profit -q
    EV_yes = p*(1-q) + (1-p)*(-q) = p - q

By symmetry the NO contract (price 1-q, pays $1 if NO):
    EV_no = q - p = -EV_yes

So "edge" = p - q, and you take whichever side your probability says is
underpriced. These are gross (frictionless); see net_edge for the fee-aware
version used to decide whether to trade.
"""

from __future__ import annotations

from hedge.decision.fees import fee_per_contract


def ev_yes(p: float, yes_price: float) -> float:
    """Expected profit per YES contract (dollars). Equals p - yes_price."""
    return p - yes_price


def ev_no(p: float, no_price: float) -> float:
    """Expected profit per NO contract (dollars).

    NO pays $1 when the event resolves NO (prob 1-p). EV = (1-p) - no_price.
    Note no_price = 1 - yes_price, so this equals yes_price - p.
    """
    return (1.0 - p) - no_price


def net_edge(win_prob: float, exec_price: float, *, maker: bool, coef: float | None = None) -> float:
    """Fee-aware expected profit per contract for the chosen side, in dollars.

    `win_prob` is the probability the side you're buying wins (p for YES,
    1-p for NO). `exec_price` is the price you actually pay for that side. Held
    to settlement there is no exit fee, so only the entry fee is subtracted.

        net_edge = win_prob - exec_price - fee(exec_price)
    """
    return win_prob - exec_price - fee_per_contract(exec_price, maker=maker, coef=coef)
