"""Strategy base class — the interface every Monte Carlo algorithm implements.

The contract is intentionally tiny: given access to market data, return zero or
more `Signal`s. The framework calls `evaluate` on a schedule, collects the
signals, and routes them through the decision/sizing/execution pipeline.

You do NOT place orders, compute edge, or size positions in here. You only
produce probabilities. That separation is deliberate: it keeps every strategy
backtestable and lets the same risk engine govern all of them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence

from hedge.signal import Signal


class MarketView:
    """Read-only snapshot of a market handed to a strategy.

    Thin wrapper around the Kalshi `GET /markets/{ticker}` payload plus the
    order book, so strategies don't each re-learn the raw API shape. Prices are
    exposed in DOLLARS (0.01-0.99) for convenience; Kalshi's raw integer cents
    are still available under `.raw`.
    """

    def __init__(self, ticker: str, raw: dict, orderbook: dict | None = None):
        self.ticker = ticker
        self.raw = raw
        self.orderbook = orderbook or {}

    def _price(self, cents_key: str, dollars_key: str) -> float | None:
        """Read a price in dollars, supporting both Kalshi payload shapes.

        Older endpoints return integer cents (`yes_bid`); newer ones return a
        dollar string (`yes_bid_dollars`). Prefer cents, fall back to dollars, so
        strategies and the price logger see a consistent dollar value either way.
        """
        c = self.raw.get(cents_key)
        if c is not None:
            return c / 100
        d = self.raw.get(dollars_key)
        return float(d) if d is not None else None

    @property
    def yes_bid(self) -> float | None:
        return self._price("yes_bid", "yes_bid_dollars")

    @property
    def yes_ask(self) -> float | None:
        return self._price("yes_ask", "yes_ask_dollars")

    @property
    def last_price(self) -> float | None:
        return self._price("last_price", "last_price_dollars")

    @property
    def mid(self) -> float | None:
        b, a = self.yes_bid, self.yes_ask
        if b is not None and a is not None:
            return (b + a) / 2
        return self.last_price


class Strategy(ABC):
    """Subclass this for each Monte Carlo algorithm.

    Minimal example::

        class MyStrategy(Strategy):
            name = "my_strategy"

            def universe(self) -> list[str]:
                return ["KXFED-26MAR19-T3.00"]

            def evaluate(self, market: MarketView) -> Signal | None:
                p = self.run_monte_carlo(market)   # your code
                return Signal(market.ticker, prob=p, n_draws=10_000,
                              strategy=self.name)
    """

    #: Unique, stable identifier used in logs and signal attribution.
    name: str = "unnamed"

    def universe(self) -> Sequence[str]:
        """Return the market tickers this strategy wants to evaluate.

        Override to point at your markets. Return an empty list to opt out of a
        cycle. The framework fetches market data for these tickers and calls
        `evaluate` once per ticker.
        """
        return []

    @abstractmethod
    def evaluate(self, market: MarketView) -> Signal | None:
        """Return a Signal for this market, or None to abstain.

        Called once per ticker in `universe()` per cycle. Returning None (or a
        signal whose edge is below threshold) means "no opinion / do nothing".
        Must be pure-ish: no order placement, no global state mutation that the
        backtester can't reproduce.
        """
        raise NotImplementedError

    def evaluate_all(self, markets: Iterable[MarketView]) -> list[Signal]:
        """Default fan-out over markets. Override only if you need cross-market
        logic (e.g. correlated baskets evaluated jointly)."""
        out: list[Signal] = []
        for m in markets:
            sig = self.evaluate(m)
            if sig is not None:
                out.append(sig)
        return out
