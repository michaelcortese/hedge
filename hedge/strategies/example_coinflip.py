"""A trivial reference strategy showing the plug-and-play shape.

Copy this file, rename the class, and replace `evaluate` with your real Monte
Carlo. This one just hard-codes a fair-coin belief — it exists so new agents can
see a complete, runnable example, NOT because it makes money.
"""

from __future__ import annotations

from hedge.signal import Signal
from hedge.strategies.base import MarketView, Strategy


class CoinflipStrategy(Strategy):
    name = "coinflip_example"

    def universe(self) -> list[str]:
        # Replace with the markets your model actually covers.
        return []

    def evaluate(self, market: MarketView) -> Signal | None:
        # Your Monte Carlo goes here. It must end by returning a probability of
        # the market resolving YES. This stub always believes it's a fair coin.
        p = 0.50
        return Signal(
            ticker=market.ticker,
            prob=p,
            n_draws=10_000,
            strategy=self.name,
            meta={"note": "reference stub — not a real edge"},
        )
