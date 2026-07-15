"""Minimal Kalshi PERPS (margin) API client — public market-data endpoints.

The perps surface lives under ``/trade-api/v2/margin/*`` on a different host
than event contracts:

    prod  https://external-api.kalshi.com/trade-api/v2
    demo  https://external-api.demo.kalshi.co/trade-api/v2

Market data (markets, orderbook, candlesticks, trades, funding rates) is
public. Order/portfolio endpoints need the same RSA-PSS signing as event
contracts (reuse ``KalshiAuth``) — not implemented here yet; this module is
read-only on purpose. OpenAPI spec: https://docs.kalshi.com/perps_openapi.yaml

Fees (fee-schedule PDF, 7.7.26): taker 12bps / maker 5bps at base tier,
tiered by 30-day volume. Funding: 3x/day (12am/8am/4pm ET), 8h TWAP of 1-min
premiums, deadband |r| < 0.01% -> 0, cap ±2%.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any

PERPS_PROD_BASE = "https://external-api.kalshi.com/trade-api/v2"
PERPS_DEMO_BASE = "https://external-api.demo.kalshi.co/trade-api/v2"

_UA = {"User-Agent": "hedge-perps/0.1"}


class PerpsClient:
    """Unauthenticated read-only client for the perps market-data endpoints."""

    def __init__(self, base_url: str = PERPS_PROD_BASE, timeout: float = 15.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _get(self, path: str) -> dict[str, Any]:
        req = urllib.request.Request(self.base_url + path, headers=_UA)
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)

    def markets(self) -> list[dict[str, Any]]:
        """All perp markets. Prices are dollars-per-contract strings; divide by
        float(contract_size) for the per-coin price."""
        return self._get("/margin/markets")["markets"]

    def market(self, ticker: str) -> dict[str, Any]:
        return self._get(f"/margin/markets/{ticker}")["market"]

    def orderbook(self, ticker: str, depth: int = 10) -> dict[str, Any]:
        return self._get(f"/margin/markets/{ticker}/orderbook?depth={depth}")

    def candlesticks(
        self, ticker: str, start_ts: int, end_ts: int, period_interval: int = 1
    ) -> list[dict[str, Any]]:
        """1/60/1440-minute candles. Candle ``end_period_ts`` labels the END of
        the interval (Coinbase labels the START — mind the off-by-one when
        aligning cross-venue series)."""
        d = self._get(
            f"/margin/markets/{ticker}/candlesticks"
            f"?start_ts={start_ts}&end_ts={end_ts}&period_interval={period_interval}"
        )
        return d.get("candlesticks", [])

    def funding_estimate(self, ticker: str) -> dict[str, Any]:
        return self._get(f"/margin/funding_rates/estimate?ticker={ticker}")

    def funding_history(self, ticker: str, limit: int = 100, cursor: str = "") -> dict[str, Any]:
        path = f"/margin/funding_rates/historical?ticker={ticker}&limit={limit}"
        if cursor:
            path += f"&cursor={cursor}"
        return self._get(path)
