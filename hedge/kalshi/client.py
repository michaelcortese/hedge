"""Minimal Kalshi REST client.

Wraps the `/trade-api/v2` endpoints the bot needs, signing every request with
`KalshiAuth`. Read-only market-data endpoints work without auth, but we sign
everything for consistency (and so a misconfigured key fails loudly early).

Base URLs (see CLAUDE.md for the full table):
    demo  https://demo-api.kalshi.co/trade-api/v2
    prod  https://api.elections.kalshi.com/trade-api/v2   (legacy, widely used)
    prod  https://external-api.kalshi.com/trade-api/v2    (newer, docs-preferred)
"""

from __future__ import annotations

from typing import Any

import requests

from hedge.kalshi.auth import KalshiAuth

DEMO_BASE = "https://demo-api.kalshi.co/trade-api/v2"
PROD_BASE = "https://api.elections.kalshi.com/trade-api/v2"
PROD_BASE_EXTERNAL = "https://external-api.kalshi.com/trade-api/v2"

# The path prefix that must be present in the SIGNED path, regardless of host.
_API_PREFIX = "/trade-api/v2"


class KalshiError(RuntimeError):
    """Raised when the Kalshi API returns a non-2xx response."""

    def __init__(self, status: int, body: str, method: str, path: str):
        self.status = status
        self.body = body
        super().__init__(f"{method} {path} -> HTTP {status}: {body[:500]}")


class KalshiClient:
    def __init__(
        self,
        auth: KalshiAuth,
        base_url: str = DEMO_BASE,
        timeout: float = 10.0,
        session: requests.Session | None = None,
    ):
        self.auth = auth
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    # ----- core request plumbing -------------------------------------------

    def _signed_path(self, path: str) -> str:
        """The path used both for the URL suffix and for signing.

        Accepts either a full "/trade-api/v2/..." path or a short "/markets"
        path and normalizes to include the prefix exactly once.
        """
        path = "/" + path.lstrip("/")
        if not path.startswith(_API_PREFIX):
            path = _API_PREFIX + path
        return path

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        signed_path = self._signed_path(path)
        url = self.base_url.replace(_API_PREFIX, "") + signed_path
        headers = self.auth.headers(method, signed_path)
        headers["Content-Type"] = "application/json"

        resp = self.session.request(
            method.upper(),
            url,
            params=params,
            json=json,
            headers=headers,
            timeout=self.timeout,
        )
        if not resp.ok:
            raise KalshiError(resp.status_code, resp.text, method, signed_path)
        if not resp.content:
            return {}
        return resp.json()

    # ----- market data ------------------------------------------------------

    def get_exchange_status(self) -> dict[str, Any]:
        """Exchange trading/up status. Cheap call — good for auth smoke tests."""
        return self._request("GET", "/exchange/status")

    def get_markets(self, **filters: Any) -> dict[str, Any]:
        """List markets. Filters: series_ticker, event_ticker, status, tickers,
        limit, cursor, etc."""
        return self._request("GET", "/markets", params=filters or None)

    def get_market(self, ticker: str) -> dict[str, Any]:
        return self._request("GET", f"/markets/{ticker}")

    def get_orderbook(self, ticker: str, depth: int | None = None) -> dict[str, Any]:
        params = {"depth": depth} if depth is not None else None
        return self._request("GET", f"/markets/{ticker}/orderbook", params=params)

    # ----- portfolio (auth required) ---------------------------------------

    def get_balance(self) -> dict[str, Any]:
        return self._request("GET", "/portfolio/balance")

    def get_positions(self, **filters: Any) -> dict[str, Any]:
        return self._request("GET", "/portfolio/positions", params=filters or None)

    def get_fills(self, **filters: Any) -> dict[str, Any]:
        return self._request("GET", "/portfolio/fills", params=filters or None)

    def get_orders(self, **filters: Any) -> dict[str, Any]:
        return self._request("GET", "/portfolio/orders", params=filters or None)

    def create_order(
        self,
        *,
        ticker: str,
        action: str,            # "buy" | "sell"
        side: str,              # "yes" | "no"
        count: int,
        type: str = "limit",    # "limit" | "market"
        yes_price: int | None = None,   # cents 1-99 (supply exactly one of
        no_price: int | None = None,    # yes_price / no_price for a limit order)
        client_order_id: str | None = None,
        time_in_force: str | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        """Place an order. Prices are integer cents (1-99).

        To bet AGAINST an outcome, buy the NO side (action="buy", side="no") —
        Kalshi has no margin short. action="sell" closes/reduces a position you
        already hold.
        """
        body: dict[str, Any] = {
            "ticker": ticker,
            "action": action,
            "side": side,
            "count": count,
            "type": type,
        }
        if yes_price is not None:
            body["yes_price"] = yes_price
        if no_price is not None:
            body["no_price"] = no_price
        if client_order_id is not None:
            body["client_order_id"] = client_order_id
        if time_in_force is not None:
            body["time_in_force"] = time_in_force
        body.update(extra)
        return self._request("POST", "/portfolio/orders", json=body)

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/portfolio/orders/{order_id}")
