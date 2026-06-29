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
        auth: KalshiAuth | None = None,
        base_url: str = DEMO_BASE,
        timeout: float = 10.0,
        session: requests.Session | None = None,
    ):
        # ``auth=None`` is a READ-ONLY client: Kalshi's market-data endpoints are
        # public, so we can capture live prod prices with no key. Any portfolio/order
        # call (which needs signing) raises instead of sending an unsigned request.
        self.auth = auth
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    @classmethod
    def read_only(cls, base_url: str = PROD_BASE, **kw: Any) -> "KalshiClient":
        """A keyless client for public market data (defaults to prod)."""
        return cls(auth=None, base_url=base_url, **kw)

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
        if self.auth is None:
            if method.upper() != "GET":
                raise KalshiError(0, "read-only client cannot place/modify orders "
                                  "(build with credentials to trade)", method, signed_path)
            headers = {}
        else:
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
        """List orders. Filters: ticker, status (resting|canceled|executed), limit, cursor.

        Reads stay on ``/portfolio/orders`` (still live); only create/cancel moved to
        the V2 ``/portfolio/events/orders`` family.
        """
        return self._request("GET", "/portfolio/orders", params=filters or None)

    def get_order(self, order_id: str) -> dict[str, Any]:
        """Fetch a single order's current state (status + fill counts) by id."""
        return self._request("GET", f"/portfolio/orders/{order_id}")

    def create_order(self, **body: Any) -> dict[str, Any]:
        """Place an order via the Kalshi **V2** endpoint (``/portfolio/events/orders``).

        The V1 ``/portfolio/orders`` create endpoint is deprecated (HTTP 410). The
        V2 model is YES-priced and side-only (``side`` bid/ask, single ``price`` in
        dollars, ``time_in_force``); the body is built by
        :func:`hedge.execution.executor.build_order_body`. Always include a
        ``client_order_id`` (UUID) for idempotency — Kalshi rejects duplicates.
        """
        return self._request("POST", "/portfolio/events/orders", json=body)

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        """Cancel a resting order by id (V2)."""
        return self._request("DELETE", f"/portfolio/events/orders/{order_id}")
