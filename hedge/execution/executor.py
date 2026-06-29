"""Decision -> signed Kalshi order, with idempotency and safety guards.

The decision engine has already decided *what* to trade and *how big*. The
executor only translates that into the Kalshi REST order body and (optionally)
sends it. Two safety rails are deliberately on by default:

  * ``dry_run=True`` — build and return the order body WITHOUT calling the API.
    The runner logs it so you can watch what it *would* do before arming it.
  * production requires ``allow_prod=True`` — even a non-dry-run executor refuses
    to touch the prod environment unless you explicitly opt in. Demo is the
    default everywhere (CLAUDE.md house rules).

Idempotency: every order carries a ``client_order_id``. We derive a deterministic
one from (ticker, side, action, price, count, key) when the caller supplies a
key, so a retried cycle reuses the same id and Kalshi de-dupes the fill; absent a
key we fall back to a random UUID.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from hedge.decision.engine import Action, Decision, Side
from hedge.kalshi.client import KalshiError

# Stable namespace so deterministic client_order_ids are reproducible across runs.
_ORDER_NS = uuid.UUID("6f9619ff-8b86-d011-b42d-00cf4fc964ff")


def _client_order_id(decision: Decision, idem_key: str | None) -> str:
    if idem_key is None:
        return str(uuid.uuid4())
    seed = (
        f"{decision.ticker}|{decision.action.value}|{decision.side.value if decision.side else '-'}"
        f"|{decision.price_cents}|{decision.count}|{idem_key}"
    )
    return str(uuid.uuid5(_ORDER_NS, seed))


def build_order_body(decision: Decision, *, idem_key: str | None = None) -> dict[str, Any]:
    """Translate a tradable ``Decision`` into a Kalshi **V2** create-order body.

    Kalshi's V2 order model (POST /portfolio/events/orders) is YES-priced and
    side-only — there is no yes/no + buy/sell:
      * ``side="bid"`` buys YES; ``side="ask"`` sells YES (which IS buying NO).
      * ``price`` is a single fixed-point **dollar** string, always the YES price.

    So a YES decision posts ``bid`` at its price; a NO decision posts ``ask`` at
    ``1 - no_price`` (selling YES at the equivalent price acquires the NO). A taker
    (crossing) decision uses ``immediate_or_cancel``; a maker rests ``good_till_canceled``
    with ``post_only``. Raises if the decision isn't a trade.
    """
    if not decision.is_trade:
        raise ValueError(f"decision is not a trade: {decision.action} {decision.ticker}")
    if decision.side is None:
        raise ValueError("tradable decision missing side")

    # V2 price is always the YES price. Our price_cents is the price of the chosen
    # contract; for NO it converts to the equivalent YES price (1 - no_price).
    yes_price = decision.price_cents / 100.0
    if decision.side is Side.NO:
        yes_price = 1.0 - yes_price
    # bid = acquire YES exposure (buy YES, or sell NO); ask = shed it (sell YES, or
    # buy NO). action matters: a SELL to close flips the side vs the same-side BUY.
    acquiring_yes = (
        (decision.action is Action.BUY and decision.side is Side.YES)
        or (decision.action is Action.SELL and decision.side is Side.NO)
    )
    v2_side = "bid" if acquiring_yes else "ask"

    body: dict[str, Any] = {
        "ticker": decision.ticker,
        "side": v2_side,
        "count": f"{decision.count:.2f}",
        "price": f"{yes_price:.4f}",
        "time_in_force": "good_till_canceled" if decision.maker else "immediate_or_cancel",
        "self_trade_prevention_type": "taker_at_cross",
        "client_order_id": _client_order_id(decision, idem_key),
    }
    if decision.maker:
        body["post_only"] = True
    return body


@dataclass
class OrderTicket:
    """Record of one attempted order: the body, whether it was sent, the result.

    ``order_id``/``status``/``fill_count`` are parsed from the Kalshi order in the
    create response (when placed) so the runner can persist fill state and reconcile.
    """

    decision: Decision
    body: dict[str, Any]
    placed: bool                 # True only if actually sent to the API
    dry_run: bool
    response: dict[str, Any] | None = None
    error: str | None = None
    order_id: str | None = None
    status: str | None = None    # Kalshi: resting | executed | canceled
    fill_count: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


def parse_order(resp: dict[str, Any] | None) -> tuple[str | None, str | None, int]:
    """Pull (order_id, status, fill_count) from a Kalshi order payload.

    Accepts either a full ``{"order": {...}}`` create response or a bare order dict.
    ``fill_count`` prefers the fractional-precision ``fill_count_fp`` field, falling
    back to ``fill_count``; both are read as an integer number of contracts.
    """
    if not resp:
        return None, None, 0
    o = resp.get("order", resp)
    oid = o.get("order_id") or o.get("id")
    status = o.get("status")
    raw_fill = o.get("fill_count_fp", o.get("fill_count", 0)) or 0
    try:
        fill = int(float(raw_fill))
    except (TypeError, ValueError):
        fill = 0
    return oid, status, fill


class Executor:
    """Places (or simulates) orders for the decision engine's verdicts.

    Args:
        client: a signed ``KalshiClient`` (may be None in pure dry-run/testing).
        env: the environment name the client points at ("demo"/"prod").
        dry_run: when True (default), never calls the API — just builds the body.
        allow_prod: must be True to send live orders against the prod environment.
    """

    def __init__(self, client: Any, *, env: str = "demo", dry_run: bool = True,
                 allow_prod: bool = False):
        self.client = client
        self.env = env
        self.dry_run = dry_run
        self.allow_prod = allow_prod
        if env == "prod" and not dry_run and not allow_prod:
            raise RuntimeError(
                "refusing to arm live trading against PROD without allow_prod=True"
            )

    def place(self, decision: Decision, *, idem_key: str | None = None) -> OrderTicket:
        """Place a single decision. HOLD/empty decisions return an unplaced ticket."""
        if not decision.is_trade:
            return OrderTicket(decision, {}, placed=False, dry_run=self.dry_run,
                               meta={"skipped": decision.reason or "hold"})
        body = build_order_body(decision, idem_key=idem_key)

        if self.dry_run:
            return OrderTicket(decision, body, placed=False, dry_run=True)
        if self.env == "prod" and not self.allow_prod:
            return OrderTicket(decision, body, placed=False, dry_run=False,
                               error="blocked: prod orders require allow_prod=True")
        if self.client is None:
            return OrderTicket(decision, body, placed=False, dry_run=False,
                               error="no client configured")
        try:
            resp = self.client.create_order(**body)
            oid, status, fill = parse_order(resp)
            return OrderTicket(decision, body, placed=True, dry_run=False, response=resp,
                               order_id=oid, status=status or "placed", fill_count=fill)
        except KalshiError as e:
            # 409 order_already_exists: the client_order_id was already used (an
            # idempotent retry, or a same-key order today). Benign — not a failure.
            if e.status == 409:
                return OrderTicket(decision, body, placed=False, dry_run=False,
                                   error="duplicate (idempotent skip)",
                                   meta={"idempotent": True})
            return OrderTicket(decision, body, placed=False, dry_run=False,
                               error=f"KalshiError: {e}")
        except Exception as e:  # noqa: BLE001 — surface, don't crash the loop
            return OrderTicket(decision, body, placed=False, dry_run=False,
                               error=f"{type(e).__name__}: {e}")
