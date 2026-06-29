"""Execution: turn a ``Decision`` into a signed Kalshi order.

This is the only layer allowed to place orders. It is intentionally thin and
boring — all the judgment already happened in the decision engine. Its jobs are:

  * Translate a ``Decision`` into the Kalshi order body (side/action/price/count).
  * Attach an idempotent ``client_order_id`` so a retry can't double-fill.
  * Guard against accidents: orders are DRY-RUN by default, and placing against
    production requires an explicit opt-in.

See ``executor.py``.
"""

from hedge.execution.executor import (
    Executor,
    OrderTicket,
    build_order_body,
)

__all__ = ["Executor", "OrderTicket", "build_order_body"]
