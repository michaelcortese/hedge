"""Durable trading state — survives restarts so the autonomous bot is crash-safe.

The live runner is otherwise in-memory: a restart would forget which orders it has
placed, how much it has lost today, and whether a fill landed. That is unacceptable
for a bot that places real orders unattended. This module persists the few things
that MUST outlive the process to a small SQLite database on the Fly volume
(``HEDGE_STATE_DIR``), the established pattern for autonomous traders (e.g. Freqtrade
persists trades/orders and reloads on restart).

What is persisted:
  * **orders** — every order we have placed, keyed by its idempotent
    ``client_order_id``: the Kalshi order id, fill state, and status. This both
    drives fill reconciliation and stops us from re-placing an order after a restart.
  * **daily_pnl** — realized P&L per UTC day, so the daily-loss stop and the
    once-a-day heartbeat survive a restart.
  * **meta** — small counters/flags (e.g. the monotonic cycle sequence).

Kalshi positions/balance remain the *source of truth* (read each cycle); this store
only remembers what the broker can't tell us cheaply (our own intent + per-day P&L).
Everything is best-effort and defensive: a corrupt/locked DB must never crash a cycle.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

#: Order statuses we treat as "still working" (not yet a settled position/closed).
OPEN_STATUSES = ("placed", "pending", "resting", "partial")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_db_path() -> Path:
    """Where the state DB lives: ``$HEDGE_STATE_DIR/hedge.db`` (Fly volume in prod)."""
    base = Path(os.environ.get("HEDGE_STATE_DIR", "data/runs/live"))
    return base / "hedge.db"


class State:
    """SQLite-backed durable state. Use ``State()`` for prod, ``State(":memory:")`` in tests."""

    def __init__(self, path: str | Path | None = None):
        if path is None:
            path = default_db_path()
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._migrate()

    def _migrate(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS orders (
                client_order_id TEXT PRIMARY KEY,
                order_id        TEXT,
                ticker          TEXT NOT NULL,
                side            TEXT NOT NULL,
                action          TEXT NOT NULL,
                price_cents     INTEGER NOT NULL,
                count           INTEGER NOT NULL,
                status          TEXT NOT NULL,
                fill_count      INTEGER NOT NULL DEFAULT 0,
                ts              TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_orders_ticker ON orders(ticker, status);
            CREATE TABLE IF NOT EXISTS daily_pnl (
                utc_date    TEXT PRIMARY KEY,
                realized    REAL NOT NULL DEFAULT 0,
                fees        REAL NOT NULL DEFAULT 0,
                updated_ts  TEXT NOT NULL
            );
            -- Tickers whose realized P&L has already been booked, so we never
            -- double-count a settlement into daily_pnl.
            CREATE TABLE IF NOT EXISTS settled (
                ticker      TEXT PRIMARY KEY,
                utc_date    TEXT NOT NULL,
                pnl         REAL NOT NULL,
                ts          TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        self.conn.commit()

    # ----- orders ------------------------------------------------------------
    def record_order(self, client_order_id: str, ticker: str, side: str, action: str,
                     price_cents: int, count: int, *, order_id: str | None = None,
                     status: str = "placed", fill_count: int = 0) -> None:
        """Insert (or upsert) a placed order keyed by its idempotent client_order_id."""
        self.conn.execute(
            """INSERT INTO orders
                 (client_order_id, order_id, ticker, side, action, price_cents,
                  count, status, fill_count, ts)
               VALUES (?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(client_order_id) DO UPDATE SET
                  order_id=COALESCE(excluded.order_id, orders.order_id),
                  status=excluded.status, fill_count=excluded.fill_count""",
            (client_order_id, order_id, ticker, side, action, int(price_cents),
             int(count), status, int(fill_count), _now_iso()),
        )
        self.conn.commit()

    def update_order(self, client_order_id: str, *, order_id: str | None = None,
                    status: str | None = None, fill_count: int | None = None) -> None:
        row = self.get_order(client_order_id)
        if row is None:
            return
        self.conn.execute(
            "UPDATE orders SET order_id=?, status=?, fill_count=? WHERE client_order_id=?",
            (order_id if order_id is not None else row["order_id"],
             status if status is not None else row["status"],
             int(fill_count) if fill_count is not None else row["fill_count"],
             client_order_id),
        )
        self.conn.commit()

    def get_order(self, client_order_id: str) -> sqlite3.Row | None:
        cur = self.conn.execute(
            "SELECT * FROM orders WHERE client_order_id=?", (client_order_id,))
        return cur.fetchone()

    def has_open_order(self, ticker: str, side: str | None = None) -> bool:
        """True if we have a still-working order on this ticker (optionally side).

        Prevents stacking a second order on a market while the first is unfilled —
        a resting maker is NOT yet a position, so the engine alone wouldn't catch it.
        """
        q = f"SELECT 1 FROM orders WHERE ticker=? AND status IN ({','.join('?' * len(OPEN_STATUSES))})"
        params: list = [ticker, *OPEN_STATUSES]
        if side is not None:
            q += " AND side=?"
            params.append(side)
        return self.conn.execute(q + " LIMIT 1", params).fetchone() is not None

    def open_orders(self) -> list[sqlite3.Row]:
        q = f"SELECT * FROM orders WHERE status IN ({','.join('?' * len(OPEN_STATUSES))})"
        return list(self.conn.execute(q, OPEN_STATUSES).fetchall())

    # ----- daily P&L (for the daily-loss stop + heartbeat) -------------------
    def book_settlement(self, ticker: str, pnl: float, *, utc_date: str | None = None) -> bool:
        """Record a settled market's realized P&L once. Returns False if already booked.

        Accumulates into ``daily_pnl`` for the settlement's UTC day so the daily-loss
        stop sees it. Idempotent per ticker via the ``settled`` table.
        """
        if self.conn.execute("SELECT 1 FROM settled WHERE ticker=?", (ticker,)).fetchone():
            return False
        day = utc_date or _now_iso()[:10]
        self.conn.execute("INSERT INTO settled (ticker, utc_date, pnl, ts) VALUES (?,?,?,?)",
                          (ticker, day, float(pnl), _now_iso()))
        self.conn.execute(
            """INSERT INTO daily_pnl (utc_date, realized, fees, updated_ts)
               VALUES (?,?,0,?)
               ON CONFLICT(utc_date) DO UPDATE SET
                 realized = daily_pnl.realized + excluded.realized,
                 updated_ts = excluded.updated_ts""",
            (day, float(pnl), _now_iso()),
        )
        self.conn.commit()
        return True

    def realized_today(self, utc_date: str | None = None) -> float:
        """Net realized P&L (dollars) for the UTC day (negative = a loss)."""
        day = utc_date or _now_iso()[:10]
        row = self.conn.execute(
            "SELECT realized FROM daily_pnl WHERE utc_date=?", (day,)).fetchone()
        return float(row["realized"]) if row else 0.0

    # ----- meta counters -----------------------------------------------------
    def next_cycle_seq(self) -> int:
        row = self.conn.execute("SELECT value FROM meta WHERE key='cycle_seq'").fetchone()
        nxt = (int(row["value"]) + 1) if row else 1
        self.conn.execute(
            "INSERT INTO meta (key, value) VALUES ('cycle_seq', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (str(nxt),))
        self.conn.commit()
        return nxt

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta (key, value) VALUES (?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, str(value)))
        self.conn.commit()

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:  # noqa: BLE001
            pass

    def __enter__(self) -> "State":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
