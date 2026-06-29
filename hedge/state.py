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

#: Columns of the ``decisions`` table, in insert order. One row per market per cycle
#: (including HOLDs) — the canonical, queryable record we learn from. Kept as a tuple
#: so :meth:`State.record_decision` can take an ergonomic dict and insert positionally.
DECISION_COLS = (
    "cycle_seq", "ts", "utc_date", "ticker", "strategy", "action", "side",
    "count", "price_cents", "prob", "sigma", "edge", "kelly_fraction",
    "yes_bid", "yes_ask", "mid", "last", "bankroll", "portfolio_at_risk",
    "placed", "dry_run", "error", "reason", "client_order_id", "meta_json",
)


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
            -- double-count a settlement into daily_pnl. Extra columns (side, entry,
            -- count, outcome) make each row a complete trade record for review.
            CREATE TABLE IF NOT EXISTS settled (
                ticker      TEXT PRIMARY KEY,
                utc_date    TEXT NOT NULL,
                pnl         REAL NOT NULL,
                ts          TEXT NOT NULL
            );
            -- Every decision, one row per market per cycle (incl. HOLD): the model's
            -- belief, the market it priced against, the order it intended, and whether
            -- it went through. This is the substrate the learning loop queries.
            CREATE TABLE IF NOT EXISTS decisions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                cycle_seq       INTEGER NOT NULL,
                ts              TEXT NOT NULL,
                utc_date        TEXT NOT NULL,
                ticker          TEXT NOT NULL,
                strategy        TEXT,
                action          TEXT NOT NULL,
                side            TEXT,
                count           INTEGER NOT NULL DEFAULT 0,
                price_cents     INTEGER NOT NULL DEFAULT 0,
                prob            REAL,
                sigma           REAL,
                edge            REAL,
                kelly_fraction  REAL,
                yes_bid         REAL,
                yes_ask         REAL,
                mid             REAL,
                last            REAL,
                bankroll        REAL,
                portfolio_at_risk REAL,
                placed          INTEGER NOT NULL DEFAULT 0,
                dry_run         INTEGER NOT NULL DEFAULT 0,
                error           TEXT,
                reason          TEXT,
                client_order_id TEXT,
                meta_json       TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_decisions_day ON decisions(utc_date, ticker);
            -- Actual broker fills (NOT intent): the authoritative entry price/size we
            -- realize P&L against. Keyed by client_order_id so partial->full upserts.
            CREATE TABLE IF NOT EXISTS fills (
                client_order_id TEXT PRIMARY KEY,
                order_id        TEXT,
                ticker          TEXT NOT NULL,
                side            TEXT NOT NULL,
                action          TEXT NOT NULL,
                fill_count      INTEGER NOT NULL DEFAULT 0,
                avg_price_cents REAL,
                fee_cents       REAL,
                status          TEXT,
                ts              TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_fills_ticker ON fills(ticker);
            CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        # Older DBs predate the richer `settled` columns; add them idempotently so a
        # deployed volume migrates in place without dropping booked P&L.
        for col, decl in (("side", "TEXT"), ("entry_cents", "REAL"),
                          ("count", "INTEGER"), ("outcome", "TEXT")):
            try:
                self.conn.execute(f"ALTER TABLE settled ADD COLUMN {col} {decl}")
            except sqlite3.OperationalError:
                pass  # column already exists
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

    def order_for_oid(self, order_id: str) -> sqlite3.Row | None:
        """Look up our recorded order by the broker's order_id (for fill matching)."""
        if not order_id:
            return None
        return self.conn.execute(
            "SELECT * FROM orders WHERE order_id=? ORDER BY ts DESC LIMIT 1",
            (order_id,)).fetchone()

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

    # ----- decisions (the learning substrate) --------------------------------
    def record_decision(self, row: dict) -> None:
        """Persist one decision (keys per :data:`DECISION_COLS`; missing -> NULL).

        One row per market per cycle, HOLDs included — this is the canonical record
        the learning loop queries. Best-effort: never let a logging row crash a cycle.
        """
        # NOT NULL columns must never receive a bare None (a HOLD row omits order
        # fields); fall back to their schema defaults. SQLite stores bools as ints.
        defaults = {"count": 0, "price_cents": 0, "placed": 0, "dry_run": 0}
        vals = [row.get(c) if row.get(c) is not None else defaults.get(c)
                for c in DECISION_COLS]
        placeholders = ",".join("?" * len(DECISION_COLS))
        self.conn.execute(
            f"INSERT INTO decisions ({','.join(DECISION_COLS)}) VALUES ({placeholders})",
            vals,
        )
        self.conn.commit()

    def decisions_for(self, utc_date: str) -> list[sqlite3.Row]:
        return list(self.conn.execute(
            "SELECT * FROM decisions WHERE utc_date=? ORDER BY id", (utc_date,)).fetchall())

    # ----- fills (broker truth — what actually traded) -----------------------
    def record_fill(self, client_order_id: str, ticker: str, side: str, action: str,
                    fill_count: int, *, order_id: str | None = None,
                    avg_price_cents: float | None = None, fee_cents: float | None = None,
                    status: str | None = None) -> None:
        """Upsert an order's realized fill (keyed by client_order_id).

        Accrues as partial fills become whole; COALESCE keeps a known price/fee if a
        later update omits it. The recorded entry is authoritative for P&L booking.
        """
        self.conn.execute(
            """INSERT INTO fills
                 (client_order_id, order_id, ticker, side, action, fill_count,
                  avg_price_cents, fee_cents, status, ts)
               VALUES (?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(client_order_id) DO UPDATE SET
                 order_id=COALESCE(excluded.order_id, fills.order_id),
                 fill_count=excluded.fill_count,
                 avg_price_cents=COALESCE(excluded.avg_price_cents, fills.avg_price_cents),
                 fee_cents=COALESCE(excluded.fee_cents, fills.fee_cents),
                 status=COALESCE(excluded.status, fills.status)""",
            (client_order_id, order_id, ticker, side, action, int(fill_count),
             avg_price_cents, fee_cents, status, _now_iso()),
        )
        self.conn.commit()

    def fills_for_ticker(self, ticker: str) -> list[sqlite3.Row]:
        return list(self.conn.execute(
            "SELECT * FROM fills WHERE ticker=? AND fill_count > 0", (ticker,)).fetchall())

    def unsettled_fill_tickers(self) -> list[str]:
        """Tickers we have real fills on that haven't been booked as settled yet."""
        rows = self.conn.execute(
            "SELECT DISTINCT ticker FROM fills WHERE fill_count > 0 "
            "AND ticker NOT IN (SELECT ticker FROM settled)").fetchall()
        return [r["ticker"] for r in rows]

    # ----- daily P&L (for the daily-loss stop + heartbeat) -------------------
    def book_settlement(self, ticker: str, pnl: float, *, utc_date: str | None = None,
                        side: str | None = None, entry_cents: float | None = None,
                        count: int | None = None, outcome: str | None = None) -> bool:
        """Record a settled market's realized P&L once. Returns False if already booked.

        Accumulates into ``daily_pnl`` for the settlement's UTC day so the daily-loss
        stop sees it. Idempotent per ticker via the ``settled`` table. The optional
        side/entry/count/outcome make the row a complete trade record for review.
        """
        if self.conn.execute("SELECT 1 FROM settled WHERE ticker=?", (ticker,)).fetchone():
            return False
        day = utc_date or _now_iso()[:10]
        self.conn.execute(
            "INSERT INTO settled (ticker, utc_date, pnl, ts, side, entry_cents, count, outcome) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (ticker, day, float(pnl), _now_iso(), side, entry_cents,
             int(count) if count is not None else None, outcome))
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
