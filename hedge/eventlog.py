"""Append-only event log — the durable audit trail that survives a redeploy.

The live SQLite store (:mod:`hedge.state`) is the bot's *working* state: rows are
upserted (orders, fills), counters advance, and — critically — the DB lives on the
ephemeral container layer in prod, so **every redeploy wipes it** (the runner writes
``data/runs/live/`` relative to its cwd ``/app`` instead of the ``/data`` volume). That
is fine for working state Kalshi can re-tell us, but it means the historical record —
what we decided, what filled, what settled, when the guard tripped — vanishes on each
deploy. That history is exactly what the calibration loop and any post-mortem need.

This module is the fix-by-addition the user asked for: a strictly **append-only**,
newline-delimited JSON log written to the **durable volume** (``HEDGE_STATE_DIR``,
i.e. ``/data`` in prod), *separate* from the wiped ``hedge.db``. Every meaningful
event — one per decision, order, fill, P&L booking, settlement, halt, heartbeat — is
appended as a single self-describing line and never rewritten. The SQLite store can
be reconstructed by replaying the log; the log itself is the source of truth for
"what happened over time".

Design rules (kept deliberately small and boring so it can never break a cycle):

  * **Append-only.** Files are opened ``"a"`` and lines are only ever added. We never
    seek, truncate, update, or delete. A corrupted tail line is survivable — readers
    skip unparseable lines (see :func:`iter_events`).
  * **Durable location.** The directory derives from ``HEDGE_STATE_DIR`` (the Fly
    volume), matching how the runner already writes ``status.json``. Falls back to
    ``data/runs/live`` for local/dev runs.
  * **Partitioned by UTC day.** ``events/events_YYYY-MM-DD.jsonl`` keeps any single
    file bounded and makes day-scoped reads cheap, mirroring the decision JSONLs.
  * **Best-effort.** Every write is wrapped: a logging failure prints and returns,
    never raising into the trading loop. A log we can't write is strictly better than
    a cycle that crashes because we couldn't log.

Each line is an object: ``{"ts": <iso8601 UTC>, "type": <str>, "seq": <int|None>,
... payload}``. ``type`` is one of :data:`EVENT_TYPES` (free-form is allowed; the
constant just documents the vocabulary the runner emits). ``seq`` is the monotonic
cycle sequence when known, so events can be grouped by the cycle that produced them.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

#: The event vocabulary the runner emits. Not enforced — :meth:`EventLog.emit` accepts
#: any string — but documenting it keeps producers and consumers honest.
EVENT_TYPES = (
    "decision",      # one market decided this cycle (BUY/SELL/HOLD), full snapshot
    "order",         # an order ticket we placed (or attempted): id, status, fills
    "fill",          # a reconciled broker fill (authoritative entry price/size)
    "intraday_pnl",  # realized P&L booked from an intraday close (the daily-stop delta)
    "settlement",    # a market settled; net-held realized P&L booked once
    "halt",          # kill-switch trip / daily-loss stop / latch written
    "guard",         # periodic guard + skill-gate assessment (calibration health)
    "heartbeat",     # once-a-day alive ping (bankroll, open positions)
    "cycle",         # per-cycle summary (markets checked, would-trade, placed)
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_event_dir() -> Path:
    """Where the append-only log lives: ``$HEDGE_STATE_DIR/events`` (Fly volume in prod).

    Mirrors :func:`hedge.state.default_db_path`'s use of ``HEDGE_STATE_DIR`` but lands
    on the DURABLE volume on purpose — this is the record that must outlive the DB.
    """
    base = Path(os.environ.get("HEDGE_STATE_DIR", "data/runs/live"))
    return base / "events"


class EventLog:
    """Append-only JSONL event sink on the durable volume.

    Use ``EventLog()`` for prod (resolves the volume path) or ``EventLog(tmp_path)``
    in tests. Construction never touches the filesystem; the day's file and its parent
    directory are created lazily on the first successful :meth:`emit`.
    """

    def __init__(self, dir: str | Path | None = None):
        self.dir = Path(dir) if dir is not None else default_event_dir()

    def _path_for(self, ts_iso: str) -> Path:
        """Day-partitioned file for an ISO timestamp (``events_YYYY-MM-DD.jsonl``)."""
        return self.dir / f"events_{ts_iso[:10]}.jsonl"

    def emit(self, event_type: str, payload: dict | None = None, *,
             seq: int | None = None, ts: str | None = None) -> None:
        """Append one event line. Best-effort: never raises into the caller.

        ``payload`` keys are merged at top level alongside the envelope
        (``ts``/``type``/``seq``); reserved envelope keys win so a payload can't shadow
        them. ``ts`` defaults to now (UTC); pass an event's own timestamp to keep the
        log faithful when backfilling. ``seq`` is the cycle sequence when known.
        """
        ts = ts or _now_iso()
        record: dict = dict(payload or {})
        record["ts"] = ts
        record["type"] = event_type
        record["seq"] = seq
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            with self._path_for(ts).open("a") as f:
                f.write(json.dumps(record, default=str) + "\n")
                f.flush()
        except Exception as e:  # noqa: BLE001 — logging must never crash a cycle
            print(f"[eventlog] emit failed ({type(e).__name__}: {e})", flush=True)

    def iter_day(self, utc_date: str) -> Iterator[dict]:
        """Yield this day's events in order (skips a corrupt/partial trailing line)."""
        yield from iter_events(self._path_for(utc_date + "T"))


def iter_events(path: str | Path) -> Iterator[dict]:
    """Yield parsed events from one JSONL file, skipping blank/corrupt lines.

    Append-only logs can have a torn final line if the process died mid-write; a
    reader must tolerate it rather than choke. Missing file yields nothing.
    """
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def read_all(dir: str | Path | None = None) -> list[dict]:
    """Read every event across all day files in ``dir``, ordered by (file, line)."""
    d = Path(dir) if dir is not None else default_event_dir()
    out: list[dict] = []
    for path in sorted(d.glob("events_*.jsonl")):
        out.extend(iter_events(path))
    return out


def _main(argv: list[str] | None = None) -> None:
    """Tiny CLI: ``python -m hedge.eventlog [tail [N] | count]`` over the durable log."""
    import sys

    args = list(argv if argv is not None else sys.argv[1:])
    cmd = args[0] if args else "tail"
    events = read_all()
    if cmd == "count":
        by_type: dict[str, int] = {}
        for e in events:
            by_type[e.get("type", "?")] = by_type.get(e.get("type", "?"), 0) + 1
        print(f"{len(events)} events in {default_event_dir()}")
        for t, n in sorted(by_type.items(), key=lambda kv: -kv[1]):
            print(f"  {n:>6}  {t}")
        return
    n = int(args[1]) if len(args) > 1 else 20
    for e in events[-n:]:
        print(json.dumps(e, default=str))


if __name__ == "__main__":
    _main()
