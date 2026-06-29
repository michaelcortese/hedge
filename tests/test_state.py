"""Durable state store: orders, idempotency, daily P&L, cycle seq, recovery."""

from __future__ import annotations

from hedge.state import State


def _fresh(tmp_path):
    return State(tmp_path / "hedge.db")


def test_record_and_get_order(tmp_path):
    s = _fresh(tmp_path)
    s.record_order("coid-1", "KXHIGHCHI-26JUN29-T80", "yes", "buy", 55, 4,
                   order_id="ord-1", status="resting", fill_count=0)
    row = s.get_order("coid-1")
    assert row["ticker"] == "KXHIGHCHI-26JUN29-T80" and row["status"] == "resting"
    assert row["order_id"] == "ord-1" and row["count"] == 4


def test_order_upsert_updates_fill(tmp_path):
    s = _fresh(tmp_path)
    s.record_order("coid-1", "T", "yes", "buy", 55, 4, status="resting")
    s.update_order("coid-1", status="executed", fill_count=4, order_id="ord-9")
    row = s.get_order("coid-1")
    assert row["status"] == "executed" and row["fill_count"] == 4 and row["order_id"] == "ord-9"


def test_has_open_order_blocks_stacking(tmp_path):
    s = _fresh(tmp_path)
    s.record_order("coid-1", "T", "yes", "buy", 55, 4, status="resting")
    assert s.has_open_order("T") is True
    assert s.has_open_order("T", side="yes") is True
    assert s.has_open_order("T", side="no") is False
    s.update_order("coid-1", status="executed")
    assert s.has_open_order("T") is False  # filled -> no longer "open"


def test_daily_pnl_idempotent_per_ticker(tmp_path):
    s = _fresh(tmp_path)
    assert s.book_settlement("T1", -3.0, utc_date="2026-06-29") is True
    assert s.book_settlement("T1", -3.0, utc_date="2026-06-29") is False  # already booked
    assert s.book_settlement("T2", 5.0, utc_date="2026-06-29") is True
    assert abs(s.realized_today("2026-06-29") - 2.0) < 1e-9   # -3 + 5
    assert s.realized_today("2026-06-30") == 0.0


def test_cycle_seq_monotonic_and_durable(tmp_path):
    s = _fresh(tmp_path)
    assert s.next_cycle_seq() == 1
    assert s.next_cycle_seq() == 2
    s.close()
    s2 = State(tmp_path / "hedge.db")   # reopen: counter survives restart
    assert s2.next_cycle_seq() == 3


def test_state_recovers_orders_after_restart(tmp_path):
    s = _fresh(tmp_path)
    s.record_order("coid-1", "T", "yes", "buy", 55, 4, status="resting")
    s.close()
    s2 = State(tmp_path / "hedge.db")
    assert s2.has_open_order("T") is True
    assert s2.get_order("coid-1")["count"] == 4
