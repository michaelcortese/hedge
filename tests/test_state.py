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


def test_record_decision_persists_row(tmp_path):
    s = _fresh(tmp_path)
    s.record_decision({
        "cycle_seq": 1, "ts": "2026-06-29T12:00:00+00:00", "utc_date": "2026-06-29",
        "ticker": "KXHIGHCHI-26JUN29-T80", "strategy": "weather_blend", "action": "buy",
        "side": "yes", "count": 4, "price_cents": 55, "prob": 0.8, "sigma": 0.02,
        "edge": 0.03, "kelly_fraction": 0.05, "yes_bid": 0.53, "yes_ask": 0.55,
        "mid": 0.54, "last": 0.54, "bankroll": 1000.0, "portfolio_at_risk": 0.0,
        "placed": 1, "dry_run": 0, "error": None, "reason": "open",
        "client_order_id": "coid-1", "meta_json": '{"strategy": "weather_blend"}'})
    rows = s.decisions_for("2026-06-29")
    assert len(rows) == 1 and rows[0]["ticker"] == "KXHIGHCHI-26JUN29-T80"
    assert rows[0]["prob"] == 0.8 and rows[0]["placed"] == 1


def test_record_decision_tolerates_missing_keys(tmp_path):
    # A HOLD row carries no order fields — missing keys must default to NULL, not crash.
    s = _fresh(tmp_path)
    s.record_decision({"cycle_seq": 1, "ts": "2026-06-29T12:00:00+00:00",
                       "utc_date": "2026-06-29", "ticker": "T", "action": "hold"})
    rows = s.decisions_for("2026-06-29")
    assert len(rows) == 1 and rows[0]["side"] is None and rows[0]["count"] == 0


def test_record_fill_upserts_and_accrues(tmp_path):
    s = _fresh(tmp_path)
    s.record_fill("coid-1", "T", "yes", "buy", 2, order_id="OID-1",
                  avg_price_cents=55.0, fee_cents=1.0, status="partial")
    s.record_fill("coid-1", "T", "yes", "buy", 4, order_id="OID-1",
                  avg_price_cents=56.0, fee_cents=2.0, status="executed")
    rows = s.fills_for_ticker("T")
    assert len(rows) == 1                       # upsert, not a second row
    assert rows[0]["fill_count"] == 4 and rows[0]["status"] == "executed"


def test_record_fill_coalesces_known_price(tmp_path):
    # A later update without a price must keep the earlier known price/fee.
    s = _fresh(tmp_path)
    s.record_fill("coid-1", "T", "yes", "buy", 2, avg_price_cents=55.0, fee_cents=1.0)
    s.record_fill("coid-1", "T", "yes", "buy", 3)   # no price this time
    row = s.fills_for_ticker("T")[0]
    assert row["avg_price_cents"] == 55.0 and row["fee_cents"] == 1.0 and row["fill_count"] == 3


def test_unsettled_fill_tickers_excludes_booked(tmp_path):
    s = _fresh(tmp_path)
    s.record_fill("c1", "T1", "yes", "buy", 4, avg_price_cents=55.0)
    s.record_fill("c2", "T2", "no", "buy", 2, avg_price_cents=40.0)
    assert set(s.unsettled_fill_tickers()) == {"T1", "T2"}
    s.book_settlement("T1", 1.8, outcome="yes", side="yes", entry_cents=55.0, count=4)
    assert s.unsettled_fill_tickers() == ["T2"]


def test_order_for_oid_maps_broker_id(tmp_path):
    s = _fresh(tmp_path)
    s.record_order("coid-1", "T", "yes", "buy", 55, 4, order_id="OID-9", status="executed")
    assert s.order_for_oid("OID-9")["client_order_id"] == "coid-1"
    assert s.order_for_oid("nope") is None


def test_settled_row_records_trade_detail(tmp_path):
    s = _fresh(tmp_path)
    s.book_settlement("T", 1.8, outcome="yes", side="yes", entry_cents=55.0,
                      count=4, utc_date="2026-06-29")
    row = s.conn.execute("SELECT * FROM settled WHERE ticker='T'").fetchone()
    assert row["outcome"] == "yes" and row["side"] == "yes"
    assert row["entry_cents"] == 55.0 and row["count"] == 4
