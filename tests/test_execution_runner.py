"""Execution + runner: order-body translation, safety guards, and one cycle."""

from __future__ import annotations

import pytest

from hedge.decision import Action, Decision, RiskConfig, Side
from hedge.execution import Executor, build_order_body
from hedge.runner import Runner
from hedge.signal import Signal
from hedge.strategies.base import MarketView, Strategy


# --------------------------------------------------------------------------- #
# Order body                                                                    #
# --------------------------------------------------------------------------- #
def _buy(side: Side, cents: int = 55, count: int = 4) -> Decision:
    return Decision(ticker="MKT", action=Action.BUY, side=side, price=cents / 100,
                    price_cents=cents, count=count)


def test_yes_order_body_is_v2_bid_at_yes_price():
    # V2: buying YES is a bid at the YES price (dollars), count as a fixed-point string.
    b = build_order_body(_buy(Side.YES, cents=55))
    assert b["side"] == "bid" and b["price"] == "0.5500"
    assert b["count"] == "4.00" and b["time_in_force"] == "immediate_or_cancel"
    assert "action" not in b and "yes_price" not in b and b["client_order_id"]


def test_no_order_body_is_v2_ask_at_complement_price():
    # V2: buying NO is an ask (sell YES) at price 1 - no_price.
    b = build_order_body(_buy(Side.NO, cents=45))
    assert b["side"] == "ask" and b["price"] == "0.5500"   # 1 - 0.45
    assert "no_price" not in b


def test_maker_order_is_gtc_post_only():
    d = Decision(ticker="MKT", action=Action.BUY, side=Side.YES, price=0.40,
                 price_cents=40, count=3, maker=True)
    b = build_order_body(d)
    assert b["time_in_force"] == "good_till_canceled" and b["post_only"] is True


def test_idempotency_key_is_deterministic():
    d = _buy(Side.YES)
    a = build_order_body(d, idem_key="2026-06-29")
    b = build_order_body(d, idem_key="2026-06-29")
    c = build_order_body(d, idem_key="2026-06-30")
    assert a["client_order_id"] == b["client_order_id"]
    assert a["client_order_id"] != c["client_order_id"]


def test_build_body_rejects_hold():
    with pytest.raises(ValueError):
        build_order_body(Decision(ticker="MKT", action=Action.HOLD))


# --------------------------------------------------------------------------- #
# Executor safety                                                               #
# --------------------------------------------------------------------------- #
class _FakeClient:
    def __init__(self):
        self.orders = []
        self.canceled = []
        self.order_status = "resting"   # what get_order reports back
        self.order_fill = 0
        self.results = {}               # ticker -> "yes"/"no" settlement result

    def create_order(self, **body):
        self.orders.append(body)
        return {"order": {"order_id": "OID", "status": "resting"}}

    def get_order(self, order_id):
        return {"order": {"order_id": order_id, "status": self.order_status,
                          "fill_count": self.order_fill}}

    def get_orders(self, **filters):
        # Broker-truth listing the runner reconciles against.
        return {"orders": getattr(self, "listing", [])}

    def get_fills(self, **filters):
        # Broker-truth fills the runner aggregates into the fills table.
        return {"fills": getattr(self, "fills", [])}

    def cancel_order(self, order_id):
        self.canceled.append(order_id)
        return {}

    def get_market(self, ticker):
        m = {"ticker": ticker}
        if ticker in self.results:
            m["result"] = self.results[ticker]
        # Optional per-ticker quote/strike fields so a held market pulled directly
        # (not via discovery) still presents a two-sided book to manage against.
        m.update(getattr(self, "market_fields", {}).get(ticker, {}))
        return {"market": m}

    def get_balance(self):
        return {"balance": 1_000_00}  # $1,000 in cents

    def get_positions(self):
        return {"market_positions": getattr(self, "positions_payload", [])}


def test_dry_run_does_not_place():
    client = _FakeClient()
    ex = Executor(client, env="demo", dry_run=True)
    t = ex.place(_buy(Side.YES))
    assert t.placed is False and t.dry_run is True and client.orders == []


def test_live_demo_places():
    client = _FakeClient()
    ex = Executor(client, env="demo", dry_run=False)
    t = ex.place(_buy(Side.YES))
    assert t.placed is True and len(client.orders) == 1


def test_prod_requires_allow_prod_at_construction():
    with pytest.raises(RuntimeError):
        Executor(_FakeClient(), env="prod", dry_run=False, allow_prod=False)


def test_duplicate_order_409_is_benign():
    from hedge.kalshi.client import KalshiError

    class _Dup(_FakeClient):
        def create_order(self, **body):
            raise KalshiError(409, '{"code":"order_already_exists"}', "POST", "/x")

    t = Executor(_Dup(), env="demo", dry_run=False).place(_buy(Side.YES))
    assert t.placed is False and t.meta.get("idempotent") is True
    assert "duplicate" in t.error


def test_hold_decision_is_noop_ticket():
    ex = Executor(_FakeClient(), env="demo", dry_run=False)
    t = ex.place(Decision(ticker="MKT", action=Action.HOLD, reason="no edge"))
    assert t.placed is False and t.body == {}


# --------------------------------------------------------------------------- #
# Runner cycle (with fakes — no network)                                        #
# --------------------------------------------------------------------------- #
class _BullStrategy(Strategy):
    name = "bull"

    def evaluate(self, market: MarketView) -> Signal | None:
        return Signal(ticker=market.ticker, prob=0.80, std_error=0.01, strategy=self.name)


class _RunnerClient(_FakeClient):
    """Adds market discovery hooks the Runner uses."""
    pass


def _market_raw(ticker: str) -> dict:
    return {"ticker": ticker, "yes_bid": 53, "yes_ask": 55}


def test_runner_cycle_dry_run_reports_trade(monkeypatch, tmp_path):
    import hedge.runner as runner_mod

    client = _RunnerClient()
    # Stub discovery so we don't hit Kalshi: one open market across one series.
    monkeypatch.setattr(runner_mod, "STATIONS", {"KXHIGHNY": object()})
    monkeypatch.setattr(runner_mod, "discover_temp_markets",
                        lambda c, s, status: [_market_raw("KXHIGHNY-26JUN29-T80")])
    monkeypatch.setattr(runner_mod, "LIVE_DIR", tmp_path)

    ex = Executor(client, env="demo", dry_run=True)
    runner = Runner(client, [_BullStrategy()], ex, RiskConfig(),
                    bankroll_override=1000.0)
    tickets = runner.run_cycle()

    assert len(tickets) == 1
    t = tickets[0]
    assert t.decision.action is Action.BUY and t.decision.side is Side.YES
    assert t.placed is False  # dry run
    assert client.orders == []  # nothing actually sent
    # decision log written
    assert list(tmp_path.glob("decisions_*.jsonl"))


def test_runner_cycle_live_places_order(monkeypatch, tmp_path):
    import hedge.runner as runner_mod

    client = _RunnerClient()
    monkeypatch.setattr(runner_mod, "STATIONS", {"KXHIGHNY": object()})
    monkeypatch.setattr(runner_mod, "discover_temp_markets",
                        lambda c, s, status: [_market_raw("KXHIGHNY-26JUN29-T80")])
    monkeypatch.setattr(runner_mod, "LIVE_DIR", tmp_path)

    ex = Executor(client, env="demo", dry_run=False)
    runner = Runner(client, [_BullStrategy()], ex, RiskConfig(), bankroll_override=1000.0)
    runner.run_cycle()
    assert len(client.orders) == 1 and client.orders[0]["side"] == "bid"  # V2: buy YES = bid


def _wire_market(monkeypatch, runner_mod, ticker, tmp_path):
    monkeypatch.setattr(runner_mod, "STATIONS", {ticker.split("-", 1)[0]: object()})
    monkeypatch.setattr(runner_mod, "discover_temp_markets",
                        lambda c, s, status: [_market_raw(ticker)])
    monkeypatch.setattr(runner_mod, "LIVE_DIR", tmp_path)


def test_prod_blocks_unvalidated_station(monkeypatch, tmp_path):
    # KXHIGHSEA is not in the station map (station_for_ticker -> None); a real PROD
    # order on an unvalidated/unknown station must be refused.
    import hedge.runner as runner_mod
    client = _RunnerClient()
    _wire_market(monkeypatch, runner_mod, "KXHIGHSEA-26JUN29-T80", tmp_path)
    ex = Executor(client, env="prod", dry_run=False, allow_prod=True)
    runner = Runner(client, [_BullStrategy()], ex, RiskConfig(), bankroll_override=1000.0)
    tickets = runner.run_cycle()
    assert client.orders == []  # gated — nothing sent
    assert tickets and tickets[0].decision.action is Action.HOLD
    assert "not validated" in tickets[0].decision.reason


def test_prod_allows_validated_station(monkeypatch, tmp_path):
    # KXHIGHCHI is validated=True -> a real PROD order goes through.
    import hedge.runner as runner_mod
    client = _RunnerClient()
    _wire_market(monkeypatch, runner_mod, "KXHIGHCHI-26JUN29-T80", tmp_path)
    ex = Executor(client, env="prod", dry_run=False, allow_prod=True)
    runner = Runner(client, [_BullStrategy()], ex, RiskConfig(), bankroll_override=1000.0)
    runner.run_cycle()
    assert len(client.orders) == 1


def test_demo_still_trades_unvalidated_station(monkeypatch, tmp_path):
    # The gate only applies to real PROD money; demo trades unvalidated for data.
    import hedge.runner as runner_mod
    client = _RunnerClient()
    _wire_market(monkeypatch, runner_mod, "KXHIGHSEA-26JUN29-T80", tmp_path)
    ex = Executor(client, env="demo", dry_run=False)
    runner = Runner(client, [_BullStrategy()], ex, RiskConfig(), bankroll_override=1000.0)
    runner.run_cycle()
    assert len(client.orders) == 1


# --------------------------------------------------------------------------- #
# Durable state: anti-stack, reconciliation, daily-loss stop, settlement       #
# --------------------------------------------------------------------------- #
def _runner(client, tmp_path, monkeypatch, ticker, cfg=None):
    import hedge.runner as runner_mod
    _wire_market(monkeypatch, runner_mod, ticker, tmp_path)
    ex = Executor(client, env="demo", dry_run=False)
    return Runner(client, [_BullStrategy()], ex, cfg or RiskConfig(), bankroll_override=1000.0)


def test_anti_stack_blocks_second_order_on_open_ticker(monkeypatch, tmp_path):
    t = "KXHIGHCHI-26JUN29-T80"
    client = _RunnerClient()
    r = _runner(client, tmp_path, monkeypatch, t)
    # A still-working order with no order_id survives reconciliation (skipped),
    # so the cycle must NOT place a second order on the same ticker/side.
    r.state.record_order("seed", t, "yes", "buy", 55, 4, status="resting")
    r.run_cycle()
    assert client.orders == []  # blocked by anti-stack


def test_reconcile_cancels_stale_resting_order(monkeypatch, tmp_path):
    t = "KXHIGHCHI-26JUN29-T80"
    client = _RunnerClient()
    client.listing = [{"order_id": "OID-1", "status": "resting", "fill_count_fp": 0}]
    r = _runner(client, tmp_path, monkeypatch, t)
    r.state.record_order("seed", t, "yes", "buy", 55, 4, order_id="OID-1", status="resting")
    r.run_cycle()
    assert "OID-1" in client.canceled                       # cancel-replace fired
    assert r.state.get_order("seed")["status"] == "canceled"


def test_reconcile_marks_executed_fill(monkeypatch, tmp_path):
    t = "KXHIGHCHI-26JUN29-T80"
    client = _RunnerClient()
    client.listing = [{"order_id": "OID-2", "status": "executed", "fill_count_fp": 4}]
    r = _runner(client, tmp_path, monkeypatch, t)
    r.state.record_order("seed", t, "yes", "buy", 55, 4, order_id="OID-2", status="resting")
    r.run_cycle()
    row = r.state.get_order("seed")
    assert row["status"] == "executed" and row["fill_count"] == 4
    assert "OID-2" not in client.canceled                   # executed != cancelled


def test_reconcile_closes_order_absent_from_listing(monkeypatch, tmp_path):
    t = "KXHIGHCHI-26JUN29-T80"
    client = _RunnerClient()
    client.listing = []                                     # terminal / aged out of the listing
    r = _runner(client, tmp_path, monkeypatch, t)
    r.state.record_order("seed", t, "yes", "buy", 55, 4, order_id="OID-3", status="resting")
    r.run_cycle()
    assert r.state.get_order("seed")["status"] == "closed"  # anti-stack released


def test_daily_loss_stop_halts_after_breach(monkeypatch, tmp_path):
    t = "KXHIGHCHI-26JUN29-T80"
    client = _RunnerClient()
    cfg = RiskConfig(daily_loss_stop_dollars=50.0)
    r = _runner(client, tmp_path, monkeypatch, t, cfg=cfg)
    # Book a losing day beyond the stop, then a cycle must halt and place nothing.
    r.state.book_settlement("OLD-LOSER", -60.0)
    r.run_cycle()
    assert client.orders == []
    assert r.is_halted()[0] is True


def test_settlement_booking_uses_actual_fills(monkeypatch, tmp_path):
    # P&L is booked from the fills table (broker truth), NOT the intended decision.
    t = "KXHIGHCHI-26JUN29-T80"
    client = _RunnerClient()
    client.results[t] = "yes"   # market settled YES
    r = _runner(client, tmp_path, monkeypatch, t)
    # Seed a real fill: bought 4 YES at 55c. (No order_id needed — settlement reads
    # the fills table directly, keyed by ticker.)
    r.state.record_fill("coid-x", t, "yes", "buy", 4, order_id="OID-X",
                        avg_price_cents=55.0, fee_cents=0.0, status="executed")
    r.run_cycle()
    # Won YES at 0.55 -> +0.45/contract * 4 = +$1.80 booked into today's P&L.
    assert abs(r.state.realized_today() - 1.80) < 1e-6


def test_settlement_ignores_unfilled_intent(monkeypatch, tmp_path):
    # A decision that never filled (no fills row) books NOTHING — the core win of
    # fill-based accounting over intent-based.
    t = "KXHIGHCHI-26JUN29-T80"
    client = _RunnerClient()
    client.results[t] = "yes"
    r = _runner(client, tmp_path, monkeypatch, t)
    r.run_cycle()
    assert r.state.realized_today() == 0.0


def test_reconcile_fills_records_broker_fill(monkeypatch, tmp_path):
    t = "KXHIGHCHI-26JUN29-T80"
    client = _RunnerClient()
    # We placed an order (so order_for_oid can match); broker reports a taker fill.
    client.fills = [{"order_id": "OID-7", "ticker": t, "side": "yes", "action": "buy",
                     "count": 3, "yes_price": 55, "no_price": 45, "is_taker": True}]
    r = _runner(client, tmp_path, monkeypatch, t)
    r.state.record_order("coid-7", t, "yes", "buy", 55, 3, order_id="OID-7", status="executed")
    r.run_cycle()
    rows = r.state.fills_for_ticker(t)
    assert len(rows) == 1 and rows[0]["fill_count"] == 3
    assert abs(rows[0]["avg_price_cents"] - 55.0) < 1e-9


def test_reconcile_fills_parses_live_kalshi_field_shape(monkeypatch, tmp_path):
    # Contract test pinned to the REAL prod /portfolio/fills payload (captured live
    # 2026-06-29): count is `count_fp` (a fixed-point STRING, can be fractional from
    # pro-rata maker matching), prices are `{yes,no}_price_dollars` (STRING dollars),
    # fee is `fee_cost` (STRING dollars), and `action`/`book_side` are YES-centric
    # (buying NO is action="sell"/book_side="ask"). Reading the old cents names —
    # the bug that left the fills table empty in prod and booked ZERO P&L — must fail
    # this test. We bought 13 NO @ 28c as a maker (fee 0).
    t = "KXHIGHAUS-26JUN29-B96.5"
    client = _RunnerClient()
    client.fills = [{
        "order_id": "OID-LIVE", "ticker": t, "market_ticker": t,
        "side": "no", "outcome_side": "no", "action": "sell", "book_side": "ask",
        "count_fp": "13.00", "no_price_dollars": "0.2800", "yes_price_dollars": "0.7200",
        "fee_cost": "0.000000", "is_taker": False,
    }]
    r = _runner(client, tmp_path, monkeypatch, t)
    r.state.record_order("coid-live", t, "no", "buy", 28, 13, order_id="OID-LIVE",
                         status="executed")
    r.run_cycle()
    rows = r.state.fills_for_ticker(t)
    assert len(rows) == 1
    assert rows[0]["fill_count"] == 13                       # parsed count_fp, not skipped
    assert abs(rows[0]["avg_price_cents"] - 28.0) < 1e-9     # no_price_dollars -> cents
    assert abs(rows[0]["fee_cents"] - 0.0) < 1e-9            # maker fill: real fee_cost = 0


def test_cycle_persists_decision_rows(monkeypatch, tmp_path):
    # Every market decided in a cycle writes a queryable decisions row.
    t = "KXHIGHCHI-26JUN29-T80"
    client = _RunnerClient()
    r = _runner(client, tmp_path, monkeypatch, t)
    r.run_cycle()
    today = __import__("datetime").datetime.now(
        __import__("zoneinfo").ZoneInfo("UTC")).strftime("%Y-%m-%d")
    rows = r.state.decisions_for(today)
    assert len(rows) == 1
    d = rows[0]
    assert d["ticker"] == t and d["action"] == "buy" and d["side"] == "yes"
    assert d["yes_bid"] == 0.53 and d["yes_ask"] == 0.55   # market snapshot captured
    assert d["prob"] is not None and d["client_order_id"]


# --------------------------------------------------------------------------- #
# Position management: read holdings correctly, re-evaluate + exit intraday     #
# --------------------------------------------------------------------------- #
def test_positions_reads_fixed_point_fields(monkeypatch, tmp_path):
    # The live Kalshi payload reports holdings as `position_fp` (signed STRING) and
    # cost basis as `market_exposure_dollars`. Reading the old `position`/
    # `market_exposure` names returned an empty dict — the bug that left every
    # position unmanaged because the engine was never handed a Position.
    t = "KXHIGHCHI-26JUN29-T80"
    client = _RunnerClient()
    client.positions_payload = [
        {"ticker": t, "position_fp": "-4.00", "market_exposure_dollars": "1.80"},
        {"ticker": "KXHIGHMIA-26JUN29-B91.5", "position_fp": "458.00",
         "market_exposure_dollars": "7.33"},
    ]
    r = _runner(client, tmp_path, monkeypatch, t)
    pos = r.positions()
    assert pos[t].side is Side.NO and pos[t].count == 4
    assert abs(pos[t].avg_price - 0.45) < 1e-9            # 1.80 / 4
    mia = pos["KXHIGHMIA-26JUN29-B91.5"]
    assert mia.side is Side.YES and mia.count == 458


def test_positions_falls_back_to_legacy_fields(monkeypatch, tmp_path):
    # Older payloads: integer `position` (contracts) + `market_exposure` (cents).
    t = "KXHIGHCHI-26JUN29-T80"
    client = _RunnerClient()
    client.positions_payload = [{"ticker": t, "position": -3, "market_exposure": 135}]
    r = _runner(client, tmp_path, monkeypatch, t)
    pos = r.positions()
    assert pos[t].side is Side.NO and pos[t].count == 3
    assert abs(pos[t].avg_price - 0.45) < 1e-9            # 135c / 3 = 45c


_MANAGE = RiskConfig(manage_positions=True)   # intraday management enabled (gated OFF by default)


def test_held_position_flips_to_exit_when_model_reverses(monkeypatch, tmp_path):
    # Hold 4 NO; the bull model now wants YES with edge -> the engine SELLs the held
    # NO to exit (intraday reversal management), not a blind hold to settlement.
    t = "KXHIGHCHI-26JUN29-T80"
    client = _RunnerClient()
    client.positions_payload = [
        {"ticker": t, "position_fp": "-4.00", "market_exposure_dollars": "1.80"}]
    r = _runner(client, tmp_path, monkeypatch, t, cfg=_MANAGE)
    tickets = r.run_cycle()
    tk = next(x for x in tickets if x.decision.ticker == t)
    assert tk.decision.action is Action.SELL and tk.decision.side is Side.NO
    assert tk.decision.count == 4 and tk.placed is True
    assert tk.decision.price_cents == 45            # exits at no_bid = 1 - yes_ask(.55)
    assert len(client.orders) == 1


def test_management_off_by_default_holds_position(monkeypatch, tmp_path):
    # With the default RiskConfig (manage_positions=False) the SAME reversing position
    # is NOT touched — the engine is never handed the position, so no exit is generated.
    t = "KXHIGHCHI-26JUN29-T80"
    client = _RunnerClient()
    client.positions_payload = [
        {"ticker": t, "position_fp": "-4.00", "market_exposure_dollars": "1.80"}]
    r = _runner(client, tmp_path, monkeypatch, t)          # default cfg: management OFF
    tickets = r.run_cycle()
    tk = next(x for x in tickets if x.decision.ticker == t)
    # Flat-treated -> the bull model just opens YES (or holds); it never SELLs the NO.
    assert tk.decision.action is not Action.SELL


def test_manages_held_ticker_absent_from_discovery(monkeypatch, tmp_path):
    # Discovery returns nothing (market aged out / fresh redeploy), but we still hold
    # the ticker. It must be pulled directly and re-evaluated -> exited, not stranded.
    import hedge.runner as runner_mod
    t = "KXHIGHCHI-26JUN29-T80"
    client = _RunnerClient()
    client.market_fields = {t: {"yes_bid": 53, "yes_ask": 55}}
    client.positions_payload = [
        {"ticker": t, "position_fp": "-4.00", "market_exposure_dollars": "1.80"}]
    monkeypatch.setattr(runner_mod, "STATIONS", {"KXHIGHCHI": object()})
    monkeypatch.setattr(runner_mod, "discover_temp_markets", lambda c, s, status: [])
    monkeypatch.setattr(runner_mod, "LIVE_DIR", tmp_path)
    ex = Executor(client, env="demo", dry_run=False)
    r = Runner(client, [_BullStrategy()], ex, _MANAGE, bankroll_override=1000.0)
    tickets = r.run_cycle()
    tk = next(x for x in tickets if x.decision.ticker == t)
    assert tk.decision.action is Action.SELL and tk.decision.side is Side.NO
    assert tk.decision.count == 4 and tk.placed is True
    assert len(client.orders) == 1


def test_sell_to_exit_not_blocked_on_unvalidated_prod_station(monkeypatch, tmp_path):
    # The validated-station gate must NEVER trap real money: a risk-reducing SELL on
    # an unvalidated/unknown station still executes (only opening BUYs are gated).
    import hedge.runner as runner_mod
    t = "KXHIGHSEA-26JUN29-T80"   # not in the station map -> unvalidated
    client = _RunnerClient()
    client.market_fields = {t: {"yes_bid": 53, "yes_ask": 55}}
    client.positions_payload = [
        {"ticker": t, "position_fp": "-4.00", "market_exposure_dollars": "1.80"}]
    monkeypatch.setattr(runner_mod, "STATIONS", {"KXHIGHSEA": object()})
    monkeypatch.setattr(runner_mod, "discover_temp_markets", lambda c, s, status: [])
    monkeypatch.setattr(runner_mod, "LIVE_DIR", tmp_path)
    ex = Executor(client, env="prod", dry_run=False, allow_prod=True)
    r = Runner(client, [_BullStrategy()], ex, _MANAGE, bankroll_override=1000.0)
    tickets = r.run_cycle()
    tk = next(x for x in tickets if x.decision.ticker == t)
    assert tk.decision.action is Action.SELL and tk.decision.side is Side.NO
    assert tk.decision.count == 4 and tk.placed is True
