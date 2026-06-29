"""Calibration kill-switch: Brier assessment, trip thresholds, latch, flatten."""

from __future__ import annotations

import json

from hedge.decision import Action, RiskConfig, Side
from hedge.execution import Executor
from hedge.guard import GuardConfig, GuardStatus, assess, brier_score
from hedge.runner import Runner
from hedge.signal import Signal
from hedge.strategies.base import MarketView, Strategy


# --------------------------------------------------------------------------- #
# Pure assessment                                                               #
# --------------------------------------------------------------------------- #
def test_brier_perfect_and_coin():
    assert brier_score([(1.0, True), (0.0, False)]) == 0.0
    assert brier_score([(0.5, True), (0.5, False)]) == 0.25
    assert brier_score([]) is None


def test_no_trip_below_min_samples():
    cfg = GuardConfig(min_samples=20, max_brier=0.25)
    # Wildly miscalibrated, but only a few samples -> must not trip yet.
    samples = [(0.9, False)] * 5
    s = assess(samples, cfg)
    assert s.tripped is False and "insufficient samples" in s.reason


def test_trips_when_brier_exceeds_ceiling():
    cfg = GuardConfig(min_samples=10, max_brier=0.25)
    samples = [(0.9, False)] * 30  # confidently wrong -> Brier 0.81
    s = assess(samples, cfg)
    assert s.tripped is True and s.brier and s.brier > 0.25


def test_calibrated_model_does_not_trip():
    cfg = GuardConfig(min_samples=10, max_brier=0.25)
    # Half resolve yes at p=0.5-ish but sharper: confident and usually right.
    samples = [(0.9, True)] * 18 + [(0.9, False)] * 2  # Brier ~ 0.1
    s = assess(samples, cfg)
    assert s.tripped is False


def test_baseline_threshold_is_tighter_than_absolute():
    base = GuardConfig(min_samples=5, baseline_brier=0.10, tolerance=0.05)
    assert base.threshold == 0.10 + 0.05
    samples = [(0.7, False)] * 10  # Brier 0.49 -> trips vs 0.15
    assert assess(samples, base).tripped is True


def test_disabled_guard_never_trips():
    cfg = GuardConfig(enabled=False, min_samples=1, max_brier=0.01)
    assert assess([(0.9, False)] * 50, cfg).tripped is False


# --------------------------------------------------------------------------- #
# Runner integration                                                            #
# --------------------------------------------------------------------------- #
class _FakeClient:
    def __init__(self, results=None, positions=None):
        self.orders = []
        self._results = results or {}
        self._positions = positions or []

    def create_order(self, **body):
        self.orders.append(body)
        return {"order": {"order_id": "OID"}}

    def get_balance(self):
        return {"balance": 1_000_00}

    def get_positions(self):
        return {"market_positions": self._positions}

    def get_market(self, ticker):
        m = {"ticker": ticker, "yes_bid": 53, "yes_ask": 55}
        if ticker in self._results:
            m["result"] = self._results[ticker]
        return {"market": m}


class _Bull(Strategy):
    name = "bull"

    def evaluate(self, market: MarketView) -> Signal | None:
        return Signal(ticker=market.ticker, prob=0.80, std_error=0.01, strategy=self.name)


def _runner(tmp_path, monkeypatch, client, guard_cfg, dry_run=True):
    import hedge.runner as rm
    monkeypatch.setattr(rm, "LIVE_DIR", tmp_path)
    monkeypatch.setattr(rm, "STATIONS", {"KX": object()})
    monkeypatch.setattr(rm, "discover_temp_markets",
                        lambda c, s, status: [{"ticker": "KX-T80", "yes_bid": 53, "yes_ask": 55}])
    ex = Executor(client, env="demo", dry_run=dry_run)
    return Runner(client, [_Bull()], ex, RiskConfig(), bankroll_override=1000.0,
                  guard_cfg=guard_cfg)


def _write_log(tmp_path, ticker, prob, action="buy"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    row = {"ticker": ticker, "prob": prob, "action": action, "ts": "2026-06-20T00:00:00+00:00"}
    (tmp_path / "decisions_2026-06-20.jsonl").write_text(json.dumps(row) + "\n")


def test_runner_trips_and_latches(tmp_path, monkeypatch):
    # Log 30 confidently-wrong settled trades -> guard should trip and latch.
    rows = [{"ticker": f"KX-T{i}", "prob": 0.9, "action": "buy",
             "ts": "2026-06-20T00:00:00+00:00"} for i in range(30)]
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "decisions_2026-06-20.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")
    results = {f"KX-T{i}": "no" for i in range(30)}  # all resolved NO -> we were wrong

    client = _FakeClient(results=results)
    guard = GuardConfig(min_samples=10, max_brier=0.25)
    runner = _runner(tmp_path, monkeypatch, client, guard)

    tickets = runner.run_cycle()
    assert tickets == []                       # trading skipped
    assert client.orders == []                 # nothing placed
    halted, reason = runner.is_halted()
    assert halted and "Brier" in reason

    # Latches: a second cycle stays halted even before re-evaluating.
    assert runner.run_cycle() == []
    assert runner.reset_guard() is True        # manual clear
    assert runner.is_halted()[0] is False


def test_runner_trades_when_calibrated(tmp_path, monkeypatch):
    # Well-calibrated history -> guard passes -> normal trading proceeds (dry-run).
    rows = [{"ticker": f"KX-T{i}", "prob": 0.9, "action": "buy",
             "ts": "2026-06-20T00:00:00+00:00"} for i in range(20)]
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "decisions_2026-06-20.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")
    results = {f"KX-T{i}": ("yes" if i >= 2 else "no") for i in range(20)}  # mostly right

    client = _FakeClient(results=results)
    runner = _runner(tmp_path, monkeypatch, client, GuardConfig(min_samples=10, max_brier=0.25))
    tickets = runner.run_cycle()
    assert any(t.decision.action is Action.BUY for t in tickets)


def test_flatten_on_trip_sells_positions(tmp_path, monkeypatch):
    rows = [{"ticker": f"KX-T{i}", "prob": 0.9, "action": "buy",
             "ts": "2026-06-20T00:00:00+00:00"} for i in range(30)]
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "decisions_2026-06-20.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")
    results = {f"KX-T{i}": "no" for i in range(30)}
    positions = [{"ticker": "HELD-1", "position": 5, "market_exposure": 275}]

    client = _FakeClient(results=results, positions=positions)
    guard = GuardConfig(min_samples=10, max_brier=0.25, flatten_on_trip=True)
    runner = _runner(tmp_path, monkeypatch, client, guard, dry_run=False)
    runner.run_cycle()
    # One sell order for the held YES position.
    assert len(client.orders) == 1
    o = client.orders[0]
    # V2: closing a long YES position is an ask (sell YES); count is a fixed-point string.
    assert o["side"] == "ask" and o["count"] == "5.00" and "action" not in o
