"""Tests for #3 — concentrate on the one real edge:

- the deterministic "impossible/certain bucket" nowcast signal + engine band bypass,
- the per-strategy λ multiplier (zero-size morning strategies),
- the market-relative skill gate (guard.market_skill),
- the tightened nowcast afternoon window.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from zoneinfo import ZoneInfo

from hedge.decision import Action, MarketQuote, RiskConfig, Side, decide
from hedge.guard import GuardConfig, market_skill
from hedge.signal import Signal
from hedge.strategies.base import MarketView
from hedge.strategies.weather_nowcast import WeatherNowcastStrategy

NY_TZ = ZoneInfo("America/New_York")


class FakeSource:
    def __init__(self, highs, obs=None):
        self._highs, self._obs = highs, obs

    def point_highs(self, station, target_date):
        return list(self._highs)

    def observed_max(self, station, target_date):
        return self._obs


def _between(lo, hi):
    return MarketView("t", {
        "ticker": "KXHIGHNY-25JUN28-B", "event_ticker": "KXHIGHNY-25JUN28",
        "strike_type": "between", "floor_strike": lo, "cap_strike": hi,
    })


def _upper_tail(floor):
    return MarketView("t", {
        "ticker": "KXHIGHNY-25JUN28-T", "event_ticker": "KXHIGHNY-25JUN28",
        "strike_type": "greater", "floor_strike": floor,
    })


AFTERNOON = datetime(2025, 6, 28, 15, tzinfo=NY_TZ)


# --------------------------------------------------------------------------- #
# Deterministic impossible / certain buckets                                  #
# --------------------------------------------------------------------------- #
def test_nowcast_flags_impossible_bucket_deterministic():
    # Observed max 83 already rounds above a 75-78 bucket -> YES impossible.
    strat = WeatherNowcastStrategy(source=FakeSource([85], obs=83), now=AFTERNOON)
    sig = strat.evaluate(_between(75, 78))
    assert sig is not None and sig.deterministic
    assert sig.prob < 1e-3 and sig.meta["deterministic"] == "impossible"


def test_nowcast_flags_certain_upper_tail_deterministic():
    # Observed max 85 already meets an "85+/above 80" threshold -> YES certain.
    strat = WeatherNowcastStrategy(source=FakeSource([88], obs=85), now=AFTERNOON)
    sig = strat.evaluate(_upper_tail(80))
    assert sig is not None and sig.deterministic
    assert sig.prob > 0.999 and sig.meta["deterministic"] == "certain"


def test_nowcast_not_deterministic_when_bucket_still_live():
    # Observed max 83 sits inside an 83-86 bucket: the high can still rise above it,
    # so it is NOT logically settled -> probabilistic path, deterministic=False.
    strat = WeatherNowcastStrategy(source=FakeSource([85], obs=83), now=AFTERNOON)
    sig = strat.evaluate(_between(83, 86))
    assert sig is not None and not sig.deterministic


def test_engine_trades_deterministic_through_band():
    # A deterministic near-certain NO sits at ~0.95 (outside the [0.10,0.90] band).
    q = MarketQuote(yes_bid=0.05, yes_ask=0.07)
    sig = Signal(ticker="KXHIGHNY-25JUN28-B", prob=1e-4, std_error=1e-6,
                 strategy="weather_nowcast", deterministic=True)
    d = decide(sig, q, 10_000.0, RiskConfig())
    assert d.action is Action.BUY and d.side is Side.NO and d.count > 0

    # The SAME signal without the deterministic flag is blocked by the price band.
    plain = replace(sig, deterministic=False)
    assert decide(plain, q, 10_000.0, RiskConfig()).action is Action.HOLD


# --------------------------------------------------------------------------- #
# Per-strategy λ multiplier                                                   #
# --------------------------------------------------------------------------- #
def test_zero_lambda_holds_with_clear_reason():
    q = MarketQuote(yes_bid=0.50, yes_ask=0.52)
    sig = Signal(ticker="MKT", prob=0.95, std_error=0.01, strategy="weather_ensemble")
    cfg0 = replace(RiskConfig(k_sigma=1.0, tau_min_cents=1.0), lambda_kelly=0.0)
    d = decide(sig, q, 10_000.0, cfg0)
    assert d.action is Action.HOLD and "zero target size" in d.reason
    # Same signal at full λ would trade — confirms it's the λ, not the edge.
    cfg1 = replace(RiskConfig(k_sigma=1.0, tau_min_cents=1.0), lambda_kelly=0.25)
    assert decide(sig, q, 10_000.0, cfg1).action is Action.BUY


# --------------------------------------------------------------------------- #
# Market-relative skill gate                                                  #
# --------------------------------------------------------------------------- #
def _samples(model_p, mid, n, yes):
    return [(model_p, mid, yes)] * n


def test_skill_gate_disabled_is_full_size():
    cfg = GuardConfig(skill_gate=False)
    assert market_skill(_samples(0.9, 0.5, 50, True), cfg).multiplier == 1.0


def test_skill_gate_floors_until_enough_samples():
    cfg = GuardConfig(skill_gate=True, skill_min_samples=30, skill_floor=0.1)
    st = market_skill(_samples(0.9, 0.5, 10, True), cfg)
    assert st.multiplier == 0.1 and st.n == 10


def test_skill_gate_ramps_to_full_when_model_beats_market():
    cfg = GuardConfig(skill_gate=True, skill_min_samples=30, skill_full_at=0.02,
                      skill_floor=0.1)
    # Model nails it (p=0.99, outcome YES); market hedged at 0.6 -> big positive skill.
    st = market_skill(_samples(0.99, 0.6, 40, True), cfg)
    assert st.skill > 0 and st.multiplier == 1.0


def test_skill_gate_stays_at_floor_when_model_worse_than_market():
    cfg = GuardConfig(skill_gate=True, skill_min_samples=30, skill_floor=0.1)
    # Market is right (0.9 -> YES), model wrong (0.2) -> negative skill -> floor.
    st = market_skill(_samples(0.2, 0.9, 40, True), cfg)
    assert st.skill < 0 and st.multiplier == 0.1


# --------------------------------------------------------------------------- #
# Tightened afternoon window                                                  #
# --------------------------------------------------------------------------- #
def test_nowcast_default_min_hour_is_afternoon():
    one_pm = datetime(2025, 6, 28, 13, tzinfo=NY_TZ)
    strat = WeatherNowcastStrategy(source=FakeSource([85], obs=80), now=one_pm)
    assert strat.evaluate(_between(84, 86)) is None         # before 14:00 -> abstain
    strat_2pm = WeatherNowcastStrategy(
        source=FakeSource([85], obs=80),
        now=datetime(2025, 6, 28, 14, tzinfo=NY_TZ))
    assert strat_2pm.evaluate(_between(84, 86)) is not None  # 14:00 -> acts
