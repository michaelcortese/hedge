"""Strategy-level behavior: ensemble shape, nowcast floor/abstention, blend routing.

All offline — strategies are driven by fake forecast sources so the tests are fast
and deterministic (no network).
"""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

from hedge.strategies.base import MarketView
from hedge.strategies.weather_blend import WeatherBlendStrategy
from hedge.strategies.weather_ensemble import WeatherEnsembleStrategy
from hedge.strategies.weather_nowcast import WeatherNowcastStrategy

NY_TZ = ZoneInfo("America/New_York")


class FakeSource:
    def __init__(self, highs, obs=None):
        self._highs = highs
        self._obs = obs

    def point_highs(self, station, target_date):
        return list(self._highs)

    def observed_max(self, station, target_date):
        return self._obs


def _bucket(lo, hi, ev_date="25JUN28"):
    return MarketView("t", {
        "ticker": f"KXHIGHNY-{ev_date}-X", "event_ticker": f"KXHIGHNY-{ev_date}",
        "strike_type": "between", "floor_strike": lo, "cap_strike": hi,
    })


def test_ensemble_abstains_with_one_model():
    strat = WeatherEnsembleStrategy(source=FakeSource([80.0]), as_of=date(2025, 6, 27))
    assert strat.evaluate(_bucket(80, 80)) is None


def test_ensemble_peaks_at_mean():
    strat = WeatherEnsembleStrategy(source=FakeSource([79, 81, 80, 82]),
                                    as_of=date(2025, 6, 27))
    p_center = strat.evaluate(_bucket(80, 81)).prob
    p_tail = strat.evaluate(_bucket(90, 91)).prob
    assert p_center > p_tail


def test_nowcast_abstains_before_min_hour():
    morning = datetime(2025, 6, 28, 8, tzinfo=NY_TZ)
    strat = WeatherNowcastStrategy(source=FakeSource([85], obs=70),
                                   now=morning, min_hour=12)
    assert strat.evaluate(_bucket(84, 86)) is None


def test_nowcast_abstains_without_observations():
    afternoon = datetime(2025, 6, 28, 15, tzinfo=NY_TZ)
    strat = WeatherNowcastStrategy(source=FakeSource([85], obs=None), now=afternoon)
    assert strat.evaluate(_bucket(84, 86)) is None


def test_nowcast_floor_zeroes_buckets_below_obs_max():
    afternoon = datetime(2025, 6, 28, 15, tzinfo=NY_TZ)
    strat = WeatherNowcastStrategy(source=FakeSource([85], obs=83), now=afternoon)
    # A bucket entirely below the observed max can't win -> ~0 (clamped).
    sig_low = strat.evaluate(_bucket(75, 78))
    sig_at = strat.evaluate(_bucket(83, 85))
    assert sig_low.prob < 0.01
    assert sig_at.prob > sig_low.prob


def test_blend_routes_to_nowcast_in_afternoon():
    afternoon = datetime(2025, 6, 28, 15, tzinfo=NY_TZ)
    strat = WeatherBlendStrategy(source=FakeSource([85], obs=83),
                                 as_of=date(2025, 6, 27), now=afternoon)
    sig = strat.evaluate(_bucket(83, 85))
    assert sig is not None and sig.meta.get("via") == "nowcast"


def test_blend_falls_back_to_ensemble_in_morning():
    morning = datetime(2025, 6, 28, 8, tzinfo=NY_TZ)
    strat = WeatherBlendStrategy(source=FakeSource([84, 85, 86, 83], obs=70),
                                 as_of=date(2025, 6, 27), now=morning)
    sig = strat.evaluate(_bucket(84, 86))
    assert sig is not None and sig.meta.get("via") == "ensemble"
