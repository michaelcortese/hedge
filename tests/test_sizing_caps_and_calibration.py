"""Tests for the catastrophe-proof sizing (#2) and honest-sigma (#1) changes:

- order-book parsing into MarketView.book_top / MarketQuote.from_view,
- the order-book participation cap and the per-event (city-day) concentration cap,
- the calibration sigma floor and the settlement-basis widening of the predictive
  distribution / reported std_error.
"""

from __future__ import annotations

import datetime as dt

import pytest

from hedge.decision import Action, MarketQuote, RiskConfig, Side, decide
from hedge.signal import Signal
from hedge.strategies.base import MarketView
from hedge.weather import calibration as calib_mod
from hedge.weather.calibration import (
    SETTLEMENT_SIGMA_DEFAULT_F,
    SIGMA_FLOOR_F,
    CalibrationTable,
    fit_calibration,
)
from hedge.weather.distribution import bucket_prob_and_se, build_distribution
from hedge.weather.markets import TempMarket
from hedge.weather.stations import STATIONS

BANKROLL = 10_000.0


def _sig(prob: float, se: float = 0.01, ticker: str = "KXHIGHMIA-26JUN30-B70.5") -> Signal:
    return Signal(ticker=ticker, prob=prob, std_error=se, strategy="t")


# --------------------------------------------------------------------------- #
# Order-book parsing                                                          #
# --------------------------------------------------------------------------- #
def _view_with_book(yes_levels, no_levels) -> MarketView:
    return MarketView(
        "KXHIGHMIA-26JUN30-B70.5", {},
        orderbook={"orderbook": {"yes": yes_levels, "no": no_levels}},
    )


def test_book_top_reconstructs_ask_and_depths():
    # best YES bid = 45c (depth 200); best NO bid = 52c (depth 150).
    v = _view_with_book([[40, 100], [45, 200]], [[50, 300], [52, 150]])
    top = v.book_top()
    assert top["yes_bid"] == pytest.approx(0.45)
    assert top["yes_bid_depth"] == 200
    # yes_ask is reconstructed from the best NO bid: 1 - 0.52 = 0.48.
    assert top["yes_ask"] == pytest.approx(0.48)
    assert top["no_bid_depth"] == 150


def test_quote_from_view_maps_book_depths():
    v = _view_with_book([[40, 100], [45, 200]], [[50, 300], [52, 150]])
    q = MarketQuote.from_view(v)
    assert q is not None
    assert q.yes_bid == pytest.approx(0.45) and q.yes_ask == pytest.approx(0.48)
    # Buying YES as taker lifts the resting NO bids -> depth = no-bid size (150).
    assert q.yes_ask_depth == 150
    # Joining the YES bid rests behind the resting YES size (200).
    assert q.yes_bid_depth == 200


def test_quote_from_view_rejects_crossed_book():
    # YES bid 55c, NO bid 50c -> yes_ask = 0.50 < yes_bid 0.55: locked/crossed.
    v = _view_with_book([[55, 100]], [[50, 100]])
    assert MarketQuote.from_view(v) is None


def test_book_preferred_over_stale_market_payload():
    # Stale GET /markets says 0.10/0.90; the live book says 0.45/0.48 -> use the book.
    v = MarketView(
        "KXHIGHMIA-26JUN30-B70.5",
        {"yes_bid": 10, "yes_ask": 90},
        orderbook={"orderbook": {"yes": [[45, 50]], "no": [[52, 50]]}},
    )
    q = MarketQuote.from_view(v)
    assert q.yes_bid == pytest.approx(0.45) and q.yes_ask == pytest.approx(0.48)


# --------------------------------------------------------------------------- #
# Participation cap                                                           #
# --------------------------------------------------------------------------- #
def test_participation_cap_shrinks_to_fraction_of_depth():
    # Both depths set so the cap binds whichever side (maker/taker) the engine picks.
    q = MarketQuote(yes_bid=0.50, yes_ask=0.52, yes_bid_depth=40, yes_ask_depth=40)
    cfg = RiskConfig(lambda_kelly=0.5, k_sigma=1.0, tau_min_cents=1.0,
                     participation_frac=0.25)
    d = decide(_sig(0.95), q, BANKROLL, cfg)
    assert d.action is Action.BUY and d.side is Side.YES
    assert d.count <= 10                      # 25% of 40 resting contracts


def test_full_depth_cap_when_participation_unset():
    q = MarketQuote(yes_bid=0.50, yes_ask=0.52, yes_bid_depth=40, yes_ask_depth=40)
    cfg = RiskConfig(lambda_kelly=0.5, k_sigma=1.0, tau_min_cents=1.0)  # no participation
    d = decide(_sig(0.95), q, BANKROLL, cfg)
    assert d.action is Action.BUY and d.count <= 40 and d.count > 10


# --------------------------------------------------------------------------- #
# Per-event (city-day) concentration cap                                      #
# --------------------------------------------------------------------------- #
def test_event_cap_blocks_when_sibling_buckets_full():
    q = MarketQuote(yes_bid=0.50, yes_ask=0.52)
    cfg = RiskConfig(lambda_kelly=0.5, k_sigma=1.0, tau_min_cents=1.0,
                     event_cap_frac=0.06)
    # The other buckets of this city-day already hold the full 6% event budget.
    d = decide(_sig(0.95), q, BANKROLL, cfg, event_at_risk=0.06 * BANKROLL)
    assert d.action is Action.HOLD and "event cap" in d.reason


def test_event_cap_allows_when_room_remains():
    q = MarketQuote(yes_bid=0.50, yes_ask=0.52)
    cfg = RiskConfig(lambda_kelly=0.5, k_sigma=1.0, tau_min_cents=1.0,
                     event_cap_frac=0.06)
    d = decide(_sig(0.95), q, BANKROLL, cfg, event_at_risk=0.0)
    assert d.action is Action.BUY and d.count > 0


# --------------------------------------------------------------------------- #
# #1 — honest sigma                                                           #
# --------------------------------------------------------------------------- #
def _tempmarket(lo: float, hi: float) -> TempMarket:
    st = STATIONS["KXHIGHMIA"]
    return TempMarket("KXHIGHMIA-26JUN30-B70.5", "KXHIGHMIA", st,
                      dt.date(2026, 6, 30), lo, hi, "between")


def test_settlement_sigma_widens_pmf_and_lowers_center_confidence():
    # The protective effect: a wider settlement-basis spread lowers the confident mass
    # piled on the center bucket (so |p-mid| shrinks and Kelly sizes smaller).
    pts = [70.0, 70.2, 69.8]
    narrow = build_distribution(pts, model_sigma=1.0, n_draws=40_000, seed=1)
    wide = build_distribution(pts, model_sigma=1.0, n_draws=40_000, seed=1,
                              settlement_sigma_f=3.0)
    assert wide.sigma > narrow.sigma
    mkt = _tempmarket(70.0, 71.0)
    assert wide.prob_for_market(mkt) < narrow.prob_for_market(mkt)


def test_structural_se_adds_to_reported_std_error():
    pts = [70.0, 70.2, 69.8]
    mkt = _tempmarket(70.0, 71.0)
    _, se0 = bucket_prob_and_se(pts, mkt, model_sigma=2.0, n_draws=40_000, seed=1)
    _, se1 = bucket_prob_and_se(pts, mkt, model_sigma=2.0, n_draws=40_000, seed=1,
                                structural_se=0.05)
    # Same seed -> sampling + param identical; structural_se folds in by quadrature.
    assert se1 == pytest.approx((se0**2 + 0.05**2) ** 0.5, abs=1e-9)
    assert se1 > se0


def test_sigma_floor_binds_on_correlated_collapse(monkeypatch):
    """A near-zero residual (forecast == truth, the correlated-source collapse) must
    floor to SIGMA_FLOOR_F, not produce an over-confident ~0°F spread."""
    days = [dt.date(2026, 5, 1) + dt.timedelta(days=i) for i in range(35)]
    realized = {d.isoformat(): 70.0 for d in days}
    forecasts = {d.isoformat(): [70.0, 70.05] for d in days}  # ~zero residual std

    monkeypatch.setattr("hedge.weather.archive.archive_daily_highs",
                        lambda st, s, e: realized)
    monkeypatch.setattr("hedge.weather.archive.historical_model_highs_range",
                        lambda st, s, e: forecasts)

    table = fit_calibration([STATIONS["KXHIGHMIA"]], days[0], days[-1])
    assert table.sigma_for("KXHIGHMIA", 1) == pytest.approx(SIGMA_FLOOR_F)
    # The lead-0 (same-day) prior is 2.0; floor must not let any lead drop under it.
    assert table.sigma_for("KXHIGHMIA", 0) >= SIGMA_FLOOR_F


def test_station_truth_source_used_when_requested(monkeypatch):
    """truth='station' must score against the IEM station max, not ERA5."""
    days = [dt.date(2026, 5, 1) + dt.timedelta(days=i) for i in range(35)]
    era5 = {d.isoformat(): 70.0 for d in days}
    station = {d.isoformat(): 73.0 for d in days}      # station runs 3°F warmer
    forecasts = {d.isoformat(): [70.0, 70.0] for d in days}

    monkeypatch.setattr("hedge.weather.archive.archive_daily_highs",
                        lambda st, s, e: era5)
    monkeypatch.setattr("hedge.weather.archive.historical_model_highs_range",
                        lambda st, s, e: forecasts)
    monkeypatch.setattr("hedge.weather.providers.iem_daily_max_f",
                        lambda st, s, e: station)

    table = fit_calibration([STATIONS["KXHIGHMIA"]], days[0], days[-1], truth="station")
    # bias = mean(forecast - station) = 70 - 73 = -3 -> the engine shifts center UP 3°F.
    assert table.bias_for("KXHIGHMIA", 1) == pytest.approx(-3.0, abs=0.2)
    # Against the station the basis is in the bias, so no extra settlement term.
    assert table.settlement_sigma_for("KXHIGHMIA") == pytest.approx(0.0)


def test_settlement_sigma_default_on_empty_table():
    # An unfit/empty table (the live calibration fallback) still gives forecast
    # strategies a conservative basis cushion rather than zero.
    assert CalibrationTable().settlement_sigma_for("KXHIGHMIA") == SETTLEMENT_SIGMA_DEFAULT_F
