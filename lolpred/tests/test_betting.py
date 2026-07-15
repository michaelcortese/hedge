"""Tests for lolpred.backtest.betting (contract section 5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lolpred.backtest.betting import (
    bootstrap_roi_ci,
    devig_proportional,
    devig_shin,
    implied_prob,
    kelly_fraction,
    make_synthetic_odds,
    select_bets,
    settle_bets,
    simulate_bankroll,
)


# ---------------------------------------------------------------- odds math


class TestImpliedAndDevig:
    def test_implied_prob_scalar(self):
        assert implied_prob(2.0) == pytest.approx(0.5)
        assert implied_prob(4.0) == pytest.approx(0.25)

    def test_implied_prob_array(self):
        out = implied_prob(np.array([2.0, 5.0]))
        np.testing.assert_allclose(out, [0.5, 0.2])

    def test_devig_proportional_sums_to_one_and_preserves_ratio(self):
        imp_a, imp_b = 0.60, 0.50  # booksum 1.10
        fair_a, fair_b = devig_proportional(imp_a, imp_b)
        assert fair_a + fair_b == pytest.approx(1.0)
        assert fair_a / fair_b == pytest.approx(imp_a / imp_b)
        assert fair_a == pytest.approx(6.0 / 11.0)

    def test_devig_proportional_roundtrip_from_odds(self):
        # fair 0.6/0.4 with 5% proportional vig -> odds -> devig recovers fair
        imp_a, imp_b = 0.6 * 1.05, 0.4 * 1.05
        odds_a, odds_b = 1 / imp_a, 1 / imp_b
        fair_a, fair_b = devig_proportional(implied_prob(odds_a), implied_prob(odds_b))
        assert fair_a == pytest.approx(0.6)
        assert fair_b == pytest.approx(0.4)

    def test_devig_proportional_vectorized(self):
        fair_a, fair_b = devig_proportional(np.array([0.63, 0.525]), np.array([0.42, 0.525]))
        np.testing.assert_allclose(fair_a + fair_b, 1.0)
        np.testing.assert_allclose(fair_a, [0.6, 0.5])


class TestDevigShin:
    def test_sums_to_one(self):
        fair_a, fair_b = devig_shin(0.63, 0.42)
        assert fair_a + fair_b == pytest.approx(1.0)

    def test_no_vig_is_identity(self):
        fair_a, fair_b = devig_shin(0.7, 0.3)
        assert fair_a == pytest.approx(0.7, abs=1e-9)
        assert fair_b == pytest.approx(0.3, abs=1e-9)

    def test_close_to_proportional_at_small_vig(self):
        # 5% vig on a 0.6/0.4 fair line: Shin and proportional nearly agree.
        imp_a, imp_b = 0.6 * 1.05, 0.4 * 1.05
        shin_a, shin_b = devig_shin(imp_a, imp_b)
        prop_a, prop_b = devig_proportional(imp_a, imp_b)
        assert abs(shin_a - prop_a) < 0.02
        assert abs(shin_b - prop_b) < 0.02
        # Shin de-vigs the longshot more: favorite prob >= proportional's.
        assert shin_a >= prop_a


class TestKelly:
    def test_spot_value(self):
        # p=0.6 at evens: f* = (0.6*2 - 1)/(2 - 1) = 0.2
        assert kelly_fraction(0.6, 2.0) == pytest.approx(0.2)

    def test_zero_when_no_edge(self):
        assert kelly_fraction(0.5, 2.0) == pytest.approx(0.0)
        assert kelly_fraction(0.3, 2.0) == 0.0  # negative edge floored at 0

    def test_zero_when_odds_leq_one(self):
        assert kelly_fraction(0.99, 1.0) == 0.0
        assert kelly_fraction(0.99, 0.5) == 0.0

    def test_vectorized(self):
        out = kelly_fraction(np.array([0.6, 0.5, 0.9]), np.array([2.0, 2.0, 1.0]))
        np.testing.assert_allclose(out, [0.2, 0.0, 0.0])


# ------------------------------------------------------------ synthetic odds


class TestMakeSyntheticOdds:
    def setup_method(self):
        rng = np.random.default_rng(123)
        self.ref = rng.uniform(0.05, 0.95, size=200)

    def test_deterministic_given_seed(self):
        a = make_synthetic_odds(self.ref, seed=7)
        b = make_synthetic_odds(self.ref, seed=7)
        pd.testing.assert_frame_equal(a, b)

    def test_different_seed_differs(self):
        a = make_synthetic_odds(self.ref, seed=7)
        b = make_synthetic_odds(self.ref, seed=8)
        assert not np.allclose(a["odds_blue"], b["odds_blue"])

    def test_columns(self):
        df = make_synthetic_odds(self.ref)
        assert list(df.columns) == [
            "odds_blue", "odds_red", "imp_blue", "imp_red", "fair_blue", "fair_red",
        ]
        assert len(df) == len(self.ref)

    def test_overround_equals_one_plus_margin(self):
        margin = 0.05
        df = make_synthetic_odds(self.ref, margin=margin, seed=1)
        book = 1.0 / df["odds_blue"] + 1.0 / df["odds_red"]
        np.testing.assert_allclose(book, 1.0 + margin)
        np.testing.assert_allclose(df["imp_blue"] + df["imp_red"], 1.0 + margin)

    def test_fair_probs_sum_to_one(self):
        df = make_synthetic_odds(self.ref, seed=2)
        np.testing.assert_allclose(df["fair_blue"] + df["fair_red"], 1.0)

    def test_fair_probs_are_clipped(self):
        # extreme reference probs -> fair prob stays inside [0.02, 0.98]
        df = make_synthetic_odds(np.array([0.999, 0.001]), noise_sd=0.0, shrink=1.0)
        assert (df["fair_blue"] <= 0.98 + 1e-12).all()
        assert (df["fair_blue"] >= 0.02 - 1e-12).all()

    def test_shrink_pulls_toward_half(self):
        # with no noise, shrink < 1 moves the book's fair prob toward 0.5
        df = make_synthetic_odds(np.array([0.8]), shrink=0.5, noise_sd=0.0)
        expected = 1.0 / (1.0 + np.exp(-0.5 * np.log(0.8 / 0.2)))
        assert df["fair_blue"].iloc[0] == pytest.approx(expected)

    def test_preserves_series_index(self):
        ser = pd.Series(self.ref[:5], index=[10, 20, 30, 40, 50])
        df = make_synthetic_odds(ser)
        assert list(df.index) == [10, 20, 30, 40, 50]


# --------------------------------------------------------------- select_bets


class TestSelectBets:
    def test_picks_blue_side(self):
        # imp_blue = 0.5, model 0.7 -> blue edge 0.2; red edge is negative.
        bets = select_bets(np.array([0.7]), np.array([2.0]), np.array([2.0]))
        assert len(bets) == 1
        assert bets["side"].iloc[0] == "blue"
        assert bets["model_p"].iloc[0] == pytest.approx(0.7)
        assert bets["odds"].iloc[0] == pytest.approx(2.0)
        assert bets["edge"].iloc[0] == pytest.approx(0.2)

    def test_picks_red_side_with_red_probability(self):
        bets = select_bets(np.array([0.3]), np.array([2.0]), np.array([2.0]))
        assert bets["side"].iloc[0] == "red"
        assert bets["model_p"].iloc[0] == pytest.approx(0.7)  # prob of chosen side
        assert bets["edge"].iloc[0] == pytest.approx(0.2)

    def test_edge_is_vs_vigged_implied(self):
        # odds 1.8 -> vigged implied 5/9 ~ 0.5556; edge = 0.7 - 0.5556
        bets = select_bets(np.array([0.7]), np.array([1.8]), np.array([1.8]), min_edge=0.01)
        assert bets["edge"].iloc[0] == pytest.approx(0.7 - 1.0 / 1.8)

    def test_min_edge_threshold(self):
        # edge below threshold must not trade
        bets = select_bets(np.array([0.53]), np.array([2.0]), np.array([2.0]), min_edge=0.04)
        assert len(bets) == 0
        bets = select_bets(np.array([0.55]), np.array([2.0]), np.array([2.0]), min_edge=0.04)
        assert len(bets) == 1
        # edge exactly equal to min_edge must NOT trade (strictly greater
        # required); 0.5625 and 0.0625 are exactly representable in binary.
        bets = select_bets(np.array([0.5625]), np.array([2.0]), np.array([2.0]), min_edge=0.0625)
        assert len(bets) == 0

    def test_stake_capped_at_max_stake(self):
        # kelly(0.7, 2.0) = 0.4; 0.25 * 0.4 = 0.1 -> capped at 0.02
        bets = select_bets(np.array([0.7]), np.array([2.0]), np.array([2.0]),
                           kelly_mult=0.25, max_stake=0.02)
        assert bets["stake_frac"].iloc[0] == pytest.approx(0.02)

    def test_stake_uncapped_when_small(self):
        # kelly(0.55, 2.0) = 0.1; 0.25 * 0.1 = 0.025 < max_stake=0.5
        bets = select_bets(np.array([0.55]), np.array([2.0]), np.array([2.0]),
                           min_edge=0.01, kelly_mult=0.25, max_stake=0.5)
        assert bets["stake_frac"].iloc[0] == pytest.approx(0.025)

    def test_nonpositive_stake_dropped(self):
        # min_edge=-1 lets a negative-edge game through the threshold, but
        # kelly floors at 0 -> stake 0 -> dropped.
        bets = select_bets(np.array([0.5]), np.array([1.9]), np.array([1.9]), min_edge=-1.0)
        assert len(bets) == 0

    def test_never_bets_both_sides(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0.1, 0.9, size=300)
        odds = make_synthetic_odds(p, seed=3)
        bets = select_bets(p, odds["odds_blue"], odds["odds_red"], min_edge=0.0)
        assert bets.index.is_unique  # at most one bet per game

    def test_min_history_filter_both_teams(self):
        p = np.array([0.7, 0.7, 0.7, 0.7])
        ob = np.full(4, 2.0)
        orr = np.full(4, 2.0)
        hist_b = np.array([20, 5, 20, 5])
        hist_r = np.array([20, 20, 5, 5])
        bets = select_bets(p, ob, orr, hist_games_blue=hist_b, hist_games_red=hist_r,
                           min_hist_games=10)
        assert list(bets.index) == [0]  # only the game where BOTH have >= 10

    def test_no_history_filter_when_not_provided(self):
        p = np.array([0.7, 0.7])
        bets = select_bets(p, np.full(2, 2.0), np.full(2, 2.0))
        assert len(bets) == 2

    def test_carries_original_index(self):
        idx = pd.Index([101, 202, 303])
        p = pd.Series([0.7, 0.5, 0.25], index=idx)
        ob = pd.Series([2.0, 2.0, 2.0], index=idx)
        orr = pd.Series([2.0, 2.0, 2.0], index=idx)
        bets = select_bets(p, ob, orr)
        assert list(bets.index) == [101, 303]  # middle game has no edge
        assert list(bets["side"]) == ["blue", "red"]

    def test_columns(self):
        bets = select_bets(np.array([0.7]), np.array([2.0]), np.array([2.0]))
        assert list(bets.columns) == ["side", "model_p", "odds", "edge", "stake_frac"]

    def test_empty_result_shape(self):
        bets = select_bets(np.array([0.5]), np.array([2.0]), np.array([2.0]))
        assert len(bets) == 0
        assert list(bets.columns) == ["side", "model_p", "odds", "edge", "stake_frac"]


# ------------------------------------------------------- settle + bankroll


def _hand_bets() -> pd.DataFrame:
    """Three bets referencing games 0, 2, 4 of a 5-game slate."""
    return pd.DataFrame(
        {
            "side": ["blue", "red", "blue"],
            "model_p": [0.6, 0.6, 0.6],
            "odds": [2.0, 3.0, 1.5],
            "edge": [0.1, 0.1, 0.1],
            "stake_frac": [0.1, 0.2, 0.5],
        },
        index=[0, 2, 4],
    )


# blue_win over the 5 games: bet1 (blue@0) wins, bet2 (red@2) loses, bet3 (blue@4) wins
_BLUE_WIN = np.array([1, 0, 1, 0, 1])


class TestSettleBets:
    def test_won_and_pnl_hand_computed(self):
        settled = settle_bets(_hand_bets(), _BLUE_WIN)
        assert list(settled["won"]) == [True, False, True]
        # pnl: 0.1*(2-1)=0.1; -0.2; 0.5*(1.5-1)=0.25
        np.testing.assert_allclose(settled["pnl"], [0.1, -0.2, 0.25])
        np.testing.assert_allclose(settled["ret"], [1.0, -1.0, 0.5])

    def test_uses_bets_index_into_games(self):
        # flipping outcomes of the un-bet games 1 and 3 changes nothing
        alt = _BLUE_WIN.copy()
        alt[[1, 3]] = 1 - alt[[1, 3]]
        settled = settle_bets(_hand_bets(), alt)
        assert list(settled["won"]) == [True, False, True]

    def test_accepts_series_outcomes(self):
        ser = pd.Series(_BLUE_WIN, index=range(5))
        settled = settle_bets(_hand_bets(), ser)
        np.testing.assert_allclose(settled["pnl"], [0.1, -0.2, 0.25])

    def test_preserves_input(self):
        bets = _hand_bets()
        settle_bets(bets, _BLUE_WIN)
        assert "won" not in bets.columns  # input not mutated


class TestSimulateBankroll:
    def test_compound_exact_numbers(self):
        settled = settle_bets(_hand_bets(), _BLUE_WIN)
        curve = simulate_bankroll(settled, start=1.0, compound=True)
        # bet1: amount 0.1 -> 1 + 0.1*1.0    = 1.10
        # bet2: amount 0.2*1.1=0.22 -> 1.1-0.22 = 0.88
        # bet3: amount 0.5*0.88=0.44 -> 0.88 + 0.44*0.5 = 1.10
        np.testing.assert_allclose(curve.to_numpy(), [1.10, 0.88, 1.10])
        assert list(curve.index) == [0, 2, 4]

    def test_flat_exact_numbers(self):
        settled = settle_bets(_hand_bets(), _BLUE_WIN)
        curve = simulate_bankroll(settled, start=1.0, compound=False)
        # fixed amounts 0.1, 0.2, 0.5 of START bankroll:
        # 1 + 0.1 = 1.10; 1.1 - 0.2 = 0.90; 0.9 + 0.5*0.5 = 1.15
        np.testing.assert_allclose(curve.to_numpy(), [1.10, 0.90, 1.15])

    def test_start_scaling(self):
        settled = settle_bets(_hand_bets(), _BLUE_WIN)
        curve = simulate_bankroll(settled, start=100.0, compound=True)
        np.testing.assert_allclose(curve.to_numpy(), [110.0, 88.0, 110.0])

    def test_flat_matches_cumulative_pnl(self):
        settled = settle_bets(_hand_bets(), _BLUE_WIN)
        curve = simulate_bankroll(settled, start=1.0, compound=False)
        np.testing.assert_allclose(curve.to_numpy(), 1.0 + settled["pnl"].cumsum())


# ----------------------------------------------------------------- bootstrap


class TestBootstrapRoiCi:
    def _settled(self) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        n = 100
        p = rng.uniform(0.35, 0.65, size=n)
        bets = pd.DataFrame(
            {
                "side": ["blue"] * n,
                "model_p": p,
                "odds": 1.0 / np.clip(p - 0.03, 0.05, 0.95),
                "edge": np.full(n, 0.05),
                "stake_frac": np.full(n, 0.02),
            }
        )
        blue_win = (rng.uniform(size=n) < p).astype(int)
        return settle_bets(bets, blue_win)

    def test_ci_contains_point(self):
        lo, hi, point = bootstrap_roi_ci(self._settled(), n=2000, seed=0)
        assert lo <= point <= hi
        assert point == pytest.approx(
            self._settled()["pnl"].sum() / self._settled()["stake_frac"].sum()
        )

    def test_deterministic_given_seed(self):
        a = bootstrap_roi_ci(self._settled(), n=2000, seed=5)
        b = bootstrap_roi_ci(self._settled(), n=2000, seed=5)
        assert a == b

    def test_level_widens_interval(self):
        settled = self._settled()
        lo95, hi95, _ = bootstrap_roi_ci(settled, n=2000, seed=1, level=0.95)
        lo50, hi50, _ = bootstrap_roi_ci(settled, n=2000, seed=1, level=0.50)
        assert lo95 <= lo50 and hi50 <= hi95


# --------------------------------------------------------------- integration


def test_end_to_end_pipeline_smoke():
    """Synthetic odds -> select -> settle -> bankroll runs and is coherent."""
    rng = np.random.default_rng(9)
    n = 500
    true_p = rng.uniform(0.2, 0.8, size=n)
    model_p = pd.Series(true_p, index=pd.RangeIndex(1000, 1000 + n))
    odds = make_synthetic_odds(model_p, seed=11)
    bets = select_bets(model_p, odds["odds_blue"], odds["odds_red"])
    assert len(bets) > 0
    assert set(bets.index).issubset(set(model_p.index))
    assert bets["stake_frac"].between(0, 0.02, inclusive="right").all()

    blue_win = pd.Series((rng.uniform(size=n) < true_p).astype(int), index=model_p.index)
    settled = settle_bets(bets, blue_win)
    curve = simulate_bankroll(settled)
    assert len(curve) == len(bets)
    assert (curve > 0).all()  # 2% caps can't bust the bankroll
    lo, hi, point = bootstrap_roi_ci(settled, n=1000, seed=0)
    assert lo <= point <= hi
