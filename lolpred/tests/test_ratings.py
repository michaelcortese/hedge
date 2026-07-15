"""Tests for lolpred.features.ratings (EloStream, BradleyTerry)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from lolpred.features.ratings import BT_UNSEEN_SE, BradleyTerry, EloStream


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

GAME_COLS = [
    "date",
    "blue_team",
    "red_team",
    "blue_win",
    "game_in_series",
    "blue_golddiffat15",
    "blue_kills",
    "red_kills",
]


def make_games(records: list[dict]) -> list:
    """Build itertuples rows (the documented input shape) from dicts."""
    filled = []
    for rec in records:
        row = {
            "date": pd.Timestamp("2024-01-01"),
            "blue_win": 1,
            "game_in_series": 1,
            "blue_golddiffat15": np.nan,
            "blue_kills": np.nan,
            "red_kills": np.nan,
        }
        row.update(rec)
        filled.append(row)
    df = pd.DataFrame(filled)
    return list(df.itertuples(index=False))


def pair_block(
    a: str, b: str, a_wins_as_blue: list[int], a_wins_as_red: list[int], date
) -> list[dict]:
    """Games between a and b, half with each on blue side."""
    recs = []
    for win in a_wins_as_blue:
        recs.append(
            {"date": date, "blue_team": a, "red_team": b, "blue_win": win}
        )
    for win in a_wins_as_red:
        recs.append(
            {"date": date, "blue_team": b, "red_team": a, "blue_win": 1 - win}
        )
    return recs


def round_robin_records(date) -> list[dict]:
    """4 teams, 10 games/pair: A wins 8/10 vs everyone, others split 5/5."""
    recs = []
    for opp in ["B", "C", "D"]:
        recs += pair_block("A", opp, [1, 1, 1, 1, 0], [1, 1, 1, 1, 0], date)
    for x, z in [("B", "C"), ("B", "D"), ("C", "D")]:
        recs += pair_block(x, z, [1, 0, 1, 0, 1], [0, 1, 0, 1, 0], date)
    return recs


# ---------------------------------------------------------------------------
# EloStream
# ---------------------------------------------------------------------------


class TestEloStream:
    def test_unseen_team_defaults(self):
        elo = EloStream()
        out = elo.pregame("A", "B")
        assert out["elo_blue"] == out["elo_red"] == 1500.0
        assert out["elo_diff"] == 0.0
        assert out["elo_games_blue"] == out["elo_games_red"] == 0.0

    def test_winner_gains_loser_loses_symmetric(self):
        elo = EloStream(mov=False)
        game = make_games(
            [{"blue_team": "A", "red_team": "B", "blue_win": 1}]
        )[0]
        elo.update(game)
        gain = elo.rating("A") - 1500.0
        loss = 1500.0 - elo.rating("B")
        assert gain > 0
        assert gain == pytest.approx(loss)
        assert elo.games_played("A") == elo.games_played("B") == 1

    def test_blue_offset_shifts_expected_score(self):
        with_offset = EloStream(side_offset_init=25.0)
        without = EloStream(side_offset_init=0.0)
        assert without.expected_blue("A", "B") == pytest.approx(0.5)
        assert with_offset.expected_blue("A", "B") > 0.5
        expected = 1.0 / (1.0 + 10.0 ** (-25.0 / 400.0))
        assert with_offset.expected_blue("A", "B") == pytest.approx(expected)

    def test_mov_multiplier_bounds_and_fallback(self):
        elo = EloStream()
        # enormous gold diff -> capped at 2.0
        assert elo.mov_multiplier(1e9, None, 0.0) == pytest.approx(2.0)
        # tiny gold diff -> floored at 0.5
        assert elo.mov_multiplier(1.0, None, 0.0) == pytest.approx(0.5)
        # both missing -> exactly 1.0
        assert elo.mov_multiplier(np.nan, None, 0.0) == 1.0
        assert elo.mov_multiplier(None, np.nan, 0.0) == 1.0
        # gold NaN -> falls back to kill diff, ln(1 + 20/5) in bounds
        expect = math.log1p(20 / 5)
        assert elo.mov_multiplier(np.nan, 20.0, 0.0) == pytest.approx(expect)
        # autocorrelation damp shrinks updates for a heavy elo favorite
        assert elo.mov_multiplier(6000.0, None, 400.0) < elo.mov_multiplier(
            6000.0, None, 0.0
        )
        # mov disabled -> always 1
        assert EloStream(mov=False).mov_multiplier(6000.0, None, 0.0) == 1.0

    def test_mov_scales_update(self):
        big = EloStream()
        small = EloStream()
        big.update(
            make_games(
                [
                    {
                        "blue_team": "A",
                        "red_team": "B",
                        "blue_win": 1,
                        "blue_golddiffat15": 6000.0,
                    }
                ]
            )[0]
        )
        small.update(
            make_games(
                [
                    {
                        "blue_team": "A",
                        "red_team": "B",
                        "blue_win": 1,
                        "blue_golddiffat15": 100.0,
                    }
                ]
            )[0]
        )
        assert big.rating("A") > small.rating("A")

    def test_new_period_regresses_toward_mean(self):
        elo = EloStream(mov=False, split_regress=0.25)
        elo.new_period("2023|spring")
        for _ in range(10):
            elo.update(
                make_games(
                    [{"blue_team": "A", "red_team": "B", "blue_win": 1}]
                )[0]
            )
        r_a, r_b = elo.rating("A"), elo.rating("B")
        assert r_a > 1500.0 > r_b
        elo.new_period("2023|summer")
        assert elo.rating("A") == pytest.approx(r_a + 0.25 * (1500.0 - r_a))
        assert elo.rating("B") == pytest.approx(r_b + 0.25 * (1500.0 - r_b))
        # same key again is a no-op
        r_a2 = elo.rating("A")
        elo.new_period("2023|summer")
        assert elo.rating("A") == r_a2

    def test_pregame_is_pure(self):
        elo = EloStream()
        elo.update(
            make_games([{"blue_team": "A", "red_team": "B", "blue_win": 1}])[0]
        )
        first = elo.pregame("A", "B")
        second = elo.pregame("A", "B")
        assert first == second
        assert elo.rating("A") == first["elo_blue"]
        assert elo.games_played("A") == 1

    def test_k_differs_bo1_vs_series(self):
        def delta_for(game_kwargs):
            elo = EloStream(k_bo1=32.0, k_series=24.0, mov=False)
            base = {"blue_team": "A", "red_team": "B", "blue_win": 1}
            base.update(game_kwargs)
            elo.update(make_games([base])[0])
            return elo.rating("A") - 1500.0

        e0 = 1.0 / (1.0 + 10.0 ** (-25.0 / 400.0))  # blue offset, equal elo
        # inferred from game_in_series when best_of absent
        assert delta_for({"game_in_series": 1}) == pytest.approx(
            32.0 * (1 - e0)
        )
        assert delta_for({"game_in_series": 2}) == pytest.approx(
            24.0 * (1 - e0)
        )
        # explicit best_of overrides the inference
        assert delta_for({"game_in_series": 1, "best_of": 5}) == pytest.approx(
            24.0 * (1 - e0)
        )
        assert delta_for({"game_in_series": 1, "best_of": 1}) == pytest.approx(
            32.0 * (1 - e0)
        )

    def test_update_accepts_dict_rows(self):
        elo = EloStream(mov=False)
        elo.update({"blue_team": "A", "red_team": "B", "blue_win": 0})
        assert elo.rating("B") > 1500.0 > elo.rating("A")


# ---------------------------------------------------------------------------
# BradleyTerry
# ---------------------------------------------------------------------------


def feed(bt: BradleyTerry, records: list[dict]) -> None:
    for row in make_games(records):
        bt.observe(row)


class TestBradleyTerry:
    def test_round_robin_ranks_dominant_team(self):
        bt = BradleyTerry(half_life_days=1e6, l2=2.0)
        feed(bt, round_robin_records(pd.Timestamp("2024-01-01")))
        query = pd.Timestamp("2024-02-01")
        for opp in ["B", "C", "D"]:
            out = bt.pregame("A", opp, query)
            assert out["bt_theta_diff"] > 0.0
        out = bt.pregame("A", "B", query)
        assert out["bt_prob_blue"] > 0.6
        assert out["bt_se_diff"] > 0.0
        # side intercept identified (data is side-balanced -> near zero)
        assert abs(out["bt_beta_side"]) < 0.5

    def test_time_decay_flips_theta_sign(self):
        old, recent = pd.Timestamp("2023-01-01"), pd.Timestamp("2024-01-01")
        records = []
        # A beats B 8 times long ago; B beats A 5 times recently
        records += pair_block("A", "B", [1, 1, 1, 1], [1, 1, 1, 1], old)
        records += pair_block("B", "A", [1, 1, 1], [1, 1], recent)
        # filler so history clears the 50-game minimum, half old half recent
        records += pair_block(
            "C", "D", [1, 0] * 6, [0, 1] * 6, old
        )
        records += pair_block(
            "C", "D", [1, 0] * 6, [0, 1] * 6, recent
        )
        query = pd.Timestamp("2024-01-02")

        fast_decay = BradleyTerry(half_life_days=30.0)
        feed(fast_decay, records)
        slow_decay = BradleyTerry(half_life_days=1e6)
        feed(slow_decay, records)

        # recent losses dominate under fast decay ...
        assert fast_decay.pregame("A", "B", query)["bt_theta_diff"] < 0.0
        # ... but A's larger win count dominates without decay
        assert slow_decay.pregame("A", "B", query)["bt_theta_diff"] > 0.0

    def test_se_shrinks_with_more_games(self):
        date = pd.Timestamp("2024-01-01")
        query = pd.Timestamp("2024-02-01")
        small = BradleyTerry(half_life_days=1e6)
        feed(small, round_robin_records(date))
        big = BradleyTerry(half_life_days=1e6)
        for _ in range(4):
            feed(big, round_robin_records(date))
        se_small = small.pregame("A", "B", query)["bt_se_diff"]
        se_big = big.pregame("A", "B", query)["bt_se_diff"]
        assert 0.0 < se_big < se_small

    def test_below_min_history_returns_defaults_without_fitting(self):
        bt = BradleyTerry()
        feed(
            bt,
            pair_block(
                "A", "B", [1] * 10, [1] * 10, pd.Timestamp("2024-01-01")
            ),
        )  # 20 games < 50
        out = bt.pregame("A", "B", pd.Timestamp("2024-02-01"))
        assert out == {
            "bt_theta_diff": 0.0,
            "bt_se_diff": BT_UNSEEN_SE,
            "bt_prob_blue": 0.5,
            "bt_beta_side": 0.0,
        }
        assert bt.n_fits == 0

    def test_unseen_team_after_fit(self):
        bt = BradleyTerry(half_life_days=1e6)
        feed(bt, round_robin_records(pd.Timestamp("2024-01-01")))
        out = bt.pregame("A", "ZZZ_NEW", pd.Timestamp("2024-02-01"))
        assert bt.n_fits == 1
        assert out["bt_theta_diff"] == 0.0
        assert out["bt_se_diff"] == BT_UNSEEN_SE
        assert 0.0 < out["bt_prob_blue"] < 1.0

    def test_refit_every_days_honored(self):
        bt = BradleyTerry(half_life_days=1e6, refit_every_days=7)
        feed(bt, round_robin_records(pd.Timestamp("2024-01-01")))
        d0 = pd.Timestamp("2024-02-01")
        bt.pregame("A", "B", d0)
        assert bt.n_fits == 1
        bt.pregame("A", "B", d0 + pd.Timedelta(days=3))
        bt.pregame("C", "D", d0 + pd.Timedelta(days=6))
        assert bt.n_fits == 1  # inside the refit window: no refit
        bt.pregame("A", "B", d0 + pd.Timedelta(days=7))
        assert bt.n_fits == 2  # boundary inclusive: date >= last_fit + 7

    def test_deterministic(self):
        records = round_robin_records(pd.Timestamp("2024-01-01"))
        query = pd.Timestamp("2024-02-01")
        a = BradleyTerry()
        feed(a, records)
        b = BradleyTerry()
        feed(b, records)
        assert a.pregame("A", "C", query) == b.pregame("A", "C", query)

    def test_pregame_speed_bound(self):
        """~10k games must fit in well under a second."""
        import time

        rng_dates = pd.date_range("2022-01-01", periods=100, freq="D")
        records = []
        teams = [f"T{i}" for i in range(20)]
        k = 0
        for d in rng_dates:
            for i in range(100):
                b, r = teams[k % 20], teams[(k + 7) % 20]
                if b == r:
                    r = teams[(k + 8) % 20]
                records.append(
                    {
                        "date": d,
                        "blue_team": b,
                        "red_team": r,
                        "blue_win": k % 2,
                    }
                )
                k += 1
        bt = BradleyTerry()
        feed(bt, records)  # 10_000 games
        t0 = time.perf_counter()
        bt.pregame("T0", "T1", pd.Timestamp("2022-05-01"))
        elapsed = time.perf_counter() - t0
        assert bt.n_fits == 1
        assert elapsed < 1.0
