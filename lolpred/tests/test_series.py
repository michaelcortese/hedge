import math

import pytest

from lolpred.series import exact_score_probs, series_win_prob, wins_needed


def test_bo3_closed_form():
    for p in (0.3, 0.5, 0.62, 0.9):
        assert series_win_prob(p, 3) == pytest.approx(p * p * (3 - 2 * p))


def test_bo5_closed_form():
    for p in (0.3, 0.5, 0.62, 0.9):
        expect = p**3 * (10 - 15 * p + 6 * p * p)
        assert series_win_prob(p, 5) == pytest.approx(expect)


def test_bo1_is_identity():
    assert series_win_prob(0.37, 1) == pytest.approx(0.37)


def test_symmetry():
    # P_A(p) + P_B(1-p) = 1
    for bo in (1, 3, 5, 7):
        for p in (0.1, 0.44, 0.5, 0.83):
            assert series_win_prob(p, bo) + series_win_prob(1 - p, bo) == pytest.approx(1.0)


def test_mid_series_states():
    # up 2-0 in a Bo5: lose out only if opponent wins 3 straight
    assert series_win_prob(0.5, 5, 2, 0) == pytest.approx(1 - 0.5**3)
    # match point converts with prob p + (1-p)*S(next state)
    p = 0.6
    assert series_win_prob(p, 5, 2, 2) == pytest.approx(p)
    # boundary states
    assert series_win_prob(0.2, 5, 3, 0) == 1.0
    assert series_win_prob(0.9, 5, 0, 3) == 0.0


def test_exact_scores_sum_to_one_and_match_series_prob():
    p, bo = 0.58, 5
    scores = exact_score_probs(p, bo)
    assert sum(scores.values()) == pytest.approx(1.0)
    w = wins_needed(bo)
    p_series = sum(v for (a, b), v in scores.items() if a == w)
    assert p_series == pytest.approx(series_win_prob(p, bo))


def test_favorite_gains_in_longer_series():
    p = 0.6
    probs = [series_win_prob(p, bo) for bo in (1, 3, 5, 7)]
    assert probs == sorted(probs)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        wins_needed(4)
    with pytest.raises(ValueError):
        series_win_prob(1.2, 3)
    with pytest.raises(ValueError):
        series_win_prob(0.5, 3, 3, 0)
