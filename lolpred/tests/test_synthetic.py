"""Tests for lolpred.data.synthetic — the synthetic canonical game table."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lolpred.data.synthetic import CANONICAL_COLUMNS, generate_synthetic_games

STAT_NAMES = (
    "kills",
    "deaths",
    "assists",
    "firstblood",
    "firstdragon",
    "firstbaron",
    "firsttower",
    "dragons",
    "barons",
    "towers",
    "goldat15",
    "xpat15",
    "csat15",
    "golddiffat15",
    "dpm",
)


@pytest.fixture(scope="module")
def games_and_truth() -> tuple[pd.DataFrame, pd.DataFrame]:
    return generate_synthetic_games(seed=0, return_truth=True)


@pytest.fixture(scope="module")
def games(games_and_truth) -> pd.DataFrame:
    return games_and_truth[0]


def test_canonical_columns_present_with_right_dtypes(games: pd.DataFrame) -> None:
    assert list(games.columns) == CANONICAL_COLUMNS

    str_cols = [
        "gameid", "league", "split", "patch", "series_id",
        "datacompleteness", "blue_team", "red_team",
        "blue_players", "red_players",
    ]
    for c in str_cols:
        assert games[c].map(lambda v: isinstance(v, str)).all(), c

    assert pd.api.types.is_datetime64_any_dtype(games["date"])
    for c in ["year", "playoffs", "game_in_series", "blue_win"]:
        assert pd.api.types.is_integer_dtype(games[c]), c
    assert pd.api.types.is_float_dtype(games["gamelength"])
    for side in ("blue", "red"):
        for stat in STAT_NAMES:
            assert pd.api.types.is_float_dtype(games[f"{side}_{stat}"]), f"{side}_{stat}"

    assert (games["playoffs"] == 0).all()
    assert games["split"].isin(["Spring", "Summer"]).all()
    assert (games["datacompleteness"] == "complete").all()
    assert games["gamelength"].between(1200, 3600).all()


def test_sorted_and_gameid_unique(games: pd.DataFrame) -> None:
    assert games["gameid"].is_unique
    key = games[["date", "gameid"]]
    assert key.equals(key.sort_values(["date", "gameid"]).reset_index(drop=True))
    assert games.index.equals(pd.RangeIndex(len(games)))


def test_blue_win_binary(games: pd.DataFrame) -> None:
    assert set(games["blue_win"].unique()) <= {0, 1}
    # Both outcomes actually occur.
    assert games["blue_win"].nunique() == 2


def test_series_consistency(games: pd.DataFrame) -> None:
    for series_id, grp in games.groupby("series_id"):
        n = len(grp)
        assert n in (2, 3), series_id
        # game_in_series consecutive 1..n.
        assert sorted(grp["game_in_series"].tolist()) == list(range(1, n + 1)), series_id
        # Same day, same team pair throughout.
        assert grp["date"].nunique() == 1, series_id
        pairs = grp.apply(lambda r: frozenset((r["blue_team"], r["red_team"])), axis=1)
        assert pairs.nunique() == 1, series_id
        # Series ends exactly when one team hits 2 wins: 2-0 or 2-1.
        winners = np.where(grp["blue_win"] == 1, grp["blue_team"], grp["red_team"])
        counts = pd.Series(winners).value_counts()
        assert counts.max() == 2, series_id
        assert counts.sum() == n, series_id
        # The clinching win is the last game of the series.
        ordered = grp.sort_values("game_in_series")
        last_winner = (
            ordered.iloc[-1]["blue_team"]
            if ordered.iloc[-1]["blue_win"] == 1
            else ordered.iloc[-1]["red_team"]
        )
        assert counts.idxmax() == last_winner, series_id


def test_golddiffat15_antisymmetry(games: pd.DataFrame) -> None:
    np.testing.assert_allclose(
        games["blue_golddiffat15"].to_numpy(),
        -games["red_golddiffat15"].to_numpy(),
    )


def test_determinism_same_seed() -> None:
    a, ta = generate_synthetic_games(n_days=120, seed=7, return_truth=True)
    b, tb = generate_synthetic_games(n_days=120, seed=7, return_truth=True)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(ta, tb)
    # Different seed actually changes the data.
    c = generate_synthetic_games(n_days=120, seed=8)
    assert not a["blue_win"].equals(c["blue_win"])


def test_signal_top_quartile_teams_win_more(games_and_truth) -> None:
    games, truth = games_and_truth
    final = truth[truth["date"] == truth["date"].max()]
    top = set(final.nlargest(len(final) // 4, "theta")["team"])

    wins = 0
    played = 0
    for t in top:
        as_blue = games[games["blue_team"] == t]
        as_red = games[games["red_team"] == t]
        wins += int(as_blue["blue_win"].sum()) + int((1 - as_red["blue_win"]).sum())
        played += len(as_blue) + len(as_red)
    assert played > 0
    assert wins / played > 0.55


def test_theta_diff_predicts_outcome_without_blue_bonus() -> None:
    games, truth = generate_synthetic_games(
        n_days=240, seed=3, blue_bonus=0.0, return_truth=True
    )
    merged = games.merge(
        truth.rename(columns={"team": "blue_team", "theta": "theta_blue"}),
        on=["date", "blue_team"],
    ).merge(
        truth.rename(columns={"team": "red_team", "theta": "theta_red"}),
        on=["date", "red_team"],
    )
    assert len(merged) == len(games)
    diff = merged["theta_blue"] - merged["theta_red"]
    corr = np.corrcoef(diff, merged["blue_win"])[0, 1]
    assert corr > 0.15

    # Stronger team wins the majority of clearly-lopsided games.
    lopsided = merged[diff.abs() > 0.5]
    hit = ((diff.loc[lopsided.index] > 0) == (lopsided["blue_win"] == 1)).mean()
    assert hit > 0.55


def test_roster_churn_happens(games: pd.DataFrame) -> None:
    # Over the default 720 days, some team's player set changes.
    rosters_per_team: dict[str, set[str]] = {}
    for side in ("blue", "red"):
        for team, players in zip(games[f"{side}_team"], games[f"{side}_players"]):
            rosters_per_team.setdefault(team, set()).add(players)
    assert any(len(v) > 1 for v in rosters_per_team.values())
    # Rosters are 5 "|"-joined, sorted player names.
    for v in rosters_per_team.values():
        for roster in v:
            names = roster.split("|")
            assert len(names) == 5
            assert names == sorted(names)


def test_cross_league_days_exist(games: pd.DataFrame) -> None:
    intl = games[games["league"] == "INTL"]
    assert len(intl) > 0
    # Cross-league: the two teams come from different regular leagues.
    prefix = lambda t: t.split("_")[0]  # noqa: E731
    assert (intl["blue_team"].map(prefix) != intl["red_team"].map(prefix)).all()
    # Regular days stay within league.
    reg = games[games["league"] != "INTL"]
    assert (reg["blue_team"].map(prefix) == reg["red_team"].map(prefix)).all()
