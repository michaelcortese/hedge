"""Leakage proof suite for the feature builder (DESIGN.md section 3).

Four attacks, each of which would catch a different class of leak:

* FUTURE-MUTATION: mutating games after a cutoff must not change any feature
  row at or before the cutoff (features are pure functions of the strict past).
* FAKE-FUTURE: appending a fabricated monster game in the future must leave
  every original row byte-identical.
* SENTINEL: on synthetic data with *zero* real signal (equal, static teams,
  no blue bonus), no feature may correlate with the current game's outcome —
  any correlation would mean current-game information leaked into features.
* WITHIN-DAY: sibling games of a same-day Bo3 must see identical state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lolpred.data.synthetic import generate_synthetic_games
from lolpred.features.build import (
    FEATURE_COLUMNS,
    META_COLUMNS,
    build_matchup_features,
)

START = pd.Timestamp("2020-01-01")
CUTOFF = START + pd.Timedelta(days=100)


@pytest.fixture(scope="module")
def base_games():
    return generate_synthetic_games(n_teams=8, n_days=200, seed=1)


@pytest.fixture(scope="module")
def base_feats(base_games):
    return build_matchup_features(base_games)


def test_future_mutation_does_not_change_past_features(base_games, base_feats):
    mutated = base_games.copy()
    after = mutated["date"] > CUTOFF
    assert after.any() and (~after).any()

    # flip outcomes and scramble stats of every post-cutoff game
    mutated.loc[after, "blue_win"] = 1 - mutated.loc[after, "blue_win"]
    mutated.loc[after, "blue_golddiffat15"] = (
        -3.0 * mutated.loc[after, "blue_golddiffat15"] + 12345.0
    )
    mutated.loc[after, "red_golddiffat15"] = -mutated.loc[after, "blue_golddiffat15"]
    for col in ("kills", "deaths", "firstblood", "firstdragon",
                "firstbaron", "firsttower"):
        b, r = f"blue_{col}", f"red_{col}"
        mutated.loc[after, [b, r]] = mutated.loc[after, [r, b]].to_numpy()
    mutated.loc[after, "blue_players"] = "IMP_P1|IMP_P2|IMP_P3|IMP_P4|IMP_P5"

    feats_mut = build_matchup_features(mutated)

    keep = base_feats["date"] <= CUTOFF
    pd.testing.assert_frame_equal(
        base_feats[keep].reset_index(drop=True),
        feats_mut[keep.to_numpy()].reset_index(drop=True),
        check_exact=True,
    )


def test_fake_future_game_does_not_change_any_row(base_games, base_feats):
    team_a = base_games["blue_team"].iloc[0]
    team_b = base_games["red_team"].iloc[0]
    future_day = base_games["date"].max() + pd.Timedelta(days=30)
    fake = {c: np.nan for c in base_games.columns}
    fake.update(
        gameid="ZZZ_FAKE",
        date=future_day,
        league="SYNTH_A",
        year=int(future_day.year),
        split="Summer",
        playoffs=0,
        patch="99.9",
        game_in_series=1,
        series_id=f"{future_day.date()}|SYNTH_A|{team_a}|{team_b}",
        datacompleteness="complete",
        blue_team=team_a,
        red_team=team_b,
        blue_win=1,
        gamelength=1000.0,
        blue_kills=100.0,
        blue_deaths=0.0,
        red_kills=0.0,
        red_deaths=100.0,
        blue_golddiffat15=50_000.0,
        red_golddiffat15=-50_000.0,
        blue_firstblood=1.0,
        red_firstblood=0.0,
        blue_players="",
        red_players="",
    )
    extended = pd.concat(
        [base_games, pd.DataFrame([fake], columns=base_games.columns)],
        ignore_index=True,
    )
    feats_ext = build_matchup_features(extended)
    assert len(feats_ext) == len(base_feats) + 1
    pd.testing.assert_frame_equal(
        base_feats,
        feats_ext.iloc[:-1].reset_index(drop=True),
        check_exact=True,
    )


def test_sentinel_no_feature_correlates_with_outcome_when_no_signal_exists():
    games = generate_synthetic_games(
        n_teams=8, n_days=250, seed=2,
        strength_sd=0.0, drift_sd=0.0, blue_bonus=0.0,
    )
    assert len(games) > 2000
    feats = build_matchup_features(games)
    y = feats["blue_win"].to_numpy(dtype=float)

    checked = 0
    for col in FEATURE_COLUMNS:
        x = feats[col].to_numpy(dtype=float)
        valid = ~np.isnan(x)
        if valid.sum() < 100:
            continue
        xv, yv = x[valid], y[valid]
        if np.std(xv) == 0 or np.std(yv) == 0:
            continue  # constant feature: cannot leak
        corr = np.corrcoef(xv, yv)[0, 1]
        checked += 1
        assert abs(corr) < 0.35, (
            f"{col} correlates with the current game's outcome "
            f"(|r|={abs(corr):.3f}) on zero-signal data -> leakage"
        )
    assert checked >= 15  # the suite actually exercised most features


def test_within_day_bo3_siblings_see_identical_state(base_games, base_feats):
    # find a Bo3 whose 3 games all share the same orientation (so features
    # must be *equal*, not negated), after some burn-in history
    meta = base_feats[base_feats["date"] > START + pd.Timedelta(days=30)]
    grouped = meta.groupby("series_id")
    target = None
    for sid, grp in grouped:
        if len(grp) == 3 and grp["blue_team"].nunique() == 1:
            target = grp
            break
    assert target is not None, "synthetic data should contain same-orientation Bo3s"

    cols = [c for c in FEATURE_COLUMNS if c != "f_game_in_series"]
    first = target.iloc[0]
    for i in (1, 2):
        row = target.iloc[i]
        for col in cols:
            assert np.array_equal(
                np.asarray(first[col]), np.asarray(row[col]), equal_nan=True
            ), (
                f"{col}: game {i + 1} of a Bo3 differs from game 1 "
                f"({first[col]} vs {row[col]}) -> within-day leakage"
            )
