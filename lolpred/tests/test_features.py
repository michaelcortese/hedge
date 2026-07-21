"""Tests for the chronological matchup feature builder."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from lolpred.data.synthetic import CANONICAL_COLUMNS, generate_synthetic_games
from lolpred.features.build import (
    FEATURE_COLUMNS,
    META_COLUMNS,
    FeatureConfig,
    build_matchup_features,
)

START = pd.Timestamp("2020-01-01")

# Symmetric context columns (invariant under orientation swap).
SYMMETRIC_COLS = [
    "f_bt_se",
    "f_bt_beta_side",
    "f_sidewr_pair",
    "f_league_bluewr",
    "f_playoffs",
    "f_game_in_series",
    "f_patch_age_days",
]
# Exchange under swap rather than staying fixed.
SWAP_PAIR = ("f_hist_games_blue", "f_hist_games_red")
DIFF_COLS = [c for c in FEATURE_COLUMNS if c.endswith("_diff")]


@pytest.fixture(scope="module")
def synth():
    return generate_synthetic_games(n_teams=8, n_days=200, seed=1)


@pytest.fixture(scope="module")
def synth_feats(synth):
    return build_matchup_features(synth)


# ---------------------------------------------------------------------------
# hand-made canonical frames
# ---------------------------------------------------------------------------

def mk_game(gameid, date, blue, red, blue_win, *, league="L", gd15=np.nan,
            blue_kills=np.nan, blue_deaths=np.nan, red_kills=np.nan,
            red_deaths=np.nan, blue_firstblood=np.nan, game_in_series=1,
            patch="13.01", playoffs=0):
    """One canonical-table row with sane defaults for unused stats."""
    date = pd.Timestamp(date)
    pair = sorted((blue, red))
    row = {c: np.nan for c in CANONICAL_COLUMNS}
    row.update(
        gameid=gameid,
        date=date,
        league=league,
        year=int(date.year),
        split="Spring" if date.month <= 6 else "Summer",
        playoffs=playoffs,
        patch=patch,
        game_in_series=game_in_series,
        series_id=f"{date.date()}|{league}|{pair[0]}|{pair[1]}",
        datacompleteness="complete",
        blue_team=blue,
        red_team=red,
        blue_win=int(blue_win),
        gamelength=1900.0,
        blue_golddiffat15=gd15,
        red_golddiffat15=-gd15 if gd15 == gd15 else np.nan,
        blue_kills=blue_kills,
        blue_deaths=blue_deaths,
        red_kills=red_kills,
        red_deaths=red_deaths,
        blue_firstblood=blue_firstblood,
        red_firstblood=(1 - blue_firstblood) if blue_firstblood == blue_firstblood else np.nan,
        blue_players="|".join(f"{blue}_P{i}" for i in range(1, 6)),
        red_players="|".join(f"{red}_P{i}" for i in range(1, 6)),
    )
    return row


def hand_frame():
    rows = [
        mk_game("g1", "2021-01-01", "A", "B", 1, gd15=1000.0, blue_kills=10,
                blue_deaths=5, red_kills=5, red_deaths=10, blue_firstblood=1.0),
        mk_game("g2", "2021-01-02", "A", "B", 0, gd15=-500.0, blue_kills=3,
                blue_deaths=8, red_kills=8, red_deaths=3, blue_firstblood=1.0),
        mk_game("g3", "2021-01-03", "B", "C", 1, gd15=200.0, blue_kills=7,
                blue_deaths=4, red_kills=4, red_deaths=7, blue_firstblood=1.0),
        mk_game("g4", "2021-01-04", "A", "C", 1, gd15=500.0, blue_kills=12,
                blue_deaths=2, red_kills=2, red_deaths=12, blue_firstblood=1.0),
        mk_game("g5", "2021-01-10", "A", "B", 1, gd15=100.0, blue_kills=9,
                blue_deaths=9, red_kills=9, red_deaths=9, blue_firstblood=1.0),
    ]
    return pd.DataFrame(rows, columns=CANONICAL_COLUMNS)


# ---------------------------------------------------------------------------
# shape / naming / meta preservation
# ---------------------------------------------------------------------------

def test_output_shape_and_columns(synth, synth_feats):
    feats = synth_feats
    assert len(feats) == len(synth)
    assert list(feats.columns) == META_COLUMNS + FEATURE_COLUMNS
    non_meta = [c for c in feats.columns if c not in META_COLUMNS]
    assert all(c.startswith("f_") for c in non_meta)
    # comparative features end _diff; declared symmetric ones must not
    for c in SYMMETRIC_COLS + list(SWAP_PAIR):
        assert not c.endswith("_diff")


def test_meta_preserved(synth, synth_feats):
    ordered = synth.sort_values(["date", "gameid"], kind="mergesort").reset_index(drop=True)
    for col in META_COLUMNS:
        pd.testing.assert_series_equal(
            synth_feats[col], ordered[col], check_names=True
        )


# ---------------------------------------------------------------------------
# hand-computed spot checks
# ---------------------------------------------------------------------------

def test_hand_computed_values():
    feats = build_matchup_features(hand_frame())
    last = feats.iloc[-1]  # g5: A (blue) vs B (red) on 2021-01-10
    k = 8.0

    # A results [1,0,1], B results [0,1,1]: identical shrunk means -> 0 diff.
    assert last["f_win10_diff"] == 0.0
    assert last["f_win30_diff"] == 0.0

    # ewm, half-life 15 games, start 0.5, shrunk toward 0.5 with n/(n+8).
    w = 1.0 - 0.5 ** (1.0 / 15.0)
    def ewm(obs):
        v = 0.5
        for o in obs:
            v = w * o + (1 - w) * v
        return 0.5 + (v - 0.5) * len(obs) / (len(obs) + k)
    assert last["f_winewm_diff"] == pytest.approx(ewm([1, 0, 1]) - ewm([0, 1, 1]))

    # gd15 per-team perspective, shrunk toward 0 with n/(n+8):
    # A [1000,-500,500] -> (1000/3)*(3/11); B [-1000,500,200] -> (-100)*(3/11)
    assert last["f_gd15_mean30_diff"] == pytest.approx(1300.0 / 11.0)
    assert last["f_gd15_mean10_diff"] == pytest.approx(1300.0 / 11.0)

    # rest days: A last played 01-04 (6d), B 01-03 (7d)
    assert last["f_rest_days_diff"] == pytest.approx(-1.0)

    # first blood: A [1,1,1] -> 1; B [0,0,1] -> 1/3
    assert last["f_fb_rate30_diff"] == pytest.approx(2.0 / 3.0)

    # K/D log ratio over window: A log(26/16), B log(21/18)
    assert last["f_kd_log30_diff"] == pytest.approx(
        math.log(26.0 / 16.0) - math.log(21.0 / 18.0)
    )

    # side preference s(t) = shrunkWR_as_blue - shrunkWR_as_red
    # A: as-blue [1,0,1] -> 0.5 + 1/22, as-red none -> 0.5 => 1/22
    # B: as-blue [1] -> 0.5 + 1/18, as-red [0,1] -> 0.5    => 1/18
    assert last["f_sidewr_diff"] == pytest.approx(1.0 / 22.0 - 1.0 / 18.0)
    assert last["f_sidewr_pair"] == pytest.approx((1.0 / 22.0 + 1.0 / 18.0) / 2.0)

    # counters / context
    assert last["f_hist_games_blue"] == 3.0
    assert last["f_hist_games_red"] == 3.0
    assert last["f_career_games_diff"] == 0.0
    assert last["f_games_on_patch_diff"] == 0.0
    assert last["f_split_games_diff"] == 0.0
    assert last["f_roster_cont_diff"] == 0.0  # stable rosters, 1.0 - 1.0
    assert last["f_league_bluewr"] == pytest.approx(0.75)  # [1,0,1,1]
    assert last["f_patch_age_days"] == pytest.approx(9.0)
    assert last["f_playoffs"] == 0.0
    assert last["f_game_in_series"] == 1.0


def test_cold_start_first_game_nan():
    feats = build_matchup_features(hand_frame())
    first = feats.iloc[0]
    for col in [
        "f_win10_diff", "f_win30_diff", "f_winewm_diff",
        "f_gd15_mean10_diff", "f_gd15_mean30_diff",
        "f_fb_rate30_diff", "f_fdragon_rate30_diff", "f_fbaron_rate30_diff",
        "f_ftower_rate30_diff", "f_sidewr_diff", "f_kd_log30_diff",
        "f_rest_days_diff", "f_league_bluewr",
    ]:
        assert np.isnan(first[col]), col
    assert first["f_hist_games_blue"] == 0.0
    assert first["f_hist_games_red"] == 0.0
    # elo starts equal, counters at zero
    assert first["f_elo_diff"] == 0.0
    assert first["f_career_games_diff"] == 0.0


def test_fewer_than_three_games_rolling_nan():
    # after 2 games, gd15/objective/kd rolling means still NaN; shrunk win
    # rates already emit.
    feats = build_matchup_features(hand_frame())
    third = feats.iloc[2]  # g3: B vs C -- B has 2 prior games, C has 0
    assert np.isnan(third["f_gd15_mean30_diff"])
    second = feats.iloc[1]  # g2: A vs B, each with exactly 1 prior game
    assert np.isnan(second["f_gd15_mean30_diff"])
    assert np.isnan(second["f_kd_log30_diff"])
    assert not np.isnan(second["f_win10_diff"])
    assert not np.isnan(second["f_winewm_diff"])


# ---------------------------------------------------------------------------
# antisymmetry
# ---------------------------------------------------------------------------

def _mirror(row: dict) -> dict:
    out = dict(row)
    out["blue_team"], out["red_team"] = row["red_team"], row["blue_team"]
    out["blue_win"] = 1 - row["blue_win"]
    for col in CANONICAL_COLUMNS:
        if col.startswith("blue_") and col not in ("blue_team", "blue_win"):
            other = "red_" + col[len("blue_"):]
            out[col], out[other] = row[other], row[col]
    return out


def test_antisymmetry_under_orientation_swap():
    hist = generate_synthetic_games(
        n_teams=4, n_days=80, seed=3, leagues=("SYNTH_A",)
    )
    last_date = hist["date"].max() + pd.Timedelta(days=5)
    game = mk_game("zz_final", last_date, "SYNA_T01", "SYNA_T02", 1,
                   gd15=800.0, blue_kills=15, blue_deaths=6, red_kills=6,
                   red_deaths=15, blue_firstblood=1.0)
    mirrored = _mirror(game)

    hist_rows = hist.to_dict("records")
    f1 = build_matchup_features(
        pd.DataFrame(hist_rows + [game], columns=CANONICAL_COLUMNS)
    ).iloc[-1]
    f2 = build_matchup_features(
        pd.DataFrame(hist_rows + [mirrored], columns=CANONICAL_COLUMNS)
    ).iloc[-1]

    for col in DIFF_COLS:
        a, b = f1[col], f2[col]
        assert np.array_equal(np.asarray(a), np.asarray(-b), equal_nan=True), (
            f"{col}: {a} vs {b} (expected exact negation)"
        )
    for col in SYMMETRIC_COLS:
        a, b = f1[col], f2[col]
        assert np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True), (
            f"{col}: {a} vs {b} (expected identical)"
        )
    # hist counters exchange under swap
    assert f1["f_hist_games_blue"] == f2["f_hist_games_red"]
    assert f1["f_hist_games_red"] == f2["f_hist_games_blue"]


# ---------------------------------------------------------------------------
# signal: elo recovers latent strength
# ---------------------------------------------------------------------------

def test_elo_diff_tracks_latent_strength():
    games, truth = generate_synthetic_games(
        n_teams=8, n_days=200, seed=1, return_truth=True
    )
    feats = build_matchup_features(games)
    theta = {(d, t): v for d, t, v in truth.itertuples(index=False)}

    burn = feats["date"] > START + pd.Timedelta(days=60)
    sub = feats[burn]
    truth_diff = np.array([
        theta[(d, b)] - theta[(d, r)]
        for d, b, r in zip(sub["date"], sub["blue_team"], sub["red_team"])
    ])
    corr = np.corrcoef(sub["f_elo_diff"].to_numpy(), truth_diff)[0, 1]
    assert corr > 0.5, f"corr(f_elo_diff, latent theta diff) = {corr:.3f}"


# ---------------------------------------------------------------------------
# within-day invisibility
# ---------------------------------------------------------------------------

def test_within_day_games_mutually_invisible():
    hist = generate_synthetic_games(
        n_teams=4, n_days=60, seed=4, leagues=("SYNTH_A",)
    )
    day = hist["date"].max() + pd.Timedelta(days=3)
    # two Bo1s, same date, same orientation, wildly different outcomes/stats
    g1 = mk_game("zz_a", day, "SYNA_T01", "SYNA_T02", 1, gd15=5000.0,
                 blue_kills=30, blue_deaths=2, red_kills=2, red_deaths=30,
                 blue_firstblood=1.0)
    g2 = mk_game("zz_b", day, "SYNA_T01", "SYNA_T02", 0, gd15=-5000.0,
                 blue_kills=1, blue_deaths=25, red_kills=25, red_deaths=1,
                 blue_firstblood=0.0)
    frame = pd.DataFrame(
        hist.to_dict("records") + [g1, g2], columns=CANONICAL_COLUMNS
    )
    feats = build_matchup_features(frame)
    r1, r2 = feats.iloc[-2], feats.iloc[-1]
    for col in FEATURE_COLUMNS:
        assert np.array_equal(
            np.asarray(r1[col]), np.asarray(r2[col]), equal_nan=True
        ), f"{col}: {r1[col]} vs {r2[col]} (same-day games must be invisible)"


def test_empty_input():
    empty = pd.DataFrame(columns=CANONICAL_COLUMNS).astype({"date": "datetime64[ns]"})
    feats = build_matchup_features(empty)
    assert len(feats) == 0
    assert list(feats.columns) == META_COLUMNS + FEATURE_COLUMNS
