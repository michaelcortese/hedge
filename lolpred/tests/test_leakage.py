"""Leakage proof suite for the feature builder (DESIGN.md section 3).

Attacks, each of which would catch a different class of leak:

* FUTURE-MUTATION: mutating games after a cutoff must not change any feature
  row at or before the cutoff (features are pure functions of the strict past).
* SAME-DAY MUTATION: mutating the outcomes/stats of every game ON a date D
  must not change the feature rows OF date D's games — within-day games
  (including a game's own outcome) are invisible to that day's features.
* FAKE-FUTURE: appending a fabricated monster game in the future must leave
  every original row byte-identical.
* SENTINEL: on synthetic data with *zero* real signal (equal, static teams,
  no blue bonus), no feature may correlate with the current game's outcome
  beyond what a label-permutation null allows — any excess correlation would
  mean current-game information leaked into features.
* WITHIN-DAY: sibling games of a same-day Bo3 must see identical state, and
  intraday timestamps must not break within-day mutual invisibility.
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


def test_same_day_mutation_does_not_change_that_days_features(base_games, base_feats):
    """Mutating all games ON date D leaves date-D feature rows byte-identical.

    Features for date D may depend only on games from strictly earlier dates
    — not on same-day games, including the row's own outcome.  Rows after D
    are allowed (and expected) to change.
    """
    D = CUTOFF  # a mid-history date with several games
    mutated = base_games.copy()
    on_d = mutated["date"] == D
    assert on_d.sum() >= 3, "need several games on the attack date"
    assert (mutated["date"] > D).any()

    mutated.loc[on_d, "blue_win"] = 1 - mutated.loc[on_d, "blue_win"]
    mutated.loc[on_d, "blue_golddiffat15"] = (
        -3.0 * mutated.loc[on_d, "blue_golddiffat15"] + 12345.0
    )
    mutated.loc[on_d, "red_golddiffat15"] = -mutated.loc[on_d, "blue_golddiffat15"]

    feats_mut = build_matchup_features(mutated)

    # feature rows FOR date D (and everything before) are untouched;
    # meta (blue_win) differs on D by construction, so compare features only.
    upto = (base_feats["date"] <= D).to_numpy()
    pd.testing.assert_frame_equal(
        base_feats.loc[upto, FEATURE_COLUMNS].reset_index(drop=True),
        feats_mut.loc[upto, FEATURE_COLUMNS].reset_index(drop=True),
        check_exact=True,
    )

    # ... and the attack has teeth: some later row must have changed.
    after = (base_feats["date"] > D).to_numpy()
    assert not base_feats.loc[after, FEATURE_COLUMNS].equals(
        feats_mut.loc[after, FEATURE_COLUMNS]
    ), "mutating date D changed nothing after D — attack is inert"


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


def _max_abs_corr(feats: pd.DataFrame, y: np.ndarray) -> tuple[float, str, int]:
    """Max |corr(feature, y)| over all usable features.

    Returns ``(max_abs_corr, argmax_column, n_features_checked)``.  Features
    with < 100 valid rows or zero variance (on their valid mask) are skipped.
    """
    best, best_col, checked = 0.0, "", 0
    for col in FEATURE_COLUMNS:
        x = feats[col].to_numpy(dtype=float)
        valid = ~np.isnan(x)
        if valid.sum() < 100:
            continue
        xv, yv = x[valid], y[valid]
        if np.std(xv) == 0 or np.std(yv) == 0:
            continue  # constant feature: cannot leak
        corr = abs(float(np.corrcoef(xv, yv)[0, 1]))
        checked += 1
        if corr > best:
            best, best_col = corr, col
    return best, best_col, checked


def test_sentinel_no_feature_correlates_with_outcome_when_no_signal_exists():
    """Permutation-null-calibrated sentinel (replaces a loose |r| < 0.35).

    On zero-signal data the features are functions of PAST outcomes only, so
    max-|corr| against the CURRENT outcome must be statistically
    indistinguishable from the same statistic under label permutation (which
    destroys any real feature->outcome relationship by construction).  The
    threshold is calibrated from 20 seeded permutations:

        real_max < max(1.5 * q95(null_maxes), 0.15)

    1.5 * the null's 95th percentile allows for the real statistic being one
    more draw from the null distribution; the 0.15 absolute floor guards
    against flakiness if the null happens to come out unusually tight.  Both
    are far stricter than the old fixed 0.35, and everything is seeded, so
    the test is deterministic.
    """
    games = generate_synthetic_games(
        n_teams=8, n_days=250, seed=2,
        strength_sd=0.0, drift_sd=0.0, blue_bonus=0.0,
    )
    assert len(games) > 2000
    feats = build_matchup_features(games)
    y = feats["blue_win"].to_numpy(dtype=float)

    real_max, worst_col, checked = _max_abs_corr(feats, y)
    assert checked >= 15  # the suite actually exercised most features

    rng = np.random.default_rng(0)
    null_maxes = np.array(
        [_max_abs_corr(feats, rng.permutation(y))[0] for _ in range(20)]
    )
    threshold = max(1.5 * float(np.quantile(null_maxes, 0.95)), 0.15)
    assert real_max < threshold, (
        f"{worst_col} correlates with the current game's outcome "
        f"(|r|={real_max:.3f} >= threshold {threshold:.3f}, "
        f"null q95={np.quantile(null_maxes, 0.95):.3f}) on zero-signal "
        "data -> leakage"
    )


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


def test_intraday_timestamps_do_not_break_within_day_invisibility(base_games):
    """Same-day games with distinct clock times stay mutually invisible.

    Give every game on a mid-history date D a distinct intraday time
    (09:00, 10:00, ...), build, then REVERSE the time assignment (the
    formerly-last game now happens first) and build again.  Since features
    for date D may only use games from strictly earlier dates, the feature
    rows for date D's games must be identical regardless of the time order
    (later dates may differ — state-update order within D can shift Elo).
    """
    D = CUTOFF
    on_d = (base_games["date"] == D).to_numpy()
    n_d = int(on_d.sum())
    assert n_d >= 2, "need multiple games on the attack date"

    times_fwd = [D + pd.Timedelta(hours=9 + i) for i in range(n_d)]

    def with_times(times) -> pd.DataFrame:
        g = base_games.copy()
        g.loc[on_d, "date"] = times
        return g

    feats_fwd = build_matchup_features(with_times(times_fwd))
    feats_rev = build_matchup_features(with_times(times_fwd[::-1]))

    day = lambda f: pd.to_datetime(f["date"]).dt.normalize() == D  # noqa: E731
    fwd_d = feats_fwd[day(feats_fwd)].set_index("gameid").sort_index()
    rev_d = feats_rev[day(feats_rev)].set_index("gameid").sort_index()
    assert len(fwd_d) == n_d and len(rev_d) == n_d

    pd.testing.assert_frame_equal(
        fwd_d[FEATURE_COLUMNS],
        rev_d[FEATURE_COLUMNS],
        check_exact=True,
    )
