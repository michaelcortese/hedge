"""Tests for lolpred.models.xgb (WinModel, EloLogisticBaseline)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import log_loss

from lolpred.models.xgb import PERSP_COL, EloLogisticBaseline, WinModel

# Small booster so the suite stays fast; overrides merge over defaults.
# n_jobs capped: unbounded OpenMP threads oversubscribe badly on small data.
FAST_PARAMS = {
    "n_estimators": 120,
    "learning_rate": 0.2,
    "max_depth": 3,
    "min_child_weight": 5,
    "n_jobs": 2,
}


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def make_data(n: int = 2000, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Synthetic matchup frame: two _diff features, one symmetric context col.

    y ~ Bernoulli(sigmoid(1.2*f_a_diff - 0.4*f_b_diff + 0.15)); the +0.15
    intercept is the "blue advantage".
    """
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    ctx = rng.normal(size=n)
    p = sigmoid(1.2 * a - 0.4 * b + 0.15)
    y = (rng.random(n) < p).astype(int)
    X = pd.DataFrame({"f_a_diff": a, "f_b_diff": b, "f_ctx": ctx})
    return X, y


def mirror(X: pd.DataFrame) -> pd.DataFrame:
    """Manually mirror: negate _diff columns, leave symmetric columns alone."""
    Xm = X.copy()
    for c in X.columns:
        if c.endswith("_diff"):
            Xm[c] = -Xm[c]
    return Xm


@pytest.fixture(scope="module")
def data():
    X, y = make_data(n=2400, seed=7)
    return X.iloc[:1600], y[:1600], X.iloc[1600:], y[1600:]


@pytest.fixture(scope="module")
def fitted_model(data):
    X_tr, y_tr, _, _ = data
    return WinModel(params=FAST_PARAMS, calibrate=None, seed=0).fit(X_tr, y_tr)


# --------------------------------------------------------------- WinModel


def test_orientation_semantics_blue_bump_survives(fitted_model, data):
    """Team-swap is a different physical game: p + p(swap) - 1 equals the
    learned blue bump (twice), which must be positive here (data intercept
    +0.15) and of plausible size — NOT forced to zero."""
    _, _, X_te, _ = data
    p = fitted_model.predict_proba(X_te)
    p_mirr = fitted_model.predict_proba(mirror(X_te))
    bump = np.mean(p + p_mirr - 1.0)
    assert 0.01 < bump < 0.2  # sigmoid(0.15)-0.5 ~= 0.037 at even matchups
    # Base rate is recovered: features are symmetric, so mean(p) tracks the
    # intercept-driven base rate sigmoid(0.15) ~= 0.537.
    assert 0.50 < p.mean() < 0.60


def test_near_antisymmetry_without_side_effect():
    """When the training data has NO side advantage, predictions should be
    approximately antisymmetric under team swap."""
    rng = np.random.default_rng(3)
    n = 2000
    a = rng.normal(size=n)
    X = pd.DataFrame({"f_a_diff": a, "f_ctx": rng.normal(size=n)})
    y = (rng.random(n) < sigmoid(1.3 * a)).astype(int)
    m = WinModel(params=FAST_PARAMS, calibrate=None, seed=0).fit(X, y)
    p = m.predict_proba(X)
    p_mirr = m.predict_proba(mirror(X))
    assert np.mean(np.abs(p + p_mirr - 1.0)) < 0.04


def test_better_than_chance_and_direction(fitted_model, data):
    _, _, X_te, y_te = data
    p = fitted_model.predict_proba(X_te)
    assert log_loss(y_te, p) < 0.69
    # Recovers the direction of the dominant feature.
    assert np.corrcoef(p, X_te["f_a_diff"].to_numpy())[0, 1] > 0


def test_early_stopping_engages(data):
    X_tr, y_tr, X_te, y_te = data
    n_estimators = 500
    model = WinModel(
        params={**FAST_PARAMS, "n_estimators": n_estimators, "learning_rate": 0.3,
                "early_stopping_rounds": 20},
        calibrate=None,
        seed=0,
    ).fit(X_tr, y_tr, X_val=X_te, y_val=y_te)
    assert model.best_iteration_ is not None
    assert model.best_iteration_ < n_estimators - 1


def test_no_val_uses_reduced_fixed_budget():
    # Without a validation set (no early stopping possible) and without an
    # explicit n_estimators override, the budget is cut to 400.
    model = WinModel(params={"n_jobs": 2}, calibrate=None, seed=0)
    X, y = make_data(n=300, seed=1)
    model.fit(X, y)
    assert model.model_.get_params()["n_estimators"] == 400
    # An explicit override is respected.
    model2 = WinModel(params=FAST_PARAMS, calibrate=None, seed=0).fit(X, y)
    assert model2.model_.get_params()["n_estimators"] == FAST_PARAMS["n_estimators"]


def test_calibration_flag_and_antisymmetry(data):
    X_tr, y_tr, X_te, y_te = data
    # Validation set -> calibrator fit on val predictions.
    m_cal = WinModel(params=FAST_PARAMS, calibrate="platt", seed=0).fit(
        X_tr, y_tr, X_val=X_te, y_val=y_te
    )
    assert m_cal.calibrated_ is True
    p = m_cal.predict_proba(X_te)
    assert np.all((p > 0) & (p < 1))
    # Calibration (with intercept) keeps the predicted base rate near the
    # observed one rather than forcing symmetry.
    assert abs(p.mean() - y_te.mean()) < 0.05
    # No validation set -> calibration silently skipped.
    m_nocal = WinModel(params=FAST_PARAMS, calibrate="platt", seed=0).fit(X_tr, y_tr)
    assert m_nocal.calibrated_ is False


def test_save_load_roundtrip(fitted_model, data, tmp_path):
    _, _, X_te, _ = data
    path = tmp_path / "model.joblib"
    fitted_model.save(path)
    loaded = WinModel.load(path)
    np.testing.assert_allclose(
        loaded.predict_proba(X_te), fitted_model.predict_proba(X_te)
    )


def test_predict_reorders_columns_by_name(fitted_model, data):
    _, _, X_te, _ = data
    shuffled = X_te[list(reversed(list(X_te.columns)))]
    np.testing.assert_allclose(
        fitted_model.predict_proba(shuffled), fitted_model.predict_proba(X_te)
    )


def test_predict_raises_on_column_mismatch(fitted_model, data):
    _, _, X_te, _ = data
    with pytest.raises(ValueError):
        fitted_model.predict_proba(X_te.drop(columns=["f_b_diff"]))
    with pytest.raises(ValueError):
        fitted_model.predict_proba(X_te.assign(f_extra_diff=1.0))


def test_reserved_persp_column_rejected():
    X, y = make_data(n=100, seed=2)
    X[PERSP_COL] = 1.0
    with pytest.raises(ValueError):
        WinModel(params=FAST_PARAMS, calibrate=None).fit(X, y)


def test_nan_handling(data):
    X_tr, y_tr, X_te, _ = data
    rng = np.random.default_rng(3)
    X_tr = X_tr.copy()
    X_tr.loc[X_tr.sample(frac=0.1, random_state=1).index, "f_a_diff"] = np.nan
    model = WinModel(params=FAST_PARAMS, calibrate=None, seed=0).fit(X_tr, y_tr)
    X_te = X_te.copy()
    X_te.iloc[rng.choice(len(X_te), size=50, replace=False),
              X_te.columns.get_loc("f_b_diff")] = np.nan
    p = model.predict_proba(X_te)
    assert np.all(np.isfinite(p))
    assert np.all((p >= 0) & (p <= 1))


def test_deterministic_given_seed(data):
    X_tr, y_tr, X_te, _ = data
    p1 = WinModel(params=FAST_PARAMS, calibrate=None, seed=42).fit(
        X_tr, y_tr
    ).predict_proba(X_te)
    p2 = WinModel(params=FAST_PARAMS, calibrate=None, seed=42).fit(
        X_tr, y_tr
    ).predict_proba(X_te)
    np.testing.assert_array_equal(p1, p2)


# ---------------------------------------------------- EloLogisticBaseline


def make_elo_data(n: int = 1500, seed: int = 5):
    rng = np.random.default_rng(seed)
    elo = rng.normal(scale=1.0, size=n)
    p = sigmoid(0.9 * elo + 0.1)
    y = (rng.random(n) < p).astype(int)
    return pd.DataFrame({"f_elo_diff": elo}), y


def test_baseline_fits_and_is_monotone():
    X, y = make_elo_data()
    base = EloLogisticBaseline(seed=0).fit(X, y)
    grid = pd.DataFrame({"f_elo_diff": np.linspace(-3, 3, 41)})
    p = base.predict_proba(grid)
    assert p.ndim == 1 and len(p) == len(grid)
    assert np.all((p > 0) & (p < 1))
    assert np.all(np.diff(p) > 0)  # monotone increasing in elo_diff


def test_baseline_uses_available_columns_and_raises_if_none():
    X, y = make_elo_data(n=400)
    X = X.assign(f_bt_theta_diff=X["f_elo_diff"] * 0.5, f_other=1.0)
    base = EloLogisticBaseline().fit(X, y)
    assert base.columns_ == ["f_elo_diff", "f_bt_theta_diff"]
    with pytest.raises(ValueError):
        EloLogisticBaseline().fit(pd.DataFrame({"f_other_diff": y * 1.0}), y)


def test_baseline_nan_handling():
    X, y = make_elo_data(n=600)
    X = X.copy()
    X.iloc[::7, 0] = np.nan
    base = EloLogisticBaseline().fit(X, y)
    p = base.predict_proba(X)
    assert np.all(np.isfinite(p))
    assert np.all((p > 0) & (p < 1))


def test_mirror_swaps_blue_red_pairs():
    """f_hist_games_blue/red must swap under mirroring so that externally
    swapping the two teams is exactly the model's internal mirror."""
    import numpy as np
    import pandas as pd
    from lolpred.models.xgb import WinModel

    rng = np.random.default_rng(7)
    n = 1500
    X = pd.DataFrame(
        {
            "f_a_diff": rng.normal(size=n),
            "f_hist_games_blue": rng.integers(0, 200, size=n).astype(float),
            "f_hist_games_red": rng.integers(0, 200, size=n).astype(float),
            "f_ctx": rng.normal(size=n),
        }
    )
    y = (rng.random(n) < 1 / (1 + np.exp(-1.5 * X["f_a_diff"]))).astype(int)
    m = WinModel(params={"n_estimators": 60, "max_depth": 3, "n_jobs": 2}, seed=0)
    m.fit(X, y)
    p = m.predict_proba(X)
    X_swapped = X.copy()
    X_swapped["f_a_diff"] = -X["f_a_diff"]
    X_swapped["f_hist_games_blue"] = X["f_hist_games_red"]
    X_swapped["f_hist_games_red"] = X["f_hist_games_blue"]
    p_swapped = m.predict_proba(X_swapped)
    # Data has no side effect, so external swap should be ~complementary.
    # (Exact complement is NOT the contract — the persp column is active.)
    assert np.mean(np.abs(p_swapped - (1.0 - p))) < 0.04
