"""Tests for lolpred.backtest.walkforward (folds + out-of-sample harness)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lolpred.backtest.walkforward import (
    FoldSpec,
    fold_masks,
    make_folds,
    run_walkforward,
)
from lolpred.models.xgb import WinModel


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _daily_dates(start: str, end: str, per_day: int = 1) -> pd.Series:
    days = pd.date_range(start, end, freq="D")
    return pd.Series(np.repeat(days, per_day))


def make_feature_frame(n=3000, start="2019-01-01", end="2021-12-31", seed=0):
    """Small synthetic feature frame: blue_win ~ Bernoulli(sigmoid(1.5*elo))."""
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, end, freq="D")
    dates = pd.Series(np.sort(rng.choice(days, size=n, replace=True)))
    elo = rng.normal(0.0, 1.0, size=n)
    noise = rng.normal(0.0, 1.0, size=n)
    y = (rng.random(n) < _sigmoid(1.5 * elo)).astype(int)
    return pd.DataFrame(
        {
            "gameid": [f"g{i:05d}" for i in range(n)],
            "date": dates,
            "league": "SYN",
            "blue_win": y,
            "f_elo_diff": elo,
            "f_x_diff": noise,
        }
    )


class SpyModel:
    """Fake model recording the index of every row it was fit on."""

    def __init__(self, log: list):
        self._log = log

    def fit(self, X, y, X_val=None, y_val=None, X_cal=None, y_cal=None):
        seen = list(X.index)
        if X_val is not None:
            seen += list(X_val.index)
        if X_cal is not None:
            seen += list(X_cal.index)
        self._log.append(seen)
        return self

    def predict_proba(self, X):
        return np.full(len(X), 0.5)


class SliceSpy:
    """Fake model recording the fit/val/cal slices separately."""

    def __init__(self, log: list):
        self._log = log

    def fit(self, X, y, X_val=None, y_val=None, X_cal=None, y_cal=None):
        self._log.append(
            {
                "fit": list(X.index),
                "val": None if X_val is None else list(X_val.index),
                "cal": None if X_cal is None else list(X_cal.index),
            }
        )
        return self

    def predict_proba(self, X):
        return np.full(len(X), 0.5)


# --------------------------------------------------------------- make_folds


class TestMakeFolds:
    def test_expected_boundaries(self):
        dates = _daily_dates("2019-01-01", "2021-12-31")
        folds = make_folds(dates, burn_in_end="2019-12-31", fold_months=6)
        assert len(folds) == 4
        expected = [
            ("2020-01-01", "2020-06-30"),
            ("2020-07-01", "2020-12-31"),
            ("2021-01-01", "2021-06-30"),
            ("2021-07-01", "2021-12-31"),
        ]
        for fold, (ts, te) in zip(folds, expected, strict=True):
            assert fold.test_start == pd.Timestamp(ts)
            assert fold.test_end == pd.Timestamp(te)
            assert fold.train_end == fold.test_start
            assert fold.is_holdout is False
            assert fold.gap_days == 7

    def test_skips_small_chunks(self):
        # Dense through 2020 H1, only 10 games in H2 -> H2 chunk dropped.
        # (Dates start inside the burn-in so the 30% fallback does not fire.)
        dense = _daily_dates("2019-07-01", "2020-06-30")
        sparse = _daily_dates("2020-07-01", "2020-07-10")
        dates = pd.concat([dense, sparse], ignore_index=True)
        folds = make_folds(dates, burn_in_end="2019-12-31", fold_months=6)
        assert len(folds) == 1
        assert folds[0].test_start == pd.Timestamp("2020-01-01")

    def test_burn_in_fallback_first_30pct(self):
        # Data starts after the default burn_in_end -> burn-in = first 30% of span.
        dates = _daily_dates("2020-01-01", "2021-12-31")
        folds = make_folds(dates)  # default burn_in_end=2018-12-31
        assert folds, "fallback should still produce folds"
        dmin, dmax = dates.min(), dates.max()
        burn_cut = dmin + 0.30 * (dmax - dmin)
        first_test = min(f.test_start for f in folds)
        assert first_test > burn_cut - pd.Timedelta(days=1)
        # nothing in the first 30% is ever a test date
        for f in folds:
            assert f.test_start > burn_cut - pd.Timedelta(days=1)

    def test_holdout_isolation(self):
        dates = _daily_dates("2019-01-01", "2021-12-31")
        holdout_start = pd.Timestamp("2021-06-01")
        folds = make_folds(
            dates, burn_in_end="2019-06-30", holdout_start=holdout_start
        )
        holdouts = [f for f in folds if f.is_holdout]
        regular = [f for f in folds if not f.is_holdout]
        assert len(holdouts) == 1
        hold = holdouts[0]
        assert hold.test_start == holdout_start
        assert hold.test_end == pd.Timestamp("2021-12-31")
        assert hold is folds[-1]
        # No non-holdout fold touches any holdout date.
        for f in regular:
            assert f.test_end < holdout_start
        # Every date >= holdout_start is covered only by the holdout fold.
        d = pd.to_datetime(dates)
        for f in regular:
            in_fold = (d >= f.test_start) & (d <= f.test_end)
            assert not (d[in_fold] >= holdout_start).any()

    def test_gap_respected_via_fake_run(self):
        """No training row (incl. validation rows) within gap of test_start."""
        feats = make_feature_frame(n=2000, seed=1)
        folds = make_folds(feats["date"], burn_in_end="2019-12-31", gap_days=7)
        assert folds
        fit_log: list[list] = []
        preds = run_walkforward(
            feats,
            folds,
            model_factory=lambda: SpyModel(fit_log),
            baseline_factory=lambda: SpyModel([]),
            verbose=False,
        )
        assert len(fit_log) == len(preds["fold_id"].unique())
        date_by_idx = feats["date"]
        fold_ids = sorted(preds["fold_id"].unique())
        for seen, fid in zip(fit_log, fold_ids, strict=True):
            fold = folds[fid]
            max_train_date = date_by_idx.loc[seen].max()
            assert max_train_date < fold.test_start - pd.Timedelta(days=fold.gap_days)


# ----------------------------------------------------------- run_walkforward


@pytest.fixture(scope="module")
def wf_result():
    feats = make_feature_frame(n=3000, seed=0)
    folds = make_folds(feats["date"])
    factory = lambda: WinModel(  # noqa: E731
        params={"n_estimators": 60, "max_depth": 3, "n_jobs": 2}, seed=0
    )
    preds = run_walkforward(feats, folds, model_factory=factory, verbose=False)
    return feats, folds, preds


class TestRunWalkforward:
    def test_probs_finite_in_open_interval(self, wf_result):
        _, _, preds = wf_result
        for col in ("model_p", "baseline_p"):
            p = preds[col].to_numpy()
            assert np.isfinite(p).all()
            assert ((p > 0.0) & (p < 1.0)).all()

    def test_every_row_out_of_sample(self, wf_result):
        _, folds, preds = wf_result
        dates = pd.to_datetime(preds["date"]).dt.normalize()
        for fid, grp in preds.groupby("fold_id"):
            fold = folds[fid]
            g = dates.loc[grp.index]
            assert (g >= fold.test_start).all()
            assert (g <= fold.test_end).all()

    def test_beats_coin_flip_on_logloss(self, wf_result):
        _, _, preds = wf_result
        y = preds["blue_win"].to_numpy(dtype=float)
        p = np.clip(preds["model_p"].to_numpy(), 1e-15, 1 - 1e-15)
        ll = -np.mean(y * np.log(p) + (1 - y) * np.log1p(-p))
        assert ll < np.log(2.0)

    def test_covers_exactly_union_of_test_slices(self, wf_result):
        feats, folds, preds = wf_result
        expected: set[str] = set()
        for fold in folds:
            _, test_mask = fold_masks(feats, fold)
            expected |= set(feats.loc[test_mask, "gameid"])
        got = list(preds["gameid"])
        assert len(got) == len(set(got)), "duplicate test rows across folds"
        assert set(got) == expected

    def test_output_columns(self, wf_result):
        _, _, preds = wf_result
        for col in ("gameid", "date", "blue_win", "model_p", "baseline_p", "fold_id", "is_holdout"):
            assert col in preds.columns
        assert not any(c.startswith("f_") for c in preds.columns)

    def test_deterministic_across_runs(self, wf_result):
        feats, folds, preds1 = wf_result
        factory = lambda: WinModel(  # noqa: E731
            params={"n_estimators": 60, "max_depth": 3, "n_jobs": 2}, seed=0
        )
        preds2 = run_walkforward(feats, folds, model_factory=factory, verbose=False)
        pd.testing.assert_frame_equal(preds1, preds2)


class TestTemporalIntegrity:
    def test_max_train_date_before_gap(self):
        """Regression: for every fold, max train date < test_start - gap."""
        feats = make_feature_frame(n=1500, seed=3)
        for gap in (1, 7, 30):
            folds = make_folds(feats["date"], burn_in_end="2019-12-31", gap_days=gap)
            assert folds
            day = pd.to_datetime(feats["date"]).dt.normalize()
            for fold in folds:
                train_mask, test_mask = fold_masks(feats, fold)
                assert train_mask.any() and test_mask.any()
                assert not (train_mask & test_mask).any()
                max_train = day[train_mask].max()
                assert max_train < fold.test_start - pd.Timedelta(days=gap)
                assert day[test_mask].min() >= fold.test_start

    def test_holdout_fold_flagged_in_output(self):
        feats = make_feature_frame(n=2000, seed=4)
        folds = make_folds(
            feats["date"], burn_in_end="2019-12-31", holdout_start="2021-06-01"
        )
        fit_log: list[list] = []
        preds = run_walkforward(
            feats,
            folds,
            model_factory=lambda: SpyModel(fit_log),
            baseline_factory=lambda: SpyModel([]),
            verbose=False,
        )
        hold_dates = pd.to_datetime(preds.loc[preds["is_holdout"], "date"])
        reg_dates = pd.to_datetime(preds.loc[~preds["is_holdout"], "date"])
        assert len(hold_dates) > 0
        assert (hold_dates >= pd.Timestamp("2021-06-01")).all()
        assert (reg_dates < pd.Timestamp("2021-06-01")).all()


class TestValCalSplit:
    """The train-tail carving: val/cal halves for big tails, val-only small."""

    @staticmethod
    def _run(n: int, calib_frac: float = 0.15):
        feats = make_feature_frame(n=n, seed=2)
        folds = make_folds(feats["date"], burn_in_end="2019-12-31")
        assert folds
        model_log: list[dict] = []
        base_log: list[list] = []
        run_walkforward(
            feats,
            folds,
            model_factory=lambda: SliceSpy(model_log),
            baseline_factory=lambda: SpyModel(base_log),
            calib_frac=calib_frac,
            verbose=False,
        )
        assert model_log
        return model_log, base_log

    def test_big_tail_split_val_then_cal(self):
        # n=6000 -> every fold's tail >= 200 -> chronological val/cal halves.
        model_log, _ = self._run(n=6000)
        for call in model_log:
            fit, val, cal = call["fit"], call["val"], call["cal"]
            assert val is not None and cal is not None
            tail = len(val) + len(cal)
            assert tail >= 200
            # halves differ by at most one row; cal gets the odd row
            assert len(cal) - len(val) in (0, 1)
            # chronological: fit < val < cal (index is date-sorted RangeIndex)
            assert max(fit) < min(val)
            assert max(val) < min(cal)

    def test_small_tail_keeps_single_slice(self):
        # n=1200 -> every fold's tail < 200 -> whole tail is val, no cal.
        model_log, _ = self._run(n=1200)
        for call in model_log:
            assert call["cal"] is None
            assert call["val"] is not None
            assert 2 <= len(call["val"]) < 200
            assert max(call["fit"]) < min(call["val"])

    def test_baseline_fit_on_full_train_slice(self):
        # The baseline (no early stopping/calibration) must see fit+val+cal.
        for n in (1200, 6000):
            model_log, base_log = self._run(n=n)
            assert len(base_log) == len(model_log)
            for m, b in zip(model_log, base_log, strict=True):
                expected = set(m["fit"])
                expected |= set(m["val"] or [])
                expected |= set(m["cal"] or [])
                assert set(b) == expected


class TestFoldSpec:
    def test_frozen(self):
        f = FoldSpec(
            train_end=pd.Timestamp("2020-01-01"),
            test_start=pd.Timestamp("2020-01-01"),
            test_end=pd.Timestamp("2020-06-30"),
        )
        with pytest.raises(Exception):
            f.test_start = pd.Timestamp("2019-01-01")  # type: ignore[misc]
