"""Walk-forward backtest (CONTRACTS.md section 6).

Expanding-window, strictly out-of-sample evaluation:

* :func:`make_folds` tiles the post-burn-in date span into ``fold_months``
  chunks; each chunk is one test fold and training uses only games dated
  strictly earlier than ``test_start - gap_days`` (an embargo gap so that
  same-week information cannot leak across the boundary).
* :func:`run_walkforward` fits a fresh model + baseline per fold and collects
  their predictions on the fold's test slice.  Every returned row is out of
  sample by construction.

Date semantics
--------------
Fold boundaries are calendar days (midnight Timestamps).  Fold membership and
the training cutoff are evaluated on *normalized* dates (``d.dt.normalize()``)
so intraday timestamps cannot fall into cracks between folds: a game belongs
to a fold iff ``test_start <= day(d) <= test_end``, and is trainable for that
fold iff ``day(d) < test_start - gap_days``.

Burn-in fallback
----------------
If the data starts after ``burn_in_end`` (e.g. synthetic 2020+ fixtures), the
fixed burn-in date is meaningless, so it falls back to the first 30% of the
observed date span: ``burn_in_end = min(dates) + 0.30 * (max(dates) -
min(dates))`` (normalized to midnight).  Folds then tile the remaining 70%.

Deviation from the pinned contract (documented): :class:`FoldSpec` carries two
extra fields beyond ``train_end/test_start/test_end`` — ``is_holdout`` (per
the holdout requirement) and ``gap_days``, because :func:`run_walkforward`
takes no gap parameter and must reproduce the exact training cutoff the folds
were built with.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from lolpred.models.xgb import PERSP_COL, EloLogisticBaseline, WinModel

__all__ = ["FoldSpec", "make_folds", "fold_masks", "run_walkforward"]

logger = logging.getLogger(__name__)

#: A tiled chunk with fewer test games than this is skipped.
MIN_TEST_GAMES = 50

#: A fold whose training slice is smaller than this is skipped (cannot fit).
_MIN_TRAIN_ROWS = 30

#: Minimum held-out tail size (rows) before it is split into separate
#: early-stopping and calibration halves.  Below this, splitting makes both
#: jobs worse (early stopping gets a noisy stopping signal AND the Platt fit
#: gets too few points), so the whole tail does double duty as val set.
_MIN_TAIL_FOR_CAL_SPLIT = 200


@dataclass(frozen=True)
class FoldSpec:
    """One walk-forward fold.

    ``train_end`` equals ``test_start``; training selection additionally
    applies the embargo: train on ``date < train_end - gap_days`` (strictly).
    ``test_start``/``test_end`` bound the test slice, both inclusive, as
    calendar days.  ``is_holdout`` marks the single final holdout fold.
    """

    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    is_holdout: bool = False
    gap_days: int = 7


def make_folds(
    dates: pd.Series,
    burn_in_end: str | pd.Timestamp = "2018-12-31",
    fold_months: int = 6,
    gap_days: int = 7,
    holdout_start: str | pd.Timestamp | None = None,
) -> list[FoldSpec]:
    """Tile the span after ``burn_in_end`` into test folds of ``fold_months``.

    Parameters
    ----------
    dates:
        Game dates (any datetime-coercible Series/array).  Only used to place
        boundaries and count games per chunk.
    burn_in_end:
        Last day of the never-tested burn-in period.  If the data starts
        after this date, falls back to the first 30% of the date span (see
        module docstring).
    fold_months:
        Calendar length of each test chunk.
    gap_days:
        Embargo between the end of a fold's training data and its
        ``test_start`` (training uses ``date < test_start - gap_days``).
    holdout_start:
        If given, no tiled fold may include any date ``>= holdout_start``;
        instead a single final fold covering ``[holdout_start, max(dates)]``
        is appended with ``is_holdout=True`` (only if it contains any games).

    Chunks with fewer than :data:`MIN_TEST_GAMES` test games are skipped.
    """
    d = pd.to_datetime(pd.Series(dates)).dropna().dt.normalize()
    if d.empty:
        raise ValueError("dates is empty")
    dmin, dmax = d.min(), d.max()

    burn = pd.Timestamp(burn_in_end)
    if dmin > burn:
        # Burn-in fallback: first 30% of the observed span (module docstring).
        # This uses max(dates), which is technically future-dependent, but it
        # is harmless: it only PLACES fold boundaries and carries no outcome
        # information — every fold still trains strictly on the past and
        # tests strictly on the future of its own boundary.
        burn = (dmin + 0.30 * (dmax - dmin)).normalize()

    hold = pd.Timestamp(holdout_start) if holdout_start is not None else None
    tile_end = dmax if hold is None else min(dmax, hold - pd.Timedelta(days=1))

    folds: list[FoldSpec] = []
    cur = burn.normalize() + pd.Timedelta(days=1)
    while cur <= tile_end:
        nxt = cur + pd.DateOffset(months=fold_months)
        test_end = min(nxt - pd.Timedelta(days=1), tile_end)
        n_test = int(((d >= cur) & (d <= test_end)).sum())
        if n_test >= MIN_TEST_GAMES:
            folds.append(
                FoldSpec(
                    train_end=cur,
                    test_start=cur,
                    test_end=test_end,
                    is_holdout=False,
                    gap_days=gap_days,
                )
            )
        else:
            logger.debug(
                "make_folds: skipping chunk %s..%s with only %d test games",
                cur.date(),
                test_end.date(),
                n_test,
            )
        cur = nxt

    if hold is not None:
        n_hold = int((d >= hold).sum())
        if n_hold > 0:
            folds.append(
                FoldSpec(
                    train_end=hold,
                    test_start=hold,
                    test_end=dmax,
                    is_holdout=True,
                    gap_days=gap_days,
                )
            )
    return folds


def fold_masks(feats: pd.DataFrame, fold: FoldSpec) -> tuple[np.ndarray, np.ndarray]:
    """Boolean (train_mask, test_mask) over ``feats`` rows for one fold.

    Evaluated on normalized (calendar-day) dates.  The train mask applies the
    embargo: ``day(date) < test_start - gap_days`` strictly.  This is the one
    choke point for temporal visibility in the walk-forward — used both by
    :func:`run_walkforward` and by the leakage regression tests.
    """
    day = pd.to_datetime(feats["date"]).dt.normalize()
    cutoff = fold.test_start - pd.Timedelta(days=fold.gap_days)
    train = (day < cutoff).to_numpy()
    test = ((day >= fold.test_start) & (day <= fold.test_end)).to_numpy()
    return train, test


def _fold_logloss(y: np.ndarray, p: np.ndarray) -> float:
    """Binary log-loss robust to single-class folds."""
    return float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1]))


def run_walkforward(
    feats: pd.DataFrame,
    folds: list[FoldSpec],
    model_factory=None,
    baseline_factory=None,
    calib_frac: float = 0.15,
    seed: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fit per fold, predict the test slice, collect out-of-sample rows.

    Parameters
    ----------
    feats:
        Feature-builder output: meta columns (must include ``date``,
        ``blue_win``; typically also ``gameid`` etc.) plus feature columns
        prefixed ``f_``.  The feature matrix passed to the models is exactly
        the ``f_`` columns — except the reserved ``f_is_blue_persp``, which
        :class:`WinModel` forbids in caller frames and is dropped defensively.
    folds:
        Output of :func:`make_folds` (each fold carries its own ``gap_days``).
    model_factory / baseline_factory:
        Zero-arg callables returning fresh unfitted models per fold; both
        must expose ``fit(X, y, X_val, y_val)`` and ``predict_proba(X)``,
        and the model factory's ``fit`` must additionally accept
        ``X_cal``/``y_cal`` keyword args (they are only passed when a
        calibration slice exists — see ``calib_frac``).  The baseline is fit
        on the FULL training slice (tail included) with no validation set:
        it uses neither early stopping nor calibration, so withholding the
        most recent rows from it would only handicap it.
        Defaults: ``WinModel(seed=seed)`` / ``EloLogisticBaseline(seed=seed)``.
    calib_frac:
        Within each fold's training slice, the chronologically LAST
        ``calib_frac`` of rows is held out of the model's training rows.
        When that tail has at least ``_MIN_TAIL_FOR_CAL_SPLIT`` (200) rows it
        is split chronologically in half: the FIRST half is the
        early-stopping validation set (``X_val``/``y_val``) and the SECOND
        (most recent) half is the calibration slice (``X_cal``/``y_cal``),
        so the Platt calibrator is not fit on the same rows the stopping
        point was chosen on.  Smaller tails keep the old single-slice
        behavior (the whole tail is ``X_val``, no ``X_cal``): splitting a
        tiny tail makes both jobs worse — a noisier stopping signal AND too
        few calibration points.  If the tail would have < 2 rows, the fold
        is fit without a validation set at all.
    verbose:
        Log per-fold progress at INFO (else DEBUG).

    Returns
    -------
    DataFrame with the meta columns of ``feats`` plus ``model_p``,
    ``baseline_p``, ``fold_id`` (index into ``folds``) and ``is_holdout``,
    one row per test game across all folds, chronological within fold.  The
    original ``feats`` index is preserved.  Folds with an empty test slice or
    fewer than 30 training rows are skipped with a warning.
    """
    if model_factory is None:
        model_factory = lambda: WinModel(seed=seed)  # noqa: E731
    if baseline_factory is None:
        baseline_factory = lambda: EloLogisticBaseline(seed=seed)  # noqa: E731
    if not 0.0 <= calib_frac < 1.0:
        raise ValueError(f"calib_frac must be in [0, 1), got {calib_frac}")
    level = logging.INFO if verbose else logging.DEBUG

    feats = feats.sort_values("date", kind="stable")
    feature_cols = [c for c in feats.columns if c.startswith("f_") and c != PERSP_COL]
    meta_cols = [c for c in feats.columns if not c.startswith("f_")]
    if not feature_cols:
        raise ValueError("feats has no 'f_' feature columns")
    if "blue_win" not in feats.columns:
        raise ValueError("feats must have a 'blue_win' column")

    X_all = feats[feature_cols]
    y_all = feats["blue_win"].to_numpy(dtype=int)

    out: list[pd.DataFrame] = []
    for fold_id, fold in enumerate(folds):
        train_mask, test_mask = fold_masks(feats, fold)
        tr_idx = np.flatnonzero(train_mask)  # chronological: feats is sorted
        te_idx = np.flatnonzero(test_mask)
        if len(te_idx) == 0 or len(tr_idx) < _MIN_TRAIN_ROWS:
            logger.warning(
                "fold %d (%s..%s): skipped (train=%d, test=%d)",
                fold_id,
                fold.test_start.date(),
                fold.test_end.date(),
                len(tr_idx),
                len(te_idx),
            )
            continue

        # Chronologically last calib_frac of the training slice -> held-out
        # tail.  Large tails are split in half: first half = early-stopping
        # val, second (most recent) half = calibration slice; small tails do
        # double duty as val only (see the calib_frac docstring).
        n_tail = min(int(round(calib_frac * len(tr_idx))), len(tr_idx) - 1)
        X_val: pd.DataFrame | None = None
        y_val: np.ndarray | None = None
        X_cal: pd.DataFrame | None = None
        y_cal: np.ndarray | None = None
        if n_tail >= 2:
            fit_idx, tail_idx = tr_idx[:-n_tail], tr_idx[-n_tail:]
            if n_tail >= _MIN_TAIL_FOR_CAL_SPLIT:
                half = n_tail // 2
                val_idx, cal_idx = tail_idx[:half], tail_idx[half:]
                X_cal, y_cal = X_all.iloc[cal_idx], y_all[cal_idx]
            else:
                val_idx = tail_idx
            X_val, y_val = X_all.iloc[val_idx], y_all[val_idx]
        else:
            fit_idx = tr_idx

        X_tr, y_tr = X_all.iloc[fit_idx], y_all[fit_idx]
        if X_cal is not None:
            model = model_factory().fit(X_tr, y_tr, X_val, y_val, X_cal=X_cal, y_cal=y_cal)
        else:
            model = model_factory().fit(X_tr, y_tr, X_val, y_val)
        # Baseline: full training slice (tail included), no val/cal — it has
        # no early stopping or calibration to feed, so it gets all the data.
        baseline = baseline_factory().fit(X_all.iloc[tr_idx], y_all[tr_idx])

        X_te, y_te = X_all.iloc[te_idx], y_all[te_idx]
        model_p = np.asarray(model.predict_proba(X_te), dtype=float)
        baseline_p = np.asarray(baseline.predict_proba(X_te), dtype=float)

        rows = feats.iloc[te_idx][meta_cols].copy()
        rows["model_p"] = model_p
        rows["baseline_p"] = baseline_p
        rows["fold_id"] = fold_id
        rows["is_holdout"] = fold.is_holdout
        out.append(rows)

        logger.log(
            level,
            "fold %d%s: test %s..%s | train=%d (val=%d cal=%d) test=%d | "
            "logloss model=%.4f baseline=%.4f",
            fold_id,
            " [holdout]" if fold.is_holdout else "",
            fold.test_start.date(),
            fold.test_end.date(),
            len(tr_idx),
            0 if X_val is None else len(X_val),
            0 if X_cal is None else len(X_cal),
            len(te_idx),
            _fold_logloss(y_te, model_p),
            _fold_logloss(y_te, baseline_p),
        )

    if not out:
        return pd.DataFrame(
            columns=meta_cols + ["model_p", "baseline_p", "fold_id", "is_holdout"]
        )
    return pd.concat(out)
