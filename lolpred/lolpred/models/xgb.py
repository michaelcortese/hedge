"""Win-probability models (CONTRACTS.md section 4).

``WinModel``
    XGBoost classifier wrapped with mirror augmentation, optional Platt
    calibration, and a strict feature-column contract.  Probabilities are
    always P(blue win) from the blue perspective of the row as given.

``EloLogisticBaseline``
    Logistic regression on the rating-diff features only — the "beat this"
    bar for the GBDT.

Orientation convention
----------------------
Per CONTRACTS.md the feature builder emits only antisymmetric ``*_diff``
columns, ``_blue``/``_red`` swap pairs, and orientation-invariant symmetric
context columns; rows are always blue-perspective.  Naive mirror augmentation
(negate ``_diff`` cols, flip labels) would destroy the real blue-side
advantage (~52-53% blue win rate in pro play), so a synthetic column
``f_is_blue_persp`` carries it:

* train time (mirror path only): ``+1`` on original rows, ``-1`` on mirrored
  rows.  The booster attributes the blue advantage to this column instead of
  leaking a bias into the antisymmetric ``_diff`` features.
* predict time: each physical game has exactly two training representations —
  (X, persp=+1, y) and (mirror(X), persp=-1, 1-y).  Prediction averages
  those two views of the SAME game: ``p = 0.5 * (f(X,+1) + 1 - f(mirror(X),-1))``.
  The perspective column stays active, so the blue bump survives into
  predictions.  Consequently ``p(team-swap) != 1 - p`` in general: swapping
  the teams is a different physical game (the other team now enjoys blue
  side), and ``p(X) + p(swap(X)) - 1`` equals the learned blue bump (twice).
  An earlier revision marginalized persp at predict time to force exact
  ``p(swap) == 1 - p``; review showed that structurally erases the blue-side
  edge (~0.025 probability bias on every row) — do not reintroduce it.

Calibration
-----------
``calibrate="platt"`` fits a 1-D logistic regression (with intercept — it
corrects residual base-rate offset) mapping the averaged booster logit ->
label.  Leakage rule: the calibrator is fit on the VALIDATION set
predictions; if no validation set is provided, calibration is silently
skipped and ``self.calibrated_`` is ``False``.

Early stopping (xgboost 3.x)
----------------------------
``early_stopping_rounds`` is a constructor argument of ``XGBClassifier`` and
requires an ``eval_set`` in ``fit``.  When no validation set is passed we fit
WITHOUT early stopping using a reduced fixed ``n_estimators=400`` (unless the
caller explicitly set ``n_estimators`` in ``params``), since the full default
of 1500 rounds is only safe when early stopping can cut it short.

Serialization
-------------
``save``/``load`` pickle the whole wrapper via ``joblib``.  XGBoost's native
JSON serialization is nicer for cross-version portability, but a wrapper
pickle is fine for this project (single pinned environment) and keeps the
calibrator + feature-column order in one artifact.  The saved feature column
order is enforced at predict time: unknown/missing columns raise, and a
same-set/different-order frame is reordered to match by name.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

__all__ = ["WinModel", "EloLogisticBaseline", "PERSP_COL"]

#: Synthetic perspective column added in the mirror-augmentation path.
PERSP_COL = "f_is_blue_persp"

_DEFAULT_PARAMS: dict = {
    "max_depth": 4,
    "learning_rate": 0.03,
    "n_estimators": 1500,
    "min_child_weight": 25,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_lambda": 8.0,
    "reg_alpha": 0.5,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
}

#: n_estimators used when fitting without a validation set (no early stopping)
#: and the caller did not explicitly set n_estimators.
_NO_VAL_N_ESTIMATORS = 400

#: Default early-stopping patience when a validation set is provided.
_DEFAULT_EARLY_STOPPING_ROUNDS = 50

_LOGIT_EPS = 1e-7


def _mirror_frame(X: pd.DataFrame) -> pd.DataFrame:
    """Return the mirrored orientation.

    * ``*_diff`` columns are negated (antisymmetric).
    * ``<stem>_blue`` / ``<stem>_red`` column pairs are swapped (e.g.
      ``f_hist_games_blue`` becomes the red team's count under mirroring).
    * Everything else (and not :data:`PERSP_COL`, which is never present in
      caller frames) is symmetric context and copied unchanged.

    NaNs pass through (negation of NaN is NaN).
    """
    Xm = X.copy()
    diff_cols = [c for c in X.columns if c.endswith("_diff")]
    if diff_cols:
        Xm[diff_cols] = -Xm[diff_cols].astype(float)
    for col_blue in X.columns:
        if col_blue.endswith("_blue"):
            col_red = col_blue[: -len("_blue")] + "_red"
            if col_red in X.columns:
                Xm[col_blue] = X[col_red].to_numpy(copy=True)
                Xm[col_red] = X[col_blue].to_numpy(copy=True)
    return Xm


def _logit(p: np.ndarray) -> np.ndarray:
    """Symmetric-clipped logit, so logit(1-p) == -logit(p) exactly."""
    p = np.clip(np.asarray(p, dtype=float), _LOGIT_EPS, 1.0 - _LOGIT_EPS)
    return np.log(p / (1.0 - p))


class WinModel:
    """XGBoost win model with mirror augmentation and Platt calibration.

    Parameters
    ----------
    params:
        Overrides merged over the pinned defaults (see module docstring).
        May include ``early_stopping_rounds`` (used only when a validation
        set is passed to :meth:`fit`).
    mirror_augment:
        Double the training data with the mirrored orientation
        (negate ``_diff`` columns, flip labels) plus the ±1
        :data:`PERSP_COL` column, and predict via the exactly antisymmetric
        two-orientation average.
    calibrate:
        ``"platt"`` (fit on validation predictions if a validation set is
        given, else silently skipped) or ``None``.
    seed:
        Passed to ``random_state`` everywhere.  ``nthread``/``n_jobs`` is
        left at the xgboost default.
    """

    def __init__(
        self,
        params: dict | None = None,
        mirror_augment: bool = True,
        calibrate: str | None = "platt",
        seed: int = 0,
    ) -> None:
        if calibrate not in (None, "platt"):
            raise ValueError(f"calibrate must be None or 'platt', got {calibrate!r}")
        params = dict(params or {})
        self.seed = int(seed)
        self.mirror_augment = bool(mirror_augment)
        self.calibrate = calibrate
        self._user_set_n_estimators = "n_estimators" in params
        merged = {**_DEFAULT_PARAMS, **params}
        self.early_stopping_rounds = int(
            merged.pop("early_stopping_rounds", _DEFAULT_EARLY_STOPPING_ROUNDS)
        )
        self.params = merged

        # Fitted state.
        self.model_: xgb.XGBClassifier | None = None
        self.calibrator_: LogisticRegression | None = None
        self.calibrated_: bool = False
        self.feature_columns_: list[str] | None = None  # caller-facing columns
        self.fit_columns_: list[str] | None = None  # booster columns (+ persp)
        self.best_iteration_: int | None = None

    # ------------------------------------------------------------------ fit

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        X_val: pd.DataFrame | None = None,
        y_val: np.ndarray | None = None,
        X_cal: pd.DataFrame | None = None,
        y_cal: np.ndarray | None = None,
    ) -> "WinModel":
        """Fit the booster (early stopping iff a validation set is given).

        If ``X_cal``/``y_cal`` are given, the Platt calibrator is fit on that
        slice instead of on the early-stopping validation set — avoiding the
        mild optimism of calibrating on the same rows the stopping point was
        chosen on. Without them, the validation set is reused (documented
        bias risk, within-train only).
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if PERSP_COL in X.columns:
            raise ValueError(
                f"{PERSP_COL!r} is reserved for internal mirror augmentation; "
                "remove it from X"
            )
        y = np.asarray(y, dtype=int).ravel()
        if len(y) != len(X):
            raise ValueError("X and y length mismatch")

        self.feature_columns_ = list(X.columns)

        params = dict(self.params)
        has_val = X_val is not None and y_val is not None
        if not has_val and not self._user_set_n_estimators:
            # No early stopping possible -> reduced fixed budget (see module
            # docstring).
            params["n_estimators"] = _NO_VAL_N_ESTIMATORS

        self.model_ = xgb.XGBClassifier(
            **params,
            random_state=self.seed,
            early_stopping_rounds=self.early_stopping_rounds if has_val else None,
        )

        X_fit, y_fit = self._augment(X, y)
        fit_kwargs: dict = {"verbose": False}
        if has_val:
            y_val_arr = np.asarray(y_val, dtype=int).ravel()
            Xv = self._check_columns(X_val)
            Xv_fit, yv_fit = self._augment(Xv, y_val_arr)
            fit_kwargs["eval_set"] = [(Xv_fit, yv_fit)]
        self.fit_columns_ = list(X_fit.columns)

        self.model_.fit(X_fit, y_fit, **fit_kwargs)
        self.best_iteration_ = getattr(self.model_, "best_iteration", None)

        # ---- calibration (leakage rule: never fit on training rows) ----
        self.calibrator_ = None
        self.calibrated_ = False
        has_cal = X_cal is not None and y_cal is not None
        if self.calibrate == "platt" and (has_cal or has_val):
            X_c, y_c = (X_cal, y_cal) if has_cal else (X_val, y_val)
            p_val = self._raw_predict(self._check_columns(X_c))
            yv = np.asarray(y_c, dtype=int).ravel()
            if len(np.unique(yv)) == 2:
                cal = LogisticRegression(
                    # The intercept is load-bearing: it lets calibration
                    # correct any residual base-rate (blue-side) offset.
                    fit_intercept=True,
                    C=1e6,
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=self.seed,
                )
                cal.fit(_logit(p_val).reshape(-1, 1), yv)
                self.calibrator_ = cal
                self.calibrated_ = True
        return self

    def _augment(
        self, X: pd.DataFrame, y: np.ndarray
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Mirror-augmented (frame, labels) — or passthrough when disabled."""
        X0 = X.reset_index(drop=True)
        if not self.mirror_augment:
            return X0, y
        Xm = _mirror_frame(X0)
        X0 = X0.copy()
        X0[PERSP_COL] = 1.0
        Xm[PERSP_COL] = -1.0
        Xa = pd.concat([X0, Xm], ignore_index=True)
        ya = np.concatenate([y, 1 - y])
        return Xa, ya

    # -------------------------------------------------------------- predict

    def _check_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """Enforce the fitted feature-column contract.

        Same set, different order -> reorder by name.  Missing or unexpected
        columns -> raise.
        """
        assert self.feature_columns_ is not None
        want = self.feature_columns_
        got = list(X.columns)
        if got == want:
            return X
        missing = [c for c in want if c not in X.columns]
        extra = [c for c in got if c not in want]
        if missing or extra:
            raise ValueError(
                f"feature columns do not match fit-time columns; "
                f"missing={missing}, unexpected={extra}"
            )
        return X[want]  # same set, wrong order -> reorder by name

    def _raw_predict(self, X: pd.DataFrame) -> np.ndarray:
        """Uncalibrated P(blue win) via the two training representations.

        Rows are blue-perspective by convention.  Each physical game has
        exactly two training representations: (X, persp=+1) with label y and
        (mirror(X), persp=-1) with label 1-y.  Prediction averages those two
        views of the SAME game: ``p = 0.5 * (f(X, +1) + 1 - f(mirror(X), -1))``.

        The perspective column stays ACTIVE (not marginalized): it carries the
        real blue-side advantage (~52-53% blue win rate in pro play).  As a
        consequence ``p(team-swap) != 1 - p(X)`` in general — swapping the
        teams is a *different physical game* in which the other team enjoys
        the blue-side edge; the difference ``p(X) + p(swap(X)) - 1`` is
        (twice) the learned blue bump, exactly as it should be.
        """
        assert self.model_ is not None and self.fit_columns_ is not None
        if not self.mirror_augment:
            return np.asarray(
                self.model_.predict_proba(X[self.fit_columns_])[:, 1], dtype=float
            )
        n = len(X)
        X0 = X.reset_index(drop=True)
        Xm = _mirror_frame(X0)
        X0 = X0.copy()
        X0[PERSP_COL] = 1.0
        Xm[PERSP_COL] = -1.0
        stacked = pd.concat(
            [X0[self.fit_columns_], Xm[self.fit_columns_]], ignore_index=True
        )
        p = np.asarray(self.model_.predict_proba(stacked)[:, 1], dtype=float)
        return 0.5 * (p[0:n] + (1.0 - p[n : 2 * n]))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """1-D array of P(blue win) for each row of ``X``."""
        if self.model_ is None:
            raise RuntimeError("WinModel is not fitted; call fit() first")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        X = self._check_columns(X)
        p = self._raw_predict(X)
        if self.calibrated_ and self.calibrator_ is not None:
            # Applied AFTER the two-orientation average; no intercept, so the
            # exact 1-p antisymmetry is preserved.
            p = np.asarray(
                self.calibrator_.predict_proba(_logit(p).reshape(-1, 1))[:, 1],
                dtype=float,
            )
        return p

    # ---------------------------------------------------------- persistence

    def save(self, path: str | Path) -> None:
        """Pickle the whole wrapper (booster + calibrator + column order)."""
        if self.model_ is None:
            raise RuntimeError("cannot save an unfitted WinModel")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "WinModel":
        """Load a wrapper saved by :meth:`save`."""
        obj = joblib.load(Path(path))
        if not isinstance(obj, cls):
            raise TypeError(f"{path} does not contain a {cls.__name__}")
        return obj


class EloLogisticBaseline:
    """Logistic regression on the rating diffs — the beat-this bar.

    Uses whichever of ``f_elo_diff`` / ``f_bt_theta_diff`` exist in the
    training frame (raises if neither does).  NaNs are median-imputed inside
    an sklearn Pipeline.  No mirroring is needed: these features are already
    antisymmetric, and ``fit_intercept=True`` absorbs the blue-side
    advantage.  Same ``predict_proba`` contract: 1-D P(blue win).
    """

    CANDIDATE_COLUMNS = ("f_elo_diff", "f_bt_theta_diff")

    def __init__(self, seed: int = 0) -> None:
        self.seed = int(seed)
        self.columns_: list[str] | None = None
        self.pipeline_: Pipeline | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        X_val: pd.DataFrame | None = None,
        y_val: np.ndarray | None = None,
    ) -> "EloLogisticBaseline":
        """Fit on the available rating-diff columns (validation args ignored)."""
        cols = [c for c in self.CANDIDATE_COLUMNS if c in X.columns]
        if not cols:
            raise ValueError(
                f"X has none of the baseline columns {self.CANDIDATE_COLUMNS}"
            )
        self.columns_ = cols
        self.pipeline_ = Pipeline(
            [
                (
                    "impute",
                    SimpleImputer(strategy="median", keep_empty_features=True),
                ),
                (
                    "logit",
                    LogisticRegression(
                        fit_intercept=True,
                        max_iter=1000,
                        random_state=self.seed,
                    ),
                ),
            ]
        )
        self.pipeline_.fit(X[cols].to_numpy(dtype=float), np.asarray(y, dtype=int))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """1-D array of P(blue win)."""
        if self.pipeline_ is None or self.columns_ is None:
            raise RuntimeError("EloLogisticBaseline is not fitted; call fit() first")
        missing = [c for c in self.columns_ if c not in X.columns]
        if missing:
            raise ValueError(f"missing baseline columns at predict time: {missing}")
        proba = self.pipeline_.predict_proba(X[self.columns_].to_numpy(dtype=float))
        return np.asarray(proba[:, 1], dtype=float)
