"""Rating streams: online Elo and lazily-refit Bradley-Terry.

Contract section 2 of ``docs/CONTRACTS.md``. Both classes are chronological
state machines driven by the feature builder:

* :class:`EloStream` — classic Elo with margin-of-victory multiplier,
  blue-side offset, per-format K, and split regression.
* :class:`BradleyTerry` — time-decayed L2 (ridge) logistic regression on a
  sparse +/-1 team design with a blue-side intercept, refit lazily inside
  :meth:`BradleyTerry.pregame`.

Row access convention
---------------------
``EloStream.update`` and ``BradleyTerry.observe`` consume one canonical-game
row. The expected shape is a pandas NamedTuple from
``DataFrame.itertuples(index=False)`` (attribute access), but plain dicts and
anything supporting ``row[name]`` work too. Attribute names used:

* always: ``blue_team``, ``red_team``, ``blue_win``, and (BT only) ``date``
* Elo, optional: ``game_in_series`` (default 1), ``best_of`` (optional,
  overrides format inference), ``blue_golddiffat15``, ``blue_kills``,
  ``red_kills``

Both streams are pure functions of the games observed so far — no file I/O,
no randomness.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression

_MISSING = object()

#: se reported by BradleyTerry.pregame when either team is unseen / no fit.
BT_UNSEEN_SE = 2.0

#: Minimum history size before BradleyTerry will fit at all.
BT_MIN_GAMES = 50


def _field(row: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from a row via attribute access, then item access."""
    value = getattr(row, name, _MISSING)
    if value is _MISSING:
        try:
            value = row[name]
        except (KeyError, IndexError, TypeError):
            return default
    return value


def _is_missing(value: Any) -> bool:
    """True for None / NaN / NaT scalars."""
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):  # non-scalar oddities
        return False


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


class EloStream:
    """Online Elo ratings over the canonical game stream.

    Parameters
    ----------
    k_bo1, k_series:
        K-factor for Bo1 games vs games inside a series. Which one applies
        is decided per game: an optional ``best_of`` attribute on the row
        wins (``best_of == 1`` -> ``k_bo1``); otherwise we infer from
        ``game_in_series`` — ``game_in_series > 1`` -> series K, else Bo1 K
        (a game 1 of a Bo5 is indistinguishable from a Bo1 without
        ``best_of``; this is the documented v1 tradeoff).
    scale:
        Logistic scale of the expected-score curve (Elo standard 400).
    mov:
        Enable the margin-of-victory multiplier (see :meth:`mov_multiplier`).
    side_offset_init:
        Elo points added to the *blue* team's effective rating when
        computing the expected score (constant in v1).
    split_regress:
        Fraction of the distance to the global mean (``init``) that every
        team's rating is regressed on a period change (:meth:`new_period`).
    init:
        Rating for unseen teams and the regression target.
    """

    def __init__(
        self,
        k_bo1: float = 32.0,
        k_series: float = 24.0,
        scale: float = 400.0,
        mov: bool = True,
        side_offset_init: float = 25.0,
        split_regress: float = 0.25,
        init: float = 1500.0,
    ) -> None:
        self.k_bo1 = float(k_bo1)
        self.k_series = float(k_series)
        self.scale = float(scale)
        self.mov = bool(mov)
        self.side_offset = float(side_offset_init)
        self.split_regress = float(split_regress)
        self.init = float(init)
        self._ratings: dict[str, float] = {}
        self._games: dict[str, int] = {}
        self._period_key: str | None = None

    # -- read-only views ---------------------------------------------------
    def rating(self, team: str) -> float:
        """Current rating of ``team`` (``init`` if unseen)."""
        return self._ratings.get(team, self.init)

    def games_played(self, team: str) -> int:
        """Number of games ``update`` has consumed for ``team``."""
        return self._games.get(team, 0)

    def expected_blue(self, blue: str, red: str) -> float:
        """Expected score of the blue team, including the blue-side offset."""
        diff = (self.rating(blue) + self.side_offset) - self.rating(red)
        return 1.0 / (1.0 + 10.0 ** (-diff / self.scale))

    def mov_multiplier(
        self,
        gold_diff: float | None,
        kill_diff: float | None,
        winner_elo_advantage: float,
    ) -> float:
        """Margin-of-victory K multiplier, clamped to [0.5, 2.0].

        Base margin term: ``ln(1 + |blue_golddiffat15| / 1500)`` when the
        gold-diff column is present, else ``ln(1 + |kill diff| / 5)`` (the
        canonical table has no final-gold column; kill diff is rescaled so a
        typical decisive game lands in a comparable range), else 1.0. The
        base is clamped to [0.5, 2.0], then a FiveThirtyEight-style
        autocorrelation damp ``2.2 / (0.001 * winner_elo_advantage + 2.2)``
        shrinks updates for wins by the already-higher-rated side; the final
        product is re-clamped to [0.5, 2.0].
        """
        if not self.mov:
            return 1.0
        if not _is_missing(gold_diff):
            base = math.log1p(abs(float(gold_diff)) / 1500.0)
        elif not _is_missing(kill_diff):
            base = math.log1p(abs(float(kill_diff)) / 5.0)
        else:
            return 1.0
        base = min(2.0, max(0.5, base))
        denom = max(0.1, 0.001 * winner_elo_advantage + 2.2)
        return min(2.0, max(0.5, base * (2.2 / denom)))

    # -- contract API ------------------------------------------------------
    def pregame(self, blue: str, red: str) -> dict[str, float]:
        """Pre-game feature dict. Never mutates state."""
        elo_blue = self.rating(blue)
        elo_red = self.rating(red)
        return {
            "elo_diff": elo_blue - elo_red,
            "elo_blue": elo_blue,
            "elo_red": elo_red,
            "elo_games_blue": float(self.games_played(blue)),
            "elo_games_red": float(self.games_played(red)),
        }

    def update(self, game: Any) -> None:
        """Consume one canonical-game row (see module docstring for fields)."""
        blue = _field(game, "blue_team")
        red = _field(game, "red_team")
        blue_win = int(_field(game, "blue_win"))

        k = self._k_for(game)
        expected = self.expected_blue(blue, red)

        mult = 1.0
        if self.mov:
            gold_diff = _field(game, "blue_golddiffat15")
            bk, rk = _field(game, "blue_kills"), _field(game, "red_kills")
            kill_diff = (
                None
                if _is_missing(bk) or _is_missing(rk)
                else float(bk) - float(rk)
            )
            eff_blue = self.rating(blue) + self.side_offset
            eff_red = self.rating(red)
            winner_adv = eff_blue - eff_red if blue_win else eff_red - eff_blue
            mult = self.mov_multiplier(gold_diff, kill_diff, winner_adv)

        delta = k * mult * (blue_win - expected)
        self._ratings[blue] = self.rating(blue) + delta
        self._ratings[red] = self.rating(red) - delta
        self._games[blue] = self.games_played(blue) + 1
        self._games[red] = self.games_played(red) + 1

    def new_period(self, year_split_key: str) -> None:
        """On a new year/split key, regress all teams toward the mean.

        Idempotent per key: repeated calls with the current key are no-ops.
        The very first key only labels the period (nothing to regress from).
        """
        if year_split_key == self._period_key:
            return
        if self._period_key is not None:
            for team, r in self._ratings.items():
                self._ratings[team] = r + self.split_regress * (self.init - r)
        self._period_key = year_split_key

    # -- internals ----------------------------------------------------------
    def _k_for(self, game: Any) -> float:
        best_of = _field(game, "best_of")
        if not _is_missing(best_of):
            return self.k_bo1 if int(best_of) == 1 else self.k_series
        gis = _field(game, "game_in_series", 1)
        gis = 1 if _is_missing(gis) else int(gis)
        return self.k_series if gis > 1 else self.k_bo1


class BradleyTerry:
    """Time-decayed ridge-logistic Bradley-Terry ratings.

    ``observe`` appends games to an internal history; ``pregame`` lazily
    refits when ``date >= last_fit_date + refit_every_days`` (or on the
    first call once at least :data:`BT_MIN_GAMES` games are buffered).

    Fit: sklearn ``LogisticRegression`` (lbfgs, ``C = 1/l2``,
    ``fit_intercept=True``) on a scipy.sparse design with one +/-1 column
    per team seen so far; the intercept is the blue-side advantage
    ``bt_beta_side``. Note sklearn also L2-penalizes the intercept slightly
    less cleanly than the team columns would like (it is *not* penalized by
    sklearn, while our hand-computed covariance below adds ``l2`` on its
    diagonal too) — an accepted, documented approximation. Sample weights
    decay exponentially: ``0.5 ** (days_before_fit_date / half_life_days)``.

    Coefficient covariance is computed directly as
    ``Sigma = inv(X'WX + l2*I)`` with ``W = diag(w_i * p_i * (1 - p_i))``
    from the fitted model, ``X`` including the intercept column.
    ``bt_se_diff`` is the se of the full linear predictor
    ``theta_blue - theta_red + beta_side`` (combination vector ``c`` with
    +1/-1 on the team columns and +1 on the intercept: ``sqrt(c' Sigma c)``).

    The reported probability is logit-normal shrunk:
    ``p = sigmoid(mu / sqrt(1 + pi * s**2 / 8))``.

    Deterministic: no randomness anywhere.
    """

    def __init__(
        self,
        half_life_days: float = 60.0,
        l2: float = 2.0,
        refit_every_days: float = 7,
    ) -> None:
        self.half_life_days = float(half_life_days)
        self.l2 = float(l2)
        self.refit_every_days = float(refit_every_days)
        self._history: list[tuple[pd.Timestamp, str, str, int]] = []
        self._last_fit_date: pd.Timestamp | None = None
        self._team_index: dict[str, int] = {}
        self._theta: np.ndarray | None = None
        self._beta_side: float = 0.0
        self._sigma: np.ndarray | None = None
        #: number of refits performed (exposed for tests/diagnostics)
        self.n_fits: int = 0

    # -- contract API ------------------------------------------------------
    def observe(self, game: Any) -> None:
        """Append one canonical-game row to the history buffer."""
        self._history.append(
            (
                pd.Timestamp(_field(game, "date")),
                str(_field(game, "blue_team")),
                str(_field(game, "red_team")),
                int(_field(game, "blue_win")),
            )
        )

    def pregame(self, blue: str, red: str, date) -> dict[str, float]:
        """Pre-game BT features; may lazily refit (the only state change)."""
        date = pd.Timestamp(date)
        if self._should_refit(date):
            self._fit(date)

        if (
            self._theta is None
            or blue not in self._team_index
            or red not in self._team_index
        ):
            return self._unseen(blue, red)

        b = self._team_index[blue]
        r = self._team_index[red]
        theta_diff = float(self._theta[b] - self._theta[r])
        mu = theta_diff + self._beta_side

        n = len(self._team_index)
        c = np.zeros(n + 1)
        c[b] += 1.0
        c[r] -= 1.0
        c[n] = 1.0  # intercept column
        var = float(c @ self._sigma @ c)
        se = math.sqrt(max(var, 0.0))

        return {
            "bt_theta_diff": theta_diff,
            "bt_se_diff": se,
            "bt_prob_blue": self._shrunk_prob(mu, se),
            "bt_beta_side": self._beta_side,
        }

    # -- internals ----------------------------------------------------------
    @staticmethod
    def _shrunk_prob(mu: float, se: float) -> float:
        return _sigmoid(mu / math.sqrt(1.0 + math.pi * se * se / 8.0))

    def _unseen(self, blue: str, red: str) -> dict[str, float]:
        prob = (
            self._shrunk_prob(self._beta_side, BT_UNSEEN_SE)
            if self._theta is not None
            else 0.5
        )
        return {
            "bt_theta_diff": 0.0,
            "bt_se_diff": BT_UNSEEN_SE,
            "bt_prob_blue": prob,
            "bt_beta_side": self._beta_side,
        }

    def _should_refit(self, date: pd.Timestamp) -> bool:
        if len(self._history) < BT_MIN_GAMES:
            return False
        if self._last_fit_date is None:
            return True
        return date >= self._last_fit_date + pd.Timedelta(
            days=self.refit_every_days
        )

    def _fit(self, fit_date: pd.Timestamp) -> None:
        history = self._history
        teams = sorted({t for _, b, r, _ in history for t in (b, r)})
        index = {t: i for i, t in enumerate(teams)}
        n_games, n_teams = len(history), len(teams)

        rows = np.empty(2 * n_games, dtype=np.int32)
        cols = np.empty(2 * n_games, dtype=np.int32)
        vals = np.empty(2 * n_games, dtype=np.float64)
        y = np.empty(n_games, dtype=np.int8)
        w = np.empty(n_games, dtype=np.float64)
        day = pd.Timedelta(days=1)
        for i, (gdate, b, r, win) in enumerate(history):
            rows[2 * i] = rows[2 * i + 1] = i
            cols[2 * i] = index[b]
            cols[2 * i + 1] = index[r]
            vals[2 * i] = 1.0
            vals[2 * i + 1] = -1.0
            y[i] = win
            days_before = max(0.0, (fit_date - gdate) / day)
            w[i] = 0.5 ** (days_before / self.half_life_days)

        X = sp.csr_matrix(
            (vals, (rows, cols)), shape=(n_games, n_teams), dtype=np.float64
        )
        model = LogisticRegression(
            C=1.0 / self.l2,  # sklearn default penalty is l2
            solver="lbfgs",
            fit_intercept=True,
            max_iter=1000,
            tol=1e-8,
        )
        model.fit(X, y, sample_weight=w)
        theta = model.coef_[0].astype(np.float64)
        beta_side = float(model.intercept_[0])

        # Covariance: Sigma = inv(X'WX + l2*I) with the intercept col in X.
        p = model.predict_proba(X)[:, 1]
        omega = w * p * (1.0 - p)
        X_full = sp.hstack(
            [X, np.ones((n_games, 1), dtype=np.float64)], format="csr"
        )
        H = (X_full.T @ sp.diags(omega) @ X_full).toarray()
        H += self.l2 * np.eye(n_teams + 1)
        sigma = np.linalg.inv(H)

        self._team_index = index
        self._theta = theta
        self._beta_side = beta_side
        self._sigma = sigma
        self._last_fit_date = fit_date
        self.n_fits += 1
