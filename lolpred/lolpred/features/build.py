"""Chronological matchup feature builder (contract section 3).

THE CARDINAL RULE — strict-date visibility
------------------------------------------
The builder makes a single chronological pass over the canonical game table,
grouped by calendar date. For each date ``d``:

1. features for **all** games on ``d`` are computed from state as of the end
   of ``d - 1`` (phase 1);
2. only then is state updated with all games of ``d`` (phase 2).

Within-day games (including Bo-series siblings) are therefore mutually
invisible. This is structurally guaranteed by two separate loops over the
day's rows in :meth:`_Builder.flush_day` — never interleave them.

Output
------
One row per input game (sorted by ``(date, gameid)``), meta columns preserved,
features prefixed ``f_``. Naming convention (load-bearing for the model's
mirror augmentation, contract section 4):

* every team-comparative feature ends in ``_diff`` and is **exactly
  antisymmetric**: swapping blue/red orientation negates it;
* symmetric context features never end in ``_diff`` and are invariant under
  orientation swap (``f_hist_games_blue`` / ``f_hist_games_red`` are the
  documented exception: they *exchange* under swap — "symmetric-ish"
  context the model and bet gate need individually).

Documented v1 choices / deviations
----------------------------------
* ``f_bt_edge_diff`` / ``f_bt_se``: :meth:`BradleyTerry.pregame` is called in
  both orientations and the results are (anti)symmetrized — the edge is the
  antisymmetric part of the shrunk win probability, the se the symmetric
  part. The blue-side intercept is carried separately by the symmetric
  ``f_bt_beta_side``.
* ``f_sidewr_diff``: the naive "blue's as-blue WR minus red's as-red WR" is
  *not* antisymmetric (the two teams' deques differ), which would break the
  mirror convention. We emit the antisymmetric side-preference difference
  ``s(blue) - s(red)`` with ``s(t) = shrunkWR_as_blue(t) - shrunkWR_as_red(t)``,
  plus the symmetric pair term ``f_sidewr_pair = (s(blue) + s(red)) / 2``
  (how much this particular pairing amplifies the blue side).
* Shrinkage: win rates (win10/win30/ewm, per-side WRs) are shrunk toward 0.5
  and gd15 means toward 0 with weight ``n / (n + shrink_k)``.
* Cold start: shrunk win rates emit from a team's first observed game;
  other rolling means (gd15, first-objective rates, K/D) require at least
  ``_MIN_ROLLING`` observations, else NaN. A team with zero prior games gets
  NaN for every rolling feature. ``f_hist_games_*`` always emit.
* ``EloStream.new_period`` is keyed globally on the day's first game's
  ``f"{year}|{split}"`` (leagues whose splits are out of phase share one
  regression clock — accepted v1 simplification).
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from lolpred.features.ratings import BradleyTerry, EloStream

__all__ = ["FeatureConfig", "build_matchup_features", "META_COLUMNS", "FEATURE_COLUMNS"]

logger = logging.getLogger(__name__)

#: Rest-days cap.
_REST_CAP = 14.0
#: League blue-side base-rate window (games).
_LEAGUE_WINDOW = 200
#: Roster history window (player-sets remembered per team).
_ROSTER_WINDOW = 10
#: Minimum observations for un-shrunk rolling means (gd15, objective rates, K/D).
_MIN_ROLLING = 3
#: Progress-log cadence (games).
_LOG_EVERY = 10_000

#: Meta columns preserved in the output (contract section 3).
META_COLUMNS: list[str] = [
    "gameid",
    "date",
    "league",
    "year",
    "split",
    "playoffs",
    "patch",
    "series_id",
    "game_in_series",
    "blue_team",
    "red_team",
    "blue_win",
]

#: Emitted feature columns, in output order.
FEATURE_COLUMNS: list[str] = [
    # rating streams
    "f_elo_diff",
    "f_elo_games_diff",
    "f_bt_theta_diff",
    "f_bt_edge_diff",
    "f_bt_se",
    "f_bt_beta_side",
    # rolling win rates
    "f_win10_diff",
    "f_win30_diff",
    "f_winewm_diff",
    # rolling gold diff at 15
    "f_gd15_mean10_diff",
    "f_gd15_mean30_diff",
    # first-objective rates (window = big window)
    "f_fb_rate30_diff",
    "f_fdragon_rate30_diff",
    "f_fbaron_rate30_diff",
    "f_ftower_rate30_diff",
    # side-specific
    "f_sidewr_diff",
    "f_sidewr_pair",
    # combat
    "f_kd_log30_diff",
    # schedule / context diffs
    "f_rest_days_diff",
    "f_games_on_patch_diff",
    "f_split_games_diff",
    "f_career_games_diff",
    "f_roster_cont_diff",
    # symmetric-ish context
    "f_hist_games_blue",
    "f_hist_games_red",
    "f_league_bluewr",
    "f_playoffs",
    "f_game_in_series",
    "f_patch_age_days",
]

_ALL_COLUMNS: list[str] = META_COLUMNS + FEATURE_COLUMNS

_NAN = float("nan")


@dataclass
class FeatureConfig:
    """Knobs for :func:`build_matchup_features`."""

    windows: tuple[int, int] = (10, 30)
    ewm_half_life_games: float = 15.0
    #: shrink rolling means toward the league mean with weight n / (n + k)
    shrink_k: float = 8.0
    elo_kwargs: dict = field(default_factory=dict)
    bt_kwargs: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# small numeric helpers
# --------------------------------------------------------------------------

def _num(value: Any) -> float | None:
    """``value`` as a finite-or-None float (None for missing/NaN/non-numeric)."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def _mean(values) -> tuple[float, int]:
    n = len(values)
    return (sum(values) / n if n else 0.0), n


def _tail(dq: deque, w: int) -> list:
    """Last ``w`` items of ``dq`` (deque maxlen == big window)."""
    if len(dq) <= w:
        return list(dq)
    return list(dq)[-w:]


def _players_set(raw: Any) -> frozenset[str]:
    if not isinstance(raw, str) or not raw:
        return frozenset()
    return frozenset(p for p in raw.split("|") if p)


class _TeamState:
    """Strictly-past rolling state for one team."""

    __slots__ = (
        "results", "ewm", "n",
        "gd15", "fb", "fd", "fbaron", "ftower",
        "blue_res", "red_res", "kills", "deaths",
        "last_played", "patch_key", "patch_games",
        "split_key", "split_games", "rosters",
    )

    def __init__(self, w_big: int) -> None:
        self.results: deque = deque(maxlen=w_big)
        self.ewm: float = 0.5          # exponential win rate, updated per game
        self.n: int = 0                # career games observed
        self.gd15: deque = deque(maxlen=w_big)
        self.fb: deque = deque(maxlen=w_big)
        self.fd: deque = deque(maxlen=w_big)
        self.fbaron: deque = deque(maxlen=w_big)
        self.ftower: deque = deque(maxlen=w_big)
        self.blue_res: deque = deque(maxlen=w_big)
        self.red_res: deque = deque(maxlen=w_big)
        self.kills: deque = deque(maxlen=w_big)
        self.deaths: deque = deque(maxlen=w_big)
        self.last_played = None        # datetime.date
        self.patch_key: str | None = None
        self.patch_games: int = 0
        self.split_key: str | None = None
        self.split_games: int = 0
        self.rosters: deque = deque(maxlen=_ROSTER_WINDOW)


class _Builder:
    """Holds all chronological state; one instance per build call."""

    def __init__(self, cfg: FeatureConfig) -> None:
        self.cfg = cfg
        self.w_small, self.w_big = int(cfg.windows[0]), int(cfg.windows[1])
        self.ewm_w = 1.0 - 0.5 ** (1.0 / float(cfg.ewm_half_life_games))
        self.k = float(cfg.shrink_k)
        self.elo = EloStream(**cfg.elo_kwargs)
        self.bt = BradleyTerry(**cfg.bt_kwargs)
        self.teams: dict[str, _TeamState] = {}
        self.league_blue: dict[str, deque] = {}
        self.patch_first: dict[str, Any] = {}  # patch -> first date seen
        self.n_done = 0

    # -- shrinkage helpers --------------------------------------------------
    def _shrunk_rate(self, values, n: int) -> float:
        """Win-rate mean shrunk toward 0.5 with n/(n+k); NaN when n == 0."""
        if n == 0:
            return _NAN
        m, _ = _mean(values)
        return 0.5 + (m - 0.5) * n / (n + self.k)

    def _shrunk_mean0(self, values, n: int) -> float:
        """Mean shrunk toward 0 with n/(n+k); NaN when n < _MIN_ROLLING."""
        if n < _MIN_ROLLING:
            return _NAN
        m, _ = _mean(values)
        return m * n / (n + self.k)

    @staticmethod
    def _rate_min(values, n: int) -> float:
        """Plain rolling mean, NaN when n < _MIN_ROLLING."""
        if n < _MIN_ROLLING:
            return _NAN
        m, _ = _mean(values)
        return m

    # -- per-team feature primitives ----------------------------------------
    def _team_feats(self, team: str, day, row: Any, side: str) -> dict[str, float]:
        ts = self.teams.get(team)
        patch_key = str(getattr(row, "patch", None))
        split_key = f"{getattr(row, 'year', None)}|{getattr(row, 'split', None)}"
        if ts is None or ts.n == 0:
            hist = 0.0 if ts is None else float(ts.n)
            return {
                "win10": _NAN, "win30": _NAN, "winewm": _NAN,
                "gd10": _NAN, "gd30": _NAN,
                "fb": _NAN, "fd": _NAN, "fbaron": _NAN, "ftower": _NAN,
                "side_pref": _NAN, "kd": _NAN, "rest": _NAN,
                "patch_games": 0.0, "split_games": 0.0, "career": hist,
                "roster_cont": 1.0, "hist": hist,
            }

        res = ts.results
        win10 = self._shrunk_rate(_tail(res, self.w_small), min(len(res), self.w_small))
        win30 = self._shrunk_rate(res, len(res))
        winewm = 0.5 + (ts.ewm - 0.5) * ts.n / (ts.n + self.k)

        gd = ts.gd15
        gd10 = self._shrunk_mean0(_tail(gd, self.w_small), min(len(gd), self.w_small))
        gd30 = self._shrunk_mean0(gd, len(gd))

        # side preference s(t) = shrunkWR_as_blue - shrunkWR_as_red; a side
        # never played contributes its shrinkage target 0.5 exactly (n = 0
        # puts zero weight on the empirical mean).
        def _side(dq: deque) -> float:
            if not dq:
                return 0.5
            m, n = _mean(dq)
            return 0.5 + (m - 0.5) * n / (n + self.k)

        side_pref = _side(ts.blue_res) - _side(ts.red_res)

        nk = len(ts.kills)
        if nk < _MIN_ROLLING:
            kd = _NAN
        else:
            kd = math.log((sum(ts.kills) + 1.0) / (sum(ts.deaths) + 1.0))

        rest = _NAN
        if ts.last_played is not None:
            rest = min(float((day - ts.last_played).days), _REST_CAP)

        patch_games = float(ts.patch_games) if ts.patch_key == patch_key else 0.0
        split_games = float(ts.split_games) if ts.split_key == split_key else 0.0

        cur = _players_set(getattr(row, f"{side}_players", None))
        if not cur or not ts.rosters:
            roster_cont = 1.0
        else:
            roster_cont = sum(
                len(cur & past) / len(cur | past) if (cur | past) else 1.0
                for past in ts.rosters
            ) / len(ts.rosters)

        return {
            "win10": win10, "win30": win30, "winewm": winewm,
            "gd10": gd10, "gd30": gd30,
            "fb": self._rate_min(ts.fb, len(ts.fb)),
            "fd": self._rate_min(ts.fd, len(ts.fd)),
            "fbaron": self._rate_min(ts.fbaron, len(ts.fbaron)),
            "ftower": self._rate_min(ts.ftower, len(ts.ftower)),
            "side_pref": side_pref, "kd": kd, "rest": rest,
            "patch_games": patch_games, "split_games": split_games,
            "career": float(ts.n), "roster_cont": roster_cont,
            "hist": float(ts.n),
        }

    # -- phase 1: features for one game (state read-only) -------------------
    def features(self, day, row: Any) -> dict[str, Any]:
        blue, red = row.blue_team, row.red_team

        e = self.elo.pregame(blue, red)
        # Both orientations -> exactly (anti)symmetric BT features.
        b_br = self.bt.pregame(blue, red, row.date)
        b_rb = self.bt.pregame(red, blue, row.date)
        bt_theta = (b_br["bt_theta_diff"] - b_rb["bt_theta_diff"]) / 2.0
        # == mean of the two orientations' P(blue) minus 0.5, written as a
        # plain difference so orientation swap negates it *bitwise*.
        bt_edge = (b_br["bt_prob_blue"] - b_rb["bt_prob_blue"]) / 2.0
        bt_se = (b_br["bt_se_diff"] + b_rb["bt_se_diff"]) / 2.0

        tb = self._team_feats(blue, day, row, "blue")
        tr = self._team_feats(red, day, row, "red")

        lg = self.league_blue.get(getattr(row, "league", None))
        league_bluewr = (sum(lg) / len(lg)) if lg else _NAN

        patch_key = str(getattr(row, "patch", None))
        first = self.patch_first.get(patch_key)
        patch_age = float((day - first).days) if first is not None else 0.0

        gis = _num(getattr(row, "game_in_series", None))
        playoffs = _num(getattr(row, "playoffs", None))

        rec = {m: getattr(row, m, None) for m in META_COLUMNS}
        rec.update(
            f_elo_diff=e["elo_diff"],
            f_elo_games_diff=e["elo_games_blue"] - e["elo_games_red"],
            f_bt_theta_diff=bt_theta,
            f_bt_edge_diff=bt_edge,
            f_bt_se=bt_se,
            f_bt_beta_side=b_br["bt_beta_side"],
            f_win10_diff=tb["win10"] - tr["win10"],
            f_win30_diff=tb["win30"] - tr["win30"],
            f_winewm_diff=tb["winewm"] - tr["winewm"],
            f_gd15_mean10_diff=tb["gd10"] - tr["gd10"],
            f_gd15_mean30_diff=tb["gd30"] - tr["gd30"],
            f_fb_rate30_diff=tb["fb"] - tr["fb"],
            f_fdragon_rate30_diff=tb["fd"] - tr["fd"],
            f_fbaron_rate30_diff=tb["fbaron"] - tr["fbaron"],
            f_ftower_rate30_diff=tb["ftower"] - tr["ftower"],
            f_sidewr_diff=tb["side_pref"] - tr["side_pref"],
            f_sidewr_pair=(tb["side_pref"] + tr["side_pref"]) / 2.0,
            f_kd_log30_diff=tb["kd"] - tr["kd"],
            f_rest_days_diff=tb["rest"] - tr["rest"],
            f_games_on_patch_diff=tb["patch_games"] - tr["patch_games"],
            f_split_games_diff=tb["split_games"] - tr["split_games"],
            f_career_games_diff=tb["career"] - tr["career"],
            f_roster_cont_diff=tb["roster_cont"] - tr["roster_cont"],
            f_hist_games_blue=tb["hist"],
            f_hist_games_red=tr["hist"],
            f_league_bluewr=league_bluewr,
            f_playoffs=playoffs if playoffs is not None else _NAN,
            f_game_in_series=gis if gis is not None else _NAN,
            f_patch_age_days=patch_age,
        )
        return rec

    # -- phase 2: state update with one game ---------------------------------
    def update(self, day, row: Any) -> None:
        self.elo.update(row)
        self.bt.observe(row)

        blue_win = int(row.blue_win)
        patch_key = str(getattr(row, "patch", None))
        split_key = f"{getattr(row, 'year', None)}|{getattr(row, 'split', None)}"

        league = getattr(row, "league", None)
        lg = self.league_blue.get(league)
        if lg is None:
            lg = self.league_blue[league] = deque(maxlen=_LEAGUE_WINDOW)
        lg.append(blue_win)
        self.patch_first.setdefault(patch_key, day)

        for side, team in (("blue", row.blue_team), ("red", row.red_team)):
            ts = self.teams.get(team)
            if ts is None:
                ts = self.teams[team] = _TeamState(self.w_big)
            result = blue_win if side == "blue" else 1 - blue_win

            ts.results.append(result)
            ts.ewm = self.ewm_w * result + (1.0 - self.ewm_w) * ts.ewm
            ts.n += 1
            (ts.blue_res if side == "blue" else ts.red_res).append(result)

            # gold diff at 15 from this team's perspective; prefer the
            # side-prefixed column, fall back to the negated opposite one.
            gd = _num(getattr(row, f"{side}_golddiffat15", None))
            if gd is None:
                other = _num(
                    getattr(row, f"{'red' if side == 'blue' else 'blue'}_golddiffat15", None)
                )
                gd = -other if other is not None else None
            if gd is not None:
                ts.gd15.append(gd)

            for attr, dq in (
                ("firstblood", ts.fb),
                ("firstdragon", ts.fd),
                ("firstbaron", ts.fbaron),
                ("firsttower", ts.ftower),
            ):
                v = _num(getattr(row, f"{side}_{attr}", None))
                if v is not None:
                    dq.append(v)

            k = _num(getattr(row, f"{side}_kills", None))
            d = _num(getattr(row, f"{side}_deaths", None))
            if k is not None and d is not None:
                ts.kills.append(k)
                ts.deaths.append(d)

            ts.last_played = day
            if ts.patch_key != patch_key:
                ts.patch_key, ts.patch_games = patch_key, 0
            ts.patch_games += 1
            if ts.split_key != split_key:
                ts.split_key, ts.split_games = split_key, 0
            ts.split_games += 1

            players = _players_set(getattr(row, f"{side}_players", None))
            if players:
                ts.rosters.append(players)

    # -- the cardinal-rule choke point ---------------------------------------
    def flush_day(self, day, rows: list, records: list) -> None:
        """Phase 1 (features, read-only) then phase 2 (updates) for one date.

        The two loops below are the structural guarantee of strict-date
        visibility: every game of the day is featurized before any game of
        the day mutates state.
        """
        first = rows[0]
        self.elo.new_period(f"{getattr(first, 'year', None)}|{getattr(first, 'split', None)}")

        for row in rows:               # phase 1 — state as of end of day-1
            records.append(self.features(day, row))
        for row in rows:               # phase 2 — only now absorb the day
            self.update(day, row)

        before = self.n_done
        self.n_done += len(rows)
        if self.n_done // _LOG_EVERY > before // _LOG_EVERY:
            logger.info("feature builder: %d games processed (through %s)",
                        self.n_done, day)


def build_matchup_features(
    games: pd.DataFrame, cfg: FeatureConfig | None = None
) -> pd.DataFrame:
    """Build strictly-past matchup features for the canonical game table.

    Args:
        games: canonical game table (contract section 1). Re-sorted by
            ``(date, gameid)`` defensively; ``date`` must be Timestamp-like.
        cfg: feature knobs; defaults to :class:`FeatureConfig()`.

    Returns:
        One row per input game with :data:`META_COLUMNS` preserved and
        :data:`FEATURE_COLUMNS` (all ``f_``-prefixed) appended, sorted by
        ``(date, gameid)``.
    """
    cfg = cfg or FeatureConfig()
    df = games.sort_values(["date", "gameid"], kind="mergesort").reset_index(drop=True)

    builder = _Builder(cfg)
    records: list[dict] = []
    day_key = None
    day_rows: list = []
    for row in df.itertuples(index=False):
        key = row.date.date()
        if day_key is not None and key != day_key:
            builder.flush_day(day_key, day_rows, records)
            day_rows = []
        day_key = key
        day_rows.append(row)
    if day_rows:
        builder.flush_day(day_key, day_rows, records)

    out = pd.DataFrame.from_records(records, columns=_ALL_COLUMNS)
    return out
