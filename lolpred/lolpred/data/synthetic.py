"""Synthetic canonical game table with known ground-truth latent strengths.

Produces the canonical game table described in docs/CONTRACTS.md section 1,
generated from a known latent-strength model so the full pipeline can be
validated end-to-end: does the feature builder / model recover the latent
strengths, and do any features leak the future?

Fully deterministic given ``seed`` (single ``np.random.default_rng`` stream,
consumed in a fixed chronological order).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

__all__ = ["generate_synthetic_games", "CANONICAL_COLUMNS"]

#: Canonical column order (contract section 1).
CANONICAL_COLUMNS: list[str] = [
    "gameid",
    "date",
    "league",
    "year",
    "split",
    "playoffs",
    "patch",
    "game_in_series",
    "series_id",
    "datacompleteness",
    "blue_team",
    "red_team",
    "blue_win",
    "gamelength",
    *[
        f"{side}_{stat}"
        for side in ("blue", "red")
        for stat in (
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
    ],
    "blue_players",
    "red_players",
]

#: Every this-many days, one day of cross-league ("international") play.
INTERNATIONAL_PERIOD_DAYS = 180
#: League label used for cross-league game days.
INTERNATIONAL_LEAGUE = "INTL"
#: Every this-many days, ~this fraction of teams swap one player.
ROSTER_CHURN_PERIOD_DAYS = 180
ROSTER_CHURN_FRAC = 0.20


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _team_prefix(league: str) -> str:
    """'SYNTH_A' -> 'SYNA' (first 3 chars of head + joined tail parts)."""
    parts = league.split("_")
    return (parts[0][:3] + "".join(parts[1:])).upper()


def generate_synthetic_games(
    n_teams: int = 20,
    n_days: int = 720,
    games_per_day: int = 4,
    seed: int = 0,
    strength_sd: float = 0.7,
    drift_sd: float = 0.02,
    blue_bonus: float = 0.10,
    leagues: tuple[str, ...] = ("SYNTH_A", "SYNTH_B"),
    return_truth: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a synthetic canonical game table (contract section 1).

    Teams are split evenly across ``leagues`` and play only within their
    league, except one cross-league day every ``INTERNATIONAL_PERIOD_DAYS``
    days.  Each team carries a latent strength ``theta ~ N(0, strength_sd)``
    that does a slow daily random walk (sd ``drift_sd``).  Each game day,
    ``games_per_day`` random disjoint pairs per league play a best-of-3
    series; per game, blue/red is assigned at random and
    ``P(blue wins) = sigmoid(theta_blue - theta_red + blue_bonus)``.

    Per-side stats are correlated with both the latent strength difference
    and the realized outcome, so downstream feature engineering has real
    signal to find.  Values are plausible-magnitude, not physically exact.

    Args:
        n_teams: total teams (split evenly across leagues).
        n_days: number of sequential calendar days starting 2020-01-01.
        games_per_day: series (matchups) per league per day, capped at
            ``teams_per_league // 2``.
        seed: RNG seed; output is fully deterministic given it.
        strength_sd: sd of the initial latent strengths.
        drift_sd: daily sd of the latent-strength random walk.
        blue_bonus: additive blue-side advantage on the logit scale.
        leagues: league names; teams are named ``f"{prefix}_T{i:02d}"``.
        return_truth: if True, also return a truth frame with columns
            ``(date, team, theta)`` — one row per team per day, the theta
            in force for that day's games.

    Returns:
        The canonical game DataFrame, or ``(games, truth)`` if
        ``return_truth`` is True.
    """
    if n_teams % len(leagues) != 0:
        raise ValueError("n_teams must divide evenly across leagues")
    per_league = n_teams // len(leagues)
    if per_league < 2:
        raise ValueError("need at least 2 teams per league")

    rng = np.random.default_rng(seed)

    # --- teams, latent strengths, rosters -------------------------------
    league_teams: dict[str, list[str]] = {
        lg: [f"{_team_prefix(lg)}_T{i + 1:02d}" for i in range(per_league)]
        for lg in leagues
    }
    teams: list[str] = [t for lg in leagues for t in league_teams[lg]]
    theta: dict[str, float] = {
        t: v for t, v in zip(teams, rng.normal(0.0, strength_sd, size=n_teams))
    }
    # 5 stable players per team; a per-team counter mints new names on churn.
    rosters: dict[str, list[str]] = {t: [f"{t}_P{k + 1}" for k in range(5)] for t in teams}
    next_player_no: dict[str, int] = {t: 6 for t in teams}

    start = pd.Timestamp("2020-01-01")
    rows: list[dict] = []
    truth_rows: list[dict] = []
    game_no = 0

    def play_series(day: pd.Timestamp, league: str, team_a: str, team_b: str) -> None:
        nonlocal game_no
        pair = sorted((team_a, team_b))
        series_id = f"{day.date()}|{league}|{pair[0]}|{pair[1]}"
        year = int(day.year)
        split = "Spring" if day.month <= 6 else "Summer"
        patch = f"{13 + year - 2020}.{1 + day.dayofyear // 14}"
        wins = {team_a: 0, team_b: 0}
        game_in_series = 0
        while max(wins.values()) < 2:
            game_in_series += 1
            game_no += 1
            # Random blue/red assignment per game.
            if rng.random() < 0.5:
                blue, red = team_a, team_b
            else:
                blue, red = team_b, team_a
            dt = theta[blue] - theta[red]
            p_blue = _sigmoid(dt + blue_bonus)
            blue_win = int(rng.random() < p_blue)
            wins[blue if blue_win else red] += 1
            sgn = 1.0 if blue_win else -1.0  # 2*(blue_win - 0.5)

            gamelength = float(np.clip(rng.normal(1900.0, 250.0), 1200.0, 3600.0))

            # Gold diff at 15: strength + outcome signal, antisymmetric.
            gd15 = float(rng.normal(1500.0 * dt + 800.0 * sgn, 1500.0))

            # Kills: winner higher, Poisson-ish around 12-16.
            lam_b = float(np.clip(13.0 + 2.5 * sgn + 1.5 * dt, 2.0, 40.0))
            lam_r = float(np.clip(13.0 - 2.5 * sgn - 1.5 * dt, 2.0, 40.0))
            blue_kills = float(rng.poisson(lam_b))
            red_kills = float(rng.poisson(lam_r))
            # Deaths = opponent kills; assists ~ 2.2x own kills.
            blue_deaths, red_deaths = red_kills, blue_kills
            blue_assists = float(rng.poisson(2.2 * blue_kills))
            red_assists = float(rng.poisson(2.2 * red_kills))

            # First objectives: Bernoulli, prob sigmoid of strength diff,
            # boosted for the eventual winner; sides are complementary.
            def first(w_dt: float, w_sgn: float) -> tuple[float, float]:
                b = float(rng.random() < _sigmoid(w_dt * dt + w_sgn * sgn))
                return b, 1.0 - b

            blue_fb, red_fb = first(0.8, 0.5)
            blue_fd, red_fd = first(0.6, 0.6)
            blue_fbaron, red_fbaron = first(1.0, 1.5)
            blue_ft, red_ft = first(0.9, 0.9)

            # Objective counts, winner-skewed.
            blue_dragons = float(rng.poisson(np.clip(2.5 + 1.0 * sgn + 0.5 * dt, 0.2, 6.0)))
            red_dragons = float(rng.poisson(np.clip(2.5 - 1.0 * sgn - 0.5 * dt, 0.2, 6.0)))
            blue_barons = float(rng.poisson(np.clip(0.8 + 0.5 * sgn + 0.2 * dt, 0.05, 3.0)))
            red_barons = float(rng.poisson(np.clip(0.8 - 0.5 * sgn - 0.2 * dt, 0.05, 3.0)))
            blue_towers = float(rng.poisson(np.clip(6.5 + 2.8 * sgn + 0.8 * dt, 1.0, 11.0)))
            red_towers = float(rng.poisson(np.clip(6.5 - 2.8 * sgn - 0.8 * dt, 1.0, 11.0)))

            # Gold/xp/cs at 15: plausible base + a share of the diff, with
            # goldat15 exactly consistent with golddiffat15.
            blue_gold15 = float(rng.normal(24000.0, 800.0)) + gd15 / 2.0
            red_gold15 = blue_gold15 - gd15
            blue_xp15 = float(rng.normal(29000.0, 900.0)) + 0.4 * gd15
            red_xp15 = float(rng.normal(29000.0, 900.0)) - 0.4 * gd15
            blue_cs15 = float(rng.normal(520.0, 25.0)) + gd15 / 50.0
            red_cs15 = float(rng.normal(520.0, 25.0)) - gd15 / 50.0

            blue_dpm = float(rng.normal(2400.0 + 250.0 * sgn + 120.0 * dt, 250.0))
            red_dpm = float(rng.normal(2400.0 - 250.0 * sgn - 120.0 * dt, 250.0))

            rows.append(
                {
                    "gameid": f"SYNTH_{game_no:07d}",
                    "date": day,
                    "league": league,
                    "year": year,
                    "split": split,
                    "playoffs": 0,
                    "patch": patch,
                    "game_in_series": game_in_series,
                    "series_id": series_id,
                    "datacompleteness": "complete",
                    "blue_team": blue,
                    "red_team": red,
                    "blue_win": blue_win,
                    "gamelength": gamelength,
                    "blue_kills": blue_kills,
                    "blue_deaths": blue_deaths,
                    "blue_assists": blue_assists,
                    "blue_firstblood": blue_fb,
                    "blue_firstdragon": blue_fd,
                    "blue_firstbaron": blue_fbaron,
                    "blue_firsttower": blue_ft,
                    "blue_dragons": blue_dragons,
                    "blue_barons": blue_barons,
                    "blue_towers": blue_towers,
                    "blue_goldat15": blue_gold15,
                    "blue_xpat15": blue_xp15,
                    "blue_csat15": blue_cs15,
                    "blue_golddiffat15": gd15,
                    "blue_dpm": blue_dpm,
                    "red_kills": red_kills,
                    "red_deaths": red_deaths,
                    "red_assists": red_assists,
                    "red_firstblood": red_fb,
                    "red_firstdragon": red_fd,
                    "red_firstbaron": red_fbaron,
                    "red_firsttower": red_ft,
                    "red_dragons": red_dragons,
                    "red_barons": red_barons,
                    "red_towers": red_towers,
                    "red_goldat15": red_gold15,
                    "red_xpat15": red_xp15,
                    "red_csat15": red_cs15,
                    "red_golddiffat15": -gd15,
                    "red_dpm": red_dpm,
                    "blue_players": "|".join(sorted(rosters[blue])),
                    "red_players": "|".join(sorted(rosters[red])),
                }
            )

    # --- day loop --------------------------------------------------------
    for d in range(n_days):
        day = start + pd.Timedelta(days=d)

        # Roster churn: every ROSTER_CHURN_PERIOD_DAYS, ~20% of teams swap
        # one player for a brand-new name.
        if d > 0 and d % ROSTER_CHURN_PERIOD_DAYS == 0:
            for t in teams:
                if rng.random() < ROSTER_CHURN_FRAC:
                    slot = int(rng.integers(0, 5))
                    rosters[t][slot] = f"{t}_P{next_player_no[t]}"
                    next_player_no[t] += 1

        # Record ground truth in force for today's games.
        for t in teams:
            truth_rows.append({"date": day, "team": t, "theta": theta[t]})

        international = (d + 1) % INTERNATIONAL_PERIOD_DAYS == 0
        if international and len(leagues) >= 2:
            # Cross-league day: pair shuffled rosters of two leagues.
            shuffled = [
                [league_teams[lg][i] for i in rng.permutation(per_league)]
                for lg in leagues
            ]
            n_series = min(games_per_day, per_league)
            for i in range(n_series):
                play_series(day, INTERNATIONAL_LEAGUE, shuffled[0][i], shuffled[1][i])
        else:
            for lg in leagues:
                order = [league_teams[lg][i] for i in rng.permutation(per_league)]
                n_series = min(games_per_day, per_league // 2)
                for i in range(n_series):
                    play_series(day, lg, order[2 * i], order[2 * i + 1])

        # Latent strengths drift after the day's games.
        drift = rng.normal(0.0, drift_sd, size=n_teams)
        for t, dv in zip(teams, drift):
            theta[t] += float(dv)

    df = pd.DataFrame(rows, columns=CANONICAL_COLUMNS)
    df = df.sort_values(["date", "gameid"], kind="mergesort").reset_index(drop=True)

    if return_truth:
        truth = pd.DataFrame(truth_rows, columns=["date", "team", "theta"])
        return df, truth
    return df
