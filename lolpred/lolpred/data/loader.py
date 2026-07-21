"""Load raw Oracle's Elixir match CSVs into the canonical one-row-per-game table.

Contract (docs/CONTRACTS.md section 1): one row per game, blue/red oriented,
sorted by (date, gameid), deduplicated on gameid. Strategies downstream only
ever see this table; all raw-data quirks (duplicate gameids across files,
"unknown team" rows, year-0 / dateless games, missing @15 columns in older
years) are handled here.
"""

from __future__ import annotations

import glob as _glob
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Raw OE columns the loader needs. Older years may lack some -> added as NaN.
RAW_COLUMNS: list[str] = [
    "gameid", "datacompleteness", "league", "year", "split", "playoffs",
    "date", "game", "patch", "participantid", "side", "position",
    "playername", "teamname", "result", "kills", "deaths", "assists",
    "firstblood", "firstdragon", "firstbaron", "firsttower", "dragons",
    "barons", "towers", "gamelength", "goldat15", "xpat15", "csat15",
    "golddiffat15", "dpm",
]
_RAW_SET = frozenset(RAW_COLUMNS)

#: Per-side stat columns emitted with ``blue_`` / ``red_`` prefixes.
TEAM_STAT_COLS: list[str] = [
    "kills", "deaths", "assists", "firstblood", "firstdragon", "firstbaron",
    "firsttower", "dragons", "barons", "towers", "goldat15", "xpat15",
    "csat15", "golddiffat15", "dpm",
]

#: Ordering of the datacompleteness levels for ``min_datacompleteness``.
_COMPLETENESS_RANK = {"any": 0, "partial": 1, "complete": 2}

_PLAYER_IDS = frozenset(str(i) for i in range(1, 11))
_TEAM_IDS = frozenset({"100", "200"})


def _resolve_paths(paths: str | Path | Sequence[str | Path]) -> list[Path]:
    """Expand the accepted path forms into a concrete list of CSV files.

    Accepts a single path, a sequence of paths, a directory (every ``*.csv``
    under it) or a glob pattern.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    out: list[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            out.extend(sorted(p.glob("*.csv")))
        elif any(ch in str(p) for ch in "*?["):
            out.extend(sorted(Path(m) for m in _glob.glob(str(p))))
        else:
            out.append(p)
    if not out:
        raise FileNotFoundError(f"no CSV files found for {paths!r}")
    missing = [p for p in out if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"raw CSV(s) not found: {missing}")
    return out


def _read_raw(path: Path) -> pd.DataFrame:
    """Read one raw OE CSV, restricted to the needed columns, everything str."""
    df = pd.read_csv(
        path,
        usecols=lambda c: c in _RAW_SET,
        dtype=str,
        low_memory=False,
    )
    for col in RAW_COLUMNS:
        if col not in df.columns:
            df[col] = pd.Series(pd.NA, index=df.index, dtype="string")
    return df[RAW_COLUMNS]


def load_games(
    paths: str | Path | Sequence[str | Path],
    min_datacompleteness: str = "any",
    verbose: bool = False,
) -> pd.DataFrame:
    """Load raw Oracle's Elixir CSV(s) into the canonical game table.

    Parameters
    ----------
    paths:
        One or many raw OE CSV paths; a directory loads every ``*.csv`` under
        it; glob patterns are expanded. When a gameid appears in several files
        (overlapping-year sources) the first file's rows win.
    min_datacompleteness:
        ``"any"`` (default) keeps everything, ``"partial"`` keeps
        partial+complete, ``"complete"`` keeps only fully-scraped games.
    verbose:
        Log dropped-game counts by reason at INFO (always logged at DEBUG).

    Returns
    -------
    pd.DataFrame
        One row per game with the contract's meta / per-side stat / roster
        columns, sorted by (date, gameid), deduplicated on gameid, index
        reset. Drop counts by reason are attached as ``df.attrs["drop_counts"]``.
    """
    if min_datacompleteness not in _COMPLETENESS_RANK:
        raise ValueError(
            f"min_datacompleteness must be one of {sorted(_COMPLETENESS_RANK)},"
            f" got {min_datacompleteness!r}"
        )
    files = _resolve_paths(paths)
    frames = []
    for i, f in enumerate(files):
        df = _read_raw(f)
        df["_src"] = i
        frames.append(df)
        logger.debug("read %s: %d rows", f, len(df))
    raw = pd.concat(frames, ignore_index=True)
    raw = raw[raw["gameid"].notna()]

    drops: dict[str, int] = {}

    # --- dedupe on gameid across files: first file containing a game wins ---
    first_src = raw.groupby("gameid")["_src"].transform("min")
    dup_mask = raw["_src"] != first_src
    drops["duplicate_gameid"] = int(raw.loc[dup_mask, "gameid"].nunique())
    raw = raw[~dup_mask]

    # --- datacompleteness gate (row-level; constant within a game) ---
    min_rank = _COMPLETENESS_RANK[min_datacompleteness]
    if min_rank > 0:
        rank = raw["datacompleteness"].map(_COMPLETENESS_RANK).fillna(0)
        before = raw["gameid"].nunique()
        raw = raw[rank >= min_rank]
        drops["datacompleteness"] = int(before - raw["gameid"].nunique())
    else:
        drops["datacompleteness"] = 0

    pid = raw["participantid"].astype("string").str.replace(
        r"\.0$", "", regex=True
    )
    side = raw["side"].astype("string")
    position = raw["position"].astype("string")

    is_team = pid.isin(_TEAM_IDS) | (position == "team")
    teams = raw[is_team & side.isin(["Blue", "Red"])].copy()
    teams["_side"] = side[teams.index]
    players = raw[~is_team & pid.isin(_PLAYER_IDS)].copy()
    players["_side"] = side[players.index]

    # --- games must have exactly one Blue and one Red team row ---
    counts = (
        teams.groupby(["gameid", "_side"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    for c in ("Blue", "Red"):
        if c not in counts.columns:
            counts[c] = 0
    valid_ids = counts.index[(counts["Blue"] == 1) & (counts["Red"] == 1)]
    all_ids = raw["gameid"].nunique()
    drops["not_one_blue_one_red"] = int(all_ids - len(valid_ids))

    teams = teams[teams["gameid"].isin(valid_ids)]
    blue = teams[teams["_side"] == "Blue"].set_index("gameid")
    red = teams[teams["_side"] == "Red"].set_index("gameid").reindex(blue.index)

    out = pd.DataFrame(index=blue.index)
    out["gameid"] = out.index.astype(str)
    out["date"] = pd.to_datetime(blue["date"], errors="coerce")
    out["league"] = blue["league"].astype(str)
    out["_year"] = pd.to_numeric(blue["year"], errors="coerce")
    out["split"] = blue["split"]
    out["playoffs"] = (
        pd.to_numeric(blue["playoffs"], errors="coerce").fillna(0).astype(int)
    )
    out["patch"] = blue["patch"]
    out["game_in_series"] = (
        pd.to_numeric(blue["game"], errors="coerce").fillna(1).astype(int)
    )
    out["datacompleteness"] = blue["datacompleteness"].astype(str)
    out["blue_team"] = blue["teamname"]
    out["red_team"] = red["teamname"]
    out["gamelength"] = pd.to_numeric(blue["gamelength"], errors="coerce")
    out["_result_blue"] = pd.to_numeric(blue["result"], errors="coerce")
    out["_result_red"] = pd.to_numeric(red["result"], errors="coerce")
    for c in TEAM_STAT_COLS:
        out[f"blue_{c}"] = pd.to_numeric(blue[c], errors="coerce")
        out[f"red_{c}"] = pd.to_numeric(red[c], errors="coerce")

    def _bad_team(s: pd.Series) -> pd.Series:
        name = s.astype("string").str.strip()
        return name.isna() | (name == "") | (name.str.lower() == "unknown team")

    def _drop(mask: pd.Series, reason: str) -> None:
        nonlocal out
        drops[reason] = int(mask.sum())
        out = out[~mask]

    _drop(_bad_team(out["blue_team"]) | _bad_team(out["red_team"]),
          "missing_or_unknown_team")
    _drop(out["_result_blue"].isna() | out["_result_red"].isna(),
          "missing_result")
    _drop(out["_result_blue"] + out["_result_red"] != 1,
          "inconsistent_result")
    _drop(out["date"].isna() | out["_year"].isna() | (out["_year"] == 0),
          "missing_date_or_year")

    out["year"] = out["_year"].astype(int)
    out["blue_win"] = out["_result_blue"].astype(int)
    out["blue_team"] = out["blue_team"].astype(str)
    out["red_team"] = out["red_team"].astype(str)

    # --- rosters: sorted "|"-joined starter names per side ---
    players = players[
        players["gameid"].isin(out.index) & players["playername"].notna()
    ]
    if len(players):
        joined = players.groupby(["gameid", "_side"], observed=True)[
            "playername"
        ].agg(lambda s: "|".join(sorted(s.astype(str))))
    else:
        joined = pd.Series(dtype=object)
    for s, col in (("Blue", "blue_players"), ("Red", "red_players")):
        m = (
            joined.xs(s, level="_side")
            if len(joined) and s in joined.index.get_level_values("_side")
            else pd.Series(dtype=object)
        )
        out[col] = out.index.to_series().map(m).fillna("").astype(str)

    out["series_id"] = [
        f"{d.date()}|{lg}|{'|'.join(sorted((b, r)))}"
        for d, lg, b, r in zip(
            out["date"], out["league"], out["blue_team"], out["red_team"]
        )
    ]

    cols = (
        ["gameid", "date", "league", "year", "split", "playoffs", "patch",
         "game_in_series", "series_id", "datacompleteness", "blue_team",
         "red_team", "blue_win", "gamelength"]
        + [f"blue_{c}" for c in TEAM_STAT_COLS]
        + [f"red_{c}" for c in TEAM_STAT_COLS]
        + ["blue_players", "red_players"]
    )
    out = out[cols].reset_index(drop=True)
    out = out.drop_duplicates(subset="gameid", keep="first")
    out = out.sort_values(["date", "gameid"]).reset_index(drop=True)

    out.attrs["drop_counts"] = drops
    log = logger.info if verbose else logger.debug
    log("load_games: %d files -> %d games; dropped by reason: %s",
        len(files), len(out), drops)
    return out


def filter_games(
    df: pd.DataFrame,
    leagues: Sequence[str] | None = None,
    min_year: int | None = None,
    complete_only: bool = False,
) -> pd.DataFrame:
    """Filter a canonical game table by league / year / datacompleteness.

    ``complete_only`` keeps only games with ``datacompleteness == "complete"``.
    Returns a copy with the index reset; ``df.attrs`` are preserved.
    """
    mask = pd.Series(True, index=df.index)
    if leagues is not None:
        mask &= df["league"].isin(list(leagues))
    if min_year is not None:
        mask &= df["year"] >= int(min_year)
    if complete_only:
        mask &= df["datacompleteness"] == "complete"
    out = df[mask].reset_index(drop=True)
    out.attrs = dict(df.attrs)
    return out
