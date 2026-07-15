"""Leaguepedia (lol.fandom.com) Cargo bridge -> canonical game-table rows.

Covers pro games missing from the primary Oracle's Elixir dataset (e.g. the
2026 season) by querying the free, key-less Cargo API table
``ScoreboardGames`` and reshaping the rows into the canonical one-row-per-game
table of docs/CONTRACTS.md section 1, so the feature builder can compute
up-to-date Elo / Bradley-Terry / win-rate state.

SIDE ASSUMPTION (important, encoded in ``datacompleteness='leaguepedia'``)
--------------------------------------------------------------------------
``ScoreboardGames`` does not directly expose blue/red side; we map
``Team1 -> blue_team`` and ``Team2 -> red_team``. Leaguepedia's own
``ScoreboardTeams`` table (which has a ``Side`` field) confirms Team1 == the
blue-side team on every game spot-checked live (2026 LCK / LEC / LPL / MSI
samples, 12/12 agreement), so the assumption is verified-by-sampling rather
than guaranteed-by-schema. Rows sourced here carry
``datacompleteness == "leaguepedia"`` so downstream consumers can identify
(and if ever needed, exclude or re-derive) side-sensitive features for these
games.

Other known limitations of this source vs Oracle's Elixir:

* ``split`` is NaN and ``playoffs`` is 0 (not cheaply knowable from
  ``OverviewPage`` without a tournament-table join);
* per-side stat columns are NaN except ``blue_kills``/``red_kills`` when the
  API exposes ``Team1Kills``/``Team2Kills``;
* ``blue_players``/``red_players`` are ``""`` (no player rows fetched);
* ``league`` is the first path segment of ``OverviewPage``
  (``"LCK/2026 Season/Rounds 1-2" -> "LCK"``) — deterministic, but the names
  do not always match Oracle's Elixir league codes.

The API is rate-limited: requests are paced (``pace_s`` seconds between
calls) and retried with exponential backoff on 429/5xx/"ratelimited". When
``api.php?action=cargoquery`` keeps rate-limiting, the fetcher falls back to
the ``Special:CargoExport`` endpoint (same Cargo tables, JSON output, much
laxer limits — keys come back with spaces instead of underscores and typed
values instead of strings; both are normalized here).
"""

from __future__ import annotations

import difflib
import html
import logging
import re
import time
from datetime import date as _date
from datetime import datetime

import pandas as pd
import requests

from lolpred.data.loader import TEAM_STAT_COLS

logger = logging.getLogger(__name__)

__all__ = ["fetch_scoreboard_games", "to_canonical", "merge_with_canonical"]

API_URL = "https://lol.fandom.com/api.php"
EXPORT_URL = "https://lol.fandom.com/wiki/Special:CargoExport"
USER_AGENT = (
    "lolpred/0.1 (LoL win-prediction research; mcortese1406@gmail.com) "
    "python-requests"
)

#: Cargo fields that must exist in ScoreboardGames.
BASE_FIELDS: list[str] = [
    "Team1", "Team2", "Winner", "DateTime_UTC", "Patch", "OverviewPage",
    "Team1Score", "Team2Score", "Gamelength_Number", "GameId", "MatchId",
    "N_GameInMatch",
]
#: Fields we take if present but drop without a fight if the API errors.
OPTIONAL_FIELDS: list[str] = ["Team1Kills", "Team2Kills"]

#: datacompleteness marker for rows from this source (also flags the
#: Team1==blue side assumption documented in the module docstring).
DATACOMPLETENESS = "leaguepedia"

_PAGE_LIMIT = 500  # Cargo's anonymous per-request maximum


def _fmt_dt(value: str | _date | datetime | pd.Timestamp, end: bool) -> str:
    """Format a date-ish value as a Cargo DateTime_UTC bound (inclusive)."""
    ts = pd.Timestamp(value)
    if end and ts == ts.normalize():
        ts = ts + pd.Timedelta(hours=23, minutes=59, seconds=59)
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def _cargo_params(fields: list[str], where: str, offset: int) -> dict:
    # Alias every field to itself: cargoquery would otherwise return keys
    # with underscores replaced by spaces ("DateTime UTC").
    return {
        "action": "cargoquery",
        "format": "json",
        "tables": "ScoreboardGames",
        "fields": ",".join(f"{f}={f}" for f in fields),
        "where": where,
        "order_by": "DateTime_UTC,GameId",  # stable ordering for paging
        "limit": _PAGE_LIMIT,
        "offset": offset,
    }


class _CargoFieldError(RuntimeError):
    """The API rejected the query for a schema reason (e.g. unknown field)."""


class _Transient(RuntimeError):
    """A retryable failure (rate limit, 5xx, network hiccup)."""


def _query_api(session: requests.Session, params: dict) -> list[dict]:
    """One ``api.php?action=cargoquery`` request -> rows (or raise)."""
    resp = session.get(API_URL, params=params,
                       headers={"User-Agent": USER_AGENT}, timeout=30)
    if resp.status_code == 429 or resp.status_code >= 500:
        raise _Transient(f"HTTP {resp.status_code}")
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        code = data["error"].get("code", "")
        info = data["error"].get("info", "")
        if code == "ratelimited":
            raise _Transient("ratelimited")
        raise _CargoFieldError(f"cargoquery error {code}: {info}")
    return [row["title"] for row in data.get("cargoquery", [])]


def _query_export(session: requests.Session, params: dict) -> list[dict]:
    """Same query via ``Special:CargoExport`` (laxer rate limits).

    Output keys use spaces instead of underscores and carry an extra
    ``__precision`` companion per datetime field; both are normalized away.
    Values are typed JSON (int/float/str) — downstream coerces.
    """
    export_params = {k: v for k, v in params.items()
                     if k not in ("action", "format")}
    export_params["format"] = "json"
    resp = session.get(EXPORT_URL, params=export_params,
                       headers={"User-Agent": USER_AGENT}, timeout=30)
    if resp.status_code == 429 or resp.status_code >= 500:
        raise _Transient(f"HTTP {resp.status_code}")
    resp.raise_for_status()
    try:
        data = resp.json()
    except ValueError as exc:  # HTML error/challenge page
        raise _Transient(f"CargoExport non-JSON response: {exc}") from exc
    return [
        {k.replace(" ", "_"): v for k, v in row.items()
         if not k.endswith("__precision")}
        for row in data
    ]


#: After this many rate-limited api.php attempts, switch to CargoExport.
_API_ATTEMPTS = 2


def _request_page(
    session: requests.Session,
    params: dict,
    pace_s: float,
    max_retries: int = 8,
) -> list[dict]:
    """One paced Cargo page with backoff; api.php first, CargoExport after.

    ``api.php?action=cargoquery`` rate-limits anonymous clients aggressively;
    after ``_API_ATTEMPTS`` transient failures the remaining retries go to
    ``Special:CargoExport``, which serves the same table data.
    """
    delay = max(pace_s, 2.0)
    last = "no attempt made"
    for attempt in range(max_retries):
        if attempt:
            logger.info("leaguepedia retry %d/%d after %s (sleep %.0fs)",
                        attempt, max_retries - 1, last, delay)
            time.sleep(delay)
            delay = min(delay * 2, 120.0)
        query = _query_api if attempt < _API_ATTEMPTS else _query_export
        try:
            return query(session, params)
        except _Transient as exc:
            last = str(exc)
        except requests.RequestException as exc:
            last = f"request error: {exc}"
    raise RuntimeError(f"leaguepedia: retries exhausted ({last})")


def fetch_scoreboard_games(
    date_from: str | _date | datetime | pd.Timestamp,
    date_to: str | _date | datetime | pd.Timestamp,
    session: requests.Session | None = None,
    pace_s: float = 2.0,
    max_pages: int = 100,
) -> pd.DataFrame:
    """Fetch raw ScoreboardGames rows for ``[date_from, date_to]`` (UTC).

    Pages through the Cargo API ``limit=500`` at a time, sleeping ``pace_s``
    seconds between calls and backing off exponentially on rate limits /
    server errors (falling back to ``Special:CargoExport`` when api.php keeps
    rate-limiting). Returns the raw rows, one DataFrame row per Cargo row
    (columns are whichever fields the API accepted; values may be str or
    typed depending on the endpoint — :func:`to_canonical` coerces).
    """
    session = session or requests.Session()
    where = (
        f'DateTime_UTC >= "{_fmt_dt(date_from, end=False)}"'
        f' AND DateTime_UTC <= "{_fmt_dt(date_to, end=True)}"'
    )
    fields = BASE_FIELDS + OPTIONAL_FIELDS
    rows: list[dict] = []
    for page in range(max_pages):
        if page:
            time.sleep(pace_s)
        params = _cargo_params(fields, where, offset=page * _PAGE_LIMIT)
        try:
            batch = _request_page(session, params, pace_s)
        except _CargoFieldError as exc:
            # Only fight the schema once: drop the optional fields and retry.
            if page == 0 and fields is not BASE_FIELDS:
                logger.warning(
                    "leaguepedia: query with optional fields failed (%s); "
                    "retrying with base fields only", exc)
                fields = BASE_FIELDS
                batch = _request_page(
                    session, _cargo_params(fields, where, offset=0), pace_s)
            else:
                raise
        rows.extend(batch)
        logger.info("leaguepedia page %d: %d rows (total %d)",
                    page, len(batch), len(rows))
        if len(batch) < _PAGE_LIMIT:
            break
    else:
        logger.warning(
            "leaguepedia: hit max_pages=%d with full pages still coming; "
            "results are likely truncated", max_pages)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# raw -> canonical
# ---------------------------------------------------------------------------

def _league_from_overview(page: str) -> str:
    """First path segment of an OverviewPage: 'LCK/2026 Season/...' -> 'LCK'."""
    return str(page).split("/", 1)[0].strip()


def to_canonical(raw: pd.DataFrame) -> pd.DataFrame:
    """Reshape raw ScoreboardGames rows into the canonical game table.

    Output is schema-compatible with ``lolpred.data.loader.load_games``
    (CONTRACTS.md section 1). Rows missing team, winner or date are dropped;
    the result is deduplicated on ``gameid`` and sorted by (date, gameid).
    """
    if raw.empty:
        raw = pd.DataFrame(columns=BASE_FIELDS)
    df = raw.copy()
    for col in BASE_FIELDS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df.astype("string")

    out = pd.DataFrame(index=df.index)
    out["date"] = pd.to_datetime(df["DateTime_UTC"], errors="coerce")
    out["league"] = df["OverviewPage"].fillna("").map(_league_from_overview)
    out["split"] = pd.Series(pd.NA, index=df.index, dtype=object)
    out["playoffs"] = 0
    out["patch"] = df["Patch"]
    out["game_in_series"] = (
        pd.to_numeric(df["N_GameInMatch"], errors="coerce")
        .fillna(1).astype(int)
    )
    out["datacompleteness"] = DATACOMPLETENESS
    # Cargo occasionally returns HTML-escaped names ("A &amp; B").
    out["blue_team"] = df["Team1"].map(html.unescape, na_action="ignore")
    out["red_team"] = df["Team2"].map(html.unescape, na_action="ignore")
    # Winner may arrive as "1"/"2" (api.php strings) or 1/2 (CargoExport).
    winner = pd.to_numeric(df["Winner"], errors="coerce")
    out["_winner"] = winner
    out["blue_win"] = (winner == 1).fillna(False).astype(int)
    out["gamelength"] = (
        pd.to_numeric(df["Gamelength_Number"], errors="coerce") * 60.0
    ).astype(float)  # plain float64: nullable Float64's pd.NA -> np.nan

    # gameid: prefer GameId; fall back to MatchId + game number.
    game_id = df["GameId"].str.strip()
    match_id = df["MatchId"].str.strip()
    fallback = match_id + "_" + out["game_in_series"].astype(str)
    base_id = game_id.where(game_id.notna() & (game_id != ""), fallback)
    out["gameid"] = "LP_" + base_id.astype(str)

    # Per-side stats: NaN floats, except kills when the API exposes them.
    for side, tnum in (("blue", "1"), ("red", "2")):
        for stat in TEAM_STAT_COLS:
            src = f"Team{tnum}Kills" if stat == "kills" else None
            if src is not None and src in df.columns:
                out[f"{side}_{stat}"] = pd.to_numeric(
                    df[src], errors="coerce").astype(float)
            else:
                out[f"{side}_{stat}"] = pd.Series(
                    float("nan"), index=df.index, dtype=float)

    out["blue_players"] = ""
    out["red_players"] = ""

    # --- drop rows with missing team / winner / date ---
    def _bad_team(s: pd.Series) -> pd.Series:
        name = s.astype("string").str.strip()
        return name.isna() | (name == "")

    keep = (
        out["date"].notna()
        & ~_bad_team(out["blue_team"])
        & ~_bad_team(out["red_team"])
        & out["_winner"].isin([1.0, 2.0])
    )
    dropped = int((~keep).sum())
    if dropped:
        logger.info("to_canonical: dropped %d rows missing team/winner/date",
                    dropped)
    out = out[keep].drop(columns="_winner")

    out["year"] = out["date"].dt.year.astype(int) if len(out) else pd.Series(
        dtype=int)
    out["blue_team"] = out["blue_team"].astype(str)
    out["red_team"] = out["red_team"].astype(str)
    out["series_id"] = _series_ids(out)

    cols = (
        ["gameid", "date", "league", "year", "split", "playoffs", "patch",
         "game_in_series", "series_id", "datacompleteness", "blue_team",
         "red_team", "blue_win", "gamelength"]
        + [f"blue_{c}" for c in TEAM_STAT_COLS]
        + [f"red_{c}" for c in TEAM_STAT_COLS]
        + ["blue_players", "red_players"]
    )
    out = out[cols].drop_duplicates(subset="gameid", keep="first")
    out = out.sort_values(["date", "gameid"]).reset_index(drop=True)
    return out


def _series_ids(df: pd.DataFrame) -> list[str]:
    """Contract series id: ``date.date()|league|sorted(team pair)``."""
    return [
        f"{d.date()}|{lg}|{'|'.join(sorted((b, r)))}"
        for d, lg, b, r in zip(
            df["date"], df["league"], df["blue_team"], df["red_team"]
        )
    ]


# ---------------------------------------------------------------------------
# merge with the Oracle's Elixir canonical table
# ---------------------------------------------------------------------------

_ORG_SUFFIXES = (" esports", " esport", " gaming")


def _normalize_team(name: str) -> str:
    """Deterministic normalization for team-name matching.

    Lowercase, strip punctuation, collapse whitespace, drop a trailing org
    suffix ("esports"/"esport"/"gaming") when something remains.
    """
    s = re.sub(r"[^a-z0-9 ]+", "", str(name).lower())
    s = re.sub(r"\s+", " ", s).strip()
    for suf in _ORG_SUFFIXES:
        if s.endswith(suf) and len(s) > len(suf):
            s = s[: -len(suf)].strip()
            break
    return s


def build_team_alias_map(
    oe_teams: set[str],
    lp_teams: set[str],
    fuzzy_cutoff: float = 0.9,
    oe_leagues: dict[str, set[str]] | None = None,
    lp_leagues: dict[str, set[str]] | None = None,
) -> tuple[dict[str, str], list[str]]:
    """Map LP team names -> OE team names for roster continuity.

    Exact normalized match wins; otherwise the best difflib match on the
    normalized names with ratio >= ``fuzzy_cutoff`` (logged); otherwise the
    LP name is kept as-is (a genuinely new team). When both league maps are
    given, a *fuzzy* (non-exact) rename additionally requires the two teams
    to share at least one league — near-name amateur teams in minor leagues
    otherwise get merged into major orgs (a live example: 'FONatic' in
    'IDL Kings Lendas' fuzzy-matches 'Fnatic' at 0.92). Returns
    ``(alias_map, new_team_names)`` — names already identical to an OE name
    are excluded from both.
    """
    oe_by_norm: dict[str, str] = {}
    for name in sorted(oe_teams):  # sorted -> deterministic collision winner
        norm = _normalize_team(name)
        if norm:  # fully non-Latin names normalize to "" — never match on it
            oe_by_norm.setdefault(norm, name)
    norm_keys = sorted(oe_by_norm)

    alias: dict[str, str] = {}
    new_teams: list[str] = []
    for lp_name in sorted(lp_teams):
        if lp_name in oe_teams:
            continue  # already continuous
        norm = _normalize_team(lp_name)
        if not norm:
            new_teams.append(lp_name)
            logger.info("new team (non-normalizable name): %r", lp_name)
            continue
        if norm in oe_by_norm:
            alias[lp_name] = oe_by_norm[norm]
            logger.info("team alias (exact-normalized): %r -> %r",
                        lp_name, alias[lp_name])
            continue
        close = difflib.get_close_matches(norm, norm_keys, n=1,
                                          cutoff=fuzzy_cutoff)
        if close:
            candidate = oe_by_norm[close[0]]
            if (oe_leagues is not None and lp_leagues is not None
                    and not (oe_leagues.get(candidate, set())
                             & lp_leagues.get(lp_name, set()))):
                new_teams.append(lp_name)
                logger.info(
                    "new team (fuzzy match %r rejected: no shared league): %r",
                    candidate, lp_name)
                continue
            alias[lp_name] = candidate
            logger.info("team alias (fuzzy %.2f): %r -> %r",
                        difflib.SequenceMatcher(None, norm, close[0]).ratio(),
                        lp_name, alias[lp_name])
        else:
            new_teams.append(lp_name)
            logger.info("new team (no OE match): %r", lp_name)
    return alias, new_teams


def _team_leagues(df: pd.DataFrame) -> dict[str, set[str]]:
    """team name -> set of leagues the team appears in (either side)."""
    out: dict[str, set[str]] = {}
    for col in ("blue_team", "red_team"):
        for team, leagues in df.groupby(col)["league"].agg(set).items():
            out.setdefault(team, set()).update(leagues)
    return out


def merge_with_canonical(
    oe_games: pd.DataFrame, lp_games: pd.DataFrame,
) -> pd.DataFrame:
    """Append Leaguepedia rows to the OE canonical table without overlap.

    Only LP rows strictly AFTER ``oe_games.date.max()`` are kept (simpler and
    safer than trying to dedupe overlapping coverage row-by-row). LP team
    names are renamed to their OE equivalents via
    :func:`build_team_alias_map` (exact normalized match, then difflib >= 0.9,
    else kept as new teams) and ``series_id`` is rebuilt for renamed rows.

    The merged frame carries ``attrs['lp_appended']``, ``attrs['lp_renames']``
    (dict LP name -> OE name) and ``attrs['lp_new_teams']`` for reporting.
    """
    cutoff = oe_games["date"].max()
    lp = lp_games[lp_games["date"] > cutoff].copy()
    logger.info("merge: OE max date %s; keeping %d/%d LP rows after cutoff",
                cutoff, len(lp), len(lp_games))

    oe_recent = oe_games[oe_games["year"] >= 2025]
    oe_teams = set(oe_recent["blue_team"]) | set(oe_recent["red_team"])
    lp_teams = set(lp["blue_team"]) | set(lp["red_team"])
    alias, new_teams = build_team_alias_map(
        oe_teams, lp_teams,
        oe_leagues=_team_leagues(oe_recent), lp_leagues=_team_leagues(lp))

    if alias:
        lp["blue_team"] = lp["blue_team"].map(lambda t: alias.get(t, t))
        lp["red_team"] = lp["red_team"].map(lambda t: alias.get(t, t))
        lp["series_id"] = _series_ids(lp)

    merged = pd.concat([oe_games, lp], ignore_index=True)
    merged = merged.drop_duplicates(subset="gameid", keep="first")
    merged = merged.sort_values(["date", "gameid"]).reset_index(drop=True)
    merged.attrs["lp_appended"] = int(len(lp))
    merged.attrs["lp_renames"] = dict(alias)
    merged.attrs["lp_new_teams"] = list(new_teams)
    return merged
