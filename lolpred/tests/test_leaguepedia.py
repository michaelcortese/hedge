"""Tests for the Leaguepedia Cargo bridge (no network except LP_NET=1)."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from lolpred.data.leaguepedia import (
    build_team_alias_map,
    fetch_scoreboard_games,
    merge_with_canonical,
    to_canonical,
)  # noqa: F401
from lolpred.data.loader import TEAM_STAT_COLS
from lolpred.data.synthetic import CANONICAL_COLUMNS


def _raw_row(**overrides) -> dict:
    """One hand-built ScoreboardGames cargo row (all strings, like the API)."""
    row = {
        "Team1": "T1",
        "Team2": "Gen.G",
        "Winner": "1",
        "DateTime_UTC": "2026-01-15 08:30:00",
        "Patch": "26.01",
        "OverviewPage": "LCK/2026 Season/Rounds 1-2",
        "Team1Score": "1",
        "Team2Score": "0",
        "Gamelength_Number": "32.5",
        "GameId": "LCK/2026 Season/Rounds 1-2_Week 1_1_1",
        "MatchId": "LCK/2026 Season/Rounds 1-2_Week 1_1",
        "N_GameInMatch": "1",
        "Team1Kills": "18",
        "Team2Kills": "7",
    }
    row.update(overrides)
    return row


def _oe_frame(rows: list[dict]) -> pd.DataFrame:
    """Small hand-built OE canonical frame (contract section 1 schema)."""
    out = []
    for i, r in enumerate(rows):
        base = {c: np.nan for c in CANONICAL_COLUMNS}
        base.update(
            gameid=f"OE_{i}",
            date=pd.Timestamp(r.get("date", "2025-06-01")),
            league=r.get("league", "LCK"),
            year=pd.Timestamp(r.get("date", "2025-06-01")).year,
            split="Summer",
            playoffs=0,
            patch="25.10",
            game_in_series=1,
            series_id="x",
            datacompleteness="complete",
            blue_team=r["blue_team"],
            red_team=r["red_team"],
            blue_win=1,
            gamelength=1800.0,
            blue_players="",
            red_players="",
        )
        out.append(base)
    return pd.DataFrame(out, columns=CANONICAL_COLUMNS)


# ---------------------------------------------------------------------------
# to_canonical
# ---------------------------------------------------------------------------

class TestToCanonical:
    def test_schema_matches_canonical_columns(self):
        got = to_canonical(pd.DataFrame([_raw_row()]))
        assert list(got.columns) == CANONICAL_COLUMNS

    def test_basic_mapping(self):
        got = to_canonical(pd.DataFrame([_raw_row()]))
        assert len(got) == 1
        g = got.iloc[0]
        assert g["gameid"] == "LP_LCK/2026 Season/Rounds 1-2_Week 1_1_1"
        assert g["gameid"].startswith("LP_")
        assert g["date"] == pd.Timestamp("2026-01-15 08:30:00")
        assert g["year"] == 2026
        assert g["blue_team"] == "T1"
        assert g["red_team"] == "Gen.G"
        assert g["blue_win"] == 1
        assert g["playoffs"] == 0
        assert g["patch"] == "26.01"
        assert g["game_in_series"] == 1
        assert g["datacompleteness"] == "leaguepedia"
        assert g["blue_players"] == "" and g["red_players"] == ""
        assert g["series_id"] == "2026-01-15|LCK|Gen.G|T1"

    @pytest.mark.parametrize("page,league", [
        ("LCK/2026 Season/Rounds 1-2", "LCK"),
        ("LEC/2026 Season/Winter Season", "LEC"),
        ("Mid-Season Invitational/2026", "Mid-Season Invitational"),
        ("LPL", "LPL"),
    ])
    def test_league_extraction(self, page, league):
        got = to_canonical(pd.DataFrame([_raw_row(OverviewPage=page)]))
        assert got.iloc[0]["league"] == league

    def test_winner_mapping(self):
        raw = pd.DataFrame([
            _raw_row(Winner="1", GameId="a"),
            _raw_row(Winner="2", GameId="b"),
        ])
        got = to_canonical(raw).set_index("gameid")
        assert got.loc["LP_a", "blue_win"] == 1
        assert got.loc["LP_b", "blue_win"] == 0

    def test_gamelength_minutes_to_seconds(self):
        got = to_canonical(pd.DataFrame([_raw_row(Gamelength_Number="32.5")]))
        assert got.iloc[0]["gamelength"] == pytest.approx(32.5 * 60)

    def test_gamelength_missing_is_nan(self):
        got = to_canonical(pd.DataFrame([_raw_row(Gamelength_Number=None)]))
        assert np.isnan(got.iloc[0]["gamelength"])

    def test_gameid_falls_back_to_matchid_plus_game(self):
        got = to_canonical(pd.DataFrame(
            [_raw_row(GameId="", MatchId="M1", N_GameInMatch="3")]))
        assert got.iloc[0]["gameid"] == "LP_M1_3"
        assert got.iloc[0]["game_in_series"] == 3

    def test_stat_columns_nan_floats_except_kills(self):
        got = to_canonical(pd.DataFrame([_raw_row()]))
        for side in ("blue", "red"):
            for stat in TEAM_STAT_COLS:
                col = f"{side}_{stat}"
                assert got[col].dtype == np.float64, col
                if stat == "kills":
                    assert not got[col].isna().any()
                else:
                    assert got[col].isna().all(), col
        assert got.iloc[0]["blue_kills"] == 18.0
        assert got.iloc[0]["red_kills"] == 7.0

    def test_kills_absent_from_api_stay_nan(self):
        raw = pd.DataFrame([_raw_row()]).drop(
            columns=["Team1Kills", "Team2Kills"])
        got = to_canonical(raw)
        assert got["blue_kills"].isna().all()
        assert got["blue_kills"].dtype == np.float64

    def test_drops_missing_team_winner_date(self):
        raw = pd.DataFrame([
            _raw_row(GameId="ok"),
            _raw_row(Team1="", GameId="no_team"),
            _raw_row(Team2=None, GameId="no_team2"),
            _raw_row(Winner="", GameId="no_winner"),
            _raw_row(Winner="3", GameId="bad_winner"),
            _raw_row(DateTime_UTC=None, GameId="no_date"),
        ])
        got = to_canonical(raw)
        assert list(got["gameid"]) == ["LP_ok"]

    def test_dedupe_and_sort(self):
        raw = pd.DataFrame([
            _raw_row(GameId="b", DateTime_UTC="2026-01-16 08:00:00"),
            _raw_row(GameId="a", DateTime_UTC="2026-01-15 08:00:00"),
            _raw_row(GameId="a", DateTime_UTC="2026-01-15 08:00:00"),
        ])
        got = to_canonical(raw)
        assert list(got["gameid"]) == ["LP_a", "LP_b"]

    def test_typed_values_cargoexport_style(self):
        # Special:CargoExport returns typed JSON, not strings.
        raw = pd.DataFrame([_raw_row(
            Winner=2, Team1Score=0, Team2Score=1,
            Gamelength_Number=29.316666666667, N_GameInMatch=1,
            Patch=26.13, Team1Kills=17, Team2Kills=11,
        )])
        got = to_canonical(raw)
        g = got.iloc[0]
        assert g["blue_win"] == 0
        assert g["gamelength"] == pytest.approx(29.316666666667 * 60)
        assert g["blue_kills"] == 17.0
        assert g["patch"] == "26.13"

    def test_empty_input(self):
        got = to_canonical(pd.DataFrame())
        assert len(got) == 0
        assert list(got.columns) == CANONICAL_COLUMNS


# ---------------------------------------------------------------------------
# merge_with_canonical
# ---------------------------------------------------------------------------

class TestMerge:
    def _lp(self, rows):
        return to_canonical(pd.DataFrame(rows))

    def test_lp_after_oe_max_kept_before_dropped(self):
        oe = _oe_frame([
            {"blue_team": "T1", "red_team": "Gen.G", "date": "2025-10-05"},
        ])
        lp = self._lp([
            _raw_row(GameId="early", DateTime_UTC="2025-10-04 08:00:00"),
            _raw_row(GameId="exact", DateTime_UTC="2025-10-05 00:00:00"),
            _raw_row(GameId="late", DateTime_UTC="2026-01-15 08:00:00"),
        ])
        merged = merge_with_canonical(oe, lp)
        lp_ids = set(merged.loc[merged["gameid"].str.startswith("LP_"),
                                "gameid"])
        assert lp_ids == {"LP_late"}
        assert merged.attrs["lp_appended"] == 1
        assert len(merged) == len(oe) + 1

    def test_alias_exact_normalized_rename(self):
        oe = _oe_frame([
            {"blue_team": "Gen.G", "red_team": "T1", "date": "2025-10-05"},
        ])
        lp = self._lp([_raw_row(GameId="g", Team1="T1",
                                Team2="Gen.G eSports",
                                DateTime_UTC="2026-01-15 08:00:00")])
        merged = merge_with_canonical(oe, lp)
        row = merged[merged["gameid"] == "LP_g"].iloc[0]
        assert row["red_team"] == "Gen.G"
        assert merged.attrs["lp_renames"] == {"Gen.G eSports": "Gen.G"}
        # series_id rebuilt with the renamed team
        assert row["series_id"] == "2026-01-15|LCK|Gen.G|T1"

    def test_alias_fuzzy_rename(self):
        oe = _oe_frame([
            {"blue_team": "Hanwha Life Esports", "red_team": "T1",
             "date": "2025-10-05"},
        ])
        lp = self._lp([_raw_row(GameId="g", Team1="Hanwha Life Esport",
                                Team2="T1",
                                DateTime_UTC="2026-01-15 08:00:00")])
        merged = merge_with_canonical(oe, lp)
        row = merged[merged["gameid"] == "LP_g"].iloc[0]
        assert row["blue_team"] == "Hanwha Life Esports"

    def test_new_team_passthrough(self):
        oe = _oe_frame([
            {"blue_team": "T1", "red_team": "Gen.G", "date": "2025-10-05"},
        ])
        lp = self._lp([_raw_row(GameId="g", Team1="Brand New Org 2026",
                                Team2="T1",
                                DateTime_UTC="2026-01-15 08:00:00")])
        merged = merge_with_canonical(oe, lp)
        row = merged[merged["gameid"] == "LP_g"].iloc[0]
        assert row["blue_team"] == "Brand New Org 2026"
        assert "Brand New Org 2026" in merged.attrs["lp_new_teams"]

    def test_identical_name_not_counted_as_rename(self):
        oe = _oe_frame([
            {"blue_team": "T1", "red_team": "Gen.G", "date": "2025-10-05"},
        ])
        lp = self._lp([_raw_row(GameId="g",
                                DateTime_UTC="2026-01-15 08:00:00")])
        merged = merge_with_canonical(oe, lp)
        assert merged.attrs["lp_renames"] == {}
        assert merged.attrs["lp_new_teams"] == []

    def test_merged_sorted_and_schema_stable(self):
        oe = _oe_frame([
            {"blue_team": "T1", "red_team": "Gen.G", "date": "2025-10-05"},
        ])
        lp = self._lp([_raw_row(GameId="g",
                                DateTime_UTC="2026-01-15 08:00:00")])
        merged = merge_with_canonical(oe, lp)
        assert list(merged.columns) == CANONICAL_COLUMNS
        assert merged["date"].is_monotonic_increasing

    def test_alias_map_only_uses_recent_oe_teams(self):
        alias, new = build_team_alias_map(
            {"Gen.G", "T1"}, {"Gen.G eSports", "T1", "Fresh Team"})
        assert alias == {"Gen.G eSports": "Gen.G"}
        assert new == ["Fresh Team"]

    def test_fuzzy_rename_blocked_without_shared_league(self):
        # 'FONatic' (Brazilian amateur league) must not merge into 'Fnatic'.
        alias, new = build_team_alias_map(
            {"Fnatic"}, {"FONatic"},
            oe_leagues={"Fnatic": {"LEC"}},
            lp_leagues={"FONatic": {"IDL Kings Lendas Season 3"}})
        assert alias == {}
        assert new == ["FONatic"]

    def test_fuzzy_rename_allowed_with_shared_league(self):
        alias, new = build_team_alias_map(
            {"Ninjas in Pyjamas"}, {"Ninjas in Pyjamas.CN"},
            oe_leagues={"Ninjas in Pyjamas": {"LPL"}},
            lp_leagues={"Ninjas in Pyjamas.CN": {"LPL"}})
        assert alias == {"Ninjas in Pyjamas.CN": "Ninjas in Pyjamas"}
        assert new == []

    def test_non_latin_names_never_match_each_other(self):
        # Both normalize to "" — must NOT be treated as an exact match.
        alias, new = build_team_alias_map({"ΣΤΕΝΑΧΩΡΕΜΕΝΟΙ"}, {"코알라"})
        assert alias == {}
        assert new == ["코알라"]

    def test_html_entities_unescaped_in_team_names(self):
        got = to_canonical(pd.DataFrame(
            [_raw_row(Team1="4 Swines &amp; A Bum")]))
        assert got.iloc[0]["blue_team"] == "4 Swines & A Bum"


# ---------------------------------------------------------------------------
# live probe (network; opt-in)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(os.environ.get("LP_NET") != "1",
                    reason="set LP_NET=1 to hit the live Leaguepedia API")
def test_live_probe_small_window():
    raw = fetch_scoreboard_games("2026-01-15", "2026-01-16", max_pages=2)
    assert len(raw) > 0
    got = to_canonical(raw)
    assert len(got) > 0
    assert (got["datacompleteness"] == "leaguepedia").all()
    assert got["gameid"].str.startswith("LP_").all()
