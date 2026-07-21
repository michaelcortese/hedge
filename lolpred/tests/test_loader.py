"""Tests for lolpred.data.loader: synthetic raw-OE fixtures + real-data integration."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lolpred.data.loader import TEAM_STAT_COLS, filter_games, load_games

REAL_2025 = Path(__file__).resolve().parents[1] / "data" / "raw" / "oe_2025.csv"

RAW_HEADER = [
    "gameid", "datacompleteness", "league", "year", "split", "playoffs",
    "date", "game", "patch", "participantid", "side", "position",
    "playername", "teamname", "result", "kills", "deaths", "assists",
    "firstblood", "firstdragon", "firstbaron", "firsttower", "dragons",
    "barons", "towers", "gamelength", "goldat15", "xpat15", "csat15",
    "golddiffat15", "dpm",
]

POSITIONS = ["top", "jng", "mid", "bot", "sup"]


def _game_rows(
    gameid: str,
    date: str,
    league: str,
    blue_team: str,
    red_team: str,
    blue_result: int,
    game: str = "1",
    blue_players: list[str] | None = None,
    red_players: list[str] | None = None,
    include_blue_team_row: bool = True,
    include_red_team_row: bool = True,
    blue_stats: dict | None = None,
    red_stats: dict | None = None,
    datacompleteness: str = "complete",
) -> list[dict]:
    """Build raw OE-format rows (10 player rows + up to 2 team rows) for one game."""
    blue_players = blue_players or [f"bp{i}" for i in range(1, 6)]
    red_players = red_players or [f"rp{i}" for i in range(1, 6)]
    base = {
        "gameid": gameid, "datacompleteness": datacompleteness,
        "league": league, "year": "2025", "split": "Winter", "playoffs": "0",
        "date": date, "game": game, "patch": "15.01", "gamelength": "1800",
    }
    rows = []
    for i, (name, pos) in enumerate(zip(blue_players, POSITIONS), start=1):
        rows.append({**base, "participantid": str(i), "side": "Blue",
                     "position": pos, "playername": name,
                     "teamname": blue_team, "result": str(blue_result)})
    for i, (name, pos) in enumerate(zip(red_players, POSITIONS), start=6):
        rows.append({**base, "participantid": str(i), "side": "Red",
                     "position": pos, "playername": name,
                     "teamname": red_team, "result": str(1 - blue_result)})
    if include_blue_team_row:
        rows.append({**base, "participantid": "100", "side": "Blue",
                     "position": "team", "playername": "",
                     "teamname": blue_team, "result": str(blue_result),
                     **(blue_stats or {})})
    if include_red_team_row:
        rows.append({**base, "participantid": "200", "side": "Red",
                     "position": "team", "playername": "",
                     "teamname": red_team, "result": str(1 - blue_result),
                     **(red_stats or {})})
    return rows


def _write_csv(path: Path, rows: list[dict]) -> Path:
    df = pd.DataFrame(rows)
    for c in RAW_HEADER:
        if c not in df.columns:
            df[c] = ""
    df[RAW_HEADER].to_csv(path, index=False)
    return path


@pytest.fixture()
def fixture_files(tmp_path: Path) -> tuple[Path, Path]:
    """Two raw CSVs: 3 games in file1 (one broken), G1 duplicated + G4 in file2."""
    g1 = _game_rows(
        "G1", "2025-01-11 08:07:23", "LCK", "T1", "Gen.G", blue_result=1,
        blue_players=["Zeus", "Oner", "Faker", "Gumayusi", "Keria"],
        red_players=["Kiin", "Canyon", "Chovy", "Peyz", "Lehends"],
        blue_stats={"kills": "20", "golddiffat15": "1500", "goldat15": "26000",
                    "firstblood": "1", "towers": "11", "dpm": "2100.5"},
        red_stats={"kills": "10", "golddiffat15": "-1500", "goldat15": "24500",
                   "firstblood": "0", "towers": "2", "dpm": "1800.0"},
    )
    # G2: red team row missing -> must be dropped (not one Blue + one Red).
    g2 = _game_rows("G2", "2025-01-12 10:00:00", "LEC", "FNC", "G2", 1,
                    include_red_team_row=False)
    g3 = _game_rows("G3", "2025-01-13 12:30:00", "LPL", "BLG", "TES",
                    blue_result=0, game="2", datacompleteness="partial")
    file1 = _write_csv(tmp_path / "file1.csv", g1 + g2 + g3)

    # G1 duplicated with a DIFFERENT blue team name: first file must win.
    g1_dup = _game_rows("G1", "2025-01-11 08:07:23", "LCK",
                        "WRONG-DUP-TEAM", "Gen.G", blue_result=0)
    g4 = _game_rows("G4", "2025-02-01 15:00:00", "LCK", "HLE", "DK", 1)
    file2 = _write_csv(tmp_path / "file2.csv", g1_dup + g4)
    return file1, file2


CANONICAL_META = [
    "gameid", "date", "league", "year", "split", "playoffs", "patch",
    "game_in_series", "series_id", "datacompleteness", "blue_team",
    "red_team", "blue_win", "gamelength",
]


class TestSyntheticFixture:
    def test_canonical_columns_and_one_row_per_valid_game(self, fixture_files):
        df = load_games(list(fixture_files))
        for c in CANONICAL_META:
            assert c in df.columns, f"missing meta column {c}"
        for stat in TEAM_STAT_COLS:
            assert f"blue_{stat}" in df.columns
            assert f"red_{stat}" in df.columns
        assert "blue_players" in df.columns and "red_players" in df.columns
        # G2 dropped (missing red team row); G1 deduped; G3, G4 kept.
        assert sorted(df["gameid"]) == ["G1", "G3", "G4"]
        assert df["gameid"].is_unique

    def test_dedupe_keeps_first_file(self, fixture_files):
        df = load_games(list(fixture_files))
        g1 = df[df["gameid"] == "G1"].iloc[0]
        assert g1["blue_team"] == "T1"  # not WRONG-DUP-TEAM from file2
        assert g1["blue_win"] == 1

    def test_orientation_and_stats(self, fixture_files):
        df = load_games(list(fixture_files))
        g1 = df.set_index("gameid").loc["G1"]
        assert g1["blue_team"] == "T1" and g1["red_team"] == "Gen.G"
        assert g1["blue_kills"] == 20.0 and g1["red_kills"] == 10.0
        assert g1["blue_golddiffat15"] == 1500.0
        assert g1["red_golddiffat15"] == -1500.0
        assert g1["blue_firstblood"] == 1.0 and g1["red_firstblood"] == 0.0
        assert g1["blue_dpm"] == pytest.approx(2100.5)
        g3 = df.set_index("gameid").loc["G3"]
        assert g3["blue_win"] == 0 and g3["game_in_series"] == 2
        assert np.isnan(g3["blue_golddiffat15"])  # not provided in fixture

    def test_meta_parsing(self, fixture_files):
        df = load_games(list(fixture_files))
        g1 = df.set_index("gameid").loc["G1"]
        assert g1["date"] == pd.Timestamp("2025-01-11 08:07:23")
        assert g1["year"] == 2025 and g1["playoffs"] == 0
        assert g1["gamelength"] == 1800.0
        assert g1["series_id"] == "2025-01-11|LCK|Gen.G|T1"
        assert list(df["gameid"]) == ["G1", "G3", "G4"]  # sorted by date

    def test_players_joined_sorted(self, fixture_files):
        df = load_games(list(fixture_files))
        g1 = df.set_index("gameid").loc["G1"]
        assert g1["blue_players"] == "Faker|Gumayusi|Keria|Oner|Zeus"
        assert g1["red_players"] == "Canyon|Chovy|Kiin|Lehends|Peyz"

    def test_drop_reasons_counted(self, fixture_files):
        df = load_games(list(fixture_files), verbose=True)
        drops = df.attrs["drop_counts"]
        assert drops["duplicate_gameid"] == 1  # G1 in file2
        assert drops["not_one_blue_one_red"] == 1  # G2
        assert drops["missing_or_unknown_team"] == 0
        assert drops["missing_result"] == 0

    def test_min_datacompleteness(self, fixture_files):
        df = load_games(list(fixture_files), min_datacompleteness="complete")
        assert "G3" not in set(df["gameid"])  # partial
        assert df.attrs["drop_counts"]["datacompleteness"] >= 1

    def test_filter_games(self, fixture_files):
        df = load_games(list(fixture_files))
        assert sorted(filter_games(df, leagues=["LCK"])["gameid"]) == ["G1", "G4"]
        assert list(filter_games(df, complete_only=True)["gameid"]) == ["G1", "G4"]
        assert len(filter_games(df, min_year=2026)) == 0

    def test_directory_input(self, fixture_files):
        df = load_games(fixture_files[0].parent)
        assert sorted(df["gameid"]) == ["G1", "G3", "G4"]


@pytest.mark.skipif(not REAL_2025.is_file(), reason="data/raw/oe_2025.csv not present")
class TestRealData2025:
    @pytest.fixture(scope="class")
    @staticmethod
    def loaded():
        t0 = time.monotonic()
        df = load_games(REAL_2025, verbose=True)
        return df, time.monotonic() - t0

    def test_load_time_and_volume(self, loaded):
        df, elapsed = loaded
        assert elapsed < 120, f"load took {elapsed:.1f}s"
        assert len(df) > 8000

    def test_blue_win_rate_sane(self, loaded):
        df, _ = loaded
        assert 0.4 < df["blue_win"].mean() < 0.62

    def test_no_duplicate_gameids_and_dates_2025(self, loaded):
        df, _ = loaded
        assert df["gameid"].is_unique
        assert df["date"].min() >= pd.Timestamp("2025-01-01")
        assert df["date"].max() < pd.Timestamp("2026-01-01")
        # Known quirk: the 2025 file labels a handful of late-2025 games
        # (e.g. HLL) as season 2026; dates are still within calendar 2025.
        assert (df["year"] == 2025).mean() > 0.95

    def test_golddiffat15_antisymmetry(self, loaded):
        df, _ = loaded
        both = df.dropna(subset=["blue_golddiffat15", "red_golddiffat15"])
        assert len(both) > 1000
        np.testing.assert_allclose(
            both["blue_golddiffat15"], -both["red_golddiffat15"], atol=1e-6
        )
