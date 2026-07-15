"""End-to-end tests for the CLI scripts (backtest.py, train.py, predict.py).

Runs the scripts as subprocesses (sys.executable) against a tiny synthetic
dataset. build_features.py consumes raw OE CSVs only, so its core (loader +
builder) is covered by test_loader/test_features; here we exercise the
scripts that consume feature/games parquets, which we create directly from
``generate_synthetic_games`` + ``build_matchup_features``.

Model sizes are kept tiny via the scripts' --model-params JSON option so the
whole module stays well under the runtime budget.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from lolpred.data.synthetic import generate_synthetic_games
from lolpred.features.build import build_matchup_features

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"

MODEL_PARAMS = '{"n_estimators": 60, "n_jobs": 2}'
PREDICT_DATE = "2020-09-15"  # after the synthetic span (starts 2020-01-01)


def run_script(script: str, *args, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / script), *map(str, args)],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    if check and proc.returncode != 0:
        raise AssertionError(
            f"{script} exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    return proc


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory) -> Path:
    """Tiny synthetic dataset: games.parquet + features.parquet."""
    d = tmp_path_factory.mktemp("scripts_data")
    games = generate_synthetic_games(n_teams=8, n_days=250, seed=3)
    feats = build_matchup_features(games)
    games.to_parquet(d / "games.parquet", index=False)
    feats.to_parquet(d / "features.parquet", index=False)
    return d


@pytest.fixture(scope="module")
def model_dir(data_dir, tmp_path_factory) -> Path:
    """Artifacts of one train.py run (shared across predict tests)."""
    d = tmp_path_factory.mktemp("model")
    run_script(
        "train.py",
        "--features", data_dir / "features.parquet",
        "--out-dir", d,
        "--model-params", MODEL_PARAMS,
        "--seed", 0,
    )
    return d


def _two_teams(data_dir: Path) -> tuple[str, str]:
    games = pd.read_parquet(data_dir / "games.parquet")
    counts = games["blue_team"].value_counts()
    return str(counts.index[0]), str(counts.index[1])


# ---------------------------------------------------------------- backtest


def test_backtest_end_to_end(data_dir, tmp_path):
    out = tmp_path / "bt"
    proc = run_script(
        "backtest.py",
        "--features", data_dir / "features.parquet",
        "--out-dir", out,
        "--burn-in-end", "2020-04-30",
        "--fold-months", 2,
        "--model-params", MODEL_PARAMS,
        "--seed", 0,
    )
    assert proc.returncode == 0  # a report, not a gate

    for name in ("preds.parquet", "bets.parquet", "report.txt", "metrics.json"):
        assert (out / name).is_file(), f"missing artifact {name}"

    report = (out / "report.txt").read_text()
    assert "SYNTHETIC ODDS" in report
    assert report in proc.stdout or "Walk-forward evaluation report" in proc.stdout

    metrics = json.loads((out / "metrics.json").read_text())
    model_m = metrics["models"]["model"]
    assert model_m["n"] > 0
    assert 0.0 < model_m["logloss"] < 2.0
    # synthetic data has real signal: the model must beat the coin flip
    assert model_m["logloss"] < metrics["models"]["const_0.5"]["logloss"]
    assert metrics["n_folds"] >= 2
    assert "momentum" in metrics

    preds = pd.read_parquet(out / "preds.parquet")
    assert {"model_p", "baseline_p", "blue_win", "fold_id"} <= set(preds.columns)
    assert preds["model_p"].between(0, 1).all()

    bets = pd.read_parquet(out / "bets.parquet")
    if len(bets):
        assert {"side", "odds", "edge", "stake_frac", "won", "pnl"} <= set(bets.columns)
        assert (bets["stake_frac"] > 0).all()
        assert (bets["stake_frac"] <= 0.02 + 1e-12).all()


# ------------------------------------------------------------------- train


def test_train_writes_artifacts(data_dir, model_dir):
    assert (model_dir / "model.joblib").is_file()
    meta = json.loads((model_dir / "training_meta.json").read_text())
    assert meta["n_games"] > 0
    assert meta["n_fit"] + meta["n_val"] == meta["n_games"]
    assert isinstance(meta["feature_columns"], list) and meta["feature_columns"]
    assert all(c.startswith("f_") for c in meta["feature_columns"])
    assert isinstance(meta["calibrated"], bool)
    assert meta["val_logloss"] is None or 0.0 < meta["val_logloss"] < 2.0
    assert meta["train_end"] >= "2020-09-01"  # default cutoff = max date

    # the saved model round-trips and honors the feature contract
    from lolpred.models.xgb import WinModel

    model = WinModel.load(model_dir / "model.joblib")
    assert model.feature_columns_ == meta["feature_columns"]


# ----------------------------------------------------------------- predict


def test_predict_json_fast_path(data_dir, model_dir):
    team_a, team_b = _two_teams(data_dir)
    proc = run_script(
        "predict.py",
        "--blue", team_a.lower(),  # case-insensitive matching
        "--red", team_b,
        "--best-of", 5,
        "--date", PREDICT_DATE,
        "--games-cache", data_dir / "games.parquet",
        "--model-dir", model_dir,
        "--score", "1-1",
        "--odds-blue", 1.8,
        "--odds-red", 2.1,
        "--json",
    )
    result = json.loads(proc.stdout)

    assert result["blue"] == team_a  # canonical casing restored
    assert result["red"] == team_b
    p = result["p_blue"]
    assert 0.0 < p < 1.0
    assert result["p_red"] == pytest.approx(1.0 - p)

    # series prob consistent with the game prob (same favorite for Bo5)
    s = result["series_p_blue"]
    assert 0.0 < s < 1.0
    assert (p - 0.5) * (s - 0.5) >= 0.0
    # Bo5 amplifies the favorite
    assert abs(s - 0.5) >= abs(p - 0.5) - 1e-12

    assert sum(result["exact_score_probs"].values()) == pytest.approx(1.0)
    assert result["fair_odds_blue"] == pytest.approx(1.0 / p)
    assert 0.0 <= result["series_from_score"]["series_p_blue"] <= 1.0

    bet = result["betting"]
    assert bet["edge_blue"] == pytest.approx(p - 1.0 / 1.8)
    assert bet["stake_frac_blue"] >= 0.0 and bet["stake_frac_red"] >= 0.0
    # both teams have long synthetic histories -> no cold-start warnings
    assert result["hist_games_blue"] >= 10
    assert result["warnings"] == []


def test_predict_features_flag_human_output(data_dir, model_dir):
    team_a, team_b = _two_teams(data_dir)
    proc = run_script(
        "predict.py",
        "--blue", team_a,
        "--red", team_b,
        "--best-of", 3,
        "--date", PREDICT_DATE,
        "--features", data_dir / "features.parquet",  # sibling games.parquet
        "--model-dir", model_dir,
    )
    assert "game win prob" in proc.stdout
    assert "series win prob" in proc.stdout
    assert team_a in proc.stdout and team_b in proc.stdout


def test_predict_unknown_team_exits_2(data_dir, model_dir):
    team_a, _ = _two_teams(data_dir)
    proc = run_script(
        "predict.py",
        "--blue", team_a,
        "--red", team_a[:-1] + "99",  # near-miss -> suggestions
        "--date", PREDICT_DATE,
        "--games-cache", data_dir / "games.parquet",
        "--model-dir", model_dir,
        check=False,
    )
    assert proc.returncode == 2
    assert "unknown team" in proc.stderr
    assert "closest matches" in proc.stderr
