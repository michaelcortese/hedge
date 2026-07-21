"""Tests for lolpred.backtest.report (metrics, calibration, momentum, summary)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lolpred.backtest.report import (
    ece,
    momentum_test,
    probability_metrics,
    reliability_table,
    summarize,
)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ------------------------------------------------------- probability_metrics


class TestProbabilityMetrics:
    def test_perfect_predictions(self):
        y = np.array([0, 1, 1, 0, 1, 0, 0, 1])
        m = probability_metrics(y, y.astype(float))
        assert m["accuracy"] == 1.0
        assert m["brier"] == 0.0
        assert m["logloss"] < 1e-10
        assert m["n"] == 8

    def test_constant_half_logloss_is_ln2(self):
        y = np.array([0, 1, 1, 0, 1, 0])
        m = probability_metrics(y, np.full(6, 0.5))
        assert np.isclose(m["logloss"], np.log(2.0))
        assert np.isclose(m["brier"], 0.25)

    def test_known_brier(self):
        # y=1, p=0.8 -> (0.2)^2; y=0, p=0.3 -> (0.3)^2
        m = probability_metrics([1, 0], [0.8, 0.3])
        assert np.isclose(m["brier"], (0.04 + 0.09) / 2)
        assert m["accuracy"] == 1.0


# ------------------------------------------------------ ece / reliability


class TestCalibration:
    def test_perfect_predictions_ece_zero(self):
        rng = np.random.default_rng(0)
        y = (rng.random(500) < 0.5).astype(int)
        assert ece(y, y.astype(float)) < 1e-12

    def test_constant_overconfident_ece_known(self):
        # p = 0.7 everywhere, y all 1 -> every bin gap is exactly 0.3.
        y = np.ones(100, dtype=int)
        p = np.full(100, 0.7)
        assert np.isclose(ece(y, p), 0.3)

    def test_calibrated_noise_small_ece(self):
        rng = np.random.default_rng(1)
        p = rng.uniform(0.1, 0.9, size=20_000)
        y = (rng.random(20_000) < p).astype(int)
        assert ece(y, p) < 0.02

    def test_reliability_table_shape_and_counts(self):
        rng = np.random.default_rng(2)
        p = rng.uniform(0, 1, 137)
        y = (rng.random(137) < p).astype(int)
        tab = reliability_table(y, p, n_bins=10)
        assert list(tab.columns) == ["bin", "n", "p_mean", "y_rate"]
        assert len(tab) == 10
        assert tab["n"].sum() == 137
        # equal-count: bin sizes differ by at most 1
        assert tab["n"].max() - tab["n"].min() <= 1
        # bins ordered by p
        assert tab["p_mean"].is_monotonic_increasing

    def test_reliability_perfect_predictions(self):
        y = np.array([0] * 50 + [1] * 50)
        tab = reliability_table(y, y.astype(float), n_bins=4)
        assert np.allclose(tab["p_mean"], tab["y_rate"])


# ------------------------------------------------------------- momentum_test


def make_series_frame(n_series=400, momentum=0.0, seed=7) -> pd.DataFrame:
    """Synthetic best-of-3 series; teams swap sides each game.

    ``model_p`` is the TRUE no-momentum probability, so any dependence of
    game k's result on the game k-1 winner beyond model_p is injected only
    via ``momentum`` (probability boost for the previous game's winner).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_series):
        a, b = f"T{s}a", f"T{s}b"
        p_a = _sigmoid(rng.normal(0.0, 1.0))  # team a's true per-game win prob
        prev_winner = None
        for k in range(1, 4):
            blue, red = (a, b) if k % 2 == 1 else (b, a)  # side swap
            p_blue = p_a if blue == a else 1.0 - p_a
            p_actual = p_blue
            if prev_winner is not None and momentum:
                p_actual += momentum if prev_winner == blue else -momentum
                p_actual = float(np.clip(p_actual, 0.01, 0.99))
            blue_win = int(rng.random() < p_actual)
            rows.append(
                {
                    "series_id": f"s{s}",
                    "game_in_series": k,
                    "blue_team": blue,
                    "red_team": red,
                    "blue_win": blue_win,
                    "model_p": p_blue,  # model unaware of any momentum
                }
            )
            prev_winner = blue if blue_win else red
    return pd.DataFrame(rows)


class TestMomentumTest:
    def test_no_momentum_unstable_sign(self):
        preds = make_series_frame(n_series=400, momentum=0.0, seed=5)
        res = momentum_test(preds, seed=0)
        assert res["n"] == 400 * 2
        assert np.isfinite(res["lag_coef"])
        assert res["sign_stability"] < 0.9

    def test_injected_momentum_detected(self):
        preds = make_series_frame(n_series=400, momentum=0.15, seed=5)
        res = momentum_test(preds, seed=0)
        assert res["lag_coef"] > 0.0
        assert res["sign_stability"] > 0.9

    def test_deterministic(self):
        preds = make_series_frame(n_series=100, momentum=0.0, seed=3)
        r1 = momentum_test(preds, seed=0)
        r2 = momentum_test(preds, seed=0)
        assert r1 == r2

    def test_caveat_field_present(self):
        preds = make_series_frame(n_series=100, momentum=0.0, seed=3)
        res = momentum_test(preds, seed=0)
        assert "caveat" in res
        assert "momentum OR model misspecification" in res["caveat"]
        # degenerate input carries the caveat too
        tiny = preds.head(3)
        res_tiny = momentum_test(tiny, seed=0)
        assert "caveat" in res_tiny


# ------------------------------------------------------------------ summarize


def make_preds(n=400, seed=5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.2, 0.8, size=n)
    return pd.DataFrame(
        {
            "gameid": [f"g{i}" for i in range(n)],
            "date": pd.date_range("2021-01-01", periods=n, freq="6h"),
            "blue_win": (rng.random(n) < p).astype(int),
            "model_p": p,
            "baseline_p": np.clip(p + rng.normal(0, 0.05, n), 0.01, 0.99),
            "fair_blue": np.clip(p + rng.normal(0, 0.03, n), 0.01, 0.99),
            "fold_id": np.repeat(np.arange(4), n // 4),
            "is_holdout": np.repeat([False, False, False, True], n // 4),
        }
    )


def make_settled_bets(n=30, seed=6) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    side = np.where(rng.random(n) < 0.5, "blue", "red")
    odds = rng.uniform(1.5, 2.8, size=n)
    stake = np.full(n, 0.02)
    won = rng.random(n) < 0.5
    pnl = np.where(won, stake * (odds - 1.0), -stake)
    return pd.DataFrame(
        {
            "side": side,
            "model_p": rng.uniform(0.4, 0.7, n),
            "odds": odds,
            "edge": rng.uniform(0.04, 0.1, n),
            "stake_frac": stake,
            "won": won,
            "pnl": pnl,
            "ret": pnl / stake,
        }
    )


class TestSummarize:
    def test_dict_keys_and_sections(self):
        preds = make_preds()  # last fold (100 rows) flagged is_holdout
        bets = make_settled_bets()  # index 0..29 -> all non-holdout rows
        result, text = summarize(preds, bets_settled=bets, synthetic_odds=True)

        assert set(result) == {
            "n", "models", "per_fold", "ece", "reliability", "momentum",
            "betting", "holdout", "logloss_diff_mean", "logloss_diff_ci_lo",
            "logloss_diff_ci_hi",
        }
        # headline n excludes the holdout rows
        n_hold = int(preds["is_holdout"].sum())
        assert result["n"] == len(preds) - n_hold
        for name in ("model", "baseline_elo_bt", "const_0.5",
                     "const_bluerate(in-sample)", "market_fair(devig)"):
            assert name in result["models"]
            assert set(result["models"][name]) == {"accuracy", "brier", "logloss", "n"}
        assert len(result["per_fold"]) == 3  # holdout fold not in headline table
        assert np.isfinite(result["ece"])

        b = result["betting"]
        for key in ("n_bets", "hit_rate", "total_staked", "total_pnl", "roi",
                    "roi_ci_lo", "roi_ci_hi", "max_drawdown", "flat_roi"):
            assert key in b, key
        assert b["n_bets"] == 30
        assert b["roi_ci_lo"] <= b["roi"] <= b["roi_ci_hi"]
        assert 0.0 <= b["max_drawdown"] <= 1.0

        assert "SYNTHETIC ODDS" in text
        assert "Betting" in text
        assert "const_bluerate(in-sample)" in text

        # paired logloss diff: keys coherent, sign convention stated in text
        assert result["logloss_diff_ci_lo"] <= result["logloss_diff_mean"]
        assert result["logloss_diff_mean"] <= result["logloss_diff_ci_hi"]
        assert "paired logloss diff (model - baseline)" in text
        assert "negative = model better" in text
        # no series_id column -> row bootstrap, momentum not computable
        assert "row-bootstrap" in text
        assert result["momentum"] is None

    def test_logloss_diff_detects_better_model(self):
        # model_p is the true probability, baseline heavily noised -> the
        # paired diff must be significantly negative (model better).
        rng = np.random.default_rng(11)
        n = 4000
        p = rng.uniform(0.2, 0.8, size=n)
        preds = pd.DataFrame(
            {
                "blue_win": (rng.random(n) < p).astype(int),
                "model_p": p,
                "baseline_p": np.clip(p + rng.normal(0, 0.25, n), 0.01, 0.99),
            }
        )
        result, _ = summarize(preds)
        assert result["logloss_diff_mean"] < 0.0
        assert result["logloss_diff_ci_hi"] < 0.0  # CI excludes zero

    def test_logloss_diff_deterministic(self):
        preds = make_preds()
        r1, t1 = summarize(preds, seed=3)
        r2, t2 = summarize(preds, seed=3)
        assert r1["logloss_diff_ci_lo"] == r2["logloss_diff_ci_lo"]
        assert r1["logloss_diff_ci_hi"] == r2["logloss_diff_ci_hi"]
        assert t1 == t2

    def test_momentum_section_with_series_columns(self):
        preds = make_series_frame(n_series=200, momentum=0.0, seed=8)
        result, text = summarize(preds)
        assert result["momentum"] is not None
        assert result["momentum"]["n"] == 200 * 2
        assert "Momentum" in text
        assert "momentum OR model misspecification" in text  # the caveat
        # series_id present -> the logloss-diff bootstrap would be
        # cluster-based, but there is no baseline_p here
        assert "logloss_diff_mean" not in result

    def test_series_cluster_bootstrap_labeled(self):
        preds = make_series_frame(n_series=200, momentum=0.0, seed=8)
        rng = np.random.default_rng(4)
        preds["baseline_p"] = np.clip(
            preds["model_p"] + rng.normal(0, 0.05, len(preds)), 0.01, 0.99
        )
        result, text = summarize(preds)
        assert "series-cluster-bootstrap" in text
        assert "logloss_diff_mean" in result

    def test_synthetic_tag_absent_when_real_odds(self):
        result, text = summarize(
            make_preds(), bets_settled=make_settled_bets(), synthetic_odds=False
        )
        assert result["betting"] is not None
        assert "Betting" in text
        assert "SYNTHETIC ODDS" not in text

    def test_no_bets_no_betting_section(self):
        result, text = summarize(make_preds(), bets_settled=None)
        assert result["betting"] is None
        assert "Betting" not in text
        assert "SYNTHETIC ODDS" not in text

    def test_holdout_separation(self):
        preds = make_preds()  # rows 300..399 are holdout
        n_hold = int(preds["is_holdout"].sum())
        assert n_hold == 100
        # bets straddle the dev/holdout boundary: index 290..309
        bets = make_settled_bets(n=20)
        bets.index = pd.RangeIndex(290, 310)
        result, text = summarize(preds, bets_settled=bets)

        assert "holdout" in result
        hold = result["holdout"]
        assert hold["n"] == n_hold
        assert result["n"] == len(preds) - n_hold
        # same probability metrics, computed on the holdout rows only
        assert set(hold["models"]["model"]) == {"accuracy", "brier", "logloss", "n"}
        assert hold["models"]["model"]["n"] == n_hold
        assert np.isfinite(hold["ece"])
        # bets split by index membership in the holdout rows
        assert result["betting"]["n_bets"] == 10
        assert hold["betting"]["n_bets"] == 10
        assert "HOLDOUT (untouched)" in text

    def test_no_holdout_rows_no_holdout_key(self):
        preds = make_preds()
        preds["is_holdout"] = False  # column present but no holdout rows
        result, text = summarize(preds)
        assert "holdout" not in result
        assert result["n"] == len(preds)
        assert "HOLDOUT" not in text

    def test_minimal_preds_columns(self):
        rng = np.random.default_rng(9)
        p = rng.uniform(0.3, 0.7, 100)
        preds = pd.DataFrame(
            {"blue_win": (rng.random(100) < p).astype(int), "model_p": p}
        )
        result, text = summarize(preds)
        assert result["per_fold"] is None
        assert "baseline_elo_bt" not in result["models"]
        assert "model" in result["models"]
        assert isinstance(text, str) and len(text) > 0
