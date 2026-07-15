"""Tests for lolpred/backtest/kalshi_eval.py — no network, no real files.

Feature frames are hand-built minimal ``f_`` frames with the meta columns the
join/pricing code needs; markets frames mirror the price-parquet schema of
``lolpred/data/kalshi_market.py::build_market_prices``.  The model is a stub
with fixed, hand-computable probabilities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lolpred.backtest.kalshi_eval import (
    DISCLAIMER,
    KALSHI_TEAM_ALIASES,
    evaluate,
    join_markets_to_series,
    kalshi_taker_fee,
    model_series_prob,
)
from lolpred.series import series_win_prob

UTC = "UTC"


# ------------------------------------------------------------------ fixtures


def mk_game(gameid, date, series_id, gis, blue, red, blue_win,
            elo_diff=0.0, h_blue=0.0, h_red=0.0, league="LCK"):
    return {
        "gameid": gameid,
        "date": pd.Timestamp(date),  # naive, UTC by convention
        "league": league,
        "series_id": series_id,
        "game_in_series": gis,
        "blue_team": blue,
        "red_team": red,
        "blue_win": blue_win,
        "f_elo_diff": elo_diff,
        "f_h_blue": h_blue,
        "f_h_red": h_red,
    }


def mk_market(ticker, yes_team, opp_team, match_start, result,
              t5_bid=np.nan, t5_ask=np.nan, t60_bid=np.nan, t60_ask=np.nan,
              volume=100.0):
    ms = pd.Timestamp(match_start, tz=UTC)
    return {
        "ticker": ticker,
        "event_ticker": f"KXLOLGAME-{ticker}",
        "yes_team": yes_team,
        "opp_team": opp_team,
        "match_start": ms,
        "result": result,
        "t5_bid": t5_bid,
        "t5_ask": t5_ask,
        "t5_mid": (t5_bid + t5_ask) / 2.0,
        "t5_ts": ms - pd.Timedelta(minutes=5),
        "t60_bid": t60_bid,
        "t60_ask": t60_ask,
        "t60_mid": (t60_bid + t60_ask) / 2.0,
        "t60_ts": ms - pd.Timedelta(minutes=60),
        "oi": 500.0,
        "volume": volume,
        "n_candles": 24,
    }


class StubModel:
    """Deliberately NOT antisymmetric, so the side-average is load-bearing:
    p = 0.5 + 0.2*elo_diff if elo_diff >= 0 else 0.5 + 0.1*elo_diff."""

    feature_columns_ = ["f_elo_diff", "f_h_blue", "f_h_red"]

    def predict_proba(self, X):
        d = X["f_elo_diff"].to_numpy(dtype=float)
        p = np.where(d >= 0, 0.5 + 0.2 * d, 0.5 + 0.1 * d)
        return np.clip(p, 0.01, 0.99)


class FlatModel:
    """Symmetric stub: p_bar == 0.5 + 0.1*elo_diff exactly (antisymmetric)."""

    feature_columns_ = ["f_elo_diff", "f_h_blue", "f_h_red"]

    def predict_proba(self, X):
        return np.clip(0.5 + 0.1 * X["f_elo_diff"].to_numpy(dtype=float), 0.01, 0.99)


@pytest.fixture()
def feats():
    rows = [
        # S1: Bo1, T1 (blue) vs Gen.G, T1 wins. elo_diff=1 -> FlatModel p_bar=0.6
        mk_game("g1", "2026-05-10 08:00", "S1", 1, "T1", "Gen.G", 1, elo_diff=1.0),
        # S2: Bo3 (2-0 Gen.G), game1 blue = Gen.G
        mk_game("g2", "2026-05-12 08:00", "S2", 1, "Gen.G", "T1", 1, elo_diff=1.0),
        mk_game("g3", "2026-05-12 09:00", "S2", 2, "T1", "Gen.G", 0, elo_diff=-1.0),
        # S3: Bo5 (3-1 KT Rolster over Hanwha Life Esports)
        mk_game("g4", "2026-05-20 08:00", "S3", 1, "KT Rolster", "Hanwha Life Esports", 1, elo_diff=0.5),
        mk_game("g5", "2026-05-20 09:00", "S3", 2, "Hanwha Life Esports", "KT Rolster", 1, elo_diff=-0.5),
        mk_game("g6", "2026-05-20 10:00", "S3", 3, "KT Rolster", "Hanwha Life Esports", 1, elo_diff=0.5),
        mk_game("g7", "2026-05-20 11:00", "S3", 4, "Hanwha Life Esports", "KT Rolster", 0, elo_diff=-0.5),
        # S4/S5: same team pair as S1, different weeks (nearest-disambiguation)
        mk_game("g8", "2026-06-01 08:00", "S4", 1, "T1", "Gen.G", 0, elo_diff=1.0),
        mk_game("g9", "2026-06-08 08:00", "S5", 1, "T1", "Gen.G", 1, elo_diff=1.0),
        # S6: Bo1 for the evaluate() P&L cases
        mk_game("g10", "2026-06-15 08:00", "S6", 1, "DRX", "BRION", 1, elo_diff=1.0),
        mk_game("g11", "2026-06-16 08:00", "S7", 1, "DRX", "BRION", 1, elo_diff=1.0,
                league="LPL"),
        mk_game("g12", "2026-06-17 08:00", "S8", 1, "DRX", "BRION", 0, elo_diff=1.0),
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- join


def test_join_exact_and_normalized_names(feats):
    markets = pd.DataFrame([
        # exact names
        mk_market("M1", "T1", "Gen.G", "2026-05-10 09:00", 1, 0.5, 0.55),
        # normalized: suffix stripped + punctuation ("Gen G" == "Gen.G")
        mk_market("M2", "Gen G", "T1 Esports", "2026-05-12 08:30", 1, 0.4, 0.45),
    ])
    j = join_markets_to_series(markets, feats)
    assert len(j) == 2
    assert j.set_index("ticker").loc["M1", "series_id"] == "S1"
    assert j.set_index("ticker").loc["M2", "series_id"] == "S2"
    assert int(j.set_index("ticker").loc["M2", "n_games_in_series"]) == 2
    assert j.set_index("ticker").loc["M1", "game1_gameid"] == "g1"
    assert j.set_index("ticker").loc["M1", "league"] == "LCK"
    assert j.attrs["n_matched"] == 2 and j.attrs["n_unmatched"] == 0


def test_join_time_window_and_unmatched(feats):
    markets = pd.DataFrame([
        # right teams, 10 days off any T1/Gen.G series -> outside window
        mk_market("M1", "T1", "Gen.G", "2026-05-25 09:00", 1, 0.5, 0.55),
        # unknown team pair
        mk_market("M2", "Fnatic", "G2", "2026-05-10 09:00", 0, 0.5, 0.55),
    ])
    j = join_markets_to_series(markets, feats, max_days=1.5)
    assert len(j) == 0
    assert j.attrs["n_unmatched"] == 2
    assert j.attrs["unmatched_reasons"] == {
        "no_series_within_time_window": 1,
        "no_series_for_team_pair": 1,
    }


def test_join_nearest_of_two(feats):
    # 2026-06-07 12:00 is within 1.5d of BOTH S4 (Jun 1? no — 6.2d) and S5
    # (Jun 8, 0.8d).  Use a start between S4 and S5 within window of both:
    # Jun 07 12:00 -> S4 is 6.2d away (out), S5 0.83d (in).  For a true
    # ambiguity use max_days=7: both in window, S5 nearer.
    markets = pd.DataFrame([
        mk_market("M1", "T1", "Gen.G", "2026-06-07 12:00", 1, 0.5, 0.55),
    ])
    j = join_markets_to_series(markets, feats, max_days=7.0)
    assert len(j) == 1
    assert j["series_id"].iloc[0] == "S5"  # nearest in time wins
    assert j.attrs["n_ambiguous"] >= 1


# ------------------------------------------------------------ alias fallback


@pytest.fixture()
def alias_feats():
    """Series whose feature-table names differ from the Kalshi spellings,
    exercising real KALSHI_TEAM_ALIASES entries."""
    rows = [
        # Kalshi says "DRX"; the feature table says "Kiwoom DRX".
        mk_game("a1", "2026-07-01 08:00", "SA1", 1,
                "Kiwoom DRX", "Hanwha Life Esports", 1, elo_diff=1.0),
        # The SAME org under both feature-table spellings (raw-first case):
        mk_game("a2", "2026-07-05 08:00", "SA2", 1,
                "Team Secret Whales", "MVK Esports", 1, elo_diff=1.0),
        mk_game("a3", "2026-07-12 08:00", "SA3", 1,
                "Team Secret (Vietnamese Team)", "Sentinels", 0, elo_diff=1.0),
    ]
    return pd.DataFrame(rows)


def test_join_via_alias(alias_feats):
    assert KALSHI_TEAM_ALIASES["DRX"] == "Kiwoom DRX"
    markets = pd.DataFrame([
        mk_market("M1", "DRX", "Hanwha Life Esports", "2026-07-01 09:00", 1,
                  0.5, 0.55),
    ])
    j = join_markets_to_series(markets, alias_feats)
    assert len(j) == 1
    assert j["series_id"].iloc[0] == "SA1"
    assert j.attrs["n_matched"] == 1 and j.attrs["n_unmatched"] == 0
    assert j.attrs["n_extended_window"] == 0


def test_join_raw_name_wins_over_alias(alias_feats):
    # Both Kalshi names are alias keys pointing at "Team Secret (Vietnamese
    # Team)", but the feature table ALSO carries the raw spellings — the raw
    # match must win so an alias can never break a pre-existing match.
    assert "Team Secret Whales" in KALSHI_TEAM_ALIASES
    markets = pd.DataFrame([
        # raw spelling exists in feats -> matched to SA2, not dropped
        mk_market("M1", "Team Secret Whales", "MVK Esports",
                  "2026-07-05 09:00", 1, 0.5, 0.55),
        # raw spelling absent -> matched via the alias to SA3
        mk_market("M2", "Team Secret", "Sentinels",
                  "2026-07-12 09:00", 0, 0.4, 0.45),
    ])
    j = join_markets_to_series(markets, alias_feats)
    assert j.set_index("ticker").loc["M1", "series_id"] == "SA2"
    assert j.set_index("ticker").loc["M2", "series_id"] == "SA3"
    assert j.attrs["n_unmatched"] == 0


def test_model_series_prob_accepts_aliased_yes_team(alias_feats):
    # FlatModel: elo_diff=+1 -> p_bar = 0.6 for game-1 blue = Kiwoom DRX.
    p_alias, bo = model_series_prob(FlatModel(), alias_feats, "SA1", "DRX")
    assert bo == 1
    assert p_alias == pytest.approx(0.6)
    p_raw, _ = model_series_prob(FlatModel(), alias_feats, "SA1", "Kiwoom DRX")
    assert p_raw == pytest.approx(p_alias)
    # aliased opponent-side name flips
    p_opp, _ = model_series_prob(FlatModel(), alias_feats, "SA1",
                                 "Hanwha Life Esports")
    assert p_opp == pytest.approx(1.0 - p_alias)


# ----------------------------------------------------- extended time window


def _one_series(date_str):
    return [mk_game(f"e{i}", d, f"SE{i}", 1, "Alpha Wolves", "Beta Bears", 1,
                    elo_diff=1.0)
            for i, d in enumerate(date_str)]


def test_extended_window_single_unambiguous_candidate():
    feats = pd.DataFrame(_one_series(["2026-07-20 08:00"]))
    # 1.83 days off: outside the strict 1.5d window, inside 2.5d, and the
    # only candidate within 4d -> rescued.
    markets = pd.DataFrame([
        mk_market("M1", "Alpha Wolves", "Beta Bears", "2026-07-22 04:00", 1,
                  0.5, 0.55),
    ])
    j = join_markets_to_series(markets, feats)
    assert len(j) == 1
    assert j["series_id"].iloc[0] == "SE0"
    assert j.attrs["n_extended_window"] == 1
    assert j.attrs["n_unmatched"] == 0


def test_extended_window_ambiguity_guard():
    # Two meetings of the same pair within 4 days of the market -> even
    # though the nearest is inside 2.5d, the guard refuses to guess.
    feats = pd.DataFrame(_one_series(["2026-07-20 08:00", "2026-07-24 20:00"]))
    markets = pd.DataFrame([
        # gaps: 1.92d and 2.58d -> none within 1.5d, two within 4d
        mk_market("M1", "Alpha Wolves", "Beta Bears", "2026-07-22 06:00", 1,
                  0.5, 0.55),
    ])
    j = join_markets_to_series(markets, feats)
    assert len(j) == 0
    assert j.attrs["unmatched_reasons"] == {"no_series_within_time_window": 1}
    assert j.attrs["n_extended_window"] == 0


def test_extended_window_hard_cap():
    feats = pd.DataFrame(_one_series(["2026-07-20 08:00"]))
    markets = pd.DataFrame([
        # 3.0 days off: single candidate, but beyond the 2.5d extension
        mk_market("M1", "Alpha Wolves", "Beta Bears", "2026-07-23 08:00", 1,
                  0.5, 0.55),
    ])
    j = join_markets_to_series(markets, feats)
    assert len(j) == 0
    assert j.attrs["unmatched_reasons"] == {"no_series_within_time_window": 1}


def test_in_window_match_never_counts_as_extended():
    feats = pd.DataFrame(_one_series(["2026-07-20 08:00"]))
    markets = pd.DataFrame([
        mk_market("M1", "Alpha Wolves", "Beta Bears", "2026-07-20 12:00", 1,
                  0.5, 0.55),
    ])
    j = join_markets_to_series(markets, feats)
    assert len(j) == 1
    assert j.attrs["n_extended_window"] == 0


# ------------------------------------------------------------- series pricing


def test_model_series_prob_side_average(feats):
    # StubModel: game1 of S1 has elo_diff=+1 -> p = 0.7; swapped row has
    # elo_diff=-1 -> p_swap = 0.4; p_bar = 0.5*(0.7 + 0.6) = 0.65.  Bo1 ->
    # series prob == p_bar.
    p, bo = model_series_prob(StubModel(), feats, "S1", "T1")
    assert bo == 1
    assert p == pytest.approx(0.65)


def test_model_series_prob_flip_for_red_yes_team(feats):
    p_blue, _ = model_series_prob(FlatModel(), feats, "S1", "T1")
    p_red, _ = model_series_prob(FlatModel(), feats, "S1", "Gen.G")
    assert p_blue == pytest.approx(0.6)
    assert p_red == pytest.approx(1.0 - p_blue)
    # normalized name also flips
    p_red2, _ = model_series_prob(FlatModel(), feats, "S1", "Gen G Esports")
    assert p_red2 == pytest.approx(p_red)
    with pytest.raises(ValueError):
        model_series_prob(FlatModel(), feats, "S1", "Fnatic")


def test_model_series_prob_best_of_inference(feats):
    # S1: 1 win -> Bo1; S2: 2 wins -> Bo3; S3: 3 wins -> Bo5.
    _, bo1 = model_series_prob(FlatModel(), feats, "S1", "T1")
    p3, bo3 = model_series_prob(FlatModel(), feats, "S2", "Gen.G")
    p5, bo5 = model_series_prob(FlatModel(), feats, "S3", "KT Rolster")
    assert (bo1, bo3, bo5) == (1, 3, 5)
    # S2 game1 blue = Gen.G, elo_diff=+1 -> p_bar = 0.6 -> Bo3 recursion.
    assert p3 == pytest.approx(series_win_prob(0.6, 3))
    # S3 game1 blue = KT, elo_diff=+0.5 -> p_bar = 0.55.
    assert p5 == pytest.approx(series_win_prob(0.55, 5))


# ----------------------------------------------------------------------- fees


def test_taker_fee_spot_values():
    assert kalshi_taker_fee(0.50) == pytest.approx(0.02)  # ceil(1.75) = 2c
    assert kalshi_taker_fee(0.10) == pytest.approx(0.01)  # ceil(0.63) = 1c
    assert kalshi_taker_fee(0.90) == pytest.approx(0.01)  # symmetric
    assert kalshi_taker_fee(0.01) == pytest.approx(0.01)  # ceil(0.0693) = 1c
    assert kalshi_taker_fee(0.25) == pytest.approx(0.02)  # ceil(1.3125) = 2c
    assert kalshi_taker_fee(0.0) == 0.0
    assert np.isnan(kalshi_taker_fee(float("nan")))


# ------------------------------------------------------------------- evaluate


def test_evaluate_end_to_end(feats):
    # FlatModel gives model_p = 0.6 for the DRX-yes Bo1 markets (elo_diff=+1).
    markets = pd.DataFrame([
        # YES bet at thr=0.05: edge = 0.60 - 0.50 = 0.10; fee(0.50)=0.02;
        # result=1 -> win -> pnl = 1 - 0.50 - 0.02 = +0.48
        mk_market("M1", "DRX", "BRION", "2026-06-15 09:00", 1,
                  t5_bid=0.45, t5_ask=0.50, t60_bid=0.35, t60_ask=0.45),
        # NO bet: (1-0.60) - (1-0.75) = 0.15; entry = 0.25, fee(0.25)=0.02;
        # result=1 -> NO loses -> pnl = -0.25 - 0.02 = -0.27
        mk_market("M2", "DRX", "BRION", "2026-06-16 09:00", 1,
                  t5_bid=0.75, t5_ask=0.80, t60_bid=0.75, t60_ask=0.85),
        # no bet at 0.05: YES edge 0.60-0.58=0.02, NO edge 0.55-0.60<0
        mk_market("M3", "DRX", "BRION", "2026-06-17 09:00", 0,
                  t5_bid=0.55, t5_ask=0.58, t60_bid=0.45, t60_ask=0.55),
    ])
    joined = join_markets_to_series(markets, feats)
    assert len(joined) == 3

    metrics, report = evaluate(joined, feats, FlatModel(),
                               edge_thresholds=(0.05,), seed=0)

    # labeling
    assert "REAL KALSHI PRICES" in report
    assert metrics["disclaimer"] == DISCLAIMER
    assert metrics["n_evaluated"] == 3

    # probability section on all 3 (valid mids)
    prob = metrics["probability"]
    assert prob["n"] == 3
    assert prob["model"]["n"] == 3 and prob["market_mid"]["n"] == 3
    for k in ("logloss", "brier", "accuracy"):
        d = prob["paired_diff"][k]
        assert d["ci_lo"] <= d["mean"] <= d["ci_hi"]
    # hand check the paired brier diff: model p = .6,.6,.6; mid = .475,.775,.565
    exp_brier_diff = np.mean([
        (0.6 - 1) ** 2 - (0.475 - 1) ** 2,
        (0.6 - 1) ** 2 - (0.775 - 1) ** 2,
        (0.6 - 0) ** 2 - (0.565 - 0) ** 2,
    ])
    assert prob["paired_diff"]["brier"]["mean"] == pytest.approx(exp_brier_diff)

    # P&L at thr=0.05, hand-computed
    (row,) = metrics["pnl"]
    assert row["threshold"] == 0.05
    assert row["n_bets"] == 2 and row["n_yes"] == 1 and row["n_no"] == 1
    assert row["hit_rate"] == pytest.approx(0.5)
    assert row["total_pnl"] == pytest.approx(0.48 - 0.27)
    assert row["total_staked"] == pytest.approx(0.52 + 0.27)
    assert row["roi"] == pytest.approx((0.48 - 0.27) / (0.52 + 0.27))
    assert row["roi_ci_lo"] <= row["roi"] <= row["roi_ci_hi"]

    # per-league breakdown at the (single) best threshold
    assert metrics["best_threshold"] == 0.05
    per_league = {r["league"]: r for r in metrics["per_league_best_threshold"]}
    assert per_league["LCK"]["total_pnl"] == pytest.approx(0.48)
    assert per_league["LPL"]["total_pnl"] == pytest.approx(-0.27)

    # line movement diagnostic exists (3 markets with both mids)
    lm = metrics["line_movement"]
    assert lm["n"] == 3
    # M1: t60 mid .40 -> t5 .475, model .6 above -> moved toward model +0.075
    # M2: t60 mid .80 -> t5 .775, model .6 below -> toward model +0.025
    # M3: t60 mid .50 -> t5 .565, model .6 above -> toward model +0.065
    assert lm["n_disagree_gt_5c"] == 3
    assert lm["mean_move_toward_model_on_disagree"] == pytest.approx(
        (0.075 + 0.025 + 0.065) / 3
    )

    # per-market frame ships one row per market with the model prob
    pm = metrics["per_market"]
    assert len(pm) == 3
    assert set(pm["model_p"].round(6)) == {0.6}


def test_evaluate_volume_filter_and_missing_prices(feats):
    markets = pd.DataFrame([
        mk_market("M1", "DRX", "BRION", "2026-06-15 09:00", 1,
                  t5_bid=0.45, t5_ask=0.50, volume=5.0),
        # no t5 quotes at all: excluded from prob scoring and untradable
        mk_market("M2", "DRX", "BRION", "2026-06-16 09:00", 1, volume=50.0),
    ])
    joined = join_markets_to_series(markets, feats)
    metrics, report = evaluate(joined, feats, FlatModel(),
                               min_volume=10.0, edge_thresholds=(0.03,))
    assert metrics["n_volume_filtered"] == 1
    assert metrics["n_evaluated"] == 1
    assert metrics["probability"] is None  # the survivor has no t5 mid
    assert metrics["pnl"][0]["n_bets"] == 0
    assert "REAL KALSHI PRICES" in report
