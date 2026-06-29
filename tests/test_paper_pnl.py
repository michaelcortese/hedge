"""Paper-tournament P&L math: edge detection, fees, gates, and scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hedge.tournament.paper import (
    RiskParams,
    decide,
    realized_pnl,
    score,
    taker_fee,
)

RISK = RiskParams(lambda_kelly=0.25, k_sigma=2.0, tau_min_cents=2.0)


def test_fee_is_small_and_peaks_midbook():
    assert taker_fee(0.5) >= taker_fee(0.1)
    assert taker_fee(0.5) <= 0.02  # ~1.75c max near 50c


def test_buys_yes_when_underpriced():
    # We think 0.70; market asks 0.55 -> strong YES edge, tight sigma passes gate.
    d = decide(0.70, sigma=0.02, yes_bid=0.53, yes_ask=0.55, risk=RISK)
    assert d.side == "yes" and d.edge > 0 and d.kelly_frac > 0


def test_buys_no_when_overpriced():
    # We think 0.30; market bid 0.55 -> buying NO at 0.45 is +EV.
    d = decide(0.30, sigma=0.02, yes_bid=0.55, yes_ask=0.57, risk=RISK)
    assert d.side == "no"


def test_abstains_when_fairly_priced():
    d = decide(0.50, sigma=0.02, yes_bid=0.49, yes_ask=0.51, risk=RISK)
    assert d.side == "none" and d.kelly_frac == 0.0


def test_noise_gate_blocks_uncertain_edge():
    # Same nominal edge but huge sigma -> k_sigma gate refuses the trade.
    d = decide(0.70, sigma=0.30, yes_bid=0.53, yes_ask=0.55, risk=RISK)
    assert d.side == "none"


def test_abstains_on_degenerate_one_sided_book():
    # No real ask (yes_ask = 1.00) and only a tiny yes bid: the YES side is not
    # fillable, so a confident YES belief must NOT become a phantom "buy at 0" edge.
    d = decide(0.80, sigma=0.01, yes_bid=0.0, yes_ask=1.0, risk=RISK)
    assert d.side == "none"
    # yes_ask = 0.00 likewise means "no ask", not a free YES contract.
    d2 = decide(0.80, sigma=0.01, yes_bid=0.0, yes_ask=0.0, risk=RISK)
    assert d2.side == "none"


def test_no_side_still_taken_against_a_real_bid():
    # A genuine yes_bid (0.60) makes the NO side fillable at 0.40; prob 0.20 -> buy NO.
    d = decide(0.20, sigma=0.01, yes_bid=0.60, yes_ask=1.0, risk=RISK)
    assert d.side == "no" and d.edge > 0


def test_realized_pnl_signs():
    d = decide(0.70, sigma=0.02, yes_bid=0.53, yes_ask=0.55, risk=RISK)
    win = realized_pnl(d, outcome_yes=True, risk=RISK)
    lose = realized_pnl(d, outcome_yes=False, risk=RISK)
    assert win > 0 > lose


def test_calibrated_edge_is_profitable_in_aggregate():
    # A strategy that is right with probability == its stated prob, always buying
    # YES at a discount, should net positive per-contract P&L over many trades.
    rng = np.random.default_rng(0)
    rows = []
    p = 0.70
    for i in range(2000):
        rows.append({"strategy": "edge", "ticker": f"T{i}", "prob": p,
                     "sigma": 0.02, "yes_bid": 0.53, "yes_ask": 0.55, "n_draws": 20000})
    df = pd.DataFrame(rows)
    outcomes = {f"T{i}": bool(rng.random() < p) for i in range(2000)}
    board = score(df, outcomes, RISK)
    assert board.loc[board.strategy == "edge", "pnl_per_contract"].iloc[0] > 0


def test_score_skips_unsettled_and_abstained():
    df = pd.DataFrame([{"strategy": "s", "ticker": "T1", "prob": 0.5, "sigma": 0.02,
                        "yes_bid": 0.49, "yes_ask": 0.51, "n_draws": 20000}])
    assert score(df, {"T1": True}, RISK).empty  # fairly priced -> no trade
