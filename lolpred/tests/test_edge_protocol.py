"""Tests for lolpred/backtest/edge_protocol.py — hand-built bet frames only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lolpred.backtest.edge_protocol import (
    ALPHA,
    MIN_BETS,
    MIN_EVENTS,
    evaluate_frozen_rule,
    exact_allwin_p,
    flip_haircut_ev,
    kelly_capped_stakes,
    minimum_detectable_edge,
    poisson_binomial_tail,
    summarize_families,
)


# ------------------------------------------------------------------ helpers


def mk_bets(entries, fees, wons, tickers=None, stakes=None):
    n = len(entries)
    df = pd.DataFrame(
        {
            "event_ticker": tickers if tickers is not None else [f"EV{i}" for i in range(n)],
            "entry_price": entries,
            "fee": fees,
            "won": wons,
        }
    )
    if stakes is not None:
        df["stake"] = stakes
    return df


def winners(n, entry=0.5, fee=0.01, per_event=1):
    """n winning bets spread over n/per_event events."""
    tickers = [f"EV{i // per_event}" for i in range(n)]
    return mk_bets([entry] * n, [fee] * n, [True] * n, tickers=tickers)


# ------------------------------------------------------- exact pnl accounting


def test_all_win_exact_pnl():
    bets = mk_bets([0.6, 0.4, 0.5], [0.02, 0.01, 0.0], [True, True, True])
    r = evaluate_frozen_rule(bets, n_families_tested=1, n_boot=200)
    assert r["n"] == 3
    assert r["n_events"] == 3
    assert r["staked"] == pytest.approx(0.62 + 0.41 + 0.50)
    assert r["pnl"] == pytest.approx(0.38 + 0.59 + 0.50)
    assert r["roi"] == pytest.approx(1.47 / 1.53)
    assert r["mean_pnl_per_bet"] == pytest.approx(1.47 / 3)
    assert r["max_drawdown"] == 0.0
    assert r["break_even_extra_cost_per_bet"] == pytest.approx(1.47 / 3)


def test_all_lose_exact_pnl():
    bets = mk_bets([0.6, 0.4, 0.5], [0.02, 0.01, 0.0], [False, False, False])
    r = evaluate_frozen_rule(bets, n_families_tested=1, n_boot=200)
    assert r["pnl"] == pytest.approx(-1.53)
    assert r["roi"] == pytest.approx(-1.0)
    assert r["break_even_extra_cost_per_bet"] == 0.0
    # losing everything: p-value should be ~1, verdict certainly not SIGNIFICANT
    assert r["p_value"] > 0.9
    assert r["verdict"] != "SIGNIFICANT"


def test_mixed_exact_pnl():
    bets = mk_bets([0.5, 0.3], [0.01, 0.005], [True, False])
    r = evaluate_frozen_rule(bets, n_families_tested=1, n_boot=200)
    assert r["pnl"] == pytest.approx(0.49 - 0.305)
    assert r["staked"] == pytest.approx(0.51 + 0.305)
    assert r["roi"] == pytest.approx(0.185 / 0.815)


def test_stake_scales_pnl():
    one = mk_bets([0.5], [0.01], [True])
    three = mk_bets([0.5], [0.01], [True], stakes=[3.0])
    r1 = evaluate_frozen_rule(one, n_families_tested=1, n_boot=200)
    r3 = evaluate_frozen_rule(three, n_families_tested=1, n_boot=200)
    assert r3["pnl"] == pytest.approx(3 * r1["pnl"])
    assert r3["staked"] == pytest.approx(3 * r1["staked"])
    assert r3["roi"] == pytest.approx(r1["roi"])  # roi is stake-invariant here


def test_max_drawdown_row_order():
    # W L L W at entry 0.5 fee 0.01: pnl +0.49, -0.51, -0.51, +0.49
    # cum: .49, -.02, -.53, -.04 -> peak-to-trough = .49 - (-.53) = 1.02
    bets = mk_bets([0.5] * 4, [0.01] * 4, [True, False, False, True])
    r = evaluate_frozen_rule(bets, n_families_tested=1, n_boot=200)
    assert r["max_drawdown"] == pytest.approx(1.02)


# ------------------------------------------------------------- determinism


def test_deterministic_given_seed():
    rng = np.random.default_rng(42)
    n = 60
    bets = mk_bets(
        rng.uniform(0.2, 0.8, n).round(2),
        [0.01] * n,
        list(rng.random(n) < 0.55),
        tickers=[f"EV{i % 25}" for i in range(n)],
    )
    a = evaluate_frozen_rule(bets, n_families_tested=5, seed=7, n_boot=1500)
    b = evaluate_frozen_rule(bets, n_families_tested=5, seed=7, n_boot=1500)
    assert a == b
    c = evaluate_frozen_rule(bets, n_families_tested=5, seed=8, n_boot=1500)
    assert c["p_value"] != a["p_value"] or c["roi_ci95"] != a["roi_ci95"]


# ------------------------------------------------- cluster vs naive bootstrap


def test_cluster_bootstrap_wider_when_clusters_correlated():
    # 10 events x 5 identical bets each (event outcome fully shared) vs the
    # same 50 bets with unique tickers (clusters of size 1 == naive bootstrap).
    outcomes = [True] * 6 + [False] * 4
    entries, fees, wons, clustered = [], [], [], []
    for e, won in enumerate(outcomes):
        for _ in range(5):
            entries.append(0.5)
            fees.append(0.0)
            wons.append(won)
            clustered.append(f"EV{e}")
    correlated = mk_bets(entries, fees, wons, tickers=clustered)
    independent = mk_bets(entries, fees, wons)  # unique ticker per bet

    rc = evaluate_frozen_rule(correlated, n_families_tested=1, n_boot=4000)
    rn = evaluate_frozen_rule(independent, n_families_tested=1, n_boot=4000)
    assert rc["roi"] == pytest.approx(rn["roi"])  # same point estimate
    width_c = rc["mean_pnl_ci95"][1] - rc["mean_pnl_ci95"][0]
    width_n = rn["mean_pnl_ci95"][1] - rn["mean_pnl_ci95"][0]
    assert width_c > 1.5 * width_n  # correlation must widen the interval
    assert rc["p_value"] > rn["p_value"]  # and weaken the evidence


# ------------------------------------------------------- multiplicity & p_adj


def test_p_adj_is_bonferroni_and_capped():
    bets = winners(40)
    r = evaluate_frozen_rule(bets, n_families_tested=8, n_variants_in_family=3, n_boot=1000)
    assert r["n_tests"] == 24
    assert r["p_adj"] == pytest.approx(min(1.0, r["p_value"] * 24))

    # a coin-flippy frame with a huge test count must cap at exactly 1
    mixed = mk_bets([0.5] * 40, [0.0] * 40, [True, False] * 20)
    r2 = evaluate_frozen_rule(mixed, n_families_tested=1000, n_boot=1000)
    assert r2["p_adj"] == 1.0


def test_p_value_never_zero():
    r = evaluate_frozen_rule(winners(40), n_families_tested=1, n_boot=1000)
    assert 0.0 < r["p_value"] <= 1.0 / 1001 + 1e-12


# -------------------------------------------- exact boundary test (audit fix)


def test_exact_allwin_p_product_formula():
    # Cluster A holds costs .9 and .95 (max .95); B is .8; C is .85.
    costs = np.array([0.9, 0.95, 0.8, 0.85])
    clusters = np.array(["A", "A", "B", "C"])
    assert exact_allwin_p(costs, clusters) == pytest.approx(0.95 * 0.8 * 0.85)


def test_allwin_p_value_is_exact_not_the_floor():
    # 3-cluster all-win toy: the bootstrap p would pin at 1/(n_boot+1); the
    # honest p is the boundary-null product over cluster max costs.
    bets = mk_bets(
        [0.9, 0.95, 0.8, 0.85],
        [0.0] * 4,
        [True] * 4,
        tickers=["A", "A", "B", "C"],
    )
    r = evaluate_frozen_rule(bets, n_families_tested=1, n_boot=1000)
    expected = 0.95 * 0.8 * 0.85
    assert r["degenerate_boot"] is True
    assert r["ci_conditional_on_no_loss"] is True
    assert r["p_boot"] == pytest.approx(1 / 1001)  # the old artifact
    assert r["p_exact"] == pytest.approx(expected)
    assert r["p_value"] == pytest.approx(expected)  # NOT the floor
    assert r["p_value"] == max(r["p_boot"], r["p_exact"])


def test_allwin_91bets_51clusters_profile():
    # The audited profile: 91 all-win bets over 51 clusters, all-in cost 0.956.
    # Honest p = 0.956**51 ~ 0.10 — nowhere near significant, floor be damned.
    tickers = [f"EV{i % 51}" for i in range(91)]
    bets = mk_bets([0.946] * 91, [0.01] * 91, [True] * 91, tickers=tickers)
    r = evaluate_frozen_rule(bets, n_families_tested=1, n_boot=2000)
    assert r["n"] == 91 and r["n_events"] == 51
    assert r["degenerate_boot"] is True
    assert r["p_exact"] == pytest.approx(0.956**51)
    assert r["p_value"] == pytest.approx(0.956**51)  # ~0.101, not 1/2001
    assert r["p_value"] > 0.09
    assert r["verdict"] == "NOT_SIGNIFICANT"


def test_mixed_p_value_guard_close_to_bootstrap():
    # With real losses the bootstrap is fine; the exact tail only guards.
    # 20/20 coin flips at 0.5: both p's sit near P(Bin(40,.5) >= 20) ~ 0.563.
    bets = mk_bets([0.5] * 40, [0.0] * 40, [True, False] * 20)
    r = evaluate_frozen_rule(bets, n_families_tested=1, n_boot=4000)
    assert r["degenerate_boot"] is False
    assert r["ci_conditional_on_no_loss"] is False
    assert r["flip_rate_ub99"] is None and r["flip_haircut_mean_pnl"] is None
    assert r["p_value"] >= r["p_boot"]
    assert r["p_value"] == max(r["p_boot"], r["p_exact"])
    assert abs(r["p_value"] - r["p_boot"]) < 0.05


def test_poisson_binomial_tail_vs_enumeration():
    import itertools

    probs = [0.3, 0.5, 0.7, 0.9]
    for k_min in range(6):
        brute = 0.0
        for outcome in itertools.product([0, 1], repeat=4):
            if sum(outcome) >= k_min:
                pr = 1.0
                for o, q in zip(outcome, probs):
                    pr *= q if o else 1.0 - q
                brute += pr
        assert poisson_binomial_tail(probs, k_min) == pytest.approx(min(1.0, brute))
    with pytest.raises(ValueError):
        poisson_binomial_tail([0.5, 1.2], 1)
    with pytest.raises(ValueError):
        poisson_binomial_tail([], 0)


def test_degenerate_flag_and_all_lose_p():
    assert evaluate_frozen_rule(winners(10), n_families_tested=1, n_boot=200)[
        "degenerate_boot"
    ]
    all_lose = mk_bets([0.5] * 10, [0.0] * 10, [False] * 10)
    r = evaluate_frozen_rule(all_lose, n_families_tested=1, n_boot=200)
    assert r["degenerate_boot"] is True
    assert r["ci_conditional_on_no_loss"] is False  # losses, not wins
    assert r["p_value"] == 1.0  # exact tail P(W >= 0) = 1
    mixed = mk_bets([0.5] * 10, [0.0] * 10, [True] * 5 + [False] * 5)
    assert not evaluate_frozen_rule(mixed, n_families_tested=1, n_boot=200)[
        "degenerate_boot"
    ]


def test_flip_haircut_ev_hand_check():
    costs = np.array([0.9, 0.8, 0.85])  # mean(1 - c) = 0.15
    clusters = np.array(["A", "A", "B"])
    assert flip_haircut_ev(costs, clusters, 0.0) == pytest.approx(0.15)
    assert flip_haircut_ev(costs, clusters, 0.10) == pytest.approx(0.05)
    assert flip_haircut_ev(costs, clusters, 0.20) == pytest.approx(-0.05)
    with pytest.raises(ValueError):
        flip_haircut_ev(costs, clusters, 1.5)
    with pytest.raises(ValueError):
        flip_haircut_ev(costs, clusters[:2], 0.1)


def test_degenerate_allwin_needs_flip_haircut_cushion():
    # All-win at cost 0.985 over 200 singleton clusters: p_exact = .985**200
    # ~ 0.049 < ALPHA and every bootstrap replicate is positive (CI99 > 0),
    # but the 99% flip-rate UB (1 - .01**(1/200) ~ 0.0228) exceeds the .015
    # per-bet margin -> the haircut gate blocks SIGNIFICANT.
    bets = mk_bets([0.985] * 200, [0.0] * 200, [True] * 200)
    r = evaluate_frozen_rule(bets, n_families_tested=1, n_boot=2000)
    assert r["p_adj"] < ALPHA
    assert r["roi_ci99"][0] > 0.0
    assert r["degenerate_boot"] is True
    assert r["flip_rate_ub99"] == pytest.approx(1.0 - 0.01 ** (1 / 200))
    assert r["flip_haircut_mean_pnl"] == pytest.approx(0.015 - r["flip_rate_ub99"])
    assert r["flip_haircut_mean_pnl"] < 0.0
    assert r["verdict"] == "NOT_SIGNIFICANT"


# ------------------------------------------------------------- verdict logic


def test_verdict_significant_clear_winner():
    # 40 events, one winning bet each, cheap entry: every bootstrap replicate
    # is positive, so p_boot ~ 1/(n_boot+1) and the CI99 lower bound is > 0.
    # Post audit-fix: p_exact = 0.51**40 ~ 2e-12 << the floor, so the honest
    # max() still reports the (more conservative) floor, and the 0.49/bet
    # margin clears the 99% flip-rate haircut (~0.109) — still SIGNIFICANT.
    r = evaluate_frozen_rule(winners(40), n_families_tested=8, n_boot=2000)
    assert r["verdict"] == "SIGNIFICANT"
    assert r["p_adj"] < ALPHA
    assert r["roi_ci99"][0] > 0.0
    assert r["p_value"] == pytest.approx(r["p_boot"])  # exact p is far smaller
    assert r["ci_conditional_on_no_loss"] is True
    assert r["flip_haircut_mean_pnl"] > 0.0
    assert r["n"] >= MIN_BETS and r["n_events"] >= MIN_EVENTS


def test_verdict_insufficient_bets():
    r = evaluate_frozen_rule(winners(MIN_BETS - 1), n_families_tested=1, n_boot=500)
    assert r["verdict"] == "INSUFFICIENT_N"


def test_verdict_insufficient_events():
    # 30 bets but only 5 event clusters -> effective n too small.
    bets = winners(30, per_event=6)
    assert bets["event_ticker"].nunique() == 5
    r = evaluate_frozen_rule(bets, n_families_tested=1, n_boot=500)
    assert r["verdict"] == "INSUFFICIENT_N"


def test_verdict_not_significant_coinflip():
    bets = mk_bets([0.5] * 40, [0.0] * 40, [True, False] * 20)
    r = evaluate_frozen_rule(bets, n_families_tested=10, n_boot=2000)
    assert r["verdict"] == "NOT_SIGNIFICANT"
    assert r["p_adj"] >= ALPHA


def test_verdict_multiplicity_can_flip_it():
    # Modest winner: significant if it were the only test, not after a big
    # multiplicity bill. 26 wins / 14 losses at 0.5 with fee.
    bets = mk_bets([0.5] * 40, [0.01] * 40, [True] * 26 + [False] * 14)
    r1 = evaluate_frozen_rule(bets, n_families_tested=1, n_boot=4000)
    r400 = evaluate_frozen_rule(bets, n_families_tested=400, n_boot=4000)
    assert r400["p_adj"] > r1["p_adj"]
    assert r400["verdict"] == "NOT_SIGNIFICANT"


# ---------------------------------------------------------------- tail stats


def test_es5_and_ci_shapes():
    r = evaluate_frozen_rule(winners(40), n_families_tested=1, n_boot=1000)
    for key in ("roi_ci95", "roi_ci99", "mean_pnl_ci95", "mean_pnl_ci99"):
        lo, hi = r[key]
        assert lo <= hi
    assert r["roi_ci99"][0] <= r["roi_ci95"][0]  # 99% CI is wider
    assert r["roi_ci95"][1] <= r["roi_ci99"][1]
    assert np.isfinite(r["es5_pnl"])
    assert r["es5_pnl"] <= r["pnl"]  # worst tail can't beat the point estimate


# ------------------------------------------------------------- input errors


def test_input_validation():
    good = winners(5)
    with pytest.raises(ValueError):
        evaluate_frozen_rule(good.drop(columns=["fee"]), n_families_tested=1)
    with pytest.raises(ValueError):
        evaluate_frozen_rule(good.iloc[0:0], n_families_tested=1)
    with pytest.raises(ValueError):
        evaluate_frozen_rule(mk_bets([1.0], [0.0], [True]), n_families_tested=1)
    with pytest.raises(ValueError):
        evaluate_frozen_rule(mk_bets([0.5], [-0.01], [True]), n_families_tested=1)
    with pytest.raises(ValueError):
        evaluate_frozen_rule(good, n_families_tested=0)
    with pytest.raises(ValueError):
        evaluate_frozen_rule(good, n_families_tested=1, n_boot=10)
    with pytest.raises(TypeError):
        evaluate_frozen_rule([{"won": True}], n_families_tested=1)


# --------------------------------------------------- minimum_detectable_edge


def test_mde_known_values():
    # n_tests=1: (z_.95 + z_.8) * 0.5 / sqrt(100) * 100c = 12.43c
    assert minimum_detectable_edge(100, n_tests=1) == pytest.approx(12.43, abs=0.01)
    # the frozen bar: 10 tests -> alpha' = 0.005
    assert minimum_detectable_edge(100, n_tests=10) == pytest.approx(17.09, abs=0.01)


def test_mde_monotonicity():
    assert minimum_detectable_edge(100) < minimum_detectable_edge(50)  # more bets
    assert minimum_detectable_edge(100, n_tests=20) > minimum_detectable_edge(100, n_tests=10)
    assert minimum_detectable_edge(100, power=0.9) > minimum_detectable_edge(100, power=0.8)
    assert minimum_detectable_edge(100, per_bet_sd=0.6) > minimum_detectable_edge(100)
    # 1/sqrt(n) scaling
    assert minimum_detectable_edge(100) == pytest.approx(2 * minimum_detectable_edge(400))


def test_mde_validation():
    with pytest.raises(ValueError):
        minimum_detectable_edge(0)
    with pytest.raises(ValueError):
        minimum_detectable_edge(100, alpha=0.0)
    with pytest.raises(ValueError):
        minimum_detectable_edge(100, n_tests=0)
    with pytest.raises(ValueError):
        minimum_detectable_edge(100, per_bet_sd=-1)


# --------------------------------------------------------------------- kelly


def test_kelly_values_and_cap():
    # full Kelly (0.6 - 0.5)/(1 - 0.5) = 0.2, capped at default 2%
    assert kelly_capped_stakes(0.6, 0.5) == pytest.approx(0.02)
    assert kelly_capped_stakes(0.6, 0.5, cap=0.5) == pytest.approx(0.2)
    assert kelly_capped_stakes(0.55, 0.5, cap=1.0) == pytest.approx(0.1)
    assert kelly_capped_stakes(0.5, 0.5) == 0.0  # no edge
    assert kelly_capped_stakes(0.4, 0.5) == 0.0  # negative edge -> never bet
    with pytest.raises(ValueError):
        kelly_capped_stakes(1.0, 0.5)
    with pytest.raises(ValueError):
        kelly_capped_stakes(0.6, 0.0)
    with pytest.raises(ValueError):
        kelly_capped_stakes(0.6, 0.5, cap=0.0)


# -------------------------------------------------------- summarize_families


def test_summarize_families_table():
    ra = dict(
        evaluate_frozen_rule(winners(40), n_families_tested=10, n_boot=500),
        family="A-market-bias",
    )
    rb = dict(
        evaluate_frozen_rule(
            mk_bets([0.5] * 40, [0.0] * 40, [True, False] * 20),
            n_families_tested=10,
            n_boot=500,
        ),
        family="H-residual",
    )
    out = summarize_families([rb, ra])
    assert "A-market-bias" in out and "H-residual" in out
    # sorted by p_adj: the winner comes first
    assert out.index("A-market-bias") < out.index("H-residual")
    assert "1 SIGNIFICANT" in out
    assert "raw p < 0.05 by luck" in out
    assert "2 families reported" in out


def test_summarize_families_empty_and_unnamed():
    assert "no family results" in summarize_families([])
    r = evaluate_frozen_rule(winners(40), n_families_tested=1, n_boot=500)
    out = summarize_families([r])
    assert "family_0" in out
