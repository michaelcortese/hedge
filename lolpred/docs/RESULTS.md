# Backtest results (real data, 2026-07-15 run)

Command: `scripts/backtest.py --features data/processed/features.parquet --holdout-start 2025-01-01 --seed 0`

```
========================================================================
Walk-forward evaluation report
========================================================================
games scored: 62375   blue win rate: 0.5289   (excludes 9423 holdout rows — see HOLDOUT section)

-- Model vs baselines (out-of-sample) --
name                            n  accuracy    brier   logloss
model                       62375    0.6293   0.2235    0.6371
baseline_elo_bt             62375    0.6242   0.2255    0.6416
const_0.5                   62375    0.5289   0.2500    0.6931
const_bluerate(in-sample)   62375    0.5289   0.2492    0.6915
market_fair(devig)          62375    0.6027   0.2340    0.6602
paired logloss diff (model - baseline): -0.0045 (95% series-cluster-bootstrap CI [-0.0055, -0.0035]); negative = model better

-- Per-fold --
fold       n  model_ll   base_ll  holdout
   0    4774    0.6378    0.6428       no
   1    3283    0.6431    0.6484       no
   2    5264    0.6410    0.6467       no
   3    4336    0.6334    0.6372       no
   4    7714    0.6349    0.6404       no
   5    4408    0.6515    0.6533       no
   6    7550    0.6313    0.6370       no
   7    4597    0.6365    0.6392       no
   8    7080    0.6397    0.6433       no
   9    3567    0.6358    0.6380       no
  10    6145    0.6285    0.6317       no
  11    3657    0.6397    0.6493       no

-- Calibration (model) --
ECE (10 equal-count bins): 0.0064
bin       n   p_mean   y_rate
  0    6238   0.2453   0.2400
  1    6238   0.3660   0.3604
  2    6238   0.4275   0.4229
  3    6238   0.4731   0.4703
  4    6238   0.5144   0.5181
  5    6237   0.5526   0.5459
  6    6237   0.5912   0.5910
  7    6237   0.6338   0.6324
  8    6237   0.6881   0.6987
  9    6237   0.7869   0.8098

-- Momentum (within-series iid-violation check) --
lag_coef: +0.3787   sign_stability: 1.000   n: 22152
caveat: lag_coef measures residual predictability = momentum OR model misspecification; conditioning on logit(model_p) de-confounds only if model_p is the true probability

-- Betting (quarter-Kelly, compounded) [SYNTHETIC ODDS — plumbing validation only, not evidence of real edge] --
n_bets:         25993
hit_rate:       0.5367
total_staked:   518.4915 (bankroll fractions)
total_pnl:      +149.2503
ROI:            +0.2879 (bootstrap 95% CI [+0.2710, +0.3047])
max_drawdown:   0.3764
final_bankroll: 2065126253051369033254115904003059335448205579643423832408064.0000 (start 1.0)

-- Betting (flat-stake comparison arm) [SYNTHETIC ODDS — plumbing validation only, not evidence of real edge] --
flat-stake ROI: +0.2874

========================================================================
-- HOLDOUT (untouched) --
========================================================================
holdout games: 9423   blue win rate: 0.5341
name                            n  accuracy    brier   logloss
model                        9423    0.6309   0.2223    0.6342
baseline_elo_bt              9423    0.6267   0.2243    0.6389
const_0.5                    9423    0.5341   0.2500    0.6931
const_bluerate(in-sample)    9423    0.5341   0.2488    0.6908
market_fair(devig)           9423    0.6068   0.2331    0.6583
ECE (holdout): 0.0121
-- Betting on holdout rows only [SYNTHETIC ODDS — plumbing validation only, not evidence of real edge] --
n_bets:         4073
hit_rate:       0.5495
total_pnl:      +22.7668
ROI:            +0.2800 (bootstrap 95% CI [+0.2369, +0.3223])
flat-stake ROI: +0.2805

NOTE: synthetic odds were generated from the out-of-sample BASELINE
predictions (Elo+BT logistic), not from the evaluated model. The baseline is
an independent-ish reference only — it is trained on the same games — so
betting numbers validate plumbing, not real market edge.
```

# Real-market evaluation vs Kalshi (2026-07-15)

Command: `scripts/eval_vs_kalshi.py --train-end 2026-05-01`

```
==========================================================================
Model vs Kalshi evaluation
REAL KALSHI PRICES — settled markets, executable-touch backtest, no slippage beyond the spread, assumes 1-contract fills
==========================================================================
window (match_start, UTC): 2026-05-09 04:00:00+00:00 .. 2026-07-15 08:30:00+00:00
join: 700 matched / 310 unmatched of 1010 markets
  unmatched reasons: {'no_series_for_team_pair': 214, 'no_series_within_time_window': 96}
  ambiguous joins (nearest series taken): 18
evaluated: 700 markets (volume filter >= 0 dropped 0; model errors dropped 0)

-- Probability quality: model vs market t5 mid --
n = 646 markets with a valid t5 mid
name             n  accuracy    brier  logloss     ece
model          646    0.6981   0.2092   0.6119  0.0499
market_mid     646    0.7074   0.1848   0.5497  0.0462
paired diffs (model - market), 95% row-bootstrap CI, n_boot=5000:
  logloss   +0.0622 [+0.0336, +0.0917]  (negative = model better)
  brier     +0.0245 [+0.0127, +0.0360]  (negative = model better)
  accuracy  -0.0093 [-0.0372, +0.0186]  (positive = model better)

-- P&L simulation (flat 1 contract, t5 executable touch, taker fees) --
  thr n_bets  yes/no    hit     pnl$  pnl/bet  staked$     ROI          ROI 95% CI
 0.03    446 229/217  0.377    -6.96   -0.016   174.96  -0.040  [-0.139, +0.062]
 0.05    365 182/183  0.337   -15.97   -0.044   138.97  -0.115  [-0.228, -0.002]
 0.08    255 129/126  0.314   -13.66   -0.054    93.66  -0.146  [-0.286, -0.004]

per-league breakdown at best threshold (by total P&L) thr=0.03:
league           n_bets    hit     pnl$     ROI
2026 Mid-Season Invitational     30  0.333    +4.16  +0.712
LEC                  17  0.706    +2.34  +0.242
LIT                   6  0.667    +1.97  +0.970
LFL                  13  0.615    +1.83  +0.297
LCK                  26  0.385    +1.47  +0.172
LCP                  25  0.600    +1.43  +0.105
Esports World Cup 2026     45  0.489    +1.24  +0.060
LPLOL                14  0.429    +1.01  +0.202
CBLOL                 4  0.500    +0.36  +0.220
2026 Asia Masters     24  0.250    +0.15  +0.026
Road Of Legends      12  0.333    +0.04  +0.010
Prime League 1st Division     48  0.458    -0.04  -0.002
LES                   6  0.333    -0.11  -0.052
Hitpoint Masters      1  0.000    -0.17  -1.000
North American Challengers League      2  0.000    -0.56  -1.000
LPL                  43  0.326    -0.83  -0.056
Circuito Desafiante     10  0.300    -0.97  -0.244
LCK CL               11  0.455    -1.30  -0.206
LCS                  10  0.400    -1.31  -0.247
LJL                   8  0.250    -1.52  -0.432
Hellenic Legends League      4  0.000    -1.66  -1.000
Liga Regional Sur     22  0.364    -1.80  -0.184
Liga Regional Norte     26  0.231    -5.57  -0.481
EMEA Masters         39  0.077    -7.12  -0.704

-- Line movement t60 -> t5 (CLV-like diagnostic) --
n = 642   corr(sign(model - t60_mid), sign(t5_mid - t60_mid)): -0.016
mean t60->t5 movement TOWARD the model when |model - t60_mid| > 5c: -0.0007 (n = 426; positive = market moved our way)

```
