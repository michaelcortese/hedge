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
