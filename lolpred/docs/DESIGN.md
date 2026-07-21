# lolpred — design synthesis

Predicting LoL pro match winners for betting/trading. This doc synthesizes an
independent multi-approach design round; each section notes which approach
family it came from. Families explored independently (registry):

| family | verdict |
|---|---|
| `rating-systems` (Elo/Glicko-2) | **adopt** as online feature stream, not standalone predictor |
| `gbdt-rolling-features` (XGBoost) | **adopt** as the main model |
| `bradley-terry-latent` (ridge logistic BT) | **adopt** as second rating stream + calibrated baseline |
| `player-composition` / draft | **v1: roster-continuity feature only**; player ratings & draft deferred (collinearity eats the gain at this data size) |
| `betting-math` | **adopt**: log-loss vs de-vigged market is the bar; quarter-Kelly; synthetic-odds clearly labeled |
| `temporal-validation-leakage` | **adopt**: expanding walk-forward, strict-date visibility, leakage test suite |
| `series-structure` | **adopt**: game-level model + exact BoN recursion; momentum treated as null |
| `data-landscape` | Oracle's Elixir game-level CSVs (see data.md) |

## Core architecture

```
raw Oracle's Elixir CSVs
  -> loader (clean, dedupe, sort by (date, gameid))
  -> chronological feature builder (single pass, strictly past-only state)
       - Elo stream (K=24/32, MOV multiplier, side offset, split regression)
       - Bradley-Terry refits (time-decayed ridge logistic, theta + std error)
       - rolling team stats (win rates, golddiffat15, first-objective rates,
         side-specific, opponent-adjusted; windows 10/30 + ewm half-life 15)
       - schedule/context (rest days, patch age, league, playoffs, roster continuity)
  -> matchup rows: blue-vs-red difference features, mirrored-row augmentation
  -> XGBoost (depth 3-4, lr 0.03, early stopping, logloss) + Platt/beta calibration on OOF
  -> game prob p -> series prob via exact BoN recursion (side-aware optional)
  -> betting layer: edge vs (de-vigged) market prob, threshold + quarter-Kelly, caps
  -> walk-forward backtest + report (Brier/logloss/calibration/ROI/drawdown/CI)
```

## Non-negotiable decisions (from the design round)

1. **Model games, not series.** Series probability = exact recursion
   `S(a,b) = p*S(a+1,b) + (1-p)*S(a,b+1)` from per-game p. 3x the training data.
2. **Antisymmetry** by difference features + mirrored-row augmentation; at
   inference predict both orientations and average p with 1-p'.
3. **Strict-date visibility.** Features for a game may use only games from
   strictly earlier dates. Within-day games (and Bo-series siblings) are
   mutually invisible. Enforced at one choke point and by unit tests
   (future-mutation, fake-future, sentinel-leak tests).
4. **Walk-forward only.** Expanding window, fold boundaries at half-year marks,
   burn-in years never tested, final holdout touched once. No random K-fold.
5. **Probability quality is the product.** Primary metric log-loss; the
   load-bearing baseline is de-vigged market implied probability. No historical
   odds available -> synthetic bookmaker (shrunk+noised reference model + margin),
   with every resulting number labeled SYNTHETIC.
6. **Quarter-Kelly, capped** (2%/bet), edge threshold tuned on validation folds
   only and frozen for the holdout. Flat-stake comparison arm as a
   miscalibration canary.
7. **Shallow trees, heavy regularization.** ~30-80k games of low-signal data;
   if XGBoost can't beat the Elo+BT logistic baseline by >=0.005 log-loss out
   of sample, ship the simpler model.
8. **Cold start:** shrink rolling features toward league means with n/(n+8)
   weight, `career_games` feature, NaN-native handling with missing flags; the
   betting layer refuses to bet when either team has <10 prior games.

## Known failure modes to watch

- Regime steps (roster swaps, big patches) that decay-based features lag on.
- Cross-region games: league offsets identified only by ~100 international
  games/year — treat as high-variance, size down.
- Bo1 vs Bo5 are different games; pool but include format/game-number features,
  and calibrate per format if reliability differs.
- Fearless draft (champion pool depletion across a series) changes late-game-
  number dynamics in some leagues from ~2025 — flagged, v2.
- A model that only beats *vigged* prices has no edge; the bar is de-vigged.
