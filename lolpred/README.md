# lolpred — LoL esports match-winner prediction for betting

A machine-learning pipeline that predicts the winner of professional League of
Legends matches, built for betting/trading research: calibrated per-game win
probabilities from an XGBoost model over strictly-past-only team features,
series (best-of-N) prices by exact recursion, and a walk-forward backtest with
a Kelly-staked betting simulation. It is honest about the hard part: esports
betting markets are reasonably efficient, and a model that merely predicts
winners well is not enough — you need to beat the **de-vigged** market price
after the bookmaker margin. Nothing in this repo demonstrates real market edge
yet (see the synthetic-odds caveat below); it demonstrates a leak-free
evaluation harness with which such a claim could eventually be tested.

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest tests/ -q          # everything should pass offline

# 1. download raw Oracle's Elixir data (see Data below for the quota caveat)
.venv/bin/python scripts/download_data.py --dest data/raw --years 2014-2026 --source auto

# 2. build the feature table (also writes the games.parquet cache predict.py uses)
.venv/bin/python scripts/build_features.py --raw data/raw --out data/processed/features.parquet

# 3. walk-forward backtest + synthetic-odds betting report
.venv/bin/python scripts/backtest.py --features data/processed/features.parquet --out-dir artifacts/backtest

# 4. train the final model on everything up to a cutoff
.venv/bin/python scripts/train.py --features data/processed/features.parquet --out-dir artifacts/model

# 5. price a matchup (case-insensitive team names; suggests near-misses).
#    With --best-of 5 the quoted odds are treated as Bo5 SERIES moneyline
#    prices: edge and Kelly are computed from the series probability (the
#    output says "pricing: Bo5 series moneyline"). Use --best-of 1 to price
#    single-game odds.
.venv/bin/python scripts/predict.py --blue T1 --red "Gen.G" --best-of 5 \
    --odds-blue 1.65 --odds-red 2.30
```

Note: `predict.py` recomputes features from the canonical games table at
prediction time (no per-team feature state is persisted), so keep
`data/processed/games.parquet` (written by `build_features.py`) or `data/raw`
around. With the cache it takes seconds.

## Data

Raw data is [Oracle's Elixir](https://oracleselixir.com/) per-game match
exports (12 rows per game: 10 players + 2 team rows), one CSV per season,
2014–present. Full source table and quirks in [docs/data.md](docs/data.md).

- **Google Drive (canonical per-year files):** Drive rate-limits downloads;
  when the quota is exceeded it serves an HTML page instead of the CSV.
  `download_data.py` detects this, deletes the bad file, and tells you to
  retry later — 2024 (and 2026) may stay missing until the quota clears.
- **Hugging Face bulk 2014–2023** and a **GitHub mirror of 2025** cover the
  rest reliably; overlapping years are deduplicated on `gameid`.
- All credit for the dataset goes to Tim "Magic" Sevenhuysen's Oracle's
  Elixir. Riot Games' data policy permits personal, non-commercial analytics
  use of esports data; this project is exactly that. Do not redistribute the
  raw files.

## Architecture

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
  -> XGBoost (depth 3-4, lr 0.03, early stopping, logloss) + Platt calibration
  -> game prob p -> series prob via exact BoN recursion
  -> betting layer: edge vs (de-vigged) market prob, threshold + quarter-Kelly, caps
  -> walk-forward backtest + report (Brier/logloss/calibration/ROI/drawdown/CI)
```

| module | one line |
|---|---|
| `lolpred/data/loader.py` | raw OE CSVs → canonical one-row-per-game blue/red table |
| `lolpred/data/synthetic.py` | synthetic games with known latent strengths, for pipeline validation |
| `lolpred/features/ratings.py` | online Elo stream + time-decayed Bradley-Terry rating streams |
| `lolpred/features/build.py` | chronological feature builder — the strict-date-visibility choke point |
| `lolpred/models/xgb.py` | `WinModel` (XGBoost + mirror augmentation + Platt) and the Elo+BT logistic baseline |
| `lolpred/series.py` | exact best-of-N recursion: game prob → series/exact-score probs |
| `lolpred/backtest/walkforward.py` | expanding-window folds with an embargo gap; out-of-sample predictions only |
| `lolpred/backtest/betting.py` | odds math, synthetic bookmaker, Kelly sizing, settlement, bankroll sim |
| `lolpred/backtest/report.py` | Brier/log-loss/ECE/reliability, momentum test, betting report text |
| `scripts/` | the CLI front doors: download → build_features → backtest → train → predict |

## Methodology

- **Model games, not series.** One training row per game (≈3× the data of
  series-level modeling); series prices come from the exact recursion
  `S(a,b) = p·S(a+1,b) + (1−p)·S(a,b+1)` in `lolpred/series.py`. Because teams
  alternate sides across a series, `predict.py` feeds the recursion the
  side-averaged per-game probability `p̄ = (p + (1 − p_swap))/2` (the model's
  prediction with the two teams' orientation swapped). The iid assumption
  behind the recursion is *tested*, not assumed: the backtest runs a
  within-series momentum regression (previous-game winner vs. next-game
  outcome, controlling for model skill).
- **Antisymmetry — with the blue-side edge preserved.** Every team-comparative
  feature is a blue-minus-red `_diff` that exactly negates under orientation
  swap; training doubles the data with mirrored rows carrying a ±1 perspective
  column, and inference averages the two training representations of the SAME
  game. The perspective column stays active at predict time because it carries
  the real blue-side advantage (~52–53% blue win rate in pro play). A
  consequence worth stating plainly: swapping the teams is a *different
  physical game* (the other team now enjoys blue side), so
  `P(A beats B) + P(B beats A) = 1` does **not** hold identically — the sum
  exceeds 1 by (twice) the learned blue bump, and complementarity holds only
  approximately, for side-neutral matchups. See the
  `lolpred/models/xgb.py` module docstring.
- **Strict-date visibility.** Features for a game may use only games from
  strictly earlier calendar days; within-day games (including Bo-series
  siblings) are mutually invisible. Enforced at a single choke point in the
  builder and by a dedicated leakage test suite (`tests/test_leakage.py`:
  future-mutation, fake-future, sentinel-leak tests).
- **Walk-forward only.** Expanding window over half-year folds with a 7-day
  embargo gap, burn-in years never tested, optional final holdout touched
  once. No random K-fold — it leaks future ratings into the past.
- **Calibration is the product.** The primary metric is log-loss; Platt
  scaling is fit only on each fold's held-out chronological tail — split into
  an early-stopping half and a calibration half when the tail has ≥200 rows —
  never on test data. `train.py` replicates the same tail convention.

## Evaluation & betting

The backtest reports accuracy, Brier score, log-loss, ECE with a reliability
table, and per-fold breakdowns — always against baselines that must be beaten:
the constant 0.5, the constant blue-side rate, the Elo+BT logistic baseline,
and the de-vigged market probability. A GBDT that cannot beat the Elo+BT
logistic by a meaningful log-loss margin out of sample should be replaced by
the simpler model. When a final holdout fold exists (`--holdout-start`), the
report keeps it out of every headline number and shows it in a separate
"HOLDOUT (untouched)" section.

**All betting numbers in this repo are computed against SYNTHETIC odds — no
historical esports odds ship with it. The synthetic bookmaker is the
out-of-sample baseline model's probability, shrunk, noised, and vigged. It is
generated from the baseline rather than the evaluated model to avoid grading
the model against itself, but the baseline is trained on the same games, so it
is only independent-ish. Synthetic-odds ROI validates the plumbing; it is NOT
evidence of real market edge.** Every betting section in the report carries
this label.

Stakes are quarter-Kelly (`kelly_mult 0.25`) capped at 2% of bankroll per bet,
with a cold-start gate that refuses to bet unless both teams have ≥10 prior
games. The report includes a compounded bankroll simulation, max drawdown, a
bootstrap 95% CI on ROI, and a flat-stake comparison arm as a miscalibration
canary.

## Results

Walk-forward backtest over the full corpus (90,827 pro games, 2014–2025;
burn-in through 2018; twelve 6-month folds 2019–2024; **2025 reserved as an
untouched holdout**). Full report: `docs/RESULTS.md` (regenerate with
`scripts/backtest.py --holdout-start 2025-01-01`).

**Probability quality, out-of-sample 2019–2024 (62,375 games):**

| model | accuracy | Brier | log-loss |
|---|---|---|---|
| XGBoost (this repo) | **0.6293** | **0.2235** | **0.6371** |
| Elo+BT logistic baseline | 0.6242 | 0.2255 | 0.6416 |
| constant blue rate | 0.5289 | 0.2492 | 0.6915 |
| synthetic de-vigged market | 0.6027 | 0.2340 | 0.6602 |

Paired log-loss difference (model − baseline): **−0.0045**, 95%
series-cluster-bootstrap CI **[−0.0055, −0.0035]** — the model's edge over the
rating-only baseline is small but decisively nonzero, and it held in **every
one of the 12 folds**. Calibration ECE 0.0064 over 10 equal-count bins.

**Untouched 2025 holdout (9,423 games, touched once):** accuracy 0.6309,
log-loss 0.6342 vs baseline 0.6389, ECE 0.0121 — the edge generalizes forward
in time.

**Betting simulation (SYNTHETIC odds — plumbing validation only, see above):**
26k bets selected at ≥5¢ edge, hit rate 0.537, ROI ≈ +29% with tight CI.
Against a synthetic book this is *by construction* beatable and says nothing
about beating a real market; it demonstrates that selection, staking,
settlement, and accounting work end to end.

**Honest finding — iid violation within series:** the momentum diagnostic
(lag coefficient +0.38, sign-stable under cluster bootstrap) shows the winner
of the previous game in a series wins the next one more often than the model's
probability implies. This is momentum *or* model misspecification (the
regression only de-confounds if the model were perfect); either way, series
prices computed from the iid recursion are slightly too kind to underdogs in
`predict.py`, and this is the first thing v2 should model.

## Limitations & roadmap

- **2026 (current season) data** needs the Google Drive quota to clear
  (`scripts/download_data.py --source gdrive --years 2026`); 2014–2025 download
  reliably from the HF/Kaggle/GitHub mirrors wired into the script.
- **No real historical odds.** Everything in the betting section is synthetic
  (see above). Acquiring real closing lines is the single highest-value next
  step — it is the only way to measure actual edge.
- **Draft/champion and player-level models deferred.** v1 carries only a
  roster-continuity feature; player ratings and pick/ban signal are v2
  (collinearity eats most of the gain at this data size).
- **Fearless draft** (champion-pool depletion across a series, in some leagues
  from ~2025) changes late-game-number dynamics; flagged, not yet modeled.
- **Market efficiency warning.** Pinnacle-class esports lines already embed
  most public information. Beating the vig persistently is rare; treat any
  backtested profit — especially synthetic-odds profit — with maximum
  suspicion, and size accordingly (quarter-Kelly is already generous).

## Repo layout

```
lolpred/
  data/
    loader.py        # raw CSVs -> canonical game table
    synthetic.py     # ground-truth synthetic games for pipeline validation
  features/
    ratings.py       # Elo + Bradley-Terry rating streams
    build.py         # chronological matchup feature builder
  models/
    xgb.py           # WinModel (XGBoost) + EloLogisticBaseline
  backtest/
    walkforward.py   # expanding-window folds + out-of-sample predictions
    betting.py       # odds math, synthetic bookmaker, Kelly, settlement
    report.py        # metrics, calibration, momentum test, report text
  series.py          # exact best-of-N series math
scripts/
  download_data.py   # fetch raw Oracle's Elixir CSVs
  build_features.py  # raw CSVs -> features.parquet (+ games.parquet cache)
  backtest.py        # walk-forward evaluation + synthetic-odds betting report
  train.py           # fit the final WinModel, save artifacts
  predict.py         # price a matchup: game/series probs, fair odds, Kelly
tests/               # unit + leakage + end-to-end script tests
docs/                # DESIGN.md, CONTRACTS.md, data.md
data/                # raw/ + processed/ (gitignored)
artifacts/           # backtest reports + trained models (gitignored)
```

## License & attribution

Code is MIT (see [LICENSE](LICENSE)). The match DATA is not covered by the
code license: it belongs to Oracle's Elixir and, ultimately, Riot Games, and
remains subject to their terms — it is used here for personal, non-commercial
analytics in line with Riot's data policy, and is not redistributed with this
repository. If you use the data, credit Oracle's Elixir.
