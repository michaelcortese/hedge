# Module contracts (pinned — implementers code to these)

Project root: `lolpred/` (package `lolpred`, tests in `tests/`, venv `.venv`).
All timestamps are pandas Timestamps; all probabilities are P from BLUE team's
perspective unless a function says otherwise.

## 1. Canonical game table (produced by `lolpred.data.loader.load_games`)

One row per game, blue/red oriented. Columns:

meta: `gameid` (str), `date` (Timestamp), `league` (str), `year` (int),
`split` (str|NaN), `playoffs` (int 0/1), `patch` (str|NaN),
`game_in_series` (int, Oracle's `game` column), `series_id` (str, derived as
`date.date()|league|sorted(team pair)`), `datacompleteness` (str),
`blue_team` (str), `red_team` (str), `blue_win` (int 0/1),
`gamelength` (float, seconds).

per-side stats, prefixed `blue_` / `red_` (NaN where source lacks them):
`kills, deaths, assists, firstblood, firstdragon, firstbaron, firsttower,
dragons, barons, towers, goldat15, xpat15, csat15, golddiffat15, dpm`.
Note `blue_golddiffat15 == -red_golddiffat15` when both present.

roster: `blue_players`, `red_players` (str: sorted "|"-joined starter names,
may be "" if player rows unavailable).

Sorted by (`date`, `gameid`). Deduplicated on `gameid`.

## 2. Rating streams (`lolpred/features/ratings.py`)

Chronological-state objects driven by the feature builder:

```python
class EloStream:
    def __init__(self, k_bo1=32.0, k_series=24.0, scale=400.0, mov=True,
                 side_offset_init=25.0, split_regress=0.25, init=1500.0): ...
    def pregame(self, blue: str, red: str) -> dict[str, float]:
        # {"elo_diff": ..., "elo_blue": ..., "elo_red": ...,
        #  "elo_games_blue": n, "elo_games_red": n}
    def update(self, game) -> None:  # game = row namespace w/ canonical cols
    def new_period(self, year_split_key: str) -> None:  # regression to mean

class BradleyTerry:
    # time-decayed ridge logistic on sparse +/-1 team design + blue intercept
    def __init__(self, half_life_days=60.0, l2=2.0, refit_every_days=7): ...
    def pregame(self, blue: str, red: str, date) -> dict[str, float]:
        # {"bt_theta_diff": ..., "bt_se_diff": ..., "bt_prob_blue": shrunk prob,
        #  "bt_beta_side": ...}; zeros/0.5 + big se when either team unseen
    def observe(self, game) -> None   # append to history buffer
    # refits lazily inside pregame() when date >= last_fit + refit_every_days
```

Both must be pure functions of the games observed so far — no file I/O.

## 3. Feature builder (`lolpred/features/build.py`)

```python
def build_matchup_features(games: pd.DataFrame, cfg: FeatureConfig | None = None
                           ) -> pd.DataFrame
```

Single chronological pass, grouped by calendar date: features for every game
on date d are computed from state as of the END of date d-1 (within-day games
mutually invisible), then state updates with all of date d. Output: one row
per game, meta columns preserved (`gameid, date, league, year, split, playoffs,
patch, series_id, game_in_series, blue_team, red_team, blue_win`), features
prefixed `f_`. Missing history -> NaN (XGBoost handles natively) plus
`f_hist_games_blue`, `f_hist_games_red` counters. Feature groups per DESIGN.md:
ratings (Elo, BT), rolling win rates (win10/win30/ewm-hl15), rolling
golddiffat15 & first-objective rates, side-specific win rates, rest days,
patch age, games-on-patch, roster continuity, league blue-side base rate,
playoffs, best-of/game number.

## 4. Model (`lolpred/models/xgb.py`)

```python
class WinModel:
    def __init__(self, params: dict | None = None, mirror_augment=True,
                 calibrate="platt", seed=0): ...
    def fit(self, X: pd.DataFrame, y: np.ndarray,
            X_val=None, y_val=None) -> "WinModel"  # early stopping if val given
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray  # P(blue win), 1-D
    def save(self, path) / @classmethod load(cls, path)
```

Mirror augmentation: X columns are ALL orientation-antisymmetric (`f_*_diff`
style) or orientation-flippable; the model knows which via column naming
convention: features ending `_diff` negate under mirror; `f_side_*` flip per a
registry; symmetric context features pass through. Simplest compliant
implementation: feature builder emits only `_diff` + symmetric columns, and
mirroring = negate `_diff` cols, relabel y -> 1-y. At inference predict both
orientations, average p and 1-p'.

Baseline model (same file): `EloLogisticBaseline` — logistic regression on
[elo_diff, bt_theta_diff, f_side constant] as the beat-this bar.

## 5. Betting layer (`lolpred/backtest/betting.py`)

```python
def implied_prob(decimal_odds: float) -> float
def devig_proportional(p_imp_a: float, p_imp_b: float) -> tuple[float, float]
def kelly_fraction(p: float, decimal_odds: float) -> float          # full Kelly, >=0
def make_synthetic_odds(ref_prob: np.ndarray, shrink=0.9, noise_sd=0.4,
                        margin=0.05, seed=0) -> pd.DataFrame
    # returns columns: odds_blue, odds_red (decimal, vigged), and the de-vigged
    # implied probs; noise applied on logit scale; DETERMINISTIC given seed
def select_bets(model_p: np.ndarray, odds_blue, odds_red, min_edge=0.04,
                kelly_mult=0.25, max_stake=0.02, min_hist_games=10,
                hist_games_blue=None, hist_games_red=None) -> pd.DataFrame
    # one row per bet: index into games, side ('blue'|'red'), model_p for the
    # bet side, odds, edge (model_p - vigged implied), stake_frac
def settle_bets(bets: pd.DataFrame, blue_win: np.ndarray) -> pd.DataFrame
    # adds won (bool), pnl (per unit bankroll, sequential fractional staking
    # NOT compounded here — plain per-bet return on stake), bankroll curve is
    # computed by simulate_bankroll
def simulate_bankroll(bets_settled: pd.DataFrame, start=1.0, compound=True)
    -> pd.Series  # bankroll after each bet, chronological
```

## 6. Walk-forward + report (`lolpred/backtest/walkforward.py`, `report.py`)

```python
@dataclass
class FoldSpec: train_end: Timestamp; test_start: Timestamp; test_end: Timestamp
def make_folds(dates, burn_in_end="2018-12-31", fold_months=6, gap_days=7,
               holdout_start=None) -> list[FoldSpec]
def run_walkforward(feats: pd.DataFrame, model_factory, folds) -> pd.DataFrame
    # returns feats meta + column model_p (out-of-sample) per test game,
    # plus baseline_p (EloLogisticBaseline trained same way)
```

`report.py`: `summarize(preds, bets)` -> dict + printable text: accuracy,
Brier, log-loss (model vs baselines: 0.5-const, blue-rate-const, baseline_p,
devigged synthetic market), ECE + reliability table, per-fold table, betting:
n_bets, hit rate, ROI, total staked, max drawdown, bootstrap 95% CI on ROI
(resample bets), flat-stake comparison. All betting numbers labeled
SYNTHETIC ODDS when synthetic.

## 7. Scripts

- `scripts/download_data.py [--years 2014-2026] [--dest data/raw]`
- `scripts/build_features.py [--raw data/raw --out data/processed/features.parquet]`
- `scripts/backtest.py [--features ...] [--report-out artifacts/]`
- `scripts/train.py` — fit final model on all data through a cutoff, save artifacts
- `scripts/predict.py --blue T1 --red "Gen.G" [--best-of 5] [--date today]`
   (loads saved model + feature state, prints game/series probs + fair odds
   and, given `--odds-blue/--odds-red`, edges and quarter-Kelly stakes)

## Style

Python 3.11+, type hints, docstrings, no prints inside library code (scripts
print). Every module gets a matching `tests/test_<module>.py` run with
`.venv/bin/python -m pytest`. Determinism: every stochastic path takes a seed.
