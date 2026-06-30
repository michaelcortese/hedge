# hedge — strategy roadmap

Tracking the weather-strategy improvement plan from the 6-lens tournament. #1, #2, and
#3 are implemented and merged/open. Remaining work is the follow-ups below — the next
step is **proving the nowcast edge in paper** (`run_paper.py skill`) before trusting the
skill-gate ramp on real money.

## ✅ #1 — Honest σ tied to the settlement instrument (done)
Calibrate against the IEM ASOS station daily-max (the value Kalshi settles on), not
ERA5 grid; σ floor so the correlated-source fit can't collapse; fold a structural
settlement-basis term into the predictive distribution and the reported std_error.
- `providers.iem_daily_max_f`, `Station.iem_id/iem_network`, `fit_calibration(truth=…, sigma_floor=…)`,
  `build_distribution(settlement_sigma_f=…)`, `bucket_prob_and_se(structural_se=…)`,
  runner now actually fits calibration, `deploy/config.yaml: sigma_model`.
- Station-truth is opt-in via `HEDGE_CALIBRATE_AGAINST=station` until validated against
  resolved settlements (measured basis: MIA +1.5°F, NY −1.3°F, gaps to 4.6°F).

## ✅ #2 — Catastrophe-proof sizing (done)
Fetch the real order book (was dead code), reconstruct true top-of-book + depth, add a
per-event (city-day) concentration cap and an order-book participation cap.
- `MarketView.book_top`, `MarketQuote.from_view` (book-aware, rejects crossed books),
  `RiskConfig.event_cap_frac/participation_frac`, `decide(event_at_risk=…)`,
  runner fetches the book per market and tracks per-event at-risk.

## ✅ #3 — Concentrate on the one real edge (done)
Run morning forecast bets at ~zero size; put size on the intraday obs-lag and the
deterministic "impossible-bucket" NO trade; gate live λ on demonstrated OOS skill vs
the **market mid**. **Do not size anything that hasn't beaten the market-mid Brier OOS.**
- **Per-strategy λ:** `strategy_lambda` config → runner scales λ per strategy
  (`weather_ensemble`/`weather_blend` = 0, `weather_nowcast` = 1); `decide` reports a
  "zero target size" hold.
- **Afternoon window:** `WeatherNowcastStrategy.min_hour` 12 → **14**; abstains before it.
- **Deterministic impossible/certain bucket:** `Signal.deterministic`; the nowcast emits
  it when the NWS-rounded observed max already settles a bucket (validated stations only);
  `decide` bypasses the price band for it but keeps every other guard.
- **Skill-vs-market gate:** `guard.market_skill` + `GuardConfig.skill_gate/skill_min_samples/
  skill_full_at/skill_floor`; runner records the market mid per decision and ramps λ as
  acted-on probs beat the mid's Brier OOS (the absolute/baseline latch stays the backstop).
- **Prove it first:** `scripts/run_paper.py skill` → Brier-by-hour vs market mid.
- deploy: `strategy_lambda` set; `skill_gate: true` with a 0.10 bootstrap floor.

**Operating note:** the skill gate is bootstrapped via the high-confidence deterministic
trades (they place ≥1 contract even at the floor and accrue settled samples). Confirm
the nowcast beats the mid in paper (`run_paper.py skill`) before raising `skill_floor`
or trusting the ramp on real money.

## Follow-ups surfaced while doing #1/#2
- **DB durability:** the live `hedge.db` writes to `/app/data/runs/live/` (ephemeral
  container layer), not the `/data` volume — every redeploy wipes trade/decision/guard
  history. Point the runner's state dir at `HEDGE_STATE_DIR`/`/data`.
- **Validate IEM == settlement:** extend `scripts/validate_stations.py` to confirm IEM
  daily-max equals the resolved Kalshi settlement on the 14-day set before flipping
  `HEDGE_CALIBRATE_AGAINST=station` on real money.
- **`nws_recent_temps_f` day filter:** it doesn't filter observations to the target
  local day (docstring claims it does) — harmless live, but corrupts any backtest/offset
  use of the observed max.
- **PIT/CRPS σ auto-inflation** (tournament idea 27 tail): tune a per-city variance
  multiplier from settled outcomes once enough have accrued; the σ floor is the interim.

See the full tournament output for honorable mentions (maker-only convergence, joint
categorical Kelly, adverse-selection guard, decorrelated vendors, °C→°F lattice).
