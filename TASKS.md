# hedge — strategy roadmap

Tracking the weather-strategy improvement plan from the 6-lens tournament. #1 and #2
are implemented on branch `weather-honest-sigma-and-sizing`; #3 is next.

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

## ⛳ #3 — Concentrate on the one real edge (NEXT)
Run morning forecast bets at ~zero size; put size on the intraday obs-lag and the
deterministic "impossible-bucket" NO trade; gate live λ on demonstrated out-of-sample
skill vs the **market mid** (not just vs climatology). Combines tournament ideas
31 + 32 + 36. **Do not size anything that hasn't beaten the market-mid Brier OOS.**

Plan:
1. **Per-strategy λ.** Let `deploy/config.yaml` set lambda per strategy so
   `weather_ensemble`/`weather_blend` run at ~0 size (paper/log only) and
   `weather_nowcast` carries size. Plumb a per-strategy multiplier through
   `_default_strategies`/`_best_decision` (the runner already picks best-edge per market).
2. **Tighten the nowcast window.** Raise `WeatherNowcastStrategy.min_hour` toward ~15
   local; return `None` before it so morning cycles abstain.
3. **Deterministic impossible-bucket NO.** In `weather_nowcast.evaluate`, when the
   fresh observed max already exceeds a bucket's upper bound, emit `Signal(prob≈ε,
   meta={'deterministic': True})`; add a `decide` branch that bypasses the
   `[min_price, max_price]` band for deterministic signals but keeps the depth /
   participation / event caps and requires a fresh obs + validated station.
4. **Skill-vs-market gate.** Extend `guard.py` to score acted-on probabilities' Brier
   against the **market mid** on settled trades (state.py has decisions/fills/outcomes);
   ramp `lambda_kelly` up only as positive market-relative skill accrues, down as it
   decays. Keep the existing absolute/baseline-Brier latch as the hard backstop.
5. **Prove it first.** `scripts/run_paper.py loop` → score Brier-by-hour; confirm the
   nowcast beats the market mid in the 15:00+ window before arming real size.

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
