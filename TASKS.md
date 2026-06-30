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

---

# Quant review (2026-06-30) — action items

Four-reviewer review (alpha/edge · risk-engine audit · proposed-changes ·
adversarial red-team) of the strategy + a proposed "more active intraday / salvage
losers" change, with direct code verification. **Headline: no edge has been measured
against the market price; the only plausible edge (deterministic obs-lag NO) is tiny,
capacity-limited, and unproven. Do NOT make the bot more active until P1+P2 clear.**
Engine arithmetic is clean (no sign/fee/idempotency bugs) — the issues are
calibration, inputs, and proof, not the math.

## P1 — correctness & honest σ (do first)
- [ ] **Nowcast must fold in `settlement_sigma`.** Verified gap: `weather_nowcast.py:132`
      calls `bucket_prob_and_se` WITHOUT `settlement_sigma_f` (cf. `weather_ensemble.py:80`).
      Roadmap #1 added the settlement-basis term to ensemble but not to the one strategy
      carrying size → its probabilistic buckets are overconfident (Jun-29 failure mode).
      Deterministic trades (`std_error=1e-6`) are unaffected; this is the probabilistic path.
- [ ] **`nws_recent_temps_f` local-day filter** (already noted L54-56). Re-confirmed:
      queries from `{date}T00:00:00+00:00` (UTC midnight = prior local evening) and appends
      every obs with no timestamp filter. **Low live risk** (nowcast `min_hour=14` → today's
      daytime max usually dominates), but a real latent bug and it **corrupts backtest/replay**
      of `obs_max`. Filter obs to the target local day (station tz).
- [ ] **Validate IEM/station == Kalshi settlement, then flip `HEDGE_CALIBRATE_AGAINST=station`**
      (already tracked L51-53; bump priority). Prod currently runs `truth=era5` *by design*
      (station-truth gated until validated) — but ERA5≠settlement is the documented Jun-29
      basis cause, so closing this is the single biggest risk reduction. Extend
      `scripts/validate_stations.py` over the resolved-settlement set first.
- [ ] **Move skill-gate / kill-switch evidence onto `/data`** (already tracked L48-50; bump
      priority). Verified: guard reads `self.state` = `LIVE_DIR/hedge.db` (`runner.py:52`,
      ephemeral), wiped every redeploy (2× in 4 min on 2026-06-30) → `skill_mult` frozen at
      the 0.10 floor, Brier kill-switch can never reach its 20/30 samples. Safety machinery
      + λ-ramp are effectively inert until this is durable.

## P2 — prove the edge before any size increase
- [ ] **Run `score_skill_vs_market` on ≥30 settled acted-on nowcast signals**, segmented
      deterministic vs probabilistic. Today it has **zero data** — edge-vs-price is literally
      unmeasured. Expectation to falsify: probabilistic skill ≈ 0 (model == mid); all skill is
      in deterministic rows → then the real question is transactable **depth** after the ~1¢ fee.
- [ ] **Stay paper-only / deterministic-NO-only until P1+P2 clear.** Collect the calibration
      set at $0 risk via `run_paper.py` rather than paying the correlated tail to gather it.

## P3 — the one defensible behavior change (low value; AFTER P1)
- [ ] **Edge-checked exit leg** (rebuild of "let it cut losers"; the rest of the proposed
      change is rejected below). Verified: in `decide()` the band/significance/`tau_min`/λ-floor
      gates (`engine.py` L261-360) all return HOLD *before* `_reconcile` (L378) and evaluate the
      opposite-side BUY (wrong leg), so non-deterministic losers can't be trimmed. Add an exit
      check that runs **ahead of** the open pipeline, gated on `manage_positions`:
      sells the **held** side on its own bid economics, **not** λ-scaled (full lot), bypasses
      the open-only band+significance gates, but **keeps** `held_bid − fair_value(held) − taker_fee
      ≥ tau_exit` (never sell a +EV hold; settlement is free). No-bid ⇒ no-op (dead buckets are
      unrecoverable). Reuse the `manage_cooldown_cycles=2` cooldown; prefer nowcast-driven exits.
      **Validate in paper vs hold-to-settlement (fee-net) before arming.** Also fix the misleading
      "exits are band-exempt" comments (`engine.py:299`, `deploy/config.yaml:41`, `signal.py:44-48`).

## WONT — rejected by the review (recorded so we don't revisit)
- **Stop-loss on win-prob/MTM threshold** — −EV "+EV-sale trap": dumping a NO the model rates
  26% at a 5¢ bid realizes a loss the model says not to take. The only defensible stop is
  "model-EV at the bid < 0", which is just the P3 exit leg.
- **Correlation-aware trim (standalone)** — backwards: holding NO across mutually-exclusive
  buckets means at most ONE loses and the rest WIN; trimming sells winners. The correlated
  case (YES across adjacent buckets) is already capped at open by `event_cap_frac=0.06`.
- **Open more / raise `portfolio_cap` / lower skill floor** — most dangerous; pours size into
  an unproven, mis-calibrated model right where it lost ~40% on Jun-29. "0 trades, over the
  cap" is the safety machinery working as designed, not a defect.

## Watch (not yet actionable)
- **Daily-loss stop is porous to correlated settlement:** the `$15` stop only gates *future*
  orders on *already-realized* P&L, but a city-day's buckets settle together in one cycle and
  blow through it. Real bound is `event_cap_frac × 4 cities ≈ 24%` of bankroll in one evening.
  Consider a pre-trade correlated-exposure cap if going live at size.
- All four settlement stations are `validated=True` (`stations.py:64-67`) — station-map risk is
  low; the live basis risk is the calibration default (P1), not the map.
