# Edge-search campaign registry

Goal: a strategy on Kalshi esports (LoL primary) with statistically
significant positive expectancy after fees/spread, surviving adversarial
audit. Aggregate model-vs-mid already failed (market sharper, +0.062
log-loss); the search targets structure the aggregate test can't see.

Protocol (binding):
- Discovery window: markets with match_start < 2026-06-20. Confirmation
  window: >= 2026-06-20 (untouched until a rule is frozen).
- A candidate must state its rule BEFORE touching the confirmation window,
  then show significant positive P&L there (cluster-bootstrap by event,
  95% CI excluding 0) AND jointly survive a multiplicity haircut across
  families tested (report family count; Bonferroni-style discount).
- Execution realism: taker legs priced at the touch (ask/1-bid) + Kalshi
  taker fee; maker legs must justify fill assumptions from actual trade
  prints, zero maker fee.

## Approach families

| id | family | idea | status |
|----|--------|------|--------|
| A | market-bias | favorite-longshot / side / league / time-of-day miscalibration in prices alone | **DEAD** (well calibrated; slices sign-unstable; first-listed-favorite curiosity n=19, unproven) |
| B | info-blend | does model logit add info over market logit? | **DEAD** (incremental coef +0.04, MDE 0.61; market subsumes model; model has real standalone signal) |
| C | series-underreaction | Bo3/Bo5 match repricing after game results vs recursion benchmark + momentum | round 2 running |
| D | stale-early | 24h->1h price-discovery window staleness; model-vs-early test | round 2 running |
| E | liquidity-provision | two-sided quoting inside ~5c spread; conservative print-based fills; maker fee 0 | round 2 running |
| E2 | maker thin-favorite | targeted: H's low-volume underconfidence captured with maker fills | round 2 running |
| F | cross-market-structure | game/map/totals/outright consistency; sibling esports | **DEAD as arb** (books coherent; 1c fee floor kills residuals). **Data win: ~18k settled markets across 13 series; KXLOLMAP n=1,988 per-game markets** |
| G | join-rescue + selection QA | | **DONE**: matched 970/1,010 via verified alias table; headline firmer (+0.066 logloss, CI [+0.043,+0.090]) |
| H | market-only-residual | drift/flow/round-number/composite | **DEAD** except lead: low-volume calibration slope 1.56 [1.22,2.03], thin favorites +2-5c gross, fails at touch by ~2c -> handed to E2; possible May->June decay |
| I | in-play | volume clock, in-play efficiency, resolution-drift capture | round 2 running |
| J | stat-protocol | acceptance bar | **DONE**: edge_protocol.py; MDE: 10c needs ~292 bets, 2c unconfirmable on LoL-only; pooled multi-series data changes this |
| K | model-vs-map-markets | game-level model vs KXLOLMAP per-game books (thinner, derivative) | queued (needs multi fetch) |
| L | pooled thin-market | H's slope test pooled across 13 esports series (~18k markets) | queued (needs multi fetch) |

Verdicts are appended below by the orchestrator as rounds complete.
