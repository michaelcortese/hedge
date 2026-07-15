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
| A | market-bias | favorite-longshot / side / league / time-of-day miscalibration in prices alone | round 1 |
| B | info-blend | does model logit add info over market logit? trade only the divergence region of the blend | round 1 |
| C | series-underreaction | Bo3/Bo5 match markets repricing after game results vs Bayesian benchmark (momentum finding says +) | pending data |
| D | stale-early | listing/t60-era prices stale vs close; bet-early drift capture | pending data |
| E | liquidity-provision | model-anchored two-sided quoting inside the ~5c spread; fills from trade prints; maker fee 0 | pending data |
| F | cross-market-structure | game vs match vs outright consistency; sum-to-1 violations; sibling esports series | round 1 |
| G | join-rescue + selection QA | raise matched n (700->~950); check match/unmatch selection bias | round 1 |
| H | market-only-residual | ML on market features only (drift continuation, volume, OI, spread) | round 1 |
| I | in-play | do markets trade during games? underreaction to elapsed state | pending data |
| J | stat-protocol | the significance/multiplicity bar all candidates must pass (this table's Protocol) | round 1 |

Verdicts are appended below by the orchestrator as rounds complete.
