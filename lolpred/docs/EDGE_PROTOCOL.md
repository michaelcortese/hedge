# EDGE_PROTOCOL — the acceptance bar (family J)

One page, plain language. The code is `lolpred/backtest/edge_protocol.py`
(`evaluate_frozen_rule` is the gate); it complements docs/EDGE_SEARCH.md.

## The pre-registration rule

Discovery data is markets with match_start **< 2026-06-20**; confirmation is
**>= 2026-06-20**. You may try anything in discovery. Before your rule touches
a single confirmation row, you write it down completely — entry condition,
side, price source, fee model, stake — and it never changes again. Then
`evaluate_frozen_rule` grades the confirmation bets. Anything else is grading
your own homework: the confirmation window only measures out-of-sample skill
if the rule couldn't have been shaped by it.

## Why cluster by event

Bets inside one `event_ticker` (same series/day, often the same teams) win or
lose together — they are not independent observations. Treating 5 bets on one
event as 5 data points inflates your effective sample size roughly 5x and
makes noise look significant. So the bootstrap resamples whole events, and the
sample-size gates count **events**, not bets. Your effective n is the number
of events you touched.

## Why Bonferroni across families — and what it costs

The campaign is testing ~8–10 hypothesis families, each with a handful of
variants. If everything were noise, running 10 tests at p < 0.05 hands you a
~40% chance of at least one false "edge". Bonferroni (`p_adj = p × families ×
variants`, capped at 1) is blunt but assumption-free: it holds the chance of
declaring even one false edge at ≤ 5% no matter how the tests correlate. The
cost is power — every extra thing anyone tried raises the bar for everyone.
That's the honest price of searching widely, paid in minimum detectable edge:

**Minimum detectable edge** (`minimum_detectable_edge`, per-bet pnl SD $0.50,
one-sided α=0.05 Bonferroni'd over 10 tests, 80% power; n = independent
event-clusters):

| n bets | MDE (cents/contract) |
|-------:|---------------------:|
|     50 |                 24.2 |
|    100 |                 17.1 |
|    200 |                 12.1 |
|    500 |                  7.6 |

Flipped around, to confirm a true edge of a given size you need roughly:
**10c → ~290 bets, 5c → ~1,170 bets, 2c → ~7,300 bets.** With ~1,010 settled
markets total (and only the post-06-20 slice usable for confirmation), a 2c
edge is unconfirmable on this data, 5c needs nearly every market to qualify,
and ~10c on a few hundred bets is the realistic detection target. A rule with
a plausible edge below its MDE gets "collect more data", never "ship it".

## Verdict criteria (frozen)

`SIGNIFICANT` requires **all** of:

1. **n ≥ 30 executed bets AND ≥ 20 distinct event clusters** on confirmation
   (otherwise `INSUFFICIENT_N` — no claim either way);
2. **p_adj < 0.05** — one-sided cluster-bootstrap p for "true pnl ≤ 0",
   times families × variants tested, capped at 1;
3. **cluster-bootstrap 99% CI lower bound on ROI > 0**.

**Degenerate (all-win / all-lose) samples — the exact boundary guard.** When
every confirmation bet wins, the cluster bootstrap can only resample wins: its
p pins at the resolution floor 1/(n_boot+1) and its CIs are conditional on
having seen zero losses — artifacts of n_boot, not evidence. The gate
therefore also runs an exact test under the least-favorable null on the H0
boundary (every bet's EV exactly zero): a claim bought at all-in cost c
(entry + fee, pays $1) wins with probability exactly c, and within-cluster
dependence is handled conservatively by treating each event cluster as a
single Bernoulli that wins with probability equal to the **maximum** cost
inside it (perfectly correlated bets win together at most as often as their
safest member). The statistic is the number of clusters whose bets all won;
`p_exact` is its Poisson-binomial upper tail, which for an all-win sample is
just the product of cluster max costs (`exact_allwin_p`) — e.g. 91 wins at
~0.956 across 51 clusters gives p ≈ 0.956^51 ≈ 0.10, nowhere near
significant however large n_boot is. The reported `p_value` is always
`max(p_boot, p_exact)`, the result carries `degenerate_boot` and
`ci_conditional_on_no_loss` flags, and a degenerate all-win sample must
additionally survive a flip-rate haircut for the ROI-CI gate to count: with
zero cluster losses observed in k clusters the one-sided 99% upper bound on
the per-cluster flip rate is 1 − 0.01^(1/k) (≈ 4.6/k), and
`flip_haircut_ev` — mean pnl per bet after flipping that fraction of
clusters to losses — must stay positive.

Also reported, read before celebrating: `es5_pnl` (mean total pnl of the worst
5% of bootstrap reruns — the unlucky-rerun loss), `max_drawdown`, and
`break_even_extra_cost_per_bet` — how many extra dollars per bet of slippage /
adverse fills / underestimated fees would zero the pnl. A "significant" edge
with a sub-1c break-even cushion is an execution accident waiting to happen.

Sizing for anything that passes: `kelly_capped_stakes` — Kelly on the fee-
inclusive entry price, hard-capped at 2% of bankroll, because a p that just
cleared this bar is still a suspect, not a fact.

## Failure modes that void a result (no exceptions)

- **Rule changed after peeking.** Any edit to the rule — threshold, league
  filter, side, timing — after any confirmation row was seen restarts the
  clock: the old confirmation window is burnt.
- **Fills assumed inside the spread without trade evidence.** Taker legs are
  priced at the touch (ask / 1−bid) plus taker fee. A maker-fill assumption is
  valid only with actual trade prints at that price during the window.
- **Fee omission.** Kalshi taker fees are ~1–1.75c at mid prices — larger than
  most edges we can detect. Every bet row carries its fee; a backtest without
  fees is not a backtest.
- **Survivor slicing post hoc.** "It works if you drop LPL" discovered after
  the confirmation run is a new variant: it increments the multiplicity count
  and needs its own untouched confirmation data. Same for date sub-windows,
  price bands, or "excluding that one weird event".
- **Undercounted multiplicity.** `n_families_tested` is everything the
  campaign tried (all ~10 families), `n_variants_in_family` is every variant
  this family peeked at in discovery — not just the ones that looked good.

`summarize_families` renders the cross-family league table and states how many
passes pure luck would produce, so the final report can't quietly forget the
denominator.
