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
| C | series-underreaction | Bo3/Bo5 match repricing after game results vs recursion benchmark + momentum | **PROMISING, frozen, INSUFFICIENT_N**: post-g1 gap -1.6c favorites (CI excl 0), momentum wedge +4.5c borderline; frozen rule (t_end+3min, ask < recursion-3c) confirmation 14/17, ROI +18.4%, p=0.097, n=17 < gate. Needs n: siblings (C2) + maps (K2) |
| D | stale-early | 24h->1h price-discovery window staleness; model-vs-early test | **REAL, NOT EXECUTABLE at touch**: early mids provably stale (Brier +0.012-0.018, CI excl 0); model incremental coef +0.41 [+0.105,+0.755] at 10-24h, zero by t5; edge 2-4c < spread+fee. Maker variant -> E3 |
| E | liquidity-provision | two-sided quoting inside ~5c spread; conservative print-based fills; maker fee 0 | **DEAD**: symmetric & model-anchored quoting adverse-selection-dominated in all 24 cells; in-play quoting toxic (-4.4c/fill); thin-fav tilt was look-ahead noise; confirmation -36% ROI n=8 |
| E2 | maker thin-favorite | targeted: H's low-volume underconfidence captured with maker fills | **INSUFFICIENT_N, likely decaying**: slope reproduced (1.67) but May +8.7c -> Jun -0.8c; 6 confirmation fills, p_adj=1.0 |
| E3 | informed early maker | one-sided model-side quoting in the 8-24h stale window | **DEAD-no-burn**: fills are informed AGAINST us (P(outcome|filled) 0.445 vs model fair 0.597); static model rots (Brier .186 May -> .242 Jun). Closes the static-model-anchor branch |
| M | timed decisive-drift | LP-clock entry +2min after decisive game end, buy winner at ask<=0.97, bid>=0.85, 5 ex-ante gates | **PASSES SIGNIFICANCE, INSUFFICIENT_N**: confirmation n=17/10 clusters, ROI +5.7%, p_adj 0.029, ROI CI99 lower +3.0%; 62/62 wins pooled; needs 30/20 -> M2 |
| N | draft-window info | draft-delta model (OE picks 2020-2025) vs market at draft lock | **DEAD-no-burn**: draft model has real historical signal (+0.0115 LL) and market ignores draft window (~1.7c movement), but increment-vs-price CI straddles 0; ~2c implied edge, 10-15x below MDE |
| K | model-vs-map-markets | game model vs KXLOLMAP per-game books | **INSUFFICIENT_N, dead-leaning**: map book softer (+0.021 deficit vs +0.066) but beats model; vol-low blend credible in discovery, confirmation negative (n=28) |
| M2 | frozen M-rule on maps | same frozen rule, per-map settlements | 74/74 wins, formally SIGNIFICANT — **then REFUTED by audit** (see below) |
| AUDIT | 3 adversarial auditors on M/M2 | stats, execution, data integrity | **NET: REFUTED.** A1(stats): bootstrap p degenerate at 100% hit (floor artifact); honest exact clustered p 0.016-0.14, fails multiplicity; map evidence shares 76% of games with tuning; fire rate collapsed post-06-20. A2(execution): UPHELD-w-conditions — fills real (0/91 phantom), timing costs n not wins, capacity 10-25 lots/bet. A3(data): FATAL — outcome-conditioned side + pause-blind LP clock (12.4% of games end >2min after LP end) books hindsight wins and skips phantom losses (found one: loser bid 0.87 -> 0.03); honest ex-ante replay: confirmation window -$0.02 on 13 physical games; 91 bets = 56 physical games (mirror double-booking) |
| M3 | cross-market clock (siblings) | sibling map-book snap (>=0.97 bid) as the game-end clock for match-book drift on CS2/VAL/Dota; side ex-ante (the side that snapped) — no look-ahead possible by construction | round 5 — the last live candidate; pre-registered before sibling data touched |
| F | cross-market-structure | game/map/totals/outright consistency; sibling esports | **DEAD as arb** (books coherent; 1c fee floor kills residuals). **Data win: ~18k settled markets across 13 series; KXLOLMAP n=1,988 per-game markets** |
| G | join-rescue + selection QA | | **DONE**: matched 970/1,010 via verified alias table; headline firmer (+0.066 logloss, CI [+0.043,+0.090]) |
| H | market-only-residual | drift/flow/round-number/composite | **DEAD** except lead: low-volume calibration slope 1.56 [1.22,2.03], thin favorites +2-5c gross, fails at touch by ~2c -> handed to E2; possible May->June decay |
| I | in-play | volume clock, in-play efficiency, resolution-drift capture | **DEAD as price-trigger** (confirmation n=74, ROI -0.03%): meat is real (~$30-40k/6wk fee-adj capacity) but lives in first ~120s after decisive game end -> requires external game-end clock -> family M |
| J | stat-protocol | acceptance bar | **DONE**: edge_protocol.py; MDE: 10c needs ~292 bets, 2c unconfirmable on LoL-only; pooled multi-series data changes this |
| K | model-vs-map-markets | game-level model vs KXLOLMAP per-game books (thinner, derivative) | queued (needs multi fetch) |
| L | pooled thin-market | H's slope test pooled across 13 esports series (~18k markets) | queued (needs multi fetch) |

Verdicts are appended below by the orchestrator as rounds complete.

## M3 pre-registration (written 2026-07-15 ~18:00 UTC, BEFORE any sibling microstructure examined)

Universe: KXCS2GAME, KXVALORANTGAME, KXDOTA2GAME settled match markets whose
fixture has same-fixture per-map markets (KX*MAP) in the fetched data.
Clock: for each fixture, track map-book settlements in sequence; the DECISIVE
moment is the first minute any map-book's bid >= 0.97 such that that side's
accumulated map wins clinch the match under the fixture's format (inferred
from the number of map markets listed, NOT from outcomes). The clinching
side at that moment is the ex-ante side. No Leaguepedia/external clock.
Entry: at clock + 2 minutes, buy the clinching side of the MATCH market at
the minute-candle ask if ask <= 0.97 and bid >= 0.85. One bet per PHYSICAL
fixture (no mirror double-booking). Taker fee. Hold to settlement.
Gates (all ex-ante): liveness (>=1 match-book print within +-10 min of clock);
map-book snap must be genuine (>=2 consecutive minutes bid >= 0.95); skip if
the match-book bid for the clinching side < 0.85 (defensive, as frozen in M).
Statistic: evaluate_frozen_rule with the exact/Poisson-binomial clustered p
(post-audit fix), clusters = fixture, stress-clustered by tournament-day;
n_families_tested = 15, n_variants_in_family = 24 (inherited; nothing tuned
here). Flip haircut: EV must remain positive at the rule-of-three 95% upper
bound on the observed loss rate per CLUSTER.
Decision rule (pre-committed): SIGNIFICANT verdict from the fixed evaluator
AND positive flip-haircut EV AND no audit-3-class timing contradiction
(entry bar must not contain match-book prints < 0.80 for the clinching side)
=> claim stands, subject to one final independent audit. Anything else =>
campaign reports the fallback (strongest derivation + exact gap). No retuning,
no second look, regardless of outcome.
