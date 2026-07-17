# Edge hunt: Kalshi word-mention markets — 2026-07-16

Campaign: multi-round fan-out (8 blind approach families → data-earned tests
against settled prices → pre-registered OOS evaluation → 4-lens adversarial
audit). Dataset built for this hunt: **15,612 settled mention markets / 1,900+
events (2025-01..2026-07) with full trade tapes, hourly candles (partial), and
minute-level bid/ask books for all 1,284 signal windows**; 6,729 earnings-call
transcripts (115 tickers); 1,597 political transcripts (Trump/Vance/Leavitt).

## SURVIVOR (final, 2026-07-17): in-event taker-NO theta decay vs TRUE event
clocks — hearings-only certified (`mention_hazard`, r3 spec)

The one rule that cleared the pre-specified bar (p<0.01 at REAL NO asks,
event-clustered, n≥15 events) after three adversarial audit rounds:

- **Hearings taker-NO at T = true_end−60min** (external gavel/stream anchor +
  liveness check, mid 20-80c): **+10.79c/contract, p=0.0085, 89 markets / 15
  events** — at exactly the minimum n; smallest-viable, paper-first.
- **Pooled flow-gated variant** (any ground-truth family, enter only after a
  taker-YES print): +3.46c, p=0.0018, 265 events — significant pooled, no
  single family clears alone. NHL near-miss (+7.59c, p=0.011, 19 ev).
- Underlying mechanism certified universally: vs true elapsed fraction, gross
  NO edge on 20-80c still-open markets grows +11.2c (τ=0.5) → +22.3c (τ=0.9),
  p<1e-4 in all 6 ground-truth families. What varies by family is whether
  taker execution captures it — books are wide exactly where modeled edge is
  big (median YES spread at T: MLB 25c, NHL 32c, hearings 20c, WC 2c).
- **Calibration shipped in the strategy** (family-dummy Logit, cluster-robust,
  n=11,571 snapshots / 264 events, `scripts/research_r3_calibration.py`):
  `logit(p_true) = 0.112 + A_fam + 0.956·logit(mid) − 0.999·τ` with
  A_hearings=−1.168 (p=1e-4), A_NHL=−0.779 (p=4e-4); C p=2.3e-12. Model beats
  raw-price Brier in every family (hearings −0.072, NHL −0.054); reliability
  worst decile gap 1.4pp; conservative σ floor 0.30.
- Spec + go-live gates (paper 4wk → guard → demo → prod): `docs/MENTION_HAZARD.md`,
  `hedge/strategies/mention_hazard.py`. Capacity honest: ~$10-30/day.
- Caveats: single 9.5-week regime (May-Jul 2026); hearings at the 15-event
  floor with p just under the bar; earnings true-ends are estimates; speech
  families (85% of original signal volume) have no external clock — out of
  scope permanently.

Everything below is the full derivation trail: two killed headline rules,
the kill ledger, the anchor falsification that invalidated all pre-r3
backtests, and the re-anchored certification.

## Round-1 candidate — KILLED in audit round 1: in-event maker-NO hazard carry

**Mechanism.** Mention markets resolve YES the instant the phrase is said; NO
must survive the whole event. Mid-event, prices under-decay: retail holds and
chases YES lottery tickets while watching the event; NO carry is
capital-inefficient (risk 50-70c to win 30-50c in <2h), so pros don't correct
it. Counterparty: in-event YES takers (entertainment flow).

**Measured miscalibration** (event-clustered): at τ=0.3 of the event window,
unresolved 30-50c markets are priced 40.4c, resolve YES 22.0% (z=−3.7).
Calibrated correction: `logit(p_true) = −0.24 + 0.93·logit(price) − 1.28·τ`
(n=7,656, τ-coefficient p=4.6e-05).

**Pre-registered rule** (`data/research/mentions/PREREGISTERED_RULE.md`,
frozen before OOS data existed; formation = 89 newest events, OOS = all older
events): at τ=0.40, unresolved, tape price 30-70c → bet NO, hold to
settlement.

**Out-of-sample results** (95 events never seen in formation, minute-book
execution):

| Path | Windows | Mean/contract | CI95 | p(≤0) | n / events |
|---|---|---|---|---|---|
| naive (last-trade+2c) | expost | +10.4c | [+5.8,+15.1] | <1e-4 | 427/95 |
| naive, slip 3c | expost | +9.4c | [+4.8,+14.1] | 1e-4 | 427/95 |
| taker (cross real NO ask) | expost | +4.9c | [−0.2,+10.1] | .028 | 344/84 |
| taker | causal onset | −2.0c | [−6.8,+2.8] | .79 | 483/211 |
| **maker (post at level, fee-free)** | **expost** | **+10.2c** | **[+5.0,+15.3]** | **<1e-4** | **349/84** |
| **maker** | **causal onset** | **+7.7c** | **[+1.7,+13.7]** | **.0078** | **299/149** |

Maker robustness: day-clustered p=0.0008 (38 days); series-clustered p<1e-4
(9 series); leave-one-series-out all significant; positive in 9/9 series
(strongest: hearings +23c, Mamdani +17c, Vance +13c, Trump +9c; weakest:
World Cup +1.7c). Formation-sample consistent (+12.4c, p=0.0026).

**The claim is maker-only.** Crossing the spread (taker) is marginal at best
and dies under causal windows. Live execution = post-only NO bids at the
standing level (zero maker fee), fixed small clips.

**Adverse selection, measured:** volume-weighting fills by
contracts-printed-through drops the mean to +2.4c/contract — heavy flow is
informed. Hence fixed clips (10-50 contracts), never flow-proportional.

**Capacity:** ~$300-400 notional per active day at 100-contract caps.
Hobby-scale; disclosed, not a objection to significance.

**Honesty note:** the maker execution model was chosen after the taker OOS
result looked marginal. The signal (moments, side, thresholds) was frozen in
the pre-registration; the maker variant independently passes causal windows,
formation, and day-clustering.

**Falsification test (live):** paper-trade 4 weeks via the runner
(`mention_hazard` strategy, post-only, clip≤20): kill if realized
maker-fill P&L ≤ 0 or fill rate < 25% of signals.

**Repo plan:** `hedge/strategies/mention_hazard.py` (done) → paper via
`run_paper.py` with an event-schedule feed → calibration guard → only then
demo-live with `post_only`.

## Audit round 1 verdicts: maker-NO dossier KILLED (3 checkable kills)

All three kills converge on one defect: **the maker fill proxy grants
front-of-queue priority**. Through-volume is outcome-correlated — winners'
median at-or-through volume is ~175 contracts (thin voluntary YES flow),
losers' is ~4,563 (the mention sweep lifts everyone) — so a fixed clip fills
~100% of losers but only partially on winners. Clip-weighted re-derivation
(zero-queue upper bound): expost C=10/25/50/100 = +8.45/+7.43/+5.41/+2.45c
(p=.0016/.0039/.029/.21); causal-onset C=10/25/50/100 = +6.54/+4.62/+2.65/
+0.86c (p=.023/.087/.22/.40). The live-implementable configuration never
clears p<0.01 at any stated clip. Requiring K contracts to print through
before counting a fill: +10.17c (K=0) → +1.21c (K=50) → −7.47c (K=200).
The profits live in a thin-print tail a resting order never receives.

What the auditors confirmed clean: numbers reproduce exactly; pick-offs at
the mention moment ARE in the P&L; posting joins (not crosses); no lookahead
in the onset detector; timezones fine; not a sold-tail-risk artifact (payoffs
symmetric); clustering and multiple-testing robust *if fills were real*.
Additional flags: prereg verifiable only by file mtimes (commit to git next
time); OOS significance concentrated May 11–Jun 21, most recent 3 weeks flat;
86% of maker P&L from political series.

**Ledger entry 5 (checkable): in-event maker-NO at joined levels, clips
10–100 — killed on queue-corrected economics.** Reopening requires: live
paper fills measuring the real winner/loser fill asymmetry, or a
price-improving 1-lot variant (expost +8.45c p=.0018, onset +6.54c p=.021 —
below bar, capacity trivial).

## Audit round 2: late-event taker-NO theta carry

The tape-microstructure analysis (independent, blind to the killed rule)
surfaced a taker-side successor immune to the queue objection: 60–180 min
before event end, YES still 20–80c → buy NO taker (+9.63c net of fee + 4c
spread, CI [+6.47,+13.00], p<1e-4, n=1,111/185 events; earnings +20c,
sports +12c, speech n.s. excluded). Round-2 panel attacks: ex-post event-end
anchor (causal re-derivation), entry realism vs fresh minute books,
multiplicity of the scanned grid, temporal stability, post-WC supply.

### Round-2 verdicts (2026-07-17): rule-as-specified KILLED, mispricing confirmed

Two checkable kills, two survives. All four independently reproduced the
headline (+9.63c, n=1,111/185 ev) — the *mispricing* is real; the *rule* is
not executable as stated.

1. **fees-execution — KILL (checkable).** Real minute-candle books at 588/1,111
   signal moments (all 44 earnings signals + 206 freshly-fetched random
   broadcast signals + 312 archived): median top-of-book spread at signal
   minutes is 8c, p75=21c, p90=39c — the +4c allowance is wrong for most
   entries; 53% of signal prints sit outside the contemporaneous bid/ask.
   Luck-corrected repricing at true NO asks (the covered random sample ran
   outcome-rich): broadcast/sports/TV = **+2.0c, CI95 [−2.5,+6.5], p=0.19**;
   pooled non-speech +3.9c p=0.048, leaning entirely on 11 earnings events.
   Surviving variants: (A) earnings-only at real asks +18.57c [+5.22,+30.98]
   p=0.0047 (n=44/11 ev — tiny); (B) book-filtered taker — cross only if live
   NO ask ≤ (100−last_print)+4c — +17.81c [+6.62,+27.91] p=0.0008, keeps 41%
   of signals; **post-hoc filter, needs fresh OOS** (round 3).
2. **settlement-counterparty — KILL (checkable).** The trigger anchors on
   ex-post event end (max close_time). Every live-knowable anchor
   constructible from Kalshi data is hours off (onset+durations median error
   −1,400 min; rule net +0.44 p=0.32; walk-forward duration +1.00 p=0.27;
   schedule-prior +0.88 p=0.35; within-series end-TOD IQR 1–12h).
   `occurrence_datetime` = close_time+59s (settlement-stamped, lookahead).
   Decisive reality check: the 218k real taker-NO prints in the same
   window/band netted **+0.56c [−3.26,+4.38] vol-weighted** — actual late
   NO-takers, who face live anchor uncertainty, earned nothing. Survives only
   conditional on an EXTERNAL end estimate good to ±30–60 min: perfect anchor
   +9.70 [+6.45,+12.93]; −60min bias +3.82 [+1.48,+6.04].
3. **statistics — SURVIVES.** Grid is small; z=5.71 survives Bonferroni
   ×5000; adjacent cells monotone. All 3 months positive (May +7.19, Jun
   +10.65, Jul +10.51; last 14d +8.96 p=.002); drop-top-20-events +4.95
   [+1.84,+8.08] p=.0005; day-clustered p<1e-4; WC only 7% of markets.
   Flags: KXTRUMPMENTION (largest series, n=186) has NO edge (−1.29);
   trimmed sizing expectation is ~+5c, not +9.6c.
4. **generalist-refuter — SURVIVES.** Selection logic clean (1,110/1,111
   still open at T; zero post-close prints; median print age 1.6 min);
   reconciles against the independent harness exactly; ex-post anchor = a
   real scheduled halt in 100% of signal events; ±60min anchor shift keeps
   +3.96 p=.0003; repriced at real asks on its (non-random) covered subsample
   +8.15 p<1e-4 — budget a ~4c haircut.

**Blind replication (independent agent, original code unread):** REPRODUCES —
+9.64c, n=1,114/184 ev, CI [+6.40,+12.92], buckets and series splits match.
New spec ambiguity found: ~all signal markets have same-timestamp multi-level
batch prints, so "last print" tie-breaking moves the mean +8.8c..+10.5c (sign
and significance survive every tie rule; use the band, not the point, for
sizing). Also confirmed: the resolved-before-entry filter is vacuous; tape
file order is mixed (79 tickers oldest-first) — sort by timestamp, never take
file order. Script: `scripts/replicate_taker_rule.py`.

**Adverse-selection audit (independent agent, round 3): third checkable
kill, converging.** P&L is anti-correlated with proven fillability: 65.3% of
signals show ZERO taker-NO volume within the 4c limit in the first 5 min
post-entry; the fillable 385 net −0.26c [−6.17,+5.46] while the unfillable
726 carry +14.87c; size-weighted by real within-limit NO flow the rule is
−17.15c [−22.58,−11.81]. Winners' first post-entry print reprices −5.42c
against the fill (beyond the 4c allowance) before the first possible trade;
losers +1.29c n.s. Priced at real NO-taker executions: +3.20c (≤5min,
p=.067) / +4.38c (≤15min, p=.0074). Realistic capacity ~$10–30/day.
Cleared: NOT front-running the mention sweep (zero YES resolutions within
15 min of entry, q10=+42min); counterparty is retail taker-YES flow (68% of
signal prints; conditional P&L +12.81 vs +2.87 n.s. when last print was
taker-NO). Surviving variant (post hoc, needs OOS, still anchor-killed):
require last print taker_side=="yes", price at real executions → +5.60c
[+2.00,+9.09] p=.0013, n=561/137 ev; non-speech +6.50 p=.0003. Scripts:
`scripts/audit_adverse_selection*.py`.

**Ledger entry 6 (checkable): late-event taker-NO theta carry, as specified
— killed on (a) non-live-computable event-end anchor, (b) entry-cost
understatement, and (c) fill-conditioned adverse selection.** The intersection that survives both kills, unproven:
external-schedule families (earnings calls: known start + 60-90min duration;
clocked sports: game state gives end-time) + book-filter (cross only when NO
ask ≤ last_print+4c). Round 3 (in flight): pre-registered OOS of the book
filter on fresh never-fetched books; external-anchor accuracy validation vs
real historical game schedules.

## BLOCKED LEDGER (data-earned kills, binding)

1. **earnings-transcript-baserate** — market Brier 0.174 beats model 0.203 at
   T-24h; encompassing β=+0.22 p=0.18; threshold trading negative after fees
   (n=519/39 events). Kalshi balances earnings strikes to exactly 50/50 —
   strike selection is adversarial to public-transcript models. Basis:
   backtest (checkable).
2. **pre-event phrase-persistence** — resolution-history Beta-Binomial has
   real skill vs outcomes (AUC 0.78, n=9,149) but adds nothing beyond price
   at T-6h (encompassing β=+0.19 p=0.54); threshold rule −7c/contract. Basis:
   backtest (checkable, partial-sample n=157 priced markets — re-testable
   with full candles).
3. **pre-event calibration fades** — mid-bucket YES overpricing exists
   (50-65c bucket: −11pp gap, z=−2.3) but simple fades don't clear
   fees+spread (all fade rules n.s. or negative, n=793/59 events). Basis:
   backtest.
4. **event-basket NO overround** — sum(mid)−sum(YES) ≈ +0.57/event, but
   NO-basket at executable bids = −1.5c/contract (spread+fees). Basis:
   backtest.

5. **hearing-transcript model (round 3 scout)** — NO-GO. 534 settled
   KXHEARINGMENTION markets / 33 events: base rate 50.6% (Kalshi balances
   hearing strikes adversarially, same as earnings); market Brier 0.139 /
   AUC 0.90 already at the correct pre-hearing snapshot (hearing-day 12:00Z —
   beware: `occurrence_datetime` is settlement-stamped, naive T-1h snapshots
   are mid-hearing leakage). Walk-forward prototype (word priors, corpus DF
   over 222 govinfo CHRG transcripts, title match): OOS Brier 0.2483 vs 0.25
   base, AUC 0.506 — zero standalone skill; encompassing β=+0.29 p=0.48;
   threshold rules −3.4..−6.9c/ct after fees at zero slippage. Corpus note:
   govinfo CHRG is keyless and works (0.4s/req) but lags months;
   congress.gov API needs a key. Basis: backtest (checkable). Script:
   `scripts/research_hearing_model.py`.

## Round-3 anchor feasibility (external ground truth) — anchor FALSIFIED

Live end-time anchoring IS feasible with free external feeds: at T =
true_end−60, med |err| / share within ±30 / ±60 min — WC 2.7min/.91/.99
(kickoff+121m or elapsed-conditional), NBA 4.9/.96/1.00, NHL 5.3/.79/1.00
(OT tail), earnings 6.8/.98/1.00 (call start + 60min median duration), MLB
9.7/.95/1.00 (inning-half remaining via StatsAPI live feed; schedule-only
fallback 13.2/.82/.95), hearings ~.87 within ±60 (needs a convened/adjourned
liveness check — one hearing was cancelled and Kalshi still batch-closed it
2 days later), scheduled TV proxy-feasible, fights INFEASIBLE.

**But the same measurement falsifies every prior backtest's anchor:**
event_end = max close_time is administratively late — MLB median +776 min
after the real game end (4% within ±60), WC +509 (24%), NBA/NHL +182/+108.
Kalshi closes mention markets in batches hours-to-days after broadcasts; the
tape keeps trading (MLB last trade median +482 min post-game). So the
round-2 "60-180 min before end" windows often sat AFTER the real event —
mention impossible, NO a near-lock, 20-80c last prints stale — which would
also explain the entry-realism and adverse-selection kills (paper edge
concentrated where real fills were impossible). `occurrence_datetime`
placeholder = scheduled start + exactly 14 days for sports (settlement-
stamped; confirms Kalshi keys events to public schedules). Ground truth on
disk: `data/research/mentions/r3/*.csv`; script
`scripts/research_r3_anchor_feasibility.py`. **Re-anchored backtest against
true end times is the decisive test (in flight).**

## Round-3 re-anchored backtest — IN-EVENT EDGE CONFIRMED (mechanism alive)

Against true end times (289 matched events: WC 102, MLB 93, earnings 34,
NBA 26, NHL 19, hearings 15; script `scripts/research_r3_reanchor.py`):

- **Diagnosis of the killed rule's matched signals** (only 170/1,111
  matchable — 85% of the original set is speech families with no external
  clock): 72% fell genuinely in-event and earned +14.93c p<1e-4; post-event
  stale prints were ~20% of signals (higher per-contract, but not the bulk).
  The original edge was NOT primarily a post-event artifact.
- **Re-anchored rule** (T = true_end−60, last print ≤120min 20-80c, NO at
  (100−p)+4c+fee): **+6.83c [+4.68,+8.92] p<1e-4, n=2,078/265 ev**; −90:
  +5.02; −120: +0.38 n.s. Per family: NHL +25.65 (100/19), hearings +23.27
  (98/15), MLB +11.46 (665/92), NBA +3.15 n.s., WC +0.98 n.s., earnings
  +1.08 n.s. Anchor-error stress degrades gracefully (−30min shift: +5.02
  p<1e-4; −60: +0.38 n.s. pooled, but MLB +4.29/NHL +13.00/hearings +9.40
  stay significant even at −60).
- **Real minute-book asks** (covered subsample 823/6,312, selected):
  hearings +17.9c (significant), WC-covered +9.95 p=.002 (favorably
  selected), MLB +2.11 n.s. — ~11c book slippage kills MLB capture.
- **Mechanism certified**: theta-decay calibration vs TRUE elapsed fraction,
  20-80c still-open markets: gross NO edge +11.22c at tau=0.5 → +14.70 at
  0.7 → +22.33 at 0.9 (all p<1e-4, all 6 families positive at every tau,
  survives fresh-print restriction). The under-decay is real and universal;
  what varies by family is whether taker execution captures it.
### Full-coverage real-ask certification (2,200 books fetched, 82.6% quote
coverage at T): pooled rule FAILS, hearings PASSES

At real NO asks (100−yes_bid at T=true_end−60, +taker fee; event-clustered):

| set | n (ev) | real c/ct | p(≤0) | modeled same-set |
|---|---|---|---|---|
| ALL | 1717 (265) | +2.04 | .025 | +6.17 |
| hearings | 89 (15) | **+10.79** | **.0085** | +22.01 |
| NHL | 80 (19) | +7.59 | .011 | +25.86 |
| WC | 764 (102) | +2.79 | .032 | +1.37 |
| MLB | 457 (92) | −1.10 | .69 | +10.43 |
| NBA / earnings | — | +1.87 / −1.63 | n.s. | — |

The modeled pooled edge was slippage artifact concentrated in wide-book
families (median YES spread at T: MLB 25c, NHL 32c, hearings 20c vs WC 2c);
slippage vs the +4c model: NHL med +20.5c, MLB +14, hearings +9. 181
signals (MLB-dominated) had NO BOOK at T — no taker entry exists. Latency
flat at −60 (not a race). Pre-specified flow gate (last print taker-YES),
quasi-OOS here: kept 61.4% → **+3.46 real p=0.0018** vs skipped −0.23 —
significant pooled but no single family clears alone. Book gate kept
+2.78 p=.027. −90: everything fails except WC book-gated +4.20 p=.0087.

**Certified per the pre-specified bar (p<0.01 real asks, n≥15 ev):
hearings-only at −60 (+10.79c, p=.0085, exactly 15 events — smallest
viable, paper-first) and the pooled flow-gated variant (+3.46c, p=.0018).**
Script `scripts/research_r3_fullbook.py`; books
`data/research/mentions/minute_candles_r3full.jsonl`.

## Round-3 blind hunt (independent, told only the kill ledger)

Data corrections: 830 events (not 1,900+); tape coverage 2026-05-10..07-16
(~9.5 weeks, one regime); `occurrence_datetime` mis-populated by up to
−4,400h on earnings series.

- **k10 NO-flow-persistence carry** (clears the stats bar, fills
  print-proven): on a market's 10th NO-taker print at 15-50c YES, buy NO at
  that print's price (+fee), hold. Ex-sports +3.75c/ct, CI99 [+1.55,+5.89],
  p<1e-4, n=4,283/534 ev; 9/11 weeks positive; drop-top-5-events +3.12.
  Dose-response monotone in k (k=1 −0.50 → k=100 +4.00) at flat entry price:
  the signal is persistent NO-taker flow, not price level. Caveats: k and
  band chosen in-sample; weakest chronological split p=.043; capacity
  ~$10-20/day; likely the same underlying mispricing as the theta-carry
  family — convergent evidence, not a new class.
- **Post-resolution sibling fade — VALIDATED AND KILLED (2026-07-17):** the
  +10..+15c teaser was small-sample selection. Pre-registered rule (buy 1 NO
  on each still-open sibling at the real ask 2 min after a mid-event YES
  resolution), frozen before the targeted 580-request minute-candle fetch:
  **−2.53c/contract, CI99 [−15.4,+7.5], p(≤0)=0.73, n=91 markets / 25
  events** — fails the bar decisively. Matched non-triggered control +0.58c;
  paired trigger−control diff −2.41c (p=0.89): the trigger adds nothing.
  High-priced siblings (65-85c) are the disaster bucket (−34.7c, 0/11 NO
  wins) — post-resolution repricing is, if anything, correct-to-slow on the
  cheap strikes only. `scripts/research_sibling_fade.py`.
- Structure: sports-broadcast NO carry is NEGATIVE (MLB −6.8c p<1e-4 —
  announcers out-talk prices; a YES long is NOT the mirror, MLB YES-taker
  still −3.3c); entertainment/politics NO side profitable (LOVEISL +11.1,
  FIGHT +8.8, TRUMPMENTIONB +7.6, LATENIGHT +6.9); MAMDANI the one negative
  non-sports series. YES-taker mimicry loses −8..−12c at every age bucket
  and nearly every series.
- Nulls: no listing/first-print bias; no time-of-day pattern; count-strike
  box-arb inversions exist (76/1,737 adjacent pairs >3c) but need book
  snapshots to prove executability.
- Scripts: `scripts/research_blind_hunt*.py`; intermediates
  `data/research/mentions/blind/`.

## Thread status at campaign close (2026-07-17)

- Certified survivor: hearings taker-NO (+ flow-gated pooled variant) — see
  top section. Strategy re-specced, docs rewritten; next gate is 4-week
  paper trading with real-ask fill pricing.
- Sibling fade: killed by its own pre-registered test (above).
- OOS book-filter test (max_close-anchored): SUPERSEDED without running —
  the anchor falsification made its frame invalid; its question (does a
  book-quality gate rescue the pooled rule?) was answered on true anchors by
  the full-coverage test's book-gate variant (+2.78c p=.027 — no).
- GDELT news-flow nowcast: pilot still collecting at close (GDELT API
  ~1 timeline/30-40min; 4/15 word timelines done). Pipeline complete and
  resumable: `data/research/mentions/gdelt/` + collector; economic test
  pre-gated on an encompassing regression (news flow must add beyond price).
  No verdict — open route, not a survivor.
- Hearing-transcript model: NO-GO (zero standalone skill vs market by 8am
  ET — see blocked ledger).

## Assets for future hunts

- `data/research/mentions/`: events.jsonl (15,612 settled markets),
  trades.jsonl (full tapes), minute_candles.jsonl, candles.jsonl (hourly,
  partial — collector resumable), REGISTRY.md (8-family hypothesis registry),
  transcripts/ (earnings + political corpora with resumable scrapers).
- `scripts/research_*.py`: collectors, harness (cluster bootstrap,
  fee/execution accounting), all tests reproducible.
- Unexplored routes: hearing transcripts (congress.gov/C-SPAN) for a
  hearing-specific model; GDELT news-flow nowcast (needs price snapshots
  between listing and event — hourly candle backfill); WC/MLB broadcast
  in-event models; count-threshold (NegBin) markets.
