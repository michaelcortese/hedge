# Edge hunt: Kalshi word-mention markets — 2026-07-16

Campaign: multi-round fan-out (8 blind approach families → data-earned tests
against settled prices → pre-registered OOS evaluation → 4-lens adversarial
audit). Dataset built for this hunt: **15,612 settled mention markets / 1,900+
events (2025-01..2026-07) with full trade tapes, hourly candles (partial), and
minute-level bid/ask books for all 1,284 signal windows**; 6,729 earnings-call
transcripts (115 tickers); 1,597 political transcripts (Trump/Vance/Leavitt).

## SURVIVOR: in-event maker-NO hazard carry (`mention_hazard`)

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

_(verdicts appended below when the panel returns)_

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
