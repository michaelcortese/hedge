# mention_hazard — in-event NO theta-decay on word-mention markets (r3 spec)

Strategy dossier and operating manual. Research provenance:
`data/research/edge-hunt-mentions-2026-07-16.md` (three audit rounds, kill
ledger, re-anchoring, full-book certification). This file supersedes the
pre-r3 maker spec, which was killed in audit round 1.

## What it does

During a mention event, still-open markets under-decay P(YES) as the window
shrinks: against TRUE event clocks, gross NO edge on 20-80c markets grows
+11c → +22c as elapsed fraction goes 0.5 → 0.9 (event-clustered p<1e-4 in
all six ground-truth families). The strategy emits `Signal(prob = p_true)`
with

    logit(p_true) = 0.1117 + A_fam + 0.9564·logit(mid) − 0.9990·τ_true

(A_fam: hearings −1.168, NHL −0.779, others 0; fit n=11,571 snapshots /
264 events, C p=2.3e-12, cluster-robust; model beats raw price Brier in
every family). The engine sees p_true << mid and buys NO **as taker**.

## The three load-bearing lessons from the audits

1. **Never anchor on Kalshi close_time.** It is an administrative batch
   time, hours after the real event (MLB median +776 min). τ must come
   from a REAL event window supplied by the runner (`EventWindow`), built
   from external feeds. Anchor recipes and measured accuracy at end−60:
   - hearings: committee/YouTube stream status + gavel time, ~87% within
     ±60 min — **must include a convened/adjourned liveness check** (one
     sampled hearing was cancelled; Kalshi still batch-closed it 2 days
     later).
   - NHL/NBA: ESPN live remaining-time by period (med err ~5 min).
   - WC: kickoff + 121 min (med err 2.7 min). MLB: StatsAPI inning-half
     remaining (9.7 min). Earnings: call start + 60 min (6.8 min).
   Edge degrades gracefully with anchor error: −30 min systematic error
   keeps +5.02c pooled; −60 min kills all but MLB/NHL/hearings.
2. **Never model a fill better than the live book.** Books are wide
   exactly where modeled edge looks biggest (median YES spread at end−60:
   MLB 25c, NHL 32c, hearings 20c, WC 2c); 181/2,078 signals had NO book
   at all. Maker capture is separately killed (queue adverse selection:
   winners fill thin, losers get swept). Taker at the real NO ask
   (100−yes_bid) is the only honest execution model.
3. **The edge is not a race** — P&L is flat to +5 min of latency at
   end−60. It dies ~2h+ before the true end (−120 min entry: +0.38 n.s.).

## Certification status (pre-specified bar: p<0.01 at real asks, ≥15 events)

| slice | real-ask P&L | p | n | status |
|---|---|---|---|---|
| hearings, end−60 | +10.79c/ct | 0.0085 | 89 mkts / 15 ev | **CERTIFIED** (at minimum n — smallest viable) |
| pooled flow-gate (enter only after a taker-YES print) | +3.46c | 0.0018 | ~1,054 / 265 ev | certified pooled; no single family clears alone |
| NHL, end−60 | +7.59c | 0.011 | 80 / 19 ev | near-miss — enable only after paper support |
| WC | +2.79c | 0.032 | 764 / 102 | not certified |
| MLB / NBA / earnings | −1.1 / +1.9 / −1.6 | n.s. | | dead at real asks |

Default enablement in the strategy is therefore **hearings only**.

## What the runner must supply

`event_windows: {event_ticker -> EventWindow(start, end_estimate,
confirmed_live)}` from external feeds (see recipes above). No window or
`confirmed_live=False` → the strategy abstains. Signals fire only for
τ ∈ [0.40, 0.92] and mid ∈ [0.20, 0.80].

## Go-live gates (in order, per house rules)

1. **Paper 4+ weeks** (`run_paper.py`, hearings only, taker fills priced
   at the real NO ask at signal time): kill if realized P&L ≤ 0 or if
   fewer than ~8 hearing events accrue (underpowered → extend, don't arm).
2. Calibration guard (`guard.py`): `max_brier` from the fit's hearings
   in-band Brier (~0.175).
3. Demo-live, clip ≤ 10 contracts/market, hearings only.
4. Prod only after demo shows positive realized P&L over ≥15 events.
   NHL second, WC/flow-gate variant only with a fresh certification.

## Capacity & honest expectations

Hearings-only is small: ~1-3 hearing events/week in season, ~6 signals
per event, ~$10-30/day at retail clips. The flow-gated pooled variant
adds similar scale. This is the repo's edge class (react to public
information — the event clock — faster than the crowd reprices), at
hobby scale. The certified numbers are one 9.5-week regime (May-Jul
2026); the paper gate exists because 15 events is the floor, not proof
of stationarity.

## Known caveats (carried from audits)

- Speech/presser families (Trump, Mamdani, Vance, press briefings) have
  no external clock — permanently out of scope for this strategy.
- Fights: end-time anchoring infeasible. Scheduled TV: proxy-only.
- Earnings true-ends in the fit are duration estimates (words/150wpm).
- `occurrence_datetime` in Kalshi payloads is settlement-stamped
  (lookahead) — never use it ex-ante.
