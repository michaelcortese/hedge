# mention_hazard — in-event NO carry on word-mention markets

Strategy dossier and operating manual. Research provenance:
`data/research/edge-hunt-mentions-2026-07-16.md` (full campaign report,
pre-registration, OOS tables, blocked ledger).

## What it does

During a mention event (speech, hearing, game broadcast, briefing), markets
that haven't resolved YES yet are systematically overpriced: the crowd does
not decay P(YES) as the event runs out of road. The strategy emits
`Signal(prob = p_true)` with

    logit(p_true) = −0.24 + 0.93·logit(mid) − 1.28·τ

for unresolved markets with mid in [0.30, 0.70] and event-time fraction τ in
[0.25, 0.75]. The engine sees p_true << mid and buys NO.

## Execution constraints (load-bearing — the edge is maker-only)

- **post_only NO bids at the standing level.** OOS maker edge +10.2c/contract
  (p<1e-4); taker crossing is +4.9c at best (p=.03) and dies under causal
  event windows. Never cross.
- **Fixed clips ≤ 20 contracts per market.** Volume-weighted edge is only
  +2.4c: heavy flow through your level is informed (the mention is
  happening). Do not scale into flow.
- **Cancel discipline:** cancel resting bids at τ>0.75 or on event end.
- Fills are picked off at the mention moment by stream-watchers — that loss
  is already inside the +10.2c net number; clips keep it bounded.

## What the runner must supply

`event_starts: {event_ticker -> UTC start}`. Sources, in order of quality:
1. real-world schedule (game kickoff, hearing notice, speech schedule),
2. tape-burst onset detector (≥10 trades/15min across the event's markets —
   `scripts/research_inevent_hazard.py::onset_windows`, live-implementable).
The backtest is robust to (2) alone: +7.7c, p=.0078.

## Go-live gates (in order, per house rules)

1. **Paper 4 weeks** (`run_paper.py` + post-only fill simulation): kill if
   maker-fill P&L ≤ 0 or fill rate < 25% of signals.
2. Calibration guard (`guard.py`) with `max_brier` set from the backtest's
   in-band Brier (~0.16).
3. Demo-live with clip=5, political series only (strongest subset:
   hearings/Trump/Vance/Mamdani).
4. Prod only after demo shows positive realized maker P&L over ≥30 events.

## Capacity & expectations

~$300-400 notional/active-day at 100-contract caps; at clip=20 expect
roughly 3-6 fills/event, ~10c/contract, i.e. **$5-15/event, $30-80/week** in
current market conditions. This is a small, real edge — treat it as the
repo's second confirmed instance of the only edge class that has ever worked
here: being structurally faster than the crowd's repricing.
