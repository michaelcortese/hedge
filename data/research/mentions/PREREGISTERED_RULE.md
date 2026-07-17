# Pre-registered rule: in-event NO carry on mention markets

Frozen at 2026-07-16 ~10:50 CT, before out-of-sample data (older events,
arriving via newest-first tape collection) was seen.

## Rule (exact)

For a mention market in series with a defined event window W (see
`research_inevent_hazard.py::WINDOWS`):

- Let event_end = max close_time across the event's markets,
  event_start = event_end − W.
- At τ = 0.40 of the window (single evaluation point per market),
  take the last tape trade price p in τ ∈ (0.25, 0.40].
- If the market is unresolved at τ and p ∈ [30c, 70c]:
  **buy NO at (100 − p) + 2c slippage**, hold to settlement.
  Fee: standard taker formula on the NO price.

## Hypothesis

Mean P&L per contract > 0 (one-sided), cluster-bootstrapped by EVENT.
Success bar: p(≤0) < 0.01 on out-of-sample events (settled before the
formation window), with ≥40 out-of-sample events; and the mean survives
slippage 3c.

## Formation sample

The 89 event tickers in `formation_events.json` (newest events, tapes
collected first). All events NOT in that list are out-of-sample.

## Mechanism claimed

Retail holds YES lottery tickets during the event and under-reacts to the
passage of event time (hazard decay); NO-side carry is capital-inefficient
(risk 60-70c to win 30-40c in <2h), so pros don't fully correct it. YES
resolves instantly on mention; NO must survive to event end — the market
prices the *unconditional* pre-event probability too long into the event.
