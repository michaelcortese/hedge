# Tape microstructure in Kalshi word-mention markets

Analysis of raw trade tapes for **15,482 settled mention markets** (828 events,
5,580,612 trades, settlements 2026-05-10 → 2026-07-16). Script:
`scripts/research_tape_micro.py` (full log: `tape_micro_full.log`).
All significance is **cluster-bootstrapped by event_ticker** (4,000 reps,
whole-cluster resampling). Fees: taker `ceil(0.07·C·P·(1−P))` cents, maker
free. "c/ct" = cents per contract. p-values are one-sided bootstrap p(mean≤0).

**Data caveat discovered en route (affects any analysis of these markets):**
`close_time` is *event-driven* — trading halts when the word is said, so YES
markets close earlier than NO markets in the same event (mixed-outcome events
share one close_time only 0.9% of the time). Any window anchored on a
market's own `close_time` peeks at the future. Implementable rules must
anchor on the **event end** (max close_time across the event ≈ scheduled
broadcast/game end); all headline numbers below do.

---

## Family 1 — fade vs follow taker bursts: **REVERSION (maker-only edge)**

Bursts = |net signed taker flow| ≥ K contracts within 5 min, 15-min cooldown,
>10 min to close. Markout = dir·(settle − price).

| K | n bursts (ev) | follow→settle | follow→30m tape | maker fade (fee-free) |
|---|---|---|---|---|
| 25 | 315,164 (828) | **−2.41** [−2.91,−1.95] p<1e-3 | −0.20 [−0.27,−0.13] | **+2.41** [+1.95,+2.91] p<1e-3 |
| 50 | 252,464 (828) | −1.94 [−2.40,−1.52] | +0.01 | +1.94 |
| 100 | 185,245 (828) | −1.63 [−2.08,−1.20] | +0.15 | +1.63 |

Strongly **asymmetric**: taker-YES bursts revert hard (follow = −4.67
[−5.64,−3.73], n=206k), taker-NO bursts continue (+1.88 [+1.08,+2.68]) —
NO flow is informed (word-not-said decay), YES flow is dumb money.
Fading taker-YES bursts by buying NO (5–95c): **as maker +5.74 [+4.54,+6.93]
p<1e-3**; **as taker −0.84 to −1.77 (dead)** — fee + spread eats it. The
reversion realizes at settlement, not within 30 min, so you cannot scalp it;
you must hold. Verdict: **edge, capturable only passively**.

## Family 1b — maker markout when filled by taker flow (mid-event focus)

Volume-weighted, prices 3–97c, fee-free, by time-to-(own-market)-close:

| window | maker sells YES into taker-YES | maker buys YES from taker-NO |
|---|---|---|
| <10 min | +4.72 [+2.89,+6.36] | −2.42 [−6.04,+2.46] |
| 10–60 min | +1.27 [−0.57,+3.03] p=.084 | −1.30 [−3.31,+0.75] |
| 1–6 h | +1.73 [−0.66,+3.89] p=.070 | +0.25 n.s. |
| 6–24 h | **+8.33** [+6.37,+10.13] | **−6.85** [−9.37,−4.34] |
| >24 h | +5.51 [+3.00,+7.93] | −6.00 [−10.05,−2.35] |

Mid-event (10 min–6 h) by price bucket: +0.9…+2.7 (marginal) at 3–80c, but
**−1.78 [−3.19,−0.25] at 80–97c** — a maker resting YES-asks above ~80c
during the event window is systematically picked off by informed
word-was-just-said flow. A NO-accumulating maker (sells YES) is fine pre-event
and at low/mid prices mid-event, but must pull high-price YES offers once the
event is live. Symmetrically, resting YES *bids* (filled by taker-NO flow)
lose everywhere except mid-event mid-range. (Selection caveat: markouts are
conditional on fills that actually happened.)

## Family 2 — last-trade calibration near close

Anchored honestly on **event end**, mid prices late in the event are grossly
too high: markets whose last print 10–180 min before event end is 20–80c
resolve YES only **7%** of the time (vs ~40c price); 60–180 min out, **28.4%**
vs 43.9c. At >6 h before close calibration is decent with a mild
favorite–longshot tilt: buy NO ≤15c +0.72 [+0.17,+1.22] p=.005, buy YES ≥85c
+0.95 [+0.21,+1.65] p=.006 — both under 1c, **not tradable** after spread.
Verdict: **big bias exists, only intra-event** (see 2b).

## Family 2b — THE strongest tradable pattern

**RULE: 60–180 min before the scheduled event end, if YES still trades
20–80c, buy NO (taker), hold to settlement.** Entry modeled at
(100−p)+4c spread, taker fee charged.

- n=1,111 markets, 185 events: **net +9.63 c/ct, CI95 [+6.47,+13.00],
  p<1e-4** (~19% return on ~60c capital, hours-long hold).
- By price: 20–40c +6.12 [+2.49,+9.46]; 40–60c +12.95 [+7.70,+18.14];
  60–80c +14.68 [+7.08,+22.66].
- By family: earnings +20.25 [+9.24,+31.83] (11 ev); broadcast/sports/TV
  "other" +11.79 [+8.30,+15.45] (151 ev); **speech +1.95 n.s. — exclude
  Trump-speech series** (open-ended windows).
- Tighter window [10,60) min: +22.62 [+12.51,+31.07] (n=47, YES rate 8.5%).
  Earlier [180,360) min: +2.38 [+0.30,+4.54] — the edge is theta decay the
  crowd doesn't price: YES holders don't mark down as the window shrinks.
- Capacity: ~16.7 qualifying markets/day; median tape volume in the final 3 h
  ≈ 608 contracts/market → realistically ~50–100 contracts/market without
  moving price ⇒ **roughly $100–200/day expected at retail size**, more as
  maker (add back fee + spread ≈ +5c/ct).
- Live requirement: an event-end estimate (broadcast schedule / game clock),
  seconds-latency is fine.

## Family 3 — taker-side imbalance: **no informed YES flow; fade it**

Encompassing logit `result ~ logit(p_last) + imbalance` (event-clustered SE,
3–97c markets): imbalance adds nothing at 60 min (coef +0.03, p=.79) and is
**negative** at 180 min (coef −0.15, p=.008, n=9,216/656 ev) — YES-heavy
taker flow predicts *lower* P(YES) given price. Economics: FOLLOW extreme
imbalance (|imb|≥0.6, ≥60 min horizon) = −6.91 [−8.75,−5.11]; FADE = +3.88
[+2.08,+5.69] p<1e-3 (fee-in, fill at last price ≈ maker-ish). Consistent
with families 1/1b: retail YES buyers are the dumb money.

## Family 4 — effective spread + capacity

Median effective spread (trade-to-trade bounce on taker-side flips) 2.5–12c
by series; Roll estimates agree. Median trade 6–14 contracts. Volume:
870–29,500 contracts/market-day. Naive maker capture (¼ of volume × ½
spread): KXLOVEISLMENTION $204/mkt-day, KXTRUMPMENTIONB $138, KXMAMDANI $111,
KXTRUMPMENTION $97, KXWCMENTION $41 — order **$500–1,000/day gross across the
board**, *before* the 1b adverse-selection adjustment (positive on the
sell-YES side except 80–97c mid-event, negative on the buy-YES side pre-event).
Practical maker book: quote both sides far from events, skew hard toward
selling YES, pull YES asks ≥80c once the event is live.

## Family 5 — anomalies

- **1,288 trades print AFTER close_time** (808 markets) — halt-timing
  artifact; do not trust close_time as a hard trading deadline.
- 53.8% of trades have fractional contract counts; 0 block trades.
- 35.6% of timestamps are multi-print sweeps (max 502 prints in one stamp) —
  bursts are single-taker sweeps, not crowds.
- Taking the favorite at ≥98c/≤2c (11.9% of all prints!) wins 99.5% but nets
  **−0.25 [−0.40,−0.13]** after fee — the penny-scalp crowd donates to the
  makers selling 99c.

## Caveats

Single season (May–Jul 2026), composition heavy in World Cup/MLB/Trump
series; outcomes have no survivorship (all settled markets collected).
Markout-based maker P&L is conditional on historical fills (depth unknown —
prints only). Family-2b live implementation needs an event-end estimate;
the [60,180) window result is robust to excluding the largest event and is
monotone across price buckets, months, and families except speech.
