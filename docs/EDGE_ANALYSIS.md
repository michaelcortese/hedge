# Edge analysis: profiting in a sea of weather-arbitrage bots

*2026-07-01. Grounded in the current code (post "Dial 1", post tournament fixes #1–#3)
and the 2026-06-29 loss post-mortem. File:line references are to `main` at 6e07c17.*

## Premise

Everyone in these markets — the market maker, the arb bots, us — reads the same free
feeds: Open-Meteo GFS/ECMWF/ICON/GEM and `api.weather.gov`. The tournament verdict
stands: **there is no durable model edge in morning/next-day forecast bets**, and the
config already runs those at λ=0 (`deploy/config.yaml` `strategy_lambda`). What remains
contestable in a commoditized field is:

1. **Speed and fidelity to the settlement-relevant observation** — who knows the
   running max first, and how close their number is to what the CLI report will print.
2. **Settlement-instrument literacy** — trading the actual NWS Climate (CLI) product
   rather than a proxy of it.
3. **Model-free microstructure** — mispricings between related contracts that need no
   forecast at all.
4. **Not donating** — plugging the ways a slow bot leaks money to fast ones.

The risk framework (σ floor, settlement basis, event caps, skill gate, kill switch) is
in good shape; it protects an edge but doesn't create one. Everything below is about
creating or defending one.

---

## P0 — Fix the false-floor day-boundary bug (defends the flagship trade)

`nws_recent_temps_f` (`hedge/weather/providers.py:216`) requests observations starting
at **00:00 UTC** of the target date and then keeps *every* returned temperature —
the docstring's "keeps those whose timestamp falls on the target local day" is not
implemented. 00:00 UTC is ~7–8pm the *previous* local evening for all four cities, so
yesterday-evening temps pollute `obs_max`.

Failure scenario: a cold front passes overnight. Yesterday 8pm was 88°F; today's true
high is 74°F. `obs_max = 88` → the nowcast's deterministic path
(`hedge/strategies/weather_nowcast.py:58`) declares the 73–74 bucket "YES impossible"
and buys NO at `std_error=1e-6` — which sails through the significance gate, is
**exempt from the price band** (`hedge/decision/engine.py:304`), and sizes to the caps.
That is a guaranteed full loss on the one trade the whole strategy is built around,
bounded only by `max_order_dollars`/event caps (~$25).

Fix: filter each observation's timestamp to the station-local calendar day before
taking the max. One function, plus a regression test with a synthetic
previous-evening-warmer fixture. Also note the settlement climate day is
midnight-to-midnight **local standard time**, so during DST the boundary is 1am LDT —
worth matching exactly while in there.

## P1 — Win the observation race (the edge is measured in minutes)

The one durable edge is intraday obs lag: live observations move the true distribution
before the book re-prices. Our current effective latency throws most of that away:

| Stage | Delay |
|---|---|
| Hourly METAR cadence (obs taken ~:51) | 0–60 min |
| `api.weather.gov` feed propagation | ~5–15 min |
| Obs cache TTL (`providers.py:220`, `ttl=600`) | 0–10 min |
| Runner cycle (`--interval 900`) | 0–15 min |

Worst-case ~100 minutes from a temperature happening to us acting on it. A bot polling
the IEM 1-minute/5-minute ASOS feeds every minute sees the same event up to ~90 minutes
earlier. In a race for a capacity-limited riskless trade, we currently finish last.

Concrete upgrades, in order of value:

1. **Higher-fidelity max, not just faster polls.** Hourly METAR temps are
   *instantaneous* readings; the day's true running max usually happens *between*
   them. The official high comes from continuous 1-minute data, so `max(hourly obs)`
   systematically **understates** the real floor by ~0.5–2°F. That understatement is
   safe-side (we never call "impossible" wrongly because of it) but it means our
   deterministic trigger fires one price level and up to an hour later than the truth
   allows — i.e., after the fast bots have eaten the bid. Sources that fix it:
   - IEM 1-minute ASOS (`asos1min` service) — near-real-time for ASOS stations;
     validate availability per station (KNYC/KMDW/KMIA/KAUS).
   - METAR remarks: the 6-hourly max group (`1sTTT`) at 00/06/12/18Z gives the true
     6-hour max; the `T`-group gives tenths-°C precision instead of whole °C.
2. **Event-driven afternoon cadence.** In the 13:00–20:00 station-local window, poll
   observations every 1–5 min with `ttl≤60`, and run the decide/execute pass **only
   when a station's rounded `obs_max` ticks up** (or a bucket boundary is crossed).
   Outside that window the current 900s cycle is fine. This concentrates API budget
   and rate-limit budget exactly where the edge lives, without spamming orders.
3. **Evening coverage.** The deterministic set grows all evening while retail bids
   linger; make sure the deployed loop covers through late local evening for the
   westernmost cities rather than stopping at a fixed early `--until`.

## P1 — Trade the settlement instrument itself: parse the CLI product

Kalshi settles on the NWS Climatological Report (Daily) — the CLI text product. NWS
issues an **afternoon CLI (~4–5pm local) that prints the official high-so-far**, then a
final one after midnight. Between the afternoon issuance and market close, that printed
value *is* the settlement number unless the temperature makes a new high later — a
climatologically rare, quantifiable event (post-5pm new daily highs are a few percent
of days; fit the actual rate per city from the archive).

Most bots proxy settlement through raw obs; reading the CLI directly removes the last
basis between our floor and the paying instrument (rounding chain, tenths-°C
conversion, station quirks — the exact class of mismatch behind the 06-29 loss). The
products are free via IEM's AFOS API (`CLINYC`, `CLIMDW`, `CLIMIA`, `CLIAUS`). This
slots in as a second, authoritative `observed_max` source in
`hedge/weather/sources.py` — highest-trust floor when present, ASOS obs as the
fallback between issuances.

## P2 — Model-free event-strip arbitrage

The buckets of one city-day event are mutually exclusive and exhaustive, so a strip of
one YES in every bucket pays exactly $1. Whenever `Σ(yes_asks) < 1 − Σfees` the strip
is riskless profit; mirror with NO legs when `Σ(yes_bids) > 1 + Σfees`. Illiquid retail
books dislocate like this after obs shocks and overnight. This needs **no weather
model at all** — it's pure settlement-structure literacy, and it monetizes the *other*
bots' aggression (their one-sided taking is what skews the strip).

The engine can't see this today: `decide()` prices one market at a time, and the
runner picks the best single-market decision. It needs a small event-level pass in the
runner (it already groups per-event for `event_at_risk`, `hedge/runner.py`): sum the
book tops across an event's buckets, and when the strip clears total fees + a margin,
emit the multi-leg order set. Fees are the binding constraint — each taker leg pays
`~0.07·P(1−P)` — so most days it won't fire; when it does it's free money and, unlike
the nowcast, it's *anti*-correlated with being slow.

A softer variant with real capacity: renormalize our own bucket probabilities against
the whole strip (they come from one distribution, so they already sum to ~1) and
prefer the bucket where the *book's* implied strip is most out of line, not just the
best per-market edge.

## P2 — Stop donating to faster bots: stale-maker-quote guard

The engine prefers maker fills (`engine.py:275`) and the executor rests them
`good_till_canceled` (`hedge/execution/executor.py:78`). In the afternoon window that
is adverse-selection bait: our quote is priced off obs that are up to ~100 min stale
(see P1), so the moment a new METAR lands, faster bots lift exactly the resting orders
that just became wrong. Being maker "saves" the fee and pays a dollar to the fast.

Guard: whenever a station's `obs_max` ticks up (or a new CLI issuance lands),
immediately cancel-replace any resting maker order in that event before re-deciding.
The runner already reconciles and cancel-replaces stale rests (`runner.py:573`); this
adds "obs changed" as a staleness trigger, not just price drift. Alternatively:
maker only outside the fast window, IOC-taker inside it. This isn't new alpha — it's
plugging a leak that grows exactly as the market gets faster.

## P2 — Calibrate against the settlement station by default, and fresher

`_live_calibration` (`hedge/runner.py:62`) defaults to `truth="era5"` with a 45-day
window **ending 7 days ago** (ERA5 lag). Two paid-for-but-unused improvements:

- **Default `HEDGE_CALIBRATE_AGAINST=station`.** The IEM station truth path already
  exists (`fit_calibration(truth="station")`, `calibration.py:98`) and absorbs the
  grid→station basis into fitted bias/residuals (then `settlement_sigma` drops to 0 —
  a *sharper honest* σ, which is what wins the significance gate more often without
  lying). It was gated on validation against resolved settlements; the stations table
  is now 14/14 validated against CLI (`stations.py:52`), so run the remaining check
  (IEM max == Kalshi settlement over the resolved history) and flip the default.
- **Freshness.** IEM station data lags ~1 day, not ~5–7, so the fit window can end
  yesterday. That directly addresses the 06-29 finding #3 (fit regime 7.5°F cooler
  than realized in AUS). Add recency weighting (e.g. last 14 days ×2) so a regime
  shift moves the fitted bias inside days, not weeks.

## P3 — Capacity: validate more cities

The nowcast edge is real but thin and capacity-limited across 4 cities with depth and
participation caps. Kalshi lists more daily-high series (Denver, Philadelphia, LA at
minimum). Each added, *validated* city scales the same edge roughly linearly with the
same code. The only hard part is the settlement-station map — the #1 correctness risk
— and `scripts/validate_stations.py` already automates the check against resolved
settlements. Rule stays: no size until 14/14 exact-bucket validation, `validated=True`
only from evidence.

## Where NOT to fight (keep the discipline)

- **Morning forecast bets stay at λ=0** until `run_paper.py skill` shows the acted-on
  Brier beating the market mid out-of-sample over ≥30 settled trades. The skill gate
  (`guard:` config) enforces this; don't hand-raise λ around it.
- **Don't out-model the consensus.** Any "better blend" of the same free feeds is
  re-deriving the mid and paying spread+fee for the privilege.
- **Expected size honesty.** With ~$50–100 bankroll, four cities, and per-event caps,
  even a perfectly-executed obs edge is single-digit dollars per day. The value of
  this phase is *proving the process* (paper skill → ramped λ) — the edges above are
  the ones that survive scaling capital and competition, because they're structural
  (speed, settlement literacy, strip identity), not statistical.

## Priority summary

| # | Item | Type | Effort | Why |
|---|---|---|---|---|
| P0 | Local-day filter in `nws_recent_temps_f` | bug fix | S | Guards the flagship deterministic trade from a guaranteed-loss false floor |
| P1 | 1-min/6-hr-max obs + event-driven afternoon cadence | speed/fidelity | M | The core edge is measured in minutes; we currently run ~last |
| P1 | Parse afternoon CLI product as authoritative floor | settlement literacy | M | Trade the instrument that pays, not a proxy of it |
| P2 | Event-strip sum arbitrage pass | model-free | M | Riskless when it fires; monetizes other bots' aggression |
| P2 | Cancel-replace maker rests on obs tick | leak plug | S | Stop being the stale quote fast bots pick off |
| P2 | Station-truth calibration by default + recency weighting | calibration | S | Sharper honest σ, regime-fresh bias; code already exists |
| P3 | Validate + enable more cities | capacity | M | Thin edge × more venues; the map is the only risk and it's automated |
