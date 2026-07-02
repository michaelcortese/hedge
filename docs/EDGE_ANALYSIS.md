# Edge analysis: profiting in a sea of weather-arbitrage bots

*2026-07-01. Grounded in the current code (post "Dial 1", post tournament fixes #1–#3)
and the 2026-06-29 loss post-mortem. File:line references are to `main` at 6e07c17.*

*Round 2 added 2026-07-01 PM after a code-level audit plus live probes of the Kalshi
public API, IEM, and NWS: the three S-effort defenses (P0 day-boundary fix, all-day
deterministic window, deterministic-crosses-spread) are **implemented on this branch**,
two round-1 claims are corrected (1-min ASOS is archival; capacity is ~4×, not +3
cities), and the priority table at the bottom is the merged, current one.*

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

## P0 — FIXED: the false-floor day-boundary bug (defends the flagship trade)

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

**Fixed (this branch).** `nws_recent_temps_f` now keeps only observations inside the
station's climatological day — midnight-to-midnight **local standard time**, i.e. a
1am boundary on the local clock during DST — and additionally drops readings whose
temperature failed NWS quality control (rejected `X` / questioned `Q`), closing the
*other* false-floor path nobody had listed: a single glitched sensor spike.
Regression tests in `tests/test_weather_providers.py` (previous-evening cold front,
DST boundary, next-day exclusion, QC, malformed rows).

## P1 — Win the observation race (the edge is measured in minutes)

The one durable edge is intraday obs lag: live observations move the true distribution
before the book re-prices. Our current effective latency throws most of that away:

| Stage | Delay |
|---|---|
| Hourly METAR cadence (obs taken ~:51) | 0–60 min |
| `api.weather.gov` feed propagation | ~5–15 min |
| Obs cache TTL (`providers.py:220`, `ttl=600`) | 0–10 min |
| Runner cycle (`--interval 900`) | 0–15 min |

Worst-case ~100 minutes from a temperature happening to us acting on it. A bot on a
genuinely real-time sub-hourly feed (MADIS/Synoptic 5-min ASOS, or just SPECIs + tight
METAR polling) sees the same event up to ~90 minutes earlier. In a race for a
capacity-limited riskless trade, we currently finish last.

Concrete upgrades, in order of value:

1. **Higher-fidelity max, not just faster polls.** Hourly METAR temps are
   *instantaneous* readings; the day's true running max usually happens *between*
   them. The official high comes from continuous 1-minute data, so `max(hourly obs)`
   systematically **understates** the real floor by ~0.5–2°F. That understatement is
   safe-side (we never call "impossible" wrongly because of it) but it means our
   deterministic trigger fires one price level and up to an hour later than the truth
   allows — i.e., after the fast bots have eaten the bid. Sources that fix it:
   - ~~IEM 1-minute ASOS (`asos1min` service) — near-real-time~~ **Corrected
     (verified 2026-07-01): the IEM 1-min feed is archival.** All four stations
     return 1-min rows for 2025 dates but nothing for yesterday (nor does DSM, IEM's
     home station) — it lags days-to-months, so it cannot win the live race. It is
     still gold for *fitting*: per-city hazard curves P(new high after hour h),
     boundary-crossing rates, and validating the intraday backtest replay.
   - METAR remarks (real-time via api.weather.gov `rawMessage` or IEM's live METAR
     feed): the 6-hourly max group (`1sTTT`) at 00/06/12/18Z gives the true 6-hour
     max — the 00Z group covers the whole afternoon at tenths-°C precision; the
     `T`-group gives tenths-°C on every hourly instead of whole °C.
2. **Event-driven afternoon cadence.** In the 13:00–20:00 station-local window, poll
   observations every 1–5 min with `ttl≤60`, and run the decide/execute pass **only
   when a station's rounded `obs_max` ticks up** (or a bucket boundary is crossed).
   Outside that window the current 900s cycle is fine. This concentrates API budget
   and rate-limit budget exactly where the edge lives, without spamming orders.
   Two mechanism notes from the code audit: persist a per-(station, climate-day)
   **obs high-water mark** in `state.py` so the floor is monotone across cycles (a
   feed dropout can't lower it) and "obs ticked up" is a well-defined trigger; and
   **phase-lock polling to the METAR minute** — obs land ~:51–:57, so a
   :00/:15/:30/:45 grid reads a fresh METAR up to 9 minutes late. Polling at ~:58 is
   a free latency win before any new data source.
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

**Verified 2026-07-01** via IEM AFOS (`/api/1/nws/afos/list.json?pil=CLI…` to list,
`/cgi-bin/afos/retrieve.py?pil=CLI…` to fetch, no auth): on 06-30 every city issued an
intraday CLI in the ~4:40–5:45pm local window (20:40Z NYC & MIA, 21:45Z CHI & AUS —
AUS also got a 12:26Z morning issuance and a 22:43Z correction) plus the final at
~2:20–3:30am. The product prints the official max **and the time it occurred**
(`MAXIMUM 87 1226 PM`), which is also exactly the label the archive needs to fit the
post-issuance new-high rates.

Most bots proxy settlement through raw obs; reading the CLI directly removes the last
basis between our floor and the paying instrument (rounding chain, tenths-°C
conversion, station quirks — the exact class of mismatch behind the 06-29 loss). The
products are free via IEM's AFOS API (`CLINYC`, `CLIMDW`, `CLIMIA`, `CLIAUS`). This
slots in as a second, authoritative `observed_max` source in
`hedge/weather/sources.py` — highest-trust floor when present, ASOS obs as the
fallback between issuances.

*Status: implemented on this branch. `providers.iem_cli_max_so_far_f` (AFOS list for
the target UTC day + the next, immutable per-product text fetch, strict
date/section/range parsing, corrections supersede rather than max, best-effort on IEM
failure) is blended into `LiveForecastSource.observed_max` with a plausibility guard
(a CLI value >5°F above every raw ob from the same sensor is a parse/station mix-up —
keep the obs floor). Live check on 2026-07-01, first cycle after the 20:36Z NYC
issuance: **NYC CLI floor 93°F vs raw-METAR max 91°F** — the official floor ran 2°F
sharper on day one, i.e. two more price levels of deterministic NO; CHI (no issuance
yet) and MIA/AUS (obs fresher than the issuance) all fell back correctly. Remaining
from this item: fit per-city post-issuance new-high rates from the archive, which
prices the 4:40pm–close window the floor opens up (and enables the post-CLI making
idea).*

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

*Status: partially done on this branch — **deterministic** signals now always cross
(IOC) instead of resting (see Round 2), which removes the worst instance of the leak:
a GTC maker resting on a bucket the whole market is about to learn is dead. The
probabilistic path still prefers maker and still needs the obs-tick trigger; note the
existing cycle-level cancel-replace already bounds any rest's staleness to one cycle
(~15 min), so this item's value is concentrated inside the future fast-cadence window.*

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

## P2 (was P3) — Capacity: the live universe is ~4× what we trade

The nowcast edge is real but thin and capacity-limited across 4 cities with depth and
participation caps. **Verified against the live API 2026-07-01:** Kalshi has open
daily-high events for at least **16 cities** — ours plus Denver (`KXHIGHDEN`),
Philadelphia (`KXHIGHPHIL`), LA (`KXHIGHLAX`), Minneapolis (`KXHIGHTMIN`), DC
(`KXHIGHTDC`), New Orleans (`KXHIGHTNOLA`), Dallas (`KXHIGHTDAL`), Las Vegas
(`KXHIGHTLV`), San Antonio (`KXHIGHTSATX`), Atlanta (`KXHIGHTATL`), Seattle
(`KXHIGHTSEA`), and OKC (`KXHIGHTOKC`), each listing today+tomorrow events. Each
added, *validated* city scales the same edge roughly linearly with the same code, and
the outer cities are plausibly less bot-covered than NYC. The only hard part is the
settlement-station map — the #1 correctness risk — and `scripts/validate_stations.py`
already automates the check against resolved settlements. Rule stays: no size until
14/14 exact-bucket validation, `validated=True` only from evidence.

## Round 2 (2026-07-01 PM): what the first pass missed

### Implemented on this branch

- **P0 day-boundary fix + QC guard** (`hedge/weather/providers.py`) — as above.
- **The deterministic window runs all day** (`hedge/strategies/weather_nowcast.py`).
  The nowcast returned `None` before `min_hour=14` *before* checking the
  deterministic floor, so on frontal-passage days — temps falling since mid-morning,
  the high locked by 10am — the cleanest riskless trade sat untouchable for hours
  while the book kept bidding dead buckets. The floor logic ("the max can only rise")
  is valid from the first observation of the climate day; only the *probabilistic*
  path needs the afternoon gate. Unlocking it was only safe together with the P0 fix
  (pre-fix, morning `obs_max` was mostly yesterday evening — the false floor at its
  worst). This also enlarges the flagship trade's time-capacity: buckets die all day,
  not just after 2pm.
- **Deterministic signals cross the spread** (`hedge/decision/engine.py`). The engine
  preferred maker whenever the maker edge cleared `tau_min` — including for
  deterministic signals, resting GTC (`executor.py`) to save a ~1¢ fee. But the fill
  for a settled-dead bucket is the *stale bids*, and they evaporate as the news
  spreads; a resting maker there only fills against someone aggressively buying a
  settled-impossible outcome. EV(taker) = edge beats EV(maker) = P(fill)·(edge + ~1¢)
  for any realistic fill probability once the edge is a few cents — and deterministic
  edges are 10–40¢. Deterministic now takes IOC; the probabilistic path keeps the
  maker preference.

### R2 — Daily-LOW markets: a second deterministic window per city

`KXLOWT*` series exist for ~10 cities and **`KXLOWTAUS` has open events right now**
(2026-07-01; more cities presumably activate seasonally). Lows are the mirror trade:
the observed min-so-far is a hard *ceiling* on the day's low, and it locks in
pre-dawn — the deterministic set is largely settled by 8–9am local, when the
afternoon-tuned bots aren't looking and overnight retail bids linger. Mechanics:
bucket parsing in `markets.py` is identical (integer-°F buckets, same strike shapes);
`TempMarket` needs a high/low `kind` read from the series prefix; the nowcast needs
the inverted deterministic check (obs-min rounding *below* a bucket's floor ⇒ YES
impossible; "X or below" met ⇒ certain) and an `observed_min` source method; the CLI
product already prints `MINIMUM` on the same line format. Medium effort, doubles the
number of daily deterministic windows per validated city, and the race dynamics are
better (who else is sniping temperature markets at 7am?).

### R3 — Hourly directional temperature markets (watch, don't build)

`KXTEMP{NYC,CHI,MIA,DC,BOS,LAX}H` list **hourly** markets ("100° or above" at
"Jul 1, 2026 5 PM EDT"), events confirmed listing today. Two catches, read off the
live market payload: they settle to **The Weather Company** ("as reported by The
Weather Company (for coordinates KNYC)") — a different settlement instrument from
the NWS/CLI everything else here is built on — and the visible books are empty
(no bid/ask/volume). A pure obs-race instrument *if* liquidity appears: one METAR is
the whole outcome. Re-probe monthly; do not build against TWC settlement basis until
there's something to trade.

### R4 — Instrument the race before buying speed

We currently cannot measure our own effective latency (obs timestamp → decision →
order → fill) or maker adverse selection (markout: did the price move through our
resting quote right after we posted?). Both are small additions to the event log —
the obs value already rides in `Signal.meta`; add the obs *timestamp* at decision
time and log book-top at fill reconciliation. The P1 speed items are ranked on an
inferred ~100-minute worst case; a week of instrumented data turns that into a
measured histogram and tells us which upgrade actually moves the tail. Cheap, do it
first.

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

## Priority summary (merged, current as of round 2)

| # | Item | Type | Effort | Why |
|---|---|---|---|---|
| ✅ | Local-day + QC filter in `nws_recent_temps_f` | bug fix | S | Done — false floors (wrong day, glitched sensor) can't reach the flagship trade |
| ✅ | Deterministic window runs all day | unlock | S | Done — frontal-day morning locks were invisible before 2pm |
| ✅ | Deterministic signals cross, don't rest | leak plug | S | Done — fill certainty beats a ~1¢ fee on a 10–40¢ edge |
| ✅ | Afternoon CLI as authoritative floor (blended into `observed_max`) | settlement literacy | M | Done — day-one live check: NYC official floor ran 2°F above raw METARs |
| P1 | Latency + markout instrumentation | measurement | S | Measure where the ~100 min goes before buying speed |
| P1 | Event-driven afternoon cadence + obs high-water mark + METAR phase-lock | speed | M | The core edge is measured in minutes; we currently run ~last |
| P2 | Post-issuance new-high rates per city (archive fit) | settlement literacy | S | Prices the 4:40pm–close window the CLI floor opens up |
| P2 | Validate + enable 12 more high-temp cities | capacity | M | ~4× venues, same code; the station map is the only risk and it's automated |
| P2 | Daily-LOW support (`KXLOWT*`) | new instrument | M | Second deterministic window per city-day, in a less contested hour |
| P2 | Event-strip sum arbitrage pass | model-free | M | Riskless when it fires; monetizes other bots' aggression |
| P2 | Cancel-replace maker rests on obs tick | leak plug | S | Remaining (probabilistic) half of the stale-quote leak; matters inside the fast window |
| P2 | Station-truth calibration by default + recency weighting | calibration | S | Sharper honest σ, regime-fresh bias; code already exists |
| P3 | 6-hr-max METAR groups / `T`-group tenths as extra floor fidelity | fidelity | M | Raises the floor ~0.5–2°F closer to truth; 1-min ASOS itself is archival (backtest fuel) |
| P3 | Hourly TWC-settled markets (`KXTEMP*H`) | watch | — | Re-probe liquidity monthly; different settlement basis (The Weather Company) |
