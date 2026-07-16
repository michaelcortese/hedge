# Kalshi Perps — Strategy Research & Recommendation

*Research date: 2026-07-15. All numbers measured live off the public Kalshi/Coinbase
APIs or pulled from settled-market history on that date unless cited otherwise.*

## Executive summary

Kalshi launched CFTC-regulated crypto perpetual futures on 2026-06-03 (13 markets,
$5.5B volume in two weeks). After ~2 hours of systematic research — live API
probing, a 663-market backtest of BTC 15-minute binaries, a 1,720-market backtest
of BTC hourly ladders, cross-venue funding comparison, lead-lag and basis
measurement, and two deep research sweeps — the honest conclusion is:

**The perp order books and the short-horizon binary books are professionally
efficient at top-of-book. The durable, retail-accessible seat is being the
*maker* on the binary wings — selling the retail lottery flow, which the trade
tape shows paying the resting side +2.3–5.4¢/contract on ~28M wing
contracts/day — with the perp as the real-time index oracle and delta hedge.
Every "obvious" edge (stale list quotes, lead-lag, basis fade, funding carry,
naive favorite-longshot fade as taker) measured dead after fees.**

Recommended build: the **Wing Maker** (§6), validated three independent ways:
Becker's 72M-trade study (crypto makers +1.34%/trade structural), my own
1.3M-trade tape reconstruction on the exact target series (§3b), and clean
top-of-book calibration that bounds what a model-anchored maker risks (§3).
A second candidate, the **Favorite Sniper** (§6b — taking stale near-ATM quotes
seconds after perp moves), is supported by the same tape but must pass a
quote-staleness decay measurement before any build.

---

## 1. Product facts (verified via API)

- **Perps API**: separate surface under `/trade-api/v2/margin/*`
  (`external-api.kalshi.com`; demo: `external-api.demo.kalshi.co`). OpenAPI spec:
  `docs.kalshi.com/perps_openapi.yaml`. Market data + funding endpoints are
  public (no auth). Separate rate-limit buckets from event contracts.
- **Contracts**: sub-$10 notional (BTC = 0.0001 BTC ≈ $6.54; ETH = 0.001; DOGE =
  100; …). Tick $0.0001. Isolated margin, leverage ≈ 6x BTC / 4.5x ETH / 2–3x alts.
- **Fees (perps)**: **12bps taker / 5bps maker** base tier, tiered by 30-day
  volume down to 2.6/0.6bps at $3B (fee-schedule PDF, 7.7.26 update). Launched
  fee-free; fees added late June 2026.
- **Fees (event binaries)**: unchanged `ceil(7% · P(1−P))` taker (max 1.75¢ at
  50¢); maker free on most series, 25%-of-taker on some — **verify the crypto
  series in the fee PDF before going live** (Cloudflare blocked programmatic
  fetch).
- **Funding**: 3×/day (12am/8am/4pm ET) = 8h TWAP of 1-minute premiums vs the CF
  Benchmarks reference index, **deadband: |rate| < 0.01% → 0**, cap ±2%/period.
- **Books/volume (2026-07-15)**: BTC spread 0.3–0.6bps with ~$500k depth within a
  few bps; alts 3–12bps wide, $10–20k at touch. 24h notional: BTC perp $173M,
  ETH $125M, DOGE $32M, ZEC $13M. OI: BTC $5.2M, ETH $1.4M, alts $80k–760k.
- **The binary complex is bigger than the perp**: KXBTC15M (BTC up/down every 15
  min) traded **$892M notional in 7 days (~$127M/day, ~$1.35M per 15-min
  market)**, 24/7 with an evening/overnight-ET tilt. Hourly ladders (KXBTCD,
  ~10–40 strikes/hour) add ~1,700 settled markets per 4 days with vol ≥ 500.
  Settlement for all BTC event horizons: 60-second average of CF Benchmarks BRTI.

## 2. Measured results — what's DEAD (and the evidence)

| Idea | Verdict | Evidence (all measured 2026-07-15) |
|---|---|---|
| Stale-quote sniping on binaries | dead | The `/markets` list endpoint serves stale cached quotes (apparent +12¢ "edges"); live `/orderbook` books are 1¢ wide and sit on perp-implied fair. Trap for the unwary: **never trade off the list endpoint.** |
| Minute-scale lead-lag (Coinbase → Kalshi perp) | dead | After fixing candle timestamp conventions (Kalshi labels candle END, Coinbase labels START): contemporaneous corr 0.97 (BTC), 0.88 (DOGE/ZEC). True 1-min-lag corr only 0.06–0.12 on alts — 12bps taker fee ≫ edge. Sub-second latency games: Polymarket analog shows sub-100ms bots capture 73%; arms race, not our seat. |
| Perp basis fade | dead (HFT-owned) | Basis vs Coinbase spot at 1-min: BTC mean +2.3bps, sd 1bp; ZEC sd 6bps, p99 ±17bps; AR(1) half-life < 1 min. Amplitude ≤ maker fees + slippage. |
| Perp market-making | dead at base tier | 5bps maker fee vs 3–12bps full spreads, against MMs paying 0.6bps with LP incentives. |
| Funding harvest / cross-venue carry | not practical today | Kalshi alt funding is mildly negative on average (BCH −0.9, SHIB −0.7, SUI −0.46 bps/8h ≈ −4…−10%/yr, i.e. longs get paid) but 60–99% of prints are 0 (deadband) — it's episodic. The US-legal short leg barely exists: Coinbase "-PERP-INTX" is **not** US-accessible; the real US product (CDE futures via CFM) has **92–99% overnight short margin on alts**, thin books (~$1.5M/day DOGE), $0.15/contract fee floor. CME lists no DOGE/LTC/BCH/ZEC/SHIB singles. Hyperliquid/dYdX still US-blocked. Revisit when Deribit-via-CFM retail routing ships (CFTC Letter 26-17). |
| Funding-window gaming | dead/illegal | 8h TWAP of 1-min premiums + deadband; moving the TWAP is manipulation. |
| Naive favorite-longshot fade (taker) on 15M/hourlies | dead | Own backtests below. |

## 3. Measured results — the binary backtests (novel data)

Academic studies (Becker; Whelan et al.) deliberately exclude the 15-min/hourly
crypto series, so these backtests are new ground.

**KXBTC15M, 663 settled markets, 7 days** (candle quotes at T−10/−5/−2min vs
settlement): **calibrated at ATM.** Aggregate implied 0.5005 vs realized 0.5189 —
the gap is directional drift, not bias (per-day realized YES rate 0.46–0.65 while
implied hugs 0.50). All taker rules lose ≈1–4¢/contract after fees. Maker-side
sims show +2¢ but with t < 1.8 and no adverse-selection modeling. KXETH15M
(384 markets) mirrors it with opposite drift sign. **No free lunch at ATM.**

**KXBTCD hourly ladders, 1,720 settled strike-markets, 4 days.** Fitting
(S, σ) per event from the strike ladder and z-scoring every strike:

- Tail calibration is clean: implied vs realized within ±2¢ in every z-bucket
  (|z| up to 3+), n ≈ 90–156/bucket. The classic "5¢ contracts win 4.18%"
  longshot overpricing (Becker, all-horizon crypto) is **already gone from
  top-of-book quotes in these series** — the paid MMs (SIG, DL Trading, Keyrock)
  harvested it.
- All taker fade rules lose (e.g. sell YES ≤ 15¢ as taker: −3.9¢/ct, t = −2.1).
  Maker-at-quote versions ≈ breakeven **before** adverse selection.

**The vol-surface finding (the important subtlety).** Ladder-implied vol was
~3.4bps/min (~36% annualized) while √t-scaled 1-minute realized vol read
5–7.5bps/min. The ladder is not "cheap": **BTC microstructure mean-reverts, and
horizon variance is only ~0.69–0.72× of √t-scaled 1-minute variance** (measured
across 6 days: per-min σ 4.85bps at 1m → 4.04bps at 15–45m horizons). The MMs
price *horizon* vol correctly. Any fair-value engine must apply this variance
ratio or estimate vol directly at the horizon scale — a naive 1-minute EWMA
overprices wings by ~40% and generates fake "buy the wings" signals.

**Vol-timing regime test** (fast-EWMA/ladder-implied ratio terciles, 92 events):
no confirmed edge; if anything realized |z| was *lower* when fast vol ran hot.
n too small to conclude more than "no obvious wedge".

**Live forward test (same day, out-of-sample)**: ~1.7h of 60s snapshots of the
VR-corrected model vs live books (1,080 obs, 32 settled markets): market-mid
Brier 0.0799 vs model 0.0843 — **the book beats the model**; divergence-taking
rules scratched or lost at every buffer. Do not take against these books on a
home-grown vol signal.

### 3b. The trade-tape result (the decisive evidence)

Contract-weighted realized P&L of **1,303,397 actual trades (≈133M contracts)**
on the same 1,720 settled KXBTCD hourlies, by taker side and YES-price bucket
(gross = before fees; maker P&L = −taker gross):

| taker side | yes-price | contracts | taker gross/ct | taker net/ct |
|---|---|---:|---:|---:|
| yes | 0.0–0.1 | 17.7M | −0.018 | −0.028 |
| yes | 0.1–0.2 | 8.8M | −0.067 | −0.079 |
| yes | 0.2–0.3 | 5.6M | −0.129 | −0.149 |
| yes | 0.3–0.4 | 4.7M | −0.117 | −0.137 |
| yes | 0.5–0.6 | 4.0M | +0.029 | +0.009 |
| yes | 0.6–0.7 | 4.1M | **+0.136** | +0.116 |
| yes | 0.7–0.8 | 4.9M | +0.120 | +0.100 |
| yes | 0.9–1.0 | 9.9M | +0.023 | +0.013 |
| no | 0.8–0.9 | 7.6M | −0.101 | −0.114 |
| no | 0.9–1.0 | 16.3M | −0.032 | −0.042 |

(NO-side mirrors YES: cheap-NO buyers at yes-px ≥0.8 lose; expensive-NO buyers
at yes-px 0.1–0.4 win.)

Two structural facts, both load-bearing for the strategy design:

1. **Wing flow pays the resting side.** Takers buying YES ≤20¢: 26.5M contracts,
   −3.4¢/ct gross → the maker collected +3.4¢/ct. Takers buying NO at yes-px
   ≥80¢: 23.8M contracts, −5.4¢/ct gross → maker +5.4¢/ct. Both wings positive
   simultaneously ⇒ drift-neutral, structural. Wing taker flow ≈ **12.6M
   contracts/day** on BTC hourlies alone.
2. **The mid-book eats makers alive.** Takers buying the *favorite* at 0.5–0.9
   extracted +3 to +14¢/ct gross from resting quotes — that is informed/latency
   flow hitting stale quotes right after spot moves. A small maker must never
   rest near-the-money without instant quote-pulling; the Wing Maker simply
   doesn't quote there.

**KXBTC15M tape confirms it at the shortest horizon** (~2M trades sampled, last
≤3,000 per market → late-window biased): wing takers lose −2.3¢/ct on BOTH
wings (makers +2.3¢/ct) on ≈ **15.5M wing contracts/day**; favorite-takers
extract even more at 15-min horizon (+18.8¢/ct gross on YES 0.6–0.7, +20.7¢ on
the NO mirror); mid-priced lottery buyers lose −20 to −22¢/ct gross.

Caveats: 4–7 days, one regime; the maker aggregates are earned by whoever holds
queue priority at those prices today (SIG/DL/Keyrock); assume you capture a
minority share and verify with paper markouts (§8). The 15M favorite-taker
magnitudes are inflated by the late-window sampling bias (informed flow
concentrates near expiry where gamma is highest).

## 4. The structural facts that survive

1. **Makers get paid on Kalshi crypto binaries; takers pay.** Becker (72.1M
   trades through Nov 2025): crypto makers +1.34%/trade, takers −1.34%. The
   "optimism tax": cheap YES lottery tickets are −41% EV at 1¢ while cheap NO is
   +23%; retail takes 41–47% of YES volume at 1–10¢ and makers sell it. Maker
   returns are symmetric by direction → structural, not forecasting skill.
2. **The flow is enormous**: ~$127M/day through BTC 15M alone, 24/7.
3. **The perp changes the maker's risk problem**: for the first time the binary
   maker can hedge net delta on the same venue, same account, same collateral,
   at 5bps maker / 12bps taker, with the perp glued to the settlement index
   family (both CF Benchmarks).
4. **Event-side maker fees are zero-to-small**, and resting orders may earn the
   Liquidity Incentive Program (runs through 2026-09-01).
5. Books are 1¢ wide on a ~50¢ instrument — the *spread*, not mispricing, is
   the margin. On wings (3–15¢ / 85–97¢) the spread is proportionally huge.

## 5. Strategy decision

Given §2–§4, taker strategies are structurally negative-EV in this complex, and
speed contests are lost by construction. What remains — and what the data
actively supports — is **joining the maker side on the binary wings**, where:

- per-contract dollar risk is smallest (wings: max loss = price paid ≈ 3–15¢ on
  NO-side sales, vs 50¢ ATM),
- gamma is lowest (quotes stay valid longer; adverse selection per fill is
  smallest far from the strike),
- retail taker flow is most biased (the documented lottery-YES buying),
- competition is thinner than at ATM (MM attention concentrates at the strike).

## 6. Recommended build: the Wing Maker

**One sentence:** rest two-sided quotes on far-from-the-money strikes of the BTC
(then ETH) hourly ladders and the 15M series, priced off a perp-driven fair value
with variance-ratio-corrected vol, skewed to sell the retail-favored side; hedge
accumulated delta with KXBTCPERP; never cross the spread.

**Fair value engine** (prototyped in `scripts/crypto_binary_scan.py`):
- Spot S: KXBTCPERP mid (0.3bp spread — effectively the index; reference_price
  as sanity check).
- Vol: EWMA of 1-min perp log-returns (fast 30m + slow 6h blend, floor at
  0.7×slow) × **variance ratio ≈ 0.70** for horizon scaling; recalibrate the VR
  weekly from perp candles.
- Fair P(≥K) = Φ(ln(S/K)/(σ₁ₘ√(VR·τ))); settlement is a 60s BRTI average —
  treat τ as minutes to window end minus ~0.5.

**Quoting rules:**
- Universe: strikes with fair ∈ [0.02, 0.20] ∪ [0.80, 0.98] (the wings), τ ∈
  [5, 55] min on hourlies; the 15M market only when it drifts off-ATM (fair
  outside [0.35, 0.65]).
- Quote YES-sell (i.e. rest NO-buy at 1−ask) at max(fair + κ·σ_fair, book ask − 1
  tick capped at fair + floor_edge); mirror for deep-ITM YES-buys. Start κ so the
  captured edge ≥ 1¢ expected per contract. Never improve inside fair ± ½ spread.
- Pull quotes when: perp moved > 0.5·σ over last 10s; τ < 5 min (gamma zone);
  funding/CPI/FOMC minute windows; vol regime break (fast/slow EWMA > 1.6).
- **Delta hedge**: net book delta = Σ position·∂P/∂S (digital delta from the same
  model). Hedge with KXBTCPERP maker orders when |Δ| > threshold (e.g. $50 of
  BTC delta); accept 5bps. This is what the perp is *for* in this strategy.
- **Self-match guard**: never rest both sides of the same strike at crossing
  prices; Kalshi prohibits self-matching (rulebook).

**Risk & capacity:**
- Per-strike cap (start: $50 max loss/strike, $300/event, $1k/day book-wide).
- Wings cap worst-case loss at the premium sold ≈ 3–15¢/contract.
- Kill-switch: reuse `hedge/guard.py` logic — realized Brier of fair-value model
  vs settlements; halt if it drifts above the market-mid Brier (the model must
  *earn* its right to lean on quotes).
- Capacity: 15M alone clears $127M/day; capturing 0.1% of flow at 1¢/contract ≈
  $1.3k/day gross ceiling — far above personal-account scale. Realistic near-term
  target after adverse selection: low tens of $/day at $1–2k at risk, scaling
  with observed fill quality. Becker's +1.34% maker margin on crypto flow is the
  structural tailwind.
- Main risk is **adverse selection**: getting filled on wings precisely when a
  fast mover knows the perp jumped. Mitigations: quote-pull triggers above,
  wing-only placement (lowest gamma), delta hedging, and measuring realized
  post-fill markout from day one (log every fill vs perp mid +30s/+120s).

**Why this can persist**: it's not a race — the seat earns the documented
maker-vs-retail margin, which SIG/Keyrock can't exhaust because flow arrives
across ~40 strikes × 24 hourly + 96 daily 15M windows × 13 assets, and their
obligation quotes concentrate at the money. The wings are where a small,
careful, model-anchored book can rest without being run over — and now the perp
lets it stay delta-flat on the same collateral.

**Expectation management (measured 2026-07-15)**: hourly-ladder books for 5–95¢
strikes are **1¢ wide around the clock** — median spread 1.0¢ in every hour of
the day (n=533 T−45 snapshots). There is no "MMs asleep at 4am" regime. In a
1¢-wide, calibrated book the maker's edge per fill is ≈ half a spread minus
adverse selection, so the P&L driver is *volume of benign fills*, not price.
Two practical consequences: (1) queue position matters — join early after each
hourly listing; (2) the wing price bands use **0.1¢ ticks**
(`tapered_deci_cent`: 0–10¢ and 90–100¢ trade in deci-cents), so price-improving
by one tick costs 0.1¢, not 1¢ — the wings are the only place a small maker can
jump the queue cheaply. Taker flow on 15M measured ~46% YES / 54% NO
(4.75M contracts, 30 markets) — direction-balanced at ATM; the exploitable
asymmetry is concentrated in the cheap-YES wing per Becker.

## 6b. Strategy B (validate first): the Favorite Sniper

The tape's mid-bucket result cuts both ways: if resting mid-book quotes bleed
+10–14¢/ct to favorite-buying takers — symmetrically on both sides, so it is not
drift — then *someone* is successfully taking stale binary quotes seconds after
perp/spot moves, and the pool is ≈ 8–9M contracts/day across the 0.5–0.9
buckets. Unlike classic HFT, the race is against the **maker's quote-refresh
latency** (practitioner reports: Kalshi binaries reprice 3–7s after spot; the
Polymarket analog compressed to ~2.7s), not against exchange matching speed.

Sketch: poll KXBTCPERP mid at 5–10Hz (read budget allows it); when it moves
≥ k·σ within ≤2s, immediately IOC-take the favorite side of near-ATM
hourly/15M binaries whose quotes haven't repriced; exit on re-quote or hold to
settle. Fee at 0.6–0.8 ≈ 1.4–1.7¢ vs the tape's +12–14¢ gross pool.

**Measured 2026-07-15 — verdict: NO-BUILD at retail REST speed.** A 25-minute
live probe (2Hz perp polling; 17 move-events ≥3bps; 58 target-market
observations; books snapshotted at +0/0.7/1.5/3/6s) found that by detection
time (~0.5s) the binary books had already repriced: taking the move direction
at the t≈0.5s quote returned **mean −1.0¢/ct before fees** (median −0.5¢);
only 6/58 observations beat the taker fee; 76% of books moved ≥0.5¢ within 6s.
The tape's favorite-taker profits are captured inside the first few hundred
milliseconds (WebSocket + colocation class). Do not chase without that
infrastructure; the probe (`staleness_probe.py`) can be re-run cheaply if
market structure changes.

## 7. Secondary / watchlist

- **Episodic funding capture** (long Kalshi alt perp when funding prints
  negative persistently, no hedge, small size, exit on deadband return) — only
  as an opportunistic overlay; sign risk is real.
- **IBIT-options anchor** (RTH hourlies): during 10am–4pm ET the IBIT chain
  gives an independent implied distribution for the 7 US-session settles;
  divergence > MM spread is a taker signal with real informational backing.
- **Deribit-via-CFM retail rollout** (CFTC Letter 26-17): when it ships, the
  cross-venue alt carry trade becomes practical — re-run the funding-spread
  analysis then.
- **Consistency scanner** across Kalshi's own crypto surfaces (15M vs hourly
  ladder vs daily ranges vs perp) — free option on fat-finger/late-repricing
  events; bounded by event fees, so alert-only.

## 8. Validation roadmap (before any live order)

1. **Paper the maker**: log resting-quote fills you *would* have received using
   the public trade tape (`GET /markets/{t}/trades`) vs your model quotes;
   measure fill rate, post-fill markout, per-fill P&L. 2 weeks minimum.
2. Verify crypto-series maker fees in the fee PDF (in-app).
3. Confirm perps API auth + order flow on demo (`external-api.demo.kalshi.co`).
4. Re-measure the variance ratio and funding stats weekly (regimes move).
5. Gate live size on the same calibration bar as the weather bot
   (`tournament/` + `guard.py` pattern): model Brier must beat market-mid Brier
   on ≥ 200 settled wings before first real order.

## 9. Research artifacts

Scratchpad (this session): `pull_funding.py`, `binary_vs_perp.py`,
`calib_backtest.py` (+ `calib_KXBTC15M/KXBTCD/KXETH15M.json`),
`analyze_calib2.py`, `analyze_ladder.py`, `regime_test.py`, `basis.py`,
`leadlag.py`, `fair_engine.py`, `forward_test.py` (+ live forward log).
Repo: `scripts/crypto_binary_scan.py` (live scanner prototype).

Key sources: Becker, *The Microstructure of Wealth Transfer in Prediction
Markets* (jbecker.dev/research/prediction-market-microstructure); Bürgi, Deng &
Whelan 2026 (karlwhelan.com/Papers/Kalshi.pdf); Kalshi perps docs
(docs.kalshi.com/perps_openapi.yaml, help.kalshi.com collection 19654073);
CFTC perps framework May 2026 (katten.com summary); Kalshi LP program
(help.kalshi.com/en/articles/15410219).
