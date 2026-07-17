# Mention-Markets Edge Campaign — Approach-Family Registry

Standing registry synthesized from the 8-agent exploration round (2026-07-16, workflow
`wf_72a33ab2-b56`). One section per approach family: hypotheses (mechanism / exact test /
kill criteria), verified data sources, quantitative anchors. Cross-family section at the
end defines the shared cost stack every backtest must respect, the Kalshi-only vs
external-corpus split, and the overlap/dedup map.

**Dataset snapshot at exploration time** (`events.jsonl`, collector still running):
459–558 events / 7,174–8,186 settled markets depending on read time; overall YES rate
42.6–42.8%; avg 14.5 markets/event (median 17, max 38); median lifetime volume ~9.3–9.7k
contracts (earnings median 3.9k); median market lifetime 2.9 days.

**Two structural discoveries that shape everything:**
1. `can_close_early=true` — YES resolves the *instant* the phrase is said (any-time hitting
   option); NO pays only at scheduled expiry. `settlement_ts` therefore timestamps exactly
   when a word was first said — a free intra-event information feed.
2. `last_price` / bid/ask in `events.jsonl` are **post-resolution and degenerate** (0–10c
   bucket realized 0.1% YES, 90–100c realized 99.9%). All calibration/pricing tests MUST
   use phase-2 candlesticks (which carry yes_bid AND yes_ask OHLC) at a pre-event snapshot.

---

## 1. TRANSCRIPT BASE RATES
Historical phrase frequency in a speaker's past transcripts as the predictor of P(mention).

### Hypotheses

**T-H1: Recency-weighted Beta-Binomial company base rate beats the market**
- Mechanism: earnings-call language is extremely persistent (TSLA said "optimus" in 6/6
  consecutive calls, 16–38x each); retail prices news salience, not lexical habit.
- Test: p_hat = recency-weighted (λ≈0.8, grid on train only) Beta-Binomial over last 8
  calls with sector-prior pseudo-counts; (A) paired Brier vs market mid at T-24h, one-sided
  paired bootstrap; (B) trade where |p_hat − price| > 2·(posterior sd + fee), mean P&L/contract
  with 95% event-clustered bootstrap CI.
- Kill: on n≥100 markets with ≥4 prior calls, model Brier ≥ market Brier (p>0.05); or
  fee-adjusted P&L CI covers 0; or rule-exact matcher unreplicable for >20% of markets.
- Est. edge: 4–8c on traded subset.

**T-H2: 8-of-8 stickiness — near-certain repeats underpriced on YES** *(≈ SKEP-H4, REC-H1)*
- Mechanism: phrases in 8/8 prior calls repeat with p>0.97 but MMs cap YES at 88–94c; fee
  at 90–95c is tiny (0.34–1.13c).
- Test: subset with 8/8 (secondary 4/4) prior-call presence; one-sided binomial/bootstrap
  that realized YES rate > mean T-24h price + mean fee, event-clustered.
- Kill: realized ≤ price+fee on n≥40; or fewer than 40 qualifying markets exist.
- Est. edge: 3–7c, high hit rate.

**T-H3: Zero-history + quiet sector ⇒ buy NO on salience-inflated phrases**
- Mechanism: novel hot topics diffuse into calls slower than into headlines; 0/8-history
  phrases priced 15–50c resolve NO far more often than priced.
- Test: bootstrap CI on (price − outcome) − fees, clustered by event AND by phrase.
- Kill: n<30 qualifying; or realized YES ≥ price − fees; or edge vanishes under phrase
  clustering (3 lucky phrases). CAVEAT from REC-H2: Kalshi selects novel phrases *because*
  of the news cycle, so zero history alone is NOT a NO signal — the sector-quietness
  condition is load-bearing.
- Est. edge: 10–20c when triggered; fat left tail.

**T-H4: Intra-season sector contagion beats static company history** *(≈ CROSS-H3)*
- Mechanism: season themes propagate (TSLA "tariff" 2→29 mentions in one quarter); later
  reporters get asked what earlier peers were asked.
- Test: logistic with company prior + current-season sector rate (only calls already
  occurred) + last-quarter dummy; b(sector)>0, LR p<0.01, OOS log-loss improvement; then
  economic test on late-season reporters (≥10 peer calls observed).
- Kill: b2 insignificant; or no incremental fee-adjusted P&L vs T-H1.
- Est. edge: 3–6c incremental, late-season only.

**T-H5: Press-briefing (Leavitt) base rates with short half-life** *(≈ REC-H3)*
- Mechanism: near-daily same-speaker cadence, heavily templated rhetoric; short recency
  window (last ~10 briefings, half-life ~3) should dominate.
- Test: paired Brier vs market mid at T-6–12h, plus fee-adjusted P&L; check edge
  concentrates in p_hat>0.9 / <0.1 tails.
- Kill: model Brier ≥ market on n≥80; or edge only at untradeable snapshots; or Factbase
  publishing lag > 1 briefing.
- Est. edge: 3–8c in tails.

**T-H6: Count-intensity (NegBin) beats binary history; prices threshold markets** *(≈ REC-H5)*
- Mechanism: "said 30x/call" vs "said 1x/call" both score 1 in binary history but imply
  different P(≥1); mention counts ≈ Gamma-Poisson.
- Test: paired OOS log-loss NegBin P(≥1) vs binary Beta-Binomial; on "N+ times" markets,
  NegBin tail P(count≥N) vs price Brier.
- Kill: no OOS improvement (p>0.05); or improvement <1c of Brier-implied price difference.
- Est. edge: 1–3c sharpening; mainly improves T-H2/T-H3 sizing.

### Data sources
| Source | URL / access | Verified |
|---|---|---|
| defeatbeta/yahoo-finance-data parquet (**backbone earnings corpus**) | `https://huggingface.co/datasets/defeatbeta/yahoo-finance-data/resolve/main/data/stock_earning_call_transcripts.parquet` — keyless; query remotely with duckdb httpfs `read_parquet(url)`; 234,273 transcripts, 6,374 symbols, 2005→2026-07-22 (days-fresh), speaker-segmented | **YES** (TSLA counts extracted) |
| Motley Fool transcripts | `https://www.fool.com/earnings-call-transcripts/` — free scrape w/ browser UA; per-article pages server-rendered; enumerate via `fool.com/news-sitemap.xml` (listing page is JS — don't scrape it); same-day publication | **YES** (multiple agents; ASML/MS full text, counts extracted) |
| Roll Call Factbase (Trump + Leavitt) | per-page fetch verified, e.g. `https://rollcall.com/factbase/trump/transcript/donald-trump-press-conference-briefing-karoline-leavitt-march-10-2026/`; slug pattern `donald-trump-press-(conference-)briefing-karoline-leavitt-{month}-{day}-{year}`; JSON API `/wp-json/factbase/v1/search` returns 0 rows anonymously — treat as GATED | **Pages YES; API NO** |
| Rev.com transcripts | `https://www.rev.com/transcripts/...` — per-briefing pages fetch free (14.5k-word page verified); enumeration index not yet scripted | **YES** (pages) |
| HF jlh-ibm/earnings_call | datasets-server rows API — verified but tiny (188 transcripts 2016–2020); unit tests only | YES (low value) |
| EarningsCall.biz | demo key works AAPL/MSFT only; paid beyond | YES (2 tickers) |
| earningscalls.dev, API Ninjas, FinancialModelingPrep | keys required / docs 404 / demo rejected — conflicting reports across agents | **NO — do not build on** |
| whitehouse.gov /briefings-statements/ | fetches 200 but content is presidential statements, NOT Leavitt briefing transcripts | YES-but-wrong-content |
| UCSB American Presidency Project | 200 but 2026 briefing coverage unconfirmed | NO |

### Anchors
- Settlement rule (from `rules_secondary`, inline in dataset): exact word, plural/possessive
  count, other inflections do NOT, said by any company representative INCLUDING the operator,
  NOT analysts; video primary, transcript secondary; `/`-bundled strikes are OR-alternatives
  (2,755/9,653 strikes ≈ 29%).
- Transcript-vs-settlement mismatch must be audited on settled markets before trusting tails;
  kill threshold >3–5% disagreement; fold residual ~2–3pts into `Signal.std_error`.

---

## 2. MARKET MICROSTRUCTURE & BEHAVIORAL BIASES (prices only, no NLP)

### Hypotheses

**MS-H1: Favorite-longshot calibration gap by price bucket × time-to-close** *(≡ SKEP-H1)*
- Mechanism: retail lottery YES demand + no shorting + payoff-timing asymmetry (YES pays
  instantly on utterance, NO locks capital to expiry) ⇒ YES overpriced 5–60c, extreme
  favorites 90–97c underpriced. Suggestive (degenerate-data) evidence: last-price 50–60c
  bucket realized 11.8% YES (n=17), 60–70c → 38.9%, 70–80c → 42.9%.
- Test: candle mids at τ ∈ {48h, 24h, 6h, 1h} pre-close (candles strictly before
  `occurrence_datetime`); bucket into deciles + [90,95) + [95,99); realized YES freq vs
  mid ± (fee + half-spread); event-cluster-robust SEs / logistic GLMM; Brier decomposition
  per bucket; BH-FDR q=0.10 across the full bucket×τ grid.
- Kill: no cell shows |realized − mid| > fee + half-spread on chronological 60/40
  train/confirm with FDR control; or gap sign flips between splits.
- Est. edge: 3–8c buying NO in 20–60c buckets; 1–3c buying YES at 90–97c.

**MS-H2: Event-basket overround — sell the basket** *(unique; highest-power test in campaign)*
- Mechanism: ~14.5 lottery-priced phrase markets settle off ONE transcript, no basket
  instrument or shorting ⇒ Σ(YES prices) persistently exceeds E[# phrases said]. Base:
  only 42.8% resolve YES (KXFIGHTMENTION 18.3%).
- Test: per event at τ=6h, D = Σmid − #YES; sign test + event-level bootstrap (events are
  the iid unit — naturally cluster-clean, 459+ obs). Predict mean D ≥ 0.5 phrases/event,
  p<0.01; secondary: D grows with #markets listed.
- Kill: mean D < 0.3 on confirm window; or basket-NO P&L after fees+half-spread ≤ 0
  (95% event-bootstrap CI containing 0).
- Est. edge: 2–5c/contract; ~30–70c expected profit per event-basket per contract-unit.

**MS-H3: Hazard under-decay in the live window ("hope premium")** *(≈ HAZ-H1/H2)*
- Mechanism: conditional on no mention after fraction f elapsed, fair value collapses
  along the hazard curve; retail anchors to pre-event price.
- Test: trade tape restricted to in-window trades on not-yet-closed markets; logistic of
  outcome on logit(price) + f + interaction, event-clustered; predict >5c shortfall at
  f>0.5, price 20–60c; P&L form: buy NO at every qualifying print.
- Kill: interaction ≈ 0; <150 events with in-window mid-price trades; NO-at-print P&L
  <2c/contract on confirm; adverse-selection check (fill-conditional vs print-conditional
  outcome) fails.
- Est. edge: 5–15c late-window; capacity $50–300/market.

**MS-H4: Taker order-flow-imbalance reversal — fade retail YES herding**
- Mechanism: tape labels taker side; same-direction YES-taker bursts in thin books
  overshoot and mean-revert; no genuine private info exists days pre-event.
- Test: hourly OFI deciles (pre-event window, volume>0); panel regression of next-24h mid
  change on OFI deciles, two-way clustered (event, calendar-hour); plus executable fade-rule
  P&L. Need ≥2,000 top-decile market-hours across ≥250 events.
- Kill: top decile predicts continuation/nothing; fade P&L ≤0 after spread+fees; effect
  only in <50-contract hours.
- Est. edge: 2–6c on flagged hours.

**MS-H5: Post-listing drift — openings are anchored seeds** *(≈ SKEP-H5)*
- Mechanism: ~14 markets/event listed days early at semi-arbitrary prices; information
  arrives as slow drift away from 50c; +24h drift direction predicts remaining drift.
- Test: (1) |p_pre-event − 0.5| > |p_open − 0.5| (event-clustered t); (2) sign(p_24 − p_open)
  predicts continuation, hit rate >52%, follow-rule (±7c trigger) P&L via event bootstrap;
  (3) Brier(p_24) < Brier(p_open).
- Kill: follow P&L ≤0; hit rate ≤51%; open already efficient — FDR jointly with MS-H1/H4.
- Est. edge: 2–5c when triggered.

### Data sources
| Source | URL / access | Verified |
|---|---|---|
| Local collector dataset | `data/research/mentions/events.jsonl` | **YES** |
| Kalshi hourly candlesticks | `https://api.elections.kalshi.com/trade-api/v2/series/{series}/markets/{ticker}/candlesticks?start_ts=&end_ts=&period_interval=60` — keyless GET; candles carry yes_bid AND yes_ask OHLC + volume_fp + OI | **YES** (97-candle NKE fetch; 113–115-candle FDX fetches by other agents) |
| Kalshi trade tape | `https://api.elections.kalshi.com/trade-api/v2/markets/trades?ticker=&limit=&cursor=` — keyless, cursor paging; per-trade created_time, price, count_fp, `taker_side` + `taker_outcome_side`, is_block_trade; full history for settled markets | **YES** (live fetch by microstructure agent) |

### Anchors
- Family skews: KXFIGHTMENTION 18.3% YES, KXMLBMENTION 50.5%, EARNINGS 50.0%, TRUMP 45.1%,
  SPORTS 43.5%, OTHER 37.0%.
- Everything must be event-clustered; chronological 60/40 split; BH-FDR q=0.10 ONCE across
  the union of pre-registered cells.

---

## 3. HAZARD / TIME-STRUCTURE MODELS (intra-event survival dynamics)

Core model: fair mid-event price p(t) = p₀(1−F(t)) / (1 − p₀F(t)), p₀ = pre-event price,
F = first-mention-time CDF among mention events. All hypotheses test whether the market
tracks this curve.

### Hypotheses

**HZ-H1: Sticky survival decay (market under-decays conditional on no mention)** — the gate.
- Test: per-(market, minute) panel over pre-mention minutes (mention times from
  captions/transcripts); Test 1: mean(mid − p_fair) ≥ 2c in middle third, event-clustered;
  Test 2 (decisive): paired Brier of hazard-adjusted price vs market mid per elapsed-fraction
  decile, paired sign test p<0.05.
- Kill: hazard price never beats mid on paired Brier; mean overpricing <2c; OR caption
  detection disagrees with Kalshi resolution on >5% of markets (**kills the whole family's
  measurement plane**).
- Est. edge: 3–10c per NO contract mid-event.

**HZ-H2: Executable NO buy at the prepared-remarks/Q&A boundary** — the tradeable claim.
- Test: at τ* (Q&A start from transcript; minute 15 for briefings), buy NO at 100−yes_bid
  for every unmentioned market with p_fair < yes_bid − fee − 1c; hold to settlement; event
  block-bootstrap; require lower 95% CI > 0 over ≥30 events / ≥150 trades.
- Kill: mean net edge ≤0; yes_bid absent/<3c in >50% of candidate minutes; <30 events with
  mid-event quotes; adverse selection (fills resolve YES at rate > p_fair + 2SE).
- Est. edge: 5–8c net; 10–100 contracts/market.

**HZ-H3: Pre-event duration/verbosity blind spot**
- Mechanism: P(mention) ≈ 1−exp(−λW); speaker word count W is persistent and forecastable;
  markets price base rates, not verbosity.
- Test: logistic resolution ~ logit(price@T-2h) + log(W_hat from trailing 4 calls / 10
  briefings), event-clustered p<0.05 AND OOS log-loss improvement ≥0.5% relative.
- Kill: β_W insignificant/unstable; implied gap <2c (inside fee floor).
- Est. edge: 1–3c; zero-infrastructure (fits standard Signal seam as-is).

**HZ-H4: Phrase-type hazard heterogeneity (narrative vs analyst-forced)** — selection layer for HZ-H2.
- Test: early-concentration E = F̂(τ_QA) per phrase (≥20 events/phrase); regress mid-event
  overpricing residual on E; slope>0, p<0.05, clustered; KS stationarity check on F̂ halves.
- Kill: slope ≤0; KS rejects (hazard drifts); <10 phrases clear the 20-event bar.
- Est. edge: +2–5c selection on top of HZ-H2.

**HZ-H5: Post-event settlement drag (terminal-hazard carry)**
- Test: NO-resolved markets, window [event_end+10min, close]: frequency/level of yes_bid≥3c;
  simulate buy-NO-at-bid; charge every adverse-YES (event-end misidentification) at full 100c.
- Kill: opportunity in <10% of post-event minutes; adverse-YES ≥ premium; <~$50/event volume.
- Est. edge: 1–3c near-riskless, capacity-limited.

**HZ-H6: Utterance-to-price latency (live YES snipe)** *(≡ SKEP-H3; test before building)*
- Test: for YES-resolved briefing markets, latency L = (first tape trade ≥80c) − (caption
  utterance time); executable gap G = 100 − post-utterance ask. Edge iff median L > 60s and
  median G > 15c after fees, ≥40 YES events.
- Kill: median L <10s (ASR bots incumbent); G <5c; ASR false-positive >2% (one false 85c
  buy erases ~17 wins). NOTE: needs a low-latency executor path outside the Signal
  contract — framework project, only on a decisive pass.
- Est. edge: 10–40c per hit if latency persists; likely already competed.

### Data sources
| Source | URL / access | Verified |
|---|---|---|
| YouTube auto-captions via yt-dlp | `yt-dlp --skip-download --write-auto-subs --sub-langs en --sub-format vtt <watch URL>`; WH briefing playlist + C-SPAN mirror; **word-level timestamps** (242KB VTT verified) | **YES** |
| Kalshi 1-MINUTE candlesticks | same endpoint, `period_interval=1`; 33 candles w/ bid/ask OHLC verified on live SECPRESS market — the critical enabler for 30–90-min events | **YES** |
| Motley Fool transcripts | call date AND start time in header; explicit Q&A boundary line ("we will now open the lineup to questions"); mention time ≈ word_offset/150wpm, ±2–3 min | **YES** |
| Kalshi trade tape | needed for HZ-H6 latency; verified by microstructure agent | YES (cross-agent) |
| faster-whisper / IR webcast audio | `https://github.com/SYSTRAN/faster-whisper` — not installed; only if ±3-min approximation proves too coarse (run sensitivity first) | NO |

### Anchors
- Count-threshold markets ("Iran 3+ times") need mention-#k time, not first-passage —
  parse `yes_sub_title` to flag them.
- Collector quirk: events endpoint with `status=settled` can return event shells without
  nested markets — pull markets per event ticker directly.

---

## 4. CROSS-MARKET LOGICAL CONSISTENCY / RELATIVE VALUE

### Hypotheses

**XM-H1: Intra-event containment/implication violation scanner** — **near-dead, keep as free scanner.**
- Adversarial finding: 0 word-boundary containment pairs in 53 settled earnings events,
  5 pairs in all 487 events — Kalshi bundles variants into `/`-alternative strikes.
- Kill: <1 executable violation-hour per 100 events or total extractable P&L <$10.

**XM-H2: Cross-company common-factor underreaction on macro words (sector RV)**
- Mechanism: same macro word ("tariff" n=40 @55% YES) priced by separate retail crowds;
  cross-sectional peer average is a better factor estimate than any single market.
- Test: pooled logistic outcome ~ logit(own price@T-24h) + logit(vw peer price same word,
  same quarter), cluster by word-quarter; peer coefficient >0, p<0.05. Secondary: Spearman
  of (peer-implied fair − own price) rank vs residual.
- Kill: insignificant/sign-unstable across quarter splits; or trading >10c gaps ≤0 after
  fees over ≥40 trades. Warning: trailing-outcome majority already FAILS to beat base rate
  (49% vs 55% on tariff) — edge must be price-relative, not outcome-momentum.
- Est. edge: 4–8c on flagged gaps.

**XM-H3: Stale repricing after a peer call settles the quarter's hot topics**
- Mechanism: early settlement (`settlement_ts`) reveals topic intensity within hours; later
  reporters' same-word markets reprice slowly.
- Test: event study of ΔP over 24h post peer-outcome; underreaction test: outcome ~
  logit(price@+24h) + peer_outcome, positive coefficient; rule: buy W-YES on later same-sector
  companies within 2h of peer W=YES if price moved <3c.
- Kill: price@+24h fully absorbs (p>0.10); median <2 open same-word pairings; 2h-rule ≤0
  after fees over ≥30 trades.
- Est. edge: 5–15c in first hours after strong peer signal.

**XM-H4: Cross-event same-word base-rate transfer (company-conditional)** *(≡ T-H1 tail version)*
- Anchors: "dividend" 75% YES, "nvidia" 7/7, "recession" 0/7; FDX "China" quoted 43–48c
  days out and settled NO — existence proof of the target shape.
- Test: Brier(transcript rate b over last 8 calls) < Brier(price@T-24h) on tail subsample
  (b<0.1 or b>0.9); strategy trading |b − price|>15c: >0 after fees, win rate >60%, ≥50 markets.
- Kill: no significant Brier improvement; transcript-vs-resolution mismatch >5%; P&L ≤0.
- Est. edge: 10–25c on lazily-priced tail words.

**XM-H5: Consecutive-event persistence in recurring speaker series** *(≈ REC-H4, T-H5)*
- Measured: P(YES|prev YES)=0.45 vs P(YES|prev NO)=0.36 (726 transitions, Trump series) — modest.
- Test: logistic outcome ~ logit(price@T-12h) + trailing-5-event YES rate; coefficient >0
  p<0.05 given price; fade set = trailing ≥0.8 or ≤0.2 with price 35–65c.
- Kill: p>0.10 given price; <15 fade cases; ≤0 after fees; or no info beyond unconditional
  word rate (LR test vs word fixed effect) → fold into XM-H4.
- Est. edge: 3–8c in tail-history words.

**XM-H6: Within-event topic-basket co-movement laggards**
- Measured within-event outcome φ: dividend–tariff 0.56, buyback–tariff 0.42 (company-mix
  confound suspected — re-test with company-type fixed effects).
- Test: (a) pre-event laggards vs cluster moves (cluster term p<0.05 given price);
  (b) intra-event: words open when ≥2 cluster-mates early-settled YES beat price by >10c.
  Keys off Kalshi's own early settlements — no audio needed.
- Kill: candles too coarse (>50% laggard flags without bid updates); insignificant given
  price; <20 intra-event situations or gap < fees+3c.
- Est. edge: 5–12c intra-event; 3–6c pre-event.

### Data sources
Local dataset + hourly candlesticks + Motley Fool (all verified — see families 1–2);
Rev.com (verified, listing fetched); earningscalls.dev (UNVERIFIED — verify one full
transcript before committing). Manual artifact: sector map for ~60 companies (static CSV).

---

## 5. EMBEDDINGS / SEMANTIC ML (supervised on settled markets, encompassing test vs price)

### Hypotheses

**EM-H1: Rule-aware speaker-corpus base rate beats the market (anchor model)** *(≡ T-H1 with rigor)*
- Test (the family's evaluation gospel): expanding-window CV by event month;
  Fair–Shiller encompassing regression y ~ logistic(b₀ + b₁·logit(p_mkt) + b₂·logit(p_model));
  reject efficiency iff b₂>0 (event-clustered Wald p<0.01) AND blend improves OOS log-loss
  vs market-only ≥3% relative (Diebold–Mariano, event-block bootstrap 10k, p<0.05).
- Kill: b₂ CI covers 0; <1% relative improvement; unstable across ≥3 OOS quarters; edge
  only below-median-volume; after-fee P&L ≤0.
- Est. edge: 3–8c on mispriced tail.

**EM-H2: News-salience overpricing — fade the hype phrase** *(≈ NF-H4 mirror)*
- Test: news_z from Google News RSS counts (14d vs 90d baseline, strictly pre-entry);
  y ~ logit(p) + hist_rate + news_z with b(news_z)<0; portfolio: NO on top-quintile news_z
  (price 25–75c), after-fee P&L >0 (event bootstrap).
- Kill: b ≥0/insignificant; <100 OOS markets; sign flips across families; absorbed by hist_rate.
- Est. edge: 2–6c on hyped mid-priced phrases.

**EM-H3: Embedding similarity to prior-call remarks adds incremental signal over exact counts**
- Test: nested comparison — counts-only vs counts + cos(phrase, prior-call centroid) +
  max-sentence-sim + time-restricted kNN-label; require ≥2% relative OOS log-loss gain,
  DM-bootstrap p<0.05. Never feed raw 384-d vectors (n~2k would overfit); freeze model
  version pre-2025.
- Kill: <1% gain → ship counts model, drop embeddings; or gain explained by kNN-label
  alone (memorized phrase identity).
- Est. edge: 1–3c incremental; model-quality multiplier.

**EM-H4: Phrase-type miscalibration — proper-noun discount, generic-business-word premium**
- **Cheapest hypothesis in the campaign — no external corpus needed.**
- Test: kNN type classifier from ~100 hand-labeled seeds (train-window only); per-type
  OOS mean(y − p_mkt) with event bootstrap + Spiegelhalter Z; predict proper-noun/geo bucket
  < −2c, generic-business > +2c, CI excluding 0, holding out-of-time.
- Kill: no bucket clears ±2c; flat OOS; fully absorbed by hist_rate.
- Est. edge: 2–5c within extreme buckets.

**EM-H5: Edge concentrates in cold-start events (deployment filter)**
- Test: Fair–Shiller with logit(p_model)×cold_start interaction >0 (event-clustered p<0.05);
  blend improvement ≥2x larger in cold-start subgroup.
- Kill: interaction ≤0; <150 cold-start markets (mark inconclusive, don't trade); edge
  entirely sub-median-volume.
- Est. edge: 4–10c cold-start, ~0 on seasoned weekly series.

### Data sources
| Source | URL / access | Verified |
|---|---|---|
| Local dataset + candles | see above | **YES** |
| Motley Fool | see family 1 | **YES** |
| SEC EDGAR full-text search + submissions | `https://efts.sec.gov/LATEST/search-index?q=%22tariff%22&forms=8-K` — free JSON, requires User-Agent w/ contact email; 232 hits verified | **YES** |
| Google News RSS | `https://news.google.com/rss/search?q=%22tariff%22+FedEx+when:7d` — free, per-item pubDate; **LIVE ONLY**, no historical backfill | **YES** |
| HF earnings datasets (jlh-ibm, Rogersurf, finosfoundation) | datasets-server rows API; existence verified, per-ticker coverage NOT sampled | Partial |
| GDELT DOC 2.0 API | `https://api.gdeltproject.org/api/v2/doc/doc` — **UNREACHABLE from this WSL host (TLS reset x2)**; run from Fly box or use raw archives | **NO (env-blocked)** |
| whitehouse.gov, Roll Call Factbase | index 200s; per-item parse not exercised here (but verified by other agents — see families 1, 8) | Partial |
| sentence-transformers (all-MiniLM-L6-v2 / bge-small) | pip install pending; 16-core box does ~500+ sentences/s CPU, ~100k-sentence corpus embeds in minutes | NO (install pending) |

### Anchors — leakage traps (binding for all model families)
1. Filter every news/filing item by timestamp < t_entry (incl. same-day 8-K).
2. Entry price from candles at t_entry — never snapshot last_price.
3. Split by event date; cluster by event_ticker (and check phrase-level clustering — "tariff"
   ×40, WC phrases ×100).
4. Matcher must replicate the settlement rule exactly or labels are mismeasured.
5. Drop result=='' (67) and "event does not qualify" strikes (280).
6. Kalshi phrase selection drifts — expanding-window CV; monitor per-quarter base rate.
7. Transcript coverage is size-biased — impute missing hist_rate with cross-company prior.

---

## 6. NEWS-FLOW NOWCASTING

### Hypotheses

**NF-H1: Post-listing news-shock underreaction (earnings)**
- Test: GDELT GKG z-score (72h vs 30d baseline, strictly pre-t=24h-pre-close); logistic
  Y ~ logit(p(t)) + z, event-clustered, one-sided b(z)>0 α=0.05; rule: buy YES at ask when
  z>2 and model p > ask + fee + 3c; event block bootstrap, 95% CI > 0.
- Kill: b≤0 or p>0.05 at n≥150 over ≥2 seasons; CI includes 0; **effect vanishes when
  price is taken from yes_ask instead of last-trade (staleness confound)**.
- Est. edge: 4–10c on the ~5–10% of markets with z>2.

**NF-H2: Morning-news-cycle underreaction in press-briefing markets**
- Test: z_morning (06:00–11:00 ET GDELT share vs prior 7 briefing days); logistic
  Y ~ logit(p_11am ask) + z_morning, cluster by briefing; rule: buy YES at 11am ask when
  z>2 and ask<70c. Lookahead guard: exclude markets whose early close fired before 11am.
- Kill: b≤0 at n≥200 over ≥25 briefings; edge only in zero-volume 10–11am markets; CI incl 0.
- Est. edge: 5–12c on shocked words.

**NF-H3: Wikipedia attention spike as cheap orthogonal nowcast**
- Test: w = log(views_{d−1}/median_28d); Y ~ logit(p) + w, and jointly with z_GDELT —
  survives only if b(w)>0 alone AND with z included (else substitute-grade fallback).
- Kill: b≤0 at n≥150; no incremental LR improvement (p>0.10); <50% of phrases mappable.
- Est. edge: 2–5c incremental; robustness feature.

**NF-H4: News-decay overpricing — fade the stale headline (buy NO)** *(≈ EM-H2)*
- Test: d = z_last24h − z_first24h; logistic Y ~ logit(p) + d event-clustered, one-sided;
  rule: buy NO at no_ask (t=12h pre-close) when d<−2, yes price 40–85c; needs ≥48h markets
  (mostly earnings family).
- Kill: b≤0 at n≥150; NO-side edge dies to fees + wide no_ask spread (kill if median no
  spread > 2× edge); fully absorbed by NF-H1's z (LR p>0.10).
- Est. edge: 3–8c per NO contract, spread-sensitive.

**NF-H5: Speaker-history conditioned news response (per-phrase news-beta)**
- Test: hierarchical logistic logit P(Y) = a_phrase + b_phrase·z, shrunk to pooled b;
  OOS paired Brier vs (i) market and (ii) pooled model, Wilcoxon α=0.05; trade only
  |model − ask| > 5c.
- Kill: doesn't beat market OOS (≥30 phrases × ≥3 OOS events); beats pooled but not market;
  b sign flips in >40% of phrases across seasons.
- Est. edge: 3–7c on high-|b| phrases during shocks.

### Data sources
| Source | URL / access | Verified |
|---|---|---|
| **GDELT 2.0 raw 15-min archives** (backbone) | `http://data.gdeltproject.org/gdeltv2/` — free, keyless, plain-HTTP zips every 15 min back to Feb 2015; ~5–10MB/slice, ~0.7GB/day; verified: downloaded tariff-shock-day slice, 2,217 records, 112 matching 'tariff' | **YES** |
| GDELT DOC API (timelinevol) | free, ~1 req/5s — **TLS-reset from this WSL host**; verify from Fly box; treat as optimization not dependency | **NO (env)** |
| Wikimedia REST pageviews | `https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/{Article}/daily/{start}/{end}` — free; verified Tariff spike 6,257→60,673 (Apr 1–3 2025); daily granularity; needs manual phrase→article map | **YES** |
| SEC EDGAR FTS | verified (see family 5) — high-precision issuer-level pre-call signal | **YES** |
| Google News RSS | verified live; **no historical queries** — production leg only, never backtest | **YES** |
| Wayback CDX | **blocked from this environment** (TLS reset + fetch refusal) — dropped | **NO** |
| Kalshi public API | verified; note `status=settled` empty for KXSECPRESSMENTION markets query — page via events/finalized | **YES** |

### Anchors
- KXSECPRESSMENTION opens ~35h pre-briefing (03:20Z for 14:00Z close) — defines the
  tradeable window for NF-H2.
- Environment: api.gdeltproject.org and web.archive.org TLS-reset from this WSL host;
  data.gdeltproject.org (plain HTTP), wikimedia, efts.sec.gov, news.google.com,
  api.elections.kalshi.com all work. Run DOC/TV-API collectors from the Fly box.

---

## 7. EFFICIENT-MARKET SKEPTIC / EDGE ACCOUNTING

### Hypotheses

**SK-H1: NO-side premium from entertainment YES flow** *(≡ MS-H1; adds fee-asymmetry rationale)*
- Extra mechanism: fading longshots is the cheapest trade on the board (0.14–0.63c fee at
  90–98c NO); retail 1-lots pay ceil-fee = up to 20% of stake.
- Test: T-24h mid buckets ([1,5)…[95,99]); realized YES freq ≥2c below mid in [1,20)
  buckets; blanket-NO rule (mid<40c) mean P&L > +0.5c/contract after ceil-fee at C=100,
  event-clustered bootstrap p<0.05.
- Kill: calibration within 1.5c of mid; portfolio ≤0.5c; edge only in <500-volume markets;
  **edge disappears in 2026H1 subsample (MMs adapted)**.

**SK-H2: Pre-event market-making — takers lose, spread collectible at zero maker fee**
- Test (markout, needs tape): mean pre-event taker markout (settlement − price − fee,
  taker-signed) NEGATIVE ≥1c/contract; in-event markout strongly POSITIVE (quotes must be
  pulled at event start). Per family and price bucket.
- Kill: pre-event markout ≥0; or negative but <0.5c (below operational cost); or fills
  concentrate in final pre-event hour where flow turns toxic.
- Est. edge: 0.5–2c per filled contract; ~500–2,000 fills/day across families.
  Implementation is Wing-Maker-adjacent (existing perps paper harness reusable).

**SK-H3: In-event stale-quote sniping** *(≡ HZ-H6; the weather obs-lag analog)*
- Test: per settled-YES market, t* = first tape trade ≥85c; snipeable volume V = contracts
  ≤92c in [t*, t*+30min]; require median V ≥100 and avg available profit ≥$15/event;
  secondary: ≥30% of repricing volume prints >60s after t* (race winnable with commodity ASR).
- Kill: >80% of volume reprices within 60s; median V <30; ASR false-positive >~5%.
- Est. edge: 5–20c per filled snipe; 100–500 contracts/event.

**SK-H4: Near-certain phrases underpriced at 90–97 (and never-said overpriced 3–10)** *(≡ T-H2/T-H3)*
- **Adversarial caveat (load-bearing):** earnings strikes resolve YES exactly 348/348 = 50.0%
  in the local data — Kalshi CHOOSES coin-flip phrases; naive "obviously said" intuition is
  what the strike committee selects against. Only quantitative transcript divergence is edge.
  FDX "China" resolved NO.
- Test: Beta-Binomial posterior from last 8 same-company calls; markets with posterior >0.97
  but T-24h ask ≤95 resolve YES ≥97.5%; +1.5c/contract after fees, n≥50. Mirror on
  posterior <0.03 with bid ≥5c.
- Kill: P&L ≤0 at ask fills; <50 instances; posterior>0.97 already quoted 97+ (check first,
  cheap); transcript-vs-settlement disagreement >3% ("ERA5≠settlement" lesson).

**SK-H5: Listing-window mispricing (first-24h book is placeholder-wide, anchored)** *(≡ MS-H5 + model)*
- Evidence: first FDX candle quoted bid 3 / ask 84 (placeholder).
- Test: Brier(first two-sided candle mid) − Brier(T-24h mid) ≥ 0.02; drift predicts outcome;
  trading form: at listing+2h take asks ≥5c inside transcript-posterior fair.
- Kill: dBrier <0.01; <5% of markets have takeable two-sided book in first 6h; fills require
  crossing ≥8c spreads.
- Est. edge: 5–15c when filled; capacity $10–50/event.

### Data sources
Local dataset, hourly candles, live markets snapshot (`/markets?series_ticker=KXSECPRESSMENTION&status=open`,
28 open markets fetched), trade tape (pending phase 3 locally; endpoint verified by
microstructure agent), HF transcript corpora (existence verified, ticker coverage unchecked).

### THE COST STACK (see cross-family section — this family's numbers are binding campaign-wide).

---

## 8. RECURRING-EVENT REGULARITIES (template language, briefing stock phrases, nickname hazard)

### Hypotheses

**RE-H1: Template-phrase carry on earnings calls (deep-ITM YES)** *(≡ T-H2/SK-H4 with streak framing)*
- Test: (1) transcript-side P(mention | said in all last 3 calls) ≥0.95; (2) settled markets
  priced 80–95c with streak s≥3 resolve YES ≥3pts above price-implied (exact binomial +
  logistic resolution ~ price + 1{s≥3}, company-clustered, p<0.05).
- Kill: persistence <0.95; realized ≤ price+2pts at N≥80; <~30 tradable plain s≥3
  markets/quarter after the RE-H2 selection filter.
- Est. edge: 4–7c net at 88–92c entry; capacity tens-to-low-hundreds of contracts.

**RE-H2: Kalshi phrase-selection & handicapping audit** — **THE GATE. Run before any
transcript-carry strategy is built.**
- Direct evidence (live TSLA Jul-22 event): near-certain phrases handicapped with qualifiers
  ("Grok (3+ times)", "Megapack 3") while some template words still listed plain
  (Gigafactory, Cybertruck, China); novel phrases selected because of the news cycle.
- Test: (1) qualified phrases show higher transcript persistence than plain (Mann-Whitney
  on streak s — proof of deliberate handicapping); (2) plain novel s=0 phrases resolve YES
  far ABOVE historical unseen-word base rate (selection-on-news); (3) DECISIVE: logistic
  resolution ~ logit(price) + s + freq_last4 on plain markets — if streak terms jointly
  insignificant (LR p>0.10, N≥300), the market already prices persistence and the
  transcript-carry family is dead.
- Output: per-family "exploitable inventory" count (plain s≥3 markets ≤95c per quarter).
  Expected outcome: shrinks RE-H1 inventory 50–80% but doesn't zero it.

**RE-H3: Leavitt briefing stock-phrase carry (KXSECPRESSMENTION)** *(≡ T-H5 + length effect)*
- Test: (1) P(mention | ≥8 of last 10 briefings) ≥0.95; (2) settled 80–95c markets with
  last-10 freq ≥0.8 resolve ≥3pts above price (event block-bootstrap); (3) positive
  log(word_count) coefficient — skip short-gaggle days.
- Kill: persistence <0.93; bucket test fails at N≥60; Kalshi prices these ≥96c (no entry);
  transcripts not enumerable within hours of settlement.
- Est. edge: 4–7c at 85–92c; **highest trade frequency of the family** (1–3x/week ×
  5–15 phrases).

**RE-H4: Trump nickname recency hazard (burst persistence, two-sided)**
- Test: per-nickname hazard by recency bucket (≤3d, 4–14d, 15–30d, >30d); predict
  h(≤3d)/h(>30d) ≥5x; market test: logistic resolution ~ logit(price) + recency buckets,
  N≥100; backtest both sides at ±(fee+3pts) disagreement.
- Kill: hazard ratio <3x; buckets insignificant beyond price (N≥80); **no free enumerable
  Trump corpus with <24h latency** (Factbase JSON gated; page enumeration unproven) —
  the only hypothesis with an unresolved data-access blocker.
- Est. edge: dormant-NO 10–20c gross (YES priced 25–40c vs true 5–10%); burst-YES 5–10c;
  net ~5–15c.

**RE-H5: Count-threshold (qualified) markets — model the count distribution** *(≡ T-H6)*
- Mechanism: the selection effect (RE-H2) creates this trade — by qualifying sure phrases
  into count markets, Kalshi moves the game to where transcript holders have the largest
  informational advantage.
- Test: NB fit on prior 8+ per-call counts (offset by call length); model Brier beats price
  Brier ≥15% relative (paired bootstrap, p<0.05, ≥50 settled qualified markets); >5pt
  disagreements resolve model-favorably >55%.
- Kill: no Brier win; dispersion blows up (posterior predictive spans 0–10x); settlement
  counts diverge from transcript counts by >1 on >20% of audits.
- Est. edge: 3–8c near 30–70c (where fees are highest — needs the disagreement filter).

### Data sources
Motley Fool (verified, pagination through page 8; full counts extractable), Rev.com
(verified per-page), Roll Call Factbase (pages verified incl. Apr-22-2026 Leavitt briefing;
JSON API gated; robots.txt disallows AI-training bots — scrape respectfully), Kalshi public
API (verified: settled SECPRESS events + live TSLA event with qualifier evidence),
API Ninjas (UNVERIFIED), UCSB Presidency Project (UNVERIFIED for 2026 coverage).

---

# CROSS-FAMILY SYNTHESIS

## A. The cost stack (binding for every backtest — from EFFICIENT-MARKET SKEPTIC, computed this round)

**Taker fee per contract at C=100** (fee = ceil(0.07·C·P·(1−P)) cents):

| price | 2c | 5c | 10c | 20c | 50c | 80c | 95c |
|---|---|---|---|---|---|---|---|
| fee/contract | 0.14c | 0.34c | 0.64c | 1.13c | 1.76c | 1.13c | 0.34c |

- Ceil rounding makes small lots brutal: C=1 pays ≥1c at ANY price (20% of a 5c stake),
  2c flat across 20–80c. Retail is structurally overtaxed; size is not.
- **Breakeven true-prob edge vs ASK as taker = fee only** (hold to settlement; settlement is
  free; never round-trip — exiting as taker doubles fee AND pays the spread again).
- **Breakeven vs MID = half-spread + fee.** At observed 1c/2c/4c spreads: 0.87/1.40/2.46c at
  5c mid; **2.25/2.75/3.75c at 50c mid**; 0.81/1.27/2.21c at 95c mid.
  ⇒ Any taker strategy needs ≥2.5–3.75c of true edge at mid prices but only ~0.8–1.4c at
  the extremes. **The fee curve says: trade the tails.**
- Maker breakeven = adverse-selection cost only (maker fee 0) — an empirical markout
  question (SK-H2).
- Live spread/depth ground truth (SECPRESS T-1d, verified): median spread 1c, mean 2c,
  max 5c; top-of-book depth median ~40–150 contracts (range 1–3,900).

**Backtest fill accounting (mandatory):**
- Fills at hourly/minute candle **yes_ask close** for YES buys, **(100 − yes_bid close)**
  for NO buys. Never trade-price, never mid, never price.mean. A strategy that only
  survives at mid is dead.
- Worst-case sensitivity: next hour's yes_ask HIGH.
- Filter placeholder no-book candles (bid=0 or ask≥99; real example: first FDX candle 3/84).
- Fees with ceil at actual lot size. Size cap min(100, 10% of next-24h volume_fp).
- **Cluster ALL inference by event** (median 17 markets/event share one transcript;
  market-level iid bootstrap badly overstates significance). Where the same phrase repeats
  across companies/events, also check phrase-level clustering.
- Chronological 60/40 train/confirm split by event close date; event-block bootstrap
  (10k resamples) for every P&L CI; **BH-FDR q=0.10 applied ONCE across the union of all
  pre-registered cells campaign-wide**, not per hypothesis.
- Early-close lookahead guard: every decision snapshot must precede
  min(decision time, early close) or the backtest buys YES on already-resolved markets.

**Capacity reality:** realistic extraction $3–30/market pre-event, $15–100/event in-event —
a **$50–300/day operation at current liquidity**. Sports-mention series (WCMENTION median
17k volume) are where capacity lives if calibration edge generalizes. Fast capital
recycling (median lifetime 2.9 days) partially compensates.

**Where edge CANNOT exist** (skeptic's dead zones): mid-priced (30–70c) taker bets from
soft priors (need >7% relative edge); high-volume sports mentions near event (1c spreads,
MM present); anything requiring exit before settlement; 1-lot sizing.

## B. Testable with ONLY the settled Kalshi dataset (events + resolutions + candles + tape)

**Kalshi-data-only (run first — no scraping, no external corpus):**
- MS-H1/SK-H1 (bucket calibration), MS-H2 (basket overround), MS-H4 (OFI reversal),
  MS-H5/SK-H5 (listing drift — statistical form), MS-H3 (hope premium — tape version,
  using pre-event price as p₀)
- SK-H2 (maker markout — needs tape phase 3)
- SK-H3/HZ-H6 capacity half (snipeable volume from tape; the *latency* half needs captions)
- XM-H1 (containment scanner), XM-H3 (peer early-settlement repricing — settlement_ts is
  in-dataset), XM-H5 (series persistence), XM-H6 (topic-basket laggards; φ clusters from
  outcomes)
- EM-H4 (phrase-type miscalibration — local embeddings only, cheapest model hypothesis)
- RE-H2 parts 1–2 (qualifier taxonomy from rules text; novel-phrase selection effect)

**Need external transcript corpora** (defeatbeta / Motley Fool / Factbase / Rev / captions):
- All of TRANSCRIPT (T-H1..H6), EM-H1/H3/H5, XM-H4, RE-H1/H3/H4/H5, RE-H2 part 3 (streak
  regression), HZ-H1/H2/H3/H4 (mention timestamps / Q&A boundaries / word counts),
  HZ-H5 (event-end times), SK-H4.

**Need external news/attention data** (GDELT raw / Wikipedia / EDGAR / RSS):
- All of NEWS-FLOW (NF-H1..H5), EM-H2.

Dependency ordering: phase-2 candles unblock the entire first group; phase-3 tape unblocks
SK-H2/SK-H3/MS-H3/MS-H4; transcripts unblock the second; GDELT sync unblocks the third.

## C. Overlaps / duplicates (merge before round 2)

| Canonical test | Duplicated in | Resolution |
|---|---|---|
| Price-bucket calibration / longshot fade | MS-H1 ≡ SK-H1 | One test, one FDR grid. SK adds fee-asymmetry rationale + 2026H1-subsample kill. |
| Listing-window mispricing | MS-H5 ≡ SK-H5 | Merge; MS supplies statistical form, SK supplies the model-conditioned trading form. |
| In-event hazard under-decay | MS-H3 ≈ HZ-H1/H2 | HZ owns it (minute candles + transcript timestamps); MS-H3's tape-only version is the corpus-free pilot. |
| Live utterance snipe | HZ-H6 ≡ SK-H3 | One latency/capacity study off the tape; build nothing unless it passes decisively. |
| Speaker base-rate model | T-H1 ≡ EM-H1 ≈ XM-H4 ≈ RE-H1 | ONE model (EM-H1's encompassing-test harness is the evaluation standard; T-H1 supplies the estimator; XM-H4/RE-H1 are its tail/streak trading expressions). Direct descendant of the repo's paper-only `speech_mention` prototype. |
| Near-certain repeats / deep-ITM YES | T-H2 ≡ SK-H4 ≡ RE-H1 | Same trade; gated by RE-H2 audit; SK-H4's 348/348=50% finding is the prior-killer. |
| Count-threshold NegBin | T-H6 ≡ RE-H5 | One implementation (variant flag on the base-rate strategy). |
| Salience fade | T-H3 ≈ EM-H2 ≈ NF-H4 | Distinct features (history-zero / RSS z / GDELT decay) but one economic claim — test jointly, report which feature survives conditioning. NF-H1 is the *opposite-sign* shock-buy: same pipeline, both cannot be independently "confirmed" without conditioning on each other. |
| Recurring-speaker persistence | XM-H5 ≈ RE-H3/H4 ≈ T-H5 | XM-H5 has an explicit fold-into-base-rate kill path; keep RE-H4 (nickname hazard) separate only if the Trump-corpus enumeration blocker resolves. |
| Sector/peer contagion | T-H4 ≈ XM-H2/H3 | XM-H3 (early-settlement repricing) is Kalshi-only and sharpest; T-H4/XM-H2 are its pre-event covariate forms. |
| Verbosity/length covariate | HZ-H3 ≈ RE-H3(3) | One covariate, two families — estimate once. |

## D. Campaign-strategy consequences

1. **RE-H2 (adversarial-selection audit) and MS-H1/SK-H1 (calibration grid) run first** —
   they are gates: RE-H2 decides whether the entire transcript-carry cluster has inventory;
   the calibration grid decides whether any prices-only fade exists and calibrates every
   family's "vs price" baseline.
2. **Everything trades against ask/bid candles** — the last_price degeneracy means no
   result from the phase-1 snapshot alone is evidence.
3. **The encompassing test is the campaign-wide bar** (EM family): a model only lives if
   blend(model, market) beats market alone OOS on event-clustered log-loss AND the
   after-ceil-fee P&L at ask fills is positive. "Predicts mentions well" is worthless.
4. **Environment**: api.gdeltproject.org, web.archive.org TLS-reset from this WSL host —
   run those collectors from the Fly box; data.gdeltproject.org (plain HTTP), wikimedia,
   efts.sec.gov, news.google.com, api.elections.kalshi.com all work locally.
5. **Capacity ceiling (~$50–300/day)** means infra spend must stay proportionate; the
   high-frequency recurring series (SECPRESS 1–3x/week, earnings seasons) and sports series
   are where any surviving edge compounds.
