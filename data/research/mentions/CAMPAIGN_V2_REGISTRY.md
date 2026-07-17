# Mentions ML Campaign V2 - Approach-Family Registry

Updated: 2026-07-17. Scope: political speeches, briefings, hearings, and
political interviews only. Success requires a leakage-safe out-of-sample edge
over the contemporaneous Kalshi market after executable costs and an
independent adversarial audit.

## Evidence bar

- Prediction time and every feature must be reconstructable at that time.
- Train/calibration/test partitions are grouped by event and ordered in time.
- Market-only probability and executable buy-and-hold are mandatory baselines.
- P&L uses the available side of the book, applicable fees, and a defensible
  fill model. Joined-book maker fills require queue evidence.
- Inference is clustered by event and day, reports effect size and confidence
  interval, and corrects for all searched specifications.
- A family is promoted only after a fresh auditor attempts to refute it.

## Family registry

| Family | Causal mechanism | Current state | Next discriminating test |
|---|---|---|---|
| Historical lexical base rates | Speaker and event-type phrase recurrence not fully incorporated by traders | Blocked in prior campaign: price encompassed persistence and transcript models | Reopen only with political-only corpus, strict prior-speech cutoff, and temporal holdout |
| Semantic / embedding retrieval | Event topic and phrase semantics identify analog speeches better than literal counts | Active, independent wave | Frozen embedding or sparse-text model versus price-only on grouped temporal OOS |
| Pre-event residual calibration | Retail favorite/longshot or midpoint bias remains after conditioning on time and series | Blocked as a tradable rule in prior campaign | Reopen only if actual books and a holdout beat fees |
| In-event survival / theta | Conditional on no mention yet, YES probability should decay as event time is consumed | Active lead, not yet accepted for political-only scope | Causal scheduled-time political-only train/test with actual taker book and ML probability calibration |
| Live transcript latency | Speech text reveals topic/absence before the book reprices | Underexplored | Timestamped captions aligned to trade/book updates; measure capturable latency |
| Maker queue / passive NO | Entertainment YES flow leaves profitable resting NO bids | Killed: zero-queue fill proxy overfilled profitable thin prints and under-modeled losing sweeps | Reopen only with observed live queue position and fill outcomes |
| Cross-contract logical constraints | Related word contracts violate implication, containment, or event-level probability bounds | Blocked: too few containment pairs; basket overround lost after execution | Reopen for ontology/semantic implications with executable simultaneous books |
| Cross-event / speaker transfer | Peer speakers or repeated event formats lead laggards | Underexplored | Walk-forward hierarchical residual model, holding out whole speakers and dates |
| Order-flow microstructure | Taker-side bursts, quote staleness, or spread state predict residual repricing | Active, independent wave | Decision-time-only features and actual next-book execution on political subset |
| Rules / settlement NLP | Contract phrase matching differs from naive transcript matching | Underexplored | Parse official rules, build auditable matcher, quantify disagreement before modeling |
| External agenda / news | Prepared remarks, schedules, bills, and contemporaneous news shift word hazards before traders react | Underexplored | Timestamped agenda/news features with source publication-time audit |
| Count-threshold model | Negative-binomial word counts improve P(>=N) over binary models | Out of current binary-only scope | Preserve for count Mentions follow-up |

## Binding blocked-route ledger

1. **Joined-level maker NO, clips 10-100.** Killed by queue-position bias.
   Requiring contracts to print ahead of the simulated order reduced the
   reported edge from +10.17c at zero queue to +1.21c at 50 contracts and
   -7.47c at 200 contracts. No maker claim may reuse zero-queue fills.
2. **Earnings transcript base rate.** Out of scope and empirically blocked:
   market Brier 0.174 versus model 0.203; threshold trading lost after costs.
3. **Resolution-history persistence.** AUC existed but added no information
   beyond price (encompassing coefficient p=0.54); threshold P&L was negative.
4. **Simple pre-event calibration fade.** Bucket miscalibration did not clear
   fees and spread on executable rules.
5. **Event-basket NO overround.** Midpoint overround disappeared at executable
   bids and fees.
6. **All-series late-event taker headline.** Statistically strong in the
   recovered audit, but not yet a qualifying result: the specification was
   selected after a broad grid search, the reported aggregate includes sports
   and media, and the political-speech-only ML comparison is not frozen or
   independently held out.

## Active wave registry

| Worker | Independent brief | Family | Status |
|---|---|---|---|
| speech_semantic_models | Historical political-speech NLP and embeddings | Semantic / retrieval | Running |
| mentions_microstructure | Transaction systems and book mechanics | Order flow / execution | Running |
| mentions_efficiency_stats | Efficient-market nulls and rigorous inference | Calibration / statistics | Running |
| root | Reproduce prior audits and construct political-only causal panel | In-event survival / theta | Running |

