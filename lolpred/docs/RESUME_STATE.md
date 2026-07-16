# Resume state — edge-search campaign

**OBSOLETE (2026-07-16): campaign CONCLUDED.** The fetch completed (CS2MAP
99.7%, VAL/Dota 100%), M3 ran once on the full pre-registered universe, and
the else-branch of the decision rule fired (INSUFFICIENT_N: 19/19 wins but
exact clustered p 0.359, negative flip-haircut EV). See the FINAL REPORT in
`docs/EDGE_SEARCH.md`. Nothing below needs resuming.

---

Everything below survives reboot. Only running processes died at shutdown.

## Where things stand

Full campaign log: `docs/EDGE_SEARCH.md` (registry + M3 pre-registration —
READ IT FIRST; the pre-registration is binding, no retuning). Summary: 13
families dead with rigor; settlement-drift (M/M2) REFUTED by adversarial
audit (degenerate bootstrap p + pause-blind-clock look-ahead; honest replay
= zero edge); acceptance machinery hardened (`lolpred/backtest/edge_protocol.py`
now has exact clustered p + flip-rate haircut; suite 283 green). One
pre-registered candidate remains: **M3** (sibling map-book snap as game-end
clock for CS2/VAL/Dota match-book drift).

## What was interrupted

1. **Kalshi data fetch** (incremental, idempotent). Complete: all LoL series
   (match/map/totals) prices+trades+1m candles; KXCS2GAME complete;
   KXCS2MAP was at ~2,000/4,570 tickers. Missing: rest of KXCS2MAP,
   KXVALORANTGAME/MAP, KXDOTA2GAME/MAP micro.
   Resume: `cd lolpred && .venv/bin/python scripts/fetch_kalshi_multi.py`
   (it skips completed tickers; see the script's args — micro phases for the
   series above). Data lives in `data/odds/` (54MB, gitignored, on disk).
2. **M3 analysis agent**: its pipeline is at
   `/home/mcortese/.claude/jobs/efe2b3d4/tmp/m3_run.py` (+ m3_bets.parquet
   preliminary at 40% coverage). Once KXCS2MAP micro is >=95% complete,
   run `m3_run.py` (or relaunch an agent with docs/EDGE_SEARCH.md's M3
   pre-registration as the brief). Judge ONLY with the post-fix
   `evaluate_frozen_rule` (n_families_tested=15, n_variants_in_family=24)
   and apply the pre-registered decision rule verbatim.

## Decision rule (pre-committed, do not renegotiate)

M3 SIGNIFICANT under the fixed evaluator + positive flip-haircut EV + no
anti-hindsight contradiction => strategy stands, subject to one final
independent audit. Anything else => final report = strongest proven
derivation + exact gap (drafted in EDGE_SEARCH.md audit row: the
settlement-latency mechanism is real and executable per audit-2, but proving
positive expectancy needs either ~200 all-win fixture-clusters under a
trustworthy clock, or forward paper trading with a live game-end feed,
which public historical data cannot simulate).

## Job-scratch inventory (survives on disk until the bg job is deleted)

`/home/mcortese/.claude/jobs/efe2b3d4/tmp/`: all audit forensics
(xaudit_*.py, dq_audit/), family artifacts (map_drift_*.parquet,
FROZEN_RULE*.md, study.parquet...), m3_run.py. If the job dir is gone,
everything essential is reproducible from `data/odds/` + the registry docs.

## Also parked (unrelated to edge search)

- The Odds API key (500 credits, unused — no esports coverage) at
  `lolpred/secrets/odds_api_key` (gitignored).
- 2026 Oracle's Elixir file still Drive-quota-blocked
  (`scripts/download_data.py --source gdrive --years 2026` to retry).
- Draft PR #13 tracks the whole project; branch `feat/lolpred`.
