# hedge — Real-Money Go-Live & Full-Logging Plan

**Goal:** get the bot placing **real trades with real money** on Kalshi prod, while
**persisting everything it sees and does to the durable DB** so that every decision —
good or bad — is reconstructable after the fact and feeds a learning/adjustment loop.

Two intertwined tracks:

- **Track A — Observability:** make the SQLite DB (`/data/hedge.db` on the Fly volume)
  the single, canonical, queryable record of *every* signal, decision, order, fill,
  and settlement. This is the substrate we learn from. **This is the priority** — we
  do not want to discover a class of mistakes only to find we never logged the inputs.
- **Track B — Go-live:** the safe, gated sequence that flips the bot from demo to
  prod with tiny caps.

> **Sequencing rule:** Track A ships and bakes on demo *before* Track B flips to prod.
> Real money with incomplete logging is the one thing we refuse to do — a loss we
> can't explain is a loss we can't fix.

---

## Current state (what's already true)

- ✅ Full pipeline wired: signal → `decide()` → `Executor` → V2 order → reconcile.
- ✅ Durable DB (`hedge/state.py`): `orders`, `daily_pnl`, `settled`, `meta` tables.
- ✅ Fill reconciliation via `get_orders()` LIST; cycle-seq idempotency; anti-stack.
- ✅ Daily-loss stop, Brier kill-switch (latching), validated-station gate (CHI/MIA/AUS).
- ✅ Containerized + on Fly (demo). Caps baked in `deploy/config.yaml` (λ=0.10, $25/order).
- ⚠️ **Decisions log to JSONL files, not the DB.** `_write_log()` → `decisions_*.jsonl`.
- ⚠️ **P&L is booked from *intent*, not fills.** `_book_settlements()` reads the
  decision log's `price_cents`/`count`, not the actual fill price/size from the broker.
- ⚠️ **No market-quote snapshot persisted** with each decision (can't reconstruct "why").
- ⚠️ **Signal/MC inputs not persisted** (forecast, dispersion, bias — the model's reasons).

---

## Phase 0 — Prerequisite: a healthy demo soak is actually running

Nothing below matters if the machine isn't looping. Confirm first:

```bash
fly status                                        # one machine, started, not autostopped
fly logs                                          # steady cycles, no crash loop
fly ssh console -C "cat /data/status.json"        # mode=live-demo, halted=false
```

If the machine is still wedged (the "configuring firecracker" hang), resolve that
first (see RUNBOOK §3 + the volume-attach diagnosis). **Gate:** ≥24h of clean demo
cycles before touching prod.

---

## Phase 1 — Persist EVERYTHING to the DB (the learning substrate) — ✅ **DONE**

Goal: after this phase, a single SQLite file answers "what did the bot believe, decide,
do, and get, for every market, every cycle?" — with zero reliance on JSONL scraping.

**Status: implemented and tested** (`hedge/state.py`, `hedge/runner.py`,
`scripts/db_report.py`, tests in `tests/test_state.py` + `tests/test_execution_runner.py`).
The `decisions` and `fills` tables exist; settlement P&L is booked from actual fills;
the query CLI is live. What remains is operational: let it bake on demo (Phase 0/3) and
eyeball the calibration view. Details of what shipped below.

### 1.1 New `decisions` table (`hedge/state.py`)
One row per market per cycle — **including HOLDs** (a HOLD with its reason is exactly
what we review when we think we *should* have traded). Schema:

| column | source | why we need it to learn |
|---|---|---|
| `cycle_seq`, `ts`, `utc_date` | runner | join key + time-series |
| `ticker`, `strategy` | decision | what & who |
| `action`, `side`, `count`, `price_cents` | decision | intended order |
| `prob`, `sigma` | signal | the model's belief + its honesty |
| `edge`, `kelly_fraction` | decision | sizing rationale |
| `yes_bid`, `yes_ask`, `mid`, `last` | `MarketQuote` | the market we were pricing against |
| `bankroll`, `portfolio_at_risk` | runner | risk context at decision time |
| `placed`, `dry_run`, `error` | ticket | did it go through |
| `reason` | decision | why HOLD / why this side |
| `meta_json` | signal.meta | **forecast, dispersion, bias, max-so-far — the model's inputs** |

Add `State.record_decision(**row)`; call it from `run_cycle` where `_log_row` /
`_write_log` are today. **Keep the JSONL write too** (cheap redundant export, and the
guard already reads it) — but the DB becomes canonical.

### 1.2 New `fills` table — actual broker truth, not intent
`_reconcile_orders()` already pulls the order LIST and `parse_order()` gives
`fill_count`. Extend to also capture **average fill price** and **fee** per order
(from the Kalshi order fields, and/or `GET /portfolio/fills`). New table `fills`:
`client_order_id, order_id, ticker, side, fill_count, avg_price_cents, fee_cents, ts`.
Add `State.record_fill(...)`; write it in `_reconcile_orders` whenever
`status == "executed"` (or partial).

### 1.3 Book P&L from fills, not from the decision
Rewrite `_book_settlements()` to compute realized P&L from the **`fills` table**
(actual entry price × actual size − actual fee), gated by `settled` outcome — instead
of the intended `price_cents`/`count` in the decision log. Store the realized outcome
(`yes`/`no`), entry, and P&L per ticker so the `settled` row is a complete trade record.
This is the difference between "we think we made $X" and "we made $X."

### 1.4 Query/export helper (`scripts/db_report.py`, new)
A read-only CLI to pull the learning views without SSH-ing into sqlite by hand:
- `decisions --day YYYY-MM-DD` — every decision that day (table).
- `trades` — placed orders joined to fills + settlement outcome + realized P&L.
- `calibration` — predicted `prob` vs realized outcome (reliability bins, Brier) by
  strategy/city — **the core "are we right?" view.**
- `--csv` flag to dump for offline analysis.

### 1.5 Tests
Extend `tests/test_state.py` + `tests/test_execution_runner.py`: assert a full cycle
writes a `decisions` row per market (incl. HOLD), a partial fill writes a `fills` row,
and a settled winner books fill-based P&L. **Gate:** `pytest` green before deploy.

> **Deliverable of Phase 1:** demo runs for a few days; `scripts/db_report.py calibration`
> produces a real reliability table from demo decisions. If that table looks sane on
> demo, we trust the logging on prod.

---

## Phase 2 — Reconciliation hardening for real money

Before real fills depend on it:

- **Positions as truth each cycle.** `positions()` already reads `get_positions()`;
  confirm P&L/exposure caps key off broker positions, not just our `orders` table,
  so a fill we missed can't desync us.
- **Partial fills.** Ensure a partially-filled-then-canceled maker books P&L for the
  *filled* portion (via the `fills` table from 1.2), not all-or-nothing.
- **Fee realism.** `tournament/paper.py:taker_fee` uses the 0.07 coefficient; for prod
  P&L accuracy, key the fee off the actual `fee_cents` captured in `fills` when present,
  falling back to the formula only for unsettled estimates.
- **Idempotency under retry.** Already cycle-seq keyed; add a test that a simulated
  network retry within one cycle reuses the `client_order_id` (no double-fill).

---

## Phase 3 — Prod-readiness gates (the demo soak checklist)

Do **not** advance until *all* hold on demo (mirrors RUNBOOK §4, now with DB checks):

- [ ] ≥24–48h of clean cycles, no crash loop (`fly logs`).
- [ ] `decisions`, `fills`, `settled` tables populating every cycle (`db_report.py`).
- [ ] At least one demo settlement booked P&L **from fills** and it matches by hand.
- [ ] Calibration table on demo decisions is sane (no wildly over-confident `prob`).
- [ ] Restart recovery: `fly machine restart` → status.json + open orders survive.
- [ ] Alerts fire: forced guard trip → CRITICAL ntfy push.
- [ ] Caps verified in logs: orders ≤ $25, λ=0.10, only CHI/MIA/AUS, climatology silent.

---

## Phase 4 — Prod go-live (tiny), with logging proven

Prod creds are **separate** from demo (generate a prod API key in the Kalshi UI; save
its PEM). Then (RUNBOOK §5):

```bash
fly secrets set KALSHI_KEY_ID="<PROD-key-id>"
fly secrets set KALSHI_PRIVATE_KEY="$(cat secrets/kalshi_prod_private_key.pem)"
fly secrets set KALSHI_ENV=prod
fly secrets set KALSHI_BASE_URL="https://external-api.kalshi.com/trade-api/v2"
fly deploy
```

Then **fund ~$100** and verify in `fly logs` + the session-start alert:
- env=prod, λ=0.10, `max_order=$25`, `daily_stop=$50`, guard=on.
- Orders only on validated cities; the station gate now actively blocks NYC (real money).
- First real settlement: `db_report.py trades` shows entry fill, outcome, realized P&L.

> `KALSHI_ENV=prod` is the real switch (the image's CMD already has `--allow-prod`).
> Step 4 is the single deliberate go-live action. Start with caps as-is; only raise
> them after a week of profitable, well-logged trades.

---

## Phase 5 — Operate & learn (the point of all the logging)

Daily/weekly loop, all driven off the DB:

1. **`db_report.py trades`** — every real trade, entry, outcome, P&L, fees.
2. **`db_report.py calibration`** — predicted vs realized by strategy/city. Drift in a
   city's reliability curve = a model or **station** problem; investigate before sizing up.
3. **Score the kill-switch margin** — how close did realized Brier get to the 0.20 trip?
4. **Adjust deliberately** — calibration fixes go through the backtest/paper tournament
   first (CLAUDE.md house rule), never hand-tuned on prod. Raise caps only on evidence.
5. **Compound** — bankroll is read live from Kalshi; profits compound automatically
   under the same caps (no withdrawals).

---

## What can lose money, and how the plan mitigates it

| Risk | Mitigation in this plan |
|---|---|
| Biased model (wrong `p`) bleeds under Kelly | Phase 1 calibration logging + Brier kill-switch; adjust via backtest only |
| Wrong settlement station → confident-but-wrong | Validated-station gate (real money = CHI/MIA/AUS only); calibration-by-city catches drift |
| Believing intent instead of fills | Phase 1.2–1.3: P&L booked from the `fills` table, not the decision |
| Runaway sizing | $25/order, 3%/market, 30%/portfolio, $50/day stop — baked, secrets-free |
| Silent desync after a missed fill | Phase 2: positions-as-truth each cycle + partial-fill accounting |
| A loss we can't explain | The whole of Track A: every input + decision + outcome in one queryable DB |

---

## Ordered task checklist

**Track A (code — ✅ done, green):**
1. ✅ `state.py`: `decisions` + `fills` tables, `record_decision` / `record_fill`,
   richer `settled` row, `order_for_oid`.
2. ✅ `runner.py`: a `decisions` row per market per cycle (incl. HOLD), with the
   `MarketQuote` snapshot + `signal.meta`.
3. ✅ `runner.py`: `_reconcile_fills()` pulls `GET /portfolio/fills` → avg entry price +
   taker fee into `fills` (covers immediate IOC fills *and* later maker fills).
4. ✅ `runner.py`: `_book_settlements` computes P&L from `fills` (filled size × actual
   entry), not intent — an unfilled order now books $0.
5. ✅ `scripts/db_report.py`: decisions / trades / calibration / pnl (+ `--csv`, `--db`).
6. ✅ Tests green (`tests/test_state.py`, `tests/test_execution_runner.py`; full suite 109).

**Track A (operational — next, on demo):**
7. Deploy to **demo**; soak; confirm `decisions`/`fills`/`settled` populate and
   `db_report.py calibration` looks sane before flipping to prod (Track B).

**Track B (after the demo gate):**
8. Pass the Phase 3 checklist.
9. Generate prod key, save PEM, fund ~$100.
10. Swap secrets to prod + `fly deploy` (Phase 4).
11. Watch the first real trades + first settlement in `db_report.py`.
12. Begin the Phase 5 operate/learn loop; raise caps only on logged evidence.
