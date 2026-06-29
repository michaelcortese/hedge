# hedge — 24/7 Autonomous Deployment Plan (Fly.io + Kalshi prod)

> Goal: run the `hedge` weather-trading bot unattended, 24/7, on Fly.io, trading
> **real money on Kalshi production**, managing its own state, risk, and alerting
> with no human in the loop.
>
> This plan is grounded in a source-cited deep-research pass (primary docs for
> Fly.io, Kalshi, Pushover/ntfy — see **Sources**) plus a prior audit of this
> codebase. Operator decisions baked in: **prod now with tiny caps**, **conservative
> envelope** (~$100 bankroll, $25/order, ~$50/day loss stop, λ=0.10), **compound the
> bankroll** (no auto-withdrawals), **push alerts** on every critical event.

---

## 1. Executive summary

Running `hedge` 24/7 on Fly.io is **feasible and cheap (~$2–3/month)** — but the
*infrastructure is the easy part*. The dominant risk is **application-level
correctness**: the bot currently runs as an in-memory foreground loop with no
durable state, no fill reconciliation, a coarse idempotency key, no daily-loss
stop, and no alerting. **None of those may be skipped before real money**, because
each is a way to silently lose funds or lose track of what you hold.

The good news: the hard trading logic is already built and tested — a wired
decision→sizing→executor→order path, fractional-Kelly with fee/edge math, a
latching Brier-drift kill-switch, per-market/portfolio/absolute-$ caps, a
validated-station gate (only **CHI/MIA/AUS** trade real money; NYC is still
unvalidated), and a non-tradeable-quote guard. This plan closes the remaining
gaps in a strict, money-safety-first order, then containerizes and ships.

**Critical path to a safe launch:** durable state → fill reconciliation → daily-loss
stop → alerting → containerize → Fly demo soak (24–48h) → promote to prod tiny.

Fly natively supports the "worker without HTTP" shape: an app with **no
`[http_service]`/`[[services]]`** is unreachable from the internet (correct for a
loop), and a **`[[restart]] policy = "always"`** keeps a crashed *or cleanly-exited*
worker alive [fly-config, fly-restart]. State persists on a **Fly Volume**
($0.15/GB-mo) [fly-pricing]. The whole thing is a single `shared-cpu-1x`/256 MB
Machine (~$2.02/mo) [fly-pricing].

---

## 2. Current state (already built — do not rebuild)

| Capability | Where | Status |
|---|---|---|
| Decision → sizing → executor → order | `hedge/decision/engine.py`, `hedge/execution/executor.py` | ✅ wired, tested |
| Fractional-Kelly + edge/fee math | `hedge/decision/{sizing,edge,fees}.py` | ✅ |
| Brier-drift kill-switch (latching, trips at 0.20) | `hedge/guard.py`, `config.yaml` | ✅ |
| Per-market / portfolio / absolute-$ caps | `hedge/decision/{config,engine}.py` | ✅ (`max_order_dollars=25`) |
| Validated-station gate (CHI/MIA/AUS only) | `hedge/runner.py`, `hedge/weather/stations.py` | ✅ |
| Non-tradeable-quote guard | `hedge/decision/engine.py`, `hedge/tournament/paper.py` | ✅ |
| RSA-PSS auth (salt=32) | `hedge/kalshi/auth.py` | ✅ matches Kalshi spec [kalshi-auth] |
| Keyless read-only prod data | `hedge/kalshi/client.py` (`read_only`) | ✅ |
| Forward paper tournament + backtest | `hedge/tournament/`, `scripts/` | ✅ |

### Gaps this plan closes (from the audit)
1. **No fill/partial-fill reconciliation** after `create_order` — the bot doesn't read the order's fill state back.
2. **Coarse per-day idempotency key** — a same-day re-run at a different price/count can stack orders.
3. **No durable state** — runs entirely in memory; a restart forgets open orders, daily P&L, and the guard latch.
4. **No baked-in scheduler** — it's a foreground loop, fine for a worker but needs a separate trigger for post-settlement scoring.
5. **No monitoring / alerting** — if it halts at 3am, nobody knows.
6. **No daily-loss stop** — only per-trade and exposure caps exist.
7. **Not containerized / deployed.**

---

## Implementation status (updated)

**Phases 1–5 are built and tested (96 tests passing); the container builds and runs
end-to-end against demo.** Remaining work is operator-run deploy — see
[`RUNBOOK.md`](RUNBOOK.md).

| Phase | Status | Artifacts |
|---|---|---|
| 1 — Durable state | ✅ done | `hedge/state.py` (SQLite), `tests/test_state.py` |
| 2 — Fill reconciliation + idempotency | ✅ done | `executor.parse_order`, `client.get_order`, runner `_reconcile_orders`/`_record_ticket` + anti-stack |
| 3 — Daily-loss stop | ✅ done | `RiskConfig.daily_loss_stop_dollars`, runner latch ($50) |
| 4 — Alerts + status file | ✅ done | `hedge/alerts.py` (ntfy), wired to trip/daily-stop/error/heartbeat/start; `/data/status.json` |
| 5 — Containerize | ✅ done | `Dockerfile`, `.dockerignore`, `fly.toml`, `deploy/config.yaml`, PEM-from-env shim |
| 6 — Fly demo soak | ⏳ operator | `RUNBOOK.md` §1–4 |
| 7 — Promote prod tiny | ⏳ operator | `RUNBOOK.md` §5 |

Two money-safety bugs were caught and fixed during Phase 5 container testing:
(a) the container had no `config.yaml` so it ignored all caps (a 1¢ market sized a
7,500-contract order) — fixed by baking the secrets-free `deploy/config.yaml`;
(b) `weather_climatology` (the null benchmark) was placing live orders — removed
from the live strategy set.

Two more were caught on the **first Fly demo deploy** and fixed:
(c) **Kalshi deprecated the V1 order endpoints (HTTP 410)** — migrated create/cancel
to the V2 `/portfolio/events/orders` family (YES-priced `bid`/`ask` model; buy NO =
`ask` at `1 - no_price`), verified live against demo; reads stay on `/portfolio/orders`
and reconciliation uses the order LIST (single-order GET is eventually-consistent);
(d) the per-day idempotency key was too coarse — Kalshi burns a `client_order_id`
forever (even after cancel), so a cancel-replaced market 409'd all day; the key now
folds in the monotonic cycle sequence (`hedge/runner.py:_idem_key`), and a 409 is
treated as a benign idempotent skip (`executor.place`).

## 3. Phased roadmap (ordered by what unblocks a safe real-money launch)

> Each phase ends with a **gate** that must pass before the next. Phases 1–4 are
> code; 5 containerizes; 6 ships to Fly on **demo**; 7 promotes to **prod tiny**.

### Phase 1 — Durable state (crash recovery) · **blocks everything**
A restart must not lose track of open orders, the guard latch, or the day's P&L.

- **New `hedge/state.py`** — a thin SQLite store on the Fly Volume (`/data/hedge.db`).
  Tables: `orders` (client_order_id PK, ticker, side, price_cents, count,
  fill_count, status, ts), `decisions` (logged signals/decisions), `daily_pnl`
  (utc_date PK, realized, fees), `guard` (latched bool, tripped_ts, reason),
  `cycle_seq` (monotonic counter). SQLite is the established autonomous-bot pattern
  (Freqtrade persists trades/orders/metadata and reloads on restart) [freqtrade-state].
- **`hedge/runner.py`** — load state on startup; persist after each cycle; treat
  Kalshi `/portfolio/{positions,orders,balance}` as **source of truth** and reconcile
  local state against it each cycle (broker truth, not memory) [kalshi-portfolio].
- **Gate:** kill the process mid-cycle, restart → it recovers open orders, the
  daily-loss tally, and a tripped guard latch with zero double-trades.

### Phase 2 — Fill reconciliation + idempotency · **blocks prod**
- **`hedge/execution/executor.py`** — after `create_order`, read the returned Order
  (`status` ∈ {resting, canceled, executed}, `fill_count_fp`, `remaining_count_fp`,
  `initial_count_fp`) and record it in `state` [kalshi-fills]. Add
  `client.get_order(order_id)` / `get_orders(status=...)` to `hedge/kalshi/client.py`.
- **`hedge/runner.py`** — each cycle, query `GET /portfolio/orders?status=resting`;
  cancel-replace stale resting makers (a maker that never fills is *not* a position).
  Reconcile actual positions from `GET /portfolio/positions` before sizing.
- **Idempotency fix** — replace the per-day key with a persisted
  `(ticker, side, price_cents, count, cycle_seq)` `client_order_id`; Kalshi rejects
  duplicates with `ORDER_ALREADY_EXISTS`, so a retry after a network blip can't
  double-fill [kalshi-idempotency]. Persist every placed `client_order_id` in
  `state.orders` so restarts don't reissue.
- **Gate:** simulate a partial fill and a network-retry on demo; positions in
  `state` exactly match `GET /portfolio/positions`; no duplicate fills.

### Phase 3 — Daily-loss stop · **blocks prod**
- **`hedge/decision/config.py`** — add `daily_loss_stop_dollars: float | None` to
  `RiskConfig` (set `50.0` in `config.yaml`).
- **`hedge/runner.py`** — at cycle start, compute the UTC-day realized P&L from
  `state.daily_pnl` (updated from `GET /portfolio/fills` + settlements); if cumulative
  loss ≥ stop, **latch a halt for the rest of the UTC day** (same mechanism as the
  guard) and fire an alert. Resets at UTC midnight.
- **Gate:** unit test that a day breaching the stop blocks all further BUYs and
  alerts, and that it clears the next UTC day.

### Phase 4 — Monitoring + push alerts · **blocks unattended operation**

**Channel decided: ntfy (free, no account). Topic: `hedge-alerts-7f3k9q2x`.**
Operator subscribed in the ntfy mobile app; a test push was confirmed delivered
(`POST https://ntfy.sh/hedge-alerts-7f3k9q2x` → received on phone). The topic name
*is* the password (anyone who knows it can publish/read), so it stays in
`fly secrets`, never in the repo or logs. Pushover remains the recommended upgrade
later for *emergency priority 2* (repeats until acknowledged) on the most critical
trips; switching is a secret change, no code [alerting].

- **New `hedge/alerts.py`** — one best-effort function
  `notify(level, title, msg)` that POSTs to the channel in `HEDGE_ALERT_URL`
  (or `alerts.url` in config.yaml), auto-detecting ntfy / Slack-webhook / Pushover.
  Requirements:
  - **Never raises** into the trading loop (a failed push must not crash/block a cycle); 5 s timeout.
  - Level → ntfy priority + emoji tag: `INFO`→3/ℹ️, `WARN`→4/⚠️, `CRITICAL`→5/🚨
    (ntfy sets `Title`, `Priority`, `Tags` headers).
  - Pushover path (if `HEDGE_PUSHOVER_TOKEN`/`HEDGE_PUSHOVER_USER` set instead of a URL)
    uses `priority=2, retry=60, expire=3600` for `CRITICAL`.
- **Wire alerts** in `hedge/runner.py` / `hedge/guard.py`: fire on (a) guard trip
  (`Runner._trip`, CRITICAL), (b) daily-loss stop (Phase 3, CRITICAL), (c) any
  uncaught cycle error (WARN), (d) drawdown > threshold (WARN), (e) a daily
  heartbeat "alive + bankroll + open positions" (INFO), (f) live-session start (INFO).
- **Status file** — write `/data/status.json` each cycle (last cycle ts, bankroll,
  open positions, guard state) for `fly ssh console` inspection.
- **Secret:** `fly secrets set HEDGE_ALERT_URL="https://ntfy.sh/hedge-alerts-7f3k9q2x"`.
- **Gate:** trip the guard on demo → push arrives within seconds; kill the worker →
  no heartbeat next day → you notice.

### Phase 5 — Containerize
- **New `Dockerfile`**, **`.dockerignore`**, **`fly.toml`** (see §4). Entry point runs
  the worker loop; the loop already exists in `hedge/runner.py` (no Fly scheduler
  needed for the main loop) [fly-schedule].
- **Gate:** `docker build` + local `docker run` against **demo** completes a cycle
  and writes state to a mounted volume.

### Phase 6 — Deploy to Fly on **demo** (24–48h soak)
- `fly launch --no-deploy`, create the Volume, set secrets (demo creds), deploy.
- Run **demo 24/7** for 24–48h. Watch: restart-survives-state, alerts fire, no
  crash loops, cycle latency healthy.
- **Gate:** 48h clean on demo — state durable across at least one forced restart,
  alerts verified, zero unhandled exceptions.

### Phase 7 — Promote to **prod tiny** + operate
- Swap secrets to **prod** creds + `external-api.kalshi.com` base [kalshi-base];
  set `KALSHI_ENV=prod`, executor `allow_prod=true`, **λ=0.10**, `max_order_dollars=25`,
  `daily_loss_stop_dollars=50`, fund the account ~$100.
- Validated cities only (CHI/MIA/AUS) — the station gate already enforces this.
- **Settlement scoring** runs daily after close (~04:59 UTC) via a separate
  **Scheduled Machine** or in-loop timer calling `run_paper score`/the guard updater
  [fly-schedule]. Compound: no withdrawals; exposure caps bound risk.
- **Operate:** daily heartbeat + the forward P&L track. Raise λ / caps only after
  realized prod P&L is positive over a few hundred contracts and the guard never trips.

---

## 4. Fly.io artifacts (copy-pasteable)

### `Dockerfile`
```dockerfile
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
COPY pyproject.toml ./
RUN pip install -e . || true
COPY . .
RUN pip install -e .
# Runs the live loop; --live arms real orders, --allow-prod gates prod (set via env/args).
CMD ["python", "-m", "hedge.runner", "--live"]
```

### `fly.toml`  (worker — **no `[http_service]`/`[[services]]` ⇒ private, off the public internet** [fly-config])
```toml
app = "hedge-bot"
primary_region = "iad"           # close to US Kalshi API

[build]

[env]
  KALSHI_ENV = "demo"            # flip to "prod" only at Phase 7
  HEDGE_STATE_DIR = "/data"

[[mounts]]
  source = "hedge_data"          # durable state + logs (SQLite, status.json)
  destination = "/data"

# Keep the single worker alive even on clean exit; no services to autostop. [fly-restart]
[[restart]]
  policy = "always"

[[vm]]
  size = "shared-cpu-1x"
  memory = "256mb"               # ~$2.02/mo [fly-pricing]
```

### Secrets + volume + deploy
```bash
fly launch --no-deploy --name hedge-bot --region iad
fly volumes create hedge_data --region iad --size 1      # $0.15/GB-mo [fly-pricing]

# Kalshi creds (RSA PEM as a multiline secret) + alert channel
fly secrets set KALSHI_KEY_ID="<uuid>"
cat secrets/kalshi_prod_private_key.pem | fly secrets import   # or: fly secrets set KALSHI_PRIVATE_KEY=- < pem
fly secrets set HEDGE_ALERT_URL="https://ntfy.sh/hedge-alerts-7f3k9q2x"   # ntfy topic (chosen); subscribe in the ntfy app

fly deploy
fly logs            # watch cycles
fly ssh console -C "cat /data/status.json"
```
> The code reads the PEM from `KALSHI_PRIVATE_KEY_PATH`; for Fly, add a tiny
> startup shim (or `hedge/config.py` branch) to materialize `KALSHI_PRIVATE_KEY`
> (the secret) to a file, or read the PEM contents directly from the env var.

### Settlement scoring (coarse interval is fine here)
Fly's built-in Scheduled Machines only do hourly/daily/weekly buckets (no cron
precision) [fly-schedule] — adequate for once-daily post-settlement scoring. Either
a daily Scheduled Machine running `python scripts/run_paper.py score --prod`, or an
in-loop timer in `hedge/runner.py` that runs scoring after ~05:00 UTC.

---

## 5. Risk & safety

**Defense in depth — every real order must pass all of these (most already built):**
1. **Validated-station gate** — only CHI/MIA/AUS trade real money; NYC blocked until validated.
2. **Non-tradeable-quote guard** — degenerate 0.00/1.00 books never become phantom edges.
3. **Per-order absolute cap** `$25` + per-market 3% + portfolio 30% of bankroll.
4. **Fractional Kelly λ=0.10** on a *conservative*, fee-netted, CI-haircut edge.
5. **Daily-loss stop** `$50` (Phase 3) — latches for the UTC day.
6. **Brier-drift kill-switch** — latches at realized binary Brier 0.20; manual reset only.
7. **Idempotent orders** (`client_order_id`) — retries can't double-fill [kalshi-idempotency].
8. **Broker-truth reconciliation** each cycle — positions/balance from Kalshi, not memory.
9. **Prod requires explicit `allow_prod`** + `KALSHI_ENV=prod`; demo is the default everywhere.

**Secrets hygiene:** RSA PEM and key id live only in `fly secrets` (encrypted at
rest, injected as env); never in the image, repo, or logs. `config.yaml`/`secrets/`
stay gitignored.

---

## 6. Go-live checklist (must all be ✅ before Phase 7)

- [ ] Phase 1–4 gates passed; `pytest` green incl. new state/fill/daily-stop/alert tests.
- [ ] Restart mid-cycle on demo → state fully recovered, zero double-trades.
- [ ] Partial-fill + network-retry on demo → local state == `GET /portfolio/positions`.
- [ ] Daily-loss stop blocks BUYs after breach and clears next UTC day.
- [ ] Guard trip + daily-stop + cycle-error + heartbeat all push alerts (verified).
- [ ] 24–48h demo soak on Fly clean (no crash loop, healthy cycle latency, alerts live).
- [ ] Prod creds set via `fly secrets`; `KALSHI_ENV=prod`, `allow_prod=true`, λ=0.10, `$25`/order, `$50`/day.
- [ ] Kalshi account funded ~$100; station gate confirmed limiting to CHI/MIA/AUS.
- [ ] `baseline_brier` refined from forward demo settlements (vs the 0.15 placeholder).
- [ ] Rollback understood: `fly deploy` is versioned; `fly releases` + redeploy prior image.

---

## 7. What could lose money — and the mitigation

| Failure mode | Mitigation |
|---|---|
| **Biased model `p`** (wrong, not just noisy) — Kelly loses fast | Backtest calibration + **forward P&L-vs-price must be positive before scaling**; kill-switch latches on drift; λ=0.10. |
| **Wrong settlement station** → confident-biased `p` | Validated-station gate; only CHI/MIA/AUS; NYC blocked. |
| **Phantom edge on a one-sided/locked book** | Non-tradeable-quote guard (engine + paper). |
| **Double-fill on retry / restart** | `client_order_id` idempotency + persisted in state; `ORDER_ALREADY_EXISTS` [kalshi-idempotency]. |
| **Believing a fill that didn't happen** (resting maker) | Phase-2 fill reconciliation reads `fill_count_fp`/`status`; positions from broker truth [kalshi-fills]. |
| **Crash forgets open orders / guard latch / daily P&L** | Durable SQLite state on Fly Volume; reload + reconcile on boot [freqtrade-state]. |
| **Runaway losing day** | Daily-loss stop `$50`, latched per UTC day. |
| **Silent halt at 3am** | Push alerts on trip/error/halt + daily heartbeat [alerting]. |
| **Thin/illiquid prod books** → bad fills | Maker-preference + depth cap + tiny size; abstain when no tradeable side. |
| **Fee/spread eats the edge** | Edge is netted of taker fee + spread before trading; `tau_min`. |
| **Infra outage / Machine dies** | `[[restart]] policy="always"` auto-heals; state on the volume survives [fly-restart]. |

---

## 8. Effort & cost

- **Engineering:** Phases 1–4 ≈ the real work (durable state, fill reconciliation,
  daily stop, alerts) — each is S–M; Phase 5–6 (containerize + Fly) ≈ S once code is
  ready. Estimate ~2–4 focused sessions to a safe demo soak.
- **Infra cost:** ~**$2.02/mo** worker + ~**$0.15/GB-mo** volume (first 10 GB
  snapshots free) ≈ **$2–3/month** all-in [fly-pricing].

---

## Sources (primary, adversarially verified)

- **[fly-config]** Fly.io app configuration — apps with no `services` are private/unreachable: https://fly.io/docs/reference/configuration/
- **[fly-restart]** Fly Machine restart policies (`always`, `on-fail` ×10): https://fly.io/docs/machines/guides-examples/machine-restart-policy/ ; autostop/autostart: https://fly.io/docs/launch/autostop-autostart/
- **[fly-pricing]** Fly.io pricing (shared-cpu-1x/256MB ≈ $2.02/mo; Volumes $0.15/GB-mo): https://fly.io/docs/about/pricing/
- **[fly-schedule]** Fly Scheduled Machines (coarse interval buckets only): https://fly.io/docs/blueprints/task-scheduling/
- **[kalshi-base]** Kalshi prod base URL (`external-api.kalshi.com`, alt `api.elections.kalshi.com`): https://docs.kalshi.com/getting_started/quick_start_authenticated_requests
- **[kalshi-auth]** Kalshi RSA-PSS auth (SHA-256/MGF1/salt=32 over `ts+METHOD+path`): https://docs.kalshi.com/getting_started/quick_start_authenticated_requests
- **[kalshi-idempotency]** `client_order_id` idempotency (`ORDER_ALREADY_EXISTS`): https://docs.kalshi.com/api-reference/orders/create-order
- **[kalshi-fills]** Order fill fields (`fill_count_fp`, `remaining_count_fp`, `status`): https://docs.kalshi.com/api-reference/orders/get-orders
- **[kalshi-portfolio]** Broker-truth endpoints (`/portfolio/{orders,fills,positions,balance}`): https://docs.kalshi.com/openapi.yaml
- **[freqtrade-state]** Durable SQLite trade/order state pattern: https://www.freqtrade.io/en/stable/sql_cheatsheet/
- **[alerting]** Pushover API (emergency priority 2): https://pushover.net/api ; ntfy publish: https://docs.ntfy.sh/publish/

*Generated from a 107-agent deep-research pass (claims verified 2-of-3 adversarial vote against primary sources) + a prior code audit of this repo.*
