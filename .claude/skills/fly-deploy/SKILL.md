---
name: fly-deploy
description: Deploy and operate the hedge Kalshi trading bot on Fly.io. Use this skill whenever the user wants to deploy, redeploy, ship, promote, soak, operate, monitor, roll back, or troubleshoot the hedge bot running 24/7 on Fly — including "deploy to demo", "promote to prod", "the machine is stuck", "fly logs show 410/409", "it autostopped", "configuring firecracker hang", checking the bot's status/P&L/calibration on the server, swapping demo↔prod Kalshi creds, or resetting a tripped kill-switch. This is the worker-on-Fly deploy procedure for this repo specifically (app hedge-bot, single always-on Python loop, durable volume), not a generic Fly tutorial.
---

# Fly.io deploy & operations for the hedge bot

This skill runs the deploy and operations workflow for the `hedge` Kalshi trading
bot — a single always-on Python loop (a **worker, not a web server**) on Fly.io with
a durable volume for state. It is the executable companion to `docs/RUNBOOK.md`;
when they disagree, the RUNBOOK is the source of truth — read it first.

## Hard constraints (do not violate)

- **The `fly`/`flyctl` CLI runs on the USER's machine**, against their Fly + Kalshi
  accounts. It is generally NOT available to repo tooling. So for any `fly ...`
  command: if you can run it, do; otherwise hand the user the exact command and ask
  them to run it (suggest the `! <command>` prefix so its output lands in the session).
  Never assume a `fly` command succeeded — verify from its output or ask.
- **Never commit or print real secrets.** Demo and prod Kalshi keys are separate and
  non-interchangeable. The PEM is passed as the secret *contents* via
  `KALSHI_PRIVATE_KEY`, never a path, never git. The ntfy topic is effectively a
  password — keep it in `fly secrets`.
- **Demo before prod, always.** Real money (`KALSHI_ENV=prod`) is the deliberate
  go-live in the Promote phase, gated on a clean demo soak. Do not skip the gate.
- **This is a worker.** `fly.toml` must have **no `[http_service]`/`[[services]]`**
  block — with one, Fly Proxy autostops the machine as "excess capacity" (no inbound
  traffic) and never restarts it, killing the loop. If `fly launch` re-adds it, delete it.

## Decide what the user wants

Map the request to one phase, then execute only that phase (don't run the whole
pipeline unless asked):

| If the user wants to… | Go to |
|---|---|
| First-time setup (app + volume) | **A. Provision** |
| Deploy / redeploy to demo | **B. Deploy (demo)** |
| Confirm it's healthy / soak | **C. Soak gate** |
| Go live with real money | **D. Promote to prod** |
| Check status / P&L / calibration on the server | **E. Operate** |
| A deploy/machine is broken | **F. Troubleshoot** |
| Undo a bad release | **G. Rollback** |

---

## A. Provision (one-time)

Run from repo root (where `fly.toml`, `Dockerfile`, `deploy/config.yaml` live):

```bash
fly launch --no-deploy --name hedge-bot --region iad --copy-config
fly volumes create data --region iad --size 1     # durable state/logs/PEM; matches the [[mounts]]
```

⚠️ If `fly launch` added an `[http_service]` block to `fly.toml`, **delete it** (see
the worker constraint above). Confirm the committed `fly.toml` still has none.

Then set secrets — **DEMO first**:
```bash
fly secrets set KALSHI_ENV=demo
fly secrets set KALSHI_KEY_ID="<demo-key-id-uuid>"
fly secrets set KALSHI_PRIVATE_KEY="$(cat secrets/kalshi_demo_private_key.pem)"
fly secrets set HEDGE_ALERT_URL="https://ntfy.sh/<your-ntfy-topic>"   # alerts
```

## B. Deploy (demo)

```bash
fly deploy
fly logs                                       # expect: [runner] env=demo ... mode=LIVE, steady cycles
fly ssh console -C "cat /data/status.json"     # mode=live-demo, halted=false
```
Expect a "trading session started" ntfy push, then a daily heartbeat. A cycle logging
`placed=0` is normal (no tradeable edge that cycle), not a failure. If the deploy hangs
or the machine misbehaves, jump to **F. Troubleshoot**.

## C. Soak gate (24–48h on demo) — the Phase 6 gate before prod

Verify ALL of these; do not promote until every one holds:
- **Steady cycles, no crash loop** — `fly logs` shows repeating clean cycles.
- **Tables populating** — `fly ssh console -C "python scripts/db_report.py decisions"`
  and `... trades` show rows; `... calibration --by city` looks sane (no wildly
  over-confident `prob`).
- **A demo settlement booked P&L from fills** and it matches by hand.
- **Restart recovery** — `fly machine list` → `fly machine restart <id>`; confirm
  `status.json` + open orders survive (state is on the volume).
- **Alerts fire** — force a guard trip (or lower `baseline_brier` in
  `deploy/config.yaml` + redeploy) → expect a CRITICAL ntfy push.
- **Caps applied** — logged orders ≤ $25, λ=0.10, only CHI/MIA/AUS; `weather_climatology`
  never trades.

## D. Promote to prod (tiny)

Prod creds are **separate** — generate a prod API key in the Kalshi UI, save its PEM to
`secrets/kalshi_prod_private_key.pem`. Then:
```bash
fly secrets set KALSHI_KEY_ID="<PROD-key-id-uuid>"
fly secrets set KALSHI_PRIVATE_KEY="$(cat secrets/kalshi_prod_private_key.pem)"
fly secrets set KALSHI_ENV=prod
fly secrets set KALSHI_BASE_URL="https://external-api.kalshi.com/trade-api/v2"
fly deploy
```
Then **fund the Kalshi account ~$100** and confirm in `fly logs` + the session-start
alert: env=prod, λ=0.10, `max_order=$25`, `daily_stop=$50`, guard=on; orders only on
validated cities (the station gate blocks NYC on real money). `KALSHI_ENV=prod` is the
real switch — the image's CMD already includes `--allow-prod`. First real settlement:
`db_report.py trades` shows the entry fill, outcome, and realized P&L.

## E. Operate (read-only; safe against a live bot)

```bash
fly logs                                            # live cycles + decisions
fly ssh console -C "cat /data/status.json"          # bankroll, positions, halted
fly ssh console -C "python scripts/db_report.py trades"                 # fills -> P&L
fly ssh console -C "python scripts/db_report.py calibration --by city"  # predicted vs realized
fly ssh console -C "python scripts/db_report.py pnl"                    # realized P&L by day
# pull the DB down to inspect locally:
fly ssh sftp get /data/hedge.db ./hedge.db && python scripts/db_report.py --db ./hedge.db calibration
```
Clear a tripped kill-switch / daily-loss latch AFTER diagnosing the cause:
```bash
fly ssh console -C "python -m hedge.runner --reset-guard"
```

## F. Troubleshoot (known failure modes)

Diagnose first with: `fly status`, `fly machine list`, `fly volumes list`, `fly logs`.

- **"configuring firecracker" hangs on deploy** → the new machine can't attach the
  `data` volume (a Fly volume binds to one machine at a time), usually because the old
  machine still holds it, or HA tried to start 2 machines for 1 volume. Fix: destroy the
  stale/duplicate machine (`fly machine destroy <id> --force`) then `fly deploy`; or
  `fly scale count 1`. Confirm the volume's region matches `iad`.
- **"App ... has excess capacity, autostopping machine…"** → there's an
  `[http_service]` in `fly.toml`. Remove it (worker constraint), commit, redeploy.
- **HTTP 410 `deprecated_v1_order_endpoint`** → order create/cancel must use V2
  (`POST`/`DELETE /portfolio/events/orders`); reads stay on `/portfolio/orders`. This is
  already handled in `hedge/kalshi/client.py` — a 410 means stale code is deployed;
  redeploy from current `main`.
- **HTTP 409 `order_already_exists`** → benign idempotency: the `client_order_id` was
  already used (Kalshi burns it forever, even after cancel). The runner's per-cycle idem
  key + benign-409 handling already covers this; if it recurs every cycle on one market,
  the anti-stack/reconcile path is the place to look, not the key.
- **Crash loop in `fly logs`** → read the traceback; `[[restart]] policy='always'` will
  keep restarting it. Fix forward or **G. Rollback**.

## G. Rollback

```bash
fly releases                       # list versions
fly releases rollback <version>    # or: fly deploy --image <prior-image>
```
A bad release never loses money silently: the kill-switch + daily-loss stop latch on the
durable volume, so a rollback inherits the halt state.

---

## After any deploy action
- Summarize what ran, what its output showed, and the single next step (e.g. "soak 24h,
  then promote", or "investigate the traceback before redeploying").
- If you handed the user commands to run, ask them to paste the output so you can verify
  rather than assuming success.
