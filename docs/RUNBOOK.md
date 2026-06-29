# hedge — Fly.io deploy & operations runbook

Concrete commands to deploy the bot to Fly.io and operate it. Phases 1–5 of
`DEPLOYMENT_PLAN.md` are **implemented and tested** (durable state, fill
reconciliation + idempotency, daily-loss stop, alerts, container). This runbook is
the human-run Phase 6 (demo soak) → Phase 7 (prod tiny).

> Everything below runs on **your** machine — it needs the Fly CLI and your Fly +
> Kalshi accounts, so it can't be done from inside the repo tooling.

---

## 0. One-time prerequisites
```bash
# Fly CLI
curl -L https://fly.io/install.sh | sh      # or: brew install flyctl
fly auth login

# ntfy (alerts) — already chosen: topic hedge-alerts-7f3k9q2x, subscribed in the app.
# Verify it pushes:
curl -d "hedge runbook test" https://ntfy.sh/hedge-alerts-7f3k9q2x
```

## 1. Create the app + volume (no deploy yet)
Run from the repo root (where `fly.toml`, `Dockerfile`, `deploy/config.yaml` live):
```bash
fly launch --no-deploy --name hedge-bot --region iad --copy-config
fly volumes create data --region iad --size 1     # durable state/logs/PEM (matches fly.toml mount)
```
> ⚠️ **`fly launch` will try to add an `[http_service]` block to `fly.toml`.** DELETE it.
> This is a worker, not a web server — with a service, Fly Proxy autostops the machine
> as "excess capacity" (no inbound traffic) and never restarts it, killing the 24/7
> loop. The committed `fly.toml` is already correct (no service). Keep it that way.

## 2. Set secrets — **DEMO first**
```bash
# Demo Kalshi creds (the PEM is passed as the secret CONTENTS, not a path):
fly secrets set KALSHI_ENV=demo
fly secrets set KALSHI_KEY_ID="<your-demo-key-id-uuid>"
fly secrets set KALSHI_PRIVATE_KEY="$(cat secrets/kalshi_demo_private_key.pem)"

# Alerts:
fly secrets set HEDGE_ALERT_URL="https://ntfy.sh/hedge-alerts-7f3k9q2x"
```

## 3. Deploy (demo) and watch it boot
```bash
fly deploy
fly logs                                         # expect a cycle + a "session started" + heartbeat push
fly ssh console -C "cat /data/status.json"       # mode=live-demo, bankroll, halted=false
```
You should get a **"trading session started"** ntfy push, then a **daily heartbeat**.

## 4. Demo soak — 24–48h (the Phase 6 gate)
Leave it running and verify:
- **Restart recovery:** `fly machine list` → `fly machine restart <id>`; confirm
  `status.json` and open orders survive (state is on the volume).
- **Alerts fire:** force a guard trip on demo (or temporarily lower `baseline_brier`
  in `deploy/config.yaml` + redeploy) → expect a CRITICAL ntfy push.
- **No crash loop:** `fly logs` shows steady cycles, no repeating tracebacks.
- **Caps applied:** logged orders are ≤ $25 and λ=0.10 (verified in the container
  smoke test; confirm again in `fly logs`).

**Do not proceed to prod until all four hold.**

## 5. Phase 7 — promote to PROD (tiny)
> Demo and prod Kalshi keys are SEPARATE — generate a prod API key in the Kalshi web
> UI first and save its PEM to `secrets/kalshi_prod_private_key.pem`.
```bash
# Swap creds to prod and flip the switch:
fly secrets set KALSHI_KEY_ID="<your-PROD-key-id-uuid>"
fly secrets set KALSHI_PRIVATE_KEY="$(cat secrets/kalshi_prod_private_key.pem)"
fly secrets set KALSHI_ENV=prod
fly secrets set KALSHI_BASE_URL="https://external-api.kalshi.com/trade-api/v2"

fly deploy
```
Then **fund the Kalshi account ~$100** and confirm in `fly logs`:
- env=prod, λ=0.10, `max_order=$25`, `daily_stop=$50`, guard=on (from the session-start alert).
- Orders only on **KXHIGHCHI / KXHIGHMIA / KXHIGHAUS** (the validated-station gate
  blocks NYC); `weather_climatology` never trades.

The image's `CMD` already includes `--allow-prod`; **`KALSHI_ENV` is the real switch**
(executor refuses prod orders unless env=prod), so step 5 is the deliberate go-live.

## 6. Operations
```bash
fly logs                                              # live cycle output + decisions
fly ssh console -C "cat /data/status.json"            # snapshot: bankroll, positions, halted

# The learning views — everything the bot believed/decided/did/realized (read-only):
fly ssh console -C "python scripts/db_report.py decisions"          # today's decisions (incl. HOLDs)
fly ssh console -C "python scripts/db_report.py trades"            # fills -> settlement -> realized P&L
fly ssh console -C "python scripts/db_report.py calibration --by city"  # predicted vs realized P(YES)
fly ssh console -C "python scripts/db_report.py pnl"               # realized P&L by day
# Or copy the DB down and inspect locally:
#   fly ssh sftp get /data/hedge.db ./hedge.db && python scripts/db_report.py --db ./hedge.db calibration

# Clear a tripped kill-switch / daily-loss latch after diagnosing:
fly ssh console -C "python -m hedge.runner --reset-guard"

# Score the forward P&L track (settlement-aware):
fly ssh console -C "python scripts/run_paper.py score --prod"
```

## 7. Rollback
```bash
fly releases                       # list versions
fly deploy --image <prior-image>   # or: fly releases rollback <version>
```
A bad release never loses money silently: the kill-switch + daily-loss stop latch on
the durable volume, so even a rollback inherits the halt state.

## 8. Settlement scoring (optional cron)
Weather markets settle ~04:59 UTC. To score/compound daily, either rely on the
in-loop settlement booking (already runs each cycle) or add a daily Scheduled Machine:
```bash
fly machine run . --schedule daily \
  --command "python scripts/run_paper.py score --prod"
```

---

### Safety recap (all enforced in code, tested)
- Validated cities only (CHI/MIA/AUS); NYC blocked from real money.
- λ=0.10, **$25/order**, **3%/market, 30%/portfolio**, **$50/day loss stop**.
- Brier kill-switch latches at 0.20; non-tradeable-quote guard; climatology never trades.
- Idempotent orders + fill reconciliation + durable state survive restarts.
- ntfy push on session start, daily heartbeat, guard trip, daily-loss stop, cycle errors.
