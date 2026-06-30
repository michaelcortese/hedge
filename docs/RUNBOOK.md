# hedge — Fly.io operations runbook

The bot is live in **production** on Fly.io (`hedge-bot`, region `iad`), trading real
money on Kalshi. This runbook covers day-to-day operations, monitoring, and recovery.

---

## Current state

- **App:** `hedge-bot`, single `shared-cpu-1x` machine, always-on worker loop
- **Env:** `KALSHI_ENV=prod`, base URL `https://api.elections.kalshi.com/trade-api/v2`
- **Interval:** 1800s (30 min) per cycle
- **Cities:** NY, CHI, MIA, AUS — all four validated, all four active
- **Caps:** λ=0.10, $25/order, 3%/market, 30%/portfolio, $50/day stop (in `deploy/config.yaml`)
- **Guard:** Brier kill-switch on, trips at 0.20, latches until manually cleared

---

## 1. Monitor

```bash
~/.fly/bin/flyctl logs                                         # live cycle output + decisions
~/.fly/bin/flyctl status                                       # machine state

# From inside the container:
~/.fly/bin/flyctl ssh console -C "cat /data/status.json"      # bankroll, halted, last cycle
~/.fly/bin/flyctl ssh console -C "python scripts/db_report.py decisions"      # today's decisions
~/.fly/bin/flyctl ssh console -C "python scripts/db_report.py trades"         # fills + P&L
~/.fly/bin/flyctl ssh console -C "python scripts/db_report.py calibration --by city"
~/.fly/bin/flyctl ssh console -C "python scripts/db_report.py pnl"

# Copy DB locally for deeper inspection:
~/.fly/bin/flyctl ssh sftp get /data/hedge.db ./hedge.db
python scripts/db_report.py --db ./hedge.db calibration
```

---

## 2. Deploy a new version

```bash
# Edit code, commit, push to main, then:
~/.fly/bin/flyctl deploy
~/.fly/bin/flyctl logs    # verify clean restart
```

The machine receives SIGINT, finishes the current cycle, then restarts on the new image.
State (DB, materialized PEM, status.json) persists on the durable volume.

---

## 3. Rotate credentials

```bash
~/.fly/bin/flyctl secrets set KALSHI_KEY_ID="<new-uuid>"
~/.fly/bin/flyctl secrets set KALSHI_PRIVATE_KEY="$(cat secrets/kalshi_prod_private_key.pem)"
# secrets set triggers an automatic rolling restart
```

---

## 4. Reset a tripped kill-switch or daily-loss latch

```bash
# Investigate first — check db_report.py calibration to understand why it tripped.
~/.fly/bin/flyctl ssh console -C "python -m hedge.runner --reset-guard"
```

The latch file is `/data/runs/live/HALTED`. The bot won't place orders until it's cleared.

---

## 5. Rollback

```bash
~/.fly/bin/flyctl releases             # list versions
~/.fly/bin/flyctl deploy --image <prior-image>
```

A rollback never silently loses money: the kill-switch and daily-loss latch persist on
the volume, so the older image inherits any halt state.

---

## 6. Secrets reference

| Secret | Value source |
|---|---|
| `KALSHI_KEY_ID` | Prod API key UUID from Kalshi web UI |
| `KALSHI_PRIVATE_KEY` | PEM contents (not path) from `secrets/kalshi_prod_private_key.pem` |
| `KALSHI_ENV` | `prod` |
| `KALSHI_BASE_URL` | `https://api.elections.kalshi.com/trade-api/v2` |
| `HEDGE_ALERT_URL` | ntfy/Pushover/Slack/Telegram webhook (single URL, comma-sep list, or configure `alerts.channels` / `alerts.url` in config) for push alerts |

---

## Safety recap

- All four cities active (NY, CHI, MIA, AUS); `weather_climatology` never trades.
- λ=0.10, **$25/order**, **3%/market, 30%/portfolio**, **$50/day loss stop** — baked
  into `deploy/config.yaml` (checked into git, no secrets).
- Brier kill-switch latches at 0.20; non-tradeable-quote guard; idempotent orders.
- Durable SQLite state survives restarts; fill reconciliation runs every cycle.
