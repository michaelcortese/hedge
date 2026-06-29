# hedge — Deployment

**Status: live in production as of 2026-06-29.**

---

## What was built

| Component | Where |
|---|---|
| Signal → decide → Kelly size → V2 order | `hedge/decision/`, `hedge/execution/` |
| RSA-PSS auth, Kalshi REST client | `hedge/kalshi/` |
| Weather Monte Carlo (forecasts → P(YES)/bucket) | `hedge/weather/` |
| Strategies: blend, nowcast, ensemble | `hedge/strategies/weather_*.py` |
| Durable SQLite state: orders, fills, decisions, P&L | `hedge/state.py` |
| Fill reconciliation, settlement booking | `hedge/runner.py` |
| Brier kill-switch, daily-loss stop | `hedge/guard.py`, `deploy/config.yaml` |
| Validated-station gate (all four cities) | `hedge/weather/stations.py` |
| DB query CLI | `scripts/db_report.py` |
| Container + Fly worker config | `Dockerfile`, `fly.toml` |
| Secrets-free risk envelope | `deploy/config.yaml` |

## Infrastructure

- **Fly.io app:** `hedge-bot`, region `iad`, single `shared-cpu-1x` machine (~$2/mo)
- **Volume:** 1 GB durable volume at `/data` (SQLite DB, materialized PEM, status.json)
- **Loop:** `python -m hedge.runner --live --interval 1800 --allow-prod`
- **Switch:** `KALSHI_ENV=prod` Fly secret — the only thing that separates demo from prod

## Risk envelope (in `deploy/config.yaml`, baked into the image)

```
lambda_kelly: 0.10
max_order_dollars: 25.0
market_cap_frac: 0.03
portfolio_cap: 0.30
daily_loss_stop_dollars: 50.0
guard: { enabled: true, baseline_brier: 0.15, tolerance: 0.05 }
```

For operations and monitoring, see `docs/RUNBOOK.md`.
