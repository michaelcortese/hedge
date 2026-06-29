# hedge — Real-Money Go-Live & Full-Logging Plan

> **Status: COMPLETE — prod live as of 2026-06-29.**
> The bot is running on Fly.io (`hedge-bot`), trading real money on Kalshi production,
> with full DB logging. All four cities (NY, CHI, MIA, AUS) are validated and active.

---

## What was built (both tracks done)

**Track A — Observability** (`hedge/state.py`, `hedge/runner.py`, `scripts/db_report.py`):

- `decisions` table: one row per market per cycle (including HOLDs), with the full
  `MarketQuote` snapshot and `signal.meta` (forecast, dispersion, bias inputs).
- `fills` table: actual broker fills from `GET /portfolio/fills`, keyed by `client_order_id`.
- P&L booked from the `fills` table (actual entry price × actual size), not from intent.
- `db_report.py`: read-only CLI — decisions / trades / calibration / pnl views.

**Track B — Go-live**:

- Prod Kalshi API key set as Fly secrets (`KALSHI_KEY_ID`, `KALSHI_PRIVATE_KEY`, `KALSHI_ENV=prod`).
- Base URL: `https://api.elections.kalshi.com/trade-api/v2`
- ~$100 funded; first cycle placed 13 orders at 18:31 UTC on 2026-06-29.

---

## Risk envelope (baked into `deploy/config.yaml`, secrets-free)

| Parameter | Value |
|---|---|
| Kelly fraction λ | 0.10 |
| Max order | $25 |
| Market cap | 3% of bankroll |
| Portfolio cap | 30% of bankroll |
| Daily loss stop | $50 (latching) |
| Brier kill-switch | trips at 0.20, latches until `--reset-guard` |
| Validated cities | NY, CHI, MIA, AUS (all four, 100% CLI match on 14+ days each) |

---

## Phase 5 — Operate & learn (active)

Daily/weekly loop, all driven off the DB:

1. **`db_report.py trades`** — every real trade, entry, outcome, P&L, fees.
2. **`db_report.py calibration`** — predicted vs realized by strategy/city. Drift in a
   city's reliability curve = a model or station problem; investigate before sizing up.
3. **Score the kill-switch margin** — how close did realized Brier get to the 0.20 trip?
4. **Adjust deliberately** — calibration fixes go through the backtest/paper tournament
   first (CLAUDE.md house rule), never hand-tuned on prod. Raise caps only on evidence.
5. **Compound** — bankroll is read live from Kalshi; profits compound automatically
   under the same caps (no withdrawals).

---

## What can lose money, and how it's mitigated

| Risk | Mitigation |
|---|---|
| Biased model (wrong `p`) bleeds under Kelly | Calibration logging + Brier kill-switch; adjust via backtest only |
| Wrong settlement station → confident-but-wrong | Validated-station gate; calibration-by-city catches drift |
| Believing intent instead of fills | P&L booked from the `fills` table, not the decision |
| Runaway sizing | $25/order, 3%/market, 30%/portfolio, $50/day stop — baked, secrets-free |
| A loss we can't explain | Every input + decision + outcome in one queryable DB |
