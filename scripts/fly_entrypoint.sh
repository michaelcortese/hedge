#!/bin/sh
# Fly worker entrypoint — runs TWO loops on the one machine/volume while real-money
# trading is PAUSED, both persisting to /data (HEDGE_STATE_DIR):
#
#   1. dry-run reconciler (background): `hedge.runner` WITHOUT --live. Places no
#      orders, but every cycle it reconciles open orders, backfills fills, and books
#      settlement P&L on the positions we already hold — so realized P&L lands in the
#      durable DB as those markets settle.
#   2. paper tournament (foreground): `run_paper.py loop --prod` snapshots every
#      strategy's signal + the live PROD quote each cycle (keyless, read-only — it
#      cannot place an order). After settlement, `run_paper.py score` turns these into
#      realized, market-priced, fee-net P&L — the edge evidence that gates re-arming.
#
# Re-arm real money later by restoring the single `hedge.runner --live` CMD.
set -eu

INTERVAL="${HEDGE_INTERVAL:-1800}"

echo "[entrypoint] starting dry-run reconciler (background) + paper loop (foreground), interval=${INTERVAL}s"

# Background: dry-run reconcile + settlement booking on existing real positions.
python -m hedge.runner --interval "$INTERVAL" --allow-prod &
RECONCILER_PID=$!

# If this script is told to stop, pass it on to the background reconciler too.
trap 'kill "$RECONCILER_PID" 2>/dev/null || true' INT TERM

# Foreground: the paper edge-evidence loop against keyless prod market data.
exec python scripts/run_paper.py loop --interval "$INTERVAL" --prod
