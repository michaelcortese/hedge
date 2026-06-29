# hedge — autonomous Kalshi weather trader (Fly.io worker, no HTTP service).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HEDGE_STATE_DIR=/data

WORKDIR /app

# Install deps first for layer caching, then the package.
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e . || true
COPY . .
RUN pip install --no-cache-dir -e .

# Bake the SECRETS-FREE risk/guard config so the conservative caps always apply.
# (The credential-bearing root config.yaml is .dockerignored; creds come via env.)
COPY deploy/config.yaml /app/config.yaml

# Durable state (SQLite, status.json, materialized PEM) lives on the mounted volume.
VOLUME ["/data"]

# Worker. KALSHI_ENV (demo|prod) is the real demo/prod switch; caps in config.yaml
# (λ, $25/order, $50/day, validated cities) bound real-money risk when armed.
#
# PAUSED (real money): the entrypoint runs a dry-run reconciler (settles/books the
# positions we already hold, places NO new orders) alongside the paper tournament
# loop (logs signals + live prod quotes to /data for edge evidence — never trades).
# Both persist to the durable volume. Re-arm real trading by restoring the single
# live CMD:  CMD ["python","-m","hedge.runner","--live","--interval","1800","--allow-prod"]
CMD ["sh", "scripts/fly_entrypoint.sh"]
