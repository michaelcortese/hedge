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

# Worker loop. KALSHI_ENV (demo|prod) is the real demo/prod switch; --allow-prod
# only PERMITS prod, the env var GATES it (defaults to demo in fly.toml). Caps in
# config.yaml (λ, $25/order, $50/day, validated cities) bound real-money risk.
CMD ["python", "-m", "hedge.runner", "--live", "--interval", "1800", "--allow-prod"]
