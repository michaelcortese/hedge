# hedge — Kalshi trading bot

A Kalshi trading bot with a strict separation of concerns:

```
Monte Carlo strategy  ->  Signal (a probability)  ->  edge/Kelly sizing  ->  Kalshi order
   (you write this)          (the contract)            (the framework)        (the framework)
```

**If you are writing a Monte Carlo strategy, you only ever produce `Signal`s.**
You do not place orders, compute edge, or size positions. That is the framework's
job, and it is deliberately kept away from you so every strategy stays
backtestable and is governed by one shared risk engine.

---

## TL;DR plug-and-play

1. Create a file in `hedge/strategies/`, e.g. `hedge/strategies/my_thing.py`.
2. Subclass `Strategy`, set a unique `name`, implement `universe()` and `evaluate()`.
3. Return a `Signal(ticker, prob=...)` from `evaluate()` — or `None` to abstain.
4. That's it. The runner discovers markets, calls you, and routes your signals.

```python
from hedge.signal import Signal
from hedge.strategies.base import MarketView, Strategy


class MyThing(Strategy):
    name = "my_thing"                       # unique, stable, used in logs

    def universe(self) -> list[str]:
        # The Kalshi market tickers your model has an opinion on.
        return ["KXFED-26MAR19-T3.00"]

    def evaluate(self, market: MarketView) -> Signal | None:
        p = self.run_monte_carlo(market)    # <-- YOUR code; return P(resolves YES)
        if p is None:
            return None                     # abstain
        return Signal(
            ticker=market.ticker,
            prob=p,                         # probability in (0, 1) that YES wins
            n_draws=10_000,                 # # of MC draws -> sets std error
            strategy=self.name,
            meta={"anything": "for logging"},
        )
```

See `hedge/strategies/example_coinflip.py` for a complete runnable stub to copy.

---

## The `Signal` contract (the only thing that matters)

`hedge/signal.py`. Fields:

| Field | Required | Meaning |
|---|---|---|
| `ticker` | yes | Kalshi market ticker, e.g. `"KXFED-26MAR19-T3.00"`. Must be a real tradable market. |
| `prob` | yes | Your probability that the market resolves **YES**. Strictly in `(0, 1)`. |
| `n_draws` | no (default 1e6) | Number of independent Monte Carlo draws behind `prob`. Used to derive sampling std error `sqrt(p(1-p)/n_draws)`. Non-sampling method? Pass a large number or set `std_error`. |
| `std_error` | no | Explicit std error of `prob`. Overrides the `n_draws`-derived value. Set this if you can quantify your own uncertainty (incl. model error, not just sampling noise). |
| `strategy` | no | Your strategy's `name`, for attribution in logs. |
| `meta` | no | Free-form dict logged alongside the signal. Never read by the decision engine. |

**Why std error matters:** the sizing engine shrinks your position when your
estimate is noisy and refuses to trade unless your edge clears a multiple of the
std error. A strategy that reports honest uncertainty gets sized correctly; one
that claims false precision (`n_draws` too high) will be over-bet. Report it
honestly. If you have structural/model uncertainty beyond sampling noise, fold it
in via `std_error = sqrt(sampling_se**2 + model_se**2)`.

### Conventions you must follow

- **Always express `prob` as P(YES).** Kalshi markets are binary YES/NO. If your
  model naturally produces P(NO), pass `1 - p`. The framework decides whether to
  buy YES or NO; you never make that call.
- **`prob` is a probability, not a price.** Don't pre-bake the market price, fees,
  or edge into it. Report what you believe; the engine compares it to the market.
- **Return `None` to abstain.** No opinion, insufficient data, or market you don't
  cover this cycle → return `None`. Don't return `prob=0.5` to mean "no opinion";
  0.5 is a real belief and may trigger a trade if the market disagrees.
- **`evaluate` must be reproducible.** No hidden global state the backtester can't
  replay. Seed your RNG from the market/time inputs if you need determinism.
- **No I/O to Kalshi from a strategy.** Read market data only from the `MarketView`
  you're handed. Placing orders from a strategy is a bug.

---

## `MarketView` — what you get to look at

`hedge/strategies/base.py`. A read-only snapshot of one market. Prices are in
**dollars** (0.01–0.99) for convenience; raw integer-cent Kalshi fields are under
`.raw`.

| Accessor | Meaning |
|---|---|
| `market.ticker` | the market ticker |
| `market.yes_bid` / `market.yes_ask` | top-of-book YES bid/ask in dollars |
| `market.last_price` | last trade price in dollars |
| `market.mid` | midpoint (falls back to last price) |
| `market.raw` | the full `GET /markets/{ticker}` payload (cents) |
| `market.orderbook` | the order-book payload if fetched |

Need more data than this exposes? Extend `MarketView` rather than reaching around
it from a strategy — keep the strategy's view of the world centralized.

---

## How sizing uses your signal (so you understand the incentives)

You don't implement any of this, but knowing it helps you report good signals.
Let `p` = your `prob`, `q` = market YES price (dollars):

- **Edge** per YES contract = `p - q`; per NO contract = `q - p`. The engine takes
  whichever side your probability says is underpriced.
- **Sizing** is fractional Kelly: `f* = (p - q)/(1 - q)` for YES, scaled by a
  fraction λ (0.25–0.5). Bigger honest edge → bigger position.
- **Fees** (≈ `0.07 · price · (1 - price)` per contract, max ~1.75¢ at 50¢) and the
  bid/ask spread are subtracted before trading. Edges under ~2¢ are usually noise
  and won't trade.
- **Uncertainty:** the engine gates on `|p - q| > k · sigma` and shrinks toward
  the market when `sigma` is large. This is why your reported std error matters.

**The load-bearing caveat:** all of this protects against variance, not against a
*biased* model. Kelly with a systematically wrong `p` loses money fast. Validate
your strategy's calibration (Brier score / reliability on resolved markets) before
trusting it with real size. Backtest harness lives in `tests/` (WIP).

---

## Repo layout

```
hedge/
  signal.py              # the Signal contract (read this first)
  strategies/
    base.py              # Strategy ABC + MarketView — your interface
    example_coinflip.py  # copy this to start a new strategy
    weather_*.py         # temperature-hedge strategies (see below)
    <your_strategy>.py   # <- you add files here
  weather/               # shared data + Monte Carlo core for temp strategies
    stations.py          # Kalshi series -> NWS settlement station (settlement-critical)
    markets.py           # parse a temp market into bucket bounds (TempMarket)
    providers.py         # free forecast/obs fetchers (Open-Meteo, NWS) + cache
    archive.py           # historical forecasts + ERA5 truth (backtest/climatology)
    distribution.py      # MC: forecasts -> predictive distribution -> bucket P(YES)
    calibration.py       # fit forecast-error spread per city/lead
    sources.py           # ForecastSource seam (live vs archive/intraday replay)
  tournament/            # compare strategies: backtest + paper P&L
    backtest.py          # grade strategies vs realized highs over history
    report.py            # Brier/log-loss/CRPS/calibration/skill leaderboard
    paper.py             # forward: log signals+prices, score realized P&L
  kalshi/
    auth.py              # RSA-PSS request signing (don't touch unless fixing auth)
    client.py            # REST client: markets, orderbook, orders, positions
  decision/              # (WIP) edge calc + Kelly sizing + risk caps
  execution/             # (WIP) decision -> signed order
  runner.py              # (WIP) main loop: signals -> decide -> execute
config.example.yaml      # copy to config.yaml (gitignored) and fill in
scripts/test_auth.py     # auth smoke test (offline + optional live)
scripts/run_backtest.py  # historical tournament -> leaderboard (no creds needed)
scripts/run_paper.py     # forward paper tournament vs Kalshi demo (snapshot/score)
```

## Weather hedge (daily-high temperature markets)

Strategies that bet on Kalshi "high of the day in city X" markets. All share the
``hedge/weather/`` Monte Carlo core (forecasts → predictive distribution of the
official rounded daily high → per-bucket ``P(YES)``); the strategy files stay thin.

- `weather_nowcast` — intraday: observed max-so-far is a hard floor; sharpest, acts
  afternoon only. **Best edge.**
- `weather_ensemble` — multi-model forecast blend; the all-day workhorse.
- `weather_blend` — ensemble early, nowcast once obs bite; the one you'd run.
- `weather_climatology` — history-only null model every strategy must beat.

Run the historical tournament (free APIs, cached, no Kalshi creds):
```bash
.venv/bin/python scripts/run_backtest.py --days 60
```
It fits calibration on a train window, grades all strategies vs ERA5 realized highs
on a disjoint test window, and writes a leaderboard to `data/runs/`. A strategy that
can't beat `weather_climatology` on Brier/log-loss is not enabled for size. The
**#1 correctness risk is the settlement-station map** in `weather/stations.py`
(`validated=False` rows are unverified) — a wrong station yields a confident-but-
biased `p`, which Kelly punishes hard.

---

## Environment & setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python scripts/test_auth.py     # verifies signing works
```

### Kalshi credentials (only needed for live/demo API calls, not for writing strategies)

1. Generate an API key in the Kalshi web UI (do it in **demo** first). You get a
   **Key ID** (UUID) and a one-time **RSA private key** PEM — save the PEM, it's
   never shown again.
2. Put the PEM at `secrets/kalshi_private_key.pem` (the `secrets/` dir and
   `config.yaml` are gitignored — **never commit keys**).
3. `cp config.example.yaml config.yaml` and fill in `key_id` / `private_key_path`,
   or set env vars `KALSHI_KEY_ID` and `KALSHI_PRIVATE_KEY_PATH`.

Demo and production credentials are **not interchangeable** — generate separately.
Develop and backtest against **demo** before pointing at prod.

### Kalshi API quick facts (for framework work, not strategies)

- **Base URLs:** demo `https://demo-api.kalshi.co/trade-api/v2`; prod
  `https://api.elections.kalshi.com/trade-api/v2` (legacy, widely used) or
  `https://external-api.kalshi.com/trade-api/v2` (newer, docs-preferred).
- **Auth:** RSA-PSS (SHA-256, MGF1-SHA256, salt length = 32). Sign
  `timestamp_ms + METHOD + path` where `path` includes `/trade-api/v2` and
  **excludes** the query string. Salt length is the #1 auth bug — keep it 32.
- **Markets:** binary YES/NO, integer cents 1–99, settle to $1/$0.
  `yes_price + no_price = 100`. There is **no shorting** — bet against an outcome
  by **buying NO**; `action="sell"` only closes an existing position.
- **Order book** returns bids only on both sides; reconstruct `yes_ask = 100 - best_no_bid`.
- **Order body:** `ticker, action(buy|sell), side(yes|no), type(limit|market),
  count, yes_price|no_price (cents), client_order_id`. Always send a
  `client_order_id` (UUID) for idempotency.
- **Fees:** taker ≈ `ceil(0.07 · C · P · (1-P))` cents, max ~1.75¢/contract at 50¢;
  maker usually free. Coefficient is **not** universally 0.07 — pull the official
  Fee Schedule PDF and key it per market for production edge math. Settlement is
  free (holding to expiry costs no exit fee).
- **Rate limits:** token bucket; Basic tier ≈ 20 reads/s, 10 writes/s.

---

## House rules

- Don't put real keys, `config.yaml`, or PEMs in git. The `.gitignore` blocks
  them; don't override it.
- Strategy files own only their `evaluate` logic. Cross-cutting changes (sizing,
  risk, execution, the `Signal`/`MarketView` shape) are framework changes —
  coordinate, don't fork them inside a strategy.
- Report uncertainty honestly. Over-confident signals get over-bet.
- New strategy → add a calibration/backtest before enabling it for real size.
