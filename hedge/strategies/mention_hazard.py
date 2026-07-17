"""In-event hazard strategy for Kalshi word-mention markets.

Mechanism (backtested on 15,612 settled mention markets, 2025-01..2026-07):
a mention market resolves YES the instant the phrase is uttered; NO must
survive to event end. Mid-event, the market under-decays P(YES) as time
elapses without a mention — retail holds YES lottery tickets, and NO carry
is capital-inefficient, so nobody corrects it. Empirically (event-clustered):
at 40% of the event window, 30-50c markets are priced ~40c but resolve YES
only ~22% of the time.

The signal: for a mention market whose event window is ACTIVE (fraction
elapsed = tau) and which is still unresolved, the calibrated true probability
is

    logit(p_true) = A + B * logit(price) + C * tau

fit on the full settled dataset with event-clustered SEs (see
scripts/research_inevent_hazard.py and data/research/mentions/REPORT.md).
The strategy reports that p_true; the framework's engine sees market price >>
p_true and buys NO. Execution note: the edge survives only on the MAKER path
(post-only NO bids at the standing level, zero maker fee, fixed small clips);
taker crossing eats most of it. Pair with post_only execution and small
per-market size caps.

Out-of-sample (pre-registered rule, events never seen in formation):
maker +10.2c/contract, CI95 [+5.0, +15.3], p<1e-4, 349 fills / 84 events;
survives causal window definition (+7.7c, p=0.008), day-clustering
(p=0.0008), and leave-one-series-out. PAPER-TRADE before sizing: see
docs/MENTION_HAZARD.md.
"""
from __future__ import annotations

import math
import re
from datetime import datetime, timezone

from hedge.signal import Signal
from hedge.strategies.base import MarketView, Strategy

# Calibrated on all settled mention markets (scripts/research_inevent_hazard.py
# fit, 2026-07-16, n=7,656 market-tau observations, event-clustered):
#   logit(p_true) = A + B*logit(price) + C*tau
A = -0.2406
B = +0.9346
C = -1.2795

# structural std-error floor: model error dominates sampling error here;
# residual dispersion of the calibration fit, kept deliberately fat.
SIGMA_FLOOR = 0.06

# price band where the effect is measured; outside it, abstain.
PRICE_LO, PRICE_HI = 0.30, 0.70
TAU_LO, TAU_HI = 0.25, 0.75

# event-window duration (hours) per series prefix — same table the backtest
# used; the window is anchored at the event's scheduled start, which the
# runner must supply (see universe()).
WINDOWS_H = {
    "KXWCMENTION": 2.5, "KXMLBMENTION": 3.5, "KXNBAMENTION": 3.0,
    "KXNHLMENTION": 3.0, "KXTRUMPMENTION": 1.5, "KXTRUMPMENTIONB": 1.5,
    "KXHEARINGMENTION": 3.0, "KXVANCEMENTION": 1.5, "KXMAMDANIMENTION": 1.5,
    "KXLOVEISLMENTION": 1.5, "KXFIGHTMENTION": 1.0, "KXLATENIGHTMENTION": 1.5,
    "KXSECPRESSMENTION": 1.0, "KXPOLITICSMENTION": 1.5,
}

_SERIES_RE = re.compile(r"^(KX[A-Z]+)-")


class MentionHazard(Strategy):
    """In-event NO-side hazard signal on word-mention markets."""

    name = "mention_hazard"

    def __init__(self, event_starts: dict[str, datetime] | None = None):
        # event_ticker -> event start (UTC). Live: fed by a schedule source
        # or the tape-burst onset detector; paper harness fills this in.
        self.event_starts = event_starts or {}

    def universe(self) -> list[str]:
        # The runner supplies mention-market tickers whose event window is
        # near/active; static discovery: all open markets in WINDOWS_H series.
        return []

    # ------------------------------------------------------------------
    def _tau(self, market: MarketView) -> float | None:
        m = _SERIES_RE.match(market.ticker)
        if not m or m.group(1) not in WINDOWS_H:
            return None
        event_ticker = market.ticker.rsplit("-", 1)[0]
        start = self.event_starts.get(event_ticker)
        if start is None:
            return None
        w_h = WINDOWS_H[m.group(1)]
        now = datetime.now(timezone.utc)
        tau = (now - start).total_seconds() / (w_h * 3600.0)
        return tau

    def evaluate(self, market: MarketView) -> Signal | None:
        tau = self._tau(market)
        if tau is None or not (TAU_LO <= tau <= TAU_HI):
            return None
        price = market.mid
        if price is None or not (PRICE_LO <= price <= PRICE_HI):
            return None
        lp = math.log(price / (1.0 - price))
        z = A + B * lp + C * tau
        p_true = 1.0 / (1.0 + math.exp(-z))
        return Signal(
            ticker=market.ticker,
            prob=p_true,
            std_error=SIGMA_FLOOR,
            strategy=self.name,
            meta={"tau": round(tau, 3), "mid": price,
                  "note": "maker-only: execute post_only, small fixed clips"},
        )
