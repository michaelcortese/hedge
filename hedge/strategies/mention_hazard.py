"""In-event theta-decay strategy for Kalshi word-mention markets (r3 spec).

Mechanism (certified against TRUE event clocks, 2026-07-17 round-3 audit):
a mention market resolves YES the instant the phrase is uttered; NO must
survive to event end. In-event, the crowd does not decay P(YES) as the
window shrinks — gross NO edge on 20-80c still-open markets grows from
+11c at 50% elapsed to +22c at 90% elapsed (event-clustered p<1e-4, all
six ground-truth families). Full campaign provenance, kill ledger, and
certification tables: data/research/edge-hunt-mentions-2026-07-16.md.

What survived three adversarial audit rounds (and what did not):
- KILLED: anchoring on Kalshi close_time (administrative, hours late —
  MLB median +776 min after the real game end). This strategy requires
  the runner to supply REAL event windows from external feeds.
- KILLED: maker/resting-order capture (queue adverse selection) and any
  entry model cheaper than the live NO ask (books are wide exactly where
  the modeled edge looked biggest).
- CERTIFIED at real NO asks, taker, T = true_end - 60min (pre-specified
  bar p<0.01, n>=15 events): hearings +10.79c/ct p=0.0085 (15 ev,
  smallest-viable) — and, pooled across families, a flow-gated variant
  (enter only after a taker-YES print) +3.46c p=0.0018.

The signal: for a still-open mention market with true-elapsed fraction
tau and price p,

    logit(p_true) = A + A_FAM[family] + B*logit(p) + C*tau

fit with cluster-robust SEs on 11,571 (market,tau) snapshots / 264
ground-truth events (scripts/research_r3_calibration.py; C p=2.3e-12;
model beats raw price Brier in every family, best in hearings/NHL, the
two families with significant extra-overpricing dummies). The strategy
reports p_true; the engine sees market price >> p_true and buys NO as
TAKER — the edge is not a race (latency-flat at end-60) but it exists
only at the real ask: never model a fill better than the book.

Default enablement is hearings-only (the certified family), and only
when the runner confirms the hearing actually convened (one sampled
hearing was cancelled and Kalshi still batch-closed it two days later).
PAPER-TRADE FIRST: see docs/MENTION_HAZARD.md for gates and kill bars.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

from hedge.signal import Signal
from hedge.strategies.base import MarketView, Strategy

# r3 theta-decay calibration (family-dummy fit): logit(P_YES) =
#   A + A_FAM[family] + B*logit(price) + C*tau_true
# fit: n=11,571 (market,tau) obs, 264 ground-truth events, 2026-07-17,
# scripts/research_r3_calibration.py (statsmodels Logit, cluster-robust
# by event). Only NHL and hearings dummies are significant; other
# families use the WC baseline intercept.
A = 0.1117
B = 0.9564
C = -0.9990
A_FAM = {
    "hearings": -1.1676,  # p=0.0001
    "NHL": -0.7793,       # p=0.0004
}

# event-level residual std at tau=0.7, px 30-60c — conservative (includes
# Bernoulli outcome noise; pure-model reference is ~0.21). Fat on purpose.
SIGMA_FLOOR = 0.30

# certified/feasible operating region (fit range 15-85c; trading band and
# tau window from the certified rule and its anchor-error stress).
PRICE_LO, PRICE_HI = 0.20, 0.80
TAU_LO, TAU_HI = 0.40, 0.92

# series prefix -> calibration family. Families outside this table have no
# ground-truth clock (speech/presser series) — the strategy abstains there.
FAMILY_BY_PREFIX = {
    "KXHEARINGMENTION": "hearings",
    "KXNHLMENTION": "NHL",
    "KXWCMENTION": "WC",
    "KXMLBMENTION": "MLB",
    "KXNBAMENTION": "NBA",
    "KXEARNINGSMENTION": "earnings",  # prefix-match: per-ticker suffixes
}

# Families the strategy will actually signal on, by default only the one
# that cleared the certification bar at real asks. NHL was a near-miss
# (p=0.011, 19 ev) — enable it explicitly once paper fills support it.
DEFAULT_ENABLED = frozenset({"hearings"})


@dataclass(frozen=True)
class EventWindow:
    """Real event window from an EXTERNAL feed (never Kalshi close_time).

    Anchor recipes per family (accuracy at end-60, r3 feasibility study):
    hearings: committee stream/gavel + liveness check (~87% within +/-60m);
    NHL/NBA: ESPN live remaining-time by period (med err ~5min);
    WC: kickoff + 121min (med 2.7min); MLB: StatsAPI inning-half remaining
    (med 9.7min); earnings: call start + 60min median duration (med 6.8min).
    """

    start: datetime          # true event start (UTC)
    end_estimate: datetime   # live best estimate of true end (UTC)
    confirmed_live: bool = True  # hearings: gavelled and not adjourned


class MentionHazard(Strategy):
    """In-event NO-side theta-decay signal on word-mention markets."""

    name = "mention_hazard"

    def __init__(
        self,
        event_windows: dict[str, EventWindow] | None = None,
        enabled_families: frozenset[str] = DEFAULT_ENABLED,
    ):
        # event_ticker -> EventWindow, maintained by the runner from
        # external schedule/live feeds. No window, no signal.
        self.event_windows = event_windows or {}
        self.enabled_families = enabled_families

    def universe(self) -> list[str]:
        # The runner supplies open mention-market tickers for events it
        # holds a live EventWindow for; static discovery is per-series.
        return []

    # ------------------------------------------------------------------
    def _family(self, ticker: str) -> str | None:
        for prefix, fam in FAMILY_BY_PREFIX.items():
            if ticker.startswith(prefix):
                return fam
        return None

    def _tau(self, market: MarketView) -> float | None:
        event_ticker = market.ticker.rsplit("-", 1)[0]
        win = self.event_windows.get(event_ticker)
        if win is None or not win.confirmed_live:
            return None
        span = (win.end_estimate - win.start).total_seconds()
        if span <= 0:
            return None
        now = datetime.now(timezone.utc)
        return (now - win.start).total_seconds() / span

    def evaluate(self, market: MarketView) -> Signal | None:
        family = self._family(market.ticker)
        if family is None or family not in self.enabled_families:
            return None
        tau = self._tau(market)
        if tau is None or not (TAU_LO <= tau <= TAU_HI):
            return None
        price = market.mid
        if price is None or not (PRICE_LO <= price <= PRICE_HI):
            return None
        lp = math.log(price / (1.0 - price))
        z = A + A_FAM.get(family, 0.0) + B * lp + C * tau
        p_true = 1.0 / (1.0 + math.exp(-z))
        return Signal(
            ticker=market.ticker,
            prob=p_true,
            std_error=SIGMA_FLOOR,
            strategy=self.name,
            meta={
                "family": family,
                "tau": round(tau, 3),
                "mid": price,
                "note": "taker at real NO ask only; never assume a fill "
                        "inside the spread (r3 certification)",
            },
        )
