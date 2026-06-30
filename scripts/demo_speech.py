"""End-to-end smoke test for the speech-mention strategy (no Kalshi creds needed).

Builds a fake MarketView from a hand-written market payload, runs the strategy
against the FAKE demo corpus, and prints the Signal plus how the decision engine
would view it against a sample market price. Proves the parse -> corpus -> model
-> Signal pipeline runs before any real transcript API is wired.

    .venv/bin/python scripts/demo_speech.py
"""

from __future__ import annotations

from datetime import datetime

from hedge.strategies.base import MarketView
from hedge.strategies.speech_mention import SpeechMentionStrategy

# A stand-in payload shaped like GET /markets/{ticker}. The phrase lives in the
# title (not the ticker) — exactly what the parser has to handle. Replace with a
# real payload once we pull one from the live mention series.
DEMO_MARKETS = [
    {
        "ticker": "KXMENTION-26JUL04-TARIFF",
        "title": "Will Donald Trump say 'tariff' during his rally?",
        "yes_bid": 55, "yes_ask": 60, "last_price": 58,
    },
    {
        "ticker": "KXMENTION-26JUL04-BORDER",
        "title": "Will Donald Trump say 'border' during his rally?",
        "yes_bid": 70, "yes_ask": 74, "last_price": 72,
    },
    {
        "ticker": "KXMENTION-26JUL04-RECESSION",
        "title": "Will Donald Trump say 'recession' during his rally?",
        "yes_bid": 20, "yes_ask": 25, "last_price": 22,
    },
]


def main() -> None:
    strat = SpeechMentionStrategy(now=datetime(2026, 7, 4, 12, 0))
    print(f"strategy: {strat.name}  (corpus providers: "
          f"{[p.name for p in strat.corpus.providers]})\n")

    for raw in DEMO_MARKETS:
        mv = MarketView(raw["ticker"], raw)
        sig = strat.evaluate(mv)
        if sig is None:
            print(f"{raw['ticker']:<28} ABSTAIN  ({raw['title']})")
            continue
        q = mv.mid  # market YES price in dollars
        edge_yes = sig.prob - q
        side = "YES" if edge_yes > 0 else "NO"
        print(
            f"{sig.ticker:<28} p={sig.prob:.3f} ± {sig.sigma:.3f}  "
            f"mkt={q:.2f}  edge={edge_yes:+.3f} -> lean {side}\n"
            f"    phrase={sig.meta['phrase']!r}  n_eff={sig.meta['n_eff']} "
            f"k_eff={sig.meta['k_eff']}  sources={sig.meta['sources']}"
        )


if __name__ == "__main__":
    main()
