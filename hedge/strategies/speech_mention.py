"""Word/phrase-mention strategy — "will <speaker> say '<phrase>' during <event>".

Thin wrapper over ``hedge/speech/``: parse the market, pull a weighted base rate
from the corpus, turn it into a Beta-Binomial P(>=1 mention) with honest std
error, emit a ``Signal``. All the modeling lives in the core; this file only
wires parse -> corpus -> model -> Signal and decides when to abstain.

Abstention is deliberate and frequent: no phrase parsed, no known speaker, or a
corpus too thin to mean anything (``min_eff_n``) all return ``None``. A bad base
rate off two stale speeches is worse than no bet — Kelly punishes a biased ``p``.

This has NOT cleared a calibration backtest (no transcript source is wired by
default — it falls back to a FAKE demo corpus). Treat live output as paper-only
until it's graded on resolved mention markets, exactly like the weather rule.
"""

from __future__ import annotations

from datetime import date, datetime

from hedge.signal import Signal
from hedge.speech.corpus import MentionCorpus
from hedge.speech.markets import parse_mention_market
from hedge.speech.model import estimate_mention_prob
from hedge.strategies.base import MarketView, Strategy


class SpeechMentionStrategy(Strategy):
    name = "speech_mention"

    def __init__(
        self,
        corpus: MentionCorpus | None = None,
        *,
        prior_alpha: float = 0.5,
        prior_beta: float = 0.5,
        model_se_floor: float = 0.05,
        min_eff_n: float = 3.0,
        now: datetime | None = None,
    ):
        # ``corpus`` defaults to whatever the keys file enables (FAKE demo if
        # nothing is configured). ``now`` is injectable so a backtest can ask the
        # corpus "what did history look like as of date D".
        self.corpus = corpus or MentionCorpus.from_config()
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.model_se_floor = model_se_floor
        self.min_eff_n = min_eff_n
        self.now = now

    def _as_of(self) -> date:
        return (self.now or datetime.now()).date()

    def evaluate(self, market: MarketView) -> Signal | None:
        mm = parse_mention_market(market.raw)
        if mm is None or mm.speaker is None:
            return None  # couldn't parse a phrase/speaker -> no opinion

        stats = self.corpus.query(mm.speaker, mm.phrase, mm.event_type, self._as_of())
        if stats is None or stats.n_eff < self.min_eff_n:
            return None  # corpus too thin to trust -> abstain rather than guess

        est = estimate_mention_prob(
            stats,
            prior_alpha=self.prior_alpha,
            prior_beta=self.prior_beta,
            model_se_floor=self.model_se_floor,
        )
        return Signal(
            ticker=market.ticker,
            prob=est.prob,
            # Not sampling-based: std_error governs sigma; n_draws is just the
            # (rounded) effective corpus size, for log readability.
            n_draws=max(1, round(stats.n_eff)),
            std_error=est.std_error,
            strategy=self.name,
            meta={
                "speaker": mm.speaker,
                "phrase": mm.phrase,
                "event_type": mm.event_type,
                "n_eff": round(stats.n_eff, 2),
                "k_eff": round(stats.k_eff, 2),
                "n_raw": stats.n_raw,
                "mean_count": stats.mean_count,
                "sources": list(stats.sources),
                "post_alpha": round(est.post_alpha, 3),
                "post_beta": round(est.post_beta, 3),
                "sampling_se": round(est.sampling_se, 4),
                "title": mm.title,
            },
        )
