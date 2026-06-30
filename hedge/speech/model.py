"""Base-rate model: corpus stats -> P(>=1 mention) + honest std error.

The market resolves YES iff the speaker says the phrase at least once during the
event, so each comparable past speech is a Bernoulli trial (``contained the
phrase?``). With a Beta prior, observing ``k`` hits in ``n`` speeches yields a
posterior ``Beta(a0+k, b0+n-k)`` whose

  * mean  -> our ``prob`` (P the next comparable speech contains the phrase), and
  * std   -> the *sampling* std error, which grows automatically as the corpus
             thins. A 2-speech history can't pin a rate, and the posterior std
             says so.

Two extra honesty knobs, both folded into the reported std error so the sizing
engine sees the real uncertainty:

  * **Effective sample size.** Speeches are weighted by recency / event-type
    similarity upstream (see ``corpus``), so ``n_eff``/``k_eff`` are weighted
    sums, not raw counts. Down-weighting shrinks the effective ``n`` and widens
    the posterior — exactly right when only a few old speeches resemble this one.
  * **Structural-error floor.** Even with infinite history, the *next* speech's
    propensity differs from the long-run average (news cycle, venue, mood). A
    pure base rate is biased for a single event, and Kelly punishes bias. We add
    a fixed ``model_se_floor`` in quadrature so we never report near-zero
    uncertainty off a base rate alone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CorpusStats:
    """Weighted base-rate evidence for one (speaker, phrase, event-type) query.

    Attributes:
        n_eff:  Effective number of comparable past speeches (sum of weights).
        k_eff:  Effective number that contained the phrase (weighted hit count).
        n_raw:  Raw count of speeches considered (for logging / abstain gates).
        mean_count: Mean mentions-per-speech among hits, if counts were available
                    (informational; the indicator model resolves on >=1).
        sources: Which providers contributed, for attribution.
    """

    n_eff: float
    k_eff: float
    n_raw: int = 0
    mean_count: float | None = None
    sources: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.n_eff < 0 or self.k_eff < 0:
            raise ValueError("n_eff/k_eff must be non-negative")
        if self.k_eff > self.n_eff + 1e-9:
            raise ValueError(f"k_eff ({self.k_eff}) > n_eff ({self.n_eff})")


@dataclass(frozen=True)
class MentionEstimate:
    prob: float
    std_error: float
    post_alpha: float
    post_beta: float
    sampling_se: float


def estimate_mention_prob(
    stats: CorpusStats,
    *,
    prior_alpha: float = 0.5,
    prior_beta: float = 0.5,
    model_se_floor: float = 0.05,
    eps: float = 1e-4,
) -> MentionEstimate:
    """Posterior P(>=1 mention) and its std error from weighted corpus evidence.

    Default prior is Jeffreys ``Beta(0.5, 0.5)`` — weakly informative, lets the
    data dominate by ~3-4 speeches while still regularizing a 0/0 corpus toward
    0.5. The reported ``std_error`` combines the posterior (sampling) std with a
    structural ``model_se_floor`` in quadrature; ``prob`` is clamped just inside
    ``(0, 1)`` so it always satisfies the ``Signal`` contract.

    Pass an informative prior (e.g. a global cross-speaker mention rate) via
    ``prior_alpha``/``prior_beta`` when you have one.
    """
    a = prior_alpha + stats.k_eff
    b = prior_beta + (stats.n_eff - stats.k_eff)
    total = a + b

    mean = a / total
    # Variance of a Beta(a, b): ab / ((a+b)^2 (a+b+1)).
    sampling_var = (a * b) / (total * total * (total + 1.0))
    sampling_se = math.sqrt(sampling_var)
    std_error = math.sqrt(sampling_se * sampling_se + model_se_floor * model_se_floor)

    prob = min(1.0 - eps, max(eps, mean))
    return MentionEstimate(
        prob=prob,
        std_error=std_error,
        post_alpha=a,
        post_beta=b,
        sampling_se=sampling_se,
    )
