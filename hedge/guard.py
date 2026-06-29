"""Model-failure circuit breaker.

Kelly sizing protects against *variance* but not against a *biased* model — a
systematically wrong ``p`` loses money fast (CLAUDE.md's #1 correctness risk). The
backtest proves calibration on history; this guard watches calibration *forward*
on the markets the bot actually acted on, and HALTS trading when it drifts badly
from what the backtest established.

The metric is the Brier score of the acted-on probabilities against realized
outcomes: ``mean((p_yes - outcome)**2)``, lower is better. Reference points:

    0.00  perfect          0.25  always guessing 0.50 (a coin)
    >0.25 worse than a coin -> the model is actively harmful

We trip when realized Brier exceeds a threshold — either an absolute ceiling
(``max_brier``) or, preferably, the backtest baseline plus a tolerance
(``baseline_brier + tolerance``) — but only once we have ``min_samples`` settled
trades, so a couple of unlucky settlements can't trip it.

The trip is a LATCH: once halted the bot stays halted until a human resets it
(``runner --reset-guard``). A circuit breaker that silently re-arms itself isn't a
circuit breaker.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GuardConfig:
    """Knobs for the calibration kill-switch (config.yaml ``guard:`` section)."""

    enabled: bool = True
    min_samples: int = 20            # settled trades required before it can trip
    max_brier: float = 0.25          # absolute ceiling (worse-than-coin) backstop
    baseline_brier: float | None = None   # backtest Brier; preferred reference
    tolerance: float = 0.05          # allowed drift above the baseline
    window_days: int = 14            # how far back to read settled decisions
    flatten_on_trip: bool = False    # also sell out of open positions when tripped

    @property
    def threshold(self) -> float:
        """The Brier value above which we trip."""
        if self.baseline_brier is not None:
            return self.baseline_brier + self.tolerance
        return self.max_brier

    @classmethod
    def from_dict(cls, d: dict) -> "GuardConfig":
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in fields})


@dataclass(frozen=True)
class GuardStatus:
    """Outcome of one calibration assessment."""

    tripped: bool
    reason: str
    n: int                       # settled samples scored
    brier: float | None          # realized Brier, None if too few samples
    threshold: float


def brier_score(samples: list[tuple[float, bool]]) -> float | None:
    """Mean squared error of P(YES) vs realized YES outcome. None if empty."""
    if not samples:
        return None
    return sum((p - (1.0 if y else 0.0)) ** 2 for p, y in samples) / len(samples)


def assess(samples: list[tuple[float, bool]], cfg: GuardConfig) -> GuardStatus:
    """Decide whether realized calibration is bad enough to halt trading.

    ``samples`` is a list of (P(YES) the bot acted on, realized YES outcome) for
    settled markets. Pure and side-effect-free so it's trivially testable.
    """
    thresh = cfg.threshold
    n = len(samples)
    if not cfg.enabled:
        return GuardStatus(False, "guard disabled", n, brier_score(samples), thresh)
    if n < cfg.min_samples:
        return GuardStatus(
            False, f"insufficient samples ({n} < {cfg.min_samples})", n, None, thresh
        )
    b = brier_score(samples)
    assert b is not None  # n >= min_samples >= 1
    if b > thresh:
        ref = (
            f"baseline {cfg.baseline_brier:.3f}+{cfg.tolerance:.3f}"
            if cfg.baseline_brier is not None
            else f"max_brier {cfg.max_brier:.3f}"
        )
        return GuardStatus(
            True,
            f"realized Brier {b:.3f} > {thresh:.3f} ({ref}) over {n} settled trades "
            "— model calibration has drifted; halting.",
            n, b, thresh,
        )
    return GuardStatus(False, f"calibration OK (Brier {b:.3f} <= {thresh:.3f}, n={n})",
                       n, b, thresh)
