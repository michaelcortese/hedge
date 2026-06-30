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

    # --- market-relative skill gate (separate from the Brier kill-switch) ----------
    # The kill-switch above asks "is the model badly miscalibrated?". This gate asks
    # the sharper question the tournament demanded: "does the model actually beat the
    # MARKET MID out-of-sample?" — because on ~efficient weather books a well-calibrated
    # model that merely matches the mid earns nothing after fees. When enabled it scales
    # lambda_kelly by demonstrated skill: ~0 until the acted-on probabilities beat the
    # market-mid Brier over enough settled trades, ramping to full as the edge proves out.
    skill_gate: bool = False         # OFF by default (library default unchanged)
    skill_min_samples: int = 30      # settled acted-on trades before size can rise off the floor
    skill_full_at: float = 0.02      # Brier-skill margin (market - model) for FULL size
    skill_floor: float = 0.0         # lambda multiplier before skill is proven (0 = no size)

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


@dataclass(frozen=True)
class SkillStatus:
    """Out-of-sample skill of acted-on probabilities vs the market mid."""

    n: int
    brier_model: float | None        # Brier of the model's acted-on P(YES)
    brier_market: float | None       # Brier of the market mid at decision time
    skill: float | None              # brier_market - brier_model (>0 => beats market)
    multiplier: float                # lambda_kelly scale in [skill_floor, 1]


def market_skill(samples: list[tuple[float, float, bool]],
                 cfg: GuardConfig) -> SkillStatus:
    """Score model-vs-market skill and derive a lambda multiplier.

    ``samples`` is ``(model_prob, market_mid, outcome_yes)`` per settled acted-on
    trade. Skill is the market's Brier minus the model's Brier: positive means the
    model's probabilities beat the market mid out-of-sample. The multiplier ramps the
    Kelly fraction from ``skill_floor`` (until ``skill_min_samples`` settle, or while
    skill is non-positive) linearly up to 1.0 at ``skill_full_at`` of Brier skill.
    Pure and side-effect-free. With the gate disabled the multiplier is always 1.0.
    """
    n = len(samples)
    if not cfg.skill_gate:
        return SkillStatus(n, None, None, None, 1.0)
    if n < cfg.skill_min_samples:
        return SkillStatus(n, None, None, None, cfg.skill_floor)
    bm = brier_score([(p, y) for p, _mid, y in samples])
    bk = brier_score([(mid, y) for _p, mid, y in samples])
    assert bm is not None and bk is not None
    skill = bk - bm
    span = cfg.skill_full_at if cfg.skill_full_at > 0 else 1e-9
    ramp = max(0.0, min(1.0, skill / span))
    mult = max(cfg.skill_floor, ramp)
    return SkillStatus(n, bm, bk, skill, mult)


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
