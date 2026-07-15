"""Series-level probability math.

The model predicts per-game win probability p. Series (best-of-N) prices come
from the exact recursion S(a, b) = p*S(a+1, b) + (1-p)*S(a, b+1), assuming
games are iid at probability p. The iid assumption is tested empirically in
the backtest report (momentum test); it is not baked in anywhere else.
"""

from __future__ import annotations

from functools import lru_cache
from math import comb


def wins_needed(best_of: int) -> int:
    if best_of < 1 or best_of % 2 == 0:
        raise ValueError(f"best_of must be a positive odd integer, got {best_of}")
    return best_of // 2 + 1


def series_win_prob(p: float, best_of: int, score_a: int = 0, score_b: int = 0) -> float:
    """P(team A wins a best-of-N series | A wins each game iid w.p. p).

    Supports mid-series states via (score_a, score_b).
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")
    w = wins_needed(best_of)
    if score_a > w or score_b > w or score_a < 0 or score_b < 0 or (score_a == w and score_b == w):
        raise ValueError(f"invalid score {score_a}-{score_b} for best-of-{best_of}")
    if score_a == w:
        return 1.0
    if score_b == w:
        return 0.0

    @lru_cache(maxsize=None)
    def s(a: int, b: int) -> float:
        if a == w:
            return 1.0
        if b == w:
            return 0.0
        return p * s(a + 1, b) + (1.0 - p) * s(a, b + 1)

    return s(score_a, score_b)


def exact_score_probs(p: float, best_of: int) -> dict[tuple[int, int], float]:
    """P(final score) for every possible final score of a fresh best-of-N.

    Keys are (wins_a, wins_b). E.g. Bo5: (3,0), (3,1), (3,2), (2,3), (1,3), (0,3).
    """
    w = wins_needed(best_of)
    out: dict[tuple[int, int], float] = {}
    for k in range(w):  # loser's win count
        # winner takes the last game; C(w-1+k, k) orderings of the first w-1+k games
        n_orders = comb(w - 1 + k, k)
        out[(w, k)] = n_orders * p**w * (1.0 - p) ** k
        out[(k, w)] = n_orders * (1.0 - p) ** w * p**k
    return out
