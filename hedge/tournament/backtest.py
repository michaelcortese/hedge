"""Backtest harness: run every strategy over historical days and grade them.

For each (station, past day) we:
  1. pull the realized daily high (ERA5) — the outcome label;
  2. build the Kalshi-style bucket grid (a complete integer partition + two tails)
     centered so the realized high is always inside;
  3. synthesize a ``MarketView`` per bucket and call each strategy's ``evaluate``,
     with forecast data supplied by an injected archive source (no lookahead);
  4. record the strategy's probability for every bucket, plus which bucket won.

The output is a tidy list of per-(strategy, day, bucket) rows that ``report.py``
turns into Brier/log-loss/CRPS/calibration/skill numbers and a leaderboard.

The grid is a *complete partition*, so a strategy's bucket probabilities are
renormalized to sum to 1 before scoring — proper multi-category scoring needs a
genuine probability vector over mutually-exclusive, exhaustive outcomes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta

from hedge.strategies.base import MarketView, Strategy
from hedge.weather.archive import archive_daily_highs
from hedge.weather.stations import Station


@dataclass(frozen=True)
class Bucket:
    lo_f: float
    hi_f: float

    def label(self) -> str:
        if math.isinf(self.lo_f):
            return f"<={int(self.hi_f)}"
        if math.isinf(self.hi_f):
            return f">={int(self.lo_f)}"
        if self.lo_f == self.hi_f:
            return f"{int(self.lo_f)}"
        return f"{int(self.lo_f)}-{int(self.hi_f)}"

    def to_raw(self, station: Station, local_date: date) -> dict:
        """A Kalshi-style market payload the strategies' parser understands."""
        ev = f"{station.series}-{local_date:%y%b%d}".upper()
        raw = {"ticker": f"{ev}-{self.label()}", "event_ticker": ev}
        if math.isinf(self.lo_f):
            raw.update(strike_type="less_or_equal", cap_strike=self.hi_f)
        elif math.isinf(self.hi_f):
            raw.update(strike_type="greater_or_equal", floor_strike=self.lo_f)
        else:
            raw.update(strike_type="between", floor_strike=self.lo_f, cap_strike=self.hi_f)
        return raw


def build_grid(center: float, *, half_width: int = 18, bucket_width: int = 1) -> list[Bucket]:
    """Complete integer partition around ``center`` plus a low and high tail."""
    c = int(round(center))
    lo = c - half_width
    hi = c + half_width
    buckets = [Bucket(-math.inf, lo - 1)]
    t = lo
    while t <= hi:
        buckets.append(Bucket(t, t + bucket_width - 1))
        t += bucket_width
    buckets.append(Bucket(t, math.inf))
    return buckets


@dataclass
class GradedDay:
    strategy: str
    series: str
    city: str
    local_date: date
    realized_high: float
    realized_idx: int                 # which grid bucket the high fell in
    grid: list[Bucket]
    probs: list[float] = field(default_factory=list)  # renormalized, aligned to grid


def _winning_index(grid: list[Bucket], high: float) -> int:
    h = round(high)
    for i, b in enumerate(grid):
        if b.lo_f <= h <= b.hi_f:
            return i
    return -1


def grade_day(
    strategy: Strategy,
    station: Station,
    local_date: date,
    realized: float,
    *,
    half_width: int = 18,
    bucket_width: int = 1,
) -> GradedDay | None:
    """Run one strategy over one day's bucket grid and grade it. None = abstained."""
    grid = build_grid(realized, half_width=half_width, bucket_width=bucket_width)
    widx = _winning_index(grid, realized)
    if widx < 0:
        return None  # grid didn't cover the realized high (shouldn't happen)

    raw_probs: list[float] = []
    any_signal = False
    for b in grid:
        sig = strategy.evaluate(MarketView("bt", b.to_raw(station, local_date)))
        if sig is None:
            raw_probs.append(0.0)
        else:
            raw_probs.append(sig.prob)
            any_signal = True
    if not any_signal:
        return None  # strategy abstained on this day entirely

    total = sum(raw_probs)
    probs = [p / total for p in raw_probs] if total > 0 else raw_probs
    return GradedDay(
        strategy=strategy.name,
        series=station.series,
        city=station.city,
        local_date=local_date,
        realized_high=realized,
        realized_idx=widx,
        grid=grid,
        probs=probs,
    )


def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def run_backtest(
    stations: list[Station],
    start: date,
    end: date,
    strategy_factory,
    *,
    lead_days: int = 1,
    half_width: int = 18,
    bucket_width: int = 1,
) -> list[GradedDay]:
    """Grade every strategy over every covered (station, day).

    ``strategy_factory(station, target_day, as_of)`` returns the list of
    ``Strategy`` instances to run for that (station, day), wired to an archive
    source and the right ``as_of`` so lead time is honest. The target day is passed
    so intraday strategies (nowcast/blend) can build their time-of-day context.
    Returns the flat list of ``GradedDay`` records.
    """
    results: list[GradedDay] = []
    for st in stations:
        # One ranged fetch of realized highs per city (vs one call per day).
        realized_by_day = archive_daily_highs(st, start, end)
        for day in daterange(start, end):
            realized = realized_by_day.get(day.isoformat())
            if realized is None:
                continue
            as_of = day - timedelta(days=lead_days)
            for strat in strategy_factory(st, day, as_of):
                graded = grade_day(
                    strat, st, day, realized,
                    half_width=half_width, bucket_width=bucket_width,
                )
                if graded is not None:
                    results.append(graded)
    return results
