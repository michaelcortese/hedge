"""The tournament: score temperature strategies against each other.

Two modes (per the approved plan):
  * ``backtest`` — replay archived forecasts vs realized highs over history. No
    Kalshi price data needed; results are immediate. This is where strategies earn
    (or lose) the right to be trusted, measured against the climatology null model.
  * ``paper`` (Phase 5) — forward-log live signals + market prices, then score
    realized P&L after settlement.

``report`` holds the scoring math (Brier, log loss, CRPS, calibration, skill) and
renders the leaderboard.
"""
