"""Model vs REAL Kalshi prices — the decisive edge test.

Joins settled Kalshi ``KXLOLGAME`` match markets (price snapshots built by
``lolpred/data/kalshi_market.py``) to series in the feature table, prices each
market with the trained :class:`~lolpred.models.xgb.WinModel` + the exact
best-of-N recursion, and evaluates:

* probability quality: model vs the market's t-5-minute mid (log-loss / Brier /
  accuracy with paired bootstrap CIs, ECE for both);
* an executable-touch P&L simulation at several edge thresholds: buy YES at
  the t-5 ask / buy NO at ``1 - t5_bid``, flat 1-contract stakes, Kalshi taker
  fees, no slippage beyond the quoted spread;
* a CLV-like line-movement diagnostic: did the t60 -> t5 price move toward
  the model?

Kalshi LoL markets are MATCH-level: a Bo1 market is one game, a Bo3/Bo5 market
is the whole series ("win the ... match" = win the series), so the model's
per-game probability is lifted to a series probability via
:func:`lolpred.series.series_win_prob`.

Every number in the report is against REAL transacted/quotable Kalshi prices —
no synthetic odds anywhere in this module.
"""

from __future__ import annotations

import logging
from collections import Counter

import numpy as np
import pandas as pd

from lolpred.backtest.betting import bootstrap_roi_ci
from lolpred.backtest.report import ece, probability_metrics
from lolpred.data.kalshi_market import normalize_team
from lolpred.models.xgb import PERSP_COL
from lolpred.series import series_win_prob

__all__ = [
    "join_markets_to_series",
    "model_series_prob",
    "evaluate",
    "kalshi_taker_fee",
    "DISCLAIMER",
]

logger = logging.getLogger(__name__)

#: Labeling contract: every report produced here carries this header.
DISCLAIMER = (
    "REAL KALSHI PRICES — settled markets, executable-touch backtest, "
    "no slippage beyond the spread, assumes 1-contract fills"
)

_EPS = 1e-15

#: Max team wins observed in a series -> the best-of length of the contract.
_MAX_WINS_TO_BEST_OF = {1: 1, 2: 3, 3: 5}


# ------------------------------------------------------------------------ fees


def kalshi_taker_fee(price: float) -> float:
    """Kalshi taker fee per contract, in DOLLARS, for a fill at ``price``.

    Kalshi's schedule: ``fee_cents = ceil(7 * price_cents * (100 -
    price_cents) / 10000)`` per contract (equivalently
    ``ceil(0.07 * P * (1 - P) * 100) / 100`` dollars).  Implemented in integer
    cents — the price is rounded to the nearest cent first, and the ceiling is
    taken with exact integer arithmetic — to match Kalshi's own rounding
    (floating-point ``math.ceil(1.75...)`` off-by-one bugs are the classic
    failure here).  Prices that round outside (0, 100) cents return 0.0
    (nothing tradable there anyway).
    """
    if price is None or not np.isfinite(price):
        return float("nan")
    c = int(round(float(price) * 100.0))
    if c <= 0 or c >= 100:
        return 0.0
    num = 7 * c * (100 - c)
    fee_cents = -(-num // 10000)  # exact integer ceiling division
    return fee_cents / 100.0


# ------------------------------------------------------------------------ join


def _to_utc(ts) -> pd.Timestamp:
    """Coerce to a tz-aware UTC Timestamp (naive timestamps are UTC by
    convention — the feature table's dates are naive UTC)."""
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")


def join_markets_to_series(
    markets: pd.DataFrame, feats: pd.DataFrame, max_days: float = 1.5
) -> pd.DataFrame:
    """Join Kalshi match markets to series in the feature table.

    Kalshi ``KXLOLGAME`` markets are MATCH-level (a Bo1 is one game; a Bo3/Bo5
    is the whole series — the title says "win the ... match"), so each market
    is joined to a SERIES, keyed by its game-1 row.

    A series is a candidate for a market when::

        {normalize_team(yes_team), normalize_team(opp_team)}
            == {normalize_team(blue_team of game 1),
                normalize_team(red_team of game 1)}

    and ``|game-1 date - match_start| <= max_days`` (game-1 dates are naive
    UTC; ``match_start`` is tz-aware UTC).  When several candidate series
    match, the nearest in time is taken and the ambiguity is counted/logged.

    Returns one row per MATCHED market: all market columns plus ``series_id``,
    ``game1_gameid``, ``game1_date``, ``n_games_in_series``, ``league``.
    Join diagnostics are stashed in ``df.attrs``: ``n_input``, ``n_matched``,
    ``n_unmatched``, ``unmatched_reasons`` (counter dict) and ``n_ambiguous``.
    """
    f = feats.sort_values(["series_id", "game_in_series"], kind="stable")
    g1 = f.drop_duplicates("series_id", keep="first")
    n_games = f.groupby("series_id").size()

    cand_series = g1["series_id"].tolist()
    cand_gameid = g1["gameid"].tolist()
    cand_date = list(pd.to_datetime(g1["date"]))
    cand_date_utc = [_to_utc(d) for d in cand_date]
    cand_league = g1["league"].tolist()

    by_key: dict[frozenset, list[int]] = {}
    for j, (b, r) in enumerate(zip(g1["blue_team"], g1["red_team"])):
        key = frozenset((normalize_team(str(b)), normalize_team(str(r))))
        by_key.setdefault(key, []).append(j)

    max_seconds = float(max_days) * 86400.0
    reasons: Counter = Counter()
    n_ambiguous = 0
    out_rows: list[dict] = []

    for _, mkt in markets.iterrows():
        try:
            ms = _to_utc(mkt["match_start"])
        except (TypeError, ValueError):
            reasons["bad_match_start"] += 1
            continue
        if pd.isna(ms):
            reasons["bad_match_start"] += 1
            continue
        key = frozenset(
            (normalize_team(str(mkt["yes_team"])), normalize_team(str(mkt["opp_team"])))
        )
        idxs = by_key.get(key)
        if not idxs:
            reasons["no_series_for_team_pair"] += 1
            continue
        dts = np.array(
            [abs((cand_date_utc[j] - ms).total_seconds()) for j in idxs], dtype=float
        )
        ok = dts <= max_seconds
        if not ok.any():
            reasons["no_series_within_time_window"] += 1
            continue
        if int(ok.sum()) > 1:
            n_ambiguous += 1
        j = idxs[int(np.argmin(np.where(ok, dts, np.inf)))]

        row = mkt.to_dict()
        sid = cand_series[j]
        row["series_id"] = sid
        row["game1_gameid"] = cand_gameid[j]
        row["game1_date"] = cand_date[j]
        row["n_games_in_series"] = int(n_games[sid])
        row["league"] = cand_league[j]
        out_rows.append(row)

    cols = list(markets.columns) + [
        "series_id", "game1_gameid", "game1_date", "n_games_in_series", "league",
    ]
    out = pd.DataFrame(out_rows, columns=cols)
    out.attrs["n_input"] = int(len(markets))
    out.attrs["n_matched"] = int(len(out))
    out.attrs["n_unmatched"] = int(len(markets) - len(out))
    out.attrs["unmatched_reasons"] = dict(reasons)
    out.attrs["n_ambiguous"] = int(n_ambiguous)
    if n_ambiguous:
        logger.info(
            "join_markets_to_series: %d markets had multiple candidate series "
            "within %.1f days (nearest taken)", n_ambiguous, max_days,
        )
    if reasons:
        logger.info("join_markets_to_series: unmatched reasons: %s", dict(reasons))
    return out


# --------------------------------------------------------------- series pricing


def _swap_orientation(X: pd.DataFrame) -> pd.DataFrame:
    """Mirror a feature frame: the same matchup with the teams' sides swapped.

    Same transform as the model's internal mirror augmentation and
    ``scripts/predict.py``: ``*_diff`` columns negated (antisymmetric),
    ``<stem>_blue`` / ``<stem>_red`` pairs swapped, symmetric context columns
    unchanged.  (predict.py is a script, not an importable module, so the
    3-line helper is replicated here.)
    """
    Xs = X.copy()
    diff_cols = [c for c in X.columns if c.endswith("_diff")]
    if diff_cols:
        Xs[diff_cols] = -Xs[diff_cols].astype(float)
    for col_blue in X.columns:
        if col_blue.endswith("_blue"):
            col_red = col_blue[: -len("_blue")] + "_red"
            if col_red in X.columns:
                Xs[col_blue] = X[col_red].to_numpy(copy=True)
                Xs[col_red] = X[col_blue].to_numpy(copy=True)
    return Xs


def model_series_prob(model, feats: pd.DataFrame, series_id, yes_team: str) -> tuple[float, int]:
    """Model probability that ``yes_team`` wins the series, plus the Bo length.

    From the series' GAME-1 feature row: ``p`` = model P(blue win) for game 1,
    side-averaged with the swapped orientation (``p_bar = 0.5 * (p + (1 -
    p_swap))``, same convention as scripts/predict.py — teams alternate sides
    across a series, so the series recursion must not bake in game 1's side
    assignment).  The series probability for game 1's blue team comes from
    :func:`lolpred.series.series_win_prob` and is flipped when ``yes_team`` is
    game 1's red team.

    Best-of inference — IMPORTANT CAVEAT: ``best_of`` is inferred from the
    series OUTCOME (max wins by either team in the recorded series: 1 -> Bo1,
    2 -> Bo3, 3 -> Bo5).  This peeks at post-match data, but it is used ONLY
    to identify which contract the market was (the Bo length is public
    knowledge before the match — we just don't have a schedule table), never
    to inform the model's belief ``p_bar``.  Acceptable for a settled-market
    backtest; a live system must read the Bo length from the schedule.
    A series whose recorded games imply an impossible win count raises
    ``ValueError`` (as does a ``yes_team`` matching neither game-1 team).
    """
    rows = feats[feats["series_id"] == series_id].sort_values(
        "game_in_series", kind="stable"
    )
    if rows.empty:
        raise ValueError(f"series_id {series_id!r} not found in feats")
    g1 = rows.iloc[[0]]

    feat_cols = getattr(model, "feature_columns_", None)
    if not feat_cols:
        feat_cols = [c for c in feats.columns if c.startswith("f_") and c != PERSP_COL]
    X = g1[list(feat_cols)]
    p = float(np.asarray(model.predict_proba(X))[0])
    p_swap = float(np.asarray(model.predict_proba(_swap_orientation(X)))[0])
    p_bar = float(np.clip(0.5 * (p + (1.0 - p_swap)), 0.0, 1.0))

    winners = np.where(
        rows["blue_win"].astype(int).to_numpy() == 1,
        rows["blue_team"].to_numpy(),
        rows["red_team"].to_numpy(),
    )
    max_wins = int(pd.Series(winners).value_counts().max())
    best_of = _MAX_WINS_TO_BEST_OF.get(max_wins)
    if best_of is None:
        raise ValueError(
            f"series {series_id!r}: max team wins {max_wins} maps to no Bo length"
        )

    p_series_blue = series_win_prob(p_bar, best_of)
    ny = normalize_team(str(yes_team))
    if ny == normalize_team(str(g1["blue_team"].iloc[0])):
        return float(p_series_blue), best_of
    if ny == normalize_team(str(g1["red_team"].iloc[0])):
        return float(1.0 - p_series_blue), best_of
    raise ValueError(
        f"yes_team {yes_team!r} matches neither team of series {series_id!r}"
    )


# -------------------------------------------------------------------- evaluate


def _per_row_logloss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    pc = np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)
    return -(y * np.log(pc) + (1.0 - y) * np.log1p(-pc))


def _boot_ci(diff: np.ndarray, rng: np.random.Generator, n_boot: int) -> dict:
    """Mean + 95% plain row-bootstrap CI of per-market paired differences."""
    n = len(diff)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = diff[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {"mean": float(diff.mean()), "ci_lo": float(lo), "ci_hi": float(hi)}


def _fmt(x, spec: str = "+.4f") -> str:
    return "nan" if x is None or not np.isfinite(x) else format(x, spec)


def evaluate(
    markets_joined: pd.DataFrame,
    feats: pd.DataFrame,
    model,
    min_volume: float = 0.0,
    edge_thresholds: tuple[float, ...] = (0.03, 0.05, 0.08),
    seed: int = 0,
    n_boot: int = 5000,
) -> tuple[dict, str]:
    """Score the model against real Kalshi prices; returns (metrics, report).

    For each joined market: ``model_p`` = :func:`model_series_prob` for the
    YES team; the market's belief is the t-5-minute mid; executable prices are
    the t-5 ask (buy YES) and ``1 - t5_bid`` (buy NO); the outcome is
    ``result`` (1 = YES team won).

    * Probability scoring (markets with a valid t5 mid): model vs market-mid
      log-loss / Brier / accuracy with paired per-market differences and 95%
      plain row-bootstrap CIs (``n_boot`` resamples, seeded), plus ECE for
      both.
    * P&L simulation at each edge threshold ``thr``: buy YES if
      ``model_p - t5_ask > thr`` (entry ``t5_ask``); else buy NO if
      ``(1 - model_p) - (1 - t5_bid) > thr`` (entry ``1 - t5_bid``).  Flat
      one-contract stakes; taker fee per :func:`kalshi_taker_fee`;
      ``pnl = (1 - entry - fee)`` if won else ``(-entry - fee)``.  ROI is on
      dollars staked (``stake = entry + fee``) with a bootstrap 95% CI
      (:func:`~lolpred.backtest.betting.bootstrap_roi_ci` semantics).  The
      per-league breakdown is reported for the best threshold (highest total
      P&L among thresholds with any bets).
    * Line movement t60 -> t5 (CLV-like): Pearson corr of
      ``sign(model_p - t60_mid)`` vs ``sign(t5_mid - t60_mid)`` on markets
      with both snapshots, and the mean movement TOWARD the model conditional
      on model-market disagreement > 5 cents.

    The metrics dict carries a ``"per_market"`` DataFrame (one row per
    evaluated market) alongside JSON-friendly scalars/lists; the text report
    is headed by :data:`DISCLAIMER`.
    """
    attrs = dict(markets_joined.attrs)
    df = markets_joined.copy()
    n_joined = len(df)

    # Volume filter (only when requested; NaN volume counts as 0).
    n_volume_filtered = 0
    if min_volume > 0:
        keep = df["volume"].fillna(0.0).astype(float) >= float(min_volume)
        n_volume_filtered = int((~keep).sum())
        df = df[keep].copy()

    # ---- model probabilities per market ------------------------------------
    model_ps: list[float] = []
    best_ofs: list[float] = []
    ok_rows: list[bool] = []
    n_model_errors = 0
    for r in df.itertuples(index=False):
        try:
            p, bo = model_series_prob(model, feats, r.series_id, r.yes_team)
            model_ps.append(p)
            best_ofs.append(bo)
            ok_rows.append(True)
        except ValueError as exc:
            logger.warning("model_series_prob failed for %s: %s", r.ticker, exc)
            n_model_errors += 1
            ok_rows.append(False)
    df = df[np.asarray(ok_rows, dtype=bool)].copy() if len(df) else df
    df["model_p"] = np.asarray(model_ps, dtype=float)
    df["best_of"] = np.asarray(best_ofs, dtype=int) if best_ofs else np.array([], dtype=int)

    n_eval = len(df)
    y = df["result"].astype(int).to_numpy() if n_eval else np.array([], dtype=int)
    model_p = df["model_p"].to_numpy(dtype=float) if n_eval else np.array([])
    mid = df["t5_mid"].to_numpy(dtype=float) if n_eval else np.array([])
    ask = df["t5_ask"].to_numpy(dtype=float) if n_eval else np.array([])
    bid = df["t5_bid"].to_numpy(dtype=float) if n_eval else np.array([])
    leagues = df["league"].astype(str).to_numpy() if n_eval else np.array([])

    rng = np.random.default_rng(seed)

    # ---- probability scoring vs t5 mid --------------------------------------
    valid_mid = np.isfinite(mid) & (mid > 0.0) & (mid < 1.0) if n_eval else np.array([], bool)
    prob: dict | None = None
    if int(valid_mid.sum()) > 0:
        yv, pv, qv = y[valid_mid].astype(float), model_p[valid_mid], mid[valid_mid]
        m_model = probability_metrics(yv, pv)
        m_market = probability_metrics(yv, qv)
        m_model["ece"] = ece(yv, pv)
        m_market["ece"] = ece(yv, qv)
        d_ll = _per_row_logloss(yv, pv) - _per_row_logloss(yv, qv)
        d_br = (pv - yv) ** 2 - (qv - yv) ** 2
        d_acc = ((pv >= 0.5) == (yv == 1)).astype(float) - (
            (qv >= 0.5) == (yv == 1)
        ).astype(float)
        prob = {
            "n": int(valid_mid.sum()),
            "model": m_model,
            "market_mid": m_market,
            # negative = model better for logloss/brier; positive = model
            # better for accuracy.
            "paired_diff": {
                "logloss": _boot_ci(d_ll, rng, n_boot),
                "brier": _boot_ci(d_br, rng, n_boot),
                "accuracy": _boot_ci(d_acc, rng, n_boot),
            },
            "n_boot": int(n_boot),
        }

    # ---- P&L simulation ------------------------------------------------------
    fee_of = np.vectorize(kalshi_taker_fee, otypes=[float])
    yes_price_ok = np.isfinite(ask) & (ask > 0.0) & (ask < 1.0) if n_eval else np.array([], bool)
    no_entry = 1.0 - bid
    no_price_ok = np.isfinite(bid) & (bid > 0.0) & (bid < 1.0) if n_eval else np.array([], bool)

    pnl_rows: list[dict] = []
    bet_frames: dict[float, pd.DataFrame] = {}
    for thr in edge_thresholds:
        buy_yes = yes_price_ok & (model_p - ask > thr)
        buy_no = no_price_ok & ((1.0 - model_p) - no_entry > thr) & ~buy_yes
        bet = buy_yes | buy_no
        entry = np.where(buy_yes, ask, no_entry)
        fee = fee_of(np.where(bet, entry, 0.5))  # placeholder off-bet, masked out
        won = np.where(buy_yes, y == 1, y == 0)
        pnl = np.where(won, 1.0 - entry - fee, -entry - fee)
        stake = entry + fee

        b = pd.DataFrame(
            {
                "pnl": pnl[bet],
                "stake_frac": stake[bet],  # bootstrap_roi_ci column contract
                "won": won[bet],
                "league": leagues[bet],
                "side": np.where(buy_yes, "yes", "no")[bet],
            }
        )
        bet_frames[thr] = b
        n_bets = int(len(b))
        lo, hi, roi = bootstrap_roi_ci(b, seed=seed) if n_bets else (float("nan"),) * 3
        pnl_rows.append(
            {
                "threshold": float(thr),
                "n_bets": n_bets,
                "n_yes": int((b["side"] == "yes").sum()),
                "n_no": int((b["side"] == "no").sum()),
                "hit_rate": float(b["won"].mean()) if n_bets else float("nan"),
                "total_pnl": float(b["pnl"].sum()),
                "pnl_per_bet": float(b["pnl"].mean()) if n_bets else float("nan"),
                "total_staked": float(b["stake_frac"].sum()),
                "roi": roi,
                "roi_ci_lo": lo,
                "roi_ci_hi": hi,
            }
        )

    best_thr = None
    with_bets = [r for r in pnl_rows if r["n_bets"] > 0]
    if with_bets:
        best_thr = max(with_bets, key=lambda r: r["total_pnl"])["threshold"]

    per_league: list[dict] | None = None
    if best_thr is not None:
        rows = []
        for lg, grp in bet_frames[best_thr].groupby("league", sort=True):
            rows.append(
                {
                    "league": str(lg),
                    "n_bets": int(len(grp)),
                    "hit_rate": float(grp["won"].mean()),
                    "total_pnl": float(grp["pnl"].sum()),
                    "roi": float(grp["pnl"].sum() / grp["stake_frac"].sum()),
                }
            )
        per_league = sorted(rows, key=lambda r: -r["total_pnl"])

    # ---- line movement t60 -> t5 (CLV-like) ----------------------------------
    line_movement: dict | None = None
    if n_eval and "t60_mid" in df.columns:
        t60 = df["t60_mid"].to_numpy(dtype=float)
        both = (
            np.isfinite(t60) & (t60 > 0.0) & (t60 < 1.0) & valid_mid
        )
        if int(both.sum()) >= 2:
            s_model = np.sign(model_p[both] - t60[both])
            s_move = np.sign(mid[both] - t60[both])
            if s_model.std() > 0 and s_move.std() > 0:
                corr = float(np.corrcoef(s_model, s_move)[0, 1])
            else:
                corr = float("nan")
            disagree = np.abs(model_p[both] - t60[both]) > 0.05
            toward = (mid[both] - t60[both]) * np.sign(model_p[both] - t60[both])
            line_movement = {
                "n": int(both.sum()),
                "sign_corr": corr,
                "n_disagree_gt_5c": int(disagree.sum()),
                "mean_move_toward_model_on_disagree": (
                    float(toward[disagree].mean()) if disagree.any() else float("nan")
                ),
            }

    # ---- per-market frame -----------------------------------------------------
    pm_cols = [
        c
        for c in (
            "ticker", "event_ticker", "yes_team", "opp_team", "match_start",
            "league", "series_id", "game1_gameid", "game1_date",
            "n_games_in_series", "best_of", "model_p",
            "t5_bid", "t5_ask", "t5_mid", "t60_mid", "volume", "oi", "result",
        )
        if c in df.columns
    ]
    per_market = df[pm_cols].reset_index(drop=True)

    window = None
    if n_eval:
        ms = pd.to_datetime(df["match_start"])
        window = {"start": str(ms.min()), "end": str(ms.max())}

    metrics: dict = {
        "disclaimer": DISCLAIMER,
        "window": window,
        "n_joined": int(n_joined),
        "n_volume_filtered": int(n_volume_filtered),
        "n_model_errors": int(n_model_errors),
        "n_evaluated": int(n_eval),
        "min_volume": float(min_volume),
        "join": attrs or None,
        "probability": prob,
        "pnl": pnl_rows,
        "best_threshold": best_thr,
        "per_league_best_threshold": per_league,
        "line_movement": line_movement,
        "per_market": per_market,
    }

    # ---- text report -----------------------------------------------------------
    lines: list[str] = []
    lines.append("=" * 74)
    lines.append("Model vs Kalshi evaluation")
    lines.append(DISCLAIMER)
    lines.append("=" * 74)
    if window:
        lines.append(f"window (match_start, UTC): {window['start']} .. {window['end']}")
    if attrs:
        lines.append(
            f"join: {attrs.get('n_matched', n_joined)} matched / "
            f"{attrs.get('n_unmatched', 'n/a')} unmatched of "
            f"{attrs.get('n_input', 'n/a')} markets"
        )
        if attrs.get("unmatched_reasons"):
            lines.append(f"  unmatched reasons: {attrs['unmatched_reasons']}")
        if attrs.get("n_ambiguous"):
            lines.append(
                f"  ambiguous joins (nearest series taken): {attrs['n_ambiguous']}"
            )
    lines.append(
        f"evaluated: {n_eval} markets "
        f"(volume filter >= {min_volume:g} dropped {n_volume_filtered}; "
        f"model errors dropped {n_model_errors})"
    )
    lines.append("")

    lines.append("-- Probability quality: model vs market t5 mid --")
    if prob is None:
        lines.append("no markets with a valid t5 mid; section skipped")
    else:
        lines.append(f"n = {prob['n']} markets with a valid t5 mid")
        lines.append(
            f"{'name':<12} {'n':>5} {'accuracy':>9} {'brier':>8} {'logloss':>8} {'ece':>7}"
        )
        for name, m in (("model", prob["model"]), ("market_mid", prob["market_mid"])):
            lines.append(
                f"{name:<12} {m['n']:>5d} {m['accuracy']:>9.4f} {m['brier']:>8.4f} "
                f"{m['logloss']:>8.4f} {m['ece']:>7.4f}"
            )
        lines.append(
            f"paired diffs (model - market), 95% row-bootstrap CI, "
            f"n_boot={prob['n_boot']}:"
        )
        pd_ = prob["paired_diff"]
        for k, better in (("logloss", "negative"), ("brier", "negative"), ("accuracy", "positive")):
            d = pd_[k]
            lines.append(
                f"  {k:<9} {_fmt(d['mean'])} "
                f"[{_fmt(d['ci_lo'])}, {_fmt(d['ci_hi'])}]  ({better} = model better)"
            )
    lines.append("")

    lines.append(
        "-- P&L simulation (flat 1 contract, t5 executable touch, taker fees) --"
    )
    lines.append(
        f"{'thr':>5} {'n_bets':>6} {'yes/no':>7} {'hit':>6} {'pnl$':>8} "
        f"{'pnl/bet':>8} {'staked$':>8} {'ROI':>7}  {'ROI 95% CI':>18}"
    )
    for r in pnl_rows:
        lines.append(
            f"{r['threshold']:>5.2f} {r['n_bets']:>6d} "
            f"{r['n_yes']:>3d}/{r['n_no']:<3d} "
            f"{_fmt(r['hit_rate'], '.3f'):>6} {r['total_pnl']:>+8.2f} "
            f"{_fmt(r['pnl_per_bet'], '+.3f'):>8} {r['total_staked']:>8.2f} "
            f"{_fmt(r['roi'], '+.3f'):>7}  "
            f"[{_fmt(r['roi_ci_lo'], '+.3f')}, {_fmt(r['roi_ci_hi'], '+.3f')}]"
        )
    if best_thr is not None and per_league:
        lines.append("")
        lines.append(
            f"per-league breakdown at best threshold (by total P&L) thr={best_thr:g}:"
        )
        lines.append(f"{'league':<16} {'n_bets':>6} {'hit':>6} {'pnl$':>8} {'ROI':>7}")
        for r in per_league:
            lines.append(
                f"{r['league']:<16} {r['n_bets']:>6d} {r['hit_rate']:>6.3f} "
                f"{r['total_pnl']:>+8.2f} {r['roi']:>+7.3f}"
            )
    lines.append("")

    lines.append("-- Line movement t60 -> t5 (CLV-like diagnostic) --")
    if line_movement is None:
        lines.append("insufficient markets with both t60 and t5 mids; skipped")
    else:
        lines.append(
            f"n = {line_movement['n']}   "
            f"corr(sign(model - t60_mid), sign(t5_mid - t60_mid)): "
            f"{_fmt(line_movement['sign_corr'], '+.3f')}"
        )
        lines.append(
            f"mean t60->t5 movement TOWARD the model when |model - t60_mid| > 5c: "
            f"{_fmt(line_movement['mean_move_toward_model_on_disagree'], '+.4f')} "
            f"(n = {line_movement['n_disagree_gt_5c']}; positive = market moved our way)"
        )
    lines.append("")

    return metrics, "\n".join(lines)
