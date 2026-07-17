"""R3 calibration: fit the certified in-event theta-decay mispricing.

Model (pre-specified, no tuning):
    logit(P(YES)) = A + B*logit(price) + C*tau_true

Observations: (market, tau) snapshots for still-open markets on the
matched-event universe (true clocks only), last print in 15-85c and
within 30 min of T_tau (fresh quotes), tau in {0.4..0.9}, one obs per
market per tau; markets drop out once resolved/closed (close_time <= T).

Fit: statsmodels Logit, cov_type='cluster', groups=event_ticker.

Also: hearings-only fit (or pooled + hearings dummy if unstable),
per-family intercept dummies, Brier vs raw price, reliability table,
event-level residual sigma (std_error floor), and sanity checks vs the
raw reanchor test-3 buckets.

Run:  .venv/bin/python scripts/research_r3_calibration.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from research_mentions_lib import load_markets  # noqa: E402
from research_r3_reanchor import R3, load_true_ends  # noqa: E402

TAUS = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
PX_LO, PX_HI = 15.0, 85.0
FRESH_MIN = 30.0


def logit(p):
    return np.log(p / (1.0 - p))


def build_sample(j: pd.DataFrame, te: pd.DataFrame) -> pd.DataFrame:
    jt = j.merge(te, on="event_ticker", how="inner")
    jt = jt[(jt.t_end > jt.t_start)]
    rows = []
    for tau in TAUS:
        Tt = jt.t_start + tau * (jt.t_end - jt.t_start)
        m = (jt.ts <= Tt) & (jt.close_time > Tt)
        w = jt[m].copy()
        w["T_tau"] = Tt[m]
        last = w.sort_values("ts").groupby("ticker").tail(1).copy()
        last["stale_min"] = (
            (last.T_tau - last.ts).dt.total_seconds() / 60.0)
        last = last[(last.stale_min <= FRESH_MIN)
                    & last.yes_price.between(PX_LO, PX_HI)]
        last["tau"] = tau
        rows.append(last)
    s = pd.concat(rows, ignore_index=True)
    s["p"] = s.yes_price / 100.0
    s["x_logit_price"] = logit(s.p)
    s["y"] = s.result.astype(float)
    return s


def fit_logit(s: pd.DataFrame, extra_cols: list[str] | None = None):
    cols = ["x_logit_price", "tau"] + (extra_cols or [])
    X = sm.add_constant(s[cols].astype(float), has_constant="add")
    mod = sm.Logit(s.y, X)
    res = mod.fit(disp=0, maxiter=200,
                  cov_type="cluster",
                  cov_kwds={"groups": s.event_ticker})
    return res, X


def coef_table(res, label):
    print(f"\n[{label}]  n={int(res.nobs)}")
    tab = pd.DataFrame({
        "coef": res.params, "se(cluster)": res.bse,
        "z": res.tvalues, "p": res.pvalues})
    with pd.option_context("display.float_format", "{:10.4f}".format):
        print(tab.to_string())
    return tab


def brier_block(s, phat, label):
    b_mod = float(np.mean((phat - s.y) ** 2))
    b_px = float(np.mean((s.p - s.y) ** 2))
    print(f"  {label:<12} n={len(s):>5}  Brier(model)={b_mod:.4f}  "
          f"Brier(price)={b_px:.4f}  delta={b_mod - b_px:+.4f}")


def main():
    te = load_true_ends()
    j = pd.read_parquet(R3 / "_j_cache.parquet")
    mk = load_markets().drop_duplicates(subset=["ticker"])  # noqa: F841
    te = te[te.event_ticker.isin(j.event_ticker.unique())].copy()
    print(f"matched-event universe: {len(te)} events "
          f"({te.fam.value_counts().to_dict()})")

    s = build_sample(j, te)
    n_ev = s.event_ticker.nunique()
    print(f"\ncalibration sample: {len(s)} (market,tau) obs, "
          f"{s.ticker.nunique()} markets, {n_ev} events")
    print("obs per tau:", s.tau.value_counts().sort_index().to_dict())
    print("obs per family:", s.fam.value_counts().to_dict())
    print(f"YES rate={s.y.mean():.3f}, mean px={s.yes_price.mean():.1f}c, "
          f"median staleness={s.stale_min.median():.1f} min")

    # ---------------------------------------------------- 0. pooled fit
    print("\n" + "=" * 78)
    print("FIT 0 — POOLED: logit(P_YES) = A + B*logit(price) + C*tau_true")
    res, X = fit_logit(s)
    print(f"n={int(res.nobs)}, events={n_ev}, "
          f"converged={res.mle_retvals['converged']}")
    tab = coef_table(res, "pooled")
    A, B, C = res.params["const"], res.params["x_logit_price"], \
        res.params["tau"]
    pC = res.pvalues["tau"]
    if not (C < 0 and pC < 0.01):
        print("\n*** WARNING: C (tau coefficient) is NOT significantly "
              f"negative at p<0.01 (C={C:+.4f}, p={pC:.4g}). This "
              "CONTRADICTS the certified theta-decay mechanism. ***")
    else:
        print(f"\nC significantly negative (C={C:+.4f}, p={pC:.3g}) — "
              "consistent with certified theta decay.")
    s["phat"] = res.predict(X)

    # ---------------------------------------------------- 1. hearings-only
    print("\n" + "=" * 78)
    print("FIT 1 — HEARINGS-ONLY (KXHEARINGMENTION)")
    sh = s[s.fam == "hearings"]
    print(f"hearings sample: {len(sh)} obs, {sh.ticker.nunique()} markets, "
          f"{sh.event_ticker.nunique()} events, YES rate={sh.y.mean():.3f}")
    hearings_ok = False
    if len(sh) >= 30 and sh.event_ticker.nunique() >= 5 \
            and 0 < sh.y.mean() < 1:
        try:
            res_h, X_h = fit_logit(sh)
            tab_h = coef_table(res_h, "hearings-only")
            max_se = float(tab_h["se(cluster)"].abs().max())
            hearings_ok = res_h.mle_retvals["converged"] and max_se < 25
            if not hearings_ok:
                print(f"  -> UNSTABLE (converged="
                      f"{res_h.mle_retvals['converged']}, max SE={max_se:.1f})")
        except Exception as e:  # separation / singular
            print(f"  -> fit FAILED: {e!r}")
    else:
        print("  -> too few obs/events or degenerate outcomes for a "
              "standalone fit")
    if not hearings_ok:
        print("  hearings-only fit unstable at this n; reporting pooled "
              "fit + hearings intercept dummy instead:")
        s["d_hearings"] = (s.fam == "hearings").astype(float)
        res_hd, _ = fit_logit(s, ["d_hearings"])
        coef_table(res_hd, "pooled + hearings dummy")

    # ---------------------------------------------------- 2. family dummies
    print("\n" + "=" * 78)
    print("FIT 2 — PER-FAMILY INTERCEPT DUMMIES (baseline = WC)")
    fams = [f for f in ["MLB", "NBA", "NHL", "hearings", "earnings"]
            if f in set(s.fam)]
    dcols = []
    for f in fams:
        c = f"d_{f}"
        s[c] = (s.fam == f).astype(float)
        dcols.append(c)
    res_f, _ = fit_logit(s, dcols)
    tab_f = coef_table(res_f, "pooled + family dummies")
    sig = [c for c in dcols if tab_f.loc[c, "p"] < 0.05]
    print(f"  significant family effects (p<0.05): "
          f"{sig if sig else 'NONE'}")

    # ---------------------------------------------------- 3. goodness
    print("\n" + "=" * 78)
    print("GOODNESS — in-sample Brier, model vs raw price")
    brier_block(s, s.phat, "ALL")
    for fam, g in s.groupby("fam"):
        brier_block(g, g.phat, f"[{fam}]")

    print("\nreliability (pooled-fit predicted deciles):")
    s["dec"] = pd.qcut(s.phat, 10, duplicates="drop")
    rel = s.groupby("dec", observed=True).agg(
        n=("y", "size"), p_pred=("phat", "mean"), p_real=("y", "mean"),
        px=("yes_price", "mean"))
    with pd.option_context("display.float_format", "{:7.3f}".format):
        print(rel.to_string())

    # ---------------------------------------------------- 4. sigma floor
    print("\n" + "=" * 78)
    print("RESIDUAL DISPERSION — event-level spread around the fit")
    z = s[(s.tau == 0.7) & s.yes_price.between(30, 60)]
    evres = z.groupby("event_ticker").apply(
        lambda g: float(np.mean(g.y - g.phat)), include_groups=False)
    print(f"tau=0.7, px 30-60c: {len(z)} obs, {len(evres)} events; "
          f"event-mean residual std = {evres.std(ddof=1):.4f} "
          f"(mean={evres.mean():+.4f})")
    # broader cut for reference
    z2 = s[s.yes_price.between(30, 60)]
    evres2 = z2.groupby("event_ticker").apply(
        lambda g: float(np.mean(g.y - g.phat)), include_groups=False)
    print(f"all tau, px 30-60c (reference): {len(z2)} obs, "
          f"{len(evres2)} events; std = {evres2.std(ddof=1):.4f}")
    sigma_floor = float(evres.std(ddof=1))

    # ---------------------------------------------------- 5. sanity
    print("\n" + "=" * 78)
    print("SANITY — implied p_true vs raw bucket YES rates")

    def implied(px_c, tau):
        eta = A + B * logit(px_c / 100.0) + C * tau
        return 1.0 / (1.0 + np.exp(-eta))

    for px_c, tau in [(40.0, 0.7), (60.0, 0.9)]:
        p_imp = implied(px_c, tau)
        lo, hi = px_c - 10, px_c + 10
        g = s[(s.tau == tau) & s.yes_price.between(lo, hi)]
        print(f"  price={px_c:.0f}c tau={tau}: model p_true={p_imp:.3f}  "
              f"| raw bucket ({lo:.0f}-{hi:.0f}c, this sample): "
              f"YES rate={g.y.mean():.3f} (n={len(g)}, "
              f"{g.event_ticker.nunique()} ev, mean px="
              f"{g.yes_price.mean():.1f}c)")
    print("  (reanchor test 3 raw zone at tau=0.7: YES 17.7%-28.6% "
          "for mid prices)")

    # ---------------------------------------------------- constants block
    print("\n" + "=" * 78)
    print("CONSTANTS BLOCK")
    print(f"""
# r3 theta-decay calibration: logit(P_YES) = A + B*logit(price) + C*tau_true
# fit: n={int(res.nobs)} (market,tau) obs, {n_ev} events, {date.today()},
#      scripts/research_r3_calibration.py (cluster-robust by event)
A = {A:.4f}
B = {B:.4f}
C = {C:.4f}
SIGMA_FLOOR = {sigma_floor:.4f}  # event-level residual std, tau=0.7 px 30-60c
""")


if __name__ == "__main__":
    main()
