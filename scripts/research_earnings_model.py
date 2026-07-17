#!/usr/bin/env python
"""Rule-aware earnings-mention probability model (Kalshi mention-market research).

Reads settled KXEARNINGSMENTION* markets from data/research/mentions/events.jsonl
and the earnings-call transcript corpus, parses each market's resolution rule,
counts phrase occurrences in transcripts, sanity-checks the matcher against the
actual Kalshi results on the covered call, then builds strictly leakage-free
features and two probability models (recency-weighted Beta-Binomial posterior +
a time-CV logistic stack). Research only — nothing here touches trading.

Output: data/research/mentions/models/earnings_baserate.csv
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "data/research/mentions/events.jsonl"
TRANSCRIPTS = ROOT / "data/research/mentions/transcripts/earnings.parquet"
OUT = ROOT / "data/research/mentions/models/earnings_baserate.csv"

K_HISTORY = 8
HALF_LIVES = (2, 4, 8)
BB_OUTPUT_HL = 4  # half-life used for the headline p_hat_bb column
SEASON_DAYS = 45
A0 = B0 = 1.0

# ---------------------------------------------------------------------------
# 1. Rule parser
# ---------------------------------------------------------------------------

RULE_TEMPLATE = re.compile(
    r"^If .+? is said by any .+? representative \(including the operator of the "
    r"call\) during the next .+? earnings call \(including the Q\+A\), then the "
    r"market resolves to Yes\.?$",
    re.DOTALL,
)
PAREN_THRESHOLD = re.compile(r"\((\d+)\+\s*times?\)", re.IGNORECASE)
RULE_THRESHOLD = re.compile(r"(\d+)\s*(?:or more|\+)\s*times", re.IGNORECASE)


@dataclass
class ParsedRule:
    phrase: str                      # display phrase (threshold parens stripped)
    variants: list[str]              # phrase variants that each count
    threshold: int                   # total count across variants required
    scope: str                       # who/what counts
    parse_ok: bool
    notes: list[str] = field(default_factory=list)


def parse_rule(yes_sub_title: str, rules_primary: str) -> ParsedRule:
    notes: list[str] = []
    sub = (yes_sub_title or "").strip()
    rules = (rules_primary or "").strip()

    threshold = 1
    m = PAREN_THRESHOLD.search(sub)
    if m:
        threshold = int(m.group(1))
        sub = PAREN_THRESHOLD.sub("", sub).strip()
    m = RULE_THRESHOLD.search(rules)
    if m:
        threshold = max(threshold, int(m.group(1)))

    # Strip any remaining parenthetical qualifiers into variants if they look
    # like alternates, else drop them as annotations.
    extra_variants: list[str] = []
    def _paren(mm: re.Match) -> str:
        inner = mm.group(1).strip()
        if inner and not re.search(r"\d", inner):
            extra_variants.append(inner)
            notes.append(f"parenthetical->variant:{inner!r}")
        return " "
    sub = re.sub(r"\(([^)]*)\)", _paren, sub).strip()

    variants = [v.strip() for v in sub.split("/") if v.strip()] + extra_variants
    parse_ok = bool(variants)

    scope = "unknown"
    if RULE_TEMPLATE.match(rules):
        scope = "any_rep_incl_operator_and_qa"
    else:
        notes.append("nonstandard_rules_template")
        if "Q+A" in rules or "Q&A" in rules:
            scope = "incl_qa_nonstandard"
        parse_ok = parse_ok and bool(rules)

    if not variants:
        notes.append("no_variants_parsed")

    return ParsedRule(
        phrase=sub, variants=variants, threshold=threshold, scope=scope,
        parse_ok=parse_ok, notes=notes,
    )


# ---------------------------------------------------------------------------
# 2. Matcher
# ---------------------------------------------------------------------------

SEP = r"[\s\-_‐-―]"  # transcripts use "_" in product names (AV_Halo)

# Kalshi resolves off audio: "Basel 3" == "Basel III" == "Basel three".
_NUM_FORMS = {
    "1": ["1", "I", "one"], "2": ["2", "II", "two"], "3": ["3", "III", "three"],
    "4": ["4", "IV", "four"], "5": ["5", "V", "five"], "6": ["6", "VI", "six"],
    "7": ["7", "VII", "seven"], "8": ["8", "VIII", "eight"],
    "9": ["9", "IX", "nine"], "10": ["10", "X", "ten"],
}


def _token_pattern(token: str, is_last: bool) -> str:
    """Regex for one token; last token gets plural forms (rules_secondary:
    plural/possessive forms count; other inflections don't)."""
    if token in _NUM_FORMS:
        return "(?:" + "|".join(re.escape(f) for f in _NUM_FORMS[token]) + ")"
    esc = re.escape(token)
    # NOTE: we deliberately do NOT match split compounds ("Testnet" vs
    # "test net"): Kalshi counted "test net" (CRCL) but refused "buy back"
    # for "Buyback" (JPM-26JUL14), so a generic split rule nets zero and
    # biases history features on verb-phrase compounds. Known cost: one
    # false negative on CRCL Testnet.
    if not is_last:
        return esc
    alts = [esc + r"(?:s|es)?"]
    if len(token) > 2 and token[-1].lower() == "y" and token[-2].lower() not in "aeiou":
        alts.append(re.escape(token[:-1]) + r"(?:ies)")
    return "(?:" + "|".join(alts) + ")"


def variant_regex(variant: str) -> str:
    tokens = [t for t in re.split(SEP + "+", variant.strip()) if t]
    if not tokens:
        return r"(?!x)x"  # never matches
    parts = [
        _token_pattern(tok, is_last=(i == len(tokens) - 1))
        for i, tok in enumerate(tokens)
    ]
    body = (SEP + "+").join(parts)
    # Manual boundaries so leading/trailing non-word chars in the phrase
    # (e.g. "M&A") still anchor correctly; apostrophes excluded so "China's"
    # still contains a bare "China" hit.
    return r"(?<![A-Za-z0-9])" + body + r"(?![A-Za-z0-9])"


_REGEX_CACHE: dict[tuple[str, ...], re.Pattern] = {}


def rule_pattern(variants: list[str]) -> re.Pattern:
    key = tuple(sorted(set(v.lower() for v in variants)))
    pat = _REGEX_CACHE.get(key)
    if pat is None:
        # Longest variant first so overlapping variants (Olympic/Olympics)
        # aren't double counted: one alternation, non-overlapping scan.
        alts = sorted(set(variants), key=len, reverse=True)
        pat = re.compile("|".join(variant_regex(v) for v in alts), re.IGNORECASE)
        _REGEX_CACHE[key] = pat
    return pat


_COUNT_CACHE: dict[tuple[int, int], int] = {}


def count_mentions(pat: re.Pattern, text: str, text_id: int) -> int:
    key = (id(pat), text_id)
    c = _COUNT_CACHE.get(key)
    if c is None:
        c = sum(1 for _ in pat.finditer(text))
        _COUNT_CACHE[key] = c
    return c


# ---------------------------------------------------------------------------
# Speaker scope: only company representatives + the operator count.
# Analysts are identified from the operator's Q&A introductions and their
# paragraphs are dropped before matching.
# ---------------------------------------------------------------------------

_NAME = r"[A-Z][A-Za-z.'’\-]+(?:\s+[A-Z][A-Za-z.'’\-]+){0,3}"
_ANALYST_INTRO = re.compile(
    rf"(?:question|questions)(?:'s)?[^.\n]{{0,80}}?"
    rf"(?:comes?|is|will\s+come|coming|come)\s+from\s+"
    rf"(?:the\s+line\s+of\s+)?({_NAME})"
    rf"|from\s+the\s+line\s+of\s+({_NAME})"
)
# Stand-ins / self-identification at the start of a question:
# "Hi, this is Cassie Shannon for Jason", "It's Karl Keirstead with UBS".
_SELF_ID = re.compile(
    rf"\b(?:[Tt]his\s+is|[Ii]t['’]?s|I['’]?m)\s+({_NAME})[,.]?\s*"
    rf"(?:(?:on|in|here)?\s*for\s+[A-Z]|with\s+[A-Z]|from\s+[A-Z]|at\s+[A-Z])"
)


def _norm_name(name: str) -> tuple[str, str]:
    toks = re.sub(r"[.'’]", "", name).lower().split()
    if not toks:
        return ("", "")
    return (toks[0], toks[-1])  # (first, last)


def build_rep_text(full_text: str) -> str:
    """Drop paragraphs spoken by analysts.

    Analysts are identified by (a) Q&A introductions ("next question comes
    from NAME with FIRM") from ANY speaker — some calls are moderated by an
    IR host, not an "Operator"; (b) self-identification with a firm or
    "on for NAME" in their own first words (catches stand-ins whose name
    differs from the introduced analyst). Everyone else, including the
    operator, counts (per the rules). NOTE: we deliberately do NOT mark the
    next speaker after an introduction — on host-moderated calls the host
    reads investor questions aloud and an executive answers next (CRCL,
    HIMS, NBIS), which would wrongly drop executives.
    """
    paras = full_text.split("\n\n")
    speakers = [p.partition(":")[0].strip() for p in paras]
    analysts: set[tuple[str, str]] = set()
    for p in paras:
        content = p.partition(":")[2]
        for m in _ANALYST_INTRO.finditer(content):
            nm = m.group(1) or m.group(2)
            if nm:
                analysts.add(_norm_name(nm))
    for i, p in enumerate(paras):
        head = p.partition(":")[2][:200]
        m = _SELF_ID.search(head)
        if m and speakers[i].lower() != "operator":
            f, l = _norm_name(m.group(1))
            sf, sl = _norm_name(speakers[i])
            if l == sl or not sl:  # self-ID matches the speaker label
                analysts.add((sf or f, sl or l))
    if not analysts:
        return full_text
    last_names = {ln for _, ln in analysts if ln}
    kept = []
    for p, spk in zip(paras, speakers):
        f, l = _norm_name(spk)
        is_analyst = False
        if l in last_names:
            # last name match; require first-name compatibility (prefix either
            # way, covers Chris/Christopher) or exact pair
            for af, al in analysts:
                if al == l and (af == f or af.startswith(f[:3]) or f.startswith(af[:3])):
                    is_analyst = True
                    break
        if not is_analyst:
            kept.append(p)
    return "\n\n".join(kept)


_REP_TEXT_CACHE: dict[int, str] = {}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_markets() -> pd.DataFrame:
    rows = []
    with open(EVENTS) as f:
        for line in f:
            ev = json.loads(line)
            et = ev.get("event_ticker", "")
            m_sym = re.match(r"KXEARNINGSMENTION([A-Z]+)-", et)
            if not m_sym:
                continue
            for mk in ev.get("markets", []):
                if mk.get("result") not in ("yes", "no"):
                    continue
                rows.append({
                    "ticker": mk["ticker"],
                    "event_ticker": et,
                    "symbol": m_sym.group(1),
                    "yes_sub_title": mk.get("yes_sub_title") or mk.get("no_sub_title") or "",
                    "rules_primary": mk.get("rules_primary", ""),
                    "result": 1 if mk["result"] == "yes" else 0,
                    "open_time": pd.Timestamp(mk["open_time"]).tz_convert(None),
                    "close_time": pd.Timestamp(mk["close_time"]).tz_convert(None),
                })
    return pd.DataFrame(rows)


def load_transcripts() -> pd.DataFrame:
    t = pd.read_parquet(TRANSCRIPTS)
    t = t[~t["is_stub"]].copy()
    t["report_date"] = pd.to_datetime(t["report_date"])
    # NBIS symbol pre-2024 is Yandex, not Nebius — exclude (README landmine 4).
    t = t[~((t["symbol"] == "NBIS") & (t["report_date"] < "2024-01-01"))]
    t = t.sort_values(["symbol", "report_date"]).reset_index(drop=True)
    t["text_id"] = np.arange(len(t))
    return t


# ---------------------------------------------------------------------------
# Beta-Binomial
# ---------------------------------------------------------------------------

def weighted_bb(hits: np.ndarray, half_life: float) -> tuple[float, float]:
    """hits: most-recent-last array of 0/1. Returns (posterior mean, sd)."""
    n = len(hits)
    if n == 0:
        a, b = A0, B0
    else:
        ages = np.arange(n - 1, -1, -1, dtype=float)  # most recent call age 0
        w = 0.5 ** (ages / half_life)
        a = A0 + float(np.sum(w * hits))
        b = B0 + float(np.sum(w * (1 - hits)))
    mean = a / (a + b)
    sd = np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
    return mean, sd


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    markets = load_markets()
    trans = load_transcripts()
    print(f"settled earnings markets: {len(markets)}  "
          f"(yes={int(markets.result.sum())}, no={int((1 - markets.result).sum())})")
    print(f"non-stub transcripts: {len(trans)} across {trans.symbol.nunique()} symbols")

    # ---- 1. parse rules -------------------------------------------------
    parsed = [parse_rule(r.yes_sub_title, r.rules_primary) for r in markets.itertuples()]
    markets["phrase"] = [p.phrase for p in parsed]
    markets["variants"] = [p.variants for p in parsed]
    markets["n_variants"] = [len(p.variants) for p in parsed]
    markets["count_threshold"] = [p.threshold for p in parsed]
    markets["scope"] = [p.scope for p in parsed]
    markets["parse_ok"] = [p.parse_ok for p in parsed]
    n_bad = int((~markets.parse_ok).sum())
    print(f"\nRULE PARSER: coverage {len(markets) - n_bad}/{len(markets)} "
          f"({(len(markets) - n_bad) / len(markets):.1%}); unparseable: {n_bad}")
    if n_bad:
        for r in markets[~markets.parse_ok].itertuples():
            print(f"  UNPARSEABLE {r.ticker}: sub={r.yes_sub_title!r}")
    nonstd = markets[markets.scope != "any_rep_incl_operator_and_qa"]
    print(f"nonstandard rule templates: {len(nonstd)}")
    for r in nonstd.head(10).itertuples():
        print(f"  nonstd {r.ticker}: {r.rules_primary[:120]}")
    thr_dist = markets.count_threshold.value_counts().sort_index()
    print(f"count-threshold distribution:\n{thr_dist.to_string()}")

    # ---- transcript index -----------------------------------------------
    by_sym: dict[str, pd.DataFrame] = {s: g for s, g in trans.groupby("symbol")}
    texts = trans["full_text"].tolist()
    text_dates = trans["report_date"].to_numpy()
    text_syms = trans["symbol"].to_numpy()

    def mcount(variants: list[str], tid: int) -> int:
        rep = _REP_TEXT_CACHE.get(tid)
        if rep is None:
            rep = build_rep_text(texts[tid])
            _REP_TEXT_CACHE[tid] = rep
        return count_mentions(rule_pattern(variants), rep, tid)

    # ---- 3. sanity check: matcher vs actual result on the covered call --
    own_tid = np.full(len(markets), -1)
    own_count = np.full(len(markets), np.nan)
    for i, r in enumerate(markets.itertuples()):
        g = by_sym.get(r.symbol)
        if g is None:
            continue
        lo = r.close_time - timedelta(days=5)
        hi = r.close_time + timedelta(days=1)
        cand = g[(g.report_date >= lo.normalize()) & (g.report_date <= hi.normalize())]
        if len(cand) == 0:
            continue
        # nearest to close date
        j = (cand.report_date - r.close_time).abs().idxmin()
        tid = int(trans.loc[j, "text_id"])
        own_tid[i] = tid
        own_count[i] = mcount(r.variants, tid) if r.variants else 0
    markets["own_tid"] = own_tid
    markets["own_count"] = own_count
    markets["matcher_pred"] = np.where(
        np.isnan(own_count), np.nan, (own_count >= markets.count_threshold).astype(float))

    have = markets[markets.own_tid >= 0]
    agree = (have.matcher_pred == have.result)
    print(f"\nSANITY CHECK (matcher vs Kalshi result on own call):")
    print(f"  markets with a covered non-stub transcript: {len(have)}/{len(markets)} "
          f"(missing/stub: {len(markets) - len(have)})")
    print(f"  agreement: {agree.mean():.3f} ({int(agree.sum())}/{len(have)})")
    for thr, g in have.groupby(have.count_threshold.gt(1).map({False: "thr=1", True: "thr>1"})):
        a = (g.matcher_pred == g.result)
        print(f"    {thr}: {a.mean():.3f} ({int(a.sum())}/{len(g)})")
    dis = have[have.matcher_pred != have.result]
    fp = dis[dis.matcher_pred == 1]
    fn = dis[dis.matcher_pred == 0]
    print(f"  disagreements: {len(dis)} (matcher-yes/kalshi-no: {len(fp)}, "
          f"matcher-no/kalshi-yes: {len(fn)})")
    for r in dis.itertuples():
        tl = int(trans.loc[trans.text_id == r.own_tid, "text_len"].iloc[0])
        print(f"    {r.ticker} phrase={r.phrase!r} thr={r.count_threshold} "
              f"count={int(r.own_count)} kalshi={'yes' if r.result else 'no'} text_len={tl}")

    markets["matcher_ok"] = np.where(
        markets.own_tid < 0, np.nan,
        (markets.matcher_pred == markets.result).astype(float))

    # ---- 4. leakage-free features ---------------------------------------
    feats = {h: ([], []) for h in HALF_LIVES}  # h -> (means, sds)
    n_prior, said_last_q, count_last, season_rate = [], [], [], []
    hist_thresh_rate = []  # fraction of prior calls whose count met THIS threshold
    for r in markets.itertuples():
        g = by_sym.get(r.symbol)
        open_d = r.open_time
        if g is not None:
            prior = g[g.report_date < open_d.normalize()].tail(K_HISTORY)
        else:
            prior = trans.iloc[0:0]
        cnts = np.array([mcount(r.variants, int(t)) for t in prior.text_id], dtype=float) \
            if r.variants else np.zeros(0)
        hits = (cnts >= r.count_threshold).astype(float)
        n_prior.append(len(hits))
        said_last_q.append(float(hits[-1]) if len(hits) else np.nan)
        count_last.append(float(cnts[-1]) if len(cnts) else np.nan)
        hist_thresh_rate.append(float(hits.mean()) if len(hits) else np.nan)
        for h in HALF_LIVES:
            mn, sd = weighted_bb(hits, h)
            feats[h][0].append(mn)
            feats[h][1].append(sd)
        # season contagion: same rule applied to OTHER companies' calls in the
        # trailing 45 days before open_time
        mask = (text_dates >= (open_d - timedelta(days=SEASON_DAYS)).to_datetime64()) & \
               (text_dates < open_d.normalize().to_datetime64()) & (text_syms != r.symbol)
        idxs = np.nonzero(mask)[0]
        if len(idxs) and r.variants:
            shits = [1.0 if mcount(r.variants, int(t)) >= r.count_threshold else 0.0
                     for t in idxs]
            season_rate.append(float(np.mean(shits)))
        else:
            season_rate.append(np.nan)

    markets["n_prior_calls"] = n_prior
    markets["said_last_q"] = said_last_q
    markets["count_last"] = count_last
    markets["hist_thresh_rate"] = hist_thresh_rate
    markets["season_rate"] = season_rate
    for h in HALF_LIVES:
        markets[f"p_hat_bb_h{h}"] = feats[h][0]
        markets[f"p_hat_sd_bb_h{h}"] = feats[h][1]
    markets["p_hat_bb"] = markets[f"p_hat_bb_h{BB_OUTPUT_HL}"]
    markets["p_hat_sd_bb"] = markets[f"p_hat_sd_bb_h{BB_OUTPUT_HL}"]

    # ---- 5. logistic stack, expanding-window 5-fold time CV -------------
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    mk = markets.sort_values("close_time").reset_index(drop=True)
    eps = 1e-4
    X = pd.DataFrame({
        "logit_bb": np.log(mk.p_hat_bb.clip(eps, 1 - eps) / (1 - mk.p_hat_bb.clip(eps, 1 - eps))),
        "said_last_q": mk.said_last_q.fillna(0.5),
        "season": mk.season_rate.fillna(mk.season_rate.mean()),
        "season_missing": mk.season_rate.isna().astype(float),
        "log_n_prior": np.log1p(mk.n_prior_calls),
        "log_count_last": np.log1p(mk.count_last.fillna(0)),
        "thr_gt1": (mk.count_threshold > 1).astype(float),
    }).to_numpy()
    y = mk.result.to_numpy()
    oos = np.full(len(mk), np.nan)
    folds = np.array_split(np.arange(len(mk)), 5)
    for i in range(1, 5):
        tr = np.concatenate(folds[:i])
        te = folds[i]
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=1.0, max_iter=2000)
        clf.fit(sc.transform(X[tr]), y[tr])
        oos[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    mk["p_hat_logit_oos"] = oos
    markets = mk

    # ---- 6. output CSV ---------------------------------------------------
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cols = ["ticker", "event_ticker", "symbol", "close_time", "result", "phrase",
            "n_variants", "count_threshold", "matcher_ok", "n_prior_calls",
            "p_hat_bb", "p_hat_sd_bb", "p_hat_logit_oos", "season_rate",
            "said_last_q", "count_last", "hist_thresh_rate",
            "p_hat_bb_h2", "p_hat_bb_h8", "own_count", "parse_ok", "scope"]
    markets[cols].to_csv(OUT, index=False)
    print(f"\nwrote {OUT} ({len(markets)} rows)")

    # ---- 7. diagnostics ---------------------------------------------------
    from sklearn.metrics import roc_auc_score

    def score(df: pd.DataFrame, col: str) -> str:
        d = df.dropna(subset=[col])
        if d.result.nunique() < 2 or len(d) < 10:
            return f"{col}: n={len(d)} (insufficient)"
        p = d[col].clip(1e-6, 1 - 1e-6)
        brier = float(np.mean((p - d.result) ** 2))
        auc = float(roc_auc_score(d.result, p))
        return f"{col:>18}: n={len(d):3d}  Brier={brier:.4f}  AUC={auc:.4f}"

    core = markets[(markets.n_prior_calls >= 4) & (markets.matcher_ok == True)]  # noqa: E712
    print(f"\nDIAGNOSTICS — core subset (n_prior>=4 & matcher_ok): {len(core)} markets, "
          f"base rate {core.result.mean():.3f}")
    const = core.assign(const=0.5)
    print(score(const, "const"), "(the 0.5 benchmark; Brier=0.25 by construction)")
    for c in ["p_hat_bb_h2", "p_hat_bb", "p_hat_bb_h8", "p_hat_logit_oos", "season_rate",
              "said_last_q"]:
        print(score(core, c))
    thr1 = core[core.count_threshold == 1]
    print(f"\nexcluding threshold>1 markets ({len(core) - len(thr1)} dropped):")
    for c in ["p_hat_bb", "p_hat_logit_oos"]:
        print(score(thr1, c))
    allm = markets
    print(f"\nall settled markets (no filters), n={len(allm)}, "
          f"base rate {allm.result.mean():.3f}:")
    for c in ["p_hat_bb", "p_hat_logit_oos"]:
        print(score(allm, c))


if __name__ == "__main__":
    sys.exit(main())
