"""Parse a Kalshi mention market payload into a structured ``MentionMarket``.

The target phrase is **not** in the ticker — Kalshi encodes it in the market
``title`` / ``subtitle`` (e.g. *"Will Donald Trump say 'tariff' during his
speech?"*). This parser is intentionally defensive: it pulls the quoted phrase,
a best-guess speaker, and an event type out of free text, and returns ``None``
when it can't find a phrase (the strategy then abstains rather than guess).

This is the #1 thing to tighten against a real payload: once we pull a live
sample from the actual series, lock the regexes / field names to its real shape.
Anything the parser is unsure about is surfaced in ``MentionMarket`` so it shows
up in signal ``meta`` for inspection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Quote glyphs Kalshi titles use: straight + curly, single + double.
_QUOTE_RE = re.compile(r"""['"‘’“”]([^'"‘’“”]{1,80})['"‘’“”]""")

# Known speakers -> canonical key. Extend as the universe grows; the canonical
# key is what the corpus providers query on, so keep it stable.
_SPEAKERS: dict[str, str] = {
    "trump": "donald_trump",
    "donald trump": "donald_trump",
    "biden": "joe_biden",
    "joe biden": "joe_biden",
    "powell": "jerome_powell",
    "jerome powell": "jerome_powell",
    "musk": "elon_musk",
    "elon musk": "elon_musk",
}

# Event-type keywords -> canonical type, used to match comparable past speeches.
_EVENT_TYPES: dict[str, str] = {
    "state of the union": "sotu",
    "sotu": "sotu",
    "inaugural": "inaugural",
    "debate": "debate",
    "press conference": "press",
    "presser": "press",
    "rally": "rally",
    "fomc": "fomc_presser",
    "press briefing": "press",
    "interview": "interview",
    "speech": "speech",
    "address": "speech",
}


@dataclass(frozen=True)
class MentionMarket:
    """Structured view of one mention market parsed from its raw payload."""

    ticker: str
    phrase: str | None          # the word/phrase that must be said, lower-cased
    speaker: str | None         # canonical speaker key, e.g. "donald_trump"
    event_type: str | None      # canonical event type, e.g. "rally" / "sotu"
    title: str                  # the raw title we parsed (for inspection)

    @property
    def parsed_ok(self) -> bool:
        return bool(self.phrase)


def _title_of(raw: dict) -> str:
    for key in ("title", "yes_sub_title", "subtitle", "rules_primary"):
        v = raw.get(key)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def _match_canonical(text: str, table: dict[str, str]) -> str | None:
    low = text.lower()
    # Prefer the longest matching key so "donald trump" beats "trump".
    for needle in sorted(table, key=len, reverse=True):
        if needle in low:
            return table[needle]
    return None


def parse_mention_market(raw: dict) -> MentionMarket | None:
    """Best-effort parse of a market payload. ``None`` if no phrase is found."""
    ticker = raw.get("ticker") or raw.get("market_ticker") or ""
    title = _title_of(raw)
    if not title:
        return None

    m = _QUOTE_RE.search(title)
    phrase = m.group(1).strip().lower() if m else None
    if not phrase:
        return None

    return MentionMarket(
        ticker=ticker,
        phrase=phrase,
        speaker=_match_canonical(title, _SPEAKERS),
        event_type=_match_canonical(title, _EVENT_TYPES) or "speech",
        title=title,
    )
