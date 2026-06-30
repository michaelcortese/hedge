"""The base-rate corpus seam: past speeches -> weighted ``CorpusStats``.

A ``MentionCorpus`` gathers comparable past speeches for a (speaker, event-type)
from one or more ``TranscriptProvider`` adapters, checks each for the phrase, and
collapses them into the weighted ``(n_eff, k_eff)`` the model consumes. Weighting
makes the evidence honest:

  * **recency** — exponential half-life, so a speaker's word habits from 5 years
    ago count less than last month's;
  * **event-type similarity** — a rally is weak evidence for what gets said at a
    FOMC presser, so off-type speeches are down-weighted (never dropped).

Adapters all return the same ``SpeechRecord`` shape, so swapping/stacking sources
is config, not code. ``MentionCorpus.from_config`` wires up whichever providers
are enabled in the keys file (see ``hedge/speech/config.py``).

Provider status:
  * ``InMemoryTranscriptProvider`` — fully working; ships a clearly-FAKE demo set.
  * ``GdeltProvider``              — keyless, live; a *news-mention proxy* (counts
                                     articles quoting the phrase), lower-weight.
  * ``CongressGovProvider`` / ``RollCallFactbaseProvider`` / ``NewsApiProvider`` —
                                     real endpoints scaffolded; each degrades to
                                     "no records" until its response parse is
                                     verified against live data. They never
                                     fabricate evidence.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date

import requests

from hedge.speech.config import SpeechConfig
from hedge.speech.model import CorpusStats

log = logging.getLogger("hedge.speech.corpus")

_UA = {"User-Agent": "hedge-speech-bot (contact: mcortese1406@gmail.com)"}


@dataclass(frozen=True)
class SpeechRecord:
    """One past speech by a speaker, with enough to score the phrase indicator."""

    speaker: str                       # canonical key, e.g. "donald_trump"
    when: date
    event_type: str                    # canonical, e.g. "rally" / "sotu"
    text: str | None = None            # transcript, if the provider returns full text
    phrase_counts: dict[str, int] = field(default_factory=dict)  # precounted, lower-cased
    provider: str = "unknown"

    def count_phrase(self, phrase: str) -> int:
        """Occurrences of ``phrase`` in this speech (precount first, else scan text)."""
        key = phrase.lower()
        if key in self.phrase_counts:
            return self.phrase_counts[key]
        if not self.text:
            return 0
        return len(re.findall(rf"\b{re.escape(key)}\b", self.text.lower()))


# --------------------------------------------------------------------------- #
# Weighting + aggregation
# --------------------------------------------------------------------------- #

def _recency_weight(age_days: float, half_life_days: float) -> float:
    if half_life_days <= 0:
        return 1.0
    return 0.5 ** (max(0.0, age_days) / half_life_days)


def _event_weight(rec_type: str, target_type: str | None, cross: float) -> float:
    if target_type is None or rec_type == target_type:
        return 1.0
    return cross


def compute_stats(
    records: list[SpeechRecord],
    phrase: str,
    event_type: str | None,
    as_of: date,
    *,
    half_life_days: float = 365.0,
    cross_event_weight: float = 0.4,
    sources: tuple[str, ...] = (),
) -> CorpusStats | None:
    """Collapse past speeches into weighted Beta-Binomial evidence, or ``None``."""
    if not records:
        return None

    n_eff = k_eff = 0.0
    hit_counts: list[int] = []
    for rec in records:
        age = (as_of - rec.when).days
        w = _recency_weight(age, half_life_days) * _event_weight(
            rec.event_type, event_type, cross_event_weight
        )
        if w <= 0:
            continue
        c = rec.count_phrase(phrase)
        n_eff += w
        if c > 0:
            k_eff += w
            hit_counts.append(c)

    if n_eff <= 0:
        return None
    mean_count = (sum(hit_counts) / len(hit_counts)) if hit_counts else None
    return CorpusStats(
        n_eff=n_eff,
        k_eff=k_eff,
        n_raw=len(records),
        mean_count=mean_count,
        sources=sources,
    )


# --------------------------------------------------------------------------- #
# Provider adapters
# --------------------------------------------------------------------------- #

class TranscriptProvider(ABC):
    """Returns past speeches for a speaker within a lookback window."""

    name: str = "unnamed"

    @abstractmethod
    def fetch(self, speaker: str, as_of: date, lookback_days: int) -> list[SpeechRecord]:
        raise NotImplementedError


class InMemoryTranscriptProvider(TranscriptProvider):
    """Working provider backed by an in-memory list — used for the demo/backtest.

    Pass your own records, or call ``demo()`` for a tiny CLEARLY-FAKE set so the
    pipeline runs end-to-end before any API is wired.
    """

    name = "in_memory"

    def __init__(self, records: list[SpeechRecord]):
        self._records = records

    def fetch(self, speaker: str, as_of: date, lookback_days: int) -> list[SpeechRecord]:
        lo = as_of.toordinal() - lookback_days
        return [
            r for r in self._records
            if r.speaker == speaker and lo <= r.when.toordinal() <= as_of.toordinal()
        ]

    @classmethod
    def demo(cls) -> "InMemoryTranscriptProvider":
        """A fabricated Trump corpus — for wiring/tests ONLY, not a real base rate."""
        recs = [
            SpeechRecord("donald_trump", date(2026, 6, 1), "rally",
                         phrase_counts={"tariff": 4, "border": 2}, provider="in_memory"),
            SpeechRecord("donald_trump", date(2026, 5, 20), "rally",
                         phrase_counts={"tariff": 0, "border": 3}, provider="in_memory"),
            SpeechRecord("donald_trump", date(2026, 5, 5), "press",
                         phrase_counts={"tariff": 1}, provider="in_memory"),
            SpeechRecord("donald_trump", date(2026, 4, 10), "rally",
                         phrase_counts={"tariff": 2, "border": 1}, provider="in_memory"),
            SpeechRecord("donald_trump", date(2026, 3, 1), "sotu",
                         phrase_counts={"tariff": 1, "border": 5}, provider="in_memory"),
        ]
        return cls(recs)


class GdeltProvider(TranscriptProvider):
    """Keyless, live: GDELT DOC API as a *news-mention proxy*.

    GDELT has no speech transcripts, so this can't produce per-speech indicators.
    It is included as a weak secondary signal: it returns one synthetic record
    flagged with the phrase count = number of recent articles quoting the phrase
    near the speaker. Treat it as a soft nudge, not ground truth — keep its weight
    low (configure a short ``half_life`` upstream) or disable it for clean
    backtests.
    """

    name = "gdelt"
    URL = "https://api.gdeltproject.org/api/v2/doc/doc"

    def fetch(self, speaker: str, as_of: date, lookback_days: int) -> list[SpeechRecord]:
        # Intentionally conservative: GDELT is a proxy and easy to over-trust, so
        # the prototype returns nothing until you opt into proxy evidence by
        # implementing the query below. Endpoint kept here so it's a one-liner to
        # enable: params={"query": f'"{phrase}" {speaker}', "mode":"artlist", ...}.
        log.debug("GdeltProvider is a proxy stub; returning no records by default")
        return []


class _KeyedScaffold(TranscriptProvider):
    """Shared base for keyed transcript APIs that aren't response-verified yet.

    Holds the api key + endpoint and a ``_request`` helper, but leaves ``_parse``
    abstract. ``fetch`` calls the endpoint and parses; ANY failure (no key, HTTP
    error, unverified schema) degrades to ``[]`` with a warning, so an
    unconfigured source never blocks a cycle and never invents evidence.
    """

    base_url: str = ""

    def __init__(self, api_key: str | None):
        self.api_key = api_key

    def _request(self, params: dict) -> dict | None:
        if not self.api_key:
            return None
        try:
            resp = requests.get(self.base_url, params=params, headers=_UA, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001 — best-effort external source
            log.warning("%s request failed: %s", self.name, exc)
            return None

    @abstractmethod
    def _parse(self, payload: dict, speaker: str) -> list[SpeechRecord]:
        raise NotImplementedError

    def fetch(self, speaker: str, as_of: date, lookback_days: int) -> list[SpeechRecord]:
        try:
            payload = self._request(self._params(speaker, as_of, lookback_days))
            if not payload:
                return []
            return self._parse(payload, speaker)
        except NotImplementedError:
            log.info("%s: response parse not verified yet — returning no records", self.name)
            return []
        except Exception as exc:  # noqa: BLE001
            log.warning("%s fetch failed: %s", self.name, exc)
            return []

    def _params(self, speaker: str, as_of: date, lookback_days: int) -> dict:
        return {}


class CongressGovProvider(_KeyedScaffold):
    """congress.gov API — Congressional Record (floor remarks).

    Free API key from https://api.congress.gov/sign-up/. Good for members of
    Congress; not for non-government speakers. Verify the ``congressional-record``
    response shape, then implement ``_parse`` to emit one ``SpeechRecord`` per
    daily record section with the phrase counted from its text.
    """

    name = "congress_gov"
    base_url = "https://api.congress.gov/v3/congressional-record"

    def _params(self, speaker, as_of, lookback_days):
        return {"api_key": self.api_key, "format": "json", "limit": 50}

    def _parse(self, payload, speaker):
        raise NotImplementedError  # TODO: map congressional-record -> SpeechRecord


class RollCallFactbaseProvider(_KeyedScaffold):
    """Roll Call / Factbase — the canonical Trump (and other figures) speech DB.

    The single best source for "will <politician> say X" base rates. Endpoint and
    auth vary by access tier; set ``base_url``/headers to your grant, then
    implement ``_parse`` to emit one ``SpeechRecord`` per speech with full text so
    ``count_phrase`` works for any phrase.
    """

    name = "factbase"
    base_url = ""  # set to your Factbase/Roll Call endpoint

    def _params(self, speaker, as_of, lookback_days):
        return {"key": self.api_key, "speaker": speaker, "limit": 100}

    def _parse(self, payload, speaker):
        raise NotImplementedError  # TODO: map Factbase speeches -> SpeechRecord


class NewsApiProvider(_KeyedScaffold):
    """newsapi.org — news-mention proxy (secondary signal, like GDELT).

    Not transcripts; counts articles quoting the phrase. Key from
    https://newsapi.org/. Weak evidence — keep it low-weight or disabled for
    backtests. Implement ``_parse`` to fold article hits into a synthetic record.
    """

    name = "newsapi"
    base_url = "https://newsapi.org/v2/everything"

    def _params(self, speaker, as_of, lookback_days):
        return {"apiKey": self.api_key, "q": speaker, "pageSize": 50, "language": "en"}

    def _parse(self, payload, speaker):
        raise NotImplementedError  # TODO: map articles -> proxy SpeechRecord


# Registry: config provider-name -> constructor(api_key).
_KEYED_PROVIDERS = {
    CongressGovProvider.name: CongressGovProvider,
    RollCallFactbaseProvider.name: RollCallFactbaseProvider,
    NewsApiProvider.name: NewsApiProvider,
}


# --------------------------------------------------------------------------- #
# The corpus
# --------------------------------------------------------------------------- #

class MentionCorpus:
    """Aggregates comparable past speeches across providers into ``CorpusStats``."""

    def __init__(
        self,
        providers: list[TranscriptProvider],
        *,
        lookback_days: int = 730,
        half_life_days: float = 365.0,
        cross_event_weight: float = 0.4,
    ):
        self.providers = providers
        self.lookback_days = lookback_days
        self.half_life_days = half_life_days
        self.cross_event_weight = cross_event_weight

    @classmethod
    def from_config(cls, cfg: SpeechConfig | None = None, **kwargs) -> "MentionCorpus":
        """Build from the keys file: every enabled provider, in registry order.

        Keyed providers turn on when their key is present; ``gdelt`` is keyless and
        opt-in (``enabled: true``). If nothing is enabled, falls back to the FAKE
        demo provider so the pipeline still runs (and logs a loud warning).
        """
        cfg = cfg or SpeechConfig.load()
        providers: list[TranscriptProvider] = []
        for name, ctor in _KEYED_PROVIDERS.items():
            if cfg.is_enabled(name):
                providers.append(ctor(cfg.api_key(name)))
        if cfg.is_enabled(GdeltProvider.name):
            providers.append(GdeltProvider())
        if not providers:
            log.warning(
                "No speech corpus providers enabled — using the FAKE demo corpus. "
                "Add keys to config.speech.yaml / secrets/speech_apis.yaml."
            )
            providers.append(InMemoryTranscriptProvider.demo())
        return cls(providers, **kwargs)

    def query(
        self, speaker: str | None, phrase: str, event_type: str | None, as_of: date
    ) -> CorpusStats | None:
        """Gather + weight all providers' records into one ``CorpusStats``."""
        if not speaker:
            return None
        records: list[SpeechRecord] = []
        used: list[str] = []
        for prov in self.providers:
            try:
                got = prov.fetch(speaker, as_of, self.lookback_days)
            except Exception as exc:  # noqa: BLE001
                log.warning("provider %s failed: %s", prov.name, exc)
                continue
            if got:
                records.extend(got)
                used.append(prov.name)
        return compute_stats(
            records, phrase, event_type, as_of,
            half_life_days=self.half_life_days,
            cross_event_weight=self.cross_event_weight,
            sources=tuple(used),
        )
