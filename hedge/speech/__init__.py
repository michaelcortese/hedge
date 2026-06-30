"""Speech/word-mention prediction core.

Shared machinery for strategies that bet on Kalshi "will <speaker> say
'<phrase>' during <event>" markets. Mirrors the ``hedge/weather/`` split so the
strategy file stays thin:

  * ``markets``  — parse a mention market payload into a ``MentionMarket``.
  * ``corpus``   — pluggable base-rate providers (stub + leading transcript APIs)
                   behind one ``MentionCorpus`` seam, plus a key-config loader.
  * ``model``    — Beta-Binomial base-rate estimator -> P(>=1 mention) + std error.

A mention market resolves YES iff the speaker says the phrase at least once
during the event. There is no physical process to simulate; the signal is a
*base rate* over comparable past speeches, estimated with honest small-sample
uncertainty (a thin/weighted corpus -> large std error -> the sizing engine
won't overbet). See ``hedge/strategies/speech_mention.py``.
"""
