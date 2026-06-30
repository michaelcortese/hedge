"""Append-only durable event log: envelope, partitioning, append-only, recovery."""

from __future__ import annotations

import json

from hedge.eventlog import EventLog, default_event_dir, iter_events, read_all


def test_emit_writes_envelope_and_payload(tmp_path):
    log = EventLog(tmp_path)
    log.emit("decision", {"ticker": "T1", "prob": 0.8}, seq=5, ts="2026-06-30T12:00:00+00:00")
    events = list(log.iter_day("2026-06-30"))
    assert len(events) == 1
    e = events[0]
    assert e["type"] == "decision" and e["seq"] == 5
    assert e["ticker"] == "T1" and e["prob"] == 0.8
    assert e["ts"] == "2026-06-30T12:00:00+00:00"


def test_is_append_only_accumulates(tmp_path):
    log = EventLog(tmp_path)
    for i in range(3):
        log.emit("cycle", {"i": i}, ts="2026-06-30T00:00:00+00:00")
    events = list(log.iter_day("2026-06-30"))
    assert [e["i"] for e in events] == [0, 1, 2]   # appended in order, none overwritten


def test_partitioned_by_utc_day(tmp_path):
    log = EventLog(tmp_path)
    log.emit("cycle", {"d": "a"}, ts="2026-06-30T23:59:00+00:00")
    log.emit("cycle", {"d": "b"}, ts="2026-07-01T00:01:00+00:00")
    assert (tmp_path / "events_2026-06-30.jsonl").exists()
    assert (tmp_path / "events_2026-07-01.jsonl").exists()
    assert [e["d"] for e in log.iter_day("2026-06-30")] == ["a"]
    assert [e["d"] for e in log.iter_day("2026-07-01")] == ["b"]


def test_reserved_envelope_keys_win(tmp_path):
    # A payload must not be able to shadow ts/type/seq.
    log = EventLog(tmp_path)
    log.emit("fill", {"type": "EVIL", "seq": 999}, seq=1, ts="2026-06-30T00:00:00+00:00")
    e = list(log.iter_day("2026-06-30"))[0]
    assert e["type"] == "fill" and e["seq"] == 1


def test_iter_events_skips_corrupt_lines(tmp_path):
    p = tmp_path / "events_2026-06-30.jsonl"
    p.write_text(
        json.dumps({"type": "a", "ts": "x"}) + "\n"
        + "{ this is not json\n"            # torn / partial line
        + "\n"                               # blank line
        + json.dumps({"type": "b", "ts": "y"}) + "\n"
    )
    types = [e["type"] for e in iter_events(p)]
    assert types == ["a", "b"]


def test_read_all_aggregates_across_days(tmp_path):
    log = EventLog(tmp_path)
    log.emit("cycle", {"n": 1}, ts="2026-06-29T00:00:00+00:00")
    log.emit("cycle", {"n": 2}, ts="2026-06-30T00:00:00+00:00")
    assert [e["n"] for e in read_all(tmp_path)] == [1, 2]


def test_iter_missing_day_is_empty(tmp_path):
    assert list(EventLog(tmp_path).iter_day("2026-01-01")) == []


def test_default_dir_honors_state_dir(monkeypatch):
    monkeypatch.setenv("HEDGE_STATE_DIR", "/data")
    assert str(default_event_dir()) == "/data/events"
    monkeypatch.delenv("HEDGE_STATE_DIR", raising=False)
    assert str(default_event_dir()) == "data/runs/live/events"


def test_emit_never_raises_on_bad_dir(tmp_path):
    # A path that can't be created (a file where a dir should be) must not raise.
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file")
    log = EventLog(blocker / "subdir")
    log.emit("cycle", {"ok": True})   # must swallow the OSError, not propagate
