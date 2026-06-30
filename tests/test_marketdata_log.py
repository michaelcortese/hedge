"""Market + weather research data log: separate stream, retention, and runner capture."""

from __future__ import annotations

import types

from hedge.decision import RiskConfig
from hedge.eventlog import EventLog, read_all
from hedge.runner import Runner
from hedge.strategies.base import MarketView


# --------------------------------------------------------------------------- #
# EventLog: prefix-selected streams + retention                                #
# --------------------------------------------------------------------------- #
def test_prefix_writes_and_reads_a_separate_stream(tmp_path):
    log = EventLog(tmp_path, prefix="data")
    log.emit("market", {"ticker": "A"}, ts="2026-06-30T12:00:00+00:00")
    log.emit("weather", {"series": "KXHIGHNY"}, ts="2026-06-30T12:00:00+00:00")
    # Lands in data_*.jsonl, NOT events_*.jsonl.
    assert (tmp_path / "data_2026-06-30.jsonl").exists()
    assert not list(tmp_path.glob("events_*.jsonl"))
    got = read_all(tmp_path, prefix="data")
    assert [e["type"] for e in got] == ["market", "weather"]
    # The default prefix doesn't see the data stream.
    assert read_all(tmp_path) == []


def test_prune_keeps_only_newest_days(tmp_path):
    log = EventLog(tmp_path, prefix="data")
    for day in ("2026-06-25", "2026-06-26", "2026-06-27", "2026-06-28"):
        log.emit("market", {"d": day}, ts=f"{day}T12:00:00+00:00")
    removed = log.prune(keep_days=2)
    remaining = sorted(p.name for p in log.files())
    assert remaining == ["data_2026-06-27.jsonl", "data_2026-06-28.jsonl"]
    assert {p.name for p in removed} == {"data_2026-06-25.jsonl", "data_2026-06-26.jsonl"}


def test_prune_zero_keeps_everything(tmp_path):
    log = EventLog(tmp_path, prefix="data")
    log.emit("market", {}, ts="2026-06-25T00:00:00+00:00")
    log.emit("market", {}, ts="2026-06-26T00:00:00+00:00")
    assert log.prune(0) == [] and len(log.files()) == 2


# --------------------------------------------------------------------------- #
# Runner capture                                                               #
# --------------------------------------------------------------------------- #
def _ny_view():
    return MarketView("KXHIGHNY-25JUN28-B", {
        "ticker": "KXHIGHNY-25JUN28-B", "event_ticker": "KXHIGHNY-25JUN28",
        "strike_type": "between", "floor_strike": 80, "cap_strike": 81,
        "yes_bid": 40, "yes_ask": 45,
    }, orderbook={"orderbook": {"yes": [[40, 100]], "no": [[55, 80]]}})


def _stub_weather(monkeypatch):
    monkeypatch.setattr("hedge.weather.providers.open_meteo_forecast",
                        lambda st, d, *a, **k: [types.SimpleNamespace(daily_high_f=85.0),
                                                types.SimpleNamespace(daily_high_f=86.0)])
    monkeypatch.setattr("hedge.weather.providers.nws_recent_temps_f",
                        lambda st, d, *a, **k: [70.0, 78.0, 74.0])
    monkeypatch.setattr("hedge.weather.providers.open_meteo_forecast_raw",
                        lambda st, d, *a, **k: {"src": "open-meteo"})
    monkeypatch.setattr("hedge.weather.providers.nws_forecast_raw",
                        lambda st, d, *a, **k: {"src": "nws-forecast"})
    monkeypatch.setattr("hedge.weather.providers.nws_observations_raw",
                        lambda st, d, *a, **k: {"src": "nws-obs"})


def _runner(tmp_path, *, capture=True):
    datalog = EventLog(tmp_path / "marketdata", prefix="data")
    # client=None is fine: _capture_market_data never touches the client.
    return Runner(None, [], executor=types.SimpleNamespace(dry_run=True, env="demo"),
                  cfg=RiskConfig(), datalog=datalog, capture_market_data=capture,
                  state=types.SimpleNamespace())


def test_capture_emits_market_and_weather(monkeypatch, tmp_path):
    _stub_weather(monkeypatch)
    r = _runner(tmp_path)
    r._capture_market_data([_ny_view()])
    ev = read_all(tmp_path / "marketdata", prefix="data")
    market = [e for e in ev if e["type"] == "market"]
    weather = [e for e in ev if e["type"] == "weather"]
    assert len(market) == 1 and len(weather) == 1
    # Market event carries the raw payload + the order book verbatim.
    assert market[0]["raw"]["floor_strike"] == 80
    assert market[0]["orderbook"]["orderbook"]["no"] == [[55, 80]]
    # Weather event carries the raw payloads + derived highs / obs-max.
    w = weather[0]
    assert w["raw_open_meteo"] == {"src": "open-meteo"}
    assert w["raw_nws_forecast"] == {"src": "nws-forecast"}
    assert w["point_highs"] == [85.0, 86.0] and w["obs_max"] == 78.0


def test_weather_emitted_once_per_station_day(monkeypatch, tmp_path):
    _stub_weather(monkeypatch)
    r = _runner(tmp_path)
    # Two buckets of the SAME city-day -> two market events, ONE weather event.
    v2 = MarketView("KXHIGHNY-25JUN28-C", {
        "ticker": "KXHIGHNY-25JUN28-C", "event_ticker": "KXHIGHNY-25JUN28",
        "strike_type": "between", "floor_strike": 82, "cap_strike": 83,
        "yes_bid": 20, "yes_ask": 25,
    })
    r._capture_market_data([_ny_view(), v2])
    ev = read_all(tmp_path / "marketdata", prefix="data")
    assert sum(e["type"] == "market" for e in ev) == 2
    assert sum(e["type"] == "weather" for e in ev) == 1


def test_capture_disabled_writes_nothing(monkeypatch, tmp_path):
    _stub_weather(monkeypatch)
    r = _runner(tmp_path, capture=False)
    r._capture_market_data([_ny_view()])
    assert read_all(tmp_path / "marketdata", prefix="data") == []
