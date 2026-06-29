"""Bucket-parsing correctness — the engine must never mis-read a strike bound."""

from __future__ import annotations

import math
from datetime import date

from hedge.weather.markets import parse_temp_market


def _raw(**kw):
    base = {"ticker": "KXHIGHNY-25JUN28-X", "event_ticker": "KXHIGHNY-25JUN28"}
    base.update(kw)
    return base


def test_closed_range_between():
    m = parse_temp_market(_raw(strike_type="between", floor_strike=72, cap_strike=73))
    assert (m.lo_f, m.hi_f) == (72.0, 73.0)
    assert m.contains(72) and m.contains(73) and not m.contains(74)
    assert not m.is_tail


def test_upper_tail_greater_is_strict():
    # Verified live: floor=84, type="greater" renders "85° or above" -> high > 84.
    m = parse_temp_market(_raw(strike_type="greater", floor_strike=84))
    assert m.lo_f == 85.0 and math.isinf(m.hi_f)
    assert m.contains(85) and not m.contains(84)
    assert m.is_tail


def test_upper_tail_greater_or_equal_is_inclusive():
    m = parse_temp_market(_raw(strike_type="greater_or_equal", floor_strike=89))
    assert m.lo_f == 89.0 and m.contains(89)


def test_lower_tail_less_is_strict():
    # Verified live: cap=77, type="less" renders "76° or below" -> high < 77.
    m = parse_temp_market(_raw(strike_type="less", cap_strike=77))
    assert math.isinf(-m.lo_f) and m.hi_f == 76.0
    assert m.contains(76) and not m.contains(77)


def test_between_is_inclusive_both_ends():
    # Verified live: floor=77 cap=78 renders "77° to 78°".
    m = parse_temp_market(_raw(strike_type="between", floor_strike=77, cap_strike=78))
    assert m.contains(77) and m.contains(78) and not m.contains(79) and not m.contains(76)


def test_subtitle_fallback_range():
    m = parse_temp_market(_raw(yes_sub_title="100° to 101°"))
    assert (m.lo_f, m.hi_f) == (100.0, 101.0)


def test_subtitle_fallback_above():
    m = parse_temp_market(_raw(yes_sub_title="74° or above"))
    assert m.lo_f == 74.0 and math.isinf(m.hi_f)


def test_date_parsed_from_ticker():
    m = parse_temp_market(_raw(strike_type="between", floor_strike=72, cap_strike=73))
    assert m.local_date == date(2025, 6, 28)


def test_unknown_series_is_none():
    assert parse_temp_market(_raw(ticker="KXHIGHZZZ-25JUN28-X", event_ticker="KXHIGHZZZ-25JUN28")) is None
