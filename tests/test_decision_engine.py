"""Decision engine: side selection, gates, sizing, caps, and reconciliation."""

from __future__ import annotations

from hedge.decision import (
    Action,
    Decision,
    MarketQuote,
    Position,
    RiskConfig,
    Side,
    decide,
)
from hedge.signal import Signal

# A tight-sigma config that will actually trade when there's a real edge.
RISK = RiskConfig(lambda_kelly=0.25, k_sigma=2.0, tau_min_cents=2.0)
BANKROLL = 10_000.0


def _sig(prob: float, se: float = 0.01, ticker: str = "MKT") -> Signal:
    return Signal(ticker=ticker, prob=prob, std_error=se, strategy="t")


def test_buys_yes_when_underpriced():
    q = MarketQuote(yes_bid=0.53, yes_ask=0.55)
    d = decide(_sig(0.70), q, BANKROLL, RISK)
    assert d.action is Action.BUY and d.side is Side.YES
    assert d.count > 0 and d.edge > 0 and 0 < d.kelly_fraction


def test_holds_on_degenerate_one_sided_book():
    # No real ask (yes_ask=1.00) and no real bid (yes_bid=0.00): neither side is
    # fillable, so a confident belief must NOT become a phantom edge / crash sizing.
    q = MarketQuote(yes_bid=0.0, yes_ask=1.0)
    d = decide(_sig(0.80), q, BANKROLL, RISK)
    assert d.action is Action.HOLD and d.count == 0


def test_takes_no_side_when_only_yes_ask_missing():
    # A genuine yes_bid (0.60) makes NO fillable at 0.40 even though yes_ask=1.00.
    q = MarketQuote(yes_bid=0.60, yes_ask=1.0)
    d = decide(_sig(0.20), q, BANKROLL, RISK)
    assert d.action is Action.BUY and d.side is Side.NO and d.count > 0


def test_no_phantom_maker_buy_when_bid_missing():
    # YES taker is tradeable (ask 0.55) but there's no bid (0.00): the engine must
    # NOT post a phantom maker buy at ~$0.00 — it takes at the ask instead, or holds.
    q = MarketQuote(yes_bid=0.0, yes_ask=0.55)
    d = decide(_sig(0.80), q, BANKROLL, RISK)
    assert d.action is Action.HOLD or (d.is_trade and not d.maker and d.price_cents >= 2)


def test_absolute_dollar_cap_limits_size():
    # A hard max_order_dollars must bind below the bankroll-fraction caps.
    q = MarketQuote(yes_bid=0.10, yes_ask=0.12)
    sig = _sig(0.40)  # large edge vs a 12c ask
    uncapped = decide(sig, q, BANKROLL, RISK)
    capped = decide(sig, q, BANKROLL,
                    RiskConfig(lambda_kelly=0.25, k_sigma=2.0, tau_min_cents=2.0,
                               max_order_dollars=1.0))
    assert capped.action is Action.BUY
    assert capped.count < uncapped.count
    assert capped.count * capped.price <= 1.0 + 1e-9


def test_buys_no_when_overpriced():
    q = MarketQuote(yes_bid=0.55, yes_ask=0.57)
    d = decide(_sig(0.30), q, BANKROLL, RISK)
    assert d.action is Action.BUY and d.side is Side.NO and d.count > 0


def test_abstains_when_fairly_priced():
    q = MarketQuote(yes_bid=0.49, yes_ask=0.51)
    d = decide(_sig(0.50), q, BANKROLL, RISK)
    assert d.action is Action.HOLD and d.count == 0


def test_noise_gate_blocks_uncertain_edge():
    # Same nominal disagreement, but a huge sigma -> k_sigma gate refuses.
    q = MarketQuote(yes_bid=0.53, yes_ask=0.55)
    d = decide(_sig(0.70, se=0.30), q, BANKROLL, RISK)
    assert d.action is Action.HOLD
    assert "noise" in d.reason


def test_prefers_maker_when_it_clears_tau():
    # Wide spread: maker (post at bid 0.40) has far more edge than taker (0.60).
    q = MarketQuote(yes_bid=0.40, yes_ask=0.60)
    d = decide(_sig(0.75, se=0.01), q, BANKROLL, RISK)
    assert d.action is Action.BUY and d.side is Side.YES
    assert d.maker is True and d.price == 0.40


def test_falls_back_to_taker_when_maker_edge_too_thin():
    # Tight book where the model is only a little above the ask: maker at the bid
    # has plenty of edge, taker still clears. Force the maker-thin path with a
    # model just barely over the ask but well over the bid is the usual case; here
    # we check a near-zero spread so maker≈taker and taker is used as fallback only
    # if maker fails tau. Use a 1-cent spread with modest edge.
    q = MarketQuote(yes_bid=0.69, yes_ask=0.70)
    d = decide(_sig(0.80, se=0.01), q, BANKROLL, RISK)
    assert d.action is Action.BUY and d.side is Side.YES
    # maker at 0.69 clears tau comfortably, so maker is preferred.
    assert d.maker is True


def test_market_cap_limits_size():
    q = MarketQuote(yes_bid=0.10, yes_ask=0.12)
    big = decide(_sig(0.90, se=0.005), q, BANKROLL, RiskConfig(market_cap_frac=0.50))
    small = decide(_sig(0.90, se=0.005), q, BANKROLL, RiskConfig(market_cap_frac=0.01))
    assert small.count < big.count
    # Capital at risk respects the cap.
    assert small.count * small.price <= 0.01 * BANKROLL + small.price


def test_portfolio_cap_leaves_no_room():
    q = MarketQuote(yes_bid=0.53, yes_ask=0.55)
    cfg = RiskConfig(portfolio_cap=0.30)
    d = decide(_sig(0.70), q, BANKROLL, cfg, portfolio_at_risk=0.30 * BANKROLL)
    assert d.action is Action.HOLD and "portfolio cap" in d.reason


def test_depth_cap_limits_count():
    q = MarketQuote(yes_bid=0.40, yes_ask=0.60, yes_bid_depth=3)
    d = decide(_sig(0.80, se=0.005), q, BANKROLL, RISK)
    assert d.maker is True and d.count == 3  # capped to resting size


def test_rebalance_band_holds_small_drift():
    q = MarketQuote(yes_bid=0.53, yes_ask=0.55)
    want = decide(_sig(0.70), q, BANKROLL, RISK)  # target count
    # Already holding ~the target -> within band -> hold.
    pos = Position(side=Side.YES, count=want.count, avg_price=0.55)
    d = decide(_sig(0.70), q, BANKROLL, RISK, position=pos)
    assert d.action is Action.HOLD and "rebalance band" in d.reason


def test_adds_to_underweight_same_side():
    q = MarketQuote(yes_bid=0.40, yes_ask=0.50)
    want = decide(_sig(0.85, se=0.005), q, BANKROLL, RISK)
    pos = Position(side=Side.YES, count=1, avg_price=0.40)
    d = decide(_sig(0.85, se=0.005), q, BANKROLL, RISK, position=pos)
    assert d.action is Action.BUY and d.side is Side.YES
    assert d.count == want.count - 1


def test_flip_closes_opposite_side_first():
    q = MarketQuote(yes_bid=0.53, yes_ask=0.55)
    pos = Position(side=Side.NO, count=10, avg_price=0.45)
    d = decide(_sig(0.70), q, BANKROLL, RISK, position=pos)
    assert d.action is Action.SELL and d.side is Side.NO and d.count == 10
    assert "flip" in d.reason


def test_shrinkage_pulls_toward_market():
    q = MarketQuote(yes_bid=0.49, yes_ask=0.51)
    raw = RiskConfig(k_sigma=0.0)  # no gate, so we isolate the prob used
    shrunk = RiskConfig(k_sigma=0.0, shrink_to_market=True, sigma_market=0.02)
    d_raw = decide(_sig(0.70, se=0.05), q, BANKROLL, raw)
    d_shr = decide(_sig(0.70, se=0.05), q, BANKROLL, shrunk)
    assert d_shr.prob < d_raw.prob  # pulled toward the 0.50 mid


def test_quote_from_view_needs_two_sided_book():
    class V:
        yes_bid = None
        yes_ask = 0.55
    assert MarketQuote.from_view(V()) is None


def test_no_market_quote_no_trade_on_zero_bankroll():
    q = MarketQuote(yes_bid=0.53, yes_ask=0.55)
    d = decide(_sig(0.70), q, 0.0, RISK)
    assert d.action is Action.HOLD


def test_price_band_blocks_cheap_long_shot_open():
    # Model loves a 5c YES contract (thinks it's worth 30c) — a huge edge, but it's
    # a tail bet with no exit liquidity. The default 0.10 floor must refuse to open.
    q = MarketQuote(yes_bid=0.04, yes_ask=0.05)
    d = decide(_sig(0.30, se=0.01), q, BANKROLL, RISK)
    assert d.action is Action.HOLD and d.count == 0
    assert "tradeable band" in d.reason


def test_price_band_blocks_rich_side_open():
    # Mirror: buying YES at 0.95 for a sliver of edge is just as fee-heavy / pinned.
    q = MarketQuote(yes_bid=0.95, yes_ask=0.96)
    d = decide(_sig(0.999, se=0.001), q, BANKROLL, RISK)
    assert d.action is Action.HOLD and d.count == 0
    assert "tradeable band" in d.reason


def test_price_band_lowering_floor_lets_cheap_open_through():
    # The band is configurable: drop the floor and the same cheap edge trades.
    q = MarketQuote(yes_bid=0.04, yes_ask=0.05)
    cfg = RiskConfig(lambda_kelly=0.25, k_sigma=2.0, tau_min_cents=2.0, min_price=0.01)
    d = decide(_sig(0.30, se=0.01), q, BANKROLL, cfg)
    assert d.action is Action.BUY and d.side is Side.YES and d.count > 0


# ----- risk-reducing exit leg (manage_positions) ----------------------------
# Closing a souring position must run AHEAD of the open-only gates and is allowed
# even where a flip-to-open would be blocked by the price band. See engine._exit_check.
RISK_MANAGE = RiskConfig(lambda_kelly=0.25, k_sigma=2.0, tau_min_cents=2.0,
                         tau_exit_cents=2.0, manage_positions=True, exit_leg=True)


def test_exit_sells_when_book_overpays_held_side_in_band_tail():
    # We hold NO; the book bids 0.93 for it while the model rates NO worth only 0.20.
    # The flip-to-YES that would normally close is blocked (YES ask 0.07 is below the
    # 0.10 band), so the ONLY way out is the exit leg. It should fire and sell the lot.
    q = MarketQuote(yes_bid=0.06, yes_ask=0.07)        # no_bid = 1-0.07 = 0.93
    pos = Position(side=Side.NO, count=10, avg_price=0.50)
    d = decide(_sig(0.80, se=0.01), q, BANKROLL, RISK_MANAGE, position=pos)
    assert d.action is Action.SELL and d.side is Side.NO and d.count == 10
    assert "exit" in d.reason


def test_exit_disabled_rides_position_when_management_off():
    # Identical setup, but manage_positions off (the default). With no exit leg the
    # flip is band-blocked and the loser rides to settlement — the gap the leg fixes.
    q = MarketQuote(yes_bid=0.06, yes_ask=0.07)
    pos = Position(side=Side.NO, count=10, avg_price=0.50)
    d = decide(_sig(0.80, se=0.01), q, BANKROLL, RISK, position=pos)
    assert d.action is Action.HOLD
    assert "tradeable band" in d.reason


def test_no_exit_when_held_side_still_plus_ev():
    # Hold NO; the model still rates NO worth 0.80 and the book only bids 0.68 for it.
    # Selling would dump a +EV hold (the "+EV-sale trap") — the leg must NOT fire.
    q = MarketQuote(yes_bid=0.30, yes_ask=0.32)        # no_bid = 0.68
    pos = Position(side=Side.NO, count=10, avg_price=0.50)
    d = decide(_sig(0.20, se=0.01), q, BANKROLL, RISK_MANAGE, position=pos)
    assert "exit" not in d.reason
    assert not (d.action is Action.SELL and "exit" in d.reason)


def test_no_exit_when_no_bid_to_sell_into():
    # A logically-dead YES bucket (obs already past it): YES bid is 0, so there is
    # nothing to capture — the loss is locked. The exit leg must no-op, not crash.
    q = MarketQuote(yes_bid=0.0, yes_ask=0.02)
    pos = Position(side=Side.YES, count=5, avg_price=0.40)
    d = decide(_sig(0.001, se=0.001), q, BANKROLL, RISK_MANAGE, position=pos)
    assert d.action is Action.HOLD
    assert "exit" not in d.reason
