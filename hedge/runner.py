"""The main trading loop: signals -> decide -> execute, repeated.

This is the live counterpart to the paper tournament. Each cycle it:

    1. reads bankroll + open positions from Kalshi,
    2. discovers the open temperature markets and builds a ``MarketView`` each,
    3. runs every strategy to collect ``Signal``s,
    4. routes each signal through the decision engine, picking the single
       best-edge decision per market (strategies share one order book, so we
       never stack conflicting orders on the same ticker),
    5. (optional, ``cfg.manage_positions``, default OFF) reconciles every OPEN
       position against the current model — trim / add / flip-to-exit — incl.
       positions discovery missed; off by default pending churn/P&L-accounting fixes,
    6. hands tradable decisions to the ``Executor``,
    7. logs every decision (acted-on or not) for later review.

Safety: the executor is DRY-RUN by default and refuses prod without an explicit
opt-in, so ``python -m hedge.runner`` out of the box only *reports* what it would
trade. Arm real orders with ``--live`` (and ``--allow-prod`` for production).
"""

from __future__ import annotations

import argparse
import json
import os
import signal as _signal
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from hedge import alerts
from hedge.config import build_client
from hedge.decision import Action, Decision, MarketQuote, Position, RiskConfig, Side, decide
from hedge.execution import Executor, OrderTicket
from hedge.execution.executor import parse_order
from hedge.guard import GuardConfig, GuardStatus, assess, market_skill
from hedge.state import State
from hedge.strategies.base import MarketView, Strategy
from hedge.strategies.weather_blend import WeatherBlendStrategy
from hedge.strategies.weather_ensemble import WeatherEnsembleStrategy
from hedge.strategies.weather_nowcast import WeatherNowcastStrategy
from hedge.weather.calibration import CalibrationTable
from hedge.weather.markets import discover_temp_markets
from hedge.weather.sources import LiveForecastSource
from hedge.weather.stations import STATIONS, station_for_ticker

LIVE_DIR = Path("data/runs/live")


def _event_key(ticker: str) -> str:
    """Group key for one Kalshi event (city-day): the series + date code, e.g.
    ``KXHIGHMIA-26JUN30`` from ``KXHIGHMIA-26JUN30-B93.5``. All buckets of a city-day
    share this key, so the engine can cap their combined (correlated) risk."""
    return "-".join(ticker.split("-")[:2])


def _live_calibration() -> CalibrationTable:
    """Fit per-city forecast-error spread/bias for live trading, with a safe fallback.

    The live bot previously ran with an EMPTY table — prior σ and ZERO bias correction,
    so the systematic grid→station settlement offset went uncorrected. We fit the same
    way the paper tournament does (recent disjoint window, ERA5-lag aware), honoring
    HEDGE_CALIBRATE_AGAINST (era5|station) for the truth source. Any failure (offline
    container, slow archive) falls back to the prior curve so a fit hiccup can never
    stop the loop. The σ floor inside fit_calibration prevents the correlated-source
    collapse regardless of truth source.
    """
    truth = os.environ.get("HEDGE_CALIBRATE_AGAINST", "era5").strip().lower()
    try:
        from datetime import timedelta

        from hedge.weather.calibration import fit_calibration
        end = datetime.now(ZoneInfo("UTC")).date() - timedelta(days=7)  # ERA5 lags ~5d
        start = end - timedelta(days=45)
        table = fit_calibration(list(STATIONS.values()), start, end, truth=truth)
        fitted = sorted({s for (s, _lead) in table.sigma})
        print(f"[calib] fit {start}..{end} truth={truth}; "
              f"calibrated series: {fitted or 'none (using priors)'}", flush=True)
        return table
    except Exception as e:  # noqa: BLE001 — never let calibration block trading
        print(f"[calib] fit failed ({type(e).__name__}: {e}); using priors.", flush=True)
        return CalibrationTable()


def _default_strategies() -> list[Strategy]:
    src = LiveForecastSource()
    calib = _live_calibration()
    # NOTE: weather_climatology is deliberately EXCLUDED from live trading — it is the
    # null benchmark every real strategy must beat in the tournament, not an alpha
    # source. Trading it with real money would bet the baseline. It stays in the
    # backtest/paper tournament only.
    return [
        WeatherBlendStrategy(src, calib),
        WeatherNowcastStrategy(src, calib),
        WeatherEnsembleStrategy(src, calib),
    ]


class Runner:
    """Drives one or many trade cycles. Pure-ish: all I/O goes through `client`."""

    def __init__(self, client, strategies, executor: Executor, cfg: RiskConfig,
                 *, bankroll_override: float | None = None,
                 guard_cfg: GuardConfig | None = None,
                 strategy_lambda: dict[str, float] | None = None,
                 state: State | None = None):
        self.client = client
        self.strategies = strategies
        self.executor = executor
        self.cfg = cfg
        self.bankroll_override = bankroll_override
        self.guard_cfg = guard_cfg or GuardConfig()
        # Per-strategy Kelly-λ multiplier (#3): run the no-edge morning forecast
        # strategies at ~0 size (paper/log only) and put size on the intraday nowcast.
        # Missing strategy -> 1.0 (full base λ). Composes with the skill multiplier.
        self.strategy_lambda = strategy_lambda or {}
        self._skill_mult = 1.0   # refreshed each cycle when the skill gate is enabled
        # Durable state lives next to the decision logs (the Fly volume in prod).
        self.state = state if state is not None else State(LIVE_DIR / "hedge.db")

    # ----- state from Kalshi -------------------------------------------------
    def bankroll(self) -> float:
        if self.bankroll_override is not None:
            return self.bankroll_override
        try:
            bal = self.client.get_balance()
            return float(bal.get("balance", 0)) / 100.0  # cents -> dollars
        except Exception as e:  # noqa: BLE001
            print(f"[runner] balance fetch failed ({e}); bankroll=0", flush=True)
            return 0.0

    def positions(self) -> dict[str, Position]:
        """Map ticker -> current Position from Kalshi (the source of truth).

        Kalshi's net holding is signed: positive = long YES, negative = long NO.
        The live API reports it as ``position_fp`` (a fixed-point STRING, e.g.
        ``"-4.00"``) and the cost basis as ``market_exposure_dollars`` (a dollar
        STRING). Older payloads used integer ``position`` (contracts) and
        ``market_exposure`` (cents); we read the dollar fields first and fall back
        to the legacy ones so a field rename can't silently blind us to our holdings
        (which is exactly the bug that let positions go unmanaged: the engine's
        flip/trim/add reconciliation only fires when it is handed a Position).
        """
        out: dict[str, Position] = {}
        try:
            data = self.client.get_positions()
        except Exception as e:  # noqa: BLE001
            print(f"[runner] positions fetch failed ({e}); assuming flat", flush=True)
            return out
        for mp in data.get("market_positions", []) or []:
            ticker = mp.get("ticker")
            pos_fp = mp.get("position_fp")
            net = round(float(pos_fp)) if pos_fp not in (None, "") \
                else int(mp.get("position", 0) or 0)
            if not ticker or net == 0:
                continue
            side = Side.YES if net > 0 else Side.NO
            count = abs(net)
            exp_d = mp.get("market_exposure_dollars")
            exposure = float(exp_d) if exp_d not in (None, "") \
                else float(mp.get("market_exposure", 0) or 0) / 100.0
            avg = exposure / count if count else 0.0
            out[ticker] = Position(side=side, count=count, avg_price=avg)
        return out

    def _held_ticker_views(self, positions: dict[str, Position],
                           covered: set[str]) -> list[MarketView]:
        """Build a MarketView for every HELD ticker that today's discovery missed.

        Discovery only returns currently-open markets in our covered series, so a
        position can fall outside it — settlement is near and it aged out, the series
        is no longer discovered, or (the case that motivates this) a freshly redeployed
        bot inherits positions and must re-evaluate them against the *current* model
        rather than holding blindly to settlement. Pulling each held market directly
        lets the same decide()/_reconcile() path trim, add to, or exit it. A market
        with no two-sided quote (e.g. already settled) yields no MarketQuote later and
        is skipped — settlement booking handles those.
        """
        extra: list[MarketView] = []
        for ticker in positions:
            if ticker in covered:
                continue
            try:
                raw = self.client.get_market(ticker).get("market", {}) or {}
            except Exception as e:  # noqa: BLE001
                print(f"[runner] manage: no market for {ticker} ({e}); skipped", flush=True)
                continue
            if raw:
                extra.append(MarketView(raw.get("ticker", ticker), raw))
        return extra

    # ----- anti-churn cooldown (per managed ticker) --------------------------
    def _manage_cooldown_active(self, ticker: str) -> bool:
        """True if we managed ``ticker`` within the last cfg.manage_cooldown_cycles."""
        last = self.state.get_meta(f"mgmt_cycle:{ticker}")
        if last is None:
            return False
        return (getattr(self, "_cycle_seq", 0) - int(last)) < self.cfg.manage_cooldown_cycles

    def _mark_manage(self, ticker: str) -> None:
        """Stamp the current cycle as the last time we managed ``ticker``."""
        self.state.set_meta(f"mgmt_cycle:{ticker}", str(getattr(self, "_cycle_seq", 0)))

    def market_views(self) -> list[MarketView]:
        views: list[MarketView] = []
        for series in STATIONS:
            try:
                for raw in discover_temp_markets(self.client, series, status="open"):
                    ticker = raw.get("ticker", "")
                    views.append(MarketView(ticker, raw, orderbook=self._orderbook(ticker)))
            except Exception as e:  # noqa: BLE001
                print(f"[runner] market discovery failed for {series} ({e})", flush=True)
        return views

    def _orderbook(self, ticker: str) -> dict | None:
        """Fetch the order book for one ticker (best-effort).

        The engine's depth / participation caps and book-reconstructed top-of-book
        need the real ladder; without it those caps are dead code and sizing is blind
        to liquidity. A fetch failure returns None so the engine falls back to the
        stale GET /markets quote rather than aborting the cycle. Reads are cheap and
        well under the Basic-tier ~20 reads/s limit for our small (4-city) universe.
        """
        if not ticker:
            return None
        try:
            return self.client.get_orderbook(ticker)
        except Exception as e:  # noqa: BLE001 — book is best-effort; degrade gracefully
            print(f"[runner] orderbook fetch failed for {ticker} ({e})", flush=True)
            return None

    # ----- calibration kill-switch ------------------------------------------
    def _halt_file(self) -> Path:
        return LIVE_DIR / "HALTED"

    def is_halted(self) -> tuple[bool, str]:
        f = self._halt_file()
        if f.exists():
            return True, f.read_text().strip()
        return False, ""

    def reset_guard(self) -> bool:
        """Clear the halt latch. Returns True if a latch was present."""
        f = self._halt_file()
        if f.exists():
            f.unlink()
            return True
        return False

    def _latch_halt(self, reason: str) -> None:
        """Write the halt latch (durable on the volume). Cleared only by --reset-guard."""
        LIVE_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(ZoneInfo("UTC")).isoformat()
        self._halt_file().write_text(f"{stamp} {reason}\n")

    def _trip(self, status: GuardStatus) -> None:
        self._latch_halt(status.reason)
        print(f"[runner] *** KILL-SWITCH TRIPPED *** {status.reason}", flush=True)
        alerts.notify(alerts.Level.CRITICAL, "hedge: kill-switch tripped", status.reason)
        if self.guard_cfg.flatten_on_trip:
            self._flatten()

    def _recent_decision_rows(self) -> list[dict]:
        """Read acted-on opening (BUY) decisions from the last window_days logs.

        Only BUYs feed the calibration kill-switch: the guard scores the Brier of our
        *entry* beliefs against outcomes. A SELL-to-exit carries the new, opposite belief
        that triggered the flip — scoring it would penalize a correct de-risking as if it
        were a miscalibrated entry and could falsely trip the latch. So sells are excluded.
        """
        if not LIVE_DIR.exists():
            return []
        rows: list[dict] = []
        for path in sorted(LIVE_DIR.glob("decisions_*.jsonl"))[-self.guard_cfg.window_days:]:
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("action") == "buy" and r.get("prob") is not None:
                    rows.append(r)
        return rows

    def _settlement_outcomes(self, tickers: set[str]) -> dict[str, bool]:
        out: dict[str, bool] = {}
        for ticker in tickers:
            try:
                m = self.client.get_market(ticker).get("market", {})
            except Exception:  # noqa: BLE001
                continue
            result = str(m.get("result", "")).lower()
            if result in ("yes", "no"):
                out[ticker] = result == "yes"
        return out

    def evaluate_guard(self) -> GuardStatus:
        """Score realized calibration of acted-on signals on settled markets."""
        if not self.guard_cfg.enabled:
            return assess([], self.guard_cfg)
        rows = self._recent_decision_rows()
        outcomes = self._settlement_outcomes({r["ticker"] for r in rows})
        # One sample per (ticker) we acted on that has since settled. De-dup by
        # ticker keeping the first acted prob, so repeated cycles on one market
        # don't over-weight it.
        seen: set[str] = set()
        samples: list[tuple[float, bool]] = []
        for r in rows:
            t = r["ticker"]
            if t in seen or t not in outcomes:
                continue
            seen.add(t)
            samples.append((float(r["prob"]), outcomes[t]))
        return assess(samples, self.guard_cfg)

    def evaluate_skill_multiplier(self) -> float:
        """λ multiplier in [skill_floor, 1] from OOS skill vs the market mid (#3).

        Scores acted-on opening probabilities against the market mid recorded at
        decision time, on markets that have since settled, and ramps size up only as
        the model demonstrably beats the mid. 1.0 (no effect) when the gate is off.
        Demo/dry-run BUY decisions are logged and scored too, so the gate can be proven
        before real size is armed. Best-effort: any hiccup yields the conservative floor.
        """
        if not self.guard_cfg.skill_gate:
            return 1.0
        try:
            rows = self._recent_decision_rows()
            outcomes = self._settlement_outcomes({r["ticker"] for r in rows})
            seen: set[str] = set()
            samples: list[tuple[float, float, bool]] = []
            for r in rows:
                t = r["ticker"]
                mid = (r.get("meta") or {}).get("mid")
                if t in seen or t not in outcomes or mid is None:
                    continue
                seen.add(t)
                samples.append((float(r["prob"]), float(mid), outcomes[t]))
            return market_skill(samples, self.guard_cfg).multiplier
        except Exception as e:  # noqa: BLE001 — gate must never crash a cycle
            print(f"[runner] skill gate failed ({e}); using floor", flush=True)
            return self.guard_cfg.skill_floor

    def _flatten(self) -> list[OrderTicket]:
        """Sell out of every open position (cross to the bid). Used on a trip."""
        tickets: list[OrderTicket] = []
        for ticker, pos in self.positions().items():
            try:
                view = MarketView(ticker, self.client.get_market(ticker).get("market", {}))
                quote = MarketQuote.from_view(view)
            except Exception as e:  # noqa: BLE001
                print(f"[runner] flatten: no quote for {ticker} ({e}); skipped", flush=True)
                continue
            if quote is None:
                continue
            exit_px = quote.yes_bid if pos.side is Side.YES else quote.no_bid
            d = Decision(
                ticker=ticker, action=Action.SELL, side=pos.side,
                price=exit_px, price_cents=max(1, min(99, round(exit_px * 100))),
                count=pos.count, maker=False, reason="kill-switch flatten",
            )
            tickets.append(self.executor.place(d, idem_key=self._idem_key(d) + "-flat"))
        return tickets

    # ----- one cycle ---------------------------------------------------------
    def run_cycle(self) -> list[OrderTicket]:
        seq = self.state.next_cycle_seq()
        self._cycle_seq = seq   # client_order_ids are keyed on this (idempotency)
        # Reconcile our orders against broker truth (fills/cancels; cancel-replace
        # stale resting makers), pull actual fill economics, book realized P&L from
        # intraday closes (so the daily-loss stop sees churn), then book any
        # newly-settled realized P&L on the net held position.
        self._reconcile_orders()
        self._reconcile_fills()
        self._book_intraday_realized()
        self._book_settlements()

        # Circuit breaker first: never trade past a tripped/halted guard.
        halted, reason = self.is_halted()
        if halted:
            print(f"[runner] HALTED — not trading. ({reason}) "
                  "Clear with: run_live.py --reset-guard", flush=True)
            return []

        # Daily-loss stop: latch a halt for the rest of the UTC day if realized
        # losses breach the cap. Independent of the calibration kill-switch.
        stop = self.cfg.daily_loss_stop_dollars
        if stop is not None:
            realized = self.state.realized_today()
            if realized <= -abs(stop):
                msg = f"daily-loss stop: realized ${realized:.2f} today <= -${abs(stop):.2f}"
                self._latch_halt(msg)
                print(f"[runner] *** DAILY-LOSS STOP *** {msg}", flush=True)
                alerts.notify(alerts.Level.CRITICAL, "hedge: daily-loss stop", msg)
                return []

        if self.guard_cfg.enabled:
            status = self.evaluate_guard()
            if status.tripped:
                self._trip(status)
                return []

        # Market-relative skill gate (#3): scale λ by demonstrated OOS skill vs the
        # market mid. 1.0 when the gate is disabled; ~0 until the edge is proven.
        self._skill_mult = self.evaluate_skill_multiplier()

        bankroll = self.bankroll()
        positions = self.positions()
        views = self.market_views()
        # Intraday position management (gated by cfg.manage_positions, default OFF).
        # When on, re-evaluate EVERY open position each cycle — not just markets in
        # today's discovery — so the engine can trim / add / flip-to-exit, including
        # positions inherited after a redeploy. When off (the safe default), we only
        # OPEN new positions; existing holdings are read for the portfolio cap but
        # never reconciled. See RiskConfig.manage_positions for why it's off.
        if self.cfg.manage_positions:
            covered = {v.ticker for v in views}
            views += self._held_ticker_views(positions, covered)
        self._heartbeat(bankroll, positions)

        # Capital already at risk in open positions seeds the portfolio cap, and —
        # grouped by event (city-day) — the per-event concentration cap.
        portfolio_at_risk = sum(p.count * p.avg_price for p in positions.values())
        event_at_risk: dict[str, float] = {}
        for tk, p in positions.items():
            event_at_risk[_event_key(tk)] = (
                event_at_risk.get(_event_key(tk), 0.0) + p.count * p.avg_price)

        tickets: list[OrderTicket] = []
        decisions_log: list[dict] = []
        for view in views:
            quote = MarketQuote.from_view(view)
            if quote is None:
                continue
            ekey = _event_key(view.ticker)
            best = self._best_decision(view, quote, bankroll, positions,
                                       portfolio_at_risk, event_at_risk.get(ekey, 0.0))
            if best is None:
                continue
            decision, strat = best
            # Risk context as it stood when this market was decided (before we reserve
            # capital for its own order below) — logged for later reconstruction.
            par_at_decision = portfolio_at_risk
            # Anti-churn cooldown: on a ticker we already HOLD and just managed, don't
            # act again for cfg.manage_cooldown_cycles cycles. Stops a noisy boundary
            # from sell->rebuy->sell flip-flopping (fee bleed) and a duplicate exit from
            # being re-sent before the close lands in positions(). Never gates a fresh
            # open (a ticker we don't hold).
            if (decision.is_trade and self.cfg.manage_positions
                    and view.ticker in positions and self._manage_cooldown_active(view.ticker)):
                decision = Decision(
                    ticker=decision.ticker, action=Action.HOLD,
                    prob=decision.prob, sigma=decision.sigma,
                    reason=f"manage cooldown active ({self.cfg.manage_cooldown_cycles} cyc)",
                )
                decision.meta["strategy"] = strat
            # Settlement-station gate: never OPEN/ADD real-money risk on a station whose
            # (series -> NWS station) mapping hasn't been validated against resolved
            # markets. A wrong station yields a confident-but-biased p that Kelly
            # punishes hard (CLAUDE.md). Demo/dry-run still trade it to gather data.
            # A SELL (reducing/closing risk) is NEVER blocked — we must always be able
            # to exit a position, even one on a since-invalidated station.
            if (decision.is_trade and decision.action is Action.BUY
                    and self._blocks_unvalidated(decision.ticker)):
                decision = Decision(
                    ticker=decision.ticker, action=Action.HOLD,
                    prob=decision.prob, sigma=decision.sigma,
                    reason=f"station not validated for real-money trading "
                           f"({decision.ticker.split('-', 1)[0]})",
                )
                decision.meta["strategy"] = strat
            # Anti-stack: if we already have a working order on this ticker/side,
            # don't place another (a resting maker is not yet a position, so the
            # engine alone wouldn't catch it). Reconciliation cancels stale rests.
            if (decision.is_trade and decision.action is Action.BUY
                    and self.state.has_open_order(decision.ticker, decision.side.value)):
                decision = Decision(
                    ticker=decision.ticker, action=Action.HOLD,
                    prob=decision.prob, sigma=decision.sigma,
                    reason="working order already open on this market",
                )
                decision.meta["strategy"] = strat
            if decision.is_trade and decision.action is Action.BUY:
                # Reserve capital so the portfolio and per-event caps hold across this cycle.
                reserved = decision.count * decision.price
                portfolio_at_risk += reserved
                event_at_risk[ekey] = event_at_risk.get(ekey, 0.0) + reserved
            # Stamp the market mid at decision time so the skill gate can later score
            # the acted-on probability against what the market thought (#3, step 4).
            decision.meta["mid"] = round(quote.mid, 4)
            ticket = self.executor.place(
                decision, idem_key=self._idem_key(decision),
            )
            self._record_ticket(ticket)
            # Start the anti-churn cooldown when we actually act on a held position.
            if (ticket.placed and decision.is_trade and self.cfg.manage_positions
                    and view.ticker in positions):
                self._mark_manage(view.ticker)
            tickets.append(ticket)
            decisions_log.append(self._log_row(decision, strat, ticket))
            # Canonical, queryable record of this decision in the DB (the learning
            # substrate). Best-effort: a logging failure must never abort a cycle.
            try:
                self.state.record_decision(self._db_decision_row(
                    decision, strat, ticket, quote, view, bankroll, par_at_decision))
            except Exception as e:  # noqa: BLE001
                print(f"[runner] state.record_decision failed ({e})", flush=True)

        self._write_log(decisions_log)
        self._write_status(bankroll, positions)
        self._print_summary(tickets, bankroll)
        return tickets

    # ----- reconciliation, settlement, status (durable-state machinery) ------
    def _record_ticket(self, ticket: OrderTicket) -> None:
        """Persist a placed order so fills/idempotency survive a restart."""
        d = ticket.decision
        if not (ticket.placed and d.is_trade and d.side is not None):
            return
        coid = ticket.body.get("client_order_id")
        if not coid:
            return
        try:
            self.state.record_order(
                coid, d.ticker, d.side.value, d.action.value, d.price_cents, d.count,
                order_id=ticket.order_id, status=ticket.status or "placed",
                fill_count=ticket.fill_count,
            )
        except Exception as e:  # noqa: BLE001 — state must never crash a cycle
            print(f"[runner] state.record_order failed ({e})", flush=True)

    def _reconcile_orders(self) -> None:
        """Update our recorded orders from broker truth; cancel-replace stale rests.

        Reads the broker order LIST (``get_orders``) as truth — the single-order GET
        is eventually-consistent on Kalshi and 404s on a just-placed order, so we map
        the listing by order_id instead. Filled orders become positions (the engine
        reconciles against fresh ``get_positions``); resting makers are cancelled so
        the next decision re-prices against current quotes rather than deadlocking the
        ticker; orders no longer in the listing are terminal — mark them closed so the
        anti-stack guard releases the market.
        """
        open_rows = self.state.open_orders()
        if not open_rows:
            return
        try:
            listing = self.client.get_orders().get("orders", []) or []
        except Exception:  # noqa: BLE001 — broker hiccup; retry next cycle
            return
        by_id = {o.get("order_id"): o for o in listing}
        for row in open_rows:
            oid, coid = row["order_id"], row["client_order_id"]
            if not oid:
                continue
            o = by_id.get(oid)
            if o is None:
                self.state.update_order(coid, status="closed")  # terminal / aged out
                continue
            _, status, fill = parse_order({"order": o})
            if status == "resting":
                try:
                    self.client.cancel_order(oid)   # cancel-replace stale maker
                except Exception:  # noqa: BLE001
                    pass
                self.state.update_order(coid, status="canceled", fill_count=fill)
            elif status in ("executed", "canceled"):
                self.state.update_order(coid, status=status, fill_count=fill)

    @staticmethod
    def _fill_count(f: dict) -> float:
        """Filled contracts on one fill. Live Kalshi uses ``count_fp`` (a fixed-point
        STRING, e.g. ``"542.00"`` — and genuinely fractional, e.g. ``"382.59"``, from
        pro-rata maker matching); older payloads used integer ``count``. Read the
        ``_fp`` field first so a renamed key can't silently zero every fill (which is
        exactly what left the fills table empty: ``count`` does not exist on the live
        payload, so every fill was skipped and no P&L was ever booked)."""
        cf = f.get("count_fp")
        if cf not in (None, ""):
            return float(cf)
        return float(f.get("count", 0) or 0)

    @staticmethod
    def _fill_price_cents(f: dict, side: str) -> float | None:
        """Per-contract fill price for ``side`` in CENTS. Live Kalshi reports
        ``{yes,no}_price_dollars`` (STRING dollars, e.g. ``"0.9800"``); older payloads
        used integer-cent ``{yes,no}_price``. Read dollars first, else legacy cents."""
        kd = "yes_price_dollars" if side == "yes" else "no_price_dollars"
        kc = "yes_price" if side == "yes" else "no_price"
        if f.get(kd) not in (None, ""):
            return float(f[kd]) * 100.0
        if f.get(kc) is not None:
            return float(f[kc])
        return None

    def _reconcile_fills(self) -> None:
        """Pull actual broker fills and persist them as the source of truth for P&L.

        Reads ``GET /portfolio/fills``, aggregates by broker ``order_id``, matches each
        back to one of our recorded orders, and upserts a ``fills`` row (filled count,
        avg entry price in cents, fee in cents). Captures both immediate IOC-taker
        fills (gone from the order listing before the next cycle) and makers that fill
        later. Uses the broker's actual ``fee_cost`` (dollars) when present — most of
        our fills are post-only makers and settle fee-free — falling back to the taker
        estimate only when the broker omits it. Best-effort: a hiccup retries next cycle.

        NOTE: the fills ``action``/``book_side`` are YES-centric (buying NO is reported
        as ``action="sell"``/``book_side="ask"``), so they do NOT by themselves mark an
        open vs a close; ``side`` (yes/no) + price is what settlement P&L is computed on.
        """
        from hedge.tournament.paper import taker_fee
        try:
            fills = (self.client.get_fills(limit=200) or {}).get("fills", []) or []
        except Exception:  # noqa: BLE001 — endpoint missing/hiccup; retry next cycle
            return
        agg: dict[str, dict] = {}
        for f in fills:
            oid = f.get("order_id")
            side = (f.get("side") or "").lower()
            if not oid or side not in ("yes", "no"):
                continue
            n = self._fill_count(f)
            px = self._fill_price_cents(f, side)
            if n <= 0 or px is None:
                continue
            a = agg.setdefault(oid, {"count": 0.0, "notional": 0.0, "fee": 0.0,
                                     "side": side, "action": (f.get("action") or "buy").lower()})
            a["count"] += n
            a["notional"] += px * n
            fc = f.get("fee_cost")
            if fc not in (None, ""):
                a["fee"] += float(fc) * 100.0                       # broker truth ($ -> cents)
            elif bool(f.get("is_taker", True)):
                a["fee"] += taker_fee(px / 100.0) * 100.0 * n       # estimate only if absent
        for oid, a in agg.items():
            row = self.state.order_for_oid(oid)
            if row is None:
                continue  # a fill we have no record of placing — leave it to manual review
            count = round(a["count"])
            if count <= 0:
                continue
            avg = a["notional"] / a["count"] if a["count"] else None
            try:
                # Use OUR order's side/action (buy=open, sell=close) — the broker's
                # fill action/side are YES-centric and can't tell an open from a close.
                # This is what lets position_accounting realize close P&L correctly.
                self.state.record_fill(
                    row["client_order_id"], row["ticker"], row["side"], row["action"],
                    count, order_id=oid, avg_price_cents=avg,
                    fee_cents=a["fee"], status="executed")
            except Exception as e:  # noqa: BLE001
                print(f"[runner] record_fill failed for {oid} ({e})", flush=True)

    def _book_intraday_realized(self) -> None:
        """Book realized P&L from intraday CLOSES so the daily-loss stop sees churn.

        Each cycle, for every ticker we hold fills on, book the increment in realized-
        from-closes P&L (sells that reduced a position before settlement). Without this,
        a flip-to-exit loss was invisible to ``realized_today()`` until settlement and
        the daily-loss stop could be drained by churn it never saw. Best-effort.
        """
        for ticker in self.state.unsettled_fill_tickers():
            try:
                self.state.book_intraday_realized(ticker)
            except Exception as e:  # noqa: BLE001
                print(f"[runner] book_intraday_realized failed for {ticker} ({e})", flush=True)

    def _book_settlements(self) -> None:
        """Book settlement P&L on the NET held position for settled markets.

        Uses average-cost accounting (``state.position_accounting``): the realized P&L
        from intraday closes is booked separately by :meth:`_book_intraday_realized`, so
        here we book ONLY the net-held-to-settlement quantity per side — won/lost by the
        outcome at the average entry cost, minus the open-side fees. Booking gross fills
        (the prior behavior) would double-count any position that was sold intraday.
        Booked once per ticker via the ``settled`` table.
        """
        tickers = self.state.unsettled_fill_tickers()
        if not tickers:
            return
        outcomes = self._settlement_outcomes(set(tickers))
        for ticker in tickers:
            if ticker not in outcomes:
                continue
            won_yes = outcomes[ticker]
            acct = self.state.position_accounting(ticker)
            pnl = -acct["buy_fees"]                      # open-side fees charged here
            net_total = 0
            entry_c = 0.0
            held_side = None
            for side, net in acct["net"].items():
                if net <= 0:
                    continue
                avg = acct["avg_cost_cents"][side] / 100.0
                won = won_yes if side == "yes" else (not won_yes)
                pnl += net * (1.0 - avg) if won else -net * avg
                net_total += net
                entry_c += acct["avg_cost_cents"][side] * net
                held_side = side if held_side is None else "mixed"
            try:
                self.state.book_settlement(
                    ticker, pnl, side=held_side,
                    entry_cents=(entry_c / net_total) if net_total else None,
                    count=net_total, outcome="yes" if won_yes else "no")
            except Exception as e:  # noqa: BLE001
                print(f"[runner] book_settlement failed for {ticker} ({e})", flush=True)

    def _heartbeat(self, bankroll: float, positions: dict[str, Position]) -> None:
        """Once per UTC day, push an 'alive' alert with bankroll + open positions."""
        today = datetime.now(ZoneInfo("UTC")).strftime("%Y-%m-%d")
        if self.state.get_meta("last_heartbeat_date") == today:
            return
        self.state.set_meta("last_heartbeat_date", today)
        mode = "DRY-RUN" if self.executor.dry_run else f"LIVE[{self.executor.env}]"
        alerts.notify(
            alerts.Level.INFO, "hedge: daily heartbeat",
            f"{mode} alive — bankroll ${bankroll:,.2f}, {len(positions)} open position(s), "
            f"realized today ${self.state.realized_today():.2f}",
        )

    def _write_status(self, bankroll: float, positions: dict[str, Position]) -> None:
        """Write a status snapshot for `fly ssh console` / monitoring inspection."""
        halted, reason = self.is_halted()
        status = {
            "ts": datetime.now(ZoneInfo("UTC")).isoformat(),
            "mode": "dry-run" if self.executor.dry_run else f"live-{self.executor.env}",
            "bankroll": round(bankroll, 2),
            "open_positions": len(positions),
            "realized_today": round(self.state.realized_today(), 2),
            "halted": halted,
            "halt_reason": reason or None,
            "skill_mult": round(self._skill_mult, 3),
        }
        try:
            base = Path(os.environ.get("HEDGE_STATE_DIR", str(LIVE_DIR)))
            base.mkdir(parents=True, exist_ok=True)
            (base / "status.json").write_text(json.dumps(status, indent=2))
        except Exception as e:  # noqa: BLE001 — status is best-effort
            print(f"[runner] status write failed ({e})", flush=True)

    def _blocks_unvalidated(self, ticker: str) -> bool:
        """True if a real-money order on this ticker must be blocked for lack of
        settlement-station validation. Only real (non-dry-run) PROD orders are
        gated; demo and dry-run still trade unvalidated stations so calibration and
        execution evidence can accrue (CLAUDE.md: validate before real size)."""
        if self.executor.dry_run or self.executor.env != "prod":
            return False
        st = station_for_ticker(ticker)
        return st is None or not st.validated

    def _best_decision(self, view, quote, bankroll, positions, portfolio_at_risk,
                       event_at_risk=0.0):
        """Decide across all strategies for one market; return (decision, strategy).

        Picks the largest-edge tradable decision. If none trade, returns the first
        HOLD so it's still logged (with its reason)."""
        # Only hand the engine the existing position when intraday management is
        # enabled; otherwise treat each market as a fresh open (no trim/flip/exit).
        position = positions.get(view.ticker) if self.cfg.manage_positions else None
        best_trade: tuple[Decision, str] | None = None
        first_hold: tuple[Decision, str] | None = None
        for strat in self.strategies:
            try:
                sig = strat.evaluate(view)
            except Exception as e:  # noqa: BLE001
                print(f"[runner] {strat.name} failed on {view.ticker} ({e})", flush=True)
                continue
            if sig is None:
                continue
            # Effective λ for THIS strategy: base × per-strategy multiplier × skill gate.
            scale = self.strategy_lambda.get(sig.strategy, 1.0) * self._skill_mult
            scfg = (self.cfg if scale == 1.0
                    else replace(self.cfg, lambda_kelly=self.cfg.lambda_kelly * scale))
            d = decide(sig, quote, bankroll, scfg,
                       position=position, portfolio_at_risk=portfolio_at_risk,
                       event_at_risk=event_at_risk)
            d.meta["strategy"] = sig.strategy
            # Carry the model's own inputs (forecast, dispersion, bias, max-so-far)
            # so a decision row records *why* the prob was what it was.
            if sig.meta:
                d.meta["signal"] = dict(sig.meta)
            if d.is_trade:
                if best_trade is None or d.edge > best_trade[0].edge:
                    best_trade = (d, sig.strategy)
            elif first_hold is None:
                first_hold = (d, sig.strategy)
        return best_trade or first_hold

    # ----- helpers -----------------------------------------------------------
    def _idem_key(self, decision: Decision) -> str:
        """Per-CYCLE idempotency: the client_order_id is stable within one cycle (so a
        network retry of the same create dedups) but fresh across cycles.

        Keying only on the day was too coarse: Kalshi burns a client_order_id forever
        (even after cancel), so a cancel-replaced or re-priced order on the same market
        would 409 ``order_already_exists`` for the rest of the day, deadlocking it.
        Folding in the monotonic cycle sequence avoids that; genuine duplicates are
        prevented by the anti-stack guard + position reconciliation, not the key."""
        day = datetime.now(ZoneInfo("UTC")).strftime("%Y-%m-%d")
        return f"{day}:{getattr(self, '_cycle_seq', 0)}"

    def _db_decision_row(self, decision: Decision, strat: str, ticket: OrderTicket,
                         quote, view, bankroll: float, portfolio_at_risk: float) -> dict:
        """Build a full ``decisions`` row: belief + market snapshot + intended order."""
        now = datetime.now(ZoneInfo("UTC"))
        body = ticket.body or {}
        return {
            "cycle_seq": getattr(self, "_cycle_seq", 0),
            "ts": now.isoformat(),
            "utc_date": now.strftime("%Y-%m-%d"),
            "ticker": decision.ticker,
            "strategy": strat,
            "action": decision.action.value,
            "side": decision.side.value if decision.side else None,
            "count": decision.count,
            "price_cents": decision.price_cents,
            "prob": decision.prob,
            "sigma": decision.sigma,
            "edge": decision.edge,
            "kelly_fraction": decision.kelly_fraction,
            "yes_bid": quote.yes_bid,
            "yes_ask": quote.yes_ask,
            "mid": quote.mid,
            "last": getattr(view, "last_price", None),
            "bankroll": round(bankroll, 4),
            "portfolio_at_risk": round(portfolio_at_risk, 4),
            "placed": int(bool(ticket.placed)),
            "dry_run": int(bool(ticket.dry_run)),
            "error": ticket.error,
            "reason": decision.reason,
            "client_order_id": body.get("client_order_id"),
            "meta_json": json.dumps(dict(decision.meta), default=str),
        }

    @staticmethod
    def _log_row(decision: Decision, strat: str, ticket: OrderTicket) -> dict:
        row = asdict(decision)
        row["action"] = decision.action.value
        row["side"] = decision.side.value if decision.side else None
        row["strategy"] = strat
        row["placed"] = ticket.placed
        row["dry_run"] = ticket.dry_run
        row["error"] = ticket.error
        row["ts"] = datetime.now(ZoneInfo("UTC")).isoformat()
        return row

    @staticmethod
    def _write_log(rows: list[dict]) -> None:
        if not rows:
            return
        LIVE_DIR.mkdir(parents=True, exist_ok=True)
        day = rows[0]["ts"][:10]
        with (LIVE_DIR / f"decisions_{day}.jsonl").open("a") as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")

    def _print_summary(self, tickets: list[OrderTicket], bankroll: float) -> None:
        trades = [t for t in tickets if t.decision.is_trade]
        placed = [t for t in trades if t.placed]
        mode = "DRY-RUN" if self.executor.dry_run else f"LIVE[{self.executor.env}]"
        now = datetime.now(ZoneInfo("UTC"))
        print(
            f"[runner {mode}] {now:%Y-%m-%d %H:%M}Z bankroll=${bankroll:,.2f} "
            f"markets_checked={len(tickets)} would_trade={len(trades)} placed={len(placed)}",
            flush=True,
        )
        for t in trades:
            d = t.decision
            tag = "PLACED" if t.placed else ("DRY" if t.dry_run else f"BLOCKED:{t.error}")
            print(
                f"    {tag} {d.action.value} {d.count} {d.side.value if d.side else '-'} "
                f"@ {d.price_cents}c {d.ticker} edge={d.edge:.3f} "
                f"f={d.kelly_fraction:.3f} [{d.meta.get('strategy', '?')}] {d.reason}",
                flush=True,
            )


# --------------------------------------------------------------------------- #
# CLI                                                                           #
# --------------------------------------------------------------------------- #
def _section_from_config(key: str) -> dict:
    try:
        import yaml
        cfg_path = Path("config.yaml")
        if cfg_path.exists():
            data = yaml.safe_load(cfg_path.read_text()) or {}
            sect = data.get(key)
            if isinstance(sect, dict):
                return sect
    except Exception as e:  # noqa: BLE001
        print(f"[runner] could not read '{key}' config ({e}); using defaults", flush=True)
    return {}


def _risk_from_config() -> RiskConfig:
    return RiskConfig.from_dict(_section_from_config("risk"))


def _guard_from_config() -> GuardConfig:
    return GuardConfig.from_dict(_section_from_config("guard"))


def _loop(runner: Runner, interval: float, until: str | None, cycles: int | None) -> None:
    stop_at = None
    if until is not None:
        hh, mm = (int(x) for x in until.split(":"))
        now = datetime.now().astimezone()
        stop_at = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if stop_at <= now:
            sys.exit(f"--until {until} already past (local now {now:%H:%M}).")

    stopping = {"flag": False}

    def _handle(signum, _frame):
        stopping["flag"] = True
        print(f"\n[runner] signal {signum}; stopping after this cycle.", flush=True)

    _signal.signal(_signal.SIGINT, _handle)
    _signal.signal(_signal.SIGTERM, _handle)

    n = 0
    while not stopping["flag"]:
        try:
            runner.run_cycle()
        except Exception as e:  # noqa: BLE001
            print(f"[runner] cycle error ({type(e).__name__}): {e}", flush=True)
            alerts.notify(alerts.Level.WARN, "hedge: cycle error",
                          f"{type(e).__name__}: {e}")
        n += 1
        if cycles is not None and n >= cycles:
            break
        if stop_at is not None and datetime.now().astimezone() >= stop_at:
            break
        slept = 0.0
        while slept < interval and not stopping["flag"]:
            if stop_at is not None and datetime.now().astimezone() >= stop_at:
                break
            time.sleep(min(1.0, interval - slept))
            slept += 1.0
    print(f"[runner] stopped after {n} cycle(s).", flush=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="hedge live/dry-run trading loop")
    ap.add_argument("--live", action="store_true",
                    help="actually place orders (default: dry-run, report only)")
    ap.add_argument("--allow-prod", action="store_true",
                    help="permit live orders against the PROD environment (dangerous)")
    ap.add_argument("--once", action="store_true", help="run a single cycle and exit")
    ap.add_argument("--interval", type=float, default=900.0,
                    help="seconds between cycles when looping (default 900)")
    ap.add_argument("--until", type=str, default=None, help="stop at local HH:MM")
    ap.add_argument("--cycles", type=int, default=None, help="stop after N cycles")
    ap.add_argument("--bankroll", type=float, default=None,
                    help="override bankroll in dollars (else read from Kalshi balance)")
    ap.add_argument("--reset-guard", action="store_true",
                    help="clear a tripped calibration kill-switch and exit")
    ap.add_argument("--no-guard", action="store_true",
                    help="disable the calibration kill-switch (not recommended)")
    args = ap.parse_args(argv)

    try:
        client, env, base = build_client()
    except RuntimeError as e:
        sys.exit(f"{e}\n(configure config.yaml or KALSHI_* env vars; see scripts/test_auth.py)")

    guard_cfg = _guard_from_config()
    if args.no_guard:
        guard_cfg = GuardConfig(enabled=False)
    executor = Executor(client, env=env, dry_run=not args.live, allow_prod=args.allow_prod)
    strategy_lambda = {str(k): float(v)
                       for k, v in _section_from_config("strategy_lambda").items()}
    runner = Runner(client, _default_strategies(), executor, _risk_from_config(),
                    bankroll_override=args.bankroll, guard_cfg=guard_cfg,
                    strategy_lambda=strategy_lambda)

    if args.reset_guard:
        cleared = runner.reset_guard()
        print("[runner] kill-switch latch cleared." if cleared
              else "[runner] no kill-switch latch was set.", flush=True)
        return

    print(f"[runner] env={env} base={base} mode={'LIVE' if args.live else 'DRY-RUN'} "
          f"guard={'on' if guard_cfg.enabled else 'OFF'}", flush=True)

    if args.live and env == "prod" and not args.allow_prod:
        sys.exit("refusing live PROD trading without --allow-prod. Test on demo first.")

    if args.live:
        alerts.notify(alerts.Level.INFO, "hedge: trading session started",
                      f"LIVE[{env}] λ={runner.cfg.lambda_kelly} max_order=${runner.cfg.max_order_dollars} "
                      f"daily_stop=${runner.cfg.daily_loss_stop_dollars} guard={'on' if guard_cfg.enabled else 'off'}")

    if args.once:
        runner.run_cycle()
    else:
        if args.interval <= 0:
            sys.exit("--interval must be > 0")
        _loop(runner, args.interval, args.until, args.cycles)


if __name__ == "__main__":
    main()
