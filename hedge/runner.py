"""The main trading loop: signals -> decide -> execute, repeated.

This is the live counterpart to the paper tournament. Each cycle it:

    1. reads bankroll + open positions from Kalshi,
    2. discovers the open temperature markets and builds a ``MarketView`` each,
    3. runs every strategy to collect ``Signal``s,
    4. routes each signal through the decision engine, picking the single
       best-edge decision per market (strategies share one order book, so we
       never stack conflicting orders on the same ticker),
    5. hands tradable decisions to the ``Executor``,
    6. logs every decision (acted-on or not) for later review.

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
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from hedge import alerts
from hedge.config import build_client
from hedge.decision import Action, Decision, MarketQuote, Position, RiskConfig, Side, decide
from hedge.execution import Executor, OrderTicket
from hedge.execution.executor import parse_order
from hedge.guard import GuardConfig, GuardStatus, assess
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


def _default_strategies() -> list[Strategy]:
    src = LiveForecastSource()
    calib = CalibrationTable()
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
                 state: State | None = None):
        self.client = client
        self.strategies = strategies
        self.executor = executor
        self.cfg = cfg
        self.bankroll_override = bankroll_override
        self.guard_cfg = guard_cfg or GuardConfig()
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
        """Map ticker -> current Position. Kalshi `position` is signed: positive =
        long YES, negative = long NO."""
        out: dict[str, Position] = {}
        try:
            data = self.client.get_positions()
        except Exception as e:  # noqa: BLE001
            print(f"[runner] positions fetch failed ({e}); assuming flat", flush=True)
            return out
        for mp in data.get("market_positions", []) or []:
            ticker = mp.get("ticker")
            net = int(mp.get("position", 0) or 0)
            if not ticker or net == 0:
                continue
            side = Side.YES if net > 0 else Side.NO
            # market_exposure is in cents and reflects cost basis when present.
            count = abs(net)
            exposure = float(mp.get("market_exposure", 0) or 0) / 100.0
            avg = exposure / count if count else 0.0
            out[ticker] = Position(side=side, count=count, avg_price=avg)
        return out

    def market_views(self) -> list[MarketView]:
        views: list[MarketView] = []
        for series in STATIONS:
            try:
                for raw in discover_temp_markets(self.client, series, status="open"):
                    views.append(MarketView(raw.get("ticker", ""), raw))
            except Exception as e:  # noqa: BLE001
                print(f"[runner] market discovery failed for {series} ({e})", flush=True)
        return views

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
        """Read acted-on (non-hold) decisions from the last window_days logs."""
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
                if r.get("action") in ("buy", "sell") and r.get("prob") is not None:
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
        # stale resting makers), pull actual fill economics, and book any
        # newly-settled realized P&L from those fills.
        self._reconcile_orders()
        self._reconcile_fills()
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

        bankroll = self.bankroll()
        positions = self.positions()
        views = self.market_views()
        self._heartbeat(bankroll, positions)

        # Capital already at risk in open positions seeds the portfolio cap.
        portfolio_at_risk = sum(p.count * p.avg_price for p in positions.values())

        tickets: list[OrderTicket] = []
        decisions_log: list[dict] = []
        for view in views:
            quote = MarketQuote.from_view(view)
            if quote is None:
                continue
            best = self._best_decision(view, quote, bankroll, positions, portfolio_at_risk)
            if best is None:
                continue
            decision, strat = best
            # Risk context as it stood when this market was decided (before we reserve
            # capital for its own order below) — logged for later reconstruction.
            par_at_decision = portfolio_at_risk
            # Settlement-station gate: never open real-money risk on a station whose
            # (series -> NWS station) mapping hasn't been validated against resolved
            # markets. A wrong station yields a confident-but-biased p that Kelly
            # punishes hard (CLAUDE.md). Demo/dry-run still trade it to gather data.
            if decision.is_trade and self._blocks_unvalidated(decision.ticker):
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
                # Reserve capital so the portfolio cap holds across this cycle.
                portfolio_at_risk += decision.count * decision.price
            ticket = self.executor.place(
                decision, idem_key=self._idem_key(decision),
            )
            self._record_ticket(ticket)
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

    def _reconcile_fills(self) -> None:
        """Pull actual broker fills and persist them as the source of truth for P&L.

        Reads ``GET /portfolio/fills`` (authoritative per-contract entry prices in
        cents), aggregates by broker ``order_id``, matches each back to one of our
        recorded orders, and upserts a ``fills`` row (filled count, avg entry price,
        taker fee). This captures both immediate IOC-taker fills (gone from the order
        listing before the next cycle) and makers that fill later. Best-effort: a
        broker hiccup just retries next cycle.
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
            n = int(float(f.get("count", 0) or 0))
            px = f.get("yes_price") if side == "yes" else f.get("no_price")
            if n <= 0 or px is None:
                continue
            px = float(px)  # cents
            a = agg.setdefault(oid, {"count": 0, "notional": 0.0, "fee": 0.0,
                                     "side": side, "action": (f.get("action") or "buy").lower()})
            a["count"] += n
            a["notional"] += px * n
            if bool(f.get("is_taker", True)):
                a["fee"] += taker_fee(px / 100.0) * 100.0 * n  # per-contract $ -> total cents
        for oid, a in agg.items():
            row = self.state.order_for_oid(oid)
            if row is None:
                continue  # a fill we have no record of placing — leave it to manual review
            avg = a["notional"] / a["count"] if a["count"] else None
            try:
                self.state.record_fill(
                    row["client_order_id"], row["ticker"], a["side"], a["action"],
                    a["count"], order_id=oid, avg_price_cents=avg,
                    fee_cents=a["fee"], status="executed")
            except Exception as e:  # noqa: BLE001
                print(f"[runner] record_fill failed for {oid} ({e})", flush=True)

    def _book_settlements(self) -> None:
        """Book realized P&L from ACTUAL FILLS (not intent) for settled markets.

        For each ticker we have real fills on that has since resolved, P&L is the sum
        over fills of the filled count at the actual entry price, won/lost by the
        settlement outcome, net of the recorded taker fee. Booked once per ticker.
        This is the difference between "we think we made $X" (intended order) and "we
        made $X" (what actually filled) — the whole point of fill-based accounting.
        """
        from hedge.tournament.paper import taker_fee
        tickers = self.state.unsettled_fill_tickers()
        if not tickers:
            return
        outcomes = self._settlement_outcomes(set(tickers))
        for ticker in tickers:
            if ticker not in outcomes:
                continue
            won_yes = outcomes[ticker]
            rows = self.state.fills_for_ticker(ticker)
            total_pnl = 0.0
            total_count = 0
            notional_c = 0.0
            sides: set[str] = set()
            for r in rows:
                n = int(r["fill_count"] or 0)
                if n <= 0 or r["avg_price_cents"] is None:
                    continue
                price = float(r["avg_price_cents"]) / 100.0
                if not (0.0 < price < 1.0):
                    continue
                side = r["side"]
                sides.add(side)
                won = won_yes if side == "yes" else (not won_yes)
                gross = n * (1.0 - price) if won else -n * price
                fee = (float(r["fee_cents"]) / 100.0) if r["fee_cents"] is not None \
                    else taker_fee(price) * n
                total_pnl += gross - fee
                total_count += n
                notional_c += float(r["avg_price_cents"]) * n
            if total_count <= 0:
                continue
            try:
                self.state.book_settlement(
                    ticker, total_pnl,
                    side=(next(iter(sides)) if len(sides) == 1 else "mixed"),
                    entry_cents=notional_c / total_count, count=total_count,
                    outcome="yes" if won_yes else "no")
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

    def _best_decision(self, view, quote, bankroll, positions, portfolio_at_risk):
        """Decide across all strategies for one market; return (decision, strategy).

        Picks the largest-edge tradable decision. If none trade, returns the first
        HOLD so it's still logged (with its reason)."""
        position = positions.get(view.ticker)
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
            d = decide(sig, quote, bankroll, self.cfg,
                       position=position, portfolio_at_risk=portfolio_at_risk)
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
    runner = Runner(client, _default_strategies(), executor, _risk_from_config(),
                    bankroll_override=args.bankroll, guard_cfg=guard_cfg)

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
