"""
The live trading loop — paper and live.

Paper and live differ in exactly one line: which Broker is constructed. Every
decision below runs identically in both, which is what makes paper results worth
anything (KNOWLEDGE.md section 2).

Loop, once per candle:
    1. square-off check      — flatten before the broker does it for you
    2. manage open positions — stops, targets, trailing, time stops
    3. scan for entries      — only if the session and risk gates allow
    4. sleep to the next candle boundary

Ordering is deliberate: exits are processed before entries every time. Capital
freed by an exit is available to the same cycle's entries, and more importantly
a position that needs to be closed is never left waiting behind a scan.
"""

from __future__ import annotations

import json
import signal as signal_module
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from .broker import INTERVAL_SECONDS, BrokerError, KiteBroker, PaperBroker
from .data import DataManager
from .execution import ExecutionEngine, PositionBook
from .indicators import is_warm
from .logs import TradingLogger
from .risk import RiskManager, RiskState, detect_market_regime
from .session import (
    DEFAULT_ENTRY_START,
    DEFAULT_LAST_ENTRY,
    DEFAULT_SQUAREOFF,
    TradingCalendar,
    check_clock_drift,
    is_market_open,
    now_ist,
    should_squareoff,
    should_trade_now,
    to_ist,
)
from .signals import STRATEGIES, add_strategy_columns
from .universe import load_universe


def _parse_time(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        hour, _, minute = value.partition(":")
        from datetime import time as time_type

        return time_type(int(hour), int(minute or 0))
    return value


class PreflightError(RuntimeError):
    """A startup check failed. Never bypass these — each one guards a live-money failure."""


class TradingRunner:
    """
    Owns the loop and the shutdown path.

    Args:
        config: merged strategy + risk configuration
        mode: 'paper' (simulated fills, real prices) or 'live' (real orders)
        universe: symbols to trade; falls back to config['universe_file']
        state_dir: where positions and risk state persist across restarts
    """

    def __init__(
        self,
        config: dict,
        *,
        mode: str = "paper",
        universe: list[str] | None = None,
        state_dir: str | Path = "state",
        log_dir: str | Path = "logs",
    ) -> None:
        if mode not in ("paper", "live"):
            raise ValueError(f"mode must be 'paper' or 'live', got {mode!r}")

        self.config = config
        self.mode = mode
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.log = TradingLogger(log_dir, debug_to_console=config.get("verbose", False))
        self.interval = config.get("interval", "minute")
        self.strategy_name = config.get("strategy", "fortress")
        self.evaluate = STRATEGIES[self.strategy_name]

        self.calendar = TradingCalendar.from_file(
            config.get("holidays_file", "config/holidays.json")
        )
        self.entry_start = _parse_time(config.get("entry_start_time"), DEFAULT_ENTRY_START)
        self.last_entry = _parse_time(config.get("last_entry_time"), DEFAULT_LAST_ENTRY)
        self.squareoff = _parse_time(config.get("squareoff_time"), DEFAULT_SQUAREOFF)

        # --- the only line that differs between paper and live
        market_data = KiteBroker(cache_dir=config.get("cache_dir", ".cache"))
        self.broker: Any = (
            market_data
            if mode == "live"
            else PaperBroker(
                market_data,
                starting_capital=config.get("capital", 1_000_000.0),
                slippage_bps=config.get("slippage_bps", 5.0),
            )
        )

        self.data = DataManager(
            market_data,
            cache_dir=config.get("cache_dir", ".cache") + "/ohlcv",
            interval=self.interval,
            calendar=self.calendar,
        )

        self.book = PositionBook(self.state_dir / "positions.json")
        self.risk = RiskManager(config, self._load_risk_state())
        self.execution = ExecutionEngine(self.broker, self.book, self.log, config)

        self.universe = universe or self._load_universe()
        self.sector_map = config.get("sector_map", {})
        self.regime_multiplier = 1.0

        self._running = False
        self._shutdown_requested = False
        signal_module.signal(signal_module.SIGINT, self._handle_shutdown)
        signal_module.signal(signal_module.SIGTERM, self._handle_shutdown)

    # ------------------------------------------------------------------ state

    def _load_universe(self) -> list[str]:
        universe_file = self.config.get("universe_file")
        if not universe_file:
            raise PreflightError("no universe: pass universe= or set config['universe_file']")
        return load_universe(universe_file, max_age_days=self.config.get("universe_max_age_days", 30))

    def _risk_state_path(self) -> Path:
        return self.state_dir / "risk_state.json"

    def _load_risk_state(self) -> RiskState:
        path = self._risk_state_path()
        if not path.exists():
            return RiskState()
        try:
            return RiskState.from_dict(json.loads(path.read_text()))
        except (json.JSONDecodeError, KeyError, ValueError):
            return RiskState()

    def _save_risk_state(self) -> None:
        """
        Persist risk counters.

        A restart must not reset the daily loss limit or the consecutive-loss
        streak — otherwise "restart the bot" becomes an accidental way to
        override the protection that just stopped you.
        """
        self._risk_state_path().write_text(json.dumps(self.risk.state.to_dict(), indent=2))

    # -------------------------------------------------------------- preflight

    def preflight(self) -> dict[str, Any]:
        """
        Startup checks. Any failure raises rather than warns.

        Every check here corresponds to a documented production failure. The bot
        refusing to start is always cheaper than the failure it prevents.
        """
        report: dict[str, Any] = {"mode": self.mode, "checks": {}}

        # 1. broker reachable and authenticated
        try:
            margin = self.broker.available_margin()
            report["checks"]["broker"] = "ok"
            report["available_margin"] = margin
        except BrokerError as exc:
            raise PreflightError(f"broker unreachable: {exc}") from exc

        # 2. NUANCE #16 — clock drift breaks candle alignment silently
        if hasattr(self.broker, "server_time"):
            within, drift = check_clock_drift(self.broker.server_time(), now_ist())
            report["checks"]["clock"] = f"drift {drift:+.2f}s"
            if not within and self.mode == "live":
                raise PreflightError(
                    f"clock drift {drift:+.2f}s exceeds tolerance — sync system time (NTP) first"
                )

        # 3. universe sane and tradeable
        if not self.universe:
            raise PreflightError("empty universe")
        report["checks"]["universe"] = f"{len(self.universe)} symbols"

        instruments = getattr(self.data.broker, "instruments", None)
        if instruments is not None:
            unknown = [s for s in self.universe if s not in instruments]
            if unknown:
                self.log.warning("symbols not in instrument dump — dropping",
                                 reason=", ".join(unknown[:10]))
                self.universe = [s for s in self.universe if s not in unknown]
                report["checks"]["dropped_symbols"] = unknown

        # 4. capital sufficient for the configured risk
        capital = self.config.get("capital", margin)
        if self.mode == "live" and margin < capital * 0.5:
            raise PreflightError(
                f"available margin {margin:,.0f} is far below configured capital "
                f"{capital:,.0f} — fix the config or fund the account"
            )

        # 5. NUANCE #2 — reconcile before doing anything else
        reconciliation = self.execution.reconcile(set(self.universe))
        report["reconciliation"] = reconciliation
        report["checks"]["reconciliation"] = "ok"

        # 6. market regime, for position size scaling
        if self.config.get("use_regime_sizing", True):
            report["regime"] = self._detect_regime()

        report["checks"]["risk_state"] = (
            f"halted: {self.risk.state.halt_reason}" if self.risk.state.halted else "ok"
        )
        return report

    def _detect_regime(self) -> dict[str, Any]:
        """Scale size by market regime. Failure is non-fatal — it falls back to 1.0x."""
        index_symbol = self.config.get("regime_index", "NIFTY 50")
        try:
            frame = self.data.broker.historical(
                index_symbol, "day", now_ist() - timedelta(days=400), now_ist()
            )
            regime = detect_market_regime(frame, self.config)
            self.regime_multiplier = regime["multiplier"]
            self.log.info("market regime", reason=f"{regime['regime']} — {regime['reason']}")
            return regime
        except Exception as exc:  # noqa: BLE001
            self.log.warning("regime detection failed, sizing at 1.0x", reason=str(exc))
            self.regime_multiplier = 1.0
            return {"regime": "UNKNOWN", "multiplier": 1.0, "reason": str(exc)}

    # ------------------------------------------------------------------- loop

    def run(self) -> None:
        """Run until square-off, shutdown signal, or market close."""
        report = self.preflight()
        self.log.info(
            f"starting in {self.mode.upper()} mode",
            reason=f"{len(self.universe)} symbols, strategy={self.strategy_name}",
        )
        for key, value in report.get("checks", {}).items():
            self.log.info(f"preflight {key}: {value}")

        if self.mode == "live":
            self.log.warning("LIVE MODE — orders will be placed with real money")

        self._running = True
        interval_seconds = INTERVAL_SECONDS[self.interval]

        try:
            while self._running and not self._shutdown_requested:
                cycle_start = now_ist()

                if not is_market_open(cycle_start, self.calendar):
                    if self.book.all():
                        self.log.warning("market closed with open positions")
                    self.log.info("market closed, stopping")
                    break

                if should_squareoff(cycle_start, self.squareoff) and self.book.all():
                    self.log.info("square-off time, flattening all positions")
                    self.execution.close_all("square off")
                    self._record_exits()
                    break

                try:
                    self._cycle(cycle_start)
                except BrokerError as exc:
                    # Transient broker errors must not kill the loop — open
                    # positions still need managing on the next pass.
                    self.log.error("broker error during cycle", exc_info=True,
                                   reason=str(exc))
                except Exception:  # noqa: BLE001
                    self.log.critical("unexpected error during cycle", exc_info=True)

                self._sleep_to_next_candle(interval_seconds)

        finally:
            self.shutdown()

    def _cycle(self, now: datetime) -> None:
        """One pass: manage, then scan."""
        self.risk.roll_day_if_needed(now)

        if hasattr(self.broker, "poll"):
            self.broker.poll()  # paper broker fills resting orders

        self._manage_positions(now)

        allowed, reason = should_trade_now(
            now,
            calendar=self.calendar,
            entry_start=self.entry_start,
            last_entry=self.last_entry,
            skip_lunch=self.config.get("skip_lunch", True),
        )
        if not allowed:
            self.log.debug("entries closed", reason=reason)
            return

        self._scan_for_entries(now)
        self._save_risk_state()

    def _manage_positions(self, now: datetime) -> None:
        positions = self.book.all()
        if not positions:
            return

        try:
            prices = self.broker.ltp([p.symbol for p in positions])
        except BrokerError as exc:
            self.log.error("could not fetch prices for open positions", reason=str(exc))
            return

        for position in positions:
            last_price = prices.get(position.symbol)
            if last_price is None:
                self.log.warning("no price for open position", symbol=position.symbol)
                continue

            exit_reason = self.execution.manage(position, last_price, now)
            if exit_reason:
                self._record_exits()

    def _record_exits(self) -> None:
        """Feed closed trades into the risk manager (cooldown, streak, daily P&L)."""
        trades_path = self.log.trades_path
        if not trades_path.exists():
            return

        seen = getattr(self, "_seen_trades", 0)
        lines = trades_path.read_text().splitlines()

        for line in lines[seen:]:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            self.risk.record_exit(record["symbol"], record["pnl"])

        self._seen_trades = len(lines)
        self._save_risk_state()

    def _scan_for_entries(self, now: datetime) -> None:
        """Evaluate every symbol we are not already in."""
        held = {p.symbol for p in self.book.all()}
        capital = self.config.get("capital", self.broker.available_margin())

        try:
            margin = self.broker.available_margin()
        except BrokerError as exc:
            self.log.error("margin unavailable, skipping entries", reason=str(exc))
            return

        for symbol in self.universe:
            if symbol in held:
                continue
            if self.risk.state.halted:
                self.log.debug("scan stopped", reason=self.risk.state.halt_reason)
                return

            remaining = self.risk.cooldown_remaining(symbol, now)
            if remaining.total_seconds() > 0:
                continue  # cheap check before an expensive data fetch

            try:
                frame = self.data.live_frame(symbol, self.config, now)
            except Exception as exc:  # noqa: BLE001
                self.log.error("data fetch failed", symbol=symbol, reason=str(exc))
                continue

            if frame.height < 2:
                continue

            frame = add_strategy_columns(frame, self.config)

            # NUANCE #6: evaluate the last COMPLETE candle. The final row of a
            # live frame is the bar currently forming; using it is the phantom
            # signal bug.
            row = frame.row(-2, named=True)
            if not is_warm(row):
                continue

            candidate = self.evaluate(
                row, symbol, self.config, self.broker.tick_size(symbol)
            )
            if candidate is None:
                continue

            ok, reason = self.risk.can_enter(
                candidate,
                open_positions=self.book.all(),
                capital=capital,
                now=now,
                sector_map=self.sector_map,
            )
            if not ok:
                self.log.signal(candidate.to_dict(), taken=False, reason=reason)
                self.log.debug("signal rejected", symbol=symbol, reason=reason)
                continue

            sizing = self.risk.size_position(
                candidate,
                capital=capital,
                available_margin=margin,
                lot_size=1,
                regime_multiplier=self.regime_multiplier,
            )
            if not sizing.is_tradeable:
                self.log.signal(candidate.to_dict(), taken=False, reason=sizing.reason)
                continue

            position = self.execution.enter(candidate, sizing.quantity,
                                            risk_note=sizing.reason)
            if position:
                self.log.signal(candidate.to_dict(), taken=True)
                held.add(symbol)
                margin -= sizing.position_value

    def _sleep_to_next_candle(self, interval_seconds: int) -> None:
        """
        Wake just after the next candle seals.

        Sleeping a fixed interval drifts out of alignment with candle boundaries
        over a session; aligning to the boundary keeps every cycle looking at a
        freshly closed bar.
        """
        now = now_ist()
        elapsed = now.timestamp() % interval_seconds
        wait = interval_seconds - elapsed + 0.75  # clear the completion buffer

        deadline = time.monotonic() + wait
        while time.monotonic() < deadline and not self._shutdown_requested:
            time.sleep(min(0.5, deadline - time.monotonic()))

    # --------------------------------------------------------------- shutdown

    def _handle_shutdown(self, signum: int, frame: Any) -> None:
        if self._shutdown_requested:
            self.log.critical("second interrupt — exiting immediately, positions left open")
            raise SystemExit(1)

        self.log.warning(f"shutdown signal {signum} received, finishing current cycle")
        self._shutdown_requested = True
        self._running = False

    def shutdown(self) -> None:
        """
        Clean stop.

        Whether to flatten on shutdown is a real decision, not a default. For
        intraday the answer is yes: an MIS position left open gets squared off by
        the broker at whatever price it likes, and nobody is watching. For
        positional it is no. `close_on_shutdown` decides, and either way the
        outcome is logged loudly enough to notice.
        """
        if getattr(self, "_shutdown_done", False):
            return
        self._shutdown_done = True

        open_positions = self.book.all()

        if open_positions and self.config.get("close_on_shutdown", True):
            self.log.warning(f"closing {len(open_positions)} open positions on shutdown")
            self.execution.close_all("shutdown")
            self._record_exits()
        elif open_positions:
            self.log.critical(
                f"LEAVING {len(open_positions)} POSITIONS OPEN — stops remain at the broker",
                reason=", ".join(p.symbol for p in open_positions),
            )

        self._save_risk_state()
        self.book.save()

        state = self.risk.state
        self.log.info(
            "session complete",
            reason=(
                f"trades={state.trades_today} realised={state.realised_pnl:,.0f} "
                f"streak={state.consecutive_losses}"
            ),
        )
