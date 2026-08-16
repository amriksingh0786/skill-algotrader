"""
Order execution, stop-loss lifecycle, and position reconciliation.

This is the module where mistakes cost money rather than accuracy. Two rules
govern everything here:

  1. A position without a working stop is an emergency, not a warning.
     Two stop orders on one position is a nuisance; zero is the ₹8,000 lesson in
     NUANCE #3. Every path is biased toward "too much protection".

  2. The broker is the source of truth (NUANCE #2). Local state is a cache, and
     on any disagreement the cache is wrong.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .broker import BrokerError, Order, Position, round_to_tick
from .costs import estimate_costs
from .logs import TradingLogger
from .session import now_ist, to_ist
from .signals import Signal


@dataclass
class ManagedPosition:
    """
    A position the bot owns, with everything needed to manage its exit.

    `sl_order_id` is the live protective order. `adopted` marks positions found
    at the broker that the bot did not open — they are protected and closed, but
    never scaled into, since their entry rationale is unknown.
    """

    symbol: str
    direction: str
    quantity: int
    entry_price: float
    stop_loss: float
    target: float
    entry_time: datetime
    strategy: str = "UNKNOWN"
    sl_order_id: str | None = None
    initial_stop: float = 0.0
    high_water_mark: float = 0.0
    partial_booked: bool = False
    adopted: bool = False
    sl_failures: int = 0
    confidence: float = 0.0
    entry_reason: str = ""

    def __post_init__(self) -> None:
        if not self.initial_stop:
            self.initial_stop = self.stop_loss
        if not self.high_water_mark:
            self.high_water_mark = self.entry_price

    @property
    def risk_amount(self) -> float:
        """Rupees lost if the stop fires now — the portfolio-heat contribution."""
        return abs(self.entry_price - self.stop_loss) * abs(self.quantity)

    @property
    def value(self) -> float:
        return abs(self.quantity) * self.entry_price

    @property
    def exit_side(self) -> str:
        return "SELL" if self.direction == "LONG" else "BUY"

    def unrealised_pnl(self, last_price: float) -> float:
        delta = last_price - self.entry_price
        return delta * self.quantity if self.direction == "LONG" else -delta * abs(self.quantity)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["entry_time"] = self.entry_time.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ManagedPosition":
        data = dict(data)
        data["entry_time"] = datetime.fromisoformat(data["entry_time"])
        return cls(**data)


class PositionBook:
    """Managed positions with crash-safe persistence."""

    def __init__(self, path: str | Path = "state/positions.json") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.positions: dict[str, ManagedPosition] = {}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text())
            self.positions = {
                symbol: ManagedPosition.from_dict(data) for symbol, data in raw.items()
            }
        except (json.JSONDecodeError, KeyError, TypeError):
            # A corrupt cache must not stop the bot: reconciliation rebuilds it
            # from the broker, which is authoritative anyway.
            self.positions = {}

    def save(self) -> None:
        # Write-then-rename: a crash mid-write leaves the previous file intact
        # rather than a truncated one.
        temp = self.path.with_suffix(".tmp")
        temp.write_text(
            json.dumps({s: p.to_dict() for s, p in self.positions.items()}, indent=2)
        )
        temp.replace(self.path)

    def add(self, position: ManagedPosition) -> None:
        self.positions[position.symbol] = position
        self.save()

    def remove(self, symbol: str) -> ManagedPosition | None:
        position = self.positions.pop(symbol, None)
        self.save()
        return position

    def get(self, symbol: str) -> ManagedPosition | None:
        return self.positions.get(symbol)

    def all(self) -> list[ManagedPosition]:
        return list(self.positions.values())

    def __len__(self) -> int:
        return len(self.positions)


class ExecutionEngine:
    """Places, protects, and closes positions."""

    def __init__(
        self,
        broker: Any,
        book: PositionBook,
        logger: TradingLogger,
        config: dict | None = None,
    ) -> None:
        self.broker = broker
        self.book = book
        self.log = logger
        self.config = config or {}
        self.product = self.config.get("product", "MIS")

    # ------------------------------------------------------------ reconcile

    def reconcile(self, universe: set[str] | None = None) -> dict[str, list[str]]:
        """
        NUANCE #2: make local state match the broker. Run at every startup and
        periodically thereafter.

        Four disagreements are possible and all are handled:
          adopted   — broker holds it, we don't (crash before persist, or a
                      manual trade). Adopt and protect it.
          closed    — we hold it, broker doesn't (stop fired while we were down).
                      Drop it.
          resized   — quantities differ (partial fill or manual trim). Trust the
                      broker.
          protected — open position with no live stop order. Place one now; this
                      is the naked-position case.

        Args:
            universe: if given, broker positions outside it are ignored so the
                      bot never touches unrelated manual holdings.
        """
        result: dict[str, list[str]] = {
            "adopted": [], "closed": [], "resized": [], "protected": []
        }

        try:
            broker_positions = {p.symbol: p for p in self.broker.positions()}
            live_orders = self.broker.orders()
        except BrokerError as exc:
            self.log.critical("reconciliation failed — refusing to trade", exc_info=True,
                              reason=str(exc))
            raise

        if universe is not None:
            broker_positions = {s: p for s, p in broker_positions.items() if s in universe}

        open_sl_orders = {
            order.symbol: order
            for order in live_orders
            if order.status in ("TRIGGER PENDING", "OPEN")
            and order.order_type in ("SL", "SL-M")
        }

        # Broker has it, we don't → adopt.
        for symbol, position in broker_positions.items():
            if symbol in self.book.positions:
                continue

            last_price = position.last_price or position.average_price
            stop_distance = last_price * self.config.get("adopted_stop_pct", 1.0) / 100
            direction = position.direction

            adopted = ManagedPosition(
                symbol=symbol,
                direction=direction,
                quantity=abs(position.quantity),
                entry_price=position.average_price,
                stop_loss=(
                    position.average_price - stop_distance
                    if direction == "LONG"
                    else position.average_price + stop_distance
                ),
                target=0.0,  # unknown intent; managed to the stop and squareoff only
                entry_time=now_ist(),
                strategy="ADOPTED",
                adopted=True,
            )
            self.book.add(adopted)
            result["adopted"].append(symbol)
            self.log.warning("adopted untracked position", symbol=symbol,
                             quantity=position.quantity, price=position.average_price)

        # We have it, broker doesn't → it closed while we weren't looking.
        for symbol in list(self.book.positions):
            if symbol not in broker_positions:
                stale = self.book.remove(symbol)
                result["closed"].append(symbol)
                self.log.warning("position closed while bot was down", symbol=symbol,
                                 reason="not present at broker")
                if stale and stale.sl_order_id:
                    self._safe_cancel(stale.sl_order_id)

        # Quantities disagree → broker wins.
        for symbol, managed in self.book.positions.items():
            broker_quantity = abs(broker_positions[symbol].quantity)
            if broker_quantity != managed.quantity:
                self.log.warning("quantity mismatch, trusting broker", symbol=symbol,
                                 reason=f"local={managed.quantity} broker={broker_quantity}")
                managed.quantity = broker_quantity
                managed.entry_price = broker_positions[symbol].average_price
                result["resized"].append(symbol)

        # Open position with no protective order → naked. Fix immediately.
        for symbol, managed in self.book.positions.items():
            sl_order = open_sl_orders.get(symbol)
            if sl_order:
                managed.sl_order_id = sl_order.order_id
                continue

            self.log.critical("NAKED POSITION — no stop loss found", symbol=symbol,
                              quantity=managed.quantity)
            if self._place_stop(managed):
                result["protected"].append(symbol)
            else:
                self.log.critical("could not protect position, exiting at market",
                                  symbol=symbol)
                self.close(managed, reason="unprotectable after reconciliation")

        self.book.save()
        return result

    # --------------------------------------------------------------- entries

    def enter(self, signal: Signal, quantity: int, *, risk_note: str = "") -> ManagedPosition | None:
        """
        Open a position and protect it.

        Entry is a MARKET order by default. A LIMIT entry that does not fill
        leaves the scan believing it holds a position it does not, and the
        bookkeeping to handle that reliably is not worth the few basis points a
        limit saves on liquid names.

        Partial fills are respected: the stop is placed for the quantity actually
        filled, never the quantity requested (KNOWLEDGE.md section 8 — placing a
        1000-share stop against a 600-share fill leaves 400 shares short).
        """
        if quantity <= 0:
            return None

        side = "BUY" if signal.direction == "LONG" else "SELL"
        entry_type = self.config.get("entry_order_type", "MARKET")

        try:
            order_id = self.broker.place_order(
                symbol=signal.symbol,
                side=side,
                quantity=quantity,
                order_type=entry_type,
                product=self.product,
                price=signal.entry_price if entry_type == "LIMIT" else 0.0,
                tag=f"ENT{signal.strategy[:3]}",
            )
        except BrokerError as exc:
            self.log.error("entry order rejected", symbol=signal.symbol,
                           quantity=quantity, reason=str(exc))
            return None

        filled_quantity, average_price = self._await_fill(order_id, signal.symbol)

        if filled_quantity == 0:
            self.log.warning("entry did not fill", symbol=signal.symbol, order_id=order_id)
            self._safe_cancel(order_id)
            return None

        if filled_quantity < quantity:
            self.log.warning("partial fill — protecting filled quantity only",
                             symbol=signal.symbol,
                             reason=f"requested={quantity} filled={filled_quantity}")

        position = ManagedPosition(
            symbol=signal.symbol,
            direction=signal.direction,
            quantity=filled_quantity,
            entry_price=average_price or signal.entry_price,
            stop_loss=signal.stop_loss,
            target=signal.target,
            entry_time=now_ist(),
            strategy=signal.strategy,
            confidence=signal.confidence,
            entry_reason=signal.reason,
        )

        if not self._place_stop(position):
            # Cannot protect it — do not keep it. An unprotected position is a
            # bigger risk than the round-trip cost of closing immediately.
            self.log.critical("stop placement failed after entry, unwinding",
                              symbol=position.symbol)
            self.close(position, reason="could not place protective stop", persist=False)
            return None

        self.book.add(position)
        self.log.info("ENTRY", symbol=position.symbol, quantity=filled_quantity,
                      price=position.entry_price,
                      reason=f"{signal.strategy} conf={signal.confidence:.2f} {risk_note}")
        return position

    def _await_fill(self, order_id: str, symbol: str, timeout_seconds: float = 10.0
                    ) -> tuple[int, float]:
        """
        Poll until the order reaches a terminal state.

        Returns (filled_quantity, average_price). A timeout returns whatever has
        filled so far, which is the honest answer — the caller protects that
        quantity and logs the shortfall.
        """
        import time

        deadline = time.monotonic() + timeout_seconds
        filled, average = 0, 0.0

        while time.monotonic() < deadline:
            # Paper broker fills synchronously; poll() is its exchange.
            if hasattr(self.broker, "poll"):
                self.broker.poll()

            try:
                orders = {o.order_id: o for o in self.broker.orders()}
            except BrokerError:
                time.sleep(0.3)
                continue

            order = orders.get(order_id)
            if not order:
                time.sleep(0.3)
                continue

            filled, average = order.filled_quantity, order.average_price

            if order.status in ("COMPLETE", "REJECTED", "CANCELLED"):
                if order.status == "REJECTED":
                    self.log.error("entry rejected by exchange", symbol=symbol,
                                   order_id=order_id)
                return filled, average

            time.sleep(0.3)

        return filled, average

    # ------------------------------------------------------------ stop loss

    def _place_stop(self, position: ManagedPosition) -> bool:
        """Place a fresh SL-M order. Records the id on the position."""
        tick = self.broker.tick_size(position.symbol)
        trigger = round_to_tick(position.stop_loss, tick)

        try:
            order_id = self.broker.place_order(
                symbol=position.symbol,
                side=position.exit_side,
                quantity=position.quantity,
                order_type="SL-M",
                product=self.product,
                trigger_price=trigger,
                tag="SL",
            )
        except BrokerError as exc:
            position.sl_failures += 1
            self.log.error("stop placement failed", symbol=position.symbol,
                           reason=str(exc))
            return False

        position.sl_order_id = order_id
        position.stop_loss = trigger
        position.sl_failures = 0
        return True

    def move_stop(self, position: ManagedPosition, new_stop: float) -> bool:
        """
        NUANCE #3: PLACE the new stop, THEN cancel the old one.

        Cancel-first leaves the position naked in the window between the two
        calls, and if the place fails it stays naked indefinitely — the exact bug
        that cost ₹8,000 in ten minutes. Place-first can leave two stops if the
        cancel fails, which is merely redundant: the first to trigger flattens
        the position and the second becomes a no-op rejection.

        Never widens a stop. A stop that moves away from price is not a stop.
        """
        tick = self.broker.tick_size(position.symbol)
        new_stop = round_to_tick(new_stop, tick)

        tightening = (
            new_stop > position.stop_loss
            if position.direction == "LONG"
            else new_stop < position.stop_loss
        )
        if not tightening:
            return False

        old_order_id = position.sl_order_id

        try:
            new_order_id = self.broker.place_order(
                symbol=position.symbol,
                side=position.exit_side,
                quantity=position.quantity,
                order_type="SL-M",
                product=self.product,
                trigger_price=new_stop,
                tag="SLTRAIL",
            )
        except BrokerError as exc:
            # Old stop is untouched and still protecting the position.
            position.sl_failures += 1
            self.log.error("new stop failed, keeping existing stop",
                           symbol=position.symbol, reason=str(exc))

            if position.sl_failures >= self.config.get("max_sl_failures", 3):
                self.log.critical("stop management failing repeatedly, closing at market",
                                  symbol=position.symbol)
                self.close(position, reason="stop management failure")
            return False

        position.sl_order_id = new_order_id
        position.stop_loss = new_stop
        position.sl_failures = 0

        if old_order_id:
            try:
                self.broker.cancel_order(old_order_id)
            except BrokerError as exc:
                # Two live stops. Safe, and self-resolving on the first trigger.
                self.log.warning("old stop cancel failed — two stops now live (safe)",
                                 symbol=position.symbol, order_id=old_order_id,
                                 reason=str(exc))

        self.book.save()
        self.log.info("stop tightened", symbol=position.symbol, price=new_stop)
        return True

    def _safe_cancel(self, order_id: str) -> None:
        try:
            self.broker.cancel_order(order_id)
        except BrokerError as exc:
            self.log.debug("cancel failed (likely already gone)", order_id=order_id,
                           reason=str(exc))

    # ---------------------------------------------------------------- exits

    def close(
        self,
        position: ManagedPosition,
        *,
        reason: str,
        quantity: int | None = None,
        persist: bool = True,
    ) -> dict[str, Any] | None:
        """
        Close all or part of a position at market.

        The protective stop is cancelled BEFORE the exit for a full close —
        otherwise the stop remains live against a flat position and can open a
        fresh position in the opposite direction if it triggers later.
        """
        exit_quantity = quantity or position.quantity
        full_close = exit_quantity >= position.quantity

        if full_close and position.sl_order_id:
            self._safe_cancel(position.sl_order_id)
            position.sl_order_id = None

        try:
            order_id = self.broker.place_order(
                symbol=position.symbol,
                side=position.exit_side,
                quantity=exit_quantity,
                order_type="MARKET",
                product=self.product,
                tag="EXIT",
            )
        except BrokerError as exc:
            self.log.critical("EXIT FAILED — position still open", symbol=position.symbol,
                              quantity=exit_quantity, reason=str(exc))
            if full_close and not position.sl_order_id:
                self._place_stop(position)  # restore protection
            return None

        filled, average = self._await_fill(order_id, position.symbol)
        exit_price = average or self.broker.ltp([position.symbol]).get(
            position.symbol, position.entry_price
        )

        gross = (
            (exit_price - position.entry_price) * filled
            if position.direction == "LONG"
            else (position.entry_price - exit_price) * filled
        )
        costs = self._estimate_costs(position.entry_price, exit_price, filled)
        net = gross - costs

        record = {
            "symbol": position.symbol,
            "direction": position.direction,
            "strategy": position.strategy,
            "quantity": filled,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "entry_time": position.entry_time.isoformat(),
            "exit_time": now_ist().isoformat(),
            "holding_minutes": round(
                (now_ist() - to_ist(position.entry_time)).total_seconds() / 60, 1
            ),
            "gross_pnl": round(gross, 2),
            "costs": round(costs, 2),
            "pnl": round(net, 2),
            "pnl_pct": round(net / (position.entry_price * filled) * 100, 4) if filled else 0.0,
            "exit_reason": reason,
            "initial_stop": position.initial_stop,
            "final_stop": position.stop_loss,
            "confidence": position.confidence,
            "entry_reason": position.entry_reason,
            "adopted": position.adopted,
        }

        if full_close:
            if persist:
                self.book.remove(position.symbol)
            self.log.info("EXIT", symbol=position.symbol, quantity=filled,
                          price=exit_price, pnl=round(net, 2), reason=reason)
        else:
            position.quantity -= filled
            position.partial_booked = True
            self.book.save()
            self.log.info("PARTIAL EXIT", symbol=position.symbol, quantity=filled,
                          price=exit_price, pnl=round(net, 2), reason=reason)
            # The remaining shares are now over-protected by a stop sized for the
            # original quantity; resize it.
            self._resize_stop(position)

        self.log.trade(record)
        return record

    def _resize_stop(self, position: ManagedPosition) -> None:
        """After a partial exit, replace the stop with one sized to what remains."""
        old_order_id = position.sl_order_id
        if self._place_stop(position) and old_order_id:
            self._safe_cancel(old_order_id)
        self.book.save()

    def _estimate_costs(self, entry: float, exit_price: float, quantity: int) -> float:
        """Costs from the shared model — identical to what the backtest charges."""
        return estimate_costs(
            entry, exit_price, quantity, intraday=self.product == "MIS"
        )

    # ----------------------------------------------------------- management

    def manage(self, position: ManagedPosition, last_price: float,
               now: datetime | None = None) -> str | None:
        """
        Apply exit rules to one open position. Called every loop.

        Order matters: target and stop first (they are why the position exists),
        then partial booking, then trailing, then time decay. Returns the exit
        reason if the position was closed.

        The stop itself is a resting order at the exchange, so it fires even if
        this process dies. These checks are a second layer, not the primary one.
        """
        now = to_ist(now or now_ist())

        if position.direction == "LONG":
            position.high_water_mark = max(position.high_water_mark, last_price)
            hit_target = position.target > 0 and last_price >= position.target
            hit_stop = last_price <= position.stop_loss
        else:
            position.high_water_mark = min(position.high_water_mark, last_price)
            hit_target = position.target > 0 and last_price <= position.target
            hit_stop = last_price >= position.stop_loss

        if hit_target:
            self.close(position, reason="target hit")
            return "target hit"

        if hit_stop:
            # The resting stop should have fired already; if we see this, it did
            # not (gap through, or a rejected order). Exit at market now.
            self.log.warning("price through stop but position still open — exiting",
                             symbol=position.symbol, price=last_price)
            self.close(position, reason="stop breached")
            return "stop breached"

        # Partial booking at 1R: takes the trade off risk and lets the rest run.
        partial_pct = self.config.get("partial_exit_pct", 0)
        if partial_pct and not position.partial_booked and not position.adopted:
            r_multiple = self._r_multiple(position, last_price)
            if r_multiple >= self.config.get("partial_exit_at_r", 1.0):
                quantity = int(position.quantity * partial_pct / 100)
                if quantity > 0:
                    self.close(position, reason=f"partial at {r_multiple:.1f}R",
                               quantity=quantity)

        # Breakeven stop once the trade has paid for itself.
        breakeven_r = self.config.get("breakeven_at_r", 1.0)
        if breakeven_r and self._r_multiple(position, last_price) >= breakeven_r:
            entry = position.entry_price
            buffer = entry * 0.0005  # cover costs, don't stop out at exactly flat
            self.move_stop(position, entry + buffer if position.direction == "LONG"
                           else entry - buffer)

        # ATR-free trailing stop: give back a fixed fraction of the best price.
        trail_pct = self.config.get("trail_pct", 0)
        if trail_pct:
            trailed = (
                position.high_water_mark * (1 - trail_pct / 100)
                if position.direction == "LONG"
                else position.high_water_mark * (1 + trail_pct / 100)
            )
            self.move_stop(position, trailed)

        # Time stop: a thesis that has not worked in N minutes is usually wrong,
        # and the capital is better used elsewhere.
        max_hold = self.config.get("max_hold_minutes", 0)
        if max_hold:
            held_minutes = (now - to_ist(position.entry_time)).total_seconds() / 60
            if held_minutes >= max_hold:
                self.close(position, reason=f"time stop ({int(held_minutes)}m)")
                return "time stop"

        return None

    @staticmethod
    def _r_multiple(position: ManagedPosition, last_price: float) -> float:
        """Profit measured in units of initial risk."""
        risk = abs(position.entry_price - position.initial_stop)
        if risk <= 0:
            return 0.0
        move = (
            last_price - position.entry_price
            if position.direction == "LONG"
            else position.entry_price - last_price
        )
        return move / risk

    def close_all(self, reason: str) -> list[dict[str, Any]]:
        """Flatten everything — square-off time, shutdown, or a halt."""
        records = []
        for position in self.book.all():
            record = self.close(position, reason=reason)
            if record:
                records.append(record)
        return records
