"""Shared fixtures. Everything here is offline — no broker, no network."""

from __future__ import annotations

import random
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.broker import Order, Position  # noqa: E402
from engine.indicators import add_indicators  # noqa: E402


def make_candles(
    bars: int = 800,
    *,
    start: datetime | None = None,
    seed: int = 42,
    drift: float = 0.0,
    volatility: float = 0.001,
    start_price: float = 1000.0,
    session_bars: int = 375,
) -> pl.DataFrame:
    """
    Synthetic 1-minute OHLCV across trading sessions.

    Sessions start at 09:15 and run `session_bars` bars, so VWAP reset and
    session-timing logic get exercised the same way real data would.
    """
    random.seed(seed)
    start = start or datetime(2026, 5, 4, 9, 15)

    rows: list[dict[str, Any]] = []
    price = start_price
    day_offset = 0
    bar_in_session = 0

    while len(rows) < bars:
        session_day = start + timedelta(days=day_offset)
        if session_day.weekday() >= 5:
            day_offset += 1
            continue

        timestamp = session_day.replace(hour=9, minute=15) + timedelta(minutes=bar_in_session)
        price *= 1 + random.gauss(drift, volatility)
        open_price = price
        close_price = price * (1 + random.gauss(drift, volatility * 0.8))
        high = max(open_price, close_price) * (1 + abs(random.gauss(0, volatility * 0.4)))
        low = min(open_price, close_price) * (1 - abs(random.gauss(0, volatility * 0.4)))

        rows.append({
            "timestamp": timestamp,
            "open": round(open_price, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "close": round(close_price, 2),
            "volume": random.randint(10_000, 80_000),
        })

        bar_in_session += 1
        if bar_in_session >= session_bars:
            bar_in_session = 0
            day_offset += 1

    return pl.DataFrame(rows)


@pytest.fixture
def candles() -> pl.DataFrame:
    return make_candles()


@pytest.fixture
def enriched(candles: pl.DataFrame) -> pl.DataFrame:
    return add_indicators(candles)


@pytest.fixture
def config() -> dict:
    return {
        "rsi_long_min": 45, "rsi_long_max": 65, "adx_min": 25, "volume_mult": 1.5,
        "min_confidence": 0.50, "sl_atr_mult": 1.2, "risk_reward": 1.5,
        "risk_pct": 1.0, "max_positions": 5, "symbol_cooldown_minutes": 45,
        "max_consecutive_losses": 3, "daily_loss_limit_pct": 3.0,
        "max_portfolio_heat_pct": 5.0, "max_trades_per_day": 20,
    }


class FakeBroker:
    """
    Scriptable broker for execution tests.

    `fail_on` makes chosen operations raise, which is how the naked-position and
    place-then-cancel paths get exercised without a live account.
    """

    def __init__(self, prices: dict[str, float] | None = None) -> None:
        self.prices = prices or {"TEST": 1000.0}
        self.placed: list[dict[str, Any]] = []
        self.cancelled: list[str] = []
        # Interleaved log — the only way to prove place-then-cancel ordering.
        self.events: list[tuple[str, str]] = []
        self._orders: dict[str, Order] = {}
        self._positions: list[Position] = []
        self.fail_on: set[str] = set()
        self.counter = 0
        self.margin = 1_000_000.0
        self.fill_quantity_override: int | None = None

    def ltp(self, symbols: list[str]) -> dict[str, float]:
        return {s: self.prices.get(s, 1000.0) for s in symbols}

    def quote(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        return {s: {"last_price": self.prices.get(s, 1000.0)} for s in symbols}

    def tick_size(self, symbol: str) -> float:
        return 0.05

    def available_margin(self) -> float:
        return self.margin

    def place_order(self, symbol: str, side: str, quantity: int, *, order_type: str = "MARKET",
                    product: str = "MIS", price: float = 0.0, trigger_price: float = 0.0,
                    tag: str = "", variety: str = "regular") -> str:
        from engine.broker import BrokerError

        if "place" in self.fail_on:
            raise BrokerError("simulated place failure")
        if order_type in ("SL", "SL-M") and "place_sl" in self.fail_on:
            raise BrokerError("simulated SL place failure")

        self.counter += 1
        order_id = f"FAKE{self.counter}"
        self.events.append(("place", f"{order_type}:{order_id}"))
        self.placed.append({
            "order_id": order_id, "symbol": symbol, "side": side, "quantity": quantity,
            "order_type": order_type, "price": price, "trigger_price": trigger_price, "tag": tag,
        })

        filled = quantity if self.fill_quantity_override is None else self.fill_quantity_override
        self._orders[order_id] = Order(
            order_id=order_id, symbol=symbol, side=side, quantity=quantity,
            order_type=order_type, product=product, price=price, trigger_price=trigger_price,
            status="COMPLETE" if order_type == "MARKET" else "TRIGGER PENDING",
            filled_quantity=filled if order_type == "MARKET" else 0,
            average_price=self.prices.get(symbol, 1000.0) if order_type == "MARKET" else 0.0,
            tag=tag,
        )
        return order_id

    def cancel_order(self, order_id: str, *, variety: str = "regular") -> None:
        from engine.broker import BrokerError

        if "cancel" in self.fail_on:
            raise BrokerError("simulated cancel failure")
        self.events.append(("cancel", order_id))
        self.cancelled.append(order_id)
        if order_id in self._orders:
            self._orders[order_id].status = "CANCELLED"

    def modify_order(self, order_id: str, *, variety: str = "regular", **kwargs: Any) -> str:
        return order_id

    def orders(self) -> list[Order]:
        return list(self._orders.values())

    def positions(self) -> list[Position]:
        return list(self._positions)

    def set_positions(self, positions: list[Position]) -> None:
        self._positions = positions


@pytest.fixture
def broker() -> FakeBroker:
    return FakeBroker()
