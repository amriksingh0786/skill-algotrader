"""
Broker abstraction: Zerodha Kite for live, a simulator for paper trading.

Both implement the same `Broker` protocol, so `runner.py` cannot tell them apart.
That is the mechanism behind backtest/live parity (KNOWLEDGE.md section 2): there
is exactly one code path, and the mode only decides which object is constructed.

Auth flow (Kite access tokens expire daily, around 07:30 IST):
    1. `python -m engine.broker login`  prints the Kite login URL
    2. log in, copy the `request_token` from the redirect URL
    3. `python -m engine.broker token <request_token>`  writes .kite_session.json
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

import polars as pl

from .session import IST, now_ist, to_ist

# Kite historical API caps the span of a single request by interval.
_MAX_SPAN_DAYS = {
    "minute": 60,
    "3minute": 90,
    "5minute": 90,
    "10minute": 90,
    "15minute": 180,
    "30minute": 180,
    "60minute": 365,
    "day": 2000,
}

INTERVAL_SECONDS = {
    "minute": 60,
    "3minute": 180,
    "5minute": 300,
    "10minute": 600,
    "15minute": 900,
    "30minute": 1800,
    "60minute": 3600,
    "day": 86400,
}


class BrokerError(RuntimeError):
    """Broker call failed in a way the caller must handle."""


class AuthError(BrokerError):
    """Missing or expired credentials — always fatal, never retried."""


@dataclass
class Order:
    """A placed order as the engine sees it."""

    order_id: str
    symbol: str
    side: str  # BUY / SELL
    quantity: int
    order_type: str  # MARKET / LIMIT / SL / SL-M
    product: str  # MIS / CNC / NRML
    price: float = 0.0
    trigger_price: float = 0.0
    status: str = "OPEN"
    filled_quantity: int = 0
    average_price: float = 0.0
    tag: str = ""
    placed_at: datetime = field(default_factory=now_ist)


@dataclass
class Position:
    """Net position in one symbol."""

    symbol: str
    quantity: int  # negative for short
    average_price: float
    last_price: float = 0.0
    pnl: float = 0.0
    product: str = "MIS"

    @property
    def is_open(self) -> bool:
        return self.quantity != 0

    @property
    def direction(self) -> str:
        return "LONG" if self.quantity > 0 else "SHORT"

    @property
    def value(self) -> float:
        return abs(self.quantity) * self.last_price


class RateLimiter:
    """
    Token-bucket limiter. Kite allows ~3 requests/second on historical data and
    will hard-block an API key that ignores that.
    """

    def __init__(self, max_calls: int, per_seconds: float) -> None:
        self.max_calls = max_calls
        self.per_seconds = per_seconds
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            while True:
                now = time.monotonic()
                while self._calls and now - self._calls[0] > self.per_seconds:
                    self._calls.popleft()

                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    return

                time.sleep(self.per_seconds - (now - self._calls[0]) + 0.01)


class Broker(Protocol):
    """The surface `runner.py` and `execution.py` are written against."""

    def ltp(self, symbols: list[str]) -> dict[str, float]: ...
    def quote(self, symbols: list[str]) -> dict[str, dict[str, Any]]: ...
    def historical(
        self, symbol: str, interval: str, start: datetime, end: datetime
    ) -> pl.DataFrame: ...
    def place_order(self, **kwargs: Any) -> str: ...
    def cancel_order(self, order_id: str) -> None: ...
    def orders(self) -> list[Order]: ...
    def positions(self) -> list[Position]: ...
    def available_margin(self) -> float: ...
    def tick_size(self, symbol: str) -> float: ...


class InstrumentMaster:
    """
    Zerodha's instrument dump: tick size, lot size, and instrument tokens.

    NUANCE #1: tick size is per-instrument. Sending a price that is not a
    multiple of it is the largest single cause of order rejections, and the error
    message ("Tick size for this script is 5.00") arrives only after the order is
    already refused.

    NUANCE #29: instrument tokens are reissued and the dump must be refreshed
    daily. A cached token from last week can point at a different instrument.
    """

    def __init__(self, cache_dir: str | Path = ".cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._by_symbol: dict[str, dict[str, Any]] = {}
        self._loaded_for: date | None = None

    @property
    def _cache_path(self) -> Path:
        return self.cache_dir / f"instruments_{date.today():%Y%m%d}.parquet"

    def load(self, kite: Any, exchange: str = "NSE", force: bool = False) -> None:
        """Load today's dump from cache, downloading it once per day."""
        if self._loaded_for == date.today() and not force:
            return

        if self._cache_path.exists() and not force:
            frame = pl.read_parquet(self._cache_path)
        else:
            records = kite.instruments(exchange)
            if not records:
                raise BrokerError(f"empty instrument dump for {exchange}")
            frame = pl.DataFrame(records, infer_schema_length=None)
            frame.write_parquet(self._cache_path)
            self._purge_old_dumps()

        wanted = ("tradingsymbol", "instrument_token", "tick_size", "lot_size",
                  "segment", "name")
        columns = [c for c in wanted if c in frame.columns]

        self._by_symbol = {
            row["tradingsymbol"]: row
            for row in frame.select(columns).iter_rows(named=True)
        }
        self._loaded_for = date.today()

    def _purge_old_dumps(self) -> None:
        for old in self.cache_dir.glob("instruments_*.parquet"):
            if old != self._cache_path:
                old.unlink(missing_ok=True)

    def tick_size(self, symbol: str, default: float = 0.05) -> float:
        record = self._by_symbol.get(symbol)
        return float(record["tick_size"]) if record and record.get("tick_size") else default

    def lot_size(self, symbol: str, default: int = 1) -> int:
        record = self._by_symbol.get(symbol)
        return int(record["lot_size"]) if record and record.get("lot_size") else default

    def token(self, symbol: str) -> int:
        record = self._by_symbol.get(symbol)
        if not record:
            raise BrokerError(
                f"{symbol} not in instrument dump — delisted, renamed, or wrong exchange"
            )
        return int(record["instrument_token"])

    def __contains__(self, symbol: str) -> bool:
        return symbol in self._by_symbol


def _normalise_candles(candles: list[dict[str, Any]]) -> pl.DataFrame:
    """
    Kite candles -> a frame with NAIVE IST timestamps.

    kiteconnect returns tz-aware datetimes at +05:30. Polars stores those
    internally as UTC, so `replace_time_zone(None)` — which drops the zone
    without converting — yields the UTC wall clock: a 09:15 IST candle becomes
    03:45, and every session check downstream reads it as pre-open. The whole
    engine then rejects every bar and backtests return zero trades with no
    error to explain it.

    Convert first, then strip. Everything downstream treats naive datetimes as
    IST (see session.to_ist), so this is the one place the conversion happens.
    """
    frame = pl.DataFrame(candles)
    timestamp = pl.col("date")

    dtype = frame.schema["date"]
    if isinstance(dtype, pl.Datetime) and dtype.time_zone is not None:
        timestamp = timestamp.dt.convert_time_zone("Asia/Kolkata").dt.replace_time_zone(None)

    return frame.select(
        timestamp.alias("timestamp"),
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.col("volume").cast(pl.Int64),
    )


_PLACEHOLDER_MARKERS = ("your_", "_here", "xxx", "changeme", "<", "placeholder")


def _reject_placeholder(name: str, value: str) -> None:
    """
    Catch an unedited .env template before Kite does.

    Left alone, a placeholder key produces "Incorrect `api_key` or
    `access_token`" from the API, which reads like an auth expiry and sends you
    re-running the login flow instead of editing the file.
    """
    lowered = value.strip().lower()
    if not lowered or any(marker in lowered for marker in _PLACEHOLDER_MARKERS):
        raise AuthError(
            f"{name} is still the template placeholder ({value[:24]!r}).\n"
            f"  Edit .env with real values from https://kite.trade/ "
            f"(Kite Connect app -> API key & secret)."
        )


def round_to_tick(price: float, tick: float) -> float:
    """
    NUANCE #1: snap a price to the instrument's tick grid.

    Rounds to the tick, then to 2 decimals — float arithmetic leaves values like
    1847.3500000000001, which Kite rejects just as readily as an off-grid price.
    """
    if tick <= 0:
        return round(price, 2)
    return round(round(price / tick) * tick, 2)


class KiteBroker:
    """Live Zerodha Kite Connect broker."""

    def __init__(
        self,
        api_key: str | None = None,
        access_token: str | None = None,
        *,
        session_file: str | Path = ".kite_session.json",
        cache_dir: str | Path = ".cache",
    ) -> None:
        try:
            from kiteconnect import KiteConnect
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise BrokerError("kiteconnect not installed: pip install kiteconnect") from exc

        self.api_key = api_key or os.getenv("KITE_API_KEY")
        if not self.api_key:
            raise AuthError("KITE_API_KEY missing (set it in .env)")
        _reject_placeholder("KITE_API_KEY", self.api_key)

        self.session_file = Path(session_file)

        # Precedence: explicit argument, then today's session file, then .env.
        # The session file is checked BEFORE the environment because it carries
        # an issued_at stamp and is rewritten by `token`. A stale
        # KITE_ACCESS_TOKEN left in .env would otherwise shadow a fresh login
        # every morning, producing "Incorrect api_key or access_token" from an
        # account that just authenticated successfully.
        token = access_token or self._load_session() or os.getenv("KITE_ACCESS_TOKEN")
        if not token:
            raise AuthError(
                "no access token. Run:\n"
                "    ./run.sh login          # opens the Kite login URL\n"
                "    ./run.sh token <request_token>\n"
                "Kite tokens expire daily around 07:30 IST."
            )
        _reject_placeholder("KITE_ACCESS_TOKEN", token)

        self.kite = KiteConnect(api_key=self.api_key)
        self.kite.set_access_token(token)

        self.instruments = InstrumentMaster(cache_dir)
        # Lazily constructed: only F&O users pay the cost of the NFO dump, which
        # is far larger than the NSE one (every strike of every expiry).
        self._derivatives: Any = None
        self._cache_dir = cache_dir
        self._quote_limiter = RateLimiter(max_calls=8, per_seconds=1.0)
        self._historical_limiter = RateLimiter(max_calls=3, per_seconds=1.0)
        self._order_limiter = RateLimiter(max_calls=8, per_seconds=1.0)

        self.instruments.load(self.kite)
        self._verify_session()

    # ---------------------------------------------------------------- session

    def _load_session(self) -> str | None:
        if not self.session_file.exists():
            return None

        data = json.loads(self.session_file.read_text())
        issued = datetime.fromisoformat(data["issued_at"])

        # Kite invalidates tokens at ~07:30 IST daily. A token issued before
        # today's cutoff is already dead; failing here beats discovering it on
        # the first order of the session.
        cutoff = now_ist().replace(hour=7, minute=30, second=0, microsecond=0)
        if to_ist(issued) < cutoff <= now_ist():
            return None

        return data.get("access_token")

    def _verify_session(self) -> None:
        try:
            self.profile = self.kite.profile()
        except Exception as exc:  # noqa: BLE001
            hint = (
                "\n  A rejected token usually means one of:\n"
                "    - the token expired (they die daily ~07:30 IST) -> ./run.sh login\n"
                "    - a request_token was pasted into .env instead of being exchanged;\n"
                "      request tokens are single-use and must go through ./run.sh token <t>\n"
                "    - KITE_ACCESS_TOKEN in .env is stale; remove that line and let\n"
                "      .kite_session.json manage it"
            )
            raise AuthError(f"access token rejected: {exc}{hint}") from exc

    @staticmethod
    def login_url(api_key: str | None = None) -> str:
        from kiteconnect import KiteConnect

        key = api_key or os.getenv("KITE_API_KEY")
        if not key:
            raise AuthError("KITE_API_KEY missing")
        return KiteConnect(api_key=key).login_url()

    @staticmethod
    def exchange_session(
        request_token: str,
        api_key: str | None = None,
        api_secret: str | None = None,
        session_file: str | Path = ".kite_session.json",
    ) -> str:
        """Trade a request_token for an access_token and persist it."""
        from kiteconnect import KiteConnect

        key = api_key or os.getenv("KITE_API_KEY")
        secret = api_secret or os.getenv("KITE_API_SECRET")
        if not key or not secret:
            raise AuthError("KITE_API_KEY and KITE_API_SECRET required")

        data = KiteConnect(api_key=key).generate_session(request_token, api_secret=secret)

        path = Path(session_file)
        path.write_text(
            json.dumps(
                {
                    "access_token": data["access_token"],
                    "user_id": data.get("user_id", ""),
                    "issued_at": now_ist().isoformat(),
                },
                indent=2,
            )
        )
        path.chmod(0o600)  # the token is a bearer credential for a funded account
        return data["access_token"]

    # ------------------------------------------------------------ market data

    @staticmethod
    def _key(symbol: str) -> str:
        return symbol if ":" in symbol else f"NSE:{symbol}"

    def ltp(self, symbols: list[str]) -> dict[str, float]:
        if not symbols:
            return {}

        result: dict[str, float] = {}
        # NUANCE #12: batch. One call for 200 symbols, not 200 calls.
        for chunk in _chunked(symbols, 200):
            self._quote_limiter.acquire()
            try:
                response = self.kite.ltp([self._key(s) for s in chunk])
            except Exception as exc:  # noqa: BLE001
                raise BrokerError(f"ltp failed: {exc}") from exc

            for key, payload in response.items():
                result[key.split(":", 1)[-1]] = float(payload["last_price"])

        return result

    def quote(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        if not symbols:
            return {}

        result: dict[str, dict[str, Any]] = {}
        for chunk in _chunked(symbols, 200):
            self._quote_limiter.acquire()
            try:
                response = self.kite.quote([self._key(s) for s in chunk])
            except Exception as exc:  # noqa: BLE001
                raise BrokerError(f"quote failed: {exc}") from exc

            for key, payload in response.items():
                result[key.split(":", 1)[-1]] = payload

        return result

    def historical(
        self, symbol: str, interval: str, start: datetime, end: datetime
    ) -> pl.DataFrame:
        """
        Historical candles, chunked to respect Kite's per-request span limits and
        rate limit. Returns an empty frame with the right schema if nothing came
        back, so callers never branch on None.
        """
        if interval not in _MAX_SPAN_DAYS:
            raise BrokerError(f"unsupported interval {interval!r}")

        token = self.instruments.token(symbol)
        span = timedelta(days=_MAX_SPAN_DAYS[interval])
        frames: list[pl.DataFrame] = []
        cursor = to_ist(start)
        end = to_ist(end)

        while cursor < end:
            chunk_end = min(cursor + span, end)
            self._historical_limiter.acquire()

            try:
                candles = self.kite.historical_data(
                    instrument_token=token,
                    from_date=cursor,
                    to_date=chunk_end,
                    interval=interval,
                )
            except Exception as exc:  # noqa: BLE001
                raise BrokerError(f"historical {symbol} {cursor:%F}-{chunk_end:%F}: {exc}") from exc

            if candles:
                frames.append(_normalise_candles(candles))

            cursor = chunk_end

        if not frames:
            return pl.DataFrame(
                schema={
                    "timestamp": pl.Datetime,
                    "open": pl.Float64,
                    "high": pl.Float64,
                    "low": pl.Float64,
                    "close": pl.Float64,
                    "volume": pl.Int64,
                }
            )

        return pl.concat(frames).unique(subset=["timestamp"], keep="first").sort("timestamp")

    # ----------------------------------------------------------------- orders

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        *,
        order_type: str = "MARKET",
        product: str = "MIS",
        price: float = 0.0,
        trigger_price: float = 0.0,
        tag: str = "",
        variety: str = "regular",
        exchange: str = "NSE",
        tick: float | None = None,
    ) -> str:
        """
        Place an order.

        Args:
            exchange: NSE for equity, NFO for futures and options. F&O quantities
                      must already be multiples of the contract lot size — the
                      exchange rejects anything else.
            tick: override the tick size, for instruments not in the NSE dump
                  (F&O contracts come from the NFO dump instead).
        """
        if quantity <= 0:
            raise BrokerError(f"quantity must be positive, got {quantity}")

        tick = tick if tick is not None else self.tick_size(symbol)
        params: dict[str, Any] = {
            "variety": variety,
            "exchange": exchange,
            "tradingsymbol": symbol,
            "transaction_type": side.upper(),
            "quantity": int(quantity),
            "product": product,
            "order_type": order_type,
            "validity": "DAY",
        }

        # NUANCE #1: every price that reaches the exchange is tick-aligned.
        if order_type in ("LIMIT", "SL"):
            params["price"] = round_to_tick(price, tick)
        if order_type in ("SL", "SL-M"):
            params["trigger_price"] = round_to_tick(trigger_price, tick)
        if tag:
            params["tag"] = tag[:20]  # Kite truncates silently past 20 chars

        self._order_limiter.acquire()
        try:
            return str(self.kite.place_order(**params))
        except Exception as exc:  # noqa: BLE001
            raise BrokerError(f"place_order {side} {quantity} {symbol}: {exc}") from exc

    def modify_order(self, order_id: str, *, variety: str = "regular", **kwargs: Any) -> str:
        self._order_limiter.acquire()
        try:
            return str(self.kite.modify_order(variety=variety, order_id=order_id, **kwargs))
        except Exception as exc:  # noqa: BLE001
            raise BrokerError(f"modify_order {order_id}: {exc}") from exc

    def cancel_order(self, order_id: str, *, variety: str = "regular") -> None:
        self._order_limiter.acquire()
        try:
            self.kite.cancel_order(variety=variety, order_id=order_id)
        except Exception as exc:  # noqa: BLE001
            raise BrokerError(f"cancel_order {order_id}: {exc}") from exc

    def orders(self) -> list[Order]:
        try:
            raw = self.kite.orders()
        except Exception as exc:  # noqa: BLE001
            raise BrokerError(f"orders failed: {exc}") from exc

        return [
            Order(
                order_id=str(o["order_id"]),
                symbol=o["tradingsymbol"],
                side=o["transaction_type"],
                quantity=int(o["quantity"]),
                order_type=o["order_type"],
                product=o["product"],
                price=float(o.get("price") or 0),
                trigger_price=float(o.get("trigger_price") or 0),
                status=o["status"],
                filled_quantity=int(o.get("filled_quantity") or 0),
                average_price=float(o.get("average_price") or 0),
                tag=o.get("tag") or "",
            )
            for o in raw
        ]

    def positions(self) -> list[Position]:
        """
        Net positions. This is the source of truth for reconciliation
        (NUANCE #2) — never the bot's own idea of what it holds.
        """
        try:
            raw = self.kite.positions()
        except Exception as exc:  # noqa: BLE001
            raise BrokerError(f"positions failed: {exc}") from exc

        return [
            Position(
                symbol=p["tradingsymbol"],
                quantity=int(p["quantity"]),
                average_price=float(p["average_price"]),
                last_price=float(p.get("last_price") or 0),
                pnl=float(p.get("pnl") or 0),
                product=p.get("product", "MIS"),
            )
            for p in raw.get("net", [])
            if int(p["quantity"]) != 0
        ]

    def available_margin(self) -> float:
        """
        NUANCE #7: use `net`, not `opening_balance`. Opening balance ignores
        margin already blocked by open positions, so sizing against it
        overcommits and the next order is rejected for insufficient funds.
        """
        try:
            margins = self.kite.margins("equity")
        except Exception as exc:  # noqa: BLE001
            raise BrokerError(f"margins failed: {exc}") from exc
        return float(margins["net"])

    def tick_size(self, symbol: str) -> float:
        return self.instruments.tick_size(symbol)

    @property
    def derivatives(self) -> Any:
        """
        NFO contract master, loaded on first use.

        F&O is contract resolution plus a different cost model; the runner does
        not yet route orders through it. See derivatives.py.
        """
        if self._derivatives is None:
            from .derivatives import DerivativeMaster

            self._derivatives = DerivativeMaster(self._cache_dir)
            self._derivatives.load(self.kite)
        return self._derivatives

    def server_time(self) -> datetime:
        """
        Broker-side timestamp, for the NUANCE #16 clock-drift check. Kite has no
        time endpoint, so the most recent order timestamp is used as a proxy;
        absent any orders, drift cannot be measured and local time is returned.
        """
        for order in self.orders():
            if order.placed_at:
                return order.placed_at
        return now_ist()


class PaperBroker:
    """
    Simulated order execution against real market data.

    Market data comes from a wrapped broker (normally KiteBroker) so paper mode
    sees exactly the prices live mode sees. Only fills are simulated.

    Fills are modelled honestly:
      - market orders cross the spread and pay slippage
      - limit orders fill only when the market trades through the limit
      - SL-M orders trigger on the trigger price and then slip

    An optimistic fill model is how a paper-profitable strategy turns into a
    live-unprofitable one, so the defaults here are deliberately pessimistic.
    """

    def __init__(
        self,
        data_source: Any,
        *,
        starting_capital: float = 1_000_000.0,
        slippage_bps: float = 5.0,
        brokerage_per_order: float = 20.0,
    ) -> None:
        self.data = data_source
        self.starting_capital = starting_capital
        self.cash = starting_capital
        self.slippage_bps = slippage_bps
        self.brokerage_per_order = brokerage_per_order

        self._orders: dict[str, Order] = {}
        self._positions: dict[str, Position] = {}
        self._order_counter = 0
        self.fills: list[dict[str, Any]] = []

    # market data delegates straight through
    def ltp(self, symbols: list[str]) -> dict[str, float]:
        return self.data.ltp(symbols)

    def quote(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        return self.data.quote(symbols)

    def historical(self, symbol: str, interval: str, start: datetime, end: datetime) -> pl.DataFrame:
        return self.data.historical(symbol, interval, start, end)

    def tick_size(self, symbol: str) -> float:
        return self.data.tick_size(symbol)

    def _next_id(self) -> str:
        self._order_counter += 1
        return f"PAPER{self._order_counter:08d}"

    def _fill_price(self, symbol: str, side: str, reference: float) -> float:
        """Reference price plus slippage in the direction that hurts."""
        slip = reference * (self.slippage_bps / 10_000)
        price = reference + slip if side.upper() == "BUY" else reference - slip
        return round_to_tick(price, self.tick_size(symbol))

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        *,
        order_type: str = "MARKET",
        product: str = "MIS",
        price: float = 0.0,
        trigger_price: float = 0.0,
        tag: str = "",
        variety: str = "regular",
    ) -> str:
        if quantity <= 0:
            raise BrokerError(f"quantity must be positive, got {quantity}")

        order_id = self._next_id()
        tick = self.tick_size(symbol)
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side.upper(),
            quantity=int(quantity),
            order_type=order_type,
            product=product,
            price=round_to_tick(price, tick) if price else 0.0,
            trigger_price=round_to_tick(trigger_price, tick) if trigger_price else 0.0,
            tag=tag,
        )
        self._orders[order_id] = order

        if order_type == "MARKET":
            last = self.ltp([symbol]).get(symbol)
            if last is None:
                raise BrokerError(f"no quote for {symbol}, cannot simulate fill")
            self._execute(order, self._fill_price(symbol, order.side, last))

        return order_id

    def poll(self) -> None:
        """
        Advance resting orders against the current market.

        The live runner calls this each loop; live mode does nothing equivalent
        because the exchange does this work. Keeping it a no-op-shaped method on
        the paper side only preserves the shared code path.
        """
        resting = [o for o in self._orders.values() if o.status == "OPEN" and o.order_type != "MARKET"]
        if not resting:
            return

        prices = self.ltp(list({o.symbol for o in resting}))

        for order in resting:
            last = prices.get(order.symbol)
            if last is None:
                continue

            if order.order_type == "LIMIT":
                crossed = (
                    (order.side == "BUY" and last <= order.price)
                    or (order.side == "SELL" and last >= order.price)
                )
                if crossed:
                    self._execute(order, order.price)  # limit fills at the limit

            elif order.order_type in ("SL", "SL-M"):
                triggered = (
                    (order.side == "BUY" and last >= order.trigger_price)
                    or (order.side == "SELL" and last <= order.trigger_price)
                )
                if triggered:
                    # A stop becomes a market order: it slips past the trigger.
                    self._execute(order, self._fill_price(order.symbol, order.side, last))

    def _execute(self, order: Order, fill_price: float) -> None:
        order.status = "COMPLETE"
        order.filled_quantity = order.quantity
        order.average_price = fill_price

        signed = order.quantity if order.side == "BUY" else -order.quantity
        existing = self._positions.get(order.symbol)

        if existing is None or existing.quantity == 0:
            self._positions[order.symbol] = Position(
                symbol=order.symbol,
                quantity=signed,
                average_price=fill_price,
                last_price=fill_price,
                product=order.product,
            )
        else:
            new_quantity = existing.quantity + signed
            if existing.quantity * signed > 0:  # adding to the position
                total_cost = existing.average_price * existing.quantity + fill_price * signed
                existing.average_price = total_cost / new_quantity
            elif new_quantity != 0 and existing.quantity * new_quantity < 0:
                existing.average_price = fill_price  # flipped through zero
            existing.quantity = new_quantity
            existing.last_price = fill_price

        # Realised cash effect plus costs.
        self.cash -= signed * fill_price
        self.cash -= self.brokerage_per_order

        self.fills.append(
            {
                "order_id": order.order_id,
                "symbol": order.symbol,
                "side": order.side,
                "quantity": order.quantity,
                "price": fill_price,
                "tag": order.tag,
                "timestamp": now_ist().isoformat(),
            }
        )

    def cancel_order(self, order_id: str, *, variety: str = "regular") -> None:
        order = self._orders.get(order_id)
        if not order:
            raise BrokerError(f"unknown order {order_id}")
        if order.status == "OPEN":
            order.status = "CANCELLED"

    def modify_order(self, order_id: str, *, variety: str = "regular", **kwargs: Any) -> str:
        order = self._orders.get(order_id)
        if not order:
            raise BrokerError(f"unknown order {order_id}")
        if "price" in kwargs:
            order.price = round_to_tick(float(kwargs["price"]), self.tick_size(order.symbol))
        if "trigger_price" in kwargs:
            order.trigger_price = round_to_tick(
                float(kwargs["trigger_price"]), self.tick_size(order.symbol)
            )
        if "quantity" in kwargs:
            order.quantity = int(kwargs["quantity"])
        return order_id

    def orders(self) -> list[Order]:
        return list(self._orders.values())

    def positions(self) -> list[Position]:
        open_positions = [p for p in self._positions.values() if p.quantity != 0]
        if open_positions:
            prices = self.ltp([p.symbol for p in open_positions])
            for position in open_positions:
                position.last_price = prices.get(position.symbol, position.last_price)
                position.pnl = (position.last_price - position.average_price) * position.quantity
        return open_positions

    def available_margin(self) -> float:
        return self.cash


def _chunked(items: list[Any], size: int) -> list[list[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _cli() -> None:  # pragma: no cover - interactive auth helper
    import sys

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    command = sys.argv[1] if len(sys.argv) > 1 else "help"

    if command == "login":
        print("\nOpen this URL, log in, then copy the request_token from the redirect:\n")
        print(f"  {KiteBroker.login_url()}\n")
        print("Then run:  python -m engine.broker token <request_token>\n")
    elif command == "token":
        if len(sys.argv) < 3:
            print("usage: python -m engine.broker token <request_token>")
            raise SystemExit(1)
        KiteBroker.exchange_session(sys.argv[2])
        print("Access token saved to .kite_session.json (valid until ~07:30 IST tomorrow)")
    else:
        print("usage: python -m engine.broker [login|token <request_token>]")


if __name__ == "__main__":  # pragma: no cover
    _cli()
