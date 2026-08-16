"""
F&O contract resolution (NFO).

Equity trading addresses an instrument by name. Derivatives do not: "buy NIFTY"
is not an order, because you must choose an expiry, and for options a strike and
a right. This module turns a signal on an underlying into a specific contract.

Everything is derived from Zerodha's NFO instrument dump rather than hardcoded.
Strike steps, lot sizes, and expiry calendars all change — NIFTY's lot size has
been 25, 50, 75 and 25 again within a few years, and SEBI moved index weeklies to
a single expiry per index in 2024. Reading them from the dump means the code does
not silently trade a stale lot size.

WHAT THIS MODULE DELIBERATELY DOES NOT DO
-----------------------------------------
Short naked options. The margin is large, the loss is unbounded, and a retail
account can be forced into liquidation on a gap. `resolve()` refuses to build a
short-option order; if you want premium selling, do it deliberately with spreads
and a margin model that understands them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import polars as pl

Segment = Literal["equity", "futures", "options"]
OptionType = Literal["CE", "PE"]

# Cash-settled index derivatives. Stock F&O is physically settled at expiry,
# which for a retail account means an unexpected delivery obligation — see
# `physical_settlement_risk`.
INDEX_UNDERLYINGS = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"}


class ContractError(RuntimeError):
    """No contract matches the request, or the request is unsafe."""


@dataclass(frozen=True)
class Contract:
    """One tradeable derivative."""

    tradingsymbol: str
    exchange: str
    instrument_token: int
    lot_size: int
    tick_size: float
    instrument_type: str          # FUT / CE / PE
    name: str                     # underlying, e.g. NIFTY
    expiry: date | None = None
    strike: float = 0.0

    @property
    def is_option(self) -> bool:
        return self.instrument_type in ("CE", "PE")

    @property
    def is_index(self) -> bool:
        return self.name in INDEX_UNDERLYINGS

    def days_to_expiry(self, today: date | None = None) -> int:
        if not self.expiry:
            return 9999
        return (self.expiry - (today or date.today())).days

    def __str__(self) -> str:
        return f"{self.exchange}:{self.tradingsymbol}"


class DerivativeMaster:
    """
    Index over the NFO instrument dump.

    Loaded once per day alongside the NSE dump (NUANCE #29: instrument tokens are
    reissued, so a cached token from last week can address a different contract).
    """

    def __init__(self, cache_dir: str | Path = ".cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._frame: pl.DataFrame | None = None
        self._loaded_for: date | None = None

    @property
    def _cache_path(self) -> Path:
        return self.cache_dir / f"instruments_NFO_{date.today():%Y%m%d}.parquet"

    def load(self, kite: Any, force: bool = False) -> None:
        if self._loaded_for == date.today() and not force:
            return

        if self._cache_path.exists() and not force:
            frame = pl.read_parquet(self._cache_path)
        else:
            records = kite.instruments("NFO")
            if not records:
                raise ContractError("empty NFO instrument dump")
            frame = pl.DataFrame(records, infer_schema_length=None)
            frame.write_parquet(self._cache_path)
            for old in self.cache_dir.glob("instruments_NFO_*.parquet"):
                if old != self._cache_path:
                    old.unlink(missing_ok=True)

        # Normalise expiry to a date column regardless of how it arrived.
        if "expiry" in frame.columns and frame["expiry"].dtype != pl.Date:
            frame = frame.with_columns(
                pl.col("expiry").cast(pl.Date, strict=False).alias("expiry")
            )

        self._frame = frame
        self._loaded_for = date.today()

    @property
    def frame(self) -> pl.DataFrame:
        if self._frame is None:
            raise ContractError("NFO instruments not loaded — call load(kite) first")
        return self._frame

    # ------------------------------------------------------------- expiries

    def expiries(self, name: str, instrument_type: str = "FUT") -> list[date]:
        """Sorted future expiries available for an underlying."""
        today = date.today()
        rows = self.frame.filter(
            (pl.col("name") == name.upper())
            & (pl.col("instrument_type") == instrument_type)
            & (pl.col("expiry").is_not_null())
            & (pl.col("expiry") >= today)
        )
        return sorted({row["expiry"] for row in rows.iter_rows(named=True)})

    def select_expiry(
        self,
        name: str,
        instrument_type: str = "FUT",
        *,
        which: str = "nearest",
        min_days: int = 0,
    ) -> date:
        """
        Pick an expiry.

        Args:
            which: 'nearest', 'next', or 'monthly'
            min_days: skip expiries closer than this. Expiry-day options have
                      violent gamma and near-zero time value — an intraday
                      strategy that is not explicitly an expiry strategy should
                      set min_days=1 and avoid the last session.
        """
        available = [
            expiry for expiry in self.expiries(name, instrument_type)
            if (expiry - date.today()).days >= min_days
        ]
        if not available:
            raise ContractError(
                f"no {instrument_type} expiry for {name} at least {min_days} days out"
            )

        if which == "nearest":
            return available[0]
        if which == "next":
            return available[1] if len(available) > 1 else available[0]
        if which == "monthly":
            # The monthly contract is the last expiry within its calendar month.
            by_month: dict[tuple[int, int], date] = {}
            for expiry in available:
                key = (expiry.year, expiry.month)
                by_month[key] = max(by_month.get(key, expiry), expiry)
            return sorted(by_month.values())[0]

        raise ContractError(f"unknown expiry selector {which!r}")

    # ------------------------------------------------------------ contracts

    def _to_contract(self, row: dict[str, Any]) -> Contract:
        return Contract(
            tradingsymbol=row["tradingsymbol"],
            exchange=row.get("exchange", "NFO"),
            instrument_token=int(row["instrument_token"]),
            lot_size=int(row["lot_size"]),
            tick_size=float(row["tick_size"]) or 0.05,
            instrument_type=row["instrument_type"],
            name=row["name"],
            expiry=row.get("expiry"),
            strike=float(row.get("strike") or 0.0),
        )

    def future(self, name: str, *, which: str = "nearest", min_days: int = 0) -> Contract:
        """Futures contract for an underlying."""
        expiry = self.select_expiry(name, "FUT", which=which, min_days=min_days)
        rows = self.frame.filter(
            (pl.col("name") == name.upper())
            & (pl.col("instrument_type") == "FUT")
            & (pl.col("expiry") == expiry)
        )
        if rows.height == 0:
            raise ContractError(f"no future for {name} expiring {expiry}")
        return self._to_contract(rows.row(0, named=True))

    def strikes(self, name: str, expiry: date, option_type: OptionType = "CE") -> list[float]:
        rows = self.frame.filter(
            (pl.col("name") == name.upper())
            & (pl.col("instrument_type") == option_type)
            & (pl.col("expiry") == expiry)
        )
        return sorted({float(r["strike"]) for r in rows.iter_rows(named=True)})

    def option(
        self,
        name: str,
        spot: float,
        option_type: OptionType,
        *,
        which: str = "nearest",
        min_days: int = 0,
        moneyness: int = 0,
    ) -> Contract:
        """
        Option contract nearest a target strike.

        Args:
            spot: underlying price, used to locate ATM
            moneyness: strike steps away from ATM. Negative is in-the-money for a
                       call, out-of-the-money for a put; positive is the reverse.

        Strike selection is by nearest available strike rather than by a
        hardcoded step, because steps differ per underlying and change over time.

        Direction note: ITM options (moneyness < 0 for CE) have higher delta, so
        they track the underlying more closely and suffer less theta — usually
        the right choice for a directional intraday signal. Far OTM options are
        cheap for a reason.
        """
        expiry = self.select_expiry(name, option_type, which=which, min_days=min_days)
        available = self.strikes(name, expiry, option_type)
        if not available:
            raise ContractError(f"no {option_type} strikes for {name} {expiry}")

        atm = min(available, key=lambda strike: abs(strike - spot))
        atm_index = available.index(atm)

        # For a call, "one step in the money" is a lower strike; for a put, higher.
        offset = moneyness if option_type == "CE" else -moneyness
        target_index = max(0, min(len(available) - 1, atm_index + offset))
        strike = available[target_index]

        rows = self.frame.filter(
            (pl.col("name") == name.upper())
            & (pl.col("instrument_type") == option_type)
            & (pl.col("expiry") == expiry)
            & (pl.col("strike") == strike)
        )
        if rows.height == 0:
            raise ContractError(f"no {name} {strike} {option_type} {expiry}")
        return self._to_contract(rows.row(0, named=True))

    def resolve(
        self,
        name: str,
        direction: str,
        spot: float,
        config: dict | None = None,
    ) -> Contract:
        """
        Turn a signal on an underlying into a contract to trade.

        Config keys:
            segment          'futures' | 'options'
            expiry_which     'nearest' | 'next' | 'monthly'   (default nearest)
            expiry_min_days  skip contracts expiring sooner    (default 1)
            option_moneyness strike steps from ATM             (default -1, ITM)

        A LONG signal buys a call, a SHORT signal buys a put. Both are long
        premium: defined risk, and no naked short option is ever constructed
        here. Selling premium needs a spread and a margin model this engine does
        not have.
        """
        config = config or {}
        segment = config.get("segment", "futures")

        if segment == "futures":
            return self.future(
                name,
                which=config.get("expiry_which", "nearest"),
                min_days=config.get("expiry_min_days", 1),
            )

        if segment == "options":
            option_type: OptionType = "CE" if direction.upper() == "LONG" else "PE"
            return self.option(
                name,
                spot,
                option_type,
                which=config.get("expiry_which", "nearest"),
                min_days=config.get("expiry_min_days", 1),
                moneyness=config.get("option_moneyness", -1),
            )

        raise ContractError(f"segment {segment!r} is not a derivative segment")


def physical_settlement_risk(contract: Contract, today: date | None = None) -> str | None:
    """
    Warn when a stock F&O position is drifting toward physical settlement.

    Index derivatives are cash settled. Stock F&O is not: an in-the-money stock
    option or an open stock future held into expiry becomes a delivery
    obligation, and the margin required jumps in the final days. For a retail
    account this shows up as an unexpected margin call or a forced square-off at
    a bad price.

    Returns a warning string, or None when there is nothing to flag.
    """
    if contract.is_index or not contract.expiry:
        return None

    days = contract.days_to_expiry(today)
    if days <= 2:
        return (
            f"{contract.tradingsymbol} expires in {days}d and is PHYSICALLY SETTLED "
            f"(stock F&O). Close it or accept a delivery obligation; margins are "
            f"already elevated."
        )
    if days <= 5:
        return (
            f"{contract.tradingsymbol} expires in {days}d — physical settlement "
            f"margins ramp up from here."
        )
    return None


def lots_to_quantity(lots: int, contract: Contract) -> int:
    """F&O orders are placed in shares, but only in multiples of the lot size."""
    return lots * contract.lot_size


def quantity_to_lots(quantity: int, contract: Contract) -> int:
    return quantity // contract.lot_size if contract.lot_size else 0
