"""
Historical data with a Parquet cache.

NUANCE #10: Parquet over JSON is roughly 28x on load, and the difference decides
whether a parameter sweep takes a coffee break or an afternoon.

NUANCE #15: backtest and live read through the SAME cache and the same code. If
the backtest reads a CSV someone cleaned by hand and live reads the API, the two
will disagree in ways that take days to find.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from .broker import INTERVAL_SECONDS
from .indicators import add_indicators, warmup_bars
from .session import IST, TradingCalendar, now_ist, to_ist

SCHEMA = {
    "timestamp": pl.Datetime,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Int64,
}


class DataError(RuntimeError):
    """Data could not be fetched, or failed a quality check."""


def empty_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=SCHEMA)


def validate(df: pl.DataFrame, symbol: str, *, strict: bool = False) -> list[str]:
    """
    Quality checks that stop a backtest from lying.

    A backtest cannot tell the difference between "this stock did not move" and
    "the data is broken" — both look like a flat line. These checks make the
    second case loud.
    """
    issues: list[str] = []
    if df.height == 0:
        return [f"{symbol}: no data"]

    # NUANCE #14: zero volume during market hours means a data gap, not a quiet
    # market. Signals computed on those bars use a stale price as if it were real.
    zero_volume = df.filter(pl.col("volume") == 0).height
    if zero_volume:
        share = zero_volume / df.height * 100
        issues.append(f"{symbol}: {zero_volume} zero-volume bars ({share:.1f}%)")

    invalid = df.filter(
        (pl.col("high") < pl.col("low"))
        | (pl.col("close") > pl.col("high"))
        | (pl.col("close") < pl.col("low"))
        | (pl.col("open") > pl.col("high"))
        | (pl.col("open") < pl.col("low"))
    ).height
    if invalid:
        issues.append(f"{symbol}: {invalid} bars violate OHLC ordering")

    if df.height > 1:
        duplicates = df.height - df.unique(subset=["timestamp"]).height
        if duplicates:
            issues.append(f"{symbol}: {duplicates} duplicate timestamps")

        # A >20% single-bar move on a liquid name is a bad print or an
        # unadjusted corporate action, not a trade you would have caught.
        jumps = df.select(
            (pl.col("close").pct_change().abs() > 0.20).sum().alias("n")
        ).item()
        if jumps:
            issues.append(f"{symbol}: {jumps} bars jump >20% (check splits/bonuses)")

    if strict and issues:
        raise DataError("; ".join(issues))

    return issues


class DataManager:
    """
    Cached OHLCV access.

    Cache layout:  {cache_dir}/{interval}/{SYMBOL}.parquet
    """

    def __init__(
        self,
        broker: Any,
        *,
        cache_dir: str | Path = ".cache/ohlcv",
        interval: str = "minute",
        calendar: TradingCalendar | None = None,
    ) -> None:
        if interval not in INTERVAL_SECONDS:
            raise DataError(f"unsupported interval {interval!r}")

        self.broker = broker
        self.interval = interval
        self.cache_dir = Path(cache_dir) / interval
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.calendar = calendar or TradingCalendar()
        self._last_fetch: dict[str, datetime] = {}

    def _path(self, symbol: str) -> Path:
        return self.cache_dir / f"{symbol}.parquet"

    def read_cache(self, symbol: str) -> pl.DataFrame:
        path = self._path(symbol)
        return pl.read_parquet(path) if path.exists() else empty_frame()

    def _write_cache(self, symbol: str, df: pl.DataFrame) -> None:
        if df.height:
            df.sort("timestamp").write_parquet(self._path(symbol), compression="zstd")

    def fetch(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        *,
        use_cache: bool = True,
        refresh_tail: bool = True,
    ) -> pl.DataFrame:
        """
        Candles for [start, end], served from cache and topped up from the API.

        Only the missing head and tail are requested — re-downloading a year of
        minute bars to add today's is the difference between a 12-second and a
        5-minute startup.

        Args:
            refresh_tail: re-fetch the last cached day. Today's bars are still
                          forming, so a cached "today" is usually incomplete.
        """
        start, end = to_ist(start), to_ist(end)
        cached = self.read_cache(symbol) if use_cache else empty_frame()

        if cached.height == 0:
            fresh = self._fetch_api(symbol, start, end)
            self._write_cache(symbol, fresh)
            return self._slice(fresh, start, end)

        cached_start = to_ist(cached["timestamp"].min())
        cached_end = to_ist(cached["timestamp"].max())
        pieces = [cached]

        if start < cached_start:
            pieces.append(self._fetch_api(symbol, start, cached_start))

        tail_from = cached_end.replace(hour=0, minute=0) if refresh_tail else cached_end
        if end > cached_end:
            pieces.append(self._fetch_api(symbol, tail_from, end))

        merged = (
            pl.concat(pieces, how="vertical_relaxed")
            .unique(subset=["timestamp"], keep="last")  # fresher bar wins
            .sort("timestamp")
        )
        self._write_cache(symbol, merged)
        return self._slice(merged, start, end)

    def _fetch_api(self, symbol: str, start: datetime, end: datetime) -> pl.DataFrame:
        if start >= end:
            return empty_frame()
        frame = self.broker.historical(symbol, self.interval, start, end)
        return frame if frame.height else empty_frame()

    @staticmethod
    def _slice(df: pl.DataFrame, start: datetime, end: datetime) -> pl.DataFrame:
        if df.height == 0:
            return df
        naive_start, naive_end = start.replace(tzinfo=None), end.replace(tzinfo=None)
        return df.filter(
            (pl.col("timestamp") >= naive_start) & (pl.col("timestamp") <= naive_end)
        )

    def with_indicators(
        self, symbol: str, start: datetime, end: datetime, config: dict | None = None,
        **kwargs: Any
    ) -> pl.DataFrame:
        """
        Fetch and enrich, extending the window backwards to cover indicator warmup.

        Without the extension, the first hour of every backtest runs on
        half-converged Wilder averages and produces trades that could not have
        been taken — an easy way to manufacture edge that does not exist.
        """
        config = config or {}
        bars_needed = warmup_bars(config)
        seconds = INTERVAL_SECONDS[self.interval]

        # Calendar padding: minute bars only exist for ~6.25h of each trading day.
        bars_per_day = max(1, int(6.25 * 3600 / seconds))
        pad_days = max(2, int(bars_needed / bars_per_day * 1.6) + 1)

        frame = self.fetch(symbol, to_ist(start) - timedelta(days=pad_days), end, **kwargs)
        if frame.height == 0:
            return frame

        enriched = add_indicators(frame, config, intraday=self.interval != "day")
        return enriched.filter(pl.col("timestamp") >= to_ist(start).replace(tzinfo=None))

    def live_frame(self, symbol: str, config: dict | None = None,
                   now: datetime | None = None) -> pl.DataFrame:
        """
        The frame the live runner evaluates: history plus today, indicators applied.

        Refetches a symbol at most once per candle interval — polling faster
        burns rate limit without producing new bars.
        """
        now = to_ist(now or now_ist())
        seconds = INTERVAL_SECONDS[self.interval]

        last = self._last_fetch.get(symbol)
        use_cache_only = last is not None and (now - last).total_seconds() < seconds

        config = config or {}
        lookback_days = max(5, warmup_bars(config) // 60 + 3)
        frame = self.with_indicators(
            symbol,
            now - timedelta(days=lookback_days),
            now,
            config,
            use_cache=True,
            refresh_tail=not use_cache_only,
        )

        if not use_cache_only:
            self._last_fetch[symbol] = now

        return frame

    def warm_cache(self, symbols: list[str], days: int = 90,
                   progress: bool = True) -> dict[str, int]:
        """
        Pre-download history for a universe.

        Run this once before backtesting. At Kite's 3 requests/second on
        historical data, 50 symbols is around 20 seconds; doing it lazily during
        a backtest instead makes the backtest look mysteriously slow.
        """
        end = now_ist()
        start = end - timedelta(days=days)
        counts: dict[str, int] = {}

        for index, symbol in enumerate(symbols, 1):
            try:
                frame = self.fetch(symbol, start, end)
                counts[symbol] = frame.height
                if progress:
                    print(f"  [{index}/{len(symbols)}] {symbol}: {frame.height} bars")
            except Exception as exc:  # noqa: BLE001 - one bad symbol must not stop the warm
                counts[symbol] = 0
                if progress:
                    print(f"  [{index}/{len(symbols)}] {symbol}: FAILED — {exc}")

        return counts
