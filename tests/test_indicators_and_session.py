"""
Indicator correctness and session timing.

The VWAP reset test is the single most valuable test in this suite — NUANCE #4
is the top cause of backtest/live divergence, and the failure is silent.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta

import polars as pl
import pytest

from engine import indicators
from engine.indicators import IndicatorError, add_indicators, is_warm, warmup_bars
from engine.session import (
    IST,
    TradingCalendar,
    market_close_for,
    check_clock_drift,
    is_candle_complete,
    is_market_open,
    is_pre_open,
    last_complete_candle_time,
    should_squareoff,
    should_trade_now,
)
from tests.conftest import make_candles


class TestVwapReset:
    def test_vwap_resets_at_every_session_boundary(self, enriched: pl.DataFrame) -> None:
        """NUANCE #4: the first bar of each session must have VWAP == its typical price."""
        firsts = (
            enriched.group_by("session_date")
            .agg([pl.col("vwap").first(), pl.col("high").first(),
                  pl.col("low").first(), pl.col("close").first()])
            .sort("session_date")
        )
        assert firsts.height >= 2, "need multiple sessions to prove a reset"

        for row in firsts.iter_rows(named=True):
            typical = (row["high"] + row["low"] + row["close"]) / 3
            assert row["vwap"] == pytest.approx(typical, rel=1e-9), (
                f"VWAP did not reset on {row['session_date']}"
            )

    def test_vwap_does_not_drift_across_days(self, enriched: pl.DataFrame) -> None:
        """A cumulative VWAP drifts further from price each day. This catches that."""
        last_per_day = (
            enriched.group_by("session_date")
            .agg([(pl.col("close") - pl.col("vwap")).abs().max().alias("max_gap"),
                  pl.col("close").mean().alias("avg_close")])
            .sort("session_date")
        )
        for row in last_per_day.iter_rows(named=True):
            assert row["max_gap"] < row["avg_close"] * 0.10, (
                "VWAP is far from price — likely accumulating across sessions"
            )


class TestIndicatorRanges:
    def test_rsi_within_bounds(self, enriched: pl.DataFrame) -> None:
        rsi = enriched["rsi"].drop_nulls()
        assert rsi.min() >= 0.0 and rsi.max() <= 100.0

    def test_adx_within_bounds(self, enriched: pl.DataFrame) -> None:
        adx = enriched["adx"].drop_nulls()
        assert adx.min() >= 0.0 and adx.max() <= 100.0

    def test_atr_positive(self, enriched: pl.DataFrame) -> None:
        assert enriched["atr"].drop_nulls().min() > 0

    def test_rsi_is_100_when_only_gains(self) -> None:
        """avg_loss == 0 must yield 100, not a division by zero."""
        rows = [
            {"timestamp": datetime(2026, 5, 4, 9, 15) + timedelta(minutes=i),
             "open": 100.0 + i, "high": 100.5 + i, "low": 99.5 + i,
             "close": 100.0 + i, "volume": 1000}
            for i in range(60)
        ]
        result = add_indicators(pl.DataFrame(rows))
        assert result["rsi"].drop_nulls().max() == pytest.approx(100.0)

    def test_avg_volume_excludes_current_bar(self) -> None:
        """
        The volume baseline must not contain the bar being tested, or a surge
        partially averages itself away and the filter under-fires.
        """
        rows = [
            {"timestamp": datetime(2026, 5, 4, 9, 15) + timedelta(minutes=i),
             "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
             "volume": 1000 if i < 40 else 100_000}
            for i in range(41)
        ]
        result = add_indicators(pl.DataFrame(rows), {"volume_lookback": 20})
        last = result.row(-1, named=True)
        assert last["avg_volume"] == pytest.approx(1000.0), (
            "current bar leaked into its own volume baseline"
        )
        assert last["volume_ratio"] == pytest.approx(100.0)

    def test_missing_columns_raise(self) -> None:
        with pytest.raises(IndicatorError, match="missing required columns"):
            add_indicators(pl.DataFrame({"timestamp": [datetime(2026, 5, 4)]}))

    def test_empty_frame_raises(self) -> None:
        with pytest.raises(IndicatorError):
            add_indicators(pl.DataFrame(schema={
                "timestamp": pl.Datetime, "open": pl.Float64, "high": pl.Float64,
                "low": pl.Float64, "close": pl.Float64, "volume": pl.Int64,
            }))

    def test_warmup_flag_false_early(self, enriched: pl.DataFrame) -> None:
        assert not is_warm(enriched.row(0, named=True))
        assert is_warm(enriched.row(-1, named=True))

    def test_warmup_bars_covers_adx(self) -> None:
        """ADX chains two Wilder smoothings, so it needs the longest warmup."""
        assert warmup_bars({"adx_period": 14}) >= 14 * 6

    def test_indicators_are_deterministic(self, candles: pl.DataFrame) -> None:
        first = add_indicators(candles)
        second = add_indicators(candles)
        assert first.equals(second)

    def test_ema_fast_reacts_faster_than_slow(self) -> None:
        rows = [
            {"timestamp": datetime(2026, 5, 4, 9, 15) + timedelta(minutes=i),
             "open": 100.0, "high": 100.0, "low": 100.0,
             "close": 100.0 if i < 50 else 120.0, "volume": 1000}
            for i in range(80)
        ]
        result = add_indicators(pl.DataFrame(rows))
        last = result.row(-1, named=True)
        assert last["ema_fast"] > last["ema_slow"]


class TestSessionTiming:
    def test_market_hours_current_rules(self) -> None:
        """NSE closes at 15:15 since 2026-08-03."""
        monday = datetime(2026, 8, 10, tzinfo=IST)
        assert not is_market_open(monday.replace(hour=9, minute=0))
        assert is_market_open(monday.replace(hour=9, minute=15))
        assert is_market_open(monday.replace(hour=15, minute=14))
        assert not is_market_open(monday.replace(hour=15, minute=15))

    def test_market_hours_before_the_change(self) -> None:
        """
        A backtest spanning 2026-08-03 must apply the close that was in force on
        each day, or every earlier session is cut 15 minutes short.
        """
        monday = datetime(2026, 5, 4, tzinfo=IST)
        assert is_market_open(monday.replace(hour=15, minute=29))
        assert not is_market_open(monday.replace(hour=15, minute=30))

    def test_market_close_lookup(self) -> None:
        assert market_close_for(date(2026, 8, 2)) == time(15, 30)
        assert market_close_for(date(2026, 8, 3)) == time(15, 15)
        assert market_close_for(date(2030, 1, 1)) == time(15, 15)

    def test_weekend_closed(self) -> None:
        saturday = datetime(2026, 5, 9, 10, 0, tzinfo=IST)
        assert saturday.weekday() == 5
        assert not is_market_open(saturday)

    def test_holiday_closed(self) -> None:
        calendar = TradingCalendar({datetime(2026, 5, 4).date()})
        assert not is_market_open(datetime(2026, 5, 4, 10, 0, tzinfo=IST), calendar)

    def test_pre_open_detected(self) -> None:
        assert is_pre_open(datetime(2026, 5, 4, 9, 5, tzinfo=IST))
        assert not is_pre_open(datetime(2026, 5, 4, 9, 20, tzinfo=IST))

    @pytest.mark.parametrize(
        "hour,minute,expected,reason_fragment",
        [
            (9, 20, False, "opening volatility"),   # NUANCE #28
            (9, 45, True, "ok"),
            (12, 0, False, "lunch"),                # NUANCE #9
            (13, 30, True, "ok"),
            (14, 40, False, "last entry"),
            (16, 0, False, "closed"),
        ],
    )
    def test_entry_windows(self, hour: int, minute: int, expected: bool,
                           reason_fragment: str) -> None:
        now = datetime(2026, 5, 4, hour, minute, tzinfo=IST)
        allowed, reason = should_trade_now(now)
        assert allowed is expected
        assert reason_fragment.lower() in reason.lower()

    def test_lunch_can_be_disabled(self) -> None:
        noon = datetime(2026, 5, 4, 12, 0, tzinfo=IST)
        assert should_trade_now(noon, skip_lunch=False)[0]

    def test_squareoff(self) -> None:
        """Default moved to 14:55 when the close moved to 15:15."""
        assert not should_squareoff(datetime(2026, 8, 10, 14, 54, tzinfo=IST))
        assert should_squareoff(datetime(2026, 8, 10, 14, 55, tzinfo=IST))


class TestCandleCompletion:
    def test_incomplete_candle_rejected(self) -> None:
        """NUANCE #6: a candle is not usable until its interval has fully elapsed."""
        start = datetime(2026, 5, 4, 9, 47, tzinfo=IST)
        assert not is_candle_complete(start, start + timedelta(seconds=30), 60)
        assert not is_candle_complete(start, start + timedelta(seconds=60), 60)
        assert is_candle_complete(start, start + timedelta(seconds=61), 60)

    def test_buffer_is_respected(self) -> None:
        start = datetime(2026, 5, 4, 9, 47, tzinfo=IST)
        just_after = start + timedelta(seconds=60, milliseconds=200)
        assert not is_candle_complete(start, just_after, 60, buffer_ms=500)
        assert is_candle_complete(start, just_after, 60, buffer_ms=100)

    def test_last_complete_candle_is_in_the_past(self) -> None:
        now = datetime(2026, 5, 4, 9, 47, 30, tzinfo=IST)
        last = last_complete_candle_time(now, 60)
        assert last < now
        assert (now - last).total_seconds() >= 60


class TestClockDrift:
    def test_drift_within_tolerance(self) -> None:
        broker_time = datetime(2026, 5, 4, 10, 0, 0, tzinfo=IST)
        within, drift = check_clock_drift(broker_time, broker_time + timedelta(seconds=1))
        assert within and drift == pytest.approx(1.0)

    def test_drift_beyond_tolerance(self) -> None:
        """NUANCE #16: a fast local clock makes the bot trade partial candles."""
        broker_time = datetime(2026, 5, 4, 10, 0, 0, tzinfo=IST)
        within, drift = check_clock_drift(broker_time, broker_time + timedelta(seconds=5))
        assert not within and drift == pytest.approx(5.0)
