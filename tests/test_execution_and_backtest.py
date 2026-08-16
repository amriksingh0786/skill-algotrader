"""
Execution safety and backtest honesty.

The tests in TestStopLossLifecycle and TestReconciliation guard the two failures
that cost real money in KNOWLEDGE.md: the naked position and the duplicate entry
after a restart. The backtest tests guard against a subtler cost — a simulation
that flatters itself and sends you live with an edge that was never there.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from engine.broker import Position, round_to_tick
from engine.costs import breakeven_move_pct, estimate_costs
from engine.execution import ExecutionEngine, ManagedPosition, PositionBook
from engine.logs import TradingLogger
from engine.session import IST
from engine.signals import Signal


@pytest.fixture
def engine(broker, tmp_path):
    book = PositionBook(tmp_path / "positions.json")
    logger = TradingLogger(tmp_path / "logs", console=False)
    config = {"product": "MIS", "max_sl_failures": 3, "adopted_stop_pct": 1.0}
    return ExecutionEngine(broker, book, logger, config), broker, book


def _signal(symbol: str = "TEST", entry: float = 1000.0) -> Signal:
    return Signal(
        symbol=symbol, direction="LONG", entry_price=entry, stop_loss=entry * 0.99,
        target=entry * 1.015, confidence=0.65, reason="test", strategy="FORTRESS",
        timestamp=datetime(2026, 5, 4, 10, 0, tzinfo=IST),
    )


def _position(symbol: str = "TEST", **overrides) -> ManagedPosition:
    base = dict(
        symbol=symbol, direction="LONG", quantity=100, entry_price=1000.0,
        stop_loss=990.0, target=1015.0, entry_time=datetime(2026, 5, 4, 10, 0, tzinfo=IST),
        strategy="FORTRESS", sl_order_id="OLD_SL",
    )
    base.update(overrides)
    return ManagedPosition(**base)


class TestStopLossLifecycle:
    def test_new_stop_is_placed_before_old_is_cancelled(self, engine) -> None:
        """
        NUANCE #3 — the ordering that prevents a naked position.

        Cancel-first opens a window with no protection; if the subsequent place
        fails the position stays naked. This asserts the actual interleaving,
        not just that both calls happened.
        """
        execution, broker, book = engine
        position = _position()
        book.add(position)
        broker.events.clear()

        assert execution.move_stop(position, 995.0)

        kinds = [kind for kind, _ in broker.events]
        assert kinds == ["place", "cancel"], f"wrong order: {broker.events}"
        assert broker.cancelled == ["OLD_SL"]
        assert position.stop_loss == 995.0
        assert position.sl_order_id != "OLD_SL"

    def test_failed_placement_keeps_the_old_stop(self, engine) -> None:
        """If the new stop cannot be placed, the old one must survive untouched."""
        execution, broker, book = engine
        position = _position()
        book.add(position)
        broker.fail_on.add("place_sl")

        assert not execution.move_stop(position, 995.0)
        assert broker.cancelled == [], "old stop was cancelled despite placement failure"
        assert position.sl_order_id == "OLD_SL"
        assert position.stop_loss == 990.0

    def test_failed_cancel_leaves_two_stops_not_zero(self, engine) -> None:
        """Two live stops is the acceptable failure mode; zero is not."""
        execution, broker, book = engine
        position = _position()
        book.add(position)
        broker.fail_on.add("cancel")

        assert execution.move_stop(position, 995.0)
        assert position.sl_order_id is not None
        assert position.stop_loss == 995.0

    def test_stop_never_widens(self, engine) -> None:
        """A stop that moves away from price is not a stop."""
        execution, broker, book = engine
        position = _position()
        book.add(position)

        assert not execution.move_stop(position, 985.0)
        assert position.stop_loss == 990.0
        assert broker.placed == []

    def test_repeated_failures_trigger_emergency_exit(self, engine) -> None:
        """NUANCE #3's escape hatch: if protection cannot be maintained, flatten."""
        execution, broker, book = engine
        position = _position()
        book.add(position)
        broker.fail_on.add("place_sl")

        for _ in range(3):
            execution.move_stop(position, position.stop_loss + 1.0)

        exit_orders = [o for o in broker.placed if o["order_type"] == "MARKET"]
        assert exit_orders, "expected an emergency market exit after repeated SL failures"

    def test_stop_trigger_is_tick_aligned(self, engine) -> None:
        execution, broker, book = engine
        position = _position(stop_loss=990.0)
        book.add(position)

        execution.move_stop(position, 993.3333)
        assert position.stop_loss == round_to_tick(993.3333, 0.05)


class TestEntry:
    def test_entry_places_order_then_stop(self, engine) -> None:
        execution, broker, book = engine
        position = execution.enter(_signal(), 100)

        assert position is not None
        assert [o["order_type"] for o in broker.placed] == ["MARKET", "SL-M"]
        assert book.get("TEST") is not None

    def test_partial_fill_protects_only_what_filled(self, engine) -> None:
        """
        KNOWLEDGE.md section 8: a stop sized for the requested quantity against a
        partial fill leaves the account short the difference.
        """
        execution, broker, book = engine
        broker.fill_quantity_override = 60

        position = execution.enter(_signal(), 100)

        assert position is not None
        assert position.quantity == 60
        stop_order = [o for o in broker.placed if o["order_type"] == "SL-M"][0]
        assert stop_order["quantity"] == 60

    def test_unfilled_entry_creates_no_position(self, engine) -> None:
        execution, broker, book = engine
        broker.fill_quantity_override = 0

        assert execution.enter(_signal(), 100) is None
        assert len(book) == 0

    def test_position_is_unwound_if_stop_cannot_be_placed(self, engine) -> None:
        """An unprotected position is worse than no position."""
        execution, broker, book = engine
        broker.fail_on.add("place_sl")

        assert execution.enter(_signal(), 100) is None
        assert len(book) == 0
        exits = [o for o in broker.placed if o["order_type"] == "MARKET" and o["side"] == "SELL"]
        assert exits, "position was not unwound after stop placement failed"

    def test_rejected_entry_returns_none(self, engine) -> None:
        execution, broker, book = engine
        broker.fail_on.add("place")
        assert execution.enter(_signal(), 100) is None

    def test_zero_quantity_is_a_no_op(self, engine) -> None:
        execution, broker, book = engine
        assert execution.enter(_signal(), 0) is None
        assert broker.placed == []


class TestExit:
    def test_stop_is_cancelled_before_market_exit(self, engine) -> None:
        """
        A stop left live against a flat position can open a NEW position in the
        opposite direction if it triggers later.
        """
        execution, broker, book = engine
        position = _position()
        book.add(position)
        broker.events.clear()

        execution.close(position, reason="target hit")

        kinds = [kind for kind, _ in broker.events]
        assert kinds[0] == "cancel"
        assert "OLD_SL" in broker.cancelled
        assert len(book) == 0

    def test_exit_records_a_trade_with_costs(self, engine, tmp_path) -> None:
        execution, broker, book = engine
        position = _position()
        book.add(position)
        broker.prices["TEST"] = 1010.0

        record = execution.close(position, reason="target hit")

        assert record is not None
        assert record["exit_reason"] == "target hit"
        assert record["costs"] > 0
        assert record["pnl"] == pytest.approx(record["gross_pnl"] - record["costs"], abs=0.01)

    def test_partial_exit_resizes_the_stop(self, engine) -> None:
        execution, broker, book = engine
        position = _position(quantity=100)
        book.add(position)

        execution.close(position, reason="partial", quantity=40)

        assert position.quantity == 60
        stop_orders = [o for o in broker.placed if o["order_type"] == "SL-M"]
        assert stop_orders[-1]["quantity"] == 60

    def test_failed_exit_restores_protection(self, engine) -> None:
        execution, broker, book = engine
        position = _position()
        book.add(position)
        broker.fail_on.add("place")

        assert execution.close(position, reason="target") is None
        assert len(book) == 1, "position must remain tracked if the exit failed"


class TestReconciliation:
    def test_adopts_untracked_broker_position(self, engine) -> None:
        """NUANCE #2: the broker is the source of truth."""
        execution, broker, book = engine
        broker.set_positions([
            Position(symbol="ORPHAN", quantity=50, average_price=500.0, last_price=505.0)
        ])

        result = execution.reconcile({"ORPHAN"})

        assert "ORPHAN" in result["adopted"]
        adopted = book.get("ORPHAN")
        assert adopted is not None and adopted.adopted
        assert adopted.stop_loss < adopted.entry_price
        assert any(o["order_type"] == "SL-M" for o in broker.placed)

    def test_drops_positions_the_broker_no_longer_has(self, engine) -> None:
        """Stop fired while the bot was down — do not re-manage a closed trade."""
        execution, broker, book = engine
        book.add(_position("GONE"))
        broker.set_positions([])

        result = execution.reconcile({"GONE"})

        assert "GONE" in result["closed"]
        assert len(book) == 0

    def test_trusts_broker_quantity_on_mismatch(self, engine) -> None:
        execution, broker, book = engine
        book.add(_position("TEST", quantity=100))
        broker.set_positions([
            Position(symbol="TEST", quantity=60, average_price=1000.0, last_price=1000.0)
        ])
        result = execution.reconcile({"TEST"})

        assert "TEST" in result["resized"]
        assert book.get("TEST").quantity == 60

    def test_protects_a_naked_position(self, engine) -> None:
        """An open position with no live stop order is the emergency case."""
        execution, broker, book = engine
        book.add(_position("NAKED", sl_order_id=None))
        broker.set_positions([
            Position(symbol="NAKED", quantity=100, average_price=1000.0, last_price=1000.0)
        ])

        result = execution.reconcile({"NAKED"})

        assert "NAKED" in result["protected"]
        assert any(o["order_type"] == "SL-M" for o in broker.placed)
        assert book.get("NAKED").sl_order_id is not None

    def test_ignores_symbols_outside_the_universe(self, engine) -> None:
        """Manual holdings and ETFs must not be touched."""
        execution, broker, book = engine
        broker.set_positions([
            Position(symbol="MANUAL_ETF", quantity=10, average_price=100.0, last_price=100.0)
        ])

        result = execution.reconcile({"TEST"})

        assert result["adopted"] == []
        assert len(book) == 0


class TestPositionBook:
    def test_survives_a_restart(self, tmp_path) -> None:
        path = tmp_path / "positions.json"
        book = PositionBook(path)
        book.add(_position("PERSIST"))

        reloaded = PositionBook(path)
        assert reloaded.get("PERSIST") is not None
        assert reloaded.get("PERSIST").quantity == 100

    def test_corrupt_state_does_not_crash(self, tmp_path) -> None:
        """Reconciliation rebuilds from the broker, so a torn file is recoverable."""
        path = tmp_path / "positions.json"
        path.write_text("{ this is not json")
        assert len(PositionBook(path)) == 0

    def test_risk_amount_reflects_stop_distance(self) -> None:
        position = _position(quantity=100, entry_price=1000.0, stop_loss=990.0)
        assert position.risk_amount == pytest.approx(1000.0)


class TestCandleNormalisation:
    """
    Kite returns tz-aware IST datetimes; the engine works in naive IST.

    Getting this wrong shifts every candle by -5:30, so 09:15 reads as 03:45,
    every session gate reports "pre-open", and backtests return zero trades with
    no error. It was found only by fetching real data.
    """

    def _candles(self) -> list[dict]:
        from zoneinfo import ZoneInfo

        ist = ZoneInfo("Asia/Kolkata")
        return [
            {"date": datetime(2026, 8, 10, 9, 15, tzinfo=ist), "open": 100.0,
             "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000},
            {"date": datetime(2026, 8, 10, 15, 14, tzinfo=ist), "open": 100.0,
             "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000},
        ]

    def test_ist_wall_clock_is_preserved(self) -> None:
        from engine.broker import _normalise_candles

        frame = _normalise_candles(self._candles())
        first, last = frame.row(0, named=True), frame.row(1, named=True)

        assert first["timestamp"].hour == 9 and first["timestamp"].minute == 15, (
            f"expected 09:15 IST, got {first['timestamp']} — timezone was stripped, not converted"
        )
        assert last["timestamp"].hour == 15 and last["timestamp"].minute == 14

    def test_result_is_timezone_naive(self) -> None:
        from engine.broker import _normalise_candles

        frame = _normalise_candles(self._candles())
        assert frame.schema["timestamp"].time_zone is None

    def test_naive_input_passes_through_unchanged(self) -> None:
        """Cached frames and synthetic data are already naive."""
        from engine.broker import _normalise_candles

        candles = [{"date": datetime(2026, 8, 10, 9, 15), "open": 100.0, "high": 101.0,
                    "low": 99.0, "close": 100.5, "volume": 1000}]
        assert _normalise_candles(candles).row(0, named=True)["timestamp"].hour == 9

    def test_session_gate_accepts_a_normalised_bar(self) -> None:
        """End to end: a real 09:45 IST candle must be inside the entry window."""
        from engine.broker import _normalise_candles
        from engine.session import should_trade_now
        from zoneinfo import ZoneInfo

        candles = [{"date": datetime(2026, 8, 10, 9, 45, tzinfo=ZoneInfo("Asia/Kolkata")),
                    "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
                    "volume": 1000}]
        timestamp = _normalise_candles(candles).row(0, named=True)["timestamp"]
        allowed, reason = should_trade_now(timestamp)
        assert allowed, f"normalised bar rejected: {reason}"


class TestCosts:
    def test_intraday_charges_stt_on_sell_only(self) -> None:
        """NUANCE #30."""
        buy_heavy = estimate_costs(1000.0, 900.0, 100, intraday=True)
        sell_heavy = estimate_costs(900.0, 1000.0, 100, intraday=True)
        assert sell_heavy > buy_heavy

    def test_delivery_costs_more_than_intraday(self) -> None:
        assert estimate_costs(1000.0, 1010.0, 100, intraday=False) > \
               estimate_costs(1000.0, 1010.0, 100, intraday=True)

    def test_brokerage_is_capped(self) -> None:
        """Zerodha caps at Rs 20 per executed order — Rs 40 round trip."""
        costs = estimate_costs(10_000.0, 10_000.0, 1000, intraday=True)
        assert costs < 10_000_000 * 0.0005

    def test_zero_quantity_is_free(self) -> None:
        assert estimate_costs(1000.0, 1010.0, 0) == 0.0

    def test_breakeven_move_is_small_but_real(self) -> None:
        move = breakeven_move_pct(1000.0, 100, intraday=True)
        assert 0.0 < move < 1.0


class TestBacktestHonesty:
    def _frame(self, closes: list[float], *, highs=None, lows=None) -> pl.DataFrame:
        from engine.indicators import add_indicators

        rows = []
        for index, close in enumerate(closes):
            rows.append({
                "timestamp": datetime(2026, 5, 4, 9, 15) + timedelta(minutes=index),
                "open": close, "close": close,
                "high": highs[index] if highs else close * 1.001,
                "low": lows[index] if lows else close * 0.999,
                "volume": 50_000,
            })
        return add_indicators(pl.DataFrame(rows))

    def test_stop_wins_when_a_bar_spans_both_levels(self) -> None:
        """
        Without tick data the order is unknowable. Assuming the target hit first
        inflates win rate; this asserts the pessimistic choice.
        """
        from engine.backtest import OpenTrade, _resolve_exit

        trade = OpenTrade(
            symbol="T", direction="LONG", quantity=100, entry_price=1000.0,
            entry_time=datetime(2026, 5, 4, 10, 0, tzinfo=IST),
            stop_loss=990.0, target=1010.0, initial_stop=990.0,
            strategy="T", confidence=0.6, reason="", high_water_mark=1000.0,
        )
        bar = {"high": 1015.0, "low": 985.0, "close": 1000.0}

        price, reason = _resolve_exit(
            trade, bar, datetime(2026, 5, 4, 10, 1, tzinfo=IST), {}, True,
            datetime(2026, 5, 4, 15, 10).time(),
        )
        assert reason == "stop loss"
        assert price == 990.0

    def test_entries_fill_at_the_next_bar_open(self, config: dict) -> None:
        """
        No lookahead: a signal computed on bar i's close cannot fill at bar i's
        close, because that price is only known once the bar has ended.
        """
        from engine import backtest

        frame = self._frame([1000.0 + i * 0.5 for i in range(400)])
        result = backtest.run({"T": frame}, {**config, "adx_min": 0, "min_confidence": 0.0,
                                             "rsi_long_min": 0, "rsi_long_max": 100},
                              strategy="fortress", starting_capital=1_000_000,
                              slippage_bps=0.0)

        for trade in result.trades:
            entry_time = datetime.fromisoformat(trade["entry_time"])
            matching = frame.filter(
                pl.col("timestamp") == entry_time.replace(tzinfo=None)
            )
            if matching.height:
                assert trade["entry_price"] == pytest.approx(
                    matching.row(0, named=True)["open"], rel=1e-6
                ), "entry did not fill at the bar open"

    def test_costs_are_charged_on_every_trade(self, config: dict) -> None:
        from engine import backtest

        frame = self._frame([1000.0 + (i % 20) * 2 for i in range(600)])
        result = backtest.run({"T": frame}, {**config, "adx_min": 0, "min_confidence": 0.0,
                                             "rsi_long_min": 0, "rsi_long_max": 100},
                              strategy="fortress", starting_capital=1_000_000)

        for trade in result.trades:
            assert trade["costs"] > 0
            assert trade["pnl"] < trade["gross_pnl"]

    def test_no_data_returns_an_error_not_a_crash(self) -> None:
        from engine import backtest

        result = backtest.run({}, {}, strategy="fortress")
        assert "error" in result.metrics

    def test_metrics_flag_small_samples(self) -> None:
        from engine.backtest import compute_metrics

        trades = [{"pnl": 100.0, "costs": 10.0, "exit_reason": "target", "symbol": "A",
                   "holding_minutes": 5.0, "r_multiple": 1.0}]
        metrics = compute_metrics(trades, [{"date": "2026-05-04", "equity": 1_000_100}],
                                  1_000_000)
        assert metrics["sample_warning"] is not None

    def test_metrics_on_empty_trades(self) -> None:
        from engine.backtest import compute_metrics

        metrics = compute_metrics([], [], 1_000_000)
        assert metrics["total_trades"] == 0

    def test_custom_strategy_columns_survive_preparation(self, config: dict) -> None:
        """
        A strategy may add its own columns (an opening range, a higher-timeframe
        trend). `prepare()` used to select a fixed allowlist, which dropped them
        — the strategy then saw None on every row and returned no signals, with
        no error and nothing in the rejection counts to explain it.
        """
        from engine.backtest import prepare

        frame = self._frame([1000.0 + i for i in range(120)]).with_columns(
            pl.lit(42.0).alias("my_custom_column")
        )
        stream = prepare({"T": frame}, config, "fortress")

        assert "my_custom_column" in stream.columns
        assert stream["my_custom_column"].drop_nulls().len() == stream.height

    def test_frames_with_different_columns_align(self, config: dict) -> None:
        """Symbols whose frames differ in shape must not crash the concat."""
        from engine.backtest import prepare

        base = self._frame([1000.0 + i for i in range(60)])
        extra = base.with_columns(pl.lit(1.0).alias("only_on_b"))

        stream = prepare({"A": base, "B": extra}, config, "fortress")
        assert "only_on_b" in stream.columns
        assert stream.height == base.height * 2
