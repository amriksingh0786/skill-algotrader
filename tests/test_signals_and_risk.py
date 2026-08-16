"""
Signal purity, prefilter correctness, and every risk limit.

The prefilter test is load-bearing for backtest/live parity: if the vectorised
prefilter ever rejects a row that `evaluate` would have accepted, the backtest
silently skips trades the live bot takes.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from engine.broker import round_to_tick
from engine.indicators import add_indicators
from engine.risk import RiskManager, RiskState, detect_market_regime, kelly_fraction
from engine.session import IST
from engine.signals import (
    PREFILTERS,
    STRATEGIES,
    Signal,
    add_strategy_columns,
    fortress,
    get_strategy,
)
from tests.conftest import make_candles


def _row(**overrides) -> dict:
    """A row that passes every fortress gate, before overrides."""
    base = {
        "timestamp": datetime(2026, 5, 4, 10, 0, tzinfo=IST),
        "open": 1000.0, "high": 1006.0, "low": 999.0, "close": 1005.0, "volume": 200_000,
        "ema_fast": 1004.0, "ema_slow": 1000.0, "rsi": 55.0, "adx": 30.0,
        "plus_di": 30.0, "minus_di": 15.0, "atr": 5.0, "vwap": 1002.0,
        "avg_volume": 100_000.0, "volume_ratio": 2.0, "vwap_distance_pct": 0.3,
        "body_pct": 0.005, "atr_pct": 0.5,
    }
    base.update(overrides)
    return base


class TestSignalPurity:
    def test_same_row_gives_same_signal(self, config: dict) -> None:
        """Pure function: no clock, no globals, no hidden state."""
        row = _row()
        first = fortress(row, "TEST", config)
        second = fortress(row, "TEST", config)
        assert first is not None
        assert first.to_dict() == second.to_dict()

    def test_signal_does_not_read_the_wall_clock(self, config: dict) -> None:
        """
        The signal timestamp must come from the candle, not datetime.now().
        A signal function that reads the clock cannot be backtested.
        """
        candle_time = datetime(2020, 1, 1, 10, 0, tzinfo=IST)
        signal = fortress(_row(timestamp=candle_time), "TEST", config)
        assert signal is not None
        assert signal.timestamp == candle_time


class TestFortressGates:
    def test_accepts_a_clean_setup(self, config: dict) -> None:
        signal = fortress(_row(), "TEST", config)
        assert signal is not None
        assert signal.direction == "LONG"
        assert signal.stop_loss < signal.entry_price < signal.target
        assert signal.confidence >= config["min_confidence"]

    def test_rejects_downtrend(self, config: dict) -> None:
        assert fortress(_row(ema_fast=999.0, ema_slow=1000.0), "TEST", config) is None

    def test_rejects_weak_trend(self, config: dict) -> None:
        """NUANCE #8: ADX below threshold means chop."""
        assert fortress(_row(adx=15.0), "TEST", config) is None

    @pytest.mark.parametrize("rsi", [30.0, 44.9, 65.1, 80.0])
    def test_rejects_rsi_outside_band(self, config: dict, rsi: float) -> None:
        assert fortress(_row(rsi=rsi), "TEST", config) is None

    def test_rejects_when_not_warm(self, config: dict) -> None:
        assert fortress(_row(adx=None), "TEST", config) is None

    def test_soft_factors_only_affect_confidence(self, config: dict) -> None:
        strong = fortress(_row(), "TEST", config)
        weak = fortress(_row(close=1001.0, vwap=1002.0, volume=100_000), "TEST", config)
        assert strong is not None
        if weak is not None:
            assert weak.confidence < strong.confidence

    def test_confidence_floor_enforced(self) -> None:
        tight = {"rsi_long_min": 45, "rsi_long_max": 65, "adx_min": 25,
                 "volume_mult": 1.5, "min_confidence": 0.95}
        assert fortress(_row(), "TEST", tight) is None

    def test_prices_are_tick_aligned(self, config: dict) -> None:
        """NUANCE #1: every price the engine emits must sit on the tick grid."""
        signal = fortress(_row(close=1005.37, atr=5.13), "TEST", config, tick_size=0.05)
        assert signal is not None
        for price in (signal.entry_price, signal.stop_loss, signal.target):
            assert price == round_to_tick(price, 0.05)
            assert abs(round(price / 0.05) * 0.05 - price) < 1e-6

    def test_records_factor_attribution(self, config: dict) -> None:
        signal = fortress(_row(), "TEST", config)
        assert signal is not None
        assert {"trend", "strength", "momentum"} <= set(signal.factors)
        assert sum(signal.factors.values()) == pytest.approx(signal.confidence, abs=1e-9)

    def test_risk_reward_matches_config(self, config: dict) -> None:
        signal = fortress(_row(), "TEST", config)
        assert signal is not None
        assert signal.risk_reward == pytest.approx(config["risk_reward"], rel=0.05)


class TestPrefilterIsSuperset:
    """
    For every strategy: evaluate() accepting a row implies the prefilter kept it.

    A violation means the backtest skips trades live would take — the parity bug
    this design exists to prevent.
    """

    @pytest.mark.parametrize("strategy_name", sorted(STRATEGIES))
    def test_prefilter_never_drops_a_tradeable_row(self, strategy_name: str,
                                                   config: dict) -> None:
        frame = add_strategy_columns(add_indicators(make_candles(1500, seed=7)), config)
        prefilter = PREFILTERS[strategy_name]
        evaluate = STRATEGIES[strategy_name]

        marked = frame.with_columns(
            prefilter(config).fill_null(False).alias("_candidate")
        )

        violations = 0
        accepted = 0
        for row in marked.iter_rows(named=True):
            signal = evaluate(row, "TEST", config)
            if signal is None:
                continue
            accepted += 1
            if not row["_candidate"]:
                violations += 1

        assert violations == 0, (
            f"{strategy_name}: prefilter dropped {violations} rows that evaluate() accepted"
        )

    @pytest.mark.parametrize("strategy_name", sorted(STRATEGIES))
    def test_prefilter_actually_filters(self, strategy_name: str, config: dict) -> None:
        """A prefilter that keeps everything is correct but useless."""
        frame = add_strategy_columns(add_indicators(make_candles(1500, seed=7)), config)
        marked = frame.with_columns(
            PREFILTERS[strategy_name](config).fill_null(False).alias("_candidate")
        )
        kept = marked["_candidate"].sum()
        assert kept < marked.height, f"{strategy_name} prefilter kept every row"

    def test_unknown_strategy_raises(self) -> None:
        with pytest.raises(KeyError, match="unknown strategy"):
            get_strategy("does_not_exist")


class TestStrategyColumns:
    def test_rolling_high_excludes_current_bar(self, config: dict) -> None:
        """
        Off-by-one here makes breakouts undetectable: a bar can never exceed a
        window that contains its own high.
        """
        rows = [
            {"timestamp": datetime(2026, 5, 4, 9, 15) + timedelta(minutes=i),
             "open": 100.0, "high": 100.0 + (50 if i == 30 else 0), "low": 99.0,
             "close": 100.0, "volume": 1000}
            for i in range(35)
        ]
        result = add_strategy_columns(pl.DataFrame(rows), {"breakout_period": 20})
        assert result.row(30, named=True)["rolling_high"] == pytest.approx(100.0)


class TestPositionSizing:
    def test_size_derives_from_stop_distance(self, config: dict) -> None:
        """
        NUANCE #19: risk a fixed fraction, not a fixed rupee amount.

        max_position_pct is lifted here so the risk budget is the binding
        constraint — the position cap has its own test below.
        """
        risk = RiskManager({**config, "max_position_pct": 100.0})
        signal = Signal(
            symbol="TEST", direction="LONG", entry_price=1000.0, stop_loss=990.0,
            target=1015.0, confidence=0.6, reason="", strategy="TEST",
            timestamp=datetime(2026, 5, 4, 10, 0, tzinfo=IST),
        )
        result = risk.size_position(signal, capital=1_000_000, available_margin=10_000_000)
        # 1% of 1,000,000 = 10,000 budget; 10 rupees of risk per share -> 1000 shares
        assert result.quantity == 1000
        assert result.binding_constraint == "risk_budget"
        assert result.capital_at_risk == pytest.approx(10_000, rel=0.01)

    def test_wider_stop_gives_smaller_size(self, config: dict) -> None:
        risk = RiskManager({**config, "max_position_pct": 100.0})
        base = dict(symbol="TEST", direction="LONG", entry_price=100.0, target=103.0,
                    confidence=0.6, reason="", strategy="TEST",
                    timestamp=datetime(2026, 5, 4, 10, 0, tzinfo=IST))
        tight = risk.size_position(Signal(stop_loss=98.0, **base),
                                   capital=1_000_000, available_margin=10_000_000)
        wide = risk.size_position(Signal(stop_loss=92.0, **base),
                                  capital=1_000_000, available_margin=10_000_000)
        assert tight.binding_constraint == wide.binding_constraint == "risk_budget"
        assert tight.quantity > wide.quantity
        # Both risk the same rupees — that is the point of the method.
        assert tight.capital_at_risk == pytest.approx(wide.capital_at_risk, rel=0.02)

    def test_position_cap_binds(self, config: dict) -> None:
        risk = RiskManager({**config, "max_position_pct": 5.0})
        signal = Signal(symbol="T", direction="LONG", entry_price=1000.0, stop_loss=999.0,
                        target=1005.0, confidence=0.6, reason="", strategy="T",
                        timestamp=datetime(2026, 5, 4, 10, 0, tzinfo=IST))
        result = risk.size_position(signal, capital=1_000_000, available_margin=1_000_000)
        assert result.binding_constraint == "position_cap"
        assert result.position_value <= 50_000 * 1.01

    def test_margin_binds(self, config: dict) -> None:
        risk = RiskManager(config)
        signal = Signal(symbol="T", direction="LONG", entry_price=1000.0, stop_loss=990.0,
                        target=1015.0, confidence=0.6, reason="", strategy="T",
                        timestamp=datetime(2026, 5, 4, 10, 0, tzinfo=IST))
        result = risk.size_position(signal, capital=1_000_000, available_margin=50_000)
        assert result.binding_constraint == "margin"
        assert result.quantity <= 48

    def test_zero_stop_distance_is_rejected(self, config: dict) -> None:
        risk = RiskManager(config)
        signal = Signal(symbol="T", direction="LONG", entry_price=1000.0, stop_loss=1000.0,
                        target=1010.0, confidence=0.6, reason="", strategy="T",
                        timestamp=datetime(2026, 5, 4, 10, 0, tzinfo=IST))
        result = risk.size_position(signal, capital=1_000_000, available_margin=1_000_000)
        assert result.quantity == 0 and not result.is_tradeable

    def test_max_risk_pct_caps_regime_boost(self, config: dict) -> None:
        """A bull-market multiplier must not push risk past the hard ceiling."""
        risk = RiskManager({**config, "risk_pct": 1.5, "max_risk_pct": 2.0})
        signal = Signal(symbol="T", direction="LONG", entry_price=1000.0, stop_loss=990.0,
                        target=1015.0, confidence=0.6, reason="", strategy="T",
                        timestamp=datetime(2026, 5, 4, 10, 0, tzinfo=IST))
        boosted = risk.size_position(signal, capital=1_000_000, available_margin=10_000_000,
                                     regime_multiplier=3.0)
        assert boosted.capital_at_risk <= 20_000 * 1.01


class TestRiskLimits:
    def _signal(self, symbol: str = "TEST") -> Signal:
        return Signal(symbol=symbol, direction="LONG", entry_price=1000.0, stop_loss=990.0,
                      target=1015.0, confidence=0.6, reason="", strategy="TEST",
                      timestamp=datetime(2026, 5, 4, 10, 0, tzinfo=IST))

    def test_cooldown_blocks_reentry(self, config: dict) -> None:
        """NUANCE #5: the HINDALCO loop."""
        risk = RiskManager(config)
        now = datetime(2026, 5, 4, 10, 0, tzinfo=IST)
        risk.record_exit("TEST", -500.0, now)

        blocked, reason = risk.can_enter(self._signal(), open_positions=[],
                                         capital=1_000_000, now=now + timedelta(minutes=10))
        assert not blocked and "cooling down" in reason

        allowed, _ = risk.can_enter(self._signal(), open_positions=[],
                                    capital=1_000_000, now=now + timedelta(minutes=46))
        assert allowed

    def test_consecutive_losses_halt_trading(self, config: dict) -> None:
        """NUANCE #20."""
        risk = RiskManager(config)
        now = datetime(2026, 5, 4, 10, 0, tzinfo=IST)
        for index in range(3):
            risk.record_exit(f"SYM{index}", -1000.0, now)

        allowed, reason = risk.can_enter(self._signal("FRESH"), open_positions=[],
                                         capital=1_000_000, now=now)
        assert not allowed and "consecutive losses" in reason
        assert risk.state.halted

    def test_a_win_resets_the_streak(self, config: dict) -> None:
        risk = RiskManager(config)
        now = datetime(2026, 5, 4, 10, 0, tzinfo=IST)
        risk.record_exit("A", -100.0, now)
        risk.record_exit("B", -100.0, now)
        risk.record_exit("C", +500.0, now)
        assert risk.state.consecutive_losses == 0

    def test_daily_loss_limit_halts(self, config: dict) -> None:
        risk = RiskManager(config)
        now = datetime(2026, 5, 4, 10, 0, tzinfo=IST)
        risk.record_exit("A", -31_000.0, now)  # >3% of 1,000,000

        allowed, reason = risk.can_enter(self._signal("FRESH"), open_positions=[],
                                         capital=1_000_000, now=now)
        assert not allowed and "daily loss limit" in reason

    def test_position_limit(self, config: dict) -> None:
        risk = RiskManager(config)

        class Held:
            def __init__(self, symbol): self.symbol, self.value, self.risk_amount = symbol, 100.0, 10.0

        held = [Held(f"S{i}") for i in range(5)]
        allowed, reason = risk.can_enter(self._signal(), open_positions=held,
                                         capital=1_000_000)
        assert not allowed and "position limit" in reason

    def test_portfolio_heat_limit(self, config: dict) -> None:
        risk = RiskManager(config)

        class Held:
            def __init__(self): self.symbol, self.value, self.risk_amount = "X", 1000.0, 45_000.0

        allowed, reason = risk.can_enter(self._signal(), open_positions=[Held()],
                                         capital=1_000_000)
        assert not allowed and "heat" in reason

    def test_duplicate_symbol_blocked(self, config: dict) -> None:
        risk = RiskManager(config)

        class Held:
            symbol, value, risk_amount = "TEST", 1000.0, 100.0

        allowed, reason = risk.can_enter(self._signal("TEST"), open_positions=[Held()],
                                         capital=1_000_000)
        assert not allowed and "already holding" in reason

    def test_min_risk_reward_enforced(self, config: dict) -> None:
        risk = RiskManager({**config, "min_risk_reward": 2.0})
        signal = Signal(symbol="T", direction="LONG", entry_price=1000.0, stop_loss=990.0,
                        target=1005.0, confidence=0.6, reason="", strategy="T",
                        timestamp=datetime(2026, 5, 4, 10, 0, tzinfo=IST))
        allowed, reason = risk.can_enter(signal, open_positions=[], capital=1_000_000)
        assert not allowed and "risk:reward" in reason

    def test_day_rollover_resets_limits(self, config: dict) -> None:
        """A new session clears the daily halt — but not the cooldown map."""
        risk = RiskManager(config)
        day_one = datetime(2026, 5, 4, 15, 0, tzinfo=IST)
        for index in range(3):
            risk.record_exit(f"S{index}", -1000.0, day_one)
        risk.can_enter(self._signal("X"), open_positions=[], capital=1_000_000, now=day_one)
        assert risk.state.halted

        risk.roll_day_if_needed(datetime(2026, 5, 5, 9, 30, tzinfo=IST))
        assert not risk.state.halted
        assert risk.state.consecutive_losses == 0

    def test_state_round_trips_through_json(self, config: dict) -> None:
        """Restarting must not reset the daily loss limit."""
        risk = RiskManager(config)
        risk.record_exit("A", -5000.0, datetime(2026, 5, 4, 10, 0, tzinfo=IST))
        restored = RiskState.from_dict(risk.state.to_dict())
        assert restored.realised_pnl == risk.state.realised_pnl
        assert restored.consecutive_losses == risk.state.consecutive_losses
        assert restored.cooldowns == risk.state.cooldowns


class TestKelly:
    def test_positive_edge(self) -> None:
        result = kelly_fraction(win_rate=0.65, avg_win=0.012, avg_loss=0.006)
        assert result["full_kelly"] > 0
        assert result["half_kelly"] == pytest.approx(result["full_kelly"] / 2)
        assert result["recommended"] <= 0.25

    def test_negative_edge_gives_zero(self) -> None:
        result = kelly_fraction(win_rate=0.30, avg_win=0.005, avg_loss=0.010)
        assert result["full_kelly"] == 0.0

    def test_degenerate_inputs(self) -> None:
        assert kelly_fraction(0.5, 0.0, 0.01)["full_kelly"] == 0.0
        assert kelly_fraction(1.5, 0.01, 0.01)["full_kelly"] == 0.0

    def test_recommendation_is_capped(self) -> None:
        """Even an absurd edge must not size past a quarter of capital."""
        result = kelly_fraction(win_rate=0.95, avg_win=0.10, avg_loss=0.001)
        assert result["recommended"] <= 0.25


class TestMarketRegime:
    def _daily(self, closes: list[float]) -> pl.DataFrame:
        return pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(len(closes))],
            "open": closes, "high": closes, "low": closes, "close": closes,
            "volume": [1000] * len(closes),
        })

    def test_bull_detected(self) -> None:
        closes = [100 + i * 0.5 for i in range(300)]
        result = detect_market_regime(self._daily(closes))
        assert result["regime"] == "BULL"
        assert result["multiplier"] > 1.0

    def test_bear_detected(self) -> None:
        closes = [250 - i * 0.5 for i in range(300)]
        result = detect_market_regime(self._daily(closes))
        assert result["regime"] == "BEAR"
        assert result["multiplier"] < 1.0

    def test_insufficient_history_is_cautious(self) -> None:
        result = detect_market_regime(self._daily([100.0] * 50))
        assert result["regime"] == "UNKNOWN"
        assert result["multiplier"] <= 0.5
