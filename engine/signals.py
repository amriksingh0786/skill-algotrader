"""
Signal generation — pure functions over a single indicator-enriched row.

DESIGN NOTE (deviation from examples/full_system.py, deliberate)
---------------------------------------------------------------
The reference example calls `can_trade_symbol()` and `should_trade_now()` from
inside the signal function, reading global cooldown state and the wall clock.
That makes signals untestable and, worse, unbacktestable: the backtest would
consult `datetime.now()` while replaying 2025 data.

Here signal functions are pure — row in, Signal or None out. Cooldown, session
timing, and portfolio limits are applied by the caller (`risk.py` / `runner.py`),
which is what lets backtest and live run the identical evaluation. The guard
rails are not weaker; they moved to the layer that owns the state.

Every strategy exposes:
    evaluate(row, symbol, config, tick_size) -> Signal | None
    prefilter(config) -> pl.Expr   # vectorised superset of evaluate's hard gates
`prefilter` exists only to skip rows in backtests. It must never reject a row
that `evaluate` would have accepted, or backtest and live diverge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

import polars as pl

from .broker import round_to_tick
from .indicators import is_warm


@dataclass
class Signal:
    """
    A trade proposal with full attribution.

    `factors` records what each contributing condition was worth. KNOWLEDGE.md
    section on signal attribution: without it you cannot tell whether a losing
    month came from the volume filter decaying or the trend filter misfiring, and
    you end up tuning at random.
    """

    symbol: str
    direction: str  # LONG / SHORT
    entry_price: float
    stop_loss: float
    target: float
    confidence: float
    reason: str
    strategy: str
    timestamp: datetime
    indicators: dict[str, float] = field(default_factory=dict)
    factors: dict[str, float] = field(default_factory=dict)

    @property
    def risk_per_share(self) -> float:
        return abs(self.entry_price - self.stop_loss)

    @property
    def reward_per_share(self) -> float:
        return abs(self.target - self.entry_price)

    @property
    def risk_reward(self) -> float:
        return self.reward_per_share / self.risk_per_share if self.risk_per_share else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "target": self.target,
            "confidence": self.confidence,
            "reason": self.reason,
            "strategy": self.strategy,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "risk_reward": round(self.risk_reward, 2),
            "indicators": {k: round(float(v), 4) for k, v in self.indicators.items()},
            "factors": self.factors,
        }


def _levels(
    entry: float, atr_value: float, direction: str, config: dict, tick: float
) -> tuple[float, float]:
    """Stop and target from ATR. NUANCE #1: both are tick-aligned before use."""
    sl_multiplier = config.get("sl_atr_mult", 1.2)
    risk_reward = config.get("risk_reward", 1.5)
    distance = atr_value * sl_multiplier

    if direction == "LONG":
        stop_loss, target = entry - distance, entry + distance * risk_reward
    else:
        stop_loss, target = entry + distance, entry - distance * risk_reward

    return round_to_tick(stop_loss, tick), round_to_tick(target, tick)


def _snapshot(row: dict) -> dict[str, float]:
    """Indicator values recorded on the signal for later attribution."""
    keys = ("ema_fast", "ema_slow", "rsi", "adx", "plus_di", "minus_di", "vwap",
            "atr", "volume", "avg_volume", "volume_ratio", "vwap_distance_pct", "atr_pct")
    return {k: float(row[k]) for k in keys if row.get(k) is not None}


# ---------------------------------------------------------------- fortress

def fortress(row: dict, symbol: str, config: dict, tick_size: float = 0.05) -> Signal | None:
    """
    Fortress: six-factor confirmation, the highest win rate strategy in
    KNOWLEDGE.md section 3.

    Three hard gates reject outright (trend, strength, momentum band); three soft
    factors only add confidence (VWAP, volume, candle body). NUANCE #24: it is
    the confluence that carries the edge — a single indicator in isolation
    produces roughly coin-flip entries on this universe.

    Returns None unless total confidence clears `min_confidence`.
    """
    if not is_warm(row):
        return None

    close = float(row["close"])
    if close <= 0:
        return None

    factors: dict[str, float] = {}

    # HARD GATE 1 — trend direction. Never fight the EMA stack.
    if not row["ema_fast"] > row["ema_slow"]:
        return None
    factors["trend"] = 0.25

    # HARD GATE 2 — trend strength (NUANCE #8: ADX is a filter, not a direction).
    if not row["adx"] > config.get("adx_min", 25):
        return None
    factors["strength"] = 0.15

    # HARD GATE 3 — momentum band. Above the top of the band the move is already
    # extended and the stop sits too far away to size sensibly.
    rsi_value = float(row["rsi"])
    if not (config.get("rsi_long_min", 45) <= rsi_value <= config.get("rsi_long_max", 65)):
        return None
    factors["momentum"] = 0.10

    confidence = sum(factors.values())

    # SOFT FACTOR — VWAP confluence (NUANCE #4 makes this meaningful).
    above_vwap = close > float(row["vwap"])
    if above_vwap:
        factors["vwap"] = 0.15
        confidence += 0.15

    # SOFT FACTOR — participation.
    volume_surge = float(row["volume"]) > float(row["avg_volume"]) * config.get("volume_mult", 1.5)
    if volume_surge:
        factors["volume"] = 0.10
        confidence += 0.10

    # SOFT FACTOR — conviction in the candle itself (a doji is indecision).
    if float(row.get("body_pct") or 0) > config.get("min_body_pct", 0.0025):
        factors["body"] = 0.05
        confidence += 0.05

    if confidence < config.get("min_confidence", 0.50):
        return None

    atr_value = float(row["atr"])
    if atr_value <= 0:
        return None

    entry = round_to_tick(close, tick_size)
    stop_loss, target = _levels(entry, atr_value, "LONG", config, tick_size)
    if stop_loss >= entry:  # degenerate ATR
        return None

    reason = f"EMA{'↑'} + ADX {row['adx']:.0f} + RSI {rsi_value:.0f}"
    if above_vwap:
        reason += " + above VWAP"
    if volume_surge:
        reason += f" + {float(row['volume_ratio']):.1f}x volume"

    return Signal(
        symbol=symbol,
        direction="LONG",
        entry_price=entry,
        stop_loss=stop_loss,
        target=target,
        confidence=round(confidence, 3),
        reason=reason,
        strategy="FORTRESS",
        timestamp=row["timestamp"],
        indicators=_snapshot(row),
        factors=factors,
    )


def fortress_prefilter(config: dict) -> pl.Expr:
    """Vectorised form of the three hard gates only."""
    return (
        (pl.col("ema_fast") > pl.col("ema_slow"))
        & (pl.col("adx") > config.get("adx_min", 25))
        & (pl.col("rsi") >= config.get("rsi_long_min", 45))
        & (pl.col("rsi") <= config.get("rsi_long_max", 65))
        & pl.col("atr").is_not_null()
        & (pl.col("atr") > 0)
        & pl.col("avg_volume").is_not_null()
    )


# ---------------------------------------------------------------- momentum

def momentum(row: dict, symbol: str, config: dict, tick_size: float = 0.05) -> Signal | None:
    """
    Breakout continuation: price clears the recent high on expanding volume.

    Requires a `rolling_high` column (see `add_strategy_columns`). KNOWLEDGE.md
    section 5: this fits midcaps, where breakouts tend to follow through;
    largecaps mean-revert against it more often than not.
    """
    if not is_warm(row) or row.get("rolling_high") is None:
        return None

    close = float(row["close"])
    breakout_level = float(row["rolling_high"])

    if close <= breakout_level:
        return None
    if not row["ema_fast"] > row["ema_slow"]:
        return None
    if not row["adx"] > config.get("adx_min", 25):
        return None

    volume_ratio = float(row.get("volume_ratio") or 0)
    if volume_ratio < config.get("volume_mult", 2.0):
        return None  # an unconfirmed breakout is usually a fake

    # RSI ceiling: buying a vertical move leaves no room before exhaustion.
    if float(row["rsi"]) > config.get("rsi_max", 75):
        return None

    factors = {
        "breakout": 0.35,
        "trend": 0.20,
        "strength": 0.15,
        "volume": min(0.20, 0.10 * volume_ratio / config.get("volume_mult", 2.0)),
    }
    confidence = sum(factors.values())

    atr_value = float(row["atr"])
    if atr_value <= 0:
        return None

    entry = round_to_tick(close, tick_size)
    stop_loss, target = _levels(entry, atr_value, "LONG", config, tick_size)

    return Signal(
        symbol=symbol,
        direction="LONG",
        entry_price=entry,
        stop_loss=stop_loss,
        target=target,
        confidence=round(confidence, 3),
        reason=f"breakout over {breakout_level:.2f} on {volume_ratio:.1f}x volume",
        strategy="MOMENTUM",
        timestamp=row["timestamp"],
        indicators=_snapshot(row),
        factors=factors,
    )


def momentum_prefilter(config: dict) -> pl.Expr:
    return (
        (pl.col("close") > pl.col("rolling_high"))
        & (pl.col("ema_fast") > pl.col("ema_slow"))
        & (pl.col("adx") > config.get("adx_min", 25))
        & (pl.col("volume_ratio") >= config.get("volume_mult", 2.0))
        & (pl.col("rsi") <= config.get("rsi_max", 75))
    )


# ----------------------------------------------------------- mean reversion

def mean_reversion(row: dict, symbol: str, config: dict, tick_size: float = 0.05) -> Signal | None:
    """
    VWAP pullback: buy a dip below VWAP while the larger trend still points up.

    KNOWLEDGE.md section 9 ranks VWAP the most reliable intraday level, and
    section 5 notes largecaps pull back to it rather than trending away.
    The trend gate is what separates a pullback from a reversal.
    """
    if not is_warm(row):
        return None

    close = float(row["close"])
    distance_pct = float(row.get("vwap_distance_pct") or 0)

    # Below VWAP but not in free fall.
    band_min = config.get("vwap_pullback_min_pct", -1.5)
    band_max = config.get("vwap_pullback_max_pct", -0.15)
    if not (band_min <= distance_pct <= band_max):
        return None

    # The trend must still be intact, or this is a falling knife.
    if not row["ema_fast"] > row["ema_slow"]:
        return None

    rsi_value = float(row["rsi"])
    if not (config.get("rsi_oversold_min", 30) <= rsi_value <= config.get("rsi_oversold_max", 50)):
        return None

    factors = {
        "vwap_pullback": 0.35,
        "trend_intact": 0.25,
        "oversold": 0.20,
    }
    confidence = sum(factors.values())

    if float(row.get("volume_ratio") or 0) > 1.2:
        factors["volume"] = 0.10
        confidence += 0.10

    atr_value = float(row["atr"])
    if atr_value <= 0:
        return None

    entry = round_to_tick(close, tick_size)
    stop_loss, _ = _levels(entry, atr_value, "LONG", config, tick_size)
    # Mean reversion targets the level it reverted from, not an ATR multiple.
    target = round_to_tick(float(row["vwap"]) * config.get("vwap_target_mult", 1.001), tick_size)

    if target <= entry:
        return None

    return Signal(
        symbol=symbol,
        direction="LONG",
        entry_price=entry,
        stop_loss=stop_loss,
        target=target,
        confidence=round(confidence, 3),
        reason=f"{abs(distance_pct):.2f}% below VWAP, trend intact, RSI {rsi_value:.0f}",
        strategy="MEAN_REVERSION",
        timestamp=row["timestamp"],
        indicators=_snapshot(row),
        factors=factors,
    )


def mean_reversion_prefilter(config: dict) -> pl.Expr:
    return (
        (pl.col("vwap_distance_pct") >= config.get("vwap_pullback_min_pct", -1.5))
        & (pl.col("vwap_distance_pct") <= config.get("vwap_pullback_max_pct", -0.15))
        & (pl.col("ema_fast") > pl.col("ema_slow"))
        & (pl.col("rsi") >= config.get("rsi_oversold_min", 30))
        & (pl.col("rsi") <= config.get("rsi_oversold_max", 50))
    )


STRATEGIES: dict[str, Callable[..., Signal | None]] = {
    "fortress": fortress,
    "momentum": momentum,
    "mean_reversion": mean_reversion,
}

PREFILTERS: dict[str, Callable[[dict], pl.Expr]] = {
    "fortress": fortress_prefilter,
    "momentum": momentum_prefilter,
    "mean_reversion": mean_reversion_prefilter,
}


def get_strategy(name: str) -> Callable[..., Signal | None]:
    if name not in STRATEGIES:
        raise KeyError(f"unknown strategy {name!r}. Available: {', '.join(STRATEGIES)}")
    return STRATEGIES[name]


def add_strategy_columns(df: pl.DataFrame, config: dict) -> pl.DataFrame:
    """
    Columns needed by specific strategies but not by the shared indicator set.

    `rolling_high` excludes the current bar — comparing a bar's close against a
    window that contains its own high makes a breakout impossible to detect, a
    subtle off-by-one that silently yields zero momentum signals.
    """
    lookback = config.get("breakout_period", 20)
    return df.with_columns(
        [
            pl.col("high").shift(1).rolling_max(window_size=lookback).alias("rolling_high"),
            pl.col("low").shift(1).rolling_min(window_size=lookback).alias("rolling_low"),
        ]
    )
