#!/usr/bin/env python3
"""
Writing your own strategy, and running it through the engine.

    python examples/full_system.py          # backtest the custom strategy offline

This file used to be a 300-line reference implementation whose helper functions
returned `True  # Placeholder`. That was honest about being a sketch, but it
meant the one thing you actually want to copy — a working strategy — was the one
thing missing. The engine now supplies the plumbing, so this shows the part that
is genuinely yours to write.

A strategy is a pure function:

    evaluate(row, symbol, config, tick_size) -> Signal | None

`row` is one bar with indicators attached. No clock, no globals, no broker. That
purity is what lets the same function run in a backtest over 2024 and in live
trading at 09:47 tomorrow without a single branch between them.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl

from engine import backtest
from engine.analytics import format_report
from engine.broker import round_to_tick
from engine.indicators import add_indicators, is_warm
from engine.signals import PREFILTERS, STRATEGIES, Signal


# ============================================================================
# A custom strategy: opening-range breakout, confirmed by VWAP and volume.
# KNOWLEDGE.md section 9.2 ranks VWAP the most reliable intraday level, and an
# ORB without volume confirmation is the textbook false breakout.
# ============================================================================

def opening_range_breakout(
    row: dict, symbol: str, config: dict, tick_size: float = 0.05
) -> Signal | None:
    """
    Long when price breaks the opening range high with volume, above VWAP.

    Requires an `or_high` column — see `add_opening_range` below. Returning None
    is the normal case; a strategy that signals often is usually a strategy that
    has stopped filtering.
    """
    if not is_warm(row) or row.get("or_high") is None:
        return None

    close = float(row["close"])
    range_high = float(row["or_high"])

    # 1. The breakout itself.
    if close <= range_high:
        return None

    # 2. Volume confirmation. An unconfirmed ORB is a fake more often than not.
    volume_ratio = float(row.get("volume_ratio") or 0)
    if volume_ratio < config.get("orb_volume_mult", 2.0):
        return None

    # 3. VWAP agreement — institutional participation on the same side.
    if close <= float(row["vwap"]):
        return None

    # 4. Trend strength (NUANCE #8: ADX filters, it does not point).
    if float(row["adx"]) < config.get("adx_min", 25):
        return None

    factors = {
        "breakout": 0.40,
        "volume": min(0.25, 0.125 * volume_ratio),
        "vwap": 0.20,
        "strength": 0.15,
    }
    confidence = sum(factors.values())

    atr_value = float(row["atr"])
    if atr_value <= 0:
        return None

    entry = round_to_tick(close, tick_size)
    # Stop below the range, not an arbitrary ATR multiple: if price re-enters the
    # range the breakout has failed and the reason for the trade is gone.
    stop_loss = round_to_tick(min(range_high, entry - atr_value * 1.2), tick_size)
    if stop_loss >= entry:
        return None

    target = round_to_tick(entry + (entry - stop_loss) * config.get("risk_reward", 1.5),
                           tick_size)

    return Signal(
        symbol=symbol,
        direction="LONG",
        entry_price=entry,
        stop_loss=stop_loss,
        target=target,
        confidence=round(confidence, 3),
        reason=f"ORB over {range_high:.2f} on {volume_ratio:.1f}x volume, above VWAP",
        strategy="ORB",
        timestamp=row["timestamp"],
        indicators={k: float(row[k]) for k in ("rsi", "adx", "vwap", "atr")
                    if row.get(k) is not None},
        factors=factors,
    )


def orb_prefilter(config: dict) -> pl.Expr:
    """
    Vectorised form of the hard gates, for backtest speed.

    MUST NOT reject anything `opening_range_breakout` would accept — a prefilter
    that is not a superset makes the backtest skip trades live would take.
    tests/test_signals_and_risk.py proves this property for every registered
    strategy, so register yours before trusting a backtest of it.
    """
    return (
        (pl.col("close") > pl.col("or_high"))
        & (pl.col("volume_ratio") >= config.get("orb_volume_mult", 2.0))
        & (pl.col("close") > pl.col("vwap"))
        & (pl.col("adx") >= config.get("adx_min", 25))
    )


# Registering makes the strategy available to the CLI, the runner, and the
# parity tests — `./run.sh backtest --strategy orb` works after this.
STRATEGIES["orb"] = opening_range_breakout
PREFILTERS["orb"] = orb_prefilter


def add_opening_range(df: pl.DataFrame, minutes: int = 15) -> pl.DataFrame:
    """
    High and low of the first N minutes of each session.

    The window is per session_date, so it resets daily for the same reason VWAP
    does. Bars inside the opening range get a null `or_high`, which makes them
    untradeable rather than comparing against a half-formed range.
    """
    from engine.session import MARKET_OPEN

    open_minutes = MARKET_OPEN.hour * 60 + MARKET_OPEN.minute
    # Cast before multiplying: dt.hour() is Int8, and 9 * 60 = 540 overflows it,
    # wrapping to 28. The resulting comparison is silently true for every row,
    # which nulls the whole opening range instead of raising.
    minute_of_day = (
        pl.col("timestamp").dt.hour().cast(pl.Int32) * 60
        + pl.col("timestamp").dt.minute().cast(pl.Int32)
    )
    in_range = (minute_of_day - open_minutes) < minutes

    return df.with_columns(
        [
            pl.when(in_range).then(pl.col("high")).otherwise(None)
            .max().over("session_date").alias("or_high"),
            pl.when(in_range).then(pl.col("low")).otherwise(None)
            .min().over("session_date").alias("or_low"),
        ]
    ).with_columns(
        # Null out the range itself so no bar can break out of a range it is in.
        pl.when(in_range).then(None).otherwise(pl.col("or_high")).alias("or_high"),
        pl.when(in_range).then(None).otherwise(pl.col("or_low")).alias("or_low"),
    )


def _synthetic_session(days: int = 12, seed: int = 11) -> pl.DataFrame:
    """Offline data so this example runs without a broker."""
    import random

    random.seed(seed)
    rows, price = [], 1000.0
    start = datetime(2026, 5, 4, 9, 15)
    day = 0

    while len({r["timestamp"].date() for r in rows}) < days:
        session = start + timedelta(days=day)
        day += 1
        if session.weekday() >= 5:
            continue

        opening_drift = random.choice([0.00015, -0.00012, 0.00002])
        for minute in range(375):
            price *= 1 + random.gauss(opening_drift if minute < 90 else 0.0, 0.0011)
            open_price = price
            close_price = price * (1 + random.gauss(0, 0.0007))
            rows.append({
                "timestamp": session.replace(hour=9, minute=15) + timedelta(minutes=minute),
                "open": round(open_price, 2),
                "high": round(max(open_price, close_price) * 1.0006, 2),
                "low": round(min(open_price, close_price) * 0.9994, 2),
                "close": round(close_price, 2),
                "volume": random.randint(15_000, 60_000) * (4 if 15 <= minute < 45 else 1),
            })

    return pl.DataFrame(rows)


def main() -> int:
    print(__doc__.strip().split("\n\n")[0])
    print("\nBacktesting the custom ORB strategy on synthetic data...\n")

    config = {
        "orb_volume_mult": 2.0, "adx_min": 20, "risk_reward": 1.5,
        "risk_pct": 1.0, "max_positions": 3, "max_hold_minutes": 60,
        "symbol_cooldown_minutes": 45, "daily_loss_limit_pct": 3.0,
        "max_consecutive_losses": 3, "max_portfolio_heat_pct": 5.0,
    }

    frame = add_opening_range(add_indicators(_synthetic_session()))

    result = backtest.run(
        {"SYNTH": frame}, config, strategy="orb", starting_capital=1_000_000
    )

    print(format_report(result.metrics))
    print("\nWhy signals were not taken:")
    for reason, count in list(result.rejections.items())[:5]:
        print(f"    {reason:45s} {count:>6,}")

    print(
        "\nThis is random data, so the result is noise plus costs — that is the"
        "\ncorrect outcome and a useful sanity check: a backtest that finds edge"
        "\nin a random walk is a broken backtest."
        "\n\nTo run it on real data:  ./run.sh backtest --strategy orb --start 2026-01-01"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
