"""
Vectorised indicators (Polars).

Formulas follow KNOWLEDGE.md section 9 exactly. RSI, ATR and ADX all use Wilder
smoothing (an EWM with alpha = 1/period), which is what the recursive loops in
that document compute — expressed here as a single vectorised expression instead
of a Python loop (KNOWLEDGE.md section 6: ~37x faster).

NUANCE #13: compute these ONCE when a candle arrives, never per signal check.
Every function here is pure: same frame in, same frame out, no hidden state.
"""

from __future__ import annotations

import polars as pl

REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")


class IndicatorError(ValueError):
    """Raised when the input frame cannot support the requested indicators."""


def _rma(expr: pl.Expr, period: int) -> pl.Expr:
    """
    Wilder's smoothing (RMA): avg = (prev * (period - 1) + value) / period.

    Algebraically identical to an EWM with alpha = 1/period and adjust=False.
    """
    return expr.ewm_mean(alpha=1.0 / period, adjust=False, min_samples=period)


def ema(column: str, period: int) -> pl.Expr:
    """Standard EMA, alpha = 2/(period+1)."""
    return pl.col(column).ewm_mean(
        alpha=2.0 / (period + 1), adjust=False, min_samples=period
    )


def rsi(column: str = "close", period: int = 14) -> pl.Expr:
    """
    Relative Strength Index, Wilder smoothed.

    An all-gains window gives avg_loss == 0; RSI is defined as 100 there rather
    than dividing by zero.
    """
    delta = pl.col(column).diff()
    gain = pl.when(delta > 0).then(delta).otherwise(0.0)
    loss = pl.when(delta < 0).then(-delta).otherwise(0.0)

    avg_gain = _rma(gain, period)
    avg_loss = _rma(loss, period)

    return (
        pl.when(avg_loss == 0)
        .then(pl.lit(100.0))
        .otherwise(100.0 - (100.0 / (1.0 + avg_gain / avg_loss)))
    )


def true_range() -> pl.Expr:
    """max(high-low, |high-prev_close|, |low-prev_close|)."""
    prev_close = pl.col("close").shift(1)
    return pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - prev_close).abs(),
        (pl.col("low") - prev_close).abs(),
    )


def atr(period: int = 14) -> pl.Expr:
    """Average True Range — the stop-loss distance unit used throughout."""
    return _rma(true_range(), period)


def adx(period: int = 14) -> dict[str, pl.Expr]:
    """
    Average Directional Index with its component +DI / -DI.

    NUANCE #8: ADX measures trend STRENGTH and is directionless. Use it to
    filter out chop; take direction from EMA or +DI/-DI, never from ADX itself.

    Returns:
        {'adx', 'plus_di', 'minus_di'} — all three are returned because the DI
        pair is what actually carries direction when you want it.
    """
    up_move = pl.col("high").diff()
    down_move = -pl.col("low").diff()

    plus_dm = pl.when((up_move > down_move) & (up_move > 0)).then(up_move).otherwise(0.0)
    minus_dm = (
        pl.when((down_move > up_move) & (down_move > 0)).then(down_move).otherwise(0.0)
    )

    atr_expr = _rma(true_range(), period)
    plus_di = 100.0 * _rma(plus_dm, period) / atr_expr
    minus_di = 100.0 * _rma(minus_dm, period) / atr_expr

    di_sum = plus_di + minus_di
    dx = pl.when(di_sum == 0).then(pl.lit(0.0)).otherwise(
        100.0 * (plus_di - minus_di).abs() / di_sum
    )

    return {"adx": _rma(dx, period), "plus_di": plus_di, "minus_di": minus_di}


def vwap_intraday() -> pl.Expr:
    """
    Session-anchored VWAP, reset at every session boundary.

    NUANCE #4 — the single largest cause of backtest/live divergence. A VWAP that
    accumulates across days drifts further from price every session, so the
    "price above VWAP" filter silently stops filtering. The `.over("session_date")`
    is what makes the reset structural instead of something a caller can forget.

    Requires a `session_date` column (add_indicators creates it).
    """
    typical_price = (pl.col("high") + pl.col("low") + pl.col("close")) / 3.0
    cumulative_tpv = (typical_price * pl.col("volume")).cum_sum().over("session_date")
    cumulative_volume = pl.col("volume").cum_sum().over("session_date")

    return (
        pl.when(cumulative_volume > 0)
        .then(cumulative_tpv / cumulative_volume)
        .otherwise(pl.col("close"))
    )


def vwap_rolling(window: int = 20) -> pl.Expr:
    """Rolling VWAP for daily/positional frames, where a session anchor is meaningless."""
    typical_price = (pl.col("high") + pl.col("low") + pl.col("close")) / 3.0
    return (typical_price * pl.col("volume")).rolling_sum(window) / pl.col(
        "volume"
    ).rolling_sum(window)


def warmup_bars(config: dict) -> int:
    """
    Bars needed before every indicator is populated.

    Wilder smoothing needs roughly 3x the period to converge from its seed, and
    ADX chains two smoothings. Feeding a signal generator fewer bars than this
    produces indicator values that are technically non-null but not yet
    meaningful — a subtle way to make a backtest lie.
    """
    periods = [
        config.get("rsi_period", 14) * 3,
        config.get("adx_period", 14) * 6,  # DI smoothing then ADX smoothing
        config.get("atr_period", 14) * 3,
        config.get("ema_slow", 21) * 3,
        config.get("volume_lookback", 20),
    ]
    return max(periods)


def add_indicators(
    df: pl.DataFrame, config: dict | None = None, *, intraday: bool = True
) -> pl.DataFrame:
    """
    Attach every indicator the signal generators need.

    Args:
        df: OHLCV frame with REQUIRED_COLUMNS, any order
        config: parameter overrides (rsi_period, ema_fast, ema_slow, ...)
        intraday: True for minute bars (session-anchored VWAP), False for daily
                  bars (rolling VWAP)

    Returns:
        New frame sorted by timestamp with indicator columns appended.

    Raises:
        IndicatorError: missing columns, or too few rows to compute anything.
    """
    config = config or {}

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise IndicatorError(f"missing required columns: {missing}")
    if df.height == 0:
        raise IndicatorError("empty frame")

    rsi_period = config.get("rsi_period", 14)
    adx_period = config.get("adx_period", 14)
    atr_period = config.get("atr_period", 14)
    ema_fast_period = config.get("ema_fast", 9)
    ema_slow_period = config.get("ema_slow", 21)
    volume_lookback = config.get("volume_lookback", 20)

    out = df.sort("timestamp").with_columns(
        pl.col("timestamp").dt.date().alias("session_date")
    )

    adx_parts = adx(adx_period)

    out = out.with_columns(
        [
            ema("close", ema_fast_period).alias("ema_fast"),
            ema("close", ema_slow_period).alias("ema_slow"),
            rsi("close", rsi_period).alias("rsi"),
            atr(atr_period).alias("atr"),
            adx_parts["adx"].alias("adx"),
            adx_parts["plus_di"].alias("plus_di"),
            adx_parts["minus_di"].alias("minus_di"),
            (vwap_intraday() if intraday else vwap_rolling(volume_lookback)).alias("vwap"),
            # Volume baseline excludes the current bar: comparing a bar against an
            # average that already contains it dampens exactly the surge we want
            # to detect.
            pl.col("volume")
            .shift(1)
            .rolling_mean(window_size=volume_lookback)
            .alias("avg_volume"),
            ((pl.col("close") - pl.col("open")).abs() / pl.col("open")).alias("body_pct"),
        ]
    )

    # Convenience ratios used by several strategies.
    return out.with_columns(
        [
            pl.when(pl.col("avg_volume") > 0)
            .then(pl.col("volume") / pl.col("avg_volume"))
            .otherwise(0.0)
            .alias("volume_ratio"),
            ((pl.col("close") - pl.col("vwap")) / pl.col("vwap") * 100).alias(
                "vwap_distance_pct"
            ),
            pl.when(pl.col("close") > 0)
            .then(pl.col("atr") / pl.col("close") * 100)
            .otherwise(0.0)
            .alias("atr_pct"),
        ]
    )


def is_warm(row: dict, config: dict | None = None) -> bool:
    """
    True when every indicator on this row is populated and usable.

    Signal generators call this first. A null ADX during warmup would otherwise
    compare False against the threshold and quietly suppress signals, which reads
    as "the strategy found nothing" rather than "the data was not ready".
    """
    required = ("ema_fast", "ema_slow", "rsi", "adx", "atr", "vwap", "avg_volume")
    return all(row.get(k) is not None for k in required)
