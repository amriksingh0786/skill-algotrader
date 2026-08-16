"""
Event-driven backtest.

PARITY IS THE WHOLE POINT (KNOWLEDGE.md section 2)
--------------------------------------------------
This engine imports the same `signals.STRATEGIES` functions and the same
`RiskManager` the live runner uses. Nothing about entry logic, sizing, or limits
is reimplemented here — if it were, the two would drift within a week and the
backtest would stop predicting anything.

What IS modelled here (and cannot be shared, because live has an exchange):
  - fills, with slippage and the real cost stack
  - intrabar stop/target resolution

HONESTY RULES BAKED IN
----------------------
  1. A signal on bar i can only be filled at bar i+1's open. Filling at bar i's
     close is lookahead: that price is only knowable once the bar is complete.
  2. If a bar's range covers both stop and target, the STOP is assumed to have
     hit first. Without tick data the order is unknowable, and the optimistic
     assumption inflates win rate by a wide margin on volatile names.
  3. Costs are charged on every trade, STT on the sell side (NUANCE #30).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Any, Callable

import polars as pl

from .costs import estimate_costs
from .indicators import is_warm
from .risk import RiskManager
from .session import (
    DEFAULT_ENTRY_START,
    DEFAULT_LAST_ENTRY,
    DEFAULT_SQUAREOFF,
    should_squareoff,
    should_trade_now,
    to_ist,
)
from .signals import PREFILTERS, STRATEGIES, Signal, add_strategy_columns

# Columns the loop needs. Selecting them explicitly keeps row iteration cheap.
_LOOP_COLUMNS = (
    "timestamp", "symbol", "open", "high", "low", "close", "volume",
    "ema_fast", "ema_slow", "rsi", "adx", "plus_di", "minus_di", "atr", "vwap",
    "avg_volume", "volume_ratio", "vwap_distance_pct", "body_pct", "atr_pct",
    "rolling_high", "rolling_low", "_candidate",
)


def _parse_time(value: Any, default: Any) -> Any:
    """Accept 'HH:MM' from JSON config, a datetime.time, or None."""
    if value is None:
        return default
    if isinstance(value, str):
        hour, _, minute = value.partition(":")
        return time(int(hour), int(minute or 0))
    return value


@dataclass
class OpenTrade:
    """A position held during the simulation."""

    symbol: str
    direction: str
    quantity: int
    entry_price: float
    entry_time: datetime
    stop_loss: float
    target: float
    initial_stop: float
    strategy: str
    confidence: float
    reason: str
    high_water_mark: float = 0.0

    @property
    def risk_amount(self) -> float:
        return abs(self.entry_price - self.stop_loss) * self.quantity

    @property
    def value(self) -> float:
        return self.entry_price * self.quantity


@dataclass
class BacktestResult:
    trades: list[dict[str, Any]] = field(default_factory=list)
    equity_curve: list[dict[str, Any]] = field(default_factory=list)
    rejections: dict[str, int] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)

    def trades_frame(self) -> pl.DataFrame:
        return pl.DataFrame(self.trades) if self.trades else pl.DataFrame()

    def equity_frame(self) -> pl.DataFrame:
        return pl.DataFrame(self.equity_curve) if self.equity_curve else pl.DataFrame()


def prepare(frames: dict[str, pl.DataFrame], config: dict, strategy: str) -> pl.DataFrame:
    """
    Merge per-symbol frames into one time-ordered event stream.

    The `_candidate` column is the vectorised prefilter — a superset of the
    strategy's hard gates. Rows that fail it cannot produce a signal, so the
    expensive per-row `evaluate()` is skipped for them. On a typical universe
    that removes well over 90% of rows while changing no outcome.
    """
    prefilter = PREFILTERS.get(strategy)
    prepared: list[pl.DataFrame] = []

    for symbol, frame in frames.items():
        if frame.height == 0:
            continue

        enriched = add_strategy_columns(frame, config).with_columns(
            pl.lit(symbol).alias("symbol")
        )
        enriched = enriched.with_columns(
            (prefilter(config) if prefilter else pl.lit(True)).fill_null(False).alias("_candidate")
        )
        prepared.append(enriched)

    if not prepared:
        return pl.DataFrame(schema={c: pl.Null for c in _LOOP_COLUMNS})

    # Keep every column, not a fixed allowlist. A custom strategy that adds its
    # own column (an opening range, a higher-timeframe trend) would otherwise
    # find it silently dropped here and return None on every row — no error, no
    # trades, and nothing in the rejection counts to explain why.
    all_columns: list[str] = []
    for frame in prepared:
        for column in frame.columns:
            if column not in all_columns:
                all_columns.append(column)

    aligned = []
    for frame in prepared:
        missing = [c for c in all_columns if c not in frame.columns]
        if missing:
            frame = frame.with_columns([pl.lit(None).alias(c) for c in missing])
        aligned.append(frame.select(all_columns))
    prepared = aligned

    # Sort by timestamp then symbol: a deterministic order matters when several
    # symbols signal on the same bar and capital is the binding constraint.
    return pl.concat(prepared, how="vertical_relaxed").sort(["timestamp", "symbol"])


def run(
    frames: dict[str, pl.DataFrame],
    config: dict | None = None,
    *,
    strategy: str = "fortress",
    starting_capital: float = 1_000_000.0,
    slippage_bps: float = 5.0,
    tick_sizes: dict[str, float] | None = None,
    intraday: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
) -> BacktestResult:
    """
    Run the simulation.

    Args:
        frames: {symbol: indicator-enriched OHLCV frame} from
                DataManager.with_indicators()
        config: strategy + risk parameters (same dict live mode uses)
        strategy: key into signals.STRATEGIES
        starting_capital: opening equity
        slippage_bps: applied to every fill, in the direction that hurts
        tick_sizes: {symbol: tick}; defaults to 0.05
        intraday: square off daily and charge intraday costs

    Returns:
        BacktestResult with trades, equity curve, rejection counts, metrics.
    """
    config = dict(config or {})
    tick_sizes = tick_sizes or {}
    evaluate = STRATEGIES[strategy]

    stream = prepare(frames, config, strategy)
    if stream.height == 0:
        return BacktestResult(metrics={"error": "no data"}, config=config)

    risk = RiskManager(config)
    result = BacktestResult(config={**config, "strategy": strategy,
                                   "starting_capital": starting_capital})

    capital = starting_capital
    equity = starting_capital
    open_trades: dict[str, OpenTrade] = {}
    pending: dict[str, Signal] = {}  # symbol -> signal awaiting next bar's open
    rejections: dict[str, int] = {}
    current_day = None
    day_start_equity = starting_capital

    total_rows = stream.height
    squareoff_time = _parse_time(config.get("squareoff_time"), DEFAULT_SQUAREOFF)
    entry_start = _parse_time(config.get("entry_start_time"), DEFAULT_ENTRY_START)
    last_entry = _parse_time(config.get("last_entry_time"), DEFAULT_LAST_ENTRY)
    skip_lunch = config.get("skip_lunch", True)

    for index, row in enumerate(stream.iter_rows(named=True)):
        symbol = row["symbol"]
        timestamp = to_ist(row["timestamp"])

        if on_progress and index % 50_000 == 0:
            on_progress(index, total_rows)

        # ---- day rollover: mark equity, reset daily risk counters
        if current_day != timestamp.date():
            if current_day is not None:
                result.equity_curve.append(
                    {"date": current_day.isoformat(), "equity": round(equity, 2),
                     "day_pnl": round(equity - day_start_equity, 2),
                     "open_positions": len(open_trades)}
                )
            current_day = timestamp.date()
            day_start_equity = equity
            risk.roll_day_if_needed(timestamp)

        # ---- 1. fill anything queued from the previous bar (no lookahead)
        signal = pending.pop(symbol, None)
        if signal and symbol not in open_trades:
            fill_price = _slip(row["open"], "BUY" if signal.direction == "LONG" else "SELL",
                               slippage_bps)
            sizing = risk.size_position(
                signal, capital=equity, available_margin=capital,
                lot_size=1, regime_multiplier=config.get("regime_multiplier", 1.0),
            )
            if sizing.is_tradeable and sizing.position_value <= capital:
                open_trades[symbol] = OpenTrade(
                    symbol=symbol,
                    direction=signal.direction,
                    quantity=sizing.quantity,
                    entry_price=fill_price,
                    entry_time=timestamp,
                    stop_loss=signal.stop_loss,
                    target=signal.target,
                    initial_stop=signal.stop_loss,
                    strategy=signal.strategy,
                    confidence=signal.confidence,
                    reason=signal.reason,
                    high_water_mark=fill_price,
                )
                capital -= sizing.position_value
            else:
                rejections[sizing.binding_constraint] = (
                    rejections.get(sizing.binding_constraint, 0) + 1
                )

        # ---- 2. manage an open position on this bar
        trade = open_trades.get(symbol)
        if trade:
            exit_price, exit_reason = _resolve_exit(
                trade, row, timestamp, config, intraday, squareoff_time
            )
            if exit_price is not None:
                exit_price = _slip(exit_price, "SELL" if trade.direction == "LONG" else "BUY",
                                   slippage_bps)
                record = _close(trade, exit_price, timestamp, exit_reason, intraday)
                result.trades.append(record)
                capital += trade.value + record["pnl"]
                equity = capital + sum(
                    t.value for s, t in open_trades.items() if s != symbol
                )
                risk.record_exit(symbol, record["pnl"], timestamp)
                del open_trades[symbol]

        # ---- 3. look for a new signal (cheap prefilter already applied)
        if not row.get("_candidate") or symbol in open_trades or symbol in pending:
            continue

        if intraday:
            allowed, reason = should_trade_now(
                timestamp,
                entry_start=entry_start,
                last_entry=last_entry,
                skip_lunch=skip_lunch,
            )
        else:
            allowed, reason = True, "ok"

        if not allowed:
            rejections[reason] = rejections.get(reason, 0) + 1
            continue

        if not is_warm(row):
            rejections["warmup"] = rejections.get("warmup", 0) + 1
            continue

        candidate = evaluate(row, symbol, config, tick_sizes.get(symbol, 0.05))
        if candidate is None:
            continue

        ok, reason = risk.can_enter(
            candidate, open_positions=list(open_trades.values()),
            capital=equity, now=timestamp,
        )
        if not ok:
            rejections[reason.split("(")[0].strip()] = (
                rejections.get(reason.split("(")[0].strip(), 0) + 1
            )
            continue

        pending[symbol] = candidate

    # ---- final mark and forced close of anything still open
    last_row = stream.row(-1, named=True)
    final_time = to_ist(last_row["timestamp"])
    for symbol, trade in list(open_trades.items()):
        last_close = float(
            stream.filter(pl.col("symbol") == symbol).row(-1, named=True)["close"]
        )
        record = _close(trade, last_close, final_time, "backtest end", intraday)
        result.trades.append(record)
        capital += trade.value + record["pnl"]
    equity = capital

    if current_day is not None:
        result.equity_curve.append(
            {"date": current_day.isoformat(), "equity": round(equity, 2),
             "day_pnl": round(equity - day_start_equity, 2), "open_positions": 0}
        )

    result.rejections = dict(sorted(rejections.items(), key=lambda kv: -kv[1]))
    result.metrics = compute_metrics(result.trades, result.equity_curve, starting_capital)
    return result


def _slip(price: float, side: str, bps: float) -> float:
    """Slippage always works against you."""
    delta = price * bps / 10_000
    return price + delta if side == "BUY" else price - delta


def _resolve_exit(
    trade: OpenTrade, row: dict, timestamp: datetime, config: dict,
    intraday: bool, squareoff_time: Any,
) -> tuple[float | None, str]:
    """
    Decide whether this bar closes the trade, and at what price.

    Stop before target when a bar spans both — see the honesty rules at the top.
    """
    high, low = float(row["high"]), float(row["low"])

    if trade.direction == "LONG":
        trade.high_water_mark = max(trade.high_water_mark, high)
        if low <= trade.stop_loss:
            return trade.stop_loss, "stop loss"
        if trade.target and high >= trade.target:
            return trade.target, "target"
    else:
        trade.high_water_mark = min(trade.high_water_mark, low)
        if high >= trade.stop_loss:
            return trade.stop_loss, "stop loss"
        if trade.target and low <= trade.target:
            return trade.target, "target"

    close = float(row["close"])

    # Breakeven and trailing stops move only after the bar's extremes are checked,
    # so a stop can never be tightened using a price the bar itself produced.
    risk_per_share = abs(trade.entry_price - trade.initial_stop)
    if risk_per_share > 0:
        r_multiple = (
            (close - trade.entry_price) / risk_per_share
            if trade.direction == "LONG"
            else (trade.entry_price - close) / risk_per_share
        )
        breakeven_r = config.get("breakeven_at_r", 0)
        if breakeven_r and r_multiple >= breakeven_r:
            buffer = trade.entry_price * 0.0005
            new_stop = (trade.entry_price + buffer if trade.direction == "LONG"
                        else trade.entry_price - buffer)
            trade.stop_loss = (max(trade.stop_loss, new_stop) if trade.direction == "LONG"
                               else min(trade.stop_loss, new_stop))

    trail_pct = config.get("trail_pct", 0)
    if trail_pct:
        trailed = (trade.high_water_mark * (1 - trail_pct / 100)
                   if trade.direction == "LONG"
                   else trade.high_water_mark * (1 + trail_pct / 100))
        trade.stop_loss = (max(trade.stop_loss, trailed) if trade.direction == "LONG"
                           else min(trade.stop_loss, trailed))

    max_hold = config.get("max_hold_minutes", 0)
    if max_hold:
        held = (timestamp - to_ist(trade.entry_time)).total_seconds() / 60
        if held >= max_hold:
            return close, "time stop"

    if intraday and should_squareoff(timestamp, squareoff_time):
        return close, "square off"

    return None, ""


def _close(trade: OpenTrade, exit_price: float, exit_time: datetime,
           reason: str, intraday: bool) -> dict[str, Any]:
    gross = (
        (exit_price - trade.entry_price) * trade.quantity
        if trade.direction == "LONG"
        else (trade.entry_price - exit_price) * trade.quantity
    )
    costs = estimate_costs(trade.entry_price, exit_price, trade.quantity, intraday=intraday)
    net = gross - costs
    risk_per_share = abs(trade.entry_price - trade.initial_stop)

    return {
        "symbol": trade.symbol,
        "direction": trade.direction,
        "strategy": trade.strategy,
        "quantity": trade.quantity,
        "entry_price": round(trade.entry_price, 2),
        "exit_price": round(exit_price, 2),
        "entry_time": trade.entry_time.isoformat(),
        "exit_time": exit_time.isoformat(),
        "holding_minutes": round((exit_time - to_ist(trade.entry_time)).total_seconds() / 60, 1),
        "gross_pnl": round(gross, 2),
        "costs": round(costs, 2),
        "pnl": round(net, 2),
        "pnl_pct": round(net / (trade.entry_price * trade.quantity) * 100, 4),
        "r_multiple": round(net / (risk_per_share * trade.quantity), 3) if risk_per_share else 0.0,
        "exit_reason": reason,
        "confidence": trade.confidence,
        "entry_reason": trade.reason,
    }


def compute_metrics(trades: list[dict], equity_curve: list[dict],
                    starting_capital: float) -> dict[str, Any]:
    """
    Performance summary.

    Sharpe is computed on daily equity returns and annualised over 252 sessions.
    Max drawdown is peak-to-trough on the daily curve — with fewer than ~30
    trades none of these numbers mean much, which `sample_warning` flags.
    """
    if not trades:
        return {"total_trades": 0, "note": "no trades — check filters and warmup"}

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    total_pnl = sum(pnls)
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))

    avg_win = gross_profit / len(wins) if wins else 0.0
    avg_loss = gross_loss / len(losses) if losses else 0.0
    win_rate = len(wins) / len(trades)

    equities = [row["equity"] for row in equity_curve] or [starting_capital + total_pnl]
    peak, max_drawdown = equities[0], 0.0
    for value in equities:
        peak = max(peak, value)
        max_drawdown = max(max_drawdown, (peak - value) / peak * 100 if peak else 0.0)

    daily_returns: list[float] = []
    for previous, current in zip(equities, equities[1:]):
        if previous:
            daily_returns.append((current - previous) / previous)

    sharpe = 0.0
    if len(daily_returns) > 1:
        mean = sum(daily_returns) / len(daily_returns)
        variance = sum((r - mean) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
        std = variance ** 0.5
        if std > 0:
            sharpe = (mean / std) * (252 ** 0.5)

    by_reason: dict[str, dict[str, Any]] = {}
    for trade in trades:
        bucket = by_reason.setdefault(trade["exit_reason"], {"count": 0, "pnl": 0.0})
        bucket["count"] += 1
        bucket["pnl"] += trade["pnl"]

    by_symbol: dict[str, float] = {}
    for trade in trades:
        by_symbol[trade["symbol"]] = by_symbol.get(trade["symbol"], 0.0) + trade["pnl"]
    ranked = sorted(by_symbol.items(), key=lambda kv: -kv[1])

    return {
        "total_trades": len(trades),
        "win_rate": round(win_rate, 4),
        "wins": len(wins),
        "losses": len(losses),
        "total_pnl": round(total_pnl, 2),
        "return_pct": round(total_pnl / starting_capital * 100, 3),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "largest_win": round(max(pnls), 2),
        "largest_loss": round(min(pnls), 2),
        "profit_factor": round(gross_profit / gross_loss, 3) if gross_loss else float("inf"),
        "expectancy": round(total_pnl / len(trades), 2),
        "expectancy_r": round(sum(t.get("r_multiple", 0) for t in trades) / len(trades), 3),
        "total_costs": round(sum(t["costs"] for t in trades), 2),
        "cost_drag_pct": round(
            sum(t["costs"] for t in trades) / starting_capital * 100, 3
        ),
        "max_drawdown_pct": round(max_drawdown, 3),
        "sharpe": round(sharpe, 3),
        "avg_holding_minutes": round(
            sum(t["holding_minutes"] for t in trades) / len(trades), 1
        ),
        "trading_days": len(equity_curve),
        "by_exit_reason": {
            k: {"count": v["count"], "pnl": round(v["pnl"], 2)} for k, v in by_reason.items()
        },
        "best_symbols": [{"symbol": s, "pnl": round(p, 2)} for s, p in ranked[:5]],
        "worst_symbols": [{"symbol": s, "pnl": round(p, 2)} for s, p in ranked[-5:]],
        "sample_warning": (
            "fewer than 30 trades — metrics are not statistically meaningful"
            if len(trades) < 30 else None
        ),
    }
