"""
Post-trade analytics and signal attribution.

Reads the JSONL streams `logs.py` writes. The questions worth answering after a
losing week are not "how much did I lose" but "which part stopped working" —
so everything here breaks P&L down by an axis you can act on: strategy, exit
reason, hour of day, and the individual signal factors.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from .backtest import compute_metrics


def load_jsonl(path: str | Path) -> pl.DataFrame:
    """Read a JSONL stream into Polars. Missing or empty file gives an empty frame."""
    file_path = Path(path)
    if not file_path.exists():
        return pl.DataFrame()

    records = []
    for line in file_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a torn final line from a crash must not break the report

    return pl.DataFrame(records, infer_schema_length=None) if records else pl.DataFrame()


def daily_report(trades_path: str | Path = "logs/trades.jsonl",
                 starting_capital: float = 1_000_000.0,
                 day: str | None = None) -> dict[str, Any]:
    """
    Summarise one day (default: the most recent day present).

    Returns the same metric shape as a backtest, so live and backtest results can
    be compared field by field — which is the whole point of tracking parity.
    """
    frame = load_jsonl(trades_path)
    if frame.height == 0:
        return {"error": "no trades logged"}

    frame = frame.with_columns(
        pl.col("exit_time").str.slice(0, 10).alias("trade_date")
    )
    target_day = day or frame["trade_date"].max()
    day_trades = frame.filter(pl.col("trade_date") == target_day)

    if day_trades.height == 0:
        return {"error": f"no trades on {target_day}"}

    trades = day_trades.to_dicts()
    equity_curve = [{
        "date": target_day,
        "equity": starting_capital + sum(t["pnl"] for t in trades),
    }]

    metrics = compute_metrics(trades, equity_curve, starting_capital)
    metrics["date"] = target_day
    metrics["by_hour"] = _by_hour(day_trades)
    metrics["by_strategy"] = _by_group(day_trades, "strategy")
    return metrics


def _by_hour(frame: pl.DataFrame) -> dict[str, dict[str, Any]]:
    """P&L by entry hour. Reliably shows whether the lunch-lull block is earning its keep."""
    if "entry_time" not in frame.columns:
        return {}

    grouped = (
        frame.with_columns(pl.col("entry_time").str.slice(11, 2).alias("hour"))
        .group_by("hour")
        .agg([pl.len().alias("trades"), pl.col("pnl").sum().alias("pnl"),
              (pl.col("pnl") > 0).mean().alias("win_rate")])
        .sort("hour")
    )
    return {
        row["hour"]: {
            "trades": row["trades"],
            "pnl": round(row["pnl"], 2),
            "win_rate": round(row["win_rate"], 3),
        }
        for row in grouped.iter_rows(named=True)
    }


def _by_group(frame: pl.DataFrame, column: str) -> dict[str, dict[str, Any]]:
    if column not in frame.columns:
        return {}

    grouped = frame.group_by(column).agg(
        [pl.len().alias("trades"), pl.col("pnl").sum().alias("pnl"),
         (pl.col("pnl") > 0).mean().alias("win_rate")]
    )
    return {
        str(row[column]): {
            "trades": row["trades"],
            "pnl": round(row["pnl"], 2),
            "win_rate": round(row["win_rate"], 3),
        }
        for row in grouped.iter_rows(named=True)
    }


def signal_attribution(signals_path: str | Path = "logs/signals.jsonl",
                       trades_path: str | Path = "logs/trades.jsonl") -> dict[str, Any]:
    """
    Which signal factors actually predicted profit?

    Joins taken signals to their outcomes and compares win rate with and without
    each factor. A factor whose presence does not change the win rate is costing
    you trades for nothing; one that inverts it is actively harmful.

    Needs a few dozen trades before it means anything — the sample sizes are
    reported so you can judge.
    """
    signals = load_jsonl(signals_path)
    trades = load_jsonl(trades_path)

    if signals.height == 0:
        return {"error": "no signals logged"}

    result: dict[str, Any] = {
        "total_signals": signals.height,
        "taken": int(signals.filter(pl.col("taken")).height) if "taken" in signals.columns else 0,
    }

    if "rejection_reason" in signals.columns:
        rejected = signals.filter(~pl.col("taken")) if "taken" in signals.columns else signals
        if rejected.height:
            counts = (
                rejected.group_by("rejection_reason")
                .agg(pl.len().alias("count"))
                .sort("count", descending=True)
            )
            result["rejection_reasons"] = {
                row["rejection_reason"]: row["count"]
                for row in counts.iter_rows(named=True)
                if row["rejection_reason"]
            }

    if trades.height == 0 or "factors" not in signals.columns:
        return result

    taken = signals.filter(pl.col("taken")).to_dicts()
    outcome_by_symbol_time: dict[tuple[str, str], float] = {}
    for trade in trades.to_dicts():
        outcome_by_symbol_time[(trade["symbol"], trade["entry_time"][:16])] = trade["pnl"]

    factor_stats: dict[str, dict[str, list[float]]] = {}
    for record in taken:
        factors = record.get("factors") or {}
        if not isinstance(factors, dict):
            continue

        pnl = None
        for (symbol, minute), value in outcome_by_symbol_time.items():
            if symbol == record.get("symbol"):
                pnl = value
                break
        if pnl is None:
            continue

        for factor in ("trend", "strength", "momentum", "vwap", "volume", "body",
                       "breakout", "vwap_pullback", "trend_intact", "oversold"):
            bucket = factor_stats.setdefault(factor, {"with": [], "without": []})
            bucket["with" if factor in factors else "without"].append(pnl)

    attribution = {}
    for factor, buckets in factor_stats.items():
        with_pnls, without_pnls = buckets["with"], buckets["without"]
        if not with_pnls:
            continue
        attribution[factor] = {
            "trades_with": len(with_pnls),
            "win_rate_with": round(sum(1 for p in with_pnls if p > 0) / len(with_pnls), 3),
            "avg_pnl_with": round(sum(with_pnls) / len(with_pnls), 2),
            "trades_without": len(without_pnls),
            "win_rate_without": (
                round(sum(1 for p in without_pnls if p > 0) / len(without_pnls), 3)
                if without_pnls else None
            ),
            "avg_pnl_without": (
                round(sum(without_pnls) / len(without_pnls), 2) if without_pnls else None
            ),
        }

    result["factor_attribution"] = attribution
    result["note"] = "needs 30+ trades per factor before these differences mean anything"
    return result


def compare_backtest_to_live(
    backtest_metrics: dict[str, Any],
    live_metrics: dict[str, Any],
    *,
    win_rate_tolerance: float = 0.10,
) -> dict[str, Any]:
    """
    Parity diagnostic: where did live diverge from the backtest?

    A win-rate gap wider than the tolerance is the signature failure in
    NUANCES #4, #6 and #15 — VWAP not resetting, incomplete candles, or the two
    modes reading different data. The suspects are listed in the order they are
    usually guilty.
    """
    issues: list[dict[str, str]] = []

    backtest_win = backtest_metrics.get("win_rate", 0.0)
    live_win = live_metrics.get("win_rate", 0.0)
    delta = live_win - backtest_win

    if backtest_win and abs(delta) > win_rate_tolerance:
        issues.append({
            "severity": "critical",
            "metric": "win_rate",
            "detail": f"backtest {backtest_win:.1%} vs live {live_win:.1%} ({delta:+.1%})",
            "check": (
                "1) VWAP resetting daily (NUANCE #4)  "
                "2) signals evaluated on complete candles only (NUANCE #6)  "
                "3) both modes reading the same cache (NUANCE #15)  "
                "4) slippage and costs modelled realistically"
            ),
        })

    backtest_hold = backtest_metrics.get("avg_holding_minutes", 0)
    live_hold = live_metrics.get("avg_holding_minutes", 0)
    if backtest_hold and abs(live_hold - backtest_hold) > backtest_hold * 0.5:
        issues.append({
            "severity": "warning",
            "metric": "avg_holding_minutes",
            "detail": f"backtest {backtest_hold:.0f}m vs live {live_hold:.0f}m",
            "check": "exits firing differently — check trailing stop and time stop config",
        })

    backtest_expectancy = backtest_metrics.get("expectancy_r", 0)
    live_expectancy = live_metrics.get("expectancy_r", 0)
    if backtest_expectancy > 0 and live_expectancy < 0:
        issues.append({
            "severity": "critical",
            "metric": "expectancy_r",
            "detail": f"backtest {backtest_expectancy:+.2f}R vs live {live_expectancy:+.2f}R",
            "check": "edge did not survive contact with the market — stop live trading and diagnose",
        })

    return {
        "backtest": {k: backtest_metrics.get(k) for k in
                     ("total_trades", "win_rate", "expectancy_r", "profit_factor")},
        "live": {k: live_metrics.get(k) for k in
                 ("total_trades", "win_rate", "expectancy_r", "profit_factor")},
        "win_rate_delta": round(delta, 4),
        "parity_issues": issues,
        "verdict": "DIVERGENT" if any(i["severity"] == "critical" for i in issues) else "ALIGNED",
    }


def format_report(metrics: dict[str, Any]) -> str:
    """Human-readable summary for the terminal."""
    if "error" in metrics:
        return f"  {metrics['error']}"

    lines = [
        f"  Trades          {metrics.get('total_trades', 0)}",
        f"  Win rate        {metrics.get('win_rate', 0):.1%} "
        f"({metrics.get('wins', 0)}W / {metrics.get('losses', 0)}L)",
        f"  Net P&L         Rs {metrics.get('total_pnl', 0):,.0f} "
        f"({metrics.get('return_pct', 0):+.2f}%)",
        f"  Expectancy      Rs {metrics.get('expectancy', 0):,.0f} "
        f"({metrics.get('expectancy_r', 0):+.2f}R per trade)",
        f"  Profit factor   {metrics.get('profit_factor', 0):.2f}",
        f"  Avg win/loss    Rs {metrics.get('avg_win', 0):,.0f} / "
        f"Rs {metrics.get('avg_loss', 0):,.0f}",
        f"  Max drawdown    {metrics.get('max_drawdown_pct', 0):.2f}%",
        f"  Sharpe          {metrics.get('sharpe', 0):.2f}",
        f"  Costs paid      Rs {metrics.get('total_costs', 0):,.0f}",
        f"  Avg hold        {metrics.get('avg_holding_minutes', 0):.0f} min",
    ]

    if metrics.get("by_exit_reason"):
        lines.append("\n  Exit reasons:")
        for reason, stats in sorted(
            metrics["by_exit_reason"].items(), key=lambda kv: -kv[1]["count"]
        ):
            lines.append(f"    {reason:20s} {stats['count']:4d} trades  "
                         f"Rs {stats['pnl']:>12,.0f}")

    if metrics.get("sample_warning"):
        lines.append(f"\n  ! {metrics['sample_warning']}")

    return "\n".join(lines)
