#!/usr/bin/env python3
"""
Minimal intraday bot — the smallest thing that is actually safe to run.

    python templates/minimal_intraday.py --mode paper

Earlier versions of this file were a scaffold full of TODOs: the guard rails
appeared as comments telling you to implement them. That is exactly backwards —
the guard rails are the hard part and the strategy is the easy part. Everything
listed below is now enforced by engine/, not by your diligence:

    tick rounding (#1)          engine/broker.py     round_to_tick
    reconciliation (#2)         engine/execution.py  ExecutionEngine.reconcile
    stop lifecycle (#3)         engine/execution.py  ExecutionEngine.move_stop
    VWAP daily reset (#4)       engine/indicators.py vwap_intraday
    symbol cooldown (#5)        engine/risk.py       RiskManager.cooldown_remaining
    candle completion (#6)      engine/session.py    is_candle_complete
    margin from 'net' (#7)      engine/broker.py     available_margin
    session timing (#9, #27-28) engine/session.py    should_trade_now
    risk per trade (#19)        engine/risk.py       size_position
    loss streak halt (#20)      engine/risk.py       can_enter

What is left for you is the part that decides whether this makes money: the
strategy, its parameters, and the universe.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.runner import TradingRunner

CONFIG = {
    "strategy": "fortress",
    "interval": "minute",
    "product": "MIS",
    "capital": 1_000_000.0,
    "universe_file": "universe/nifty50.json",

    # Entry filters — the tested values from KNOWLEDGE.md section 3.
    "rsi_long_min": 45,
    "rsi_long_max": 65,
    "adx_min": 25,
    "volume_mult": 1.5,
    "min_confidence": 0.50,

    # Exits. A 1.2 ATR stop with a 1.5 reward target needs roughly a 40% win
    # rate to break even before costs — check that against your backtest.
    "sl_atr_mult": 1.2,
    "risk_reward": 1.5,
    "max_hold_minutes": 45,
    "breakeven_at_r": 1.0,

    # Risk. These are the numbers that keep a bad day from being a bad year.
    "risk_pct": 1.0,
    "max_positions": 5,
    "max_portfolio_heat_pct": 5.0,
    "daily_loss_limit_pct": 3.0,
    "max_consecutive_losses": 3,
    "symbol_cooldown_minutes": 45,

    # Session (IST).
    "entry_start_time": "09:30",
    "last_entry_time": "14:45",
    "squareoff_time": "15:10",
    "skip_lunch": True,
    "close_on_shutdown": True,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["paper", "live"], default="paper")
    args = parser.parse_args()

    if args.mode == "live" and input("Type LIVE to confirm: ").strip() != "LIVE":
        return 1

    TradingRunner(CONFIG, mode=args.mode).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
