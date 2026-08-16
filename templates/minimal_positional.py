#!/usr/bin/env python3
"""
Minimal positional bot — daily bars, held for weeks.

    python templates/minimal_positional.py --mode paper

Differences from the intraday template, and why each one matters:

  CNC not MIS          delivery product; T+1 settlement applies (NUANCE #17), so
                       capital from a sale is not immediately redeployable
  daily bars           VWAP is anchored per session, which is meaningless on
                       daily data — engine/indicators.py switches to a rolling
                       VWAP automatically when intraday=False
  no square-off        positions survive the close, so `close_on_shutdown` is
                       False: stopping the bot must not liquidate the portfolio
  wider stops          2.5 ATR instead of 1.2 — a daily bar's noise is an
                       intraday bar's trend
  trailing stop        8% give-back, since the edge is in letting winners run
  delivery costs       STT is charged on BOTH sides for delivery (0.1% each),
                       which is 4x the intraday sell-side rate — engine/costs.py
                       handles this via the intraday flag

The stop still lives at the broker as a resting order, so it protects the
position overnight and while the bot is not running.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.runner import TradingRunner

CONFIG = {
    "strategy": "fortress",
    "interval": "day",
    "product": "CNC",
    "capital": 1_000_000.0,
    "universe_file": "universe/nifty100.json",

    "rsi_long_min": 45,
    "rsi_long_max": 70,
    "adx_min": 22,
    "volume_mult": 1.3,
    "min_confidence": 0.50,

    "sl_atr_mult": 2.5,
    "risk_reward": 3.0,
    "max_hold_minutes": 0,      # no time stop; the trailing stop does the work
    "trail_pct": 8.0,
    "breakeven_at_r": 1.5,

    "risk_pct": 1.0,
    "max_positions": 10,
    "max_portfolio_heat_pct": 8.0,
    "daily_loss_limit_pct": 5.0,
    "max_consecutive_losses": 5,
    "symbol_cooldown_minutes": 1440,   # one day, not 45 minutes
    "sector_max_pct": 30.0,            # KNOWLEDGE.md section 5

    "squareoff_time": None,
    "close_on_shutdown": False,
    "skip_lunch": False,

    # Regime scaling matters far more on multi-week holds than intraday:
    # a positional long book in a bear regime is a slow bleed.
    "use_regime_sizing": True,
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
