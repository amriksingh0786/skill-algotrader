---
name: algotrader
description: Quantitative trading engine and expert for Indian equity markets (NSE/Zerodha Kite). Use when building, reviewing, debugging, backtesting, or running algorithmic trading code — signal generation, backtest-vs-live parity, order placement, stop-loss handling, position sizing, risk management, stock universe selection, or OHLCV/indicator pipelines. Triggers on Zerodha, kiteconnect, Kite API, NSE, Nifty, intraday/swing/positional bots, RSI/MACD/ADX/ATR/VWAP/EMA indicators, tick size, backtest parity, Kelly criterion, position reconciliation, paper trading, or trading bot generation.
---

# AlgoTrader — Quantitative Trading for Indian Equity Markets

A working trading engine plus the production learnings behind it. The engine
enforces the mechanical guard rails so they cannot be forgotten; the two
reference documents explain why each one exists and what it cost to learn.

## Two things live here

**`engine/`** — runnable code. Import it, don't reimplement it.
**`NUANCES.md` / `KNOWLEDGE.md`** — 30 numbered gotchas and 10 domain sections.
Grep them rather than reading end to end.

## Rules for working in this codebase

**Never hand-roll what the engine already enforces.** If a task involves order
prices, stops, position sizing, session timing, or reconciliation, the answer is
almost always an import, not new code. Reimplementing these is how the guard
rails get lost.

**Signal functions must stay pure** — `(row, symbol, config, tick_size) -> Signal | None`,
with no clock, no globals, no broker. This is what makes backtest and live run
the same code path. If a strategy needs the time, it comes from `row["timestamp"]`.

**Every new strategy needs a prefilter that is a superset of its hard gates.**
`tests/test_signals_and_risk.py::TestPrefilterIsSuperset` proves the property for
every registered strategy; a prefilter that rejects a tradeable row makes the
backtest silently skip trades live would take.

**Read the relevant NUANCES.md entry before touching its area.** Grep for the
topic — e.g. `grep -n -A25 "### 3\." NUANCES.md` for the stop-loss lifecycle.

## Where things are

| Concern | Module | Key entry points |
|---|---|---|
| Index constituents | `engine/universe.py` | `fetch_index_constituents`, `load_universe`, `filter_universe` |
| Indicators | `engine/indicators.py` | `add_indicators`, `warmup_bars`, `is_warm` |
| Market data + cache | `engine/data.py` | `DataManager.with_indicators`, `.live_frame`, `.warm_cache` |
| Broker | `engine/broker.py` | `KiteBroker`, `PaperBroker`, `round_to_tick`, `InstrumentMaster` |
| Strategies | `engine/signals.py` | `STRATEGIES`, `PREFILTERS`, `Signal` |
| Risk | `engine/risk.py` | `RiskManager.can_enter`, `.size_position`, `kelly_fraction`, `detect_market_regime` |
| Orders + stops | `engine/execution.py` | `ExecutionEngine.enter`, `.move_stop`, `.close`, `.reconcile` |
| Session timing | `engine/session.py` | `should_trade_now`, `is_candle_complete`, `TradingCalendar` |
| Costs | `engine/costs.py` | `estimate_costs`, `breakeven_move_pct` |
| Backtest | `engine/backtest.py` | `run`, `compute_metrics` |
| Live loop | `engine/runner.py` | `TradingRunner.run`, `.preflight` |
| Analytics | `engine/analytics.py` | `daily_report`, `signal_attribution`, `compare_backtest_to_live` |
| Logging | `engine/logs.py` | `TradingLogger` |

`KNOWLEDGE.md` sections: 1 Zerodha integration · 2 Backtest-live parity ·
3 Signal generation · 4 Rebalancing · 5 Universe selection · 6 Performance ·
7 Indian market specifics · 8 Failure handling · 9 Indicators & formulas ·
10 Multi-timeframe.

## What the engine already guarantees

Do not re-implement, and do not remove:

- **Tick alignment** (#1) — every price crossing the wire goes through `round_to_tick`.
- **Reconciliation** (#2) — `reconcile()` runs at startup; the broker is truth. Adopts orphans, drops ghosts, resizes on mismatch, protects naked positions.
- **Place-then-cancel stops** (#3) — new stop placed before the old is cancelled. Two live stops is acceptable; zero is not. Three failures triggers a market exit.
- **Session-anchored VWAP** (#4) — `.over("session_date")` makes the daily reset structural.
- **Symbol cooldown** (#5) — 45 min default after an exit.
- **Complete candles only** (#6) — live evaluates row `-2`, never the forming bar.
- **Margin from `net`** (#7).
- **ADX as a filter, not a direction** (#8).
- **Session gates** (#9, #27, #28) — no entries before 09:30, through lunch, or after 14:45.
- **Risk-based sizing** (#19) — quantity derives from stop distance, never from capital alone.
- **Loss-streak and daily-loss halts** (#20) — persisted across restarts, so restarting cannot clear a limit.

## Commands

```bash
./run.sh login                          # then: ./run.sh token <request_token>
./run.sh universe --indices nifty50     # live NSE constituent CSVs
./run.sh warm --days 90                 # fill the Parquet cache
./run.sh backtest --start 2026-01-01 --strategy fortress
./run.sh run --mode paper               # simulated fills, real prices
./run.sh run --mode live                # requires typing LIVE to confirm
./run.sh report                         # analytics + factor attribution
./run.sh check <path>                   # scan code against 12 failure patterns
./run.sh wizard                         # generate a standalone bot
./venv/bin/python -m pytest tests/ -q   # 128 tests
```

Kite access tokens expire daily around 07:30 IST; `login`/`token` is a morning
ritual. Historical data and live quotes need the **paid Connect plan** — the
free Personal plan explicitly excludes both.

## Sequencing advice

The correct order is backtest → paper → live, and the gap between paper and live
results is diagnostic, not noise. `analytics.compare_backtest_to_live` names the
usual culprits in the order they are usually guilty. A win-rate gap wider than
10 points means something is mechanically wrong (VWAP, candle completion, or the
two modes reading different data), not that "live is harder".

## Honest limits

The engine handles mechanics, not edge. It cannot tell you whether a strategy
makes money — only a backtest on real data followed by paper trading can, and
both can still mislead. The shipped parameters are starting points from
KNOWLEDGE.md, not tuned values. Backtested figures quoted in the docs (65% win
rate, 51% CAGR) are historical claims about a different implementation, not
predictions and not a property of this code.

Confirm with the user before anything runs in `--mode live`, and never place a
live order on your own initiative.
