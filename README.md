# AlgoTrader

A quantitative trading engine for Indian equity markets (NSE / Zerodha Kite),
packaged as a Claude Code skill.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Zerodha](https://img.shields.io/badge/Zerodha-Kite%20API-orange.svg)](https://kite.trade/)
[![Tests](https://img.shields.io/badge/tests-152%20passing-green.svg)](tests/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Two things live here. **`engine/`** is runnable code that backtests strategies and
places real orders. **`KNOWLEDGE.md`** and **`NUANCES.md`** are the production
learnings the engine encodes — 10 domain sections and 30 numbered failure modes,
each with the mistake, the symptom, the fix, and what it cost.

The engine exists so the guard rails cannot be forgotten. Tick rounding, position
reconciliation, the stop-loss lifecycle, daily VWAP reset, symbol cooldowns,
candle completion, risk-based sizing, loss-streak halts — all enforced in code,
each traceable to a numbered nuance.

## Install

As a Claude Code skill:

```bash
git clone https://github.com/javajack/skill-algotrader.git ~/.claude/skills/algotrader
cd ~/.claude/skills/algotrader && ./run.sh help    # creates venv/ on first run
```

Then `/algotrader` in Claude Code, or use the CLI directly.

**Requirements:** Python 3.10+, and a **paid Zerodha Kite Connect** subscription.
The free "Personal" app type explicitly excludes historical chart data and live
quotes, both of which the engine needs.

## Setup

**1. Create a Kite Connect app** at <https://kite.trade/> (type: **Connect**).

- **Redirect URL**: `http://127.0.0.1:5000/` — nothing needs to listen there.
  After login the browser lands on it with `?request_token=...` in the address
  bar, which is all you need.
- **Postback URL**: leave blank. The engine polls order status.

**2. Credentials** in `.env`:

```bash
KITE_API_KEY=your_key
KITE_API_SECRET=your_secret
```

Do not put `KITE_ACCESS_TOKEN` here — `.kite_session.json` manages it.

**3. Authenticate.** Kite access tokens expire daily around 07:30 IST, so this is
a morning ritual:

```bash
./run.sh login          # opens the browser, catches the redirect, saves the token
```

It starts a local listener on the redirect URL, so the single-use `request_token`
never touches your clipboard — the usual source of "token invalid or expired".

**On macOS, port 5000 is taken by AirPlay Receiver.** Either disable it
(System Settings → General → AirDrop & Handoff) or use another port and set the
app's Redirect URL to match:

```bash
./run.sh login --port 5555        # app Redirect URL: http://127.0.0.1:5555/
./run.sh login --manual           # print the URL, paste the token yourself
./run.sh token <request_token>    # the manual second step
```

If Kite answers `Error generating request_token`, the setup is fine and the
browser session is not: log in as the exact Zerodha account the app is bound to
(Kite Connect apps are restricted to one client ID), make sure TOTP 2FA is
enrolled (mandatory for API logins), and start from a freshly generated URL in a
private window.

## Use

```bash
./run.sh universe --indices nifty50      # live NSE constituents
./run.sh warm --days 90                  # fill the Parquet cache (rate limited)
./run.sh backtest --start 2026-01-01     # simulate
./run.sh run --mode paper                # simulated fills, real prices
./run.sh run --mode live                 # requires typing LIVE
./run.sh report                          # analytics + factor attribution
./run.sh check ./my_bot.py               # scan code for known failure patterns
./run.sh wizard                          # generate a standalone bot
```

Run them in that order. The gap between backtest, paper, and live results is
diagnostic information, not noise.

## How it fits together

```
universe.py ─→ data.py ─→ indicators.py ─→ signals.py ─→ risk.py ─→ execution.py
                  ↑            ↑               ↑            ↑            ↑
              broker.py    (pure exprs)   (pure fns)   (stateful)   (broker calls)
                                                │
                        backtest.py ────────────┴──────────── runner.py
```

**Backtest and live share one code path.** `backtest.py` imports the same
strategy functions and the same `RiskManager` as `runner.py`; nothing is
reimplemented. Paper and live differ by one line — which `Broker` is constructed.
This is the whole design, and everything else follows from it.

**Strategies are pure functions**, `(row, symbol, config, tick_size) -> Signal | None`,
with no clock and no globals. Adding one means writing that function plus a
vectorised prefilter, and registering both:

```python
from engine.signals import STRATEGIES, PREFILTERS
STRATEGIES["my_strategy"] = my_evaluate
PREFILTERS["my_strategy"] = my_prefilter
```

See `examples/full_system.py` for a complete worked example (an opening-range
breakout). The test suite then automatically proves your prefilter is a superset
of your entry gates — if it is not, the backtest would silently skip trades that
live trading takes.

## What is enforced

| Guard rail | Where | Nuance |
|---|---|---|
| Every price tick-aligned before it reaches the exchange | `broker.py` | #1 |
| Broker reconciled at startup; adopts orphans, drops ghosts, protects naked positions | `execution.py` | #2 |
| New stop placed *before* the old is cancelled | `execution.py` | #3 |
| VWAP anchored per session | `indicators.py` | #4 |
| 45-minute symbol cooldown after an exit | `risk.py` | #5 |
| Live evaluates the last *complete* candle | `runner.py` | #6 |
| Margin from `net`, not `opening_balance` | `broker.py` | #7 |
| ADX filters strength; direction comes from EMA/DI | `signals.py` | #8 |
| No entries before 09:30, through lunch, or after 14:45 | `session.py` | #9, #27, #28 |
| Size derived from stop distance | `risk.py` | #19 |
| Daily-loss and loss-streak halts that survive a restart | `risk.py` | #20 |

## Backtest honesty

Three rules are built in, because removing any of them inflates results:

1. A signal on bar *i* fills at bar *i+1*'s **open**. Anything else is lookahead.
2. When a bar spans both stop and target, the **stop** is assumed to have hit
   first. Without tick data the order is unknowable, and the optimistic
   assumption meaningfully inflates win rate.
3. Costs are charged on every trade, from the same `costs.py` live execution uses.

A useful sanity check: run it on random-walk data. If it finds edge there, the
backtest is broken.

## Strategies included

| Name | Idea | Fits |
|---|---|---|
| `fortress` | Six-factor confirmation: trend, ADX strength, RSI band (hard gates), then VWAP, volume, candle body (confidence) | Largecap intraday |
| `momentum` | Breakout over the recent high, confirmed by volume | Midcaps, where breakouts follow through |
| `mean_reversion` | VWAP pullback within an intact trend | Largecaps, which revert to VWAP |

Shipped parameters are starting points from KNOWLEDGE.md, not tuned values.

## Tests

```bash
./venv/bin/pip install -r requirements-dev.txt
./venv/bin/python -m pytest tests/ -q      # 152 tests, ~0.5s, no network, no broker
```

The suite runs fully offline against synthetic OHLCV and a scriptable fake
broker. The tests worth reading first are the VWAP reset test, the
place-then-cancel ordering test, and `TestPrefilterIsSuperset`.

## F&O status

Contract resolution (`engine/derivatives.py`) and the F&O cost models
(`engine/costs.py`) are implemented and tested: expiry selection, ATM/ITM strike
resolution from the live NFO dump, lot arithmetic, physical-settlement warnings
for stock F&O, and premium-based option costs. **The live runner does not yet
route orders through them** — equity is the only wired path today.

## Limits

The engine handles mechanics, not edge. It cannot tell you whether a strategy
makes money; only a backtest on real data followed by weeks of paper trading can,
and both can still mislead.

Performance figures in the older documents (65% win rate, 51% CAGR, 28x Parquet,
37x Polars) describe a previous implementation. They are historical claims, not
predictions and not properties of this code.

`QUICKSTART.md`, `SETUP_COMPLETE.md`, `FINAL_SUMMARY.md` and
`START_SCRIPT_GUIDE.md` are one-time setup logs from the original author's
machine and are outdated. This file and `SKILL.md` are the maintained entry
points.

## Disclaimer

Educational and engineering guidance for building trading systems. Not investment
advice. **Trading involves risk; only trade capital you can afford to lose.**
Past backtested performance does not guarantee future results.

## License

MIT — see [LICENSE](LICENSE).
