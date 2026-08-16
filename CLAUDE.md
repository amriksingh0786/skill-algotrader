# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A Claude Code skill that is also a working trading engine. It installs to
`~/.claude/skills/algotrader` and is invoked as `/algotrader`.

Two halves, and the distinction matters when deciding where a change belongs:

- **`engine/`** — runnable Python that places real orders against Zerodha Kite.
- **`KNOWLEDGE.md` (1,780 lines) / `NUANCES.md` (1,044 lines)** — the production
  learnings the engine encodes. When a change could go in either the code or the
  docs, ask whether it is a *mechanism* (code) or a *reason* (docs).

Claude Code discovers the skill via **`SKILL.md`** (YAML frontmatter). `skill.json`
is a legacy manifest kept for the skills.sh directory listing — Claude Code does
not read it.

## Commands

```bash
./run.sh <command>                      # auto-creates venv/ on first run
./venv/bin/python -m pytest tests/ -q   # 128 tests, ~0.4s, fully offline
./venv/bin/python -m pytest tests/test_signals_and_risk.py::TestPrefilterIsSuperset -q
```

Skill commands: `login`, `token <request_token>`, `universe`, `warm`, `backtest`,
`run --mode paper|live`, `report`, `check <path>`, `wizard`.

`start.sh` additionally launches a generated bot: if argv[1] is not a known
command but is a directory, it runs `<dir>/main.py --mode paper|live`.

There is no linter configured. The test suite needs no network and no broker —
`tests/conftest.py` provides synthetic OHLCV and a scriptable `FakeBroker`.

**Three places list commands and all must stay in sync**: `algotrader.py`
subparsers, `start.sh`'s `known_commands` array, and `skill.json`. A command
missing from `start.sh` is treated as a bot directory and fails confusingly.

## Architecture

Data flows in one direction, and each module is importable on its own:

```
universe.py ─→ data.py ─→ indicators.py ─→ signals.py ─→ risk.py ─→ execution.py
                  ↑            ↑               ↑            ↑            ↑
              broker.py    (pure exprs)   (pure fns)   (stateful)   (broker calls)
                                                │
                        backtest.py ────────────┴──────────── runner.py
                        (simulated fills)                     (live loop)
```

**The parity mechanism.** `backtest.py` and `runner.py` import the *same*
`signals.STRATEGIES` functions and the *same* `RiskManager`. Nothing about entry
logic, sizing, or limits is reimplemented in the backtest. Paper and live differ
by exactly one line in `runner.py` — which Broker is constructed. Preserving this
is the single most important architectural constraint in the repo; a "small
tweak" that duplicates strategy logic into the backtest breaks it silently.

**Signal functions are pure**: `(row, symbol, config, tick_size) -> Signal | None`.
No clock, no globals, no broker. This deliberately differs from the pattern in
older versions of `examples/full_system.py`, which called `datetime.now()` inside
the signal — untestable, and unbacktestable. Cooldown, session gating, and
portfolio limits live in `risk.py`/`runner.py`, the layers that own that state.

**Prefilters must be supersets.** Each strategy pairs `evaluate()` with a
vectorised `prefilter()` used to skip rows in backtests. If a prefilter ever
rejects a row `evaluate()` would accept, the backtest skips trades live takes.
`TestPrefilterIsSuperset` proves this for every registered strategy — register
new strategies in `STRATEGIES`/`PREFILTERS` so they are covered.

**Broker is a Protocol.** `KiteBroker` (live) and `PaperBroker` (simulated fills
over real market data) are interchangeable. `PaperBroker.poll()` is its stand-in
for the exchange; live mode has no equivalent, which is why `execution.py` guards
it with `hasattr`.

## Non-negotiable invariants

Each maps to a numbered NUANCES.md entry and a test. Do not remove or work around:

| Invariant | Where | Nuance |
|---|---|---|
| Every wire price passes `round_to_tick` | `broker.py` | #1 |
| Broker is truth; reconcile at startup | `execution.py:reconcile` | #2 |
| Place new stop *before* cancelling old | `execution.py:move_stop` | #3 |
| VWAP resets via `.over("session_date")` | `indicators.py` | #4 |
| Cooldown after exit | `risk.py` | #5 |
| Live evaluates row `-2`, never the forming bar | `runner.py` | #6 |
| Margin from `net` | `broker.py` | #7 |
| ADX filters strength, never direction | `signals.py` | #8 |
| Size from stop distance, not capital | `risk.py:size_position` | #19 |
| Loss-streak/daily halts persist across restart | `risk.py`, `runner.py` | #20 |

Two rules govern `execution.py` specifically: an unprotected position is an
emergency (two stops is fine, zero is not), and on any disagreement with the
broker, local state is wrong.

## Backtest honesty rules

Baked in; removing any of them inflates results:

1. A signal on bar *i* fills at bar *i+1*'s **open**. Filling at bar *i*'s close
   is lookahead.
2. When a bar's range spans both stop and target, the **stop** is assumed first.
3. Costs are charged on every trade via the shared `costs.py` — the same function
   live execution uses.

A backtest that finds edge in random-walk data is broken. `examples/full_system.py`
and the synthetic fixtures exercise exactly that check.

## Facts established against live data (2026-08-17)

These were found by running against a real Kite account and are not in any
document that predates it:

- **NSE moved the equity close from 15:30 to 15:15 on 2026-08-03.** Sessions went
  from 375 one-minute candles to 360. Verified across RELIANCE, TCS and HDFCBANK
  over 52 sessions. `session.SESSION_CLOSE_HISTORY` holds both values and
  `market_close_for(day)` picks the one in force, so backtests spanning the
  change stay correct. Defaults moved with it: squareoff 15:10 → 14:55, last
  entry 14:45 → 14:30.
- **Kite returns tz-aware IST datetimes.** `replace_time_zone(None)` drops the
  zone *without converting*, so 09:15 IST became 03:45 and every session gate
  reported "pre-open" — backtests returned zero trades with no error.
  `broker._normalise_candles` converts then strips; do not bypass it.
- **Tick sizes are not universally 0.05.** RELIANCE and INFY are 0.10. Always
  read from `InstrumentMaster`, never assume.
- **The default `fortress` parameters lost money on Nifty 50 largecaps** over
  2026-06-20 → 08-17: -10.4% on 1-minute bars, -6.9% on 15-minute. Costs were
  54% of a winning trade's gross move at 1-minute resolution. The window was a
  sideways market (NIFTY +1.5% in a 4.9% band over 40 sessions), which is the
  regime a trend-following strategy is expected to bleed in. Treat the shipped
  parameters as untested starting points, not a working strategy.

## Gotchas discovered here

- **`dt.hour()` returns Int8.** `hour * 60` overflows (540 wraps to 28), silently
  producing an always-true comparison. Cast to `Int32` before arithmetic on
  datetime components.
- **`prepare()` keeps all columns, not an allowlist.** Custom strategy columns
  (opening ranges, higher-timeframe trends) used to be dropped, so the strategy
  saw `None` on every row and produced no trades, no error, and nothing in the
  rejection counts.
- **`run.sh`/`start.sh` resolve symlinks** before computing `SCRIPT_DIR`. Without
  it, invoking the symlinked copy under `~/.claude/skills/` builds a second
  275 MB venv there.
- **Polars renamed `min_periods` to `min_samples`** in 1.21.

## The NUANCE numbering convention

`NUANCES.md` gotchas are numbered 1–30 and cited by number from `engine/`,
`templates/`, and `examples/`. The 10 `KNOWLEDGE.md` sections work the same way.
**Append; never renumber or reorder.** Match the existing format — the mistake as
wrong code, the observed symptom, the fix as right code, the real-world impact —
and use only parameters that came from actual testing.

## External dependencies and their failure modes

- **NSE constituents** come from `nsearchives.nseindia.com/content/indices/*.csv`.
  The old `api/equity-stockIndices` JSON endpoint is dead (403 on the homepage
  defeats cookie priming; the API returns 404). The CSV needs only a browser
  User-Agent. `INDEX_FILES` carries an expected row count per index and rejects
  responses outside ±15% — verified against live data for all 8 indices.
  **Failures raise; there is no silent fallback to a hardcoded list.**
- **Kite access tokens expire daily ~07:30 IST.** `login` → `token` is a morning
  ritual. Historical data and live quotes require the **paid Connect plan**; the
  free Personal plan excludes both.
- **NSE holidays** have no API. `config/holidays.json` is hand-maintained; an
  empty list only means the bot idles on a holiday, never a wrong trade.

## Documentation caveats

`QUICKSTART.md`, `SETUP_COMPLETE.md`, `FINAL_SUMMARY.md`, and
`START_SCRIPT_GUIDE.md` are one-time setup logs from the original author's
machine — they reference `/home/rakesh/...`, and their file inventories predate
`engine/`. Treat them as history. `README.md` and `SKILL.md` are the maintained
entry points.

Performance figures quoted in the older docs (28x Parquet, 37x Polars, 65% win
rate, 51% CAGR) describe a previous implementation. The first two are plausible
and directionally reproduced here; the last two are historical claims about a
different codebase and must not be presented as properties of this one.
