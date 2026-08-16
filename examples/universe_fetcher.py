#!/usr/bin/env python3
"""
Fetching index constituents from NSE.

    python examples/universe_fetcher.py

HISTORY: this file used to carry its own copy of the fetch logic, pointed at
`nseindia.com/api/equity-stockIndices`. That endpoint is dead — the homepage now
returns 403 to non-browser clients, so the cookie-priming step fails and the API
answers 404. Worse, the old code fell back to a hardcoded list on failure, so it
kept "working" while silently serving a stale universe.

It now delegates to engine/universe.py. One implementation, one place to fix.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.universe import (
    INDEX_FILES,
    UniverseFetchError,
    fetch_index_constituents,
    filter_universe,
    save_universe,
)


def main() -> int:
    print("Available indices:", ", ".join(sorted(INDEX_FILES)))

    for index in ("nifty50", "midcap150"):
        try:
            data = fetch_index_constituents(index)
        except UniverseFetchError as exc:
            # Note what is NOT happening here: no fallback to a hardcoded list.
            # Trading a stale universe is worse than not trading.
            print(f"\n{index}: FAILED — {exc}")
            continue

        print(f"\n{data['index']}: {data['count']} constituents")
        print(f"  source: {data['source']}")
        for stock in data["stocks"][:5]:
            print(f"  {stock['symbol']:14s} {stock['sector']:32s} {stock['company']}")

        path = save_universe(data, f"universe/{index}.json")
        print(f"  saved -> {path}")

    print(
        "\nLiquidity filtering needs live quotes, so it requires a broker:\n"
        "    from engine.broker import KiteBroker\n"
        "    from engine.universe import filter_universe\n"
        "    broker = KiteBroker()\n"
        "    quotes = broker.quote([s['symbol'] for s in data['stocks']])\n"
        "    tradeable = filter_universe(data['stocks'], quotes,\n"
        "                                min_avg_volume=100_000, max_spread_pct=0.3)\n"
        "\nKNOWLEDGE.md section 5: illiquid names produce backtest profits that do\n"
        "not survive live slippage. Filter before you backtest, not after."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
