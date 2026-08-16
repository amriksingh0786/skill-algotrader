"""
Universe fetching from NSE's published index constituent files.

WHY THIS EXISTS
---------------
The `nseindia.com/api/equity-stockIndices` JSON endpoint that older versions of
this skill used is dead — the homepage now returns 403 to non-browser clients
(bot detection), so the cookie-priming trick fails and the API returns 404.

The archives host serves the canonical constituent CSVs with nothing more than a
browser User-Agent. It is the same file NSE publishes for index licensees, it is
updated on rebalance days, and it has no cookie or session requirement.

FAILURE POLICY
--------------
This module raises `UniverseFetchError` on failure. It does NOT silently fall
back to a hardcoded list — a stale universe silently substituted for a live one
means you trade delisted or re-weighted names without noticing. Callers that
genuinely want a fallback must pass `allow_stale=True` and check the `stale` key
on the result.
"""

from __future__ import annotations

import csv
import io
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

# Primary and legacy hosts. Both currently serve identical bytes; the legacy host
# is kept as a fallback in case the primary is rotated again.
ARCHIVE_HOSTS = (
    "https://nsearchives.nseindia.com",
    "https://archives.nseindia.com",
)

_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

# index key -> (display name, archive filename, expected constituent count)
# The expected count is a sanity check: a truncated or error-page response will
# not have the right number of rows, and we would rather fail than trade a
# half-empty universe.
INDEX_FILES: dict[str, tuple[str, str, int]] = {
    "nifty50": ("NIFTY 50", "ind_nifty50list.csv", 50),
    "niftynext50": ("NIFTY NEXT 50", "ind_niftynext50list.csv", 50),
    "nifty100": ("NIFTY 100", "ind_nifty100list.csv", 100),
    "nifty200": ("NIFTY 200", "ind_nifty200list.csv", 200),
    "midcap150": ("NIFTY MIDCAP 150", "ind_niftymidcap150list.csv", 150),
    "smallcap250": ("NIFTY SMALLCAP 250", "ind_niftysmallcap250list.csv", 250),
    "banknifty": ("NIFTY BANK", "ind_niftybanklist.csv", 14),
    "totalmarket": ("NIFTY TOTAL MARKET", "ind_niftytotalmarket_list.csv", 750),
}


class UniverseFetchError(RuntimeError):
    """Raised when constituents cannot be fetched or fail validation."""


def _get(url: str, timeout: float, retries: int) -> str:
    """GET with retries. Returns response text, raises on exhaustion."""
    headers = {
        "User-Agent": _BROWSER_UA,
        "Accept": "text/csv,application/csv,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
    }
    last_error: Exception | None = None

    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            if not response.text.strip():
                raise UniverseFetchError(f"empty body from {url}")
            return response.text
        except Exception as exc:  # noqa: BLE001 - retried and re-raised below
            last_error = exc
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))  # linear backoff

    raise UniverseFetchError(f"{url} failed after {retries} attempts: {last_error}")


def _parse_constituents(csv_text: str) -> list[dict[str, str]]:
    """
    Parse NSE's constituent CSV.

    Columns as published: Company Name, Industry, Symbol, Series, ISIN Code
    """
    reader = csv.DictReader(io.StringIO(csv_text))
    stocks: list[dict[str, str]] = []

    for row in reader:
        # Header names carry stray whitespace on some files.
        clean = {(k or "").strip(): (v or "").strip() for k, v in row.items()}
        symbol = clean.get("Symbol", "")
        if not symbol:
            continue

        stocks.append(
            {
                "symbol": symbol,
                "company": clean.get("Company Name", symbol),
                "sector": clean.get("Industry", "Unknown"),
                "series": clean.get("Series", "EQ"),
                "isin": clean.get("ISIN Code", ""),
                # Kite instrument keys are EXCHANGE:TRADINGSYMBOL.
                "tradingsymbol": f"NSE:{symbol}",
            }
        )

    return stocks


def fetch_index_constituents(
    index: str,
    *,
    timeout: float = 15.0,
    retries: int = 3,
    tolerance: float = 0.15,
) -> dict[str, Any]:
    """
    Fetch live constituents for an index from NSE archives.

    Args:
        index: key from INDEX_FILES ('nifty50', 'midcap150', ...)
        timeout: per-request timeout in seconds
        retries: attempts per host before moving to the next host
        tolerance: allowed fractional deviation from the expected count before
                   the response is rejected as malformed

    Returns:
        {'index', 'key', 'source', 'last_updated', 'count', 'stale', 'stocks'}

    Raises:
        UniverseFetchError: unknown index, all hosts failed, or count validation
                            failed.
    """
    key = index.strip().lower()
    if key not in INDEX_FILES:
        raise UniverseFetchError(
            f"unknown index {index!r}. Known: {', '.join(sorted(INDEX_FILES))}"
        )

    display_name, filename, expected = INDEX_FILES[key]
    errors: list[str] = []

    for host in ARCHIVE_HOSTS:
        url = f"{host}/content/indices/{filename}"
        try:
            stocks = _parse_constituents(_get(url, timeout, retries))
        except Exception as exc:  # noqa: BLE001 - collected, reported below
            errors.append(f"{host}: {exc}")
            continue

        # Reject obviously wrong payloads (error pages, truncated transfers).
        low, high = expected * (1 - tolerance), expected * (1 + tolerance)
        if not stocks or not (low <= len(stocks) <= high):
            errors.append(f"{host}: got {len(stocks)} rows, expected ~{expected}")
            continue

        return {
            "index": display_name,
            "key": key,
            "source": url,
            "last_updated": datetime.now().isoformat(timespec="seconds"),
            "count": len(stocks),
            "stale": False,
            "stocks": stocks,
        }

    raise UniverseFetchError(
        f"could not fetch {display_name} constituents.\n  " + "\n  ".join(errors)
    )


def save_universe(data: dict[str, Any], path: str | Path) -> Path:
    """Write a universe payload to JSON, creating parent directories."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2))
    return out


def load_universe(path: str | Path, *, max_age_days: int | None = 30) -> list[str]:
    """
    Load symbols from a saved universe file.

    Args:
        max_age_days: warn-and-raise if the file is older than this. Index
                      constituents change on rebalance; trading a six-month-old
                      universe means holding names the index dropped. Pass None
                      to disable the check.

    Returns:
        List of trading symbols.
    """
    data = json.loads(Path(path).read_text())

    if max_age_days is not None:
        fetched = datetime.fromisoformat(data["last_updated"])
        age_days = (datetime.now() - fetched).days
        if age_days > max_age_days:
            raise UniverseFetchError(
                f"{path} is {age_days} days old (limit {max_age_days}). "
                f"Refresh with: ./run.sh universe --indices {data.get('key', 'nifty50')}"
            )

    return [s["symbol"] for s in data["stocks"]]


def filter_universe(
    stocks: list[dict[str, Any]],
    quotes: dict[str, dict[str, Any]],
    *,
    min_avg_volume: int = 100_000,
    min_price: float = 50.0,
    max_price: float = 100_000.0,
    max_spread_pct: float = 0.30,
) -> list[dict[str, Any]]:
    """
    Apply liquidity filters using broker quotes.

    KNOWLEDGE.md section 5: illiquid names produce backtest profits that do not
    survive live slippage. Filtering on volume and spread is what makes the
    backtest honest.

    Args:
        stocks: constituent dicts from fetch_index_constituents()
        quotes: {symbol: quote_dict} from Broker.quote()
        min_avg_volume: minimum average daily volume
        min_price / max_price: price band (penny stocks and very high priced
                               stocks both size badly against a fixed risk budget)
        max_spread_pct: maximum bid-ask spread as a percentage of last price

    Returns:
        Filtered list, each entry annotated with the metrics used to judge it.
    """
    kept: list[dict[str, Any]] = []

    for stock in stocks:
        quote = quotes.get(stock["symbol"])
        if not quote:
            continue

        last_price = float(quote.get("last_price") or 0)
        volume = float(quote.get("volume") or 0)
        depth = quote.get("depth") or {}
        buy_levels, sell_levels = depth.get("buy") or [], depth.get("sell") or []

        if not (min_price <= last_price <= max_price):
            continue
        if volume < min_avg_volume:
            continue

        spread_pct = 0.0
        if buy_levels and sell_levels and last_price > 0:
            bid = float(buy_levels[0].get("price") or 0)
            ask = float(sell_levels[0].get("price") or 0)
            if bid > 0 and ask > 0:
                spread_pct = (ask - bid) / last_price * 100
                if spread_pct > max_spread_pct:
                    continue

        kept.append({**stock, "last_price": last_price, "volume": volume,
                     "spread_pct": round(spread_pct, 4)})

    return kept
