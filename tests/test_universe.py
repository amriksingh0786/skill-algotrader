"""
Universe fetching and validation.

The failure policy is the important part: a stale universe substituted silently
for a live one means trading names the index dropped. These tests pin that a
failure raises rather than degrades.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from engine.universe import (
    INDEX_FILES,
    UniverseFetchError,
    _parse_constituents,
    fetch_index_constituents,
    filter_universe,
    load_universe,
    save_universe,
)

SAMPLE_CSV = """Company Name,Industry,Symbol,Series,ISIN Code
Reliance Industries Ltd.,Oil Gas & Consumable Fuels,RELIANCE,EQ,INE002A01018
Tata Consultancy Services Ltd.,Information Technology,TCS,EQ,INE467B01029
HDFC Bank Ltd.,Financial Services,HDFCBANK,EQ,INE040A01034
"""


class TestParsing:
    def test_parses_published_columns(self) -> None:
        stocks = _parse_constituents(SAMPLE_CSV)
        assert len(stocks) == 3
        assert stocks[0]["symbol"] == "RELIANCE"
        assert stocks[0]["company"] == "Reliance Industries Ltd."
        assert stocks[0]["sector"] == "Oil Gas & Consumable Fuels"
        assert stocks[0]["isin"] == "INE002A01018"

    def test_builds_kite_instrument_keys(self) -> None:
        """Kite addresses instruments as EXCHANGE:TRADINGSYMBOL."""
        assert _parse_constituents(SAMPLE_CSV)[0]["tradingsymbol"] == "NSE:RELIANCE"

    def test_skips_rows_without_a_symbol(self) -> None:
        csv_text = SAMPLE_CSV + ",,,,\n"
        assert len(_parse_constituents(csv_text)) == 3

    def test_tolerates_whitespace_in_headers(self) -> None:
        messy = SAMPLE_CSV.replace("Company Name", " Company Name ")
        assert _parse_constituents(messy)[0]["company"] == "Reliance Industries Ltd."


class TestFetchFailure:
    def test_unknown_index_raises(self) -> None:
        with pytest.raises(UniverseFetchError, match="unknown index"):
            fetch_index_constituents("nifty_does_not_exist")

    def test_all_known_indices_have_expected_counts(self) -> None:
        for key, (name, filename, expected) in INDEX_FILES.items():
            assert filename.endswith(".csv")
            assert expected > 0, f"{key} needs an expected count for validation"

    def test_no_silent_fallback(self, monkeypatch) -> None:
        """
        A network failure must raise. The old implementation returned a hardcoded
        January list here, which meant trading a stale universe without knowing.
        """
        def boom(*args, **kwargs):
            raise RuntimeError("network down")

        monkeypatch.setattr("engine.universe._get", boom)

        with pytest.raises(UniverseFetchError):
            fetch_index_constituents("nifty50")

    def test_wrong_row_count_is_rejected(self, monkeypatch) -> None:
        """A truncated transfer or an error page must not pass as a universe."""
        monkeypatch.setattr("engine.universe._get", lambda *a, **k: SAMPLE_CSV)

        with pytest.raises(UniverseFetchError, match="expected"):
            fetch_index_constituents("nifty50")  # 3 rows where ~50 are required

    def test_count_within_tolerance_is_accepted(self, monkeypatch) -> None:
        """Index membership drifts by a name or two between rebalances."""
        header = "Company Name,Industry,Symbol,Series,ISIN Code\n"
        rows = "".join(f"Co {i},Sector,SYM{i},EQ,INE{i:09d}\n" for i in range(48))
        monkeypatch.setattr("engine.universe._get", lambda *a, **k: header + rows)

        data = fetch_index_constituents("nifty50")
        assert data["count"] == 48
        assert data["stale"] is False


class TestSaveLoad:
    def test_round_trip(self, tmp_path, monkeypatch) -> None:
        header = "Company Name,Industry,Symbol,Series,ISIN Code\n"
        rows = "".join(f"Co {i},Sector,SYM{i},EQ,INE{i:09d}\n" for i in range(50))
        monkeypatch.setattr("engine.universe._get", lambda *a, **k: header + rows)

        data = fetch_index_constituents("nifty50")
        path = save_universe(data, tmp_path / "nifty50.json")

        symbols = load_universe(path)
        assert len(symbols) == 50
        assert symbols[0] == "SYM0"

    def test_stale_file_is_rejected(self, tmp_path) -> None:
        """Constituents change on rebalance; a six-month-old file is not a universe."""
        old = {
            "index": "NIFTY 50", "key": "nifty50",
            "last_updated": (datetime.now() - timedelta(days=200)).isoformat(),
            "count": 1, "stocks": [{"symbol": "RELIANCE"}],
        }
        path = tmp_path / "old.json"
        path.write_text(json.dumps(old))

        with pytest.raises(UniverseFetchError, match="days old"):
            load_universe(path, max_age_days=30)

        assert load_universe(path, max_age_days=None) == ["RELIANCE"]


class TestLiquidityFilter:
    def _stocks(self) -> list[dict]:
        return [{"symbol": s, "company": s, "sector": "X"} for s in ("LIQUID", "THIN", "WIDE")]

    def test_filters_on_volume_and_spread(self) -> None:
        quotes = {
            "LIQUID": {"last_price": 1000.0, "volume": 5_000_000,
                       "depth": {"buy": [{"price": 999.9}], "sell": [{"price": 1000.1}]}},
            "THIN": {"last_price": 1000.0, "volume": 5_000,
                     "depth": {"buy": [{"price": 999.9}], "sell": [{"price": 1000.1}]}},
            "WIDE": {"last_price": 1000.0, "volume": 5_000_000,
                     "depth": {"buy": [{"price": 995.0}], "sell": [{"price": 1005.0}]}},
        }
        kept = filter_universe(self._stocks(), quotes, min_avg_volume=100_000,
                               max_spread_pct=0.30)
        assert [s["symbol"] for s in kept] == ["LIQUID"]

    def test_price_band_excludes_penny_stocks(self) -> None:
        quotes = {"LIQUID": {"last_price": 5.0, "volume": 10_000_000, "depth": {}}}
        kept = filter_universe([{"symbol": "LIQUID"}], quotes, min_price=50.0)
        assert kept == []

    def test_missing_quote_drops_the_symbol(self) -> None:
        assert filter_universe(self._stocks(), {}) == []

    def test_annotates_kept_symbols_with_metrics(self) -> None:
        quotes = {"LIQUID": {"last_price": 1000.0, "volume": 5_000_000,
                             "depth": {"buy": [{"price": 999.5}], "sell": [{"price": 1000.5}]}}}
        kept = filter_universe([{"symbol": "LIQUID"}], quotes)
        assert kept[0]["spread_pct"] == pytest.approx(0.1, abs=0.01)
