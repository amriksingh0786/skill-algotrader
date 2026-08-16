"""
F&O contract resolution and cost models.

STATUS: contract resolution and costs are implemented and tested. The live
runner does NOT yet route orders through them — see SKILL.md. These tests pin
the behaviour so the wiring, when it happens, has something to build on.
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from engine.costs import (
    estimate_costs,
    estimate_costs_for_segment,
    estimate_futures_costs,
    estimate_options_costs,
)
from engine.derivatives import (
    Contract,
    ContractError,
    DerivativeMaster,
    lots_to_quantity,
    physical_settlement_risk,
    quantity_to_lots,
)


class FakeKite:
    """Serves a synthetic NFO dump: NIFTY (index) and RELIANCE (stock) F&O."""

    def __init__(self) -> None:
        today = date.today()
        self.weekly = today + timedelta(days=3)
        self.monthly = today + timedelta(days=24)
        self.far_monthly = today + timedelta(days=52)
        self.records: list[dict] = []
        token = 1000

        for expiry in (self.monthly, self.far_monthly):
            for name, lot in (("NIFTY", 75), ("RELIANCE", 500)):
                token += 1
                self.records.append({
                    "instrument_token": token, "tradingsymbol": f"{name}{expiry:%y%b}FUT".upper(),
                    "name": name, "expiry": expiry, "strike": 0.0, "lot_size": lot,
                    "tick_size": 0.05, "instrument_type": "FUT", "exchange": "NFO",
                    "segment": "NFO-FUT",
                })

        for expiry in (self.weekly, self.monthly):
            for strike in range(24_500, 25_600, 50):
                for option_type in ("CE", "PE"):
                    token += 1
                    self.records.append({
                        "instrument_token": token,
                        "tradingsymbol": f"NIFTY{expiry:%y%m%d}{strike}{option_type}",
                        "name": "NIFTY", "expiry": expiry, "strike": float(strike),
                        "lot_size": 75, "tick_size": 0.05, "instrument_type": option_type,
                        "exchange": "NFO", "segment": "NFO-OPT",
                    })

        for strike in range(2800, 3200, 20):
            for option_type in ("CE", "PE"):
                token += 1
                self.records.append({
                    "instrument_token": token,
                    "tradingsymbol": f"RELIANCE{self.monthly:%y%b}{strike}{option_type}".upper(),
                    "name": "RELIANCE", "expiry": self.monthly, "strike": float(strike),
                    "lot_size": 500, "tick_size": 0.05, "instrument_type": option_type,
                    "exchange": "NFO", "segment": "NFO-OPT",
                })

    def instruments(self, exchange: str) -> list[dict]:
        return self.records


@pytest.fixture
def master(tmp_path) -> DerivativeMaster:
    instance = DerivativeMaster(tmp_path)
    instance.load(FakeKite())
    return instance


class TestExpirySelection:
    def test_lists_future_expiries_only(self, master: DerivativeMaster) -> None:
        expiries = master.expiries("NIFTY", "FUT")
        assert expiries == sorted(expiries)
        assert all(expiry >= date.today() for expiry in expiries)

    def test_nearest_and_next_differ(self, master: DerivativeMaster) -> None:
        nearest = master.select_expiry("NIFTY", "FUT", which="nearest")
        following = master.select_expiry("NIFTY", "FUT", which="next")
        assert nearest < following

    def test_min_days_skips_imminent_expiry(self, master: DerivativeMaster) -> None:
        """Expiry-day options are a different game — gamma, not direction."""
        weekly = master.select_expiry("NIFTY", "CE", which="nearest", min_days=0)
        later = master.select_expiry("NIFTY", "CE", which="nearest", min_days=7)
        assert later > weekly

    def test_impossible_min_days_raises(self, master: DerivativeMaster) -> None:
        with pytest.raises(ContractError, match="at least"):
            master.select_expiry("NIFTY", "FUT", min_days=9999)

    def test_unknown_underlying_raises(self, master: DerivativeMaster) -> None:
        with pytest.raises(ContractError):
            master.future("NOT_LISTED")


class TestContractResolution:
    def test_future_carries_lot_size_from_the_dump(self, master: DerivativeMaster) -> None:
        """Lot sizes change; reading them beats hardcoding 50 and being wrong."""
        contract = master.future("NIFTY")
        assert contract.lot_size == 75
        assert contract.instrument_type == "FUT"
        assert contract.exchange == "NFO"
        assert contract.is_index

    def test_atm_option_picks_nearest_available_strike(self, master: DerivativeMaster) -> None:
        contract = master.option("NIFTY", spot=25_013.0, option_type="CE", moneyness=0)
        assert contract.strike == 25_000.0
        assert contract.instrument_type == "CE"

    def test_moneyness_direction_is_right_sided(self, master: DerivativeMaster) -> None:
        """
        For a call, in-the-money means a LOWER strike; for a put, HIGHER.
        Getting this backwards buys far OTM options that decay to zero.
        """
        itm_call = master.option("NIFTY", 25_000.0, "CE", moneyness=-1)
        otm_call = master.option("NIFTY", 25_000.0, "CE", moneyness=+1)
        assert itm_call.strike < 25_000.0 < otm_call.strike

        itm_put = master.option("NIFTY", 25_000.0, "PE", moneyness=-1)
        otm_put = master.option("NIFTY", 25_000.0, "PE", moneyness=+1)
        assert otm_put.strike < 25_000.0 < itm_put.strike

    def test_moneyness_clamps_at_the_chain_edge(self, master: DerivativeMaster) -> None:
        contract = master.option("NIFTY", 25_000.0, "CE", moneyness=500)
        assert contract.strike == max(master.strikes("NIFTY", contract.expiry, "CE"))

    def test_resolve_long_buys_a_call_short_buys_a_put(self, master: DerivativeMaster) -> None:
        """Both directions are long premium — no naked short is ever built."""
        config = {"segment": "options", "option_moneyness": 0}
        assert master.resolve("NIFTY", "LONG", 25_000.0, config).instrument_type == "CE"
        assert master.resolve("NIFTY", "SHORT", 25_000.0, config).instrument_type == "PE"

    def test_resolve_futures(self, master: DerivativeMaster) -> None:
        contract = master.resolve("NIFTY", "LONG", 25_000.0, {"segment": "futures"})
        assert contract.instrument_type == "FUT"

    def test_resolve_rejects_equity_segment(self, master: DerivativeMaster) -> None:
        with pytest.raises(ContractError, match="not a derivative"):
            master.resolve("NIFTY", "LONG", 25_000.0, {"segment": "equity"})


class TestPhysicalSettlement:
    def _stock_contract(self, days: int) -> Contract:
        return Contract(
            tradingsymbol="RELIANCE25AUGFUT", exchange="NFO", instrument_token=1,
            lot_size=500, tick_size=0.05, instrument_type="FUT", name="RELIANCE",
            expiry=date.today() + timedelta(days=days),
        )

    def test_index_contracts_are_cash_settled(self, master: DerivativeMaster) -> None:
        assert physical_settlement_risk(master.future("NIFTY")) is None

    def test_stock_fno_warns_near_expiry(self) -> None:
        warning = physical_settlement_risk(self._stock_contract(1))
        assert warning and "PHYSICALLY SETTLED" in warning

    def test_stock_fno_warns_earlier_too(self) -> None:
        assert "margins" in (physical_settlement_risk(self._stock_contract(4)) or "")

    def test_no_warning_far_from_expiry(self) -> None:
        assert physical_settlement_risk(self._stock_contract(20)) is None


class TestLotArithmetic:
    def test_round_trip(self, master: DerivativeMaster) -> None:
        contract = master.future("NIFTY")
        assert lots_to_quantity(3, contract) == 225
        assert quantity_to_lots(225, contract) == 3

    def test_partial_lot_floors(self, master: DerivativeMaster) -> None:
        """The exchange rejects anything that is not a whole multiple."""
        assert quantity_to_lots(200, master.future("NIFTY")) == 2


class TestDerivativeCosts:
    def test_options_are_charged_on_premium_not_notional(self) -> None:
        """
        A NIFTY 25000 CE at Rs 120 with a 75 lot has ~Rs 18.7L notional but only
        Rs 9,000 of premium turnover. Charging notional would overstate costs by
        two orders of magnitude.
        """
        costs = estimate_options_costs(120.0, 130.0, 75)
        premium_turnover = (120.0 + 130.0) * 75
        assert costs < premium_turnover * 0.05
        assert costs > 40.0  # two flat Rs 20 brokerages at minimum

    def test_option_costs_are_a_large_share_of_a_small_move(self) -> None:
        """The reason scalping options rarely works for retail."""
        costs = estimate_options_costs(100.0, 105.0, 75)
        gross = (105.0 - 100.0) * 75
        assert costs / gross > 0.10

    def test_futures_stt_is_sell_side(self) -> None:
        assert estimate_futures_costs(100.0, 200.0, 75) > estimate_futures_costs(200.0, 100.0, 75)

    def test_futures_cheaper_than_equity_on_notional(self) -> None:
        """Futures carry no stamp/STT burden comparable to delivery equity."""
        assert estimate_futures_costs(1000.0, 1010.0, 500) < estimate_costs(
            1000.0, 1010.0, 500, intraday=False
        )

    def test_dispatch_matches_direct_calls(self) -> None:
        assert estimate_costs_for_segment(100.0, 110.0, 75, segment="options") == \
            estimate_options_costs(100.0, 110.0, 75)
        assert estimate_costs_for_segment(1000.0, 1010.0, 75, segment="futures") == \
            estimate_futures_costs(1000.0, 1010.0, 75)
        assert estimate_costs_for_segment(1000.0, 1010.0, 100, segment="equity") == \
            estimate_costs(1000.0, 1010.0, 100)

    def test_zero_quantity_is_free(self) -> None:
        assert estimate_futures_costs(100.0, 110.0, 0) == 0.0
        assert estimate_options_costs(100.0, 110.0, 0) == 0.0
