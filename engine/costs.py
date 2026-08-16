"""
NSE equity transaction costs.

One implementation, imported by both the backtest and live execution. A backtest
that charges different costs than the live bot pays is the most common way a
"profitable" strategy turns out to be a rounding error — KNOWLEDGE.md section 4
walks through a rebalancing scheme that was pure profit on paper and pure
brokerage in practice.

Rates as of 2026. They change with SEBI circulars and budget announcements;
verify against your contract note before trusting the numbers.
"""

from __future__ import annotations

# Zerodha: 0.03% or ₹20 per executed order, whichever is lower, for intraday.
BROKERAGE_PCT = 0.0003
BROKERAGE_CAP = 20.0

STT_INTRADAY_SELL = 0.00025  # NUANCE #30: sell side only
STT_DELIVERY_BOTH = 0.001    # both sides for delivery

EXCHANGE_TXN_PCT = 0.0000297  # NSE
SEBI_FEE_PCT = 0.000001
STAMP_DUTY_INTRADAY = 0.00003   # buy side only
STAMP_DUTY_DELIVERY = 0.00015   # buy side only
GST_PCT = 0.18                  # on brokerage + exchange + SEBI


def estimate_costs(
    entry_price: float, exit_price: float, quantity: int, *, intraday: bool = True
) -> float:
    """
    Total round-trip cost in rupees.

    Args:
        entry_price / exit_price: fill prices
        quantity: shares (one leg — the round trip is both legs)
        intraday: MIS/intraday cost stack if True, delivery if False
    """
    if quantity <= 0:
        return 0.0

    entry_value = entry_price * quantity
    exit_value = exit_price * quantity
    turnover = entry_value + exit_value

    if intraday:
        brokerage = min(BROKERAGE_CAP, entry_value * BROKERAGE_PCT) + min(
            BROKERAGE_CAP, exit_value * BROKERAGE_PCT
        )
        stt = exit_value * STT_INTRADAY_SELL
        stamp_duty = entry_value * STAMP_DUTY_INTRADAY
    else:
        brokerage = 0.0  # zero-brokerage delivery is the common retail plan
        stt = turnover * STT_DELIVERY_BOTH
        stamp_duty = entry_value * STAMP_DUTY_DELIVERY

    exchange_fee = turnover * EXCHANGE_TXN_PCT
    sebi_fee = turnover * SEBI_FEE_PCT
    gst = (brokerage + exchange_fee + sebi_fee) * GST_PCT

    return brokerage + stt + exchange_fee + sebi_fee + stamp_duty + gst


# ---------------------------------------------------------------------------
# F&O (NFO). Rates reflect the October 2024 STT increase.
# Options are charged on PREMIUM, not on notional — a 25,000-strike NIFTY call
# at Rs 120 with lot size 75 has a notional of ~Rs 18.7L but a premium turnover
# of Rs 9,000, and every percentage below applies to the latter.
# ---------------------------------------------------------------------------

FUT_BROKERAGE_PCT = 0.0003
FUT_BROKERAGE_CAP = 20.0
FUT_STT_SELL = 0.0002          # 0.02% on sell, on notional
FUT_EXCHANGE_TXN = 0.0000173   # NSE futures
FUT_STAMP_BUY = 0.00002

OPT_BROKERAGE_FLAT = 20.0      # per executed order, not percentage
OPT_STT_SELL = 0.001           # 0.10% on sell, on PREMIUM
OPT_EXCHANGE_TXN = 0.0003503   # NSE options, on premium
OPT_STAMP_BUY = 0.00003


def estimate_futures_costs(entry_price: float, exit_price: float, quantity: int) -> float:
    """Round-trip cost for one futures position (quantity = lots x lot_size)."""
    if quantity <= 0:
        return 0.0

    entry_value, exit_value = entry_price * quantity, exit_price * quantity
    turnover = entry_value + exit_value

    brokerage = min(FUT_BROKERAGE_CAP, entry_value * FUT_BROKERAGE_PCT) + min(
        FUT_BROKERAGE_CAP, exit_value * FUT_BROKERAGE_PCT
    )
    stt = exit_value * FUT_STT_SELL
    exchange_fee = turnover * FUT_EXCHANGE_TXN
    sebi_fee = turnover * SEBI_FEE_PCT
    stamp_duty = entry_value * FUT_STAMP_BUY
    gst = (brokerage + exchange_fee + sebi_fee) * GST_PCT

    return brokerage + stt + exchange_fee + sebi_fee + stamp_duty + gst


def estimate_options_costs(entry_premium: float, exit_premium: float, quantity: int) -> float:
    """
    Round-trip cost for a long option position.

    Costs are brutal relative to premium: on a Rs 100 premium with a 75 lot, the
    round trip is roughly Rs 60-70 against Rs 7,500 of turnover — close to 1%.
    A strategy targeting a 5% move in premium gives up a fifth of it to costs
    before slippage, and option spreads are far wider than equity spreads.
    """
    if quantity <= 0:
        return 0.0

    entry_value, exit_value = entry_premium * quantity, exit_premium * quantity
    turnover = entry_value + exit_value

    brokerage = OPT_BROKERAGE_FLAT * 2
    stt = exit_value * OPT_STT_SELL
    exchange_fee = turnover * OPT_EXCHANGE_TXN
    sebi_fee = turnover * SEBI_FEE_PCT
    stamp_duty = entry_value * OPT_STAMP_BUY
    gst = (brokerage + exchange_fee + sebi_fee) * GST_PCT

    return brokerage + stt + exchange_fee + sebi_fee + stamp_duty + gst


def estimate_costs_for_segment(
    entry_price: float, exit_price: float, quantity: int, *,
    segment: str = "equity", intraday: bool = True,
) -> float:
    """Dispatch to the right cost model. `segment` is equity | futures | options."""
    if segment == "futures":
        return estimate_futures_costs(entry_price, exit_price, quantity)
    if segment == "options":
        return estimate_options_costs(entry_price, exit_price, quantity)
    return estimate_costs(entry_price, exit_price, quantity, intraday=intraday)


def breakeven_move_pct(price: float, quantity: int, *, intraday: bool = True) -> float:
    """
    Percentage move needed just to cover costs.

    Worth checking before committing to a strategy: if the average winner is
    0.35% and costs eat 0.12%, a third of the edge is gone before slippage. This
    is the number that kills high-frequency retail strategies.
    """
    if price <= 0 or quantity <= 0:
        return 0.0
    costs = estimate_costs(price, price, quantity, intraday=intraday)
    return costs / (price * quantity) * 100
