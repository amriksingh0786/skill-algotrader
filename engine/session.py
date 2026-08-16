"""
NSE session timing, candle completion, and clock sanity.

Every function here takes an explicit `now` rather than reading the clock, so
backtest and live run the identical code path (KNOWLEDGE.md section 2). The live
runner passes datetime.now(IST); the backtest passes the candle timestamp.
"""

from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

PRE_OPEN_START = time(9, 0)
MARKET_OPEN = time(9, 15)

# NSE shortened the equity session on 2026-08-03: the close moved from 15:30 to
# 15:15, taking sessions from 375 one-minute candles to 360. Verified against
# live Kite data across RELIANCE, TCS and HDFCBANK over 52 sessions.
#
# Backtests that span the change need the close that applied on each day, or
# every pre-August session gets squared off 15 minutes early — use
# `market_close_for(day)` rather than the constant. Newest entry first.
SESSION_CLOSE_HISTORY: tuple[tuple[date, time], ...] = (
    (date(2026, 8, 3), time(15, 15)),
    (date(1900, 1, 1), time(15, 30)),
)

MARKET_CLOSE = SESSION_CLOSE_HISTORY[0][1]

# NUANCE #28: the first 15 minutes are wild — gaps, auction spillover, and
# indicator values computed off two or three candles. Most strategies do better
# waiting for 9:30.
DEFAULT_ENTRY_START = time(9, 30)

# NUANCE #9: 11:30-13:00 is the lunch lull. Low volume, choppy, and the source of
# a disproportionate share of losing trades. These bounds were calibrated on the
# old 09:15-15:30 session; on the shorter session they cover proportionally more
# of the day, so re-check them against your own by-hour attribution before
# trusting them (`./run.sh report` breaks P&L down by entry hour).
LUNCH_START = time(11, 30)
LUNCH_END = time(13, 0)

# Square off ahead of the broker's own MIS auto-square-off. With a 15:15 close
# that cutoff moved earlier too, so 14:55 leaves room to exit on your own terms —
# a forced square-off fills at whatever the book offers.
DEFAULT_SQUAREOFF = time(14, 55)
DEFAULT_LAST_ENTRY = time(14, 30)


def market_close_for(day: date) -> time:
    """
    Closing time in effect on a given date.

    Exists because NSE changed it mid-2026 and a backtest spanning the change
    would otherwise apply today's rules to last month's data.
    """
    for effective_from, close_time in SESSION_CLOSE_HISTORY:
        if day >= effective_from:
            return close_time
    return SESSION_CLOSE_HISTORY[-1][1]


def now_ist() -> datetime:
    """Current time in IST. The only place the wall clock is read."""
    return datetime.now(IST)


def to_ist(dt: datetime) -> datetime:
    """Normalise any datetime to IST, assuming naive datetimes are already IST."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=IST)
    return dt.astimezone(IST)


class TradingCalendar:
    """
    Trading-day calendar: weekends plus an explicit NSE holiday list.

    NSE publishes holidays annually and there is no stable API for them, so the
    list is a maintained JSON file. An empty or outdated list means the bot wakes
    up on a holiday, finds no ticks, and idles — noisy but not dangerous. It
    never causes a wrong trade.
    """

    def __init__(self, holidays: set[date] | None = None) -> None:
        self.holidays = holidays or set()

    @classmethod
    def from_file(cls, path: str | Path) -> "TradingCalendar":
        """Load ["2026-01-26", ...] from JSON. Missing file yields weekends-only."""
        file_path = Path(path)
        if not file_path.exists():
            return cls()

        raw = json.loads(file_path.read_text())
        entries = raw.get("holidays", raw) if isinstance(raw, dict) else raw
        return cls({date.fromisoformat(d) for d in entries})

    def is_trading_day(self, day: date) -> bool:
        return day.weekday() < 5 and day not in self.holidays

    def previous_trading_day(self, day: date) -> date:
        cursor = day - timedelta(days=1)
        while not self.is_trading_day(cursor):
            cursor -= timedelta(days=1)
        return cursor


def is_market_open(now: datetime, calendar: TradingCalendar | None = None) -> bool:
    """True during continuous trading (09:15-15:30) on a trading day."""
    now = to_ist(now)
    calendar = calendar or TradingCalendar()

    if not calendar.is_trading_day(now.date()):
        return False

    return MARKET_OPEN <= now.time() < market_close_for(now.date())


def is_pre_open(now: datetime) -> bool:
    """
    NUANCE #27: 09:00-09:15 is the pre-open auction. Quotes exist but are
    indicative — placing orders against them is a good way to get filled at a
    price that never traded.
    """
    return PRE_OPEN_START <= to_ist(now).time() < MARKET_OPEN


def should_trade_now(
    now: datetime,
    *,
    calendar: TradingCalendar | None = None,
    entry_start: time = DEFAULT_ENTRY_START,
    last_entry: time = DEFAULT_LAST_ENTRY,
    skip_lunch: bool = True,
) -> tuple[bool, str]:
    """
    Should we look for NEW entries right now?

    This gates entries only. Exits and stop-loss management must keep running
    whenever the market is open — never gate them on this function.

    Returns:
        (allowed, reason). The reason is logged so that a quiet session is
        explainable after the fact rather than a mystery.
    """
    now = to_ist(now)
    calendar = calendar or TradingCalendar()

    if not calendar.is_trading_day(now.date()):
        return False, "not a trading day"

    current = now.time()

    if current < MARKET_OPEN:
        return False, "pre-open, market not yet open"
    if current >= market_close_for(now.date()):
        return False, "market closed"
    if current < entry_start:
        return False, f"opening volatility window until {entry_start:%H:%M}"
    if skip_lunch and LUNCH_START <= current < LUNCH_END:
        return False, "lunch lull (11:30-13:00)"
    if current >= last_entry:
        return False, f"past last entry time {last_entry:%H:%M}"

    return True, "ok"


def should_squareoff(now: datetime, squareoff_time: time = DEFAULT_SQUAREOFF) -> bool:
    """True once intraday positions must be closed."""
    return to_ist(now).time() >= squareoff_time


def is_candle_complete(
    candle_start: datetime,
    now: datetime,
    interval_seconds: int,
    buffer_ms: int = 500,
) -> bool:
    """
    NUANCE #6: has the candle that STARTED at `candle_start` finished?

    Acting on a partially formed candle is the classic phantom-signal bug: the
    candle looks like a breakout at 09:47:10 and closes red at 09:48:00. The
    buffer absorbs the lag between the exchange sealing a candle and the broker
    serving it.

    Args:
        candle_start: timestamp of the candle's opening tick
        now: current time
        interval_seconds: candle width (60 for 1-minute)
        buffer_ms: grace period after the theoretical close
    """
    candle_end = to_ist(candle_start) + timedelta(seconds=interval_seconds)
    return to_ist(now) >= candle_end + timedelta(milliseconds=buffer_ms)


def last_complete_candle_time(now: datetime, interval_seconds: int,
                              buffer_ms: int = 500) -> datetime:
    """Start timestamp of the most recent candle that is definitely complete."""
    now = to_ist(now)
    epoch_seconds = int(now.timestamp()) - (buffer_ms / 1000)
    bucket_start = int(epoch_seconds // interval_seconds) * interval_seconds
    return datetime.fromtimestamp(bucket_start - interval_seconds, tz=IST)


def minutes_since_open(now: datetime) -> int:
    """Minutes elapsed since 09:15. Negative before the open."""
    now = to_ist(now)
    open_dt = now.replace(hour=MARKET_OPEN.hour, minute=MARKET_OPEN.minute,
                          second=0, microsecond=0)
    return int((now - open_dt).total_seconds() // 60)


def check_clock_drift(broker_time: datetime, local_time: datetime,
                      max_drift_seconds: float = 2.0) -> tuple[bool, float]:
    """
    NUANCE #16: clock drift breaks candle alignment silently.

    If the local clock runs fast, `is_candle_complete` returns True early and the
    bot trades partial candles — reproducing the phantom-signal bug even though
    the buffer logic is correct. Check at startup and refuse to trade if the
    machine is not synced.

    Returns:
        (within_tolerance, drift_seconds) where drift is local minus broker.
    """
    drift = (to_ist(local_time) - to_ist(broker_time)).total_seconds()
    return abs(drift) <= max_drift_seconds, drift
