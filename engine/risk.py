"""
Position sizing and portfolio risk limits.

This module is the last thing between a signal and real money. Every limit here
exists because its absence cost someone a bad day (KNOWLEDGE.md section 8,
NUANCES #5, #19, #20). Limits are checked in cheapest-first order and every
rejection carries a reason string, so a quiet day is explainable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

import polars as pl

from .session import now_ist, to_ist
from .signals import Signal


@dataclass
class SizingResult:
    """Outcome of a sizing decision — quantity plus why it came out that way."""

    quantity: int
    capital_at_risk: float
    position_value: float
    reason: str
    binding_constraint: str

    @property
    def is_tradeable(self) -> bool:
        return self.quantity > 0


def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> dict[str, float]:
    """
    Kelly criterion for position sizing.

    f* = (b*p - q) / b, where b = avg_win/avg_loss, p = win rate, q = 1 - p.

    Full Kelly is the growth-optimal fraction only if your win rate and payoff
    estimates are exact. They never are, and Kelly is brutally asymmetric about
    overestimates — half Kelly gives up about a quarter of the growth for a large
    reduction in drawdown, which is why it is the recommended output here.
    """
    if avg_loss <= 0 or avg_win <= 0 or not (0 < win_rate < 1):
        return {"full_kelly": 0.0, "half_kelly": 0.0, "recommended": 0.0, "edge": 0.0}

    payoff = avg_win / avg_loss
    edge = win_rate * payoff - (1 - win_rate)
    full = edge / payoff

    full = max(0.0, min(full, 1.0))
    return {
        "full_kelly": round(full, 4),
        "half_kelly": round(full / 2, 4),
        # Cap at 25% of capital regardless of what Kelly says: a single
        # gap-through-stop at Kelly size can end the account.
        "recommended": round(min(full / 2, 0.25), 4),
        "edge": round(edge, 4),
    }


def detect_market_regime(index_df: pl.DataFrame, config: dict | None = None) -> dict[str, Any]:
    """
    Classify the market as BULL / BEAR / SIDEWAYS from index daily bars.

    Long-only intraday strategies degrade badly in bear regimes — the same
    signals fire and fail. Scaling size by regime is the cheapest available
    protection (KNOWLEDGE.md section 7).

    Args:
        index_df: NIFTY daily OHLCV, oldest first, at least 200 rows.

    Returns:
        {'regime', 'multiplier', 'reason', 'above_200dma', 'slope_pct'}
    """
    config = config or {}
    if index_df.height < 200:
        return {
            "regime": "UNKNOWN",
            "multiplier": config.get("unknown_multiplier", 0.5),
            "reason": f"only {index_df.height} bars, need 200",
            "above_200dma": None,
            "slope_pct": None,
        }

    frame = index_df.sort("timestamp").with_columns(
        [
            pl.col("close").rolling_mean(200).alias("dma200"),
            pl.col("close").rolling_mean(50).alias("dma50"),
        ]
    )
    last = frame.row(-1, named=True)
    close, dma200, dma50 = float(last["close"]), float(last["dma200"]), float(last["dma50"])

    # 20-day slope of the 50-day average: direction of the medium-term trend.
    lookback = min(20, frame.height - 1)
    past_dma50 = frame.row(-1 - lookback, named=True)["dma50"]
    slope_pct = ((dma50 - past_dma50) / past_dma50 * 100) if past_dma50 else 0.0

    if close > dma200 and slope_pct > 0.5:
        regime, multiplier = "BULL", config.get("bull_multiplier", 1.2)
        reason = f"index {((close/dma200-1)*100):.1f}% above 200DMA, 50DMA rising {slope_pct:.1f}%"
    elif close < dma200 and slope_pct < -0.5:
        regime, multiplier = "BEAR", config.get("bear_multiplier", 0.5)
        reason = f"index {((1-close/dma200)*100):.1f}% below 200DMA, 50DMA falling {slope_pct:.1f}%"
    else:
        regime, multiplier = "SIDEWAYS", config.get("sideways_multiplier", 0.8)
        reason = f"index near 200DMA, 50DMA slope {slope_pct:.1f}%"

    return {
        "regime": regime,
        "multiplier": multiplier,
        "reason": reason,
        "above_200dma": close > dma200,
        "slope_pct": round(slope_pct, 3),
    }


@dataclass
class RiskState:
    """Mutable risk state for one trading day. Persisted so restarts don't reset limits."""

    trading_day: date = field(default_factory=lambda: now_ist().date())
    consecutive_losses: int = 0
    realised_pnl: float = 0.0
    trades_today: int = 0
    cooldowns: dict[str, str] = field(default_factory=dict)  # symbol -> ISO timestamp
    halted: bool = False
    halt_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "trading_day": self.trading_day.isoformat(),
            "consecutive_losses": self.consecutive_losses,
            "realised_pnl": self.realised_pnl,
            "trades_today": self.trades_today,
            "cooldowns": self.cooldowns,
            "halted": self.halted,
            "halt_reason": self.halt_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RiskState":
        return cls(
            trading_day=date.fromisoformat(data["trading_day"]),
            consecutive_losses=int(data.get("consecutive_losses", 0)),
            realised_pnl=float(data.get("realised_pnl", 0.0)),
            trades_today=int(data.get("trades_today", 0)),
            cooldowns=dict(data.get("cooldowns", {})),
            halted=bool(data.get("halted", False)),
            halt_reason=data.get("halt_reason", ""),
        )


class RiskManager:
    """
    Enforces every portfolio-level limit.

    Config keys (all optional, defaults are the values KNOWLEDGE.md recommends):
        risk_pct                 1.0    percent of capital risked per trade
        max_risk_pct             2.0    hard ceiling; sizing is clamped to it
        max_position_pct         20.0   percent of capital in any one position
        max_portfolio_heat_pct   5.0    summed open risk across all positions
        max_positions            5      concurrent open positions
        max_trades_per_day       20     churn brake
        daily_loss_limit_pct     3.0    stop trading after losing this much
        max_consecutive_losses   3      stop after this many losses in a row
        symbol_cooldown_minutes  45     re-entry block after an exit
        sector_max_pct           30.0   percent of capital in one sector
    """

    def __init__(self, config: dict | None = None, state: RiskState | None = None) -> None:
        self.config = config or {}
        self.state = state or RiskState()

    # ------------------------------------------------------------ day rollover

    def roll_day_if_needed(self, now: datetime | None = None) -> bool:
        """Reset per-day counters at the session boundary. Returns True if rolled."""
        today = to_ist(now or now_ist()).date()
        if today == self.state.trading_day:
            return False

        self.state = RiskState(trading_day=today, cooldowns=self.state.cooldowns)
        return True

    # ---------------------------------------------------------------- cooldown

    def start_cooldown(self, symbol: str, now: datetime | None = None) -> None:
        """
        NUANCE #5: block re-entry for `symbol_cooldown_minutes` after an exit.

        Without this the bot re-enters the name that just stopped it out, on the
        same conditions that triggered the first entry — the HINDALCO loop, which
        can round-trip a dozen times in an afternoon and pay brokerage each way.
        """
        self.state.cooldowns[symbol] = to_ist(now or now_ist()).isoformat()

    def cooldown_remaining(self, symbol: str, now: datetime | None = None) -> timedelta:
        started_at = self.state.cooldowns.get(symbol)
        if not started_at:
            return timedelta(0)

        minutes = self.config.get("symbol_cooldown_minutes", 45)
        elapsed = to_ist(now or now_ist()) - to_ist(datetime.fromisoformat(started_at))
        remaining = timedelta(minutes=minutes) - elapsed
        return max(remaining, timedelta(0))

    # -------------------------------------------------------------- entry gate

    def can_enter(
        self,
        signal: Signal,
        *,
        open_positions: list[Any],
        capital: float,
        now: datetime | None = None,
        sector_map: dict[str, str] | None = None,
    ) -> tuple[bool, str]:
        """
        Should this signal become a position? Checked cheapest-first.

        Returns (allowed, reason). The reason is always populated — on rejection
        it says which limit bound, and on acceptance it says "ok".
        """
        now = to_ist(now or now_ist())
        self.roll_day_if_needed(now)

        if self.state.halted:
            return False, f"trading halted: {self.state.halt_reason}"

        # NUANCE #20: consecutive losses usually mean the regime stopped matching
        # the strategy. Stopping is cheaper than finding out how long the streak runs.
        max_streak = self.config.get("max_consecutive_losses", 3)
        if self.state.consecutive_losses >= max_streak:
            self.halt(f"{self.state.consecutive_losses} consecutive losses")
            return False, self.state.halt_reason

        loss_limit = capital * self.config.get("daily_loss_limit_pct", 3.0) / 100
        if self.state.realised_pnl <= -loss_limit:
            self.halt(f"daily loss limit hit ({self.state.realised_pnl:,.0f})")
            return False, self.state.halt_reason

        max_trades = self.config.get("max_trades_per_day", 20)
        if self.state.trades_today >= max_trades:
            return False, f"daily trade cap reached ({max_trades})"

        remaining = self.cooldown_remaining(signal.symbol, now)
        if remaining > timedelta(0):
            return False, f"{signal.symbol} cooling down for {remaining.seconds // 60}m more"

        if any(getattr(p, "symbol", None) == signal.symbol for p in open_positions):
            return False, f"already holding {signal.symbol}"

        max_positions = self.config.get("max_positions", 5)
        if len(open_positions) >= max_positions:
            return False, f"at position limit ({max_positions})"

        heat = self.portfolio_heat(open_positions, capital)
        max_heat = self.config.get("max_portfolio_heat_pct", 5.0)
        incoming = self.config.get("risk_pct", 1.0)
        if heat + incoming > max_heat:
            return False, f"portfolio heat {heat:.1f}% + {incoming:.1f}% exceeds {max_heat}%"

        if sector_map:
            sector = sector_map.get(signal.symbol)
            if sector:
                exposure = sum(
                    getattr(p, "value", 0.0)
                    for p in open_positions
                    if sector_map.get(getattr(p, "symbol", "")) == sector
                )
                sector_cap = capital * self.config.get("sector_max_pct", 30.0) / 100
                if exposure >= sector_cap:
                    return False, f"sector {sector} at exposure cap"

        min_rr = self.config.get("min_risk_reward", 1.0)
        if signal.risk_reward < min_rr:
            return False, f"risk:reward {signal.risk_reward:.2f} below {min_rr}"

        return True, "ok"

    # ------------------------------------------------------------------ sizing

    def size_position(
        self,
        signal: Signal,
        *,
        capital: float,
        available_margin: float,
        lot_size: int = 1,
        regime_multiplier: float = 1.0,
    ) -> SizingResult:
        """
        Risk-based sizing: quantity = (capital x risk%) / (entry - stop).

        NUANCE #19: size from the STOP DISTANCE, never from a fixed rupee amount.
        A fixed amount takes wildly different risk depending on where the stop
        sits; this keeps the loss identical across trades, which is what makes a
        win rate mean anything.

        The result is then clamped by position cap, available margin, and lot
        size, and reports which constraint actually bound.
        """
        risk_per_share = signal.risk_per_share
        if risk_per_share <= 0:
            return SizingResult(0, 0.0, 0.0, "stop equals entry", "invalid_stop")

        risk_pct = min(
            self.config.get("risk_pct", 1.0) * regime_multiplier,
            self.config.get("max_risk_pct", 2.0),
        )
        risk_budget = capital * risk_pct / 100

        quantity = int(risk_budget // risk_per_share)
        binding = "risk_budget"

        max_value = capital * self.config.get("max_position_pct", 20.0) / 100
        by_position_cap = int(max_value // signal.entry_price)
        if by_position_cap < quantity:
            quantity, binding = by_position_cap, "position_cap"

        # Keep a buffer: margin reported at signal time is stale by the time the
        # order lands, and a rejection mid-sequence leaves the position unhedged.
        usable_margin = available_margin * self.config.get("margin_utilisation", 0.95)
        by_margin = int(usable_margin // signal.entry_price)
        if by_margin < quantity:
            quantity, binding = by_margin, "margin"

        if lot_size > 1:
            quantity = (quantity // lot_size) * lot_size

        if quantity <= 0:
            return SizingResult(
                0, 0.0, 0.0,
                f"{binding} allows 0 shares at {signal.entry_price:.2f}", binding,
            )

        return SizingResult(
            quantity=quantity,
            capital_at_risk=round(quantity * risk_per_share, 2),
            position_value=round(quantity * signal.entry_price, 2),
            reason=(
                f"{quantity} shares, risking {quantity * risk_per_share:,.0f} "
                f"({risk_pct:.2f}% of {capital:,.0f})"
            ),
            binding_constraint=binding,
        )

    # ------------------------------------------------------------- bookkeeping

    def portfolio_heat(self, open_positions: list[Any], capital: float) -> float:
        """
        Total open risk as a percent of capital — what you lose if every stop
        fires at once. Correlated names all stop out together often enough that
        this is the number that matters, not position count.

        Positions must expose `.risk_amount`; those that don't (reconciled from
        the broker with no known stop) are charged their full value, which is the
        honest worst case.
        """
        if capital <= 0:
            return 0.0

        total = 0.0
        for position in open_positions:
            risk = getattr(position, "risk_amount", None)
            total += float(risk) if risk is not None else float(getattr(position, "value", 0.0))

        return round(total / capital * 100, 3)

    def record_exit(self, symbol: str, pnl: float, now: datetime | None = None) -> None:
        """Record a closed trade: streak, daily P&L, cooldown."""
        self.roll_day_if_needed(now)
        self.state.realised_pnl += pnl
        self.state.trades_today += 1
        self.state.consecutive_losses = self.state.consecutive_losses + 1 if pnl < 0 else 0
        self.start_cooldown(symbol, now)

    def halt(self, reason: str) -> None:
        """Stop opening new positions. Exits and stop management continue."""
        self.state.halted = True
        self.state.halt_reason = reason

    def resume(self) -> None:
        self.state.halted = False
        self.state.halt_reason = ""
