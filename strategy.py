"""
NY Open 15-Minuten Breakout Strategie - NQ Only
================================================

Kernlogik der Strategie:
1. Warte auf 09:30 ET (NY Open)
2. Erfasse Hoch und Tief der ersten 15 Minuten (09:30 - 09:45)
3. Nach 09:45: Warte auf Breakout über das Hoch (Long) oder unter das Tief (Short)
4. Setze Stop-Loss und Take-Profit
5. Maximal ein Trade pro Tag
"""

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from enum import Enum
from typing import Optional

import config


class Direction(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class TradeStatus(Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    CLOSED = "CLOSED"


@dataclass
class Bar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int = 0


@dataclass
class OpeningRange:
    """Die 15-Minuten Opening Range nach NY Open."""
    high: float = 0.0
    low: float = float("inf")
    is_complete: bool = False
    bar_count: int = 0

    def update(self, bar: Bar) -> None:
        if self.is_complete:
            return
        self.high = max(self.high, bar.high)
        self.low = min(self.low, bar.low)
        self.bar_count += 1

    def complete(self) -> None:
        self.is_complete = True

    @property
    def range_points(self) -> float:
        return self.high - self.low

    @property
    def midpoint(self) -> float:
        return (self.high + self.low) / 2

    def is_valid(self) -> bool:
        return (
            self.is_complete
            and config.MIN_RANGE_POINTS <= self.range_points <= config.MAX_RANGE_POINTS
        )


@dataclass
class Trade:
    direction: Direction
    entry_price: float
    stop_loss: float
    take_profit: float
    contracts: int
    entry_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: str = ""
    status: TradeStatus = TradeStatus.PENDING
    highest_since_entry: float = 0.0
    lowest_since_entry: float = float("inf")
    trailing_stop: Optional[float] = None

    @property
    def pnl_points(self) -> float:
        if self.exit_price is None:
            return 0.0
        if self.direction == Direction.LONG:
            return self.exit_price - self.entry_price
        return self.entry_price - self.exit_price

    @property
    def pnl_usd(self) -> float:
        return self.pnl_points * config.POINT_VALUE * self.contracts

    @property
    def risk_points(self) -> float:
        return abs(self.entry_price - self.stop_loss)

    @property
    def r_multiple(self) -> float:
        if self.risk_points == 0:
            return 0.0
        return self.pnl_points / self.risk_points


class NYOpen15MinStrategy:
    """
    New York Open 15-Minuten Breakout Strategie für NQ.

    Ablauf pro Handelstag:
    1. PHASE_WAIT:    Warten auf 09:30 ET
    2. PHASE_RANGE:   09:30-09:45 - Opening Range aufbauen
    3. PHASE_TRADE:   09:45-16:00 - Auf Breakout warten und handeln
    4. PHASE_DONE:    Tag abgeschlossen
    """

    def __init__(self) -> None:
        self.opening_range: Optional[OpeningRange] = None
        self.current_trade: Optional[Trade] = None
        self.daily_trades: list[Trade] = []
        self.all_trades: list[Trade] = []
        self.current_date: Optional[datetime] = None
        self._phase: str = "WAIT"

    # ──────────────────────────────────────────
    # Zeitprüfungen
    # ──────────────────────────────────────────

    @staticmethod
    def _ny_open_start() -> time:
        return time(config.NY_OPEN_HOUR, config.NY_OPEN_MINUTE)

    @staticmethod
    def _range_end() -> time:
        total_min = config.NY_OPEN_MINUTE + config.RANGE_MINUTES
        hour = config.NY_OPEN_HOUR + total_min // 60
        minute = total_min % 60
        return time(hour, minute)

    @staticmethod
    def _eod_close() -> time:
        return time(config.EOD_CLOSE_HOUR, config.EOD_CLOSE_MINUTE)

    @staticmethod
    def _trade_end() -> time:
        return time(config.TRADE_END_HOUR, config.TRADE_END_MINUTE)

    def _is_new_day(self, bar: Bar) -> bool:
        return self.current_date is None or bar.timestamp.date() != self.current_date.date()

    # ──────────────────────────────────────────
    # Tagesreset
    # ──────────────────────────────────────────

    def _reset_day(self, bar: Bar) -> None:
        self.current_date = bar.timestamp
        self.opening_range = OpeningRange()
        self.current_trade = None
        self.daily_trades = []
        self._phase = "WAIT"

    # ──────────────────────────────────────────
    # Entry-Berechnung
    # ──────────────────────────────────────────

    def _calc_long_entry(self) -> float:
        buffer = config.BREAKOUT_BUFFER_TICKS * config.TICK_SIZE
        return self.opening_range.high + buffer

    def _calc_short_entry(self) -> float:
        buffer = config.BREAKOUT_BUFFER_TICKS * config.TICK_SIZE
        return self.opening_range.low - buffer

    def _calc_stop_loss(self, direction: Direction, entry_price: float) -> float:
        if config.STOP_LOSS_MODE == "fixed":
            if direction == Direction.LONG:
                return entry_price - config.STOP_LOSS_FIXED_POINTS
            return entry_price + config.STOP_LOSS_FIXED_POINTS

        # Range-basierter Stop: gegenüberliegende Seite + Puffer
        buffer = config.STOP_LOSS_RANGE_BUFFER_TICKS * config.TICK_SIZE
        if direction == Direction.LONG:
            return self.opening_range.low - buffer
        return self.opening_range.high + buffer

    def _calc_take_profit(self, direction: Direction, entry_price: float, stop_loss: float) -> float:
        if config.TAKE_PROFIT_MODE == "fixed":
            if direction == Direction.LONG:
                return entry_price + config.TAKE_PROFIT_FIXED_POINTS
            return entry_price - config.TAKE_PROFIT_FIXED_POINTS

        # Risk-Reward basiert
        risk = abs(entry_price - stop_loss)
        reward = risk * config.TAKE_PROFIT_RR_RATIO
        if direction == Direction.LONG:
            return entry_price + reward
        return entry_price - reward

    def _calc_contracts(self, entry_price: float, stop_loss: float) -> int:
        risk_per_contract = abs(entry_price - stop_loss) * config.POINT_VALUE
        if risk_per_contract == 0:
            return 0
        max_by_risk = int(config.MAX_RISK_PER_TRADE_USD / risk_per_contract)
        return max(1, min(max_by_risk, config.MAX_CONTRACTS))

    # ──────────────────────────────────────────
    # Trade-Verwaltung
    # ──────────────────────────────────────────

    def _open_trade(self, direction: Direction, price: float, bar: Bar) -> Trade:
        stop_loss = self._calc_stop_loss(direction, price)
        take_profit = self._calc_take_profit(direction, price, stop_loss)
        contracts = self._calc_contracts(price, stop_loss)

        trade = Trade(
            direction=direction,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            contracts=contracts,
            entry_time=bar.timestamp,
            status=TradeStatus.OPEN,
            highest_since_entry=price,
            lowest_since_entry=price,
        )
        self.current_trade = trade
        self.daily_trades.append(trade)
        return trade

    def _close_trade(self, price: float, bar: Bar, reason: str) -> Trade:
        trade = self.current_trade
        trade.exit_price = price
        trade.exit_time = bar.timestamp
        trade.exit_reason = reason
        trade.status = TradeStatus.CLOSED
        self.all_trades.append(trade)
        self.current_trade = None
        return trade

    # ──────────────────────────────────────────
    # Trailing Stop
    # ──────────────────────────────────────────

    def _update_trailing_stop(self, bar: Bar) -> None:
        if not config.USE_TRAILING_STOP or self.current_trade is None:
            return

        trade = self.current_trade
        if trade.direction == Direction.LONG:
            trade.highest_since_entry = max(trade.highest_since_entry, bar.high)
            current_r = (trade.highest_since_entry - trade.entry_price) / trade.risk_points
            if current_r >= config.TRAILING_STOP_ACTIVATION_RR:
                new_trail = trade.highest_since_entry - config.TRAILING_STOP_DISTANCE_POINTS
                if trade.trailing_stop is None or new_trail > trade.trailing_stop:
                    trade.trailing_stop = new_trail
        else:
            trade.lowest_since_entry = min(trade.lowest_since_entry, bar.low)
            current_r = (trade.entry_price - trade.lowest_since_entry) / trade.risk_points
            if current_r >= config.TRAILING_STOP_ACTIVATION_RR:
                new_trail = trade.lowest_since_entry + config.TRAILING_STOP_DISTANCE_POINTS
                if trade.trailing_stop is None or new_trail < trade.trailing_stop:
                    trade.trailing_stop = new_trail

    # ──────────────────────────────────────────
    # Hauptlogik: Bar-für-Bar verarbeiten
    # ──────────────────────────────────────────

    def on_bar(self, bar: Bar) -> Optional[Trade]:
        """
        Verarbeitet eine neue Bar. Gibt einen abgeschlossenen Trade zurück,
        falls einer geschlossen wurde, sonst None.
        """
        bar_time = bar.timestamp.time()

        # Neuer Tag?
        if self._is_new_day(bar):
            # Falls noch ein Trade offen ist vom Vortag, schließen
            if self.current_trade and self.current_trade.status == TradeStatus.OPEN:
                self._close_trade(bar.open, bar, "NEW_DAY_CLOSE")
            self._reset_day(bar)

        # ── Phase: WAIT (vor 09:30) ──
        if self._phase == "WAIT":
            if bar_time >= self._ny_open_start():
                self._phase = "RANGE"
                self.opening_range.update(bar)
            return None

        # ── Phase: RANGE (09:30 - 09:45) ──
        if self._phase == "RANGE":
            if bar_time < self._range_end():
                self.opening_range.update(bar)
                return None
            else:
                # Range abgeschlossen
                self.opening_range.complete()
                if not self.opening_range.is_valid():
                    self._phase = "DONE"
                    return None
                self._phase = "TRADE"
                # Diese Bar könnte bereits einen Breakout enthalten
                # → weiter zur TRADE-Phase

        # ── Phase: TRADE (09:45 - EOD) ──
        if self._phase == "TRADE":
            # EOD-Schließung
            if config.CLOSE_AT_EOD and bar_time >= self._eod_close():
                if self.current_trade and self.current_trade.status == TradeStatus.OPEN:
                    closed = self._close_trade(bar.close, bar, "EOD_CLOSE")
                    self._phase = "DONE"
                    return closed
                self._phase = "DONE"
                return None

            # Handelszeit vorbei?
            if bar_time >= self._trade_end():
                self._phase = "DONE"
                return None

            # ── Offener Trade verwalten ──
            if self.current_trade and self.current_trade.status == TradeStatus.OPEN:
                return self._manage_open_trade(bar)

            # ── Auf Breakout warten ──
            if config.ONLY_FIRST_BREAKOUT and len(self.daily_trades) > 0:
                self._phase = "DONE"
                return None

            return self._check_breakout(bar)

        # ── Phase: DONE ──
        return None

    def _check_breakout(self, bar: Bar) -> Optional[Trade]:
        """Prüft ob ein Breakout über/unter der Opening Range stattfindet."""
        long_entry = self._calc_long_entry()
        short_entry = self._calc_short_entry()

        # Long Breakout: Bar-Hoch durchbricht das Range-Hoch
        if bar.high >= long_entry:
            self._open_trade(Direction.LONG, long_entry, bar)
            # Prüfe ob Stop/TP in derselben Bar getroffen wurde
            return self._manage_open_trade(bar)

        # Short Breakout: Bar-Tief durchbricht das Range-Tief
        if bar.low <= short_entry:
            self._open_trade(Direction.SHORT, short_entry, bar)
            return self._manage_open_trade(bar)

        return None

    def _manage_open_trade(self, bar: Bar) -> Optional[Trade]:
        """Verwaltet einen offenen Trade: Stop-Loss, Take-Profit, Trailing Stop."""
        trade = self.current_trade
        if trade is None:
            return None

        # Trailing Stop aktualisieren
        self._update_trailing_stop(bar)

        if trade.direction == Direction.LONG:
            # Stop-Loss getroffen?
            effective_stop = trade.trailing_stop if trade.trailing_stop else trade.stop_loss
            if bar.low <= effective_stop:
                reason = "TRAILING_STOP" if trade.trailing_stop and bar.low <= trade.trailing_stop else "STOP_LOSS"
                return self._close_trade(effective_stop, bar, reason)
            # Take-Profit getroffen?
            if bar.high >= trade.take_profit:
                return self._close_trade(trade.take_profit, bar, "TAKE_PROFIT")
        else:
            # Short
            effective_stop = trade.trailing_stop if trade.trailing_stop else trade.stop_loss
            if bar.high >= effective_stop:
                reason = "TRAILING_STOP" if trade.trailing_stop and bar.high >= trade.trailing_stop else "STOP_LOSS"
                return self._close_trade(effective_stop, bar, reason)
            if bar.low <= trade.take_profit:
                return self._close_trade(trade.take_profit, bar, "TAKE_PROFIT")

        return None

    # ──────────────────────────────────────────
    # Status
    # ──────────────────────────────────────────

    def get_status(self) -> dict:
        return {
            "phase": self._phase,
            "date": str(self.current_date.date()) if self.current_date else None,
            "opening_range": {
                "high": self.opening_range.high if self.opening_range else None,
                "low": self.opening_range.low if self.opening_range else None,
                "range_points": self.opening_range.range_points if self.opening_range and self.opening_range.is_complete else None,
                "is_valid": self.opening_range.is_valid() if self.opening_range and self.opening_range.is_complete else None,
            },
            "current_trade": {
                "direction": self.current_trade.direction.value,
                "entry": self.current_trade.entry_price,
                "stop": self.current_trade.stop_loss,
                "target": self.current_trade.take_profit,
            } if self.current_trade else None,
            "daily_trades": len(self.daily_trades),
            "total_trades": len(self.all_trades),
        }
