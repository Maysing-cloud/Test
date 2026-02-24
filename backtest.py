"""
Backtest-Engine für die NY Open 15-Minuten Breakout Strategie
=============================================================

Liest 1-Minuten-Bardaten (CSV) ein und simuliert die Strategie.
Erzeugt Statistiken und eine Trade-Liste.
"""

import csv
from datetime import datetime
from typing import Optional

import config
from strategy import Bar, NYOpen15MinStrategy, Trade


def load_bars_from_csv(filepath: str) -> list[Bar]:
    """
    Lädt 1-Minuten-Bars aus einer CSV-Datei.

    Erwartetes Format:
        timestamp,open,high,low,close,volume
        2024-01-02 09:30:00,16800.00,16810.25,16795.50,16805.75,1234

    Der Timestamp muss in Eastern Time sein.
    """
    bars = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bar = Bar(
                timestamp=datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S"),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=int(row.get("volume", 0)),
            )
            bars.append(bar)
    return bars


def generate_sample_data() -> list[Bar]:
    """
    Generiert Beispieldaten für einen Handelstag zur Demonstration.
    Simuliert einen typischen NQ-Tag mit Breakout nach oben.
    """
    import random

    random.seed(42)
    bars = []
    base_price = 17500.0
    current_price = base_price

    # Generiere Bars von 09:00 bis 16:00
    dt = datetime(2024, 6, 3, 9, 0, 0)  # Montag
    for minute in range(420):  # 7 Stunden * 60 Minuten
        hour = 9 + minute // 60
        min_of_hour = minute % 60
        if hour >= 16:
            break

        dt = datetime(2024, 6, 3, hour, min_of_hour, 0)

        # Mehr Volatilität in den ersten 15 Minuten
        if hour == 9 and min_of_hour < 45:
            volatility = random.uniform(3, 8)
        else:
            volatility = random.uniform(1, 4)

        # Leichter Aufwärtstrend nach dem Open
        drift = 0.1 if minute > 15 else 0
        change = random.gauss(drift, volatility)
        current_price += change

        bar_open = current_price
        bar_high = bar_open + abs(random.gauss(0, volatility * 0.5))
        bar_low = bar_open - abs(random.gauss(0, volatility * 0.5))
        bar_close = bar_open + random.gauss(drift, volatility * 0.3)

        bars.append(Bar(
            timestamp=dt,
            open=round(bar_open, 2),
            high=round(max(bar_high, bar_open, bar_close), 2),
            low=round(min(bar_low, bar_open, bar_close), 2),
            close=round(bar_close, 2),
            volume=random.randint(500, 5000),
        ))

    return bars


class BacktestResult:
    """Ergebnisse eines Backtests."""

    def __init__(self, trades: list[Trade]) -> None:
        self.trades = trades

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def winners(self) -> list[Trade]:
        return [t for t in self.trades if t.pnl_points > 0]

    @property
    def losers(self) -> list[Trade]:
        return [t for t in self.trades if t.pnl_points < 0]

    @property
    def breakeven(self) -> list[Trade]:
        return [t for t in self.trades if t.pnl_points == 0]

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return len(self.winners) / self.total_trades * 100

    @property
    def total_pnl_points(self) -> float:
        return sum(t.pnl_points for t in self.trades)

    @property
    def total_pnl_usd(self) -> float:
        return sum(t.pnl_usd for t in self.trades)

    @property
    def avg_winner_points(self) -> float:
        if not self.winners:
            return 0.0
        return sum(t.pnl_points for t in self.winners) / len(self.winners)

    @property
    def avg_loser_points(self) -> float:
        if not self.losers:
            return 0.0
        return sum(t.pnl_points for t in self.losers) / len(self.losers)

    @property
    def avg_r_multiple(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return sum(t.r_multiple for t in self.trades) / self.total_trades

    @property
    def max_drawdown_usd(self) -> float:
        if not self.trades:
            return 0.0
        equity = 0.0
        peak = 0.0
        max_dd = 0.0
        for trade in self.trades:
            equity += trade.pnl_usd
            peak = max(peak, equity)
            dd = peak - equity
            max_dd = max(max_dd, dd)
        return max_dd

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.pnl_usd for t in self.winners)
        gross_loss = abs(sum(t.pnl_usd for t in self.losers))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    def print_summary(self) -> None:
        print("=" * 60)
        print("  NY OPEN 15-MIN BREAKOUT STRATEGIE - NQ BACKTEST")
        print("=" * 60)
        print()
        print(f"  Gesamtanzahl Trades:     {self.total_trades}")
        print(f"  Gewinner:                {len(self.winners)}")
        print(f"  Verlierer:               {len(self.losers)}")
        print(f"  Breakeven:               {len(self.breakeven)}")
        print(f"  Win-Rate:                {self.win_rate:.1f}%")
        print()
        print(f"  Gesamt P&L (Punkte):     {self.total_pnl_points:+.2f}")
        print(f"  Gesamt P&L (USD):        ${self.total_pnl_usd:+,.2f}")
        print(f"  Avg Winner (Punkte):     {self.avg_winner_points:+.2f}")
        print(f"  Avg Loser (Punkte):      {self.avg_loser_points:+.2f}")
        print(f"  Avg R-Multiple:          {self.avg_r_multiple:+.2f}R")
        print(f"  Profit Factor:           {self.profit_factor:.2f}")
        print(f"  Max Drawdown (USD):      ${self.max_drawdown_usd:,.2f}")
        print()
        print("-" * 60)
        print("  TRADE-DETAILS")
        print("-" * 60)
        for i, trade in enumerate(self.trades, 1):
            print(
                f"  #{i:>3}  {trade.entry_time.strftime('%Y-%m-%d %H:%M') if trade.entry_time else 'N/A'}"
                f"  {trade.direction.value:>5}"
                f"  Entry: {trade.entry_price:>10.2f}"
                f"  Exit: {trade.exit_price:>10.2f}"
                f"  P&L: {trade.pnl_points:>+8.2f} pts"
                f"  ({trade.r_multiple:>+.1f}R)"
                f"  [{trade.exit_reason}]"
            )
        print("=" * 60)


def run_backtest(bars: list[Bar]) -> BacktestResult:
    """Führt den Backtest mit den gegebenen Bars durch."""
    strategy = NYOpen15MinStrategy()

    for bar in bars:
        strategy.on_bar(bar)

    # Falls am Ende noch ein Trade offen ist
    if strategy.current_trade and strategy.current_trade.status.value == "OPEN":
        last_bar = bars[-1]
        strategy._close_trade(last_bar.close, last_bar, "BACKTEST_END")

    return BacktestResult(strategy.all_trades)


def run_backtest_from_csv(filepath: str) -> BacktestResult:
    """Lädt Daten aus CSV und führt den Backtest durch."""
    bars = load_bars_from_csv(filepath)
    print(f"  {len(bars)} Bars geladen aus {filepath}")
    return run_backtest(bars)
