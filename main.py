#!/usr/bin/env python3
"""
NY Open 15-Minuten Breakout Strategie - NQ Only
================================================

Hauptskript zum Ausführen der Strategie.

Verwendung:
    # Mit Beispieldaten (Demo):
    python main.py

    # Mit eigenen CSV-Daten:
    python main.py --csv daten/nq_1min_2024.csv

    # Live-Status anzeigen (Simulation):
    python main.py --demo-live
"""

import argparse
import sys
import time
from datetime import datetime

from backtest import generate_sample_data, run_backtest, run_backtest_from_csv
from strategy import NYOpen15MinStrategy


def run_demo_backtest() -> None:
    """Führt einen Backtest mit generierten Beispieldaten durch."""
    print()
    print("  Generiere Beispieldaten für NQ...")
    from backtest import generate_sample_data
    bars = generate_sample_data()
    print(f"  {len(bars)} Bars generiert (1 Handelstag)")
    print()

    result = run_backtest(bars)
    result.print_summary()


def run_csv_backtest(filepath: str) -> None:
    """Führt einen Backtest mit CSV-Daten durch."""
    print()
    print(f"  Lade Daten aus: {filepath}")
    result = run_backtest_from_csv(filepath)
    result.print_summary()


def run_live_demo() -> None:
    """
    Simuliert die Strategie Bar-für-Bar mit Live-Ausgabe.
    Nützlich um die Strategie-Logik zu verstehen.
    """
    print()
    print("  ╔══════════════════════════════════════════════════╗")
    print("  ║  NY OPEN 15-MIN BREAKOUT - NQ LIVE SIMULATION   ║")
    print("  ╚══════════════════════════════════════════════════╝")
    print()

    bars = generate_sample_data()
    strategy = NYOpen15MinStrategy()

    for bar in bars:
        closed_trade = strategy.on_bar(bar)
        status = strategy.get_status()

        # Nur wichtige Momente anzeigen
        bar_time = bar.timestamp.time()

        # Range-Phase
        if status["phase"] == "RANGE":
            print(
                f"  [{bar.timestamp.strftime('%H:%M')}]"
                f"  RANGE BUILDING  |  H: {status['opening_range']['high']:.2f}"
                f"  L: {status['opening_range']['low']:.2f}"
                f"  |  Close: {bar.close:.2f}"
            )

        # Range fertig
        if status["phase"] == "TRADE" and status["opening_range"]["range_points"] and status["daily_trades"] == 0 and not status["current_trade"]:
            rng = status["opening_range"]
            if bar_time.hour == 9 and bar_time.minute == 45:
                print()
                print(f"  ═══ OPENING RANGE COMPLETE ═══")
                print(f"      Hoch:   {rng['high']:.2f}")
                print(f"      Tief:   {rng['low']:.2f}")
                print(f"      Range:  {rng['range_points']:.2f} Punkte")
                print(f"      Valid:  {rng['is_valid']}")
                print()

        # Trade eröffnet
        if status["current_trade"] and not closed_trade:
            t = status["current_trade"]
            if bar_time == strategy.current_trade.entry_time.time():
                print(
                    f"  [{bar.timestamp.strftime('%H:%M')}]"
                    f"  ▶ ENTRY {t['direction']}"
                    f"  @ {t['entry']:.2f}"
                    f"  |  SL: {t['stop']:.2f}"
                    f"  |  TP: {t['target']:.2f}"
                )

        # Trade geschlossen
        if closed_trade:
            emoji = "✓" if closed_trade.pnl_points > 0 else "✗"
            print(
                f"  [{bar.timestamp.strftime('%H:%M')}]"
                f"  {emoji} EXIT {closed_trade.direction.value}"
                f"  @ {closed_trade.exit_price:.2f}"
                f"  |  P&L: {closed_trade.pnl_points:+.2f} pts"
                f"  (${closed_trade.pnl_usd:+,.2f})"
                f"  |  {closed_trade.exit_reason}"
            )

    # Zusammenfassung
    from backtest import BacktestResult
    result = BacktestResult(strategy.all_trades)
    print()
    result.print_summary()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NY Open 15-Min Breakout Strategie - NQ Only",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python main.py                    Demo mit Beispieldaten
  python main.py --csv data.csv     Backtest mit eigenen Daten
  python main.py --demo-live        Schritt-für-Schritt Simulation

CSV-Format (1-Min Bars, Eastern Time):
  timestamp,open,high,low,close,volume
  2024-01-02 09:30:00,16800.00,16810.25,16795.50,16805.75,1234
        """,
    )
    parser.add_argument("--csv", type=str, help="Pfad zur CSV-Datei mit 1-Min-Bars")
    parser.add_argument("--demo-live", action="store_true", help="Live-Simulation mit Beispieldaten")

    args = parser.parse_args()

    if args.csv:
        run_csv_backtest(args.csv)
    elif args.demo_live:
        run_live_demo()
    else:
        run_demo_backtest()


if __name__ == "__main__":
    main()
