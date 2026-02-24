"""
Konfiguration für die NY Open 15-Minuten Breakout Strategie - NQ Only
=====================================================================

Strategie-Logik:
- Erfasse die Hoch/Tief-Range der ersten 15 Minuten nach NY Open (09:30-09:45 ET)
- Long bei Breakout über das 15-Min-Hoch
- Short bei Breakout unter das 15-Min-Tief
- Nur ein Trade pro Tag (erster Breakout zählt)
- Feste Stop-Loss und Take-Profit basierend auf der Range-Größe
"""

# ──────────────────────────────────────────────
# Instrument
# ──────────────────────────────────────────────
SYMBOL = "NQ"  # Nasdaq 100 E-mini Futures
TICK_SIZE = 0.25
TICK_VALUE = 5.00  # USD pro Tick
POINT_VALUE = 20.00  # USD pro Punkt (1 Punkt = 4 Ticks)

# ──────────────────────────────────────────────
# Zeitfenster (Eastern Time)
# ──────────────────────────────────────────────
NY_OPEN_HOUR = 9
NY_OPEN_MINUTE = 30
RANGE_MINUTES = 15  # Dauer der Opening Range
TRADE_END_HOUR = 16  # Letzter Zeitpunkt für offene Positionen
TRADE_END_MINUTE = 0

# ──────────────────────────────────────────────
# Entry-Regeln
# ──────────────────────────────────────────────
BREAKOUT_BUFFER_TICKS = 2  # Ticks über/unter dem Range-Hoch/-Tief für Entry
ONLY_FIRST_BREAKOUT = True  # Nur den ersten Breakout handeln

# ──────────────────────────────────────────────
# Risk Management
# ──────────────────────────────────────────────
STOP_LOSS_MODE = "range"  # "range" = gegenüberliegende Seite der Range, "fixed" = feste Punkte
STOP_LOSS_FIXED_POINTS = 30  # Nur relevant wenn STOP_LOSS_MODE = "fixed"
STOP_LOSS_RANGE_BUFFER_TICKS = 4  # Zusätzlicher Puffer beim Range-basierten Stop

TAKE_PROFIT_MODE = "rr"  # "rr" = Risk-Reward basiert, "fixed" = feste Punkte
TAKE_PROFIT_RR_RATIO = 2.0  # Risk-Reward Verhältnis (z.B. 2.0 = 2:1)
TAKE_PROFIT_FIXED_POINTS = 60  # Nur relevant wenn TAKE_PROFIT_MODE = "fixed"

# ──────────────────────────────────────────────
# Positionsgröße
# ──────────────────────────────────────────────
MAX_CONTRACTS = 1
MAX_RISK_PER_TRADE_USD = 500  # Maximaler Verlust pro Trade in USD

# ──────────────────────────────────────────────
# Filter
# ──────────────────────────────────────────────
MIN_RANGE_POINTS = 10  # Mindestgröße der 15-Min-Range in Punkten
MAX_RANGE_POINTS = 80  # Maximale Range-Größe (zu volatil = kein Trade)

# ──────────────────────────────────────────────
# Trail-Stop (optional)
# ──────────────────────────────────────────────
USE_TRAILING_STOP = False
TRAILING_STOP_ACTIVATION_RR = 1.0  # Ab welchem R-Vielfachen der Trail aktiviert wird
TRAILING_STOP_DISTANCE_POINTS = 15  # Abstand des Trailing Stops

# ──────────────────────────────────────────────
# End-of-Day
# ──────────────────────────────────────────────
CLOSE_AT_EOD = True  # Position am Ende des Handelstages schließen
EOD_CLOSE_HOUR = 15
EOD_CLOSE_MINUTE = 55

# ──────────────────────────────────────────────
# Daten
# ──────────────────────────────────────────────
DATA_TIMEZONE = "US/Eastern"
BAR_INTERVAL_SECONDS = 60  # 1-Minuten-Bars für Berechnung
