# NY Open 15-Minuten Breakout Strategie - NQ Only

## Übersicht

Diese Strategie handelt den **Nasdaq 100 E-mini Futures (NQ)** basierend auf der **Opening Range der ersten 15 Minuten** nach dem New York Market Open (09:30 ET).

## Strategie-Logik

```
09:00 - 09:30  →  Warten auf NY Open
09:30 - 09:45  →  Opening Range erfassen (Hoch & Tief)
09:45 - 15:55  →  Breakout handeln
15:55          →  Offene Position schließen (EOD)
```

### Entry-Regeln

| Signal | Bedingung |
|--------|-----------|
| **Long** | Preis bricht über das 15-Min-Hoch + Buffer (2 Ticks) |
| **Short** | Preis bricht unter das 15-Min-Tief - Buffer (2 Ticks) |

- Nur der **erste Breakout** pro Tag wird gehandelt
- Range muss zwischen **10 und 80 Punkten** liegen (konfigurierbar)

### Risk Management

| Parameter | Default |
|-----------|---------|
| **Stop-Loss** | Gegenüberliegende Seite der Opening Range + 4 Ticks Puffer |
| **Take-Profit** | 2:1 Risk-Reward Ratio |
| **Max Contracts** | 1 |
| **Max Risiko/Trade** | $500 USD |

### Optionaler Trailing Stop

- Aktivierung ab 1.0R im Gewinn
- 15 Punkte Trailing-Abstand

## Dateien

```
config.py    - Alle konfigurierbaren Parameter
strategy.py  - Kernlogik der Strategie (NYOpen15MinStrategy)
backtest.py  - Backtest-Engine mit Statistiken
main.py      - Hauptskript zum Ausführen
```

## Verwendung

```bash
# Demo mit Beispieldaten
python main.py

# Backtest mit eigenen 1-Min-Daten (CSV)
python main.py --csv pfad/zu/nq_1min_daten.csv

# Live-Simulation (Schritt-für-Schritt)
python main.py --demo-live
```

## CSV-Datenformat

Die CSV-Datei muss 1-Minuten-Bars in **Eastern Time** enthalten:

```csv
timestamp,open,high,low,close,volume
2024-01-02 09:30:00,16800.00,16810.25,16795.50,16805.75,1234
2024-01-02 09:31:00,16805.75,16812.00,16803.25,16810.50,987
```

## Konfiguration anpassen

Alle Parameter können in `config.py` angepasst werden:

- **Zeitfenster**: NY Open, Range-Dauer, Handelsende
- **Entry**: Breakout-Buffer, erster/alle Breakouts
- **Stop-Loss**: Range-basiert oder feste Punkte
- **Take-Profit**: Risk-Reward oder feste Punkte
- **Trailing Stop**: Aktivierung, Abstand
- **Filter**: Min/Max Range-Größe
- **Position Sizing**: Max Contracts, Max Risiko

## Hinweis

Diese Strategie dient zu **Bildungszwecken**. Vergangene Performance ist kein Indikator für zukünftige Ergebnisse. Handel mit Futures beinhaltet erhebliche Risiken.
