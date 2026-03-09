"""
Verarbeitet Koordinatendaten und erstellt eine DXF-Datei mit Rechtecken um jede Linie.

Eingabe: CSV mit Punktpaaren (je 2 aufeinanderfolgende Punkte = 1 Linie)
Ausgabe: DXF-Datei mit:
  - Den Originallinien (Layer "LINIEN")
  - Rechtecken um jede Linie, 2m Abstand links/rechts (Layer "RECHTECKE")

Koordinatenformat: XXXXXXX:CC.MMM (Meter:Zentimeter.Millimeter)
  -> Vollwert in Metern = XXXXXXX + CC.MMM / 100
"""

import csv
import math
import ezdxf

INPUT_FILE = "coordinate_data_raw.csv"
OUTPUT_FILE = "output_rectangles.dxf"
OFFSET = 2.0  # Meter links und rechts von der Linie


def parse_coordinate(raw: str) -> float:
    """Parst das Format 'XXXXXXX:CC.MMM' in einen Meterwert."""
    sign = 1.0
    raw = raw.strip()
    if raw.startswith("-"):
        sign = -1.0
        raw = raw[1:]
    parts = raw.split(":")
    meters = int(parts[0])
    centimeters = float(parts[1])
    return sign * (meters + centimeters / 100.0)


def compute_rectangle(p1, p2, offset):
    """Berechnet die 4 Eckpunkte eines Rechtecks um eine Linie p1->p2.

    Das Rechteck ist so lang wie die Linie und reicht 'offset' Meter
    senkrecht auf jeder Seite der Linie hinaus.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = math.hypot(dx, dy)
    if length == 0:
        return None

    # Einheitsvektor senkrecht zur Linie
    nx = -dy / length
    ny = dx / length

    # 4 Ecken: Verschiebung um +/- offset senkrecht zur Linie
    corners = [
        (p1[0] + nx * offset, p1[1] + ny * offset),
        (p2[0] + nx * offset, p2[1] + ny * offset),
        (p2[0] - nx * offset, p2[1] - ny * offset),
        (p1[0] - nx * offset, p1[1] - ny * offset),
    ]
    return corners


def main():
    # Koordinaten einlesen
    points = []
    with open(INPUT_FILE, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x = parse_coordinate(row["x_raw"])
            y = parse_coordinate(row["y_raw"])
            z = parse_coordinate(row["z_raw"])
            points.append((int(row["point_id"]), x, y, z))

    print(f"{len(points)} Punkte eingelesen.")
    print(f"{len(points) // 2} Linien erkannt (je 2 aufeinanderfolgende Punkte).")

    # DXF erstellen
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    # Layer anlegen
    doc.layers.add("LINIEN", color=6)       # Magenta (wie im Bild)
    doc.layers.add("RECHTECKE", color=3)     # Gruen

    line_count = 0
    rect_count = 0

    for i in range(0, len(points) - 1, 2):
        pid1, x1, y1, z1 = points[i]
        pid2, x2, y2, z2 = points[i + 1]

        p1 = (x1, y1)
        p2 = (x2, y2)

        # Linie zeichnen
        msp.add_line(p1, p2, dxfattribs={"layer": "LINIEN"})
        line_count += 1

        # Rechteck berechnen und zeichnen
        corners = compute_rectangle(p1, p2, OFFSET)
        if corners is not None:
            # Geschlossene Polylinie fuer das Rechteck
            msp.add_lwpolyline(
                corners + [corners[0]],  # schliessen
                dxfattribs={"layer": "RECHTECKE"},
            )
            rect_count += 1

    doc.saveas(OUTPUT_FILE)
    print(f"\n{line_count} Linien gezeichnet.")
    print(f"{rect_count} Rechtecke erstellt (je 2m links/rechts = 4m breit).")
    print(f"DXF gespeichert: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
