"""Korridor-Bewertung im Bildraum — deterministische Geometrie, kein ML.

Zwei Stufen, beide rein geometrisch auf einer bereits vorliegenden
Segmentierungsmaske:

A.4 Fluchtpunkt-Geometrie. Aus den linken und rechten Wegraendern werden zwei
Geraden gefittet; ihr Schnittpunkt ist der Fluchtpunkt. Die Linien von den
unteren Bildecken zum Fluchtpunkt spannen das einzige relevante Dreieck auf.
Alles oberhalb des Fluchtpunkts ist Himmel oder Ferne und wird gar nicht erst
ausgewertet — ein billiger Vorfilter vor der eigentlichen Pruefung.

A.3 Korridor-Bewertung. Drei feste Korridore (Mitte, Rechts, Links) laufen vom
unteren Bildrand auf den Fluchtpunkt zu. Geprueft wird ausschliesslich die
Breite: passt in jeder ausgewerteten Zeile ein durchgehend freier Streifen von
Fahrzeugbreite plus Sicherheitszuschlag in den Korridor? Die Laengsdimension
ist bewusst irrelevant — es wird nicht bewertet, wie weit voraus der Korridor
frei bleibt, und deshalb auch keine Distanz berichtet.

Das Ergebnis ist eine KI-gestuetzte Einschaetzung auf einer vorhergesagten
Maske und ausdruecklich keine sicherheitsrelevante Fahrfreigabe.
"""

import numpy as np

CORRIDOR_SCHEMA_VERSION = "1.0"

# Fahrzeugmasse und Kamerakalibrierung. Die Kalibrierung ist die einzige
# Groesse, die nicht aus dem Bild folgt: wie viele Meter Boden die volle
# Bildbreite in der untersten Zeile abdeckt. Sie haengt an Kamerahoehe und
# Oeffnungswinkel und muss pro Aufbau gemessen werden.
DEFAULT_VEHICLE_WIDTH_M = 1.2
DEFAULT_CLEARANCE_M = 0.10
DEFAULT_GROUND_WIDTH_AT_BOTTOM_M = 4.0

# Der Korridor ist das Suchband, der Streifen ist das, was hineinpassen muss.
# Ohne Spielraum waere "Streifen im Korridor frei" gleichbedeutend mit "Korridor
# vollstaendig frei" und ein einzelnes Stoerpixel wuerde jede Zeile sperren.
CORRIDOR_SEARCH_FACTOR = 1.5
# Seitliche Lage der drei Korridore in Streifenbreiten, gemessen in der
# untersten Bildzeile. Mitte liegt dort, wo das Fahrzeug steht.
CORRIDOR_OFFSETS = {"mitte": 0.0, "rechts": 1.0, "links": -1.0}
CORRIDOR_LABELS = {
    "mitte": "Mitte",
    "rechts": "Rechts",
    "links": "Links",
}
CORRIDOR_MEANINGS = {
    "mitte": "Standard bei schmalen Wald- und Feldwegen",
    "rechts": "Rechtsfahrgebot: Radwege und landwirtschaftliche Wege",
    "links": "Ausweichoption",
}
STATUS_LABELS = {"free": "frei", "blocked": "blockiert", "uncertain": "unsicher"}

# Sicherheitsabstand unter dem Fluchtpunkt: die letzten Zeilen davor sind nur
# wenige Pixel breit und rauschen stark.
VANISHING_MARGIN_ROWS = 2
# Unter so wenigen auswertbaren Zeilen ist keine belastbare Aussage moeglich.
MIN_EVALUATED_ROWS = 8
# Zeilen, in denen der Korridor zu weit aus dem Bild laeuft, sind unbekannt,
# nicht blockiert.
MAX_CLIPPED_FRACTION = 0.2
# "Durchgehend" mit Rasterreserve: einzelne gesperrte Zeilen entstehen schon
# durch Pixelrundung an der Wegkante und duerfen nicht als Hindernis gelten.
# Sie machen den Korridor aber unsicher, nie still frei.
MAX_BLOCKED_ROW_FRACTION = 0.05
# Mindestbreite eines Wegrand-Laufs, damit die Zeile in den Geradenfit eingeht.
MIN_EDGE_RUN_PX = 2

# Abstufung aus GRADE_ONTOLOGY (path_model): 0 unbewertet, 1 sicher, 2 gut,
# 3 knapp, 4 riskant, 5 Problemzone. Fuer die Korridorpruefung zaehlt nur
# sicher und gut als frei; knapp und riskant machen die Zeile unsicher.
FREE_GRADES = frozenset({1, 2})
UNCERTAIN_GRADES = frozenset({3, 4})


def _widest_run(values: np.ndarray):
    """Laengster zusammenhaengender True-Lauf: (start, laenge)."""
    if not values.any():
        return 0, 0
    padded = np.concatenate([[False], values, [False]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    starts, stops = changes[::2], changes[1::2]
    lengths = stops - starts
    best = int(np.argmax(lengths))
    return int(starts[best]), int(lengths[best])


def _edge_samples(free: np.ndarray):
    rows, lefts, rights = [], [], []
    for row in range(free.shape[0]):
        start, length = _widest_run(free[row])
        if length >= MIN_EDGE_RUN_PX:
            rows.append(row)
            lefts.append(start)
            rights.append(start + length - 1)
    return np.asarray(rows, np.float64), np.asarray(lefts, np.float64), np.asarray(rights, np.float64)


def estimate_vanishing_point(free: np.ndarray):
    """Schnittpunkt der gefitteten linken und rechten Wegrandgeraden.

    Der Weg verbreitert sich nach unten, die Raender laufen also nach oben
    aufeinander zu. Schlaegt der Fit fehl, greift eine dokumentierte Notloesung
    statt einer stillen Falschaussage.
    """
    height, width = free.shape
    rows, lefts, rights = _edge_samples(free)
    fallback_y = height * 0.45
    fallback_x = width / 2
    if len(rows) >= 4:
        left_slope, left_offset = np.polyfit(rows, lefts, 1)
        right_slope, right_offset = np.polyfit(rows, rights, 1)
        separation = left_slope - right_slope
        if abs(separation) > 1e-6:
            y = (right_offset - left_offset) / separation
            x = left_slope * y + left_offset
            if 0 <= y < height - 1 and -width <= x <= 2 * width:
                residual = float(
                    np.sqrt(
                        np.mean((left_slope * rows + left_offset - lefts) ** 2)
                        + np.mean((right_slope * rows + right_offset - rights) ** 2)
                    )
                )
                return {
                    "x": round(float(np.clip(x, 0, width - 1)), 3),
                    "y": round(float(y), 3),
                    "source": "path_edge_line_intersection",
                    "rows_used": len(rows),
                    "residual_px": round(residual, 3),
                }
    if len(rows):
        topmost = int(rows.min())
        index = int(np.argmin(rows))
        return {
            "x": round(float((lefts[index] + rights[index]) / 2), 3),
            "y": float(topmost),
            "source": "topmost_mask_row",
            "rows_used": len(rows),
            "residual_px": None,
        }
    return {
        "x": round(float(fallback_x), 3),
        "y": round(float(fallback_y), 3),
        "source": "image_center_assumption",
        "rows_used": 0,
        "residual_px": None,
    }


def image_decomposition(shape, vanishing_point):
    """A.4: das relevante Dreieck und die sicher irrelevante Zone darueber."""
    height, width = shape
    vanishing_row = float(vanishing_point["y"])
    first_row = int(min(height - 1, max(0, np.ceil(vanishing_row + VANISHING_MARGIN_ROWS))))
    evaluated_rows = height - first_row
    return {
        "vanishing_point": vanishing_point,
        "relevant_triangle": [
            [0, height - 1],
            [width - 1, height - 1],
            [round(float(vanishing_point["x"]), 3), round(vanishing_row, 3)],
        ],
        "irrelevant_zone": {
            "kind": "above_vanishing_point",
            "first_evaluated_row": first_row,
            "rows_skipped": first_row,
            "image_fraction_skipped": round(first_row / height, 5),
            "reason": "Himmel und Ferne oberhalb des Fluchtpunkts werden nicht ausgewertet.",
        },
        "evaluated_rows": evaluated_rows,
    }


def _classify(mask: np.ndarray, grade_mask: np.ndarray | None):
    """Pro Pixel: frei, unsicher oder blockiert."""
    if grade_mask is None:
        free = mask.astype(bool)
        return free, np.zeros_like(free)
    free = np.isin(grade_mask, list(FREE_GRADES))
    uncertain = np.isin(grade_mask, list(UNCERTAIN_GRADES))
    return free, uncertain


def _corridor_geometry(shape, vanishing_point, strip_px_bottom: float, offset: float):
    """Mittellinie und Breite je Zeile, perspektivisch zum Fluchtpunkt skaliert.

    Parallele Bodenlinien schneiden sich im Fluchtpunkt; eine feste reale Breite
    bildet sich deshalb linear mit dem Abstand zur Fluchtpunktzeile ab.
    """
    height, width = shape
    vanishing_row = float(vanishing_point["y"])
    span = max(1e-6, (height - 1) - vanishing_row)
    anchor_bottom = width / 2 + offset * strip_px_bottom
    rows = np.arange(height, dtype=np.float64)
    scale = np.clip((rows - vanishing_row) / span, 0.0, None)
    centers = vanishing_point["x"] + (anchor_bottom - vanishing_point["x"]) * scale
    strip = strip_px_bottom * scale
    return centers, strip, strip * CORRIDOR_SEARCH_FACTOR


def _row_status(free_row, uncertain_row, required_px: float, clipped_fraction: float):
    required = max(1, int(np.ceil(required_px)))
    if clipped_fraction > MAX_CLIPPED_FRACTION:
        return "uncertain"
    if _widest_run(free_row)[1] >= required:
        return "free"
    if _widest_run(free_row | uncertain_row)[1] >= required:
        return "uncertain"
    return "blocked"


def evaluate_corridors(
    mask: np.ndarray,
    grade_mask: np.ndarray | None = None,
    *,
    vehicle_width_m: float = DEFAULT_VEHICLE_WIDTH_M,
    clearance_m: float = DEFAULT_CLEARANCE_M,
    ground_width_at_bottom_m: float = DEFAULT_GROUND_WIDTH_AT_BOTTOM_M,
):
    """A.3: Status je Korridor fuer genau einen Frame."""
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.size == 0:
        raise ValueError("Die Maske muss ein nicht leeres 2D-Raster sein")
    if grade_mask is not None:
        grade_mask = np.asarray(grade_mask)
        if grade_mask.shape != mask.shape:
            raise ValueError("Abstufungsmaske und Maske müssen dieselbe Größe haben")
    if ground_width_at_bottom_m <= 0:
        raise ValueError("Die Bodenbreite am unteren Bildrand muss größer als null sein")

    height, width = mask.shape
    free, uncertain = _classify(mask, grade_mask)
    vanishing_point = estimate_vanishing_point(free)
    decomposition = image_decomposition(mask.shape, vanishing_point)
    first_row = decomposition["irrelevant_zone"]["first_evaluated_row"]

    strip_m = vehicle_width_m + clearance_m
    strip_px_bottom = strip_m / ground_width_at_bottom_m * width

    corridors = []
    for name, offset in CORRIDOR_OFFSETS.items():
        centers, strips, bands = _corridor_geometry(mask.shape, vanishing_point, strip_px_bottom, offset)
        counts = {"free": 0, "uncertain": 0, "blocked": 0}
        for row in range(first_row, height):
            band = bands[row]
            if band < 1:
                counts["uncertain"] += 1
                continue
            start = centers[row] - band / 2
            stop = centers[row] + band / 2
            clipped_start = int(np.clip(np.floor(start), 0, width))
            clipped_stop = int(np.clip(np.ceil(stop), 0, width))
            outside = max(0.0, -start) + max(0.0, stop - width)
            status = _row_status(
                free[row, clipped_start:clipped_stop],
                uncertain[row, clipped_start:clipped_stop],
                strips[row],
                outside / band,
            )
            counts[status] += 1
        evaluated = sum(counts.values())
        if evaluated < MIN_EVALUATED_ROWS:
            status = "uncertain"
            reason = "Zu wenige auswertbare Zeilen unterhalb des Fluchtpunkts."
        elif counts["blocked"] > MAX_BLOCKED_ROW_FRACTION * evaluated:
            status = "blocked"
            reason = f"In {counts['blocked']} von {evaluated} Zeilen passt kein freier Streifen in den Korridor."
        elif counts["blocked"]:
            status = "uncertain"
            reason = (
                f"Nur {counts['blocked']} von {evaluated} Zeilen sind gesperrt — das liegt in der Rasterreserve "
                "und wird nicht als Hindernis gewertet, aber auch nicht als frei."
            )
        elif counts["uncertain"]:
            status = "uncertain"
            reason = f"In {counts['uncertain']} von {evaluated} Zeilen ist der Streifen nicht sicher frei."
        else:
            status = "free"
            reason = f"In allen {evaluated} ausgewerteten Zeilen passt ein freier Streifen in den Korridor."
        corridors.append(
            {
                "corridor": name,
                "label": CORRIDOR_LABELS[name],
                "meaning": CORRIDOR_MEANINGS[name],
                "status": status,
                "status_label": STATUS_LABELS[status],
                "reason": reason,
                "rows": {"evaluated": evaluated, **counts},
                "bottom_center_x": round(float(width / 2 + offset * strip_px_bottom), 3),
            }
        )

    return {
        "schema_version": CORRIDOR_SCHEMA_VERSION,
        "kind": "image_space_corridor_check",
        "mask_size": {"width": width, "height": height},
        "decomposition": decomposition,
        "strip": {
            "vehicle_width_m": vehicle_width_m,
            "clearance_m": clearance_m,
            "required_width_m": round(strip_m, 4),
            "ground_width_at_bottom_m": ground_width_at_bottom_m,
            "required_width_px_at_bottom": round(float(strip_px_bottom), 3),
            "scaling": "linear mit dem Zeilenabstand zum Fluchtpunkt",
            "search_band_factor": CORRIDOR_SEARCH_FACTOR,
        },
        "corridors": corridors,
        "graded_input": grade_mask is not None,
        "limitations": [
            "Nur Breitenprüfung: wie weit voraus ein Korridor frei bleibt, wird bewusst nicht bewertet.",
            "Die Bodenbreite am unteren Bildrand ist eine Kalibrierung pro Kameraaufbau und keine Messung aus dem Bild.",
            "Deterministische Geometrie auf einer vorhergesagten Maske — keine sicherheitsrelevante Fahrfreigabe.",
        ],
    }
