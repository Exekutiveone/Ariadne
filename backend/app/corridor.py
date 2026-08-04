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
# Stuetzstellen der ausgelieferten Polylinien. Genug fuer eine glatte Darstellung,
# klein genug fuer eine Antwort pro Frame.
GEOMETRY_SAMPLE_ROWS = 28
# Fensterbreite der Glaettung der KI-Trajektorie in Bildzeilen.
TRAJECTORY_SMOOTHING = 9
# Reihenfolge bei gleichem Status: Mitte ist der Standard, Rechts folgt dem
# Rechtsfahrgebot, Links ist die Ausweichoption.
CORRIDOR_PREFERENCE = ("mitte", "rechts", "links")
STATUS_PREFERENCE = {"free": 0, "uncertain": 1, "blocked": 2}

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
    """Pro Pixel: frei, unsicher, und ob das Modell ueberhaupt geurteilt hat.

    Stufe 0 heisst "nicht bewertet", nicht "nicht befahrbar". Ohne diese
    Unterscheidung waere jede Zeile ohne Modellurteil — Himmel und Ferne knapp
    unter dem Fluchtpunkt — automatisch ein Hindernis. Realdaten-Befund vom
    04.08.2026: in echten Waldframes waren so 31 von 37 gesperrten Zeilen zu
    100 % unbewertet, also gar kein Hindernis.
    """
    if grade_mask is None:
        # Ohne Abstufung ist die Binaermaske vollstaendig: 0 heisst dort wirklich
        # "kein Weg", nicht "keine Aussage".
        free = mask.astype(bool)
        return free, np.zeros_like(free), np.ones_like(free)
    free = np.isin(grade_mask, list(FREE_GRADES))
    uncertain = np.isin(grade_mask, list(UNCERTAIN_GRADES))
    return free, uncertain, grade_mask != 0


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


def _normalized(points, shape):
    """Bildpunkte auf 0..1 normieren — das Frontend skaliert sie auf das
    Videoelement, dessen Aufloesung nicht die des Modellrasters ist."""
    height, width = shape
    return [
        [round(float(np.clip(x, 0, width - 1)) / max(1, width - 1), 5), round(float(y) / max(1, height - 1), 5)]
        for x, y in points
    ]


def _sample(rows, count: int = GEOMETRY_SAMPLE_ROWS):
    """Gleichmaessige Stuetzstellen; die Polylinien bleiben klein genug, um sie
    pro Frame durchs Netz zu schicken."""
    if len(rows) <= count:
        return list(rows)
    picks = np.linspace(0, len(rows) - 1, count).round().astype(int)
    return [rows[index] for index in dict.fromkeys(picks.tolist())]


def _smooth(values, window: int = TRAJECTORY_SMOOTHING):
    """Gleitender Mittelwert gegen das Zeilenrauschen der freien Laeufe."""
    if len(values) < 3:
        return list(values)
    padded = np.pad(np.asarray(values, np.float64), (window // 2, window // 2), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")[: len(values)].tolist()


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
    free, uncertain, decided = _classify(mask, grade_mask)
    vanishing_point = estimate_vanishing_point(free)
    decomposition = image_decomposition(mask.shape, vanishing_point)
    first_row = decomposition["irrelevant_zone"]["first_evaluated_row"]

    strip_m = vehicle_width_m + clearance_m
    strip_px_bottom = strip_m / ground_width_at_bottom_m * width

    corridors = []
    for name, offset in CORRIDOR_OFFSETS.items():
        centers, strips, bands = _corridor_geometry(mask.shape, vanishing_point, strip_px_bottom, offset)
        counts = {"free": 0, "uncertain": 0, "blocked": 0}
        undecided_rows = 0
        # Zeilen, in denen ein freier Streifen passt, samt seiner Mitte: daraus
        # entsteht die vorgeschlagene Trajektorie.
        drivable_rows, drivable_x = [], []
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
            if not decided[row, clipped_start:clipped_stop].any():
                # Das Modell hat hier gar nicht geurteilt. Solche Zeilen gehen
                # nicht in die Bewertung ein, statt still als Hindernis zu zaehlen.
                undecided_rows += 1
                continue
            slice_free = free[row, clipped_start:clipped_stop]
            slice_uncertain = uncertain[row, clipped_start:clipped_stop]
            status = _row_status(slice_free, slice_uncertain, strips[row], outside / band)
            counts[status] += 1
            if status != "blocked":
                # Der beste Platz in dieser Zeile ist die Mitte des breitesten
                # befahrbaren Laufs — nicht die Mitte des Korridors.
                usable = slice_free if status == "free" else (slice_free | slice_uncertain)
                run_start, run_length = _widest_run(usable)
                if run_length:
                    drivable_rows.append(row)
                    drivable_x.append(clipped_start + run_start + (run_length - 1) / 2)
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
        drawn = _sample(list(range(first_row, height)))
        smoothed = _smooth(drivable_x)
        trajectory_rows = _sample(drivable_rows)
        trajectory = [(smoothed[drivable_rows.index(row)], row) for row in trajectory_rows if row in drivable_rows]
        corridors.append(
            {
                "corridor": name,
                "label": CORRIDOR_LABELS[name],
                "meaning": CORRIDOR_MEANINGS[name],
                "status": status,
                "status_label": STATUS_LABELS[status],
                "reason": reason,
                "rows": {"evaluated": evaluated, **counts, "undecided": undecided_rows},
                "bottom_center_x": round(float(width / 2 + offset * strip_px_bottom), 3),
                # Normierte Polylinien fuer die Darstellung: Mittellinie und die
                # beiden Raender des Suchbands, von der Fluchtpunktzeile abwaerts.
                "geometry": {
                    "center": _normalized([(centers[row], row) for row in drawn], mask.shape),
                    "left": _normalized([(centers[row] - bands[row] / 2, row) for row in drawn], mask.shape),
                    "right": _normalized([(centers[row] + bands[row] / 2, row) for row in drawn], mask.shape),
                },
                "trajectory": {
                    "points": _normalized(trajectory, mask.shape),
                    "rows": len(drivable_rows),
                    "source": "widest_drivable_run_center_per_row",
                },
            }
        )

    # Vorschlag der KI: der bevorzugte Korridor, der noch befahrbar ist. Er ist
    # ein Angebot zum Weiterbearbeiten, keine Fahrempfehlung.
    candidates = sorted(
        (item for item in corridors if item["trajectory"]["points"]),
        key=lambda item: (STATUS_PREFERENCE[item["status"]], CORRIDOR_PREFERENCE.index(item["corridor"])),
    )
    best = candidates[0] if candidates else None
    proposed = (
        None
        if best is None
        else {
            "corridor": best["corridor"],
            "label": best["label"],
            "status": best["status"],
            "status_label": STATUS_LABELS[best["status"]],
            "points": best["trajectory"]["points"],
            "source": best["trajectory"]["source"],
            "note": (
                f"Vorschlag im Korridor {best['label']} ({STATUS_LABELS[best['status']]}): je Zeile die Mitte des "
                "breitesten befahrbaren Laufs. Als Ausgangspunkt zum Nachbessern gedacht."
            ),
        }
    )

    return {
        "schema_version": CORRIDOR_SCHEMA_VERSION,
        "kind": "image_space_corridor_check",
        "mask_size": {"width": width, "height": height},
        "decomposition": {
            **decomposition,
            "vanishing_point_normalized": _normalized([(vanishing_point["x"], vanishing_point["y"])], mask.shape)[0],
            "relevant_triangle_normalized": _normalized(
                [(0, height - 1), (width - 1, height - 1), (vanishing_point["x"], vanishing_point["y"])],
                mask.shape,
            ),
            "first_evaluated_row_normalized": round(first_row / max(1, height - 1), 5),
        },
        "proposed_trajectory": proposed,
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
