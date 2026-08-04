"""Bildraum-Korridore mit fester Nahfeld-Geometrie.

Das Modul bewertet drei direkt vergleichbare, parallele Fahrstreifen im
sichtbaren Nahfeld der Wegmaske: Mitte, rechts und links. Es gibt keinen
Geradenfit der Wegraender und keine Hochrechnung in die Bildferne. Die drei
Streifen sind ausschliesslich eine nachvollziehbare Pruefgeometrie auf der
aktuellen Maske.

Die Ausgabe ist eine KI-gestuetzte Einschaetzung und keine Fahrfreigabe.
"""

import numpy as np

CORRIDOR_SCHEMA_VERSION = "2.0"

DEFAULT_VEHICLE_WIDTH_M = 1.2
DEFAULT_CLEARANCE_M = 0.10
DEFAULT_GROUND_WIDTH_AT_BOTTOM_M = 4.0

CORRIDOR_SEARCH_FACTOR = 1.5
CORRIDOR_OFFSETS = {"mitte": 0.0, "rechts": 1.0, "links": -1.0}
CORRIDOR_LABELS = {"mitte": "Mitte", "rechts": "Rechts", "links": "Links"}
CORRIDOR_MEANINGS = {
    "mitte": "Standardspur im sichtbaren Nahfeld",
    "rechts": "Rechte Ausweich- oder Fahrspur",
    "links": "Linke Ausweich- oder Fahrspur",
}
STATUS_LABELS = {"free": "frei", "blocked": "blockiert", "uncertain": "unsicher"}

# Nur das untere Sichtfeld wird bewertet. Das ist kein Horizont- oder
# Perspektivenschaetzer, sondern eine feste Schutzregel gegen ferne, zu kleine
# Bildbereiche, in denen eine Fahrzeugbreite nicht sinnvoll pruefbar ist.
NEAR_FIELD_START_FRACTION = 0.42
MIN_EVALUATED_ROWS = 8
MAX_CLIPPED_FRACTION = 0.2
MAX_BLOCKED_ROW_FRACTION = 0.05
GEOMETRY_SAMPLE_ROWS = 28
TRAJECTORY_SMOOTHING = 9
CORRIDOR_PREFERENCE = ("mitte", "rechts", "links")
STATUS_PREFERENCE = {"free": 0, "uncertain": 1, "blocked": 2}

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


def _classify(mask: np.ndarray, grade_mask: np.ndarray | None):
    if grade_mask is None:
        free = mask.astype(bool)
        return free, np.zeros_like(free), np.ones_like(free)
    free = np.isin(grade_mask, list(FREE_GRADES))
    uncertain = np.isin(grade_mask, list(UNCERTAIN_GRADES))
    return free, uncertain, grade_mask != 0


def _near_field(shape):
    height, _ = shape
    first_row = min(height - 1, max(0, int(round(height * NEAR_FIELD_START_FRACTION))))
    return {
        "kind": "fixed_near_field_band",
        "first_evaluated_row": first_row,
        "rows_skipped": first_row,
        "image_fraction_skipped": round(first_row / max(1, height), 5),
        "reason": "Nur das untere sichtbare Nahfeld wird auf Fahrzeugbreite geprueft.",
        "evaluated_rows": height - first_row,
        "first_evaluated_row_normalized": round(first_row / max(1, height - 1), 5),
    }


def _corridor_geometry(shape, strip_px: float, offset: float, first_row: int):
    """Drei parallele Streifen ohne perspektivische Verjuengung."""
    height, width = shape
    rows = np.arange(first_row, height, dtype=np.float64)
    center = width / 2 + offset * strip_px
    centers = np.full_like(rows, center)
    strip = np.full_like(rows, strip_px)
    return rows, centers, strip, strip * CORRIDOR_SEARCH_FACTOR


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
    height, width = shape
    return [
        [round(float(np.clip(x, 0, width - 1)) / max(1, width - 1), 5), round(float(y) / max(1, height - 1), 5)]
        for x, y in points
    ]


def _sample(rows, count: int = GEOMETRY_SAMPLE_ROWS):
    if len(rows) <= count:
        return list(rows)
    picks = np.linspace(0, len(rows) - 1, count).round().astype(int)
    return [rows[index] for index in dict.fromkeys(picks.tolist())]


def _smooth(values, window: int = TRAJECTORY_SMOOTHING):
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
    """Status je explizit ausgewaehltem Korridor fuer einen einzelnen Frame."""
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.size == 0:
        raise ValueError("Die Maske muss ein nicht leeres 2D-Raster sein")
    if grade_mask is not None:
        grade_mask = np.asarray(grade_mask)
        if grade_mask.shape != mask.shape:
            raise ValueError("Abstufungsmaske und Maske muessen dieselbe Groesse haben")
    if ground_width_at_bottom_m <= 0:
        raise ValueError("Die Bodenbreite am unteren Bildrand muss groesser als null sein")

    height, width = mask.shape
    free, uncertain, decided = _classify(mask, grade_mask)
    region = _near_field(mask.shape)
    first_row = region["first_evaluated_row"]

    strip_m = vehicle_width_m + clearance_m
    strip_px = strip_m / ground_width_at_bottom_m * width
    corridors = []
    for name, offset in CORRIDOR_OFFSETS.items():
        rows, centers, strips, bands = _corridor_geometry(mask.shape, strip_px, offset, first_row)
        counts = {"free": 0, "uncertain": 0, "blocked": 0}
        undecided_rows = 0
        drivable_rows, drivable_x = [], []
        for index, row_value in enumerate(rows):
            row = int(row_value)
            band = bands[index]
            start = centers[index] - band / 2
            stop = centers[index] + band / 2
            clipped_start = int(np.clip(np.floor(start), 0, width))
            clipped_stop = int(np.clip(np.ceil(stop), 0, width))
            outside = max(0.0, -start) + max(0.0, stop - width)
            if clipped_start >= clipped_stop or not decided[row, clipped_start:clipped_stop].any():
                undecided_rows += 1
                continue
            slice_free = free[row, clipped_start:clipped_stop]
            slice_uncertain = uncertain[row, clipped_start:clipped_stop]
            status = _row_status(slice_free, slice_uncertain, strips[index], outside / max(1.0, band))
            counts[status] += 1
            if status != "blocked":
                usable = slice_free if status == "free" else (slice_free | slice_uncertain)
                run_start, run_length = _widest_run(usable)
                if run_length:
                    drivable_rows.append(row)
                    drivable_x.append(clipped_start + run_start + (run_length - 1) / 2)

        evaluated = sum(counts.values())
        if evaluated < MIN_EVALUATED_ROWS:
            status, reason = "uncertain", "Zu wenige auswertbare Zeilen im sichtbaren Nahfeld."
        elif counts["blocked"] > MAX_BLOCKED_ROW_FRACTION * evaluated:
            status = "blocked"
            reason = f"In {counts['blocked']} von {evaluated} Zeilen passt kein freier Streifen in den Korridor."
        elif counts["blocked"]:
            status = "uncertain"
            reason = f"{counts['blocked']} von {evaluated} Zeilen sind gesperrt; der Korridor bleibt deshalb unsicher."
        elif counts["uncertain"]:
            status = "uncertain"
            reason = f"In {counts['uncertain']} von {evaluated} Zeilen ist der Streifen nicht sicher frei."
        else:
            status = "free"
            reason = f"In allen {evaluated} ausgewerteten Zeilen passt ein freier Streifen in den Korridor."

        drawn_indexes = _sample(list(range(len(rows))))
        smoothed = _smooth(drivable_x)
        trajectory = [
            (smoothed[drivable_rows.index(row)], row)
            for row in _sample(drivable_rows)
            if row in drivable_rows
        ]
        corridors.append(
            {
                "corridor": name,
                "label": CORRIDOR_LABELS[name],
                "meaning": CORRIDOR_MEANINGS[name],
                "status": status,
                "status_label": STATUS_LABELS[status],
                "reason": reason,
                "rows": {"evaluated": evaluated, **counts, "undecided": undecided_rows},
                "bottom_center_x": round(float(width / 2 + offset * strip_px), 3),
                "geometry": {
                    "center": _normalized([(centers[index], rows[index]) for index in drawn_indexes], mask.shape),
                    "left": _normalized([(centers[index] - bands[index] / 2, rows[index]) for index in drawn_indexes], mask.shape),
                    "right": _normalized([(centers[index] + bands[index] / 2, rows[index]) for index in drawn_indexes], mask.shape),
                },
                "trajectory": {
                    "points": _normalized(trajectory, mask.shape),
                    "rows": len(drivable_rows),
                    "source": "widest_drivable_run_center_in_parallel_near_field_band",
                },
            }
        )

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
            "note": f"Vorschlag im Korridor {best['label']} ({STATUS_LABELS[best['status']]}). Als Ausgangspunkt zum Nachbessern gedacht.",
        }
    )

    return {
        "schema_version": CORRIDOR_SCHEMA_VERSION,
        "kind": "parallel_near_field_corridor_check",
        "mask_size": {"width": width, "height": height},
        "region": region,
        "proposed_trajectory": proposed,
        "strip": {
            "vehicle_width_m": vehicle_width_m,
            "clearance_m": clearance_m,
            "required_width_m": round(strip_m, 4),
            "ground_width_at_bottom_m": ground_width_at_bottom_m,
            "required_width_px_at_bottom": round(float(strip_px), 3),
            "scaling": "konstante parallele Streifen im sichtbaren Nahfeld",
            "search_band_factor": CORRIDOR_SEARCH_FACTOR,
        },
        "corridors": corridors,
        "graded_input": grade_mask is not None,
        "limitations": [
            "Nur Breitenpruefung im sichtbaren Nahfeld; freie Strecke in die Ferne wird nicht bewertet.",
            "Die Bodenbreite am unteren Bildrand ist eine Kalibrierung pro Kameraaufbau und keine Messung aus dem Bild.",
            "Deterministische Geometrie auf einer vorhergesagten Maske — keine sicherheitsrelevante Fahrfreigabe.",
        ],
    }
