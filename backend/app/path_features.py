"""Merkmale und Klassifikator des Wegmodells.

Reine Numerik ohne Dateizugriff: Pixelmerkmale, Random-Feature-Ridge und die
Abstufung der Vorhersage. Alles, was Pfade oder Videos anfasst, liegt in
`path_dataset`; das Training und die Laufverwaltung in `path_model`.

Aendert sich `pixel_features`, werden alle gespeicherten Modelle ungueltig —
dann muss MODEL_SCHEMA_VERSION in `path_model` steigen. `test_path_model_core`
pinnt die Merkmalszahl genau deshalb.
"""

import math

import cv2
import numpy as np

MODEL_WIDTH = 160
RANDOM_FEATURES = 64
SAMPLES_PER_CLASS_PER_FRAME = 450
RIDGE_LAMBDA = 0.08
RANDOM_SEED = 42
# Abstufung der binären Wegmaske in sechs Anzeigeklassen (AGENT_ANWEISUNG.md).
# Farb- und Wertetabelle ist mit dem Nutzer abgestimmt und im Frontend exakt
# zu übernehmen; Aufbau wie GROUND_TRUTH_ONTOLOGY in annotations.py.
GRADE_ONTOLOGY = {
    "unrated": {"value": 0, "label": "Nicht bewertet / Umgebung", "color": "#00000000"},
    "safe": {"value": 1, "label": "Sicher befahrbar", "color": "#1e8c46"},
    "good": {"value": 2, "label": "Gut befahrbar", "color": "#55d96f"},
    "marginal": {"value": 3, "label": "Knapp befahrbar", "color": "#a3ecb4"},
    "risky": {"value": 4, "label": "Potenziell befahrbar, mit Risiko", "color": "#f08c3a"},
    "problem": {"value": 5, "label": "Problemzone / Hindernis", "color": "#e05b52"},
}
# Bandgrenzen auf dem normierten Abstand m = (score - threshold) / max(1e-6, 1 - threshold).
# Startwerte laut Arbeitsanweisung; sie stehen in jeder API-Antwort (grading-Block),
# damit Ergebnisse reproduzierbar bleiben.
GRADE_SAFE_MIN_MARGIN = 0.6
GRADE_GOOD_MIN_MARGIN = 0.25
GRADE_RISKY_MIN_MARGIN = -0.2
# Rote Problemzonen: nur zusammenhängende sicher-negative Flächen, die mindestens
# diesen Bildanteil belegen und der Grünfläche nahe kommen (Dilatationsradius in
# Pixeln des Modellrasters, MODEL_WIDTH breit).
GRADE_PROBLEM_MIN_AREA_FRACTION = 0.002
GRADE_PROBLEM_NEIGHBOURHOOD = 9
# Qualifizierte Komponenten färben nur ihre Pixel in diesem Band um den Fahrbereich
# (Kernelgröße der Dilatation, Reichweite ~12 px im Modellraster). Nötig, weil in
# echten Waldframes Wald und Himmel EINE zusammenhängende Negativkomponente bilden,
# die den Weg immer irgendwo berührt — ohne Begrenzung würde der gesamte Hintergrund
# rot. Realdaten-Befund vom 03.08.2026: ~50 % Rotanteil pro Frame statt Hindernissen
# direkt am Weg. Himmel und ferne Umgebung bleiben mit dem Band transparent.
GRADE_PROBLEM_CLIP = 25
# Die fuenf reinen Positionskanaele haengen nur von der Rastergroesse ab und
# werden pro Shape einmal berechnet. Werte sind identisch zur Direktberechnung;
# es treten nur wenige Shapes auf (MODEL_WIDTH x videoabhaengige Hoehe).
_POSITION_CACHE: dict = {}


def _position_channels(height: int, width: int):
    cached = _POSITION_CACHE.get((height, width))
    if cached is None:
        x = np.linspace(0, 1, width, dtype=np.float32)[None, :].repeat(height, axis=0)
        y = np.linspace(0, 1, height, dtype=np.float32)[:, None].repeat(width, axis=1)
        cached = (x, y, x**2, y**2, np.abs(x - 0.5))
        _POSITION_CACHE[(height, width)] = cached
    return cached


def pixel_features(image: np.ndarray):
    pixels = image.astype(np.float32) / 255.0
    blue, green, red = cv2.split(pixels)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
    hue = hsv[:, :, 0] * (2 * math.pi / 180.0)
    saturation = hsv[:, :, 1] / 255.0
    value = hsv[:, :, 2] / 255.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.clip(np.sqrt(gradient_x**2 + gradient_y**2), 0, 1)
    local_mean = cv2.blur(gray, (9, 9))
    local_square = cv2.blur(gray**2, (9, 9))
    local_std = np.sqrt(np.maximum(0, local_square - local_mean**2))
    height, width = gray.shape
    x, y, x_squared, y_squared, x_center_distance = _position_channels(height, width)
    excess_green = np.clip(2 * green - red - blue, -1, 1)
    channels = [
        blue,
        green,
        red,
        np.sin(hue),
        np.cos(hue),
        saturation,
        value,
        lab[:, :, 0],
        lab[:, :, 1],
        lab[:, :, 2],
        excess_green,
        gradient,
        local_mean,
        local_std,
        x,
        y,
        x_squared,
        y_squared,
        x_center_distance,
        green * y,
        saturation * y,
        value * y,
    ]
    return np.stack(channels, axis=-1).reshape(-1, len(channels)).astype(np.float32)


def sample_training_pixels(frames, samples_per_class_per_frame: int, rng: np.random.Generator):
    """Zieht Trainingspixel je Frame — ausbalanciert, wo beide Klassen da sind.

    Ein Frame ganz ohne Wegflaeche (Off-Path-Intervall, komplett nicht
    befahrbares Video) braucht kein Positivgegenstueck: er liefert reine
    Negativbeispiele. Ohne diesen Sonderfall wuerde `min(count, len(positive),
    len(negative))` mit `len(positive) == 0` immer 0 ergeben — die Balance-Regel
    wuerfe genau die Frames komplett aus dem Training, die extra dafuer markiert
    wurden, das Modell zu korrigieren. Ihr Effekt bliebe auf die berichteten
    Metriken beschraenkt, ohne je die gelernten Gewichte zu veraendern.
    """
    samples, labels = [], []
    for item in frames:
        features = pixel_features(item["image"])
        flat = item["mask"].reshape(-1)
        positive = np.flatnonzero(flat == 1)
        negative = np.flatnonzero(flat == 0)
        if len(positive) and len(negative):
            count = min(samples_per_class_per_frame, len(positive), len(negative))
            if not count:
                continue
            selected_positive = rng.choice(positive, count, replace=False)
            selected_negative = rng.choice(negative, count, replace=False)
            samples.append(features[np.concatenate([selected_positive, selected_negative])])
            labels.append(np.concatenate([np.ones(count, np.uint8), np.zeros(count, np.uint8)]))
        elif len(negative) and not len(positive):
            count = min(samples_per_class_per_frame, len(negative))
            samples.append(features[rng.choice(negative, count, replace=False)])
            labels.append(np.zeros(count, np.uint8))
    return samples, labels


def _standardize(values, mean, scale):
    return (values - mean) / scale


def _random_projection(values, projection, phase):
    return np.sqrt(2.0 / projection.shape[1]) * np.cos(values @ projection + phase)


def fit_kernel_classifier(samples, labels, random_features: int, ridge_lambda: float, seed: int):
    mean = samples.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = samples.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-5] = 1
    normalized = _standardize(samples, mean, scale)
    rng = np.random.default_rng(seed)
    projection = rng.normal(0, 1 / math.sqrt(samples.shape[1]), size=(samples.shape[1], random_features)).astype(
        np.float32
    )
    phase = rng.uniform(0, 2 * math.pi, size=random_features).astype(np.float32)
    hidden = _random_projection(normalized, projection, phase)
    design = np.column_stack([hidden, np.ones(len(hidden), np.float32)])
    targets = labels.astype(np.float32) * 2 - 1
    gram = design.T @ design
    gram.flat[:: gram.shape[0] + 1] += ridge_lambda
    weights = np.linalg.solve(gram.astype(np.float64), (design.T @ targets).astype(np.float64)).astype(np.float32)
    return {"mean": mean, "scale": scale, "projection": projection, "phase": phase, "weights": weights}


def predict_scores(features, model, chunk_size: int = 120_000):
    # Mathematisch identisch zu standardize -> projizieren -> cos -> gewichten,
    # aber Standardisierung und Kosinus-Amplitude sind in Projektionsmatrix und
    # Gewichte gefaltet: ((x - m) / s) @ P == x @ (P / s) - (m / s) @ P. Das
    # spart die grossen Zwischenarrays der Normalisierung und Skalierung; cos
    # laeuft in-place. Training (fit_kernel_classifier) nutzt weiter den
    # Referenzweg, Abweichungen liegen im float32-Rundungsbereich.
    inverse_scale = (1.0 / model["scale"]).astype(np.float32)
    projection = model["projection"] * inverse_scale[:, None]
    offset = (model["phase"] - (model["mean"] * inverse_scale) @ model["projection"]).astype(np.float32)
    amplitude = np.float32(math.sqrt(2.0 / model["projection"].shape[1]))
    cosine_weights = amplitude * model["weights"][:-1]
    bias = model["weights"][-1]
    output = np.empty(len(features), np.float32)
    for start in range(0, len(features), chunk_size):
        stop = min(len(features), start + chunk_size)
        hidden = features[start:stop] @ projection
        hidden += offset
        np.cos(hidden, out=hidden)
        output[start:stop] = hidden @ cosine_weights + bias
    return output


def clean_prediction(scores, shape, threshold):
    mask = (scores.reshape(shape) >= threshold).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return mask


def grade_prediction(scores, prediction, threshold, shape):
    """Stuft die bereinigte Binärmaske in die sechs Klassen von GRADE_ONTOLOGY ab.

    Rein additiv zur bestehenden Inferenz: Innerhalb von prediction==1 entstehen
    ausschließlich die Grünstufen 1-3, außerhalb ausschließlich 0, 4 oder 5.
    """
    # 3x3-Mittelung der Abstände glättet die Stufengrenzen, damit das Overlay im
    # Video nicht flimmert (leichte Glättung analog clean_prediction).
    margins = (scores.reshape(shape).astype(np.float32) - threshold) / max(1e-6, 1.0 - threshold)
    margins = cv2.blur(margins, (3, 3))
    inside = prediction.astype(bool)
    grades = np.zeros(shape, np.uint8)
    grades[inside] = GRADE_ONTOLOGY["marginal"]["value"]
    grades[inside & (margins >= GRADE_GOOD_MIN_MARGIN)] = GRADE_ONTOLOGY["good"]["value"]
    grades[inside & (margins >= GRADE_SAFE_MIN_MARGIN)] = GRADE_ONTOLOGY["safe"]["value"]
    grades[~inside & (margins >= GRADE_RISKY_MIN_MARGIN)] = GRADE_ONTOLOGY["risky"]["value"]
    # Rot nur für sicher-negative, zusammenhängende Flächen mit Mindestgröße in
    # unmittelbarer Nachbarschaft des Fahrbereichs; alles Übrige bleibt transparent.
    candidates = (~inside & (margins < GRADE_RISKY_MIN_MARGIN)).astype(np.uint8)
    candidates = cv2.morphologyEx(candidates, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if candidates.any() and inside.any():
        kernel = np.ones((GRADE_PROBLEM_NEIGHBOURHOOD, GRADE_PROBLEM_NEIGHBOURHOOD), np.uint8)
        near_path = cv2.dilate(inside.astype(np.uint8), kernel).astype(bool)
        clip_kernel = np.ones((GRADE_PROBLEM_CLIP, GRADE_PROBLEM_CLIP), np.uint8)
        clip_band = cv2.dilate(inside.astype(np.uint8), clip_kernel).astype(bool)
        min_area = max(1, int(GRADE_PROBLEM_MIN_AREA_FRACTION * shape[0] * shape[1]))
        count, components = cv2.connectedComponents(candidates, connectivity=8)
        for component in range(1, count):
            member = components == component
            if int(member.sum()) >= min_area and bool((member & near_path).any()):
                grades[member & clip_band] = GRADE_ONTOLOGY["problem"]["value"]
    return grades


def grading_summary(threshold: float):
    return {
        "margin": "m = (score - threshold) / max(1e-6, 1 - threshold)",
        "threshold": round(float(threshold), 6),
        "bands": {
            "safe_min_margin": GRADE_SAFE_MIN_MARGIN,
            "good_min_margin": GRADE_GOOD_MIN_MARGIN,
            "risky_min_margin": GRADE_RISKY_MIN_MARGIN,
        },
        "problem_min_area_fraction": GRADE_PROBLEM_MIN_AREA_FRACTION,
        "problem_neighbourhood_px": GRADE_PROBLEM_NEIGHBOURHOOD,
        "problem_clip_px": GRADE_PROBLEM_CLIP,
        "smoothing": "3x3 mean blur on margins, 3x3 opening on problem candidates",
        "note": "KI-Einschätzung der Befahrbarkeit, keine sicherheitsrelevante Fahrfreigabe.",
    }
