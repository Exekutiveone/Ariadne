"""Missionsübergreifende Evaluation des CPU-Wegmodells.

Beantwortet die Frage, die das bestehende Validierungsschema nicht beantworten
kann: Wie gut arbeitet ein auf einer Mission trainiertes Modell auf einer
anderen Mission? Der bestehende Trainingscode wird ausschließlich importiert
und nicht verändert.

Zwei Punkte entscheiden über die Ehrlichkeit der Zahlen:

1. Die Entscheidungsschwelle wird ausschließlich auf zurückgehaltenen Frames
   der TRAININGSMISSION gewählt. Die Frames der Evaluationsmission werden bis
   zur finalen Messung nie berührt — weder für die Schwelle noch sonst.
2. Die bestehende In-Mission-Metrik trainiert und validiert auf Frames
   derselben Missionen (jeder 5. Frame zurückgehalten). Benachbarte Videoframes
   sind visuell hochkorreliert, deshalb ist diese Zahl systematisch optimistisch
   und taugt nicht als Aussage über neue Waldstücke.

`choose_threshold` aus path_model.py kann hier nicht wiederverwendet werden,
weil es intern `pixel_features` mit fester Merkmalszahl aufruft und damit die
Merkmalsvariante (mit/ohne Positionsmerkmale) ignorieren würde. Die
Schwellenwahl ist unten deshalb erneut formuliert — bewusst mit denselben
Kandidatenquantilen und demselben symmetrischen Kriterium.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .path_dataset import confirmed_annotations, frame_split, read_frames
from .path_features import (
    MODEL_WIDTH,
    RANDOM_FEATURES,
    RANDOM_SEED,
    RIDGE_LAMBDA,
    SAMPLES_PER_CLASS_PER_FRAME,
    clean_prediction,
    fit_kernel_classifier,
    pixel_features,
    predict_scores,
)
from .path_masks import confusion_counts, symmetric_metrics

# Die letzten acht Spalten von pixel_features sind ortsabhängig: x, y, x², y²,
# |x-0.5| sowie die drei Produkte green*y, saturation*y, value*y. Ein Modell
# kann darüber lernen "unten im Bild ist Weg", was innerhalb einer Mission
# funktioniert und auf fremdem Gelände zusammenbrechen kann. test_eval_cross_mission
# pinnt diese Annahme gegen Änderungen an pixel_features.
POSITION_FEATURE_COUNT = 8
THRESHOLD_QUANTILES = np.linspace(0.12, 0.88, 33)


def _features_for(image: np.ndarray, include_position: bool):
    """Merkmale eines Frames, wahlweise ohne die ortsabhängigen Spalten."""
    features = pixel_features(image)
    return features if include_position else features[:, :-POSITION_FEATURE_COUNT]


def _sample_training_pixels(frames, include_position: bool, seed: int = RANDOM_SEED):
    rng = np.random.default_rng(seed)
    samples, labels = [], []
    for item in frames:
        features = _features_for(item["image"], include_position)
        flat = item["mask"].reshape(-1)
        positive = np.flatnonzero(flat == 1)
        negative = np.flatnonzero(flat == 0)
        count = min(SAMPLES_PER_CLASS_PER_FRAME, len(positive), len(negative))
        if not count:
            continue
        indices = np.concatenate(
            [rng.choice(positive, count, replace=False), rng.choice(negative, count, replace=False)]
        )
        samples.append(features[indices])
        labels.append(np.concatenate([np.ones(count, np.uint8), np.zeros(count, np.uint8)]))
    if not samples:
        raise ValueError("Keine verwertbaren Trainingspixel in den Trainingsframes")
    training_samples = np.vstack(samples)
    training_labels = np.concatenate(labels)
    order = rng.permutation(len(training_labels))
    return training_samples[order], training_labels[order]


def _score_frames(frames, model, include_position: bool):
    return [predict_scores(_features_for(item["image"], include_position), model) for item in frames]


def _choose_threshold_on(frames, scores):
    """Schwellenwahl auf genau den übergebenen Frames — hier nie die Evaluationsmission."""
    merged = np.concatenate(scores)
    candidates = np.unique(np.quantile(merged, THRESHOLD_QUANTILES))
    best = None
    for threshold in candidates:
        total = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
        for item, frame_scores in zip(frames, scores, strict=True):
            counts = confusion_counts(
                item["mask"], clean_prediction(frame_scores, item["mask"].shape, float(threshold))
            )
            for key in total:
                total[key] += counts[key]
        metrics = symmetric_metrics(total)
        if best is None or metrics["symmetric_penalty_points"] < best[1]["symmetric_penalty_points"]:
            best = (float(threshold), metrics)
    if best is None:
        raise ValueError("Keine Schwelle bestimmbar")
    return best


def _evaluate_frames(frames, model, threshold: float, include_position: bool):
    total = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    per_frame = []
    for item in frames:
        scores = predict_scores(_features_for(item["image"], include_position), model)
        prediction = clean_prediction(scores, item["mask"].shape, threshold)
        counts = confusion_counts(item["mask"], prediction)
        for key in total:
            total[key] += counts[key]
        per_frame.append({"item": item, "prediction": prediction, "metrics": symmetric_metrics(counts)})
    return symmetric_metrics(total), per_frame


def pick_evidence_frames(per_frame, worst: int = 3, best: int = 2):
    """Die schlechtesten und besten Frames nach IoU, ohne Doppelnennung."""
    ranked = sorted(per_frame, key=lambda entry: entry["metrics"]["iou"])
    chosen = [("worst", entry) for entry in ranked[:worst]]
    used = {id(entry) for _, entry in chosen}
    for entry in reversed(ranked):
        if len(chosen) >= worst + best:
            break
        if id(entry) not in used:
            chosen.append(("best", entry))
            used.add(id(entry))
    return chosen


def write_evidence(directory: Path, run_key: str, per_frame):
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for rank, (kind, entry) in enumerate(pick_evidence_frames(per_frame)):
        item = entry["item"]
        image = item["image"]
        truth = item["mask"].astype(bool)
        prediction = entry["prediction"].astype(bool)
        colours = np.zeros_like(image)
        colours[truth & prediction] = (70, 210, 90)
        colours[truth & ~prediction] = (55, 55, 235)
        colours[~truth & prediction] = (40, 210, 240)
        overlay = image.copy()
        changed = truth | prediction
        overlay[changed] = cv2.addWeighted(image, 0.38, colours, 0.62, 0)[changed]
        record = item["record"]
        name = f"{run_key}-{rank}-{kind}-{record['video_id'][:8]}-{record['frame_index']:07d}.jpg"
        cv2.imwrite(str(directory / name), np.hstack([image, overlay]), [cv2.IMWRITE_JPEG_QUALITY, 88])
        written.append(
            {
                "kind": kind,
                "file": name,
                "iou": entry["metrics"]["iou"],
                **{"video_id": record["video_id"], "frame_index": record["frame_index"]},
            }
        )
    return written


def run_single(train_frames, eval_frames, include_position: bool):
    """Ein Evaluationslauf: fitten, Schwelle auf Trainingsmission, dann messen."""
    fit_frames, threshold_frames = frame_split([item["record"] for item in train_frames])
    by_key = {(item["record"]["video_id"], item["record"]["frame_index"]): item for item in train_frames}
    fit_items = [
        by_key[(r["video_id"], r["frame_index"])] for r in fit_frames if (r["video_id"], r["frame_index"]) in by_key
    ]
    threshold_items = [
        by_key[(r["video_id"], r["frame_index"])]
        for r in threshold_frames
        if (r["video_id"], r["frame_index"]) in by_key
    ]
    if not fit_items or not threshold_items:
        raise ValueError("Trainingsmission liefert zu wenige Frames für einen internen Split")

    samples, labels = _sample_training_pixels(fit_items, include_position)
    model = fit_kernel_classifier(samples, labels, RANDOM_FEATURES, RIDGE_LAMBDA, RANDOM_SEED)
    threshold, threshold_metrics = _choose_threshold_on(
        threshold_items, _score_frames(threshold_items, model, include_position)
    )
    metrics, per_frame = _evaluate_frames(eval_frames, model, threshold, include_position)
    return {
        "threshold": round(threshold, 6),
        "threshold_selected_on": "held_out_frames_of_training_mission_only",
        "threshold_metrics": threshold_metrics,
        "fit_frames": len(fit_items),
        "threshold_frames": len(threshold_items),
        "eval_frames": len(eval_frames),
        "feature_count": int(model["mean"].shape[0]),
        "metrics": metrics,
        "per_frame": per_frame,
    }


def render_report(rows, baseline, generated_at: str, missions=None):
    """Markdown-Tabelle; `rows` enthält je Lauf bereits fertige Zahlen."""
    lines = [
        "# Missionsübergreifende Evaluation des CPU-Wegmodells",
        "",
        f"Erzeugt: {generated_at}",
        "",
    ]
    if missions:
        lines += [
            "## Datenbasis",
            "",
            "| Mission | Bestätigte Frames | In dieser Evaluation |",
            "|---|---|---|",
        ]
        selected = {missions["first"]["name"], missions["second"]["name"]}
        for item in missions.get("available", []):
            lines.append(
                f"| {item['name']} | {item['confirmed_frames']} | {'ja' if item['name'] in selected else 'nein'} |"
            )
        lines.append("")
    lines += [
        "Jeder Lauf trainiert auf **allen** bestätigten Frames einer Mission und misst auf",
        "**allen** bestätigten Frames der jeweils anderen. Die Entscheidungsschwelle wird",
        "ausschließlich auf zurückgehaltenen Frames der Trainingsmission gewählt; die",
        "Evaluationsframes fließen an keiner Stelle in Training oder Schwellenwahl ein.",
        "",
        "## Ergebnisse",
        "",
        "| Lauf | Training | Evaluation | Positionsmerkmale | Merkmale | IoU | Precision | Recall | Eval-Frames |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        metrics = row["metrics"]
        lines.append(
            f"| {row['key']} | {row['train_mission']} | {row['eval_mission']} | "
            f"{'ja' if row['include_position'] else 'nein'} | {row['feature_count']} | "
            f"{metrics['iou']:.3f} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | {row['eval_frames']} |"
        )
    lines += [
        "",
        "## Vergleichswert: bisherige In-Mission-Metrik",
        "",
        "| Quelle | Training | Validierung | IoU | Precision | Recall |",
        "|---|---|---|---|---|---|",
    ]
    for item in baseline:
        metrics = item["metrics"]
        lines.append(
            f"| {item['label']} | {item['train']} | {item['validation']} | "
            f"{metrics['iou']:.3f} | {metrics['precision']:.3f} | {metrics['recall']:.3f} |"
        )
    lines += [
        "",
        "Diese Zahl hält jeden 5. Frame derselben Missionen zurück. Benachbarte Videoframes",
        "sind visuell hochkorreliert, deshalb ist sie systematisch optimistisch und keine",
        "Aussage über unbekanntes Gelände.",
        "",
        "## Evidenzframes",
        "",
        "Je Lauf die drei schlechtesten und zwei besten Frames nach IoU, links das",
        "Originalbild, rechts das Overlay: grün korrekt, rot übersehene Wegfläche,",
        "gelb fälschlich erkannte Wegfläche.",
        "",
    ]
    for row in rows:
        lines.append(f"### {row['key']} — {row['train_mission']} → {row['eval_mission']}")
        lines.append("")
        for item in row["evidence"]:
            lines.append(f"- `{item['file']}` — {item['kind']}, IoU {item['iou']:.3f}, Frame {item['frame_index'] + 1}")
        lines.append("")
    lines += [
        "## Vorbehalt",
        "",
        "Die Ausgabe ist eine KI-gestützte Einschätzung und keine sicherheitsrelevante",
        "Fahrfreigabe.",
        "",
    ]
    return "\n".join(lines)


def resolve_missions(candidates, selection=None):
    """Waehlt die zwei zu vergleichenden Missionen.

    Ohne explizite Auswahl werden die beiden Missionen mit den **meisten**
    bestaetigten Frames genommen — nicht die juengsten. Eine frisch angelegte
    Mission mit einer Handvoll Labels wuerde sonst stillschweigend die
    Datenbasis der Evaluation bestimmen.
    """
    if selection:
        chosen = []
        for wanted in selection:
            match = next((item for item in candidates if wanted in (item["id"], item["name"])), None)
            if match is None:
                known = ", ".join(sorted(item["name"] for item in candidates))
                raise ValueError(f"Mission {wanted!r} nicht gefunden oder ohne bestätigte Labels. Verfügbar: {known}")
            chosen.append(match)
        if chosen[0]["id"] == chosen[1]["id"]:
            raise ValueError("Für eine missionsübergreifende Evaluation werden zwei verschiedene Missionen benötigt")
        return chosen[0], chosen[1]
    if len(candidates) < 2:
        raise ValueError("Für eine missionsübergreifende Evaluation werden zwei Missionen mit Labels benötigt")
    ranked = sorted(candidates, key=lambda item: item["confirmed"], reverse=True)
    return ranked[0], ranked[1]


def evaluate_cross_mission(store, selection=None, output_dir: Path | None = None, width: int = MODEL_WIDTH):
    candidates = []
    for mission in store.list():
        records = confirmed_annotations(store.root / mission.id)
        if records:
            candidates.append({"id": mission.id, "name": mission.name, "mission": mission, "confirmed": len(records)})
    first_entry, second_entry = resolve_missions(candidates, selection)
    first, second = first_entry["mission"], second_entry["mission"]

    # Frames einmal dekodieren und über alle vier Läufe wiederverwenden.
    decoded = {}
    for mission in (first, second):
        mission_dir = store.root / mission.id
        records = confirmed_annotations(mission_dir)
        decoded[mission.id] = read_frames(mission, mission_dir, records, width)
        if len(decoded[mission.id]) < 4:
            raise ValueError(f"Mission {mission.name} liefert zu wenige dekodierbare Labelframes")

    plan = [
        ("A", first, second, True),
        ("B", second, first, True),
        ("C", first, second, False),
        ("D", second, first, False),
    ]
    output = output_dir or (store.root.parent / "global_models" / "cross_mission_eval")
    evidence_dir = output / "evidence"
    rows = []
    for key, train_mission, eval_mission, include_position in plan:
        result = run_single(decoded[train_mission.id], decoded[eval_mission.id], include_position)
        evidence = write_evidence(evidence_dir, key, result["per_frame"])
        rows.append(
            {
                "key": key,
                "train_mission": train_mission.name,
                "eval_mission": eval_mission.name,
                "include_position": include_position,
                "feature_count": result["feature_count"],
                "threshold": result["threshold"],
                "threshold_selected_on": result["threshold_selected_on"],
                "fit_frames": result["fit_frames"],
                "threshold_frames": result["threshold_frames"],
                "eval_frames": result["eval_frames"],
                "metrics": result["metrics"],
                "evidence": evidence,
            }
        )

    baseline = []
    try:
        pointer = json.loads((store.root.parent / "global_models" / "path_model" / "current.json").read_text("utf-8"))
        active = json.loads(
            (store.root.parent / "global_models" / "path_model" / "runs" / pointer["run_id"] / "result.json").read_text(
                "utf-8"
            )
        )
        baseline.append(
            {
                "label": f"Aktives globales Modell ({active['run_id']})",
                "train": f"{active['split']['train_frames']} Frames beider Missionen",
                "validation": f"{active['split']['validation_frames']} Frames derselben Missionen",
                "metrics": active["validation_metrics"],
            }
        )
    except (OSError, ValueError, KeyError):
        pass

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    output.mkdir(parents=True, exist_ok=True)
    mission_summary = {
        "first": {"name": first.name, "id": first.id, "confirmed_frames": first_entry["confirmed"]},
        "second": {"name": second.name, "id": second.id, "confirmed_frames": second_entry["confirmed"]},
        "available": [{"name": item["name"], "confirmed_frames": item["confirmed"]} for item in candidates],
    }
    (output / "evaluation_cross_mission.md").write_text(
        render_report(rows, baseline, generated_at, mission_summary), encoding="utf-8"
    )
    summary = {
        "generated_at": generated_at,
        "missions": mission_summary,
        "runs": [{key: row[key] for key in row if key != "evidence"} | {"evidence": row["evidence"]} for row in rows],
        "baseline": baseline,
    }
    (output / "evaluation_cross_mission.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    import os
    import sys

    from .storage import MissionStore

    root = Path(os.getenv("ARIADNE_DATA_DIR", Path(__file__).resolve().parents[2] / "data" / "missions"))
    # Optional zwei Missionsnamen oder -ids als Argumente; sonst die beiden
    # Missionen mit den meisten bestätigten Frames.
    selection = tuple(sys.argv[1:3]) if len(sys.argv) > 2 else None
    result = evaluate_cross_mission(MissionStore(root), selection)
    chosen = result["missions"]
    print(
        f"Missionen: {chosen['first']['name']} ({chosen['first']['confirmed_frames']} Labels) "
        f"vs. {chosen['second']['name']} ({chosen['second']['confirmed_frames']} Labels)"
    )
    for run in result["runs"]:
        metrics = run["metrics"]
        print(
            f"{run['key']}: {run['train_mission']} -> {run['eval_mission']} "
            f"(Position {'ja' if run['include_position'] else 'nein'}, {run['feature_count']} Merkmale) "
            f"IoU {metrics['iou']:.3f} P {metrics['precision']:.3f} R {metrics['recall']:.3f}"
        )
