import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from .models import MissionRecord
from .path_model import current_path_model_dir, select_path_model_run, train_path_model
from .path_training_jobs import read_job, write_job


def _existing_best(mission_dir: Path):
    try:
        directory = current_path_model_dir(mission_dir)
        result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        return result["run_id"], float(result["validation_metrics"]["symmetric_score"])
    except (OSError, ValueError, KeyError):
        return None, -1.0


def run(mission_dir: Path, profile: str, duration_hours: float, job_id: str):
    mission = MissionRecord.model_validate_json((mission_dir / "mission.json").read_text(encoding="utf-8"))
    maximum = 1 if profile == "quick" else 128
    deadline = time.time() + duration_hours * 3600
    best_run_id, best_score = _existing_best(mission_dir)
    completed = 0
    for index in range(maximum):
        if index and time.time() >= deadline:
            break
        if profile == "quick":
            configuration = {
                "width": 160,
                "random_features": 64,
                "samples_per_class_per_frame": 450,
                "ridge_lambda": 0.08,
                "seed": 42,
            }
        else:
            widths = [160, 192, 224]
            feature_counts = [64, 96, 128]
            sample_counts = [450, 650]
            ridge_values = [0.03, 0.08, 0.16]
            configuration = {
                "width": widths[index % len(widths)],
                "random_features": feature_counts[(index // len(widths)) % len(feature_counts)],
                "samples_per_class_per_frame": sample_counts[(index // 9) % len(sample_counts)],
                "ridge_lambda": ridge_values[(index // 18) % len(ridge_values)],
                "seed": 42 + index * 17,
            }
        result = train_path_model(mission, mission_dir, **configuration)
        completed += 1
        score = float(result["validation_metrics"]["symmetric_score"])
        if score > best_score:
            best_run_id, best_score = result["run_id"], score
        if best_run_id:
            select_path_model_run(mission_dir, best_run_id)
        state = read_job(mission_dir) or {"job_id": job_id, "profile": profile}
        state.update(
            {
                "status": "running",
                "candidates_completed": completed,
                "best_run_id": best_run_id,
                "best_validation_score": best_score,
                "last_candidate_run_id": result["run_id"],
                "last_configuration": configuration,
                "message": f"Kandidat {completed}/{maximum} abgeschlossen; bester Score {best_score:.2f}/100",
            }
        )
        write_job(mission_dir, state)
    if best_run_id:
        select_path_model_run(mission_dir, best_run_id)
    state = read_job(mission_dir) or {"job_id": job_id, "profile": profile}
    state.update(
        {
            "status": "completed",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "candidates_completed": completed,
            "best_run_id": best_run_id,
            "best_validation_score": best_score,
            "message": f"Hintergrundtraining abgeschlossen; bester Score {best_score:.2f}/100",
        }
    )
    write_job(mission_dir, state)


if __name__ == "__main__":
    mission_directory = Path(sys.argv[1]).resolve()
    selected_profile = sys.argv[2]
    hours = float(sys.argv[3])
    selected_job_id = sys.argv[4]
    try:
        run(mission_directory, selected_profile, hours, selected_job_id)
    except Exception as exc:
        state = read_job(mission_directory) or {"job_id": selected_job_id, "profile": selected_profile}
        state.update(
            {
                "status": "failed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "message": str(exc),
                "traceback": traceback.format_exc(limit=8),
            }
        )
        write_job(mission_directory, state)
        raise
