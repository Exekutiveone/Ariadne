import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def _job_path(mission_dir: Path):
    return mission_dir / "derived" / "path_training_job.json"


def write_job(mission_dir: Path, state: dict):
    path = _job_path(mission_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def read_job(mission_dir: Path):
    path = _job_path(mission_dir)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _process_alive(pid: int):
    try:
        os.kill(pid, 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def training_job_status(mission_dir: Path):
    state = read_job(mission_dir)
    if (
        state
        and state.get("status") in {"queued", "running"}
        and state.get("pid")
        and not _process_alive(int(state["pid"]))
    ):
        state = {
            **state,
            "status": "interrupted",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "message": "Trainingsprozess läuft nicht mehr",
        }
        write_job(mission_dir, state)
    return state


def start_training_job(mission_dir: Path, profile: str, duration_hours: float):
    if profile not in {"quick", "overnight"}:
        raise ValueError("Trainingsprofil muss quick oder overnight sein")
    duration_hours = max(0.05, min(12.0, float(duration_hours)))
    current = training_job_status(mission_dir)
    if current and current.get("status") in {"queued", "running"}:
        raise ValueError("Für diese Mission läuft bereits ein CPU-Training")
    job_id = f"train-{uuid4().hex[:10]}"
    try:
        initial_run_id = json.loads(
            (mission_dir / "derived" / "path_model_current.json").read_text(encoding="utf-8")
        ).get("run_id")
    except (OSError, ValueError):
        initial_run_id = None
    log_dir = mission_dir / "derived" / "path_training_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{job_id}.out.log"
    stderr_path = log_dir / f"{job_id}.err.log"
    command = [
        sys.executable,
        "-m",
        "backend.app.path_model_worker",
        str(mission_dir),
        profile,
        str(duration_hours),
        job_id,
    ]
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[2],
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            creationflags=creationflags,
        )
    state = {
        "job_id": job_id,
        "status": "running",
        "profile": profile,
        "duration_hours": duration_hours,
        "pid": process.pid,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "candidates_completed": 0,
        "maximum_candidates": 1 if profile == "quick" else 128,
        "best_run_id": None,
        "initial_run_id": initial_run_id,
        "best_validation_score": None,
        "message": "CPU-Hintergrundtraining gestartet; der Browser kann geschlossen werden.",
        "stdout_log": str(stdout_path.relative_to(mission_dir)),
        "stderr_log": str(stderr_path.relative_to(mission_dir)),
    }
    write_job(mission_dir, state)
    return state
