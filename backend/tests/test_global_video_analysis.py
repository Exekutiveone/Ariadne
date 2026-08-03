import json
from types import SimpleNamespace

from backend.app.global_video_analysis import (
    _analysis_dir,
    _portable_analysis_dir,
    _write_json,
    _write_status,
    global_video_analysis_status,
)


def test_dead_global_video_worker_is_reported_as_interrupted(tmp_path, monkeypatch):
    monkeypatch.setenv("ARIADNE_RUNTIME_DIR", str(tmp_path / "runtime"))
    missions_root = tmp_path / "missions"
    model_root = tmp_path / "global_models" / "path_model"
    run = model_root / "runs" / "global-run-1"
    run.mkdir(parents=True)
    model_root.mkdir(parents=True, exist_ok=True)
    (model_root / "current.json").write_text(json.dumps({"run_id": "global-run-1"}), encoding="utf-8")
    directory = _analysis_dir(missions_root, "global-run-1", "mission-1", "video-1")
    directory.mkdir(parents=True)
    (directory / "status.json").write_text(json.dumps({"status": "running", "pid": 2147483000}), encoding="utf-8")
    store = SimpleNamespace(root=missions_root)

    status = global_video_analysis_status(store, "mission-1", "video-1")

    assert status["status"] == "interrupted"
    assert status["finished_at"]


def test_atomic_json_write_retries_transient_sync_provider_lock(tmp_path, monkeypatch):
    real_replace = __import__("os").replace
    attempts = {"count": 0}

    def flaky_replace(source, target):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise PermissionError(5, "temporarily locked")
        return real_replace(source, target)

    monkeypatch.setattr("backend.app.global_video_analysis.os.replace", flaky_replace)
    monkeypatch.setattr("backend.app.global_video_analysis.time.sleep", lambda _: None)
    path = tmp_path / "status.json"

    _write_json(path, {"status": "running"})

    assert attempts["count"] == 3
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "running"


def test_status_write_failure_does_not_abort_inference(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "backend.app.global_video_analysis.os.replace", lambda *_: (_ for _ in ()).throw(PermissionError(5, "locked"))
    )
    monkeypatch.setattr("backend.app.global_video_analysis.time.sleep", lambda _: None)

    assert _write_status(tmp_path / "status.json", {"status": "running"}) is False


def test_completed_portable_cache_is_available_after_clone(tmp_path, monkeypatch):
    monkeypatch.setenv("ARIADNE_RUNTIME_DIR", str(tmp_path / "empty-runtime"))
    missions_root = tmp_path / "data" / "missions"
    model_root = tmp_path / "data" / "global_models" / "path_model"
    (model_root / "runs" / "global-run-1").mkdir(parents=True)
    (model_root / "current.json").write_text(json.dumps({"run_id": "global-run-1"}), encoding="utf-8")
    portable = _portable_analysis_dir(missions_root, "global-run-1", "mission-1", "video-1")
    portable.mkdir(parents=True)
    (portable / "status.json").write_text(json.dumps({"status": "completed", "portable_cache": True}), encoding="utf-8")
    store = SimpleNamespace(root=missions_root)

    status = global_video_analysis_status(store, "mission-1", "video-1")

    assert status == {"status": "completed", "portable_cache": True}
