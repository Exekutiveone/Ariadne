from types import SimpleNamespace

from backend.app.path_training_jobs import read_job, start_training_job


def test_background_training_is_launched_as_persistent_worker(tmp_path, monkeypatch):
    launched = {}

    def fake_popen(command, **kwargs):
        launched["command"] = command
        launched["kwargs"] = kwargs
        return SimpleNamespace(pid=43210)

    monkeypatch.setattr("backend.app.path_training_jobs.subprocess.Popen", fake_popen)

    state = start_training_job(tmp_path, "overnight", 8)
    persisted = read_job(tmp_path)

    assert state == persisted
    assert state["status"] == "running"
    assert state["profile"] == "overnight"
    assert state["duration_hours"] == 8
    assert state["maximum_candidates"] == 128
    assert state["pid"] == 43210
    assert launched["command"][1:3] == ["-m", "backend.app.path_model_worker"]
    assert launched["kwargs"]["stdin"] is not None
