import json
from types import SimpleNamespace

from backend.app.global_path_model import global_dataset_summary


def test_global_dashboard_counts_labels_and_refinements_across_missions(tmp_path):
    missions = [SimpleNamespace(id="mission-1", name="Mission 1"), SimpleNamespace(id="mission-2", name="Mission 2")]
    for mission in missions:
        ground_truth = tmp_path / mission.id / "ground_truth" / f"video-{mission.id[-1]}"
        ground_truth.mkdir(parents=True)
        (ground_truth / "000000001.json").write_text(
            json.dumps(
                {
                    "status": "confirmed",
                    "video_id": f"video-{mission.id[-1]}",
                    "frame_index": 1,
                    "polygons": [{"points": [[0, 0], [1, 0], [1, 1]]}],
                }
            ),
            encoding="utf-8",
        )
    refinement = tmp_path / "mission-2" / "path_refinements" / "video-2"
    refinement.mkdir(parents=True)
    (refinement / "000000001.json").write_text(json.dumps({"items": [{"id": "a"}, {"id": "b"}]}), encoding="utf-8")
    store = SimpleNamespace(root=tmp_path, list=lambda: missions)

    summary = global_dataset_summary(store)

    assert summary["totals"] == {
        "missions": 2,
        "confirmed_frames": 2,
        "videos": 2,
        "refinements": 2,
        "critical_flags": 0,
    }
