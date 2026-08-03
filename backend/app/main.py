import json
import logging
import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import ValidationError

from .annotations import delete_annotation, get_annotation, list_annotations, save_annotation
from .critical_flags import delete_critical_flag, list_critical_flags, save_critical_flag
from .global_path_model import (
    current_global_model,
    current_global_model_dir,
    global_dataset_summary,
    predict_global_path_frame,
    train_global_path_model,
)
from .global_video_analysis import (
    global_video_analysis_result,
    global_video_analysis_status,
    start_global_video_analysis,
)
from .labeling import list_labeling_videos
from .models import (
    CriticalFlagInput,
    GroundTruthAnnotationInput,
    MissionRecord,
    PathRefinementInput,
    SurveyPayload,
    TerrainTrainingInput,
    TerrainVideoPredictionInput,
    VideoMeta,
    VideoTerrainCategoryInput,
)
from .path_model import current_path_model_dir, predict_path_frame, save_path_refinement, train_path_model
from .path_training_jobs import start_training_job, training_job_status
from .processor import autonomous_loop, current_run_dir
from .reconstruction import current_reconstruction_dir, reconstruct
from .segmentation import current_segmentation_dir, process_segmentation
from .storage import MissionStore
from .terrain_model import (
    current_terrain_model,
    list_terrain_runs,
    predict_terrain_frame,
    predict_terrain_video,
    terrain_dataset_summary,
    terrain_prediction_run,
    train_terrain_model,
)

logging.basicConfig(level=os.getenv("ARIADNE_LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ariadne")
data_dir = Path(os.getenv("ARIADNE_DATA_DIR", Path(__file__).resolve().parents[2] / "data" / "missions"))
store = MissionStore(data_dir)
app = FastAPI(title="ARIADNE Survey API", version="1.0.0")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/v1/path-model/global/dashboard")
def global_model_dashboard():
    dataset = global_dataset_summary(store)
    try:
        model = current_global_model(store.root)
    except (OSError, ValueError, KeyError):
        model = None
    return {"dataset": dataset, "model": model}


@app.post("/api/v1/path-model/global/train")
def global_model_train():
    try:
        return train_global_path_model(store)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.get("/api/v1/path-model/global/evidence/{evidence_name}")
def global_model_evidence(evidence_name: str):
    if Path(evidence_name).name != evidence_name or Path(evidence_name).suffix.lower() not in {".jpg", ".jpeg"}:
        raise HTTPException(400, "Ungültiger Evidenzdateiname")
    try:
        path = current_global_model_dir(store.root) / "evidence" / evidence_name
    except (OSError, ValueError, KeyError):
        raise HTTPException(404, "Globale Modell-Evidenz nicht gefunden")
    if not path.is_file():
        raise HTTPException(404, "Globale Modell-Evidenz nicht gefunden")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/v1/path-model/global/predict/{mission_id}/{video_id}/{frame_index}")
def global_model_predict(mission_id: str, video_id: str, frame_index: int):
    try:
        return predict_global_path_frame(store, mission_id, video_id, frame_index)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(409, f"Globales Wegmodell nicht verfügbar: {exc}") from exc


@app.post("/api/v1/path-model/global/analyze-video/{mission_id}/{video_id}", status_code=202)
def global_video_analyze(mission_id: str, video_id: str):
    try:
        return start_global_video_analysis(store, mission_id, video_id)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(409, str(exc)) from exc


@app.get("/api/v1/path-model/global/analyze-video/{mission_id}/{video_id}/status")
def global_video_analyze_status(mission_id: str, video_id: str):
    try:
        state = global_video_analysis_status(store, mission_id, video_id)
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(409, str(exc)) from exc
    if not state:
        raise HTTPException(404, "Noch keine globale Videoanalyse gestartet")
    return state


@app.get("/api/v1/path-model/global/analyze-video/{mission_id}/{video_id}/result")
def global_video_analyze_result(mission_id: str, video_id: str):
    try:
        return global_video_analysis_result(store, mission_id, video_id)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(409, str(exc)) from exc


NO_TERRAIN_MODEL = "Es wurde noch kein Terrainmodell trainiert"


@app.get("/api/v1/terrain-model/dashboard")
def terrain_model_dashboard():
    try:
        model = current_terrain_model(store.root)
    except (OSError, ValueError, KeyError):
        model = None
    return {"dataset": terrain_dataset_summary(store), "model": model, "runs": list_terrain_runs(store.root)}


@app.post("/api/v1/terrain-model/train", status_code=201)
def terrain_model_train(payload: TerrainTrainingInput):
    try:
        return train_terrain_model(store, payload.frame_stride, payload.confidence_threshold)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.get("/api/v1/terrain-model/predict/{mission_id}/{video_id}/{frame_index}")
def terrain_model_predict_frame(mission_id: str, video_id: str, frame_index: int):
    try:
        return predict_terrain_frame(store, mission_id, video_id, frame_index)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(409, NO_TERRAIN_MODEL) from exc
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(409, str(exc)) from exc


@app.post("/api/v1/terrain-model/predict-video/{mission_id}/{video_id}", status_code=201)
def terrain_model_predict_video(mission_id: str, video_id: str, payload: TerrainVideoPredictionInput):
    try:
        return predict_terrain_video(store, mission_id, video_id, payload.frame_stride, payload.confidence_threshold)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(409, NO_TERRAIN_MODEL) from exc
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(409, str(exc)) from exc


@app.get("/api/v1/terrain-model/predictions/{run_id}")
def terrain_model_prediction_run(run_id: str):
    try:
        return terrain_prediction_run(store.root, run_id)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(409, str(exc)) from exc


@app.get("/api/v1/missions", response_model=list[MissionRecord])
def list_missions():
    return store.list()


@app.get("/api/v1/missions/{mission_id}", response_model=MissionRecord)
def get_mission(mission_id: str):
    record = store.get(mission_id)
    if not record:
        raise HTTPException(404, "Mission nicht gefunden")
    return record


@app.post("/api/v1/missions", response_model=MissionRecord, status_code=201)
async def create_mission(
    survey: str = Form(...), video_metadata: str = Form(...), videos: list[UploadFile] = File(...)
):
    try:
        payload = SurveyPayload.model_validate_json(survey)
        metadata = [VideoMeta.model_validate(item) for item in json.loads(video_metadata)]
    except (ValidationError, ValueError, TypeError) as exc:
        raise HTTPException(422, f"Ungültige Missionsdaten: {exc}") from exc
    if not 1 <= len(videos) <= 4:
        raise HTTPException(422, "Es sind ein bis vier Videos erforderlich")
    if len(metadata) != len(videos):
        raise HTTPException(422, "Metadaten fehlen für mindestens ein Video")
    allowed_extensions = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".mts", ".m2ts", ".3gp"}
    invalid = [
        v.filename or "Unbekannte Datei"
        for v in videos
        if not (v.content_type or "").startswith("video/")
        and Path(v.filename or "").suffix.lower() not in allowed_extensions
    ]
    if invalid:
        raise HTTPException(415, f"Nicht unterstütztes Videoformat: {', '.join(invalid)}")
    try:
        record = await store.create(payload, videos, metadata)
        log.info("mission_saved id=%s videos=%d", record.id, len(videos))
        return record
    except Exception:
        log.exception("mission_save_failed")
        raise HTTPException(500, "Mission konnte nicht sicher gespeichert werden")


@app.get("/api/v1/missions/{mission_id}/videos/{video_id}/content")
def video_content(mission_id: str, video_id: str):
    found = store.video_path(mission_id, video_id)
    if not found:
        raise HTTPException(404, "Video nicht gefunden")
    path, meta = found
    return FileResponse(path, media_type=meta.content_type, filename=meta.original_name)


@app.patch("/api/v1/missions/{mission_id}/videos/{video_id}")
def update_video_metadata(mission_id: str, video_id: str, payload: VideoTerrainCategoryInput):
    record = store.get(mission_id)
    if not record:
        raise HTTPException(404, "Mission nicht gefunden")
    current = next((item for item in record.videos if item.id == video_id), None)
    if not current:
        raise HTTPException(404, "Video nicht gefunden")
    updated_video = current.model_copy(update={"terrain_category": payload.terrain_category})
    updated_videos = [updated_video if item.id == video_id else item for item in record.videos]
    updated = record.model_copy(update={"videos": updated_videos})
    try:
        store.save(updated)
    except OSError as exc:
        raise HTTPException(409, f"Video-Metadaten konnten nicht gespeichert werden: {exc}") from exc
    return updated_video


@app.post("/api/v1/missions/{mission_id}/analysis", status_code=201)
def run_analysis(mission_id: str):
    record = store.get(mission_id)
    if not record:
        raise HTTPException(404, "Mission nicht gefunden")
    try:
        return autonomous_loop(record, store.root / mission_id)
    except Exception as exc:
        log.exception("analysis_failed mission=%s", mission_id)
        raise HTTPException(500, "Analyse fehlgeschlagen") from exc


@app.get("/api/v1/missions/{mission_id}/analysis")
def get_analysis(mission_id: str):
    try:
        path = current_run_dir(store.root / mission_id) / "analysis.json"
    except (OSError, ValueError, KeyError):
        raise HTTPException(404, "Noch keine Analyse vorhanden")
    if not path.is_file():
        raise HTTPException(404, "Noch keine Analyse vorhanden")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/v1/missions/{mission_id}/analysis/frames/{frame_name}")
def analysis_frame(mission_id: str, frame_name: str):
    if Path(frame_name).name != frame_name or not frame_name.endswith(".jpg"):
        raise HTTPException(400, "Ungültiger Frame")
    try:
        path = current_run_dir(store.root / mission_id) / "frames" / frame_name
    except (OSError, ValueError, KeyError):
        raise HTTPException(404, "Noch keine Analyse vorhanden")
    if not path.is_file():
        raise HTTPException(404, "Frame nicht gefunden")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/v1/missions/{mission_id}/analysis/report")
def analysis_report(mission_id: str):
    try:
        path = current_run_dir(store.root / mission_id) / "evaluation.md"
    except (OSError, ValueError, KeyError):
        raise HTTPException(404, "Bericht nicht gefunden")
    if not path.is_file():
        raise HTTPException(404, "Bericht nicht gefunden")
    return FileResponse(path, media_type="text/markdown", filename=f"ariadne-{mission_id}-evaluation.md")


@app.post("/api/v1/missions/{mission_id}/reconstruction", status_code=201)
def run_reconstruction(mission_id: str):
    record = store.get(mission_id)
    if not record:
        raise HTTPException(404, "Mission nicht gefunden")
    try:
        return reconstruct(record, store.root / mission_id)
    except Exception as exc:
        log.exception("reconstruction_failed mission=%s", mission_id)
        raise HTTPException(500, "Rekonstruktion fehlgeschlagen") from exc


@app.get("/api/v1/missions/{mission_id}/reconstruction")
def get_reconstruction(mission_id: str):
    try:
        path = current_reconstruction_dir(store.root / mission_id) / "reconstruction.json"
    except (OSError, ValueError, KeyError):
        raise HTTPException(404, "Noch keine Rekonstruktion vorhanden")
    if not path.is_file():
        raise HTTPException(404, "Noch keine Rekonstruktion vorhanden")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/v1/missions/{mission_id}/reconstruction/geojson")
def get_route_geojson(mission_id: str):
    try:
        path = current_reconstruction_dir(store.root / mission_id) / "route.geojson"
    except (OSError, ValueError, KeyError):
        raise HTTPException(404, "Noch keine Rekonstruktion vorhanden")
    if not path.is_file():
        raise HTTPException(404, "Noch keine Rekonstruktion vorhanden")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/v1/missions/{mission_id}/reconstruction/report")
def reconstruction_report(mission_id: str):
    try:
        path = current_reconstruction_dir(store.root / mission_id) / "evaluation.md"
    except (OSError, ValueError, KeyError):
        raise HTTPException(404, "Rekonstruktionsbericht nicht gefunden")
    if not path.is_file():
        raise HTTPException(404, "Rekonstruktionsbericht nicht gefunden")
    return FileResponse(path, media_type="text/markdown", filename=f"ariadne-{mission_id}-goal3-evaluation.md")


@app.post("/api/v1/missions/{mission_id}/segmentation", status_code=201)
def run_segmentation(mission_id: str):
    record = store.get(mission_id)
    if not record:
        raise HTTPException(404, "Mission nicht gefunden")
    try:
        return process_segmentation(record, store.root / mission_id)
    except Exception as exc:
        log.exception("segmentation_failed mission=%s", mission_id)
        raise HTTPException(500, "Segmentierung fehlgeschlagen") from exc


@app.get("/api/v1/missions/{mission_id}/segmentation")
def get_segmentation(mission_id: str):
    try:
        path = current_segmentation_dir(store.root / mission_id) / "segmentation.json"
    except (OSError, ValueError, KeyError):
        raise HTTPException(404, "Noch keine Segmentierung vorhanden")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/v1/missions/{mission_id}/segmentation/report")
def segmentation_report(mission_id: str):
    try:
        path = current_segmentation_dir(store.root / mission_id) / "evaluation.md"
    except (OSError, ValueError, KeyError):
        raise HTTPException(404, "Segmentierungsbericht nicht gefunden")
    if not path.is_file():
        raise HTTPException(404, "Segmentierungsbericht nicht gefunden")
    return FileResponse(path, media_type="text/markdown", filename=f"ariadne-{mission_id}-goal4-evaluation.md")


@app.get("/api/v1/missions/{mission_id}/segmentation/evidence/{evidence_name}")
def segmentation_evidence(mission_id: str, evidence_name: str):
    if Path(evidence_name).name != evidence_name or Path(evidence_name).suffix.lower() not in {".jpg", ".jpeg"}:
        raise HTTPException(400, "Ungültiger Evidenzdateiname")
    try:
        evidence_dir = (current_segmentation_dir(store.root / mission_id) / "evidence").resolve()
        path = (evidence_dir / evidence_name).resolve()
    except (OSError, ValueError, KeyError):
        raise HTTPException(404, "Terrain-Evidenz nicht gefunden")
    if path.parent != evidence_dir or not path.is_file():
        raise HTTPException(404, "Terrain-Evidenz nicht gefunden")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/v1/missions/{mission_id}/ground-truth")
def ground_truth_list(mission_id: str, video_id: str | None = None, include_geometry: bool = False):
    record = store.get(mission_id)
    if not record:
        raise HTTPException(404, "Mission nicht gefunden")
    try:
        return list_annotations(record, store.root / mission_id, video_id, include_geometry)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/api/v1/missions/{mission_id}/critical-flags")
def critical_flags_list(mission_id: str, video_id: str | None = None):
    record = store.get(mission_id)
    if not record:
        raise HTTPException(404, "Mission nicht gefunden")
    try:
        return list_critical_flags(record, store.root / mission_id, video_id)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.post("/api/v1/missions/{mission_id}/critical-flags/{video_id}/{frame_index}", status_code=201)
def critical_flags_put(mission_id: str, video_id: str, frame_index: int, payload: CriticalFlagInput):
    record = store.get(mission_id)
    if not record:
        raise HTTPException(404, "Mission nicht gefunden")
    try:
        return save_critical_flag(record, store.root / mission_id, video_id, frame_index, payload)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(409, str(exc)) from exc


@app.delete("/api/v1/missions/{mission_id}/critical-flags/{video_id}/{frame_index}", status_code=204)
def critical_flags_delete(mission_id: str, video_id: str, frame_index: int):
    record = store.get(mission_id)
    if not record:
        raise HTTPException(404, "Mission nicht gefunden")
    if not any(video.id == video_id for video in record.videos):
        raise HTTPException(404, "Video nicht gefunden")
    if not delete_critical_flag(store.root / mission_id, video_id, frame_index):
        raise HTTPException(404, "Fuer diesen Frame gibt es keine Meldung")


@app.get("/api/v1/missions/{mission_id}/labeling/videos")
def labeling_videos(mission_id: str):
    record = store.get(mission_id)
    if not record:
        raise HTTPException(404, "Mission nicht gefunden")
    try:
        return list_labeling_videos(record, store.root / mission_id)
    except (LookupError, ValueError) as exc:
        raise HTTPException(409, str(exc)) from exc


@app.get("/api/v1/missions/{mission_id}/ground-truth/{video_id}/{frame_index}")
def ground_truth_get(mission_id: str, video_id: str, frame_index: int):
    record = store.get(mission_id)
    if not record:
        raise HTTPException(404, "Mission nicht gefunden")
    if not any(video.id == video_id for video in record.videos):
        raise HTTPException(404, "Video nicht gefunden")
    annotation = get_annotation(store.root / mission_id, video_id, frame_index)
    if not annotation:
        raise HTTPException(404, "Für diesen Frame gibt es noch keine Ground Truth")
    return annotation


@app.put("/api/v1/missions/{mission_id}/ground-truth/{video_id}/{frame_index}")
def ground_truth_put(mission_id: str, video_id: str, frame_index: int, payload: GroundTruthAnnotationInput):
    record = store.get(mission_id)
    if not record:
        raise HTTPException(404, "Mission nicht gefunden")
    try:
        return save_annotation(record, store.root / mission_id, video_id, frame_index, payload)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.delete("/api/v1/missions/{mission_id}/ground-truth/{video_id}/{frame_index}", status_code=204)
def ground_truth_delete(mission_id: str, video_id: str, frame_index: int):
    record = store.get(mission_id)
    if not record:
        raise HTTPException(404, "Mission nicht gefunden")
    if not any(video.id == video_id for video in record.videos):
        raise HTTPException(404, "Video nicht gefunden")
    if not delete_annotation(store.root / mission_id, video_id, frame_index):
        raise HTTPException(404, "Für diesen Frame gibt es keine Ground Truth")


@app.post("/api/v1/missions/{mission_id}/path-model/train", status_code=201)
def path_model_train(mission_id: str):
    record = store.get(mission_id)
    if not record:
        raise HTTPException(404, "Mission nicht gefunden")
    try:
        return train_path_model(record, store.root / mission_id)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    except Exception as exc:
        log.exception("path_model_training_failed mission=%s", mission_id)
        raise HTTPException(500, "CPU-Wegmodell konnte nicht trainiert werden") from exc


@app.get("/api/v1/missions/{mission_id}/path-model")
def path_model_get(mission_id: str):
    if not store.get(mission_id):
        raise HTTPException(404, "Mission nicht gefunden")
    try:
        path = current_path_model_dir(store.root / mission_id) / "result.json"
    except (OSError, ValueError, KeyError):
        raise HTTPException(404, "Noch kein CPU-Wegmodell vorhanden")
    if not path.is_file():
        raise HTTPException(404, "Noch kein CPU-Wegmodell vorhanden")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/v1/missions/{mission_id}/path-model/evidence/{evidence_name}")
def path_model_evidence(mission_id: str, evidence_name: str):
    if Path(evidence_name).name != evidence_name or Path(evidence_name).suffix.lower() not in {".jpg", ".jpeg"}:
        raise HTTPException(400, "Ungültiger Evidenzdateiname")
    try:
        evidence_dir = (current_path_model_dir(store.root / mission_id) / "evidence").resolve()
        path = (evidence_dir / evidence_name).resolve()
    except (OSError, ValueError, KeyError):
        raise HTTPException(404, "Modell-Evidenz nicht gefunden")
    if path.parent != evidence_dir or not path.is_file():
        raise HTTPException(404, "Modell-Evidenz nicht gefunden")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/v1/missions/{mission_id}/path-model/predict/{video_id}/{frame_index}")
def path_model_predict(mission_id: str, video_id: str, frame_index: int):
    record = store.get(mission_id)
    if not record:
        raise HTTPException(404, "Mission nicht gefunden")
    try:
        return predict_path_frame(record, store.root / mission_id, video_id, frame_index)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(409, f"CPU-Wegmodell nicht verfügbar: {exc}") from exc


@app.post("/api/v1/missions/{mission_id}/path-model/refinements/{video_id}/{frame_index}", status_code=201)
def path_model_refinement(mission_id: str, video_id: str, frame_index: int, payload: PathRefinementInput):
    record = store.get(mission_id)
    if not record:
        raise HTTPException(404, "Mission nicht gefunden")
    try:
        return save_path_refinement(record, store.root / mission_id, video_id, frame_index, payload)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(409, str(exc)) from exc


@app.post("/api/v1/missions/{mission_id}/path-model/train-background", status_code=202)
def path_model_train_background(mission_id: str, profile: str = "overnight", duration_hours: float = 8):
    if not store.get(mission_id):
        raise HTTPException(404, "Mission nicht gefunden")
    try:
        return start_training_job(store.root / mission_id, profile, duration_hours)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.get("/api/v1/missions/{mission_id}/path-model/train-background")
def path_model_training_status(mission_id: str):
    if not store.get(mission_id):
        raise HTTPException(404, "Mission nicht gefunden")
    state = training_job_status(store.root / mission_id)
    if not state:
        raise HTTPException(404, "Noch kein Hintergrundtraining gestartet")
    return state
