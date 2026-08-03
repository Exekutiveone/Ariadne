"""Run-Registry: eine Wahrheitsquelle fuer die Kette Run -> Frames -> Masken -> Training.

Ein Run ist genau ein Originalvideo. Die Registry scannt den Videoordner bei
jedem Lesezugriff, legt neue Aufnahmen automatisch an und haelt pro Run drei
Dinge fest, die nirgendwo sonst stehen: Bearbeitungsstatus, vorherrschender
Untergrund und eine Freitextnotiz.

Aufteilung der Verantwortung, damit es genau eine Wahrheit gibt:

- Die Terrainkategorie bleibt in `mission.json` zuhause, weil sie dort schon
  vom Labeler und vom Terrainmodell gelesen wird. Die Registry schreibt sie
  durch (`MissionStore.save`) und spiegelt sie bei jedem Scan zurueck — sie
  haelt also nie einen abweichenden zweiten Stand.
- Status und Notiz gibt es nur hier. Sie sind Handarbeit und nicht
  reproduzierbar; die Datenbankdatei gehoert deshalb in eine Sicherung.

Die SQLite-Datei liegt neben den Missionen unter `data/registry.sqlite` und
wird bewusst nur fuer die Dauer eines Aufrufs geoeffnet: das Repo liegt in
OneDrive, und eine dauerhaft offene Datenbank mit Journal waere genau die
Sync-Sperre, die laufende Zugriffe abbricht.
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

RUN_REGISTRY_SCHEMA_VERSION = "1.0"

RUN_STATUSES = ("unlabeled", "queued_for_labeling", "labeled", "training_ready")
RUN_STATUS_LABELS = {
    "unlabeled": "Ungelabelt",
    "queued_for_labeling": "Zum Labeln vorgemerkt",
    "labeled": "Gelabelt",
    "training_ready": "Trainingsbereit",
}
# Reihenfolge des Arbeitsablaufs, rein zur Anzeige.
RUN_STATUS_ORDER = {status: index for index, status in enumerate(RUN_STATUSES)}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id           TEXT PRIMARY KEY,
    mission_id       TEXT NOT NULL,
    video_id         TEXT NOT NULL,
    mission_name     TEXT NOT NULL,
    original_name    TEXT NOT NULL,
    source_path      TEXT NOT NULL,
    size_bytes       INTEGER NOT NULL,
    status           TEXT NOT NULL,
    terrain_category TEXT,
    note             TEXT NOT NULL DEFAULT '',
    discovered_at    TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS runs_by_mission ON runs (mission_id);
"""


def registry_path(missions_root: Path) -> Path:
    return missions_root.parent / "registry.sqlite"


def _now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _connect(missions_root: Path):
    """Verbindung fuer die Dauer genau eines Aufrufs.

    Bewusst ein eigener Kontextmanager: `with sqlite3.connect(...)` committet
    zwar, schliesst die Verbindung aber **nicht**. Unter OneDrive bliebe die
    Datei damit dauerhaft offen — genau die Sync-Sperre, die vermieden werden soll.
    """
    path = registry_path(missions_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10)
    try:
        connection.row_factory = sqlite3.Row
        # Kein WAL: die zusaetzlichen -wal/-shm-Dateien wuerden ebenfalls offen
        # bleiben und den Sync blockieren.
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.executescript(_SCHEMA)
        yield connection
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def _video_file(missions_root: Path, mission_id: str, video_id: str):
    matches = sorted((missions_root / mission_id / "videos").glob(f"{video_id}.*"))
    return matches[0] if matches else None


def _row(record: sqlite3.Row):
    status = record["status"]
    return {
        "run_id": record["run_id"],
        "mission_id": record["mission_id"],
        "video_id": record["video_id"],
        "mission_name": record["mission_name"],
        "original_name": record["original_name"],
        "video_available": bool(record["source_path"]),
        "size_bytes": record["size_bytes"],
        "status": status,
        "status_label": RUN_STATUS_LABELS.get(status, status),
        "terrain_category": record["terrain_category"],
        "note": record["note"],
        "discovered_at": record["discovered_at"],
        "updated_at": record["updated_at"],
    }


def scan_runs(store):
    """Legt neue Aufnahmen an und frischt die abgeleiteten Felder auf.

    Handarbeit wird dabei nie ueberschrieben: Status und Notiz bleiben stehen,
    die Terrainkategorie wird aus `mission.json` gespiegelt, weil sie dort ihre
    Heimat hat.
    """
    now = _now()
    seen, added = set(), 0
    with _connect(store.root) as connection:
        known = {record["run_id"] for record in connection.execute("SELECT run_id FROM runs")}
        for mission in store.list():
            for video in mission.videos:
                run_id = f"{mission.id}/{video.id}"
                seen.add(run_id)
                path = _video_file(store.root, mission.id, video.id)
                size = path.stat().st_size if path and path.is_file() else 0
                if run_id in known:
                    connection.execute(
                        "UPDATE runs SET mission_name=?, original_name=?, source_path=?, size_bytes=?, "
                        "terrain_category=? WHERE run_id=?",
                        (mission.name, video.original_name, str(path or ""), size, video.terrain_category, run_id),
                    )
                    continue
                connection.execute(
                    "INSERT INTO runs (run_id, mission_id, video_id, mission_name, original_name, source_path, "
                    "size_bytes, status, terrain_category, note, discovered_at, updated_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        run_id,
                        mission.id,
                        video.id,
                        mission.name,
                        video.original_name,
                        str(path or ""),
                        size,
                        "unlabeled",
                        video.terrain_category,
                        "",
                        now,
                        now,
                    ),
                )
                added += 1
        removed = sorted(known - seen)
        for run_id in removed:
            connection.execute("DELETE FROM runs WHERE run_id=?", (run_id,))
    return {"added": added, "removed": removed, "total": len(seen)}


def list_runs(store, *, rescan: bool = True):
    scan = scan_runs(store) if rescan else {"added": 0, "removed": [], "total": None}
    with _connect(store.root) as connection:
        records = [_row(record) for record in connection.execute("SELECT * FROM runs")]
    records.sort(
        key=lambda item: (RUN_STATUS_ORDER.get(item["status"], 99), item["mission_name"], item["original_name"])
    )
    counts = {status: sum(item["status"] == status for item in records) for status in RUN_STATUSES}
    categories: dict[str, int] = {}
    for item in records:
        if item["terrain_category"]:
            categories[item["terrain_category"]] = categories.get(item["terrain_category"], 0) + 1
    return {
        "schema_version": RUN_REGISTRY_SCHEMA_VERSION,
        "database": str(registry_path(store.root)),
        "scan": scan,
        "statuses": [{"value": status, "label": RUN_STATUS_LABELS[status]} for status in RUN_STATUSES],
        "counts": counts,
        "totals": {
            "runs": len(records),
            "with_terrain_category": sum(bool(item["terrain_category"]) for item in records),
            "missing_video_file": sum(not item["video_available"] for item in records),
        },
        "terrain_categories": [{"terrain_category": name, "runs": categories[name]} for name in sorted(categories)],
        "runs": records,
        "note": (
            "Status und Notiz sind Handarbeit und stehen nur in dieser Datenbank. "
            "Die Terrainkategorie wird nach mission.json durchgeschrieben."
        ),
    }


def get_run(store, run_id: str):
    with _connect(store.root) as connection:
        record = connection.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
    if record is None:
        raise LookupError("Run nicht gefunden")
    return _row(record)


def update_run(store, run_id: str, *, status=None, terrain_category=..., note=None):
    """Aendert Status, Untergrund oder Notiz eines Runs.

    `terrain_category` benutzt Ellipsis als Vorgabe, damit "nicht mitgeschickt"
    von "ausdruecklich auf null gesetzt" unterscheidbar bleibt.
    """
    existing = get_run(store, run_id)
    if status is not None and status not in RUN_STATUSES:
        raise ValueError(f"Unbekannter Status: {status}")

    if terrain_category is not Ellipsis:
        mission = store.get(existing["mission_id"])
        if not mission:
            raise LookupError("Mission nicht gefunden")
        video = next((item for item in mission.videos if item.id == existing["video_id"]), None)
        if not video:
            raise LookupError("Video nicht gefunden")
        # Durchschreiben statt zweiter Wahrheit: mission.json bleibt die Heimat
        # der Terrainkategorie, damit Labeler und Terrainmodell sie dort finden.
        updated_videos = [
            item.model_copy(update={"terrain_category": terrain_category}) if item.id == video.id else item
            for item in mission.videos
        ]
        store.save(mission.model_copy(update={"videos": updated_videos}))

    with _connect(store.root) as connection:
        connection.execute(
            "UPDATE runs SET status=COALESCE(?, status), note=COALESCE(?, note), "
            "terrain_category=CASE WHEN ?=1 THEN ? ELSE terrain_category END, updated_at=? WHERE run_id=?",
            (
                status,
                note,
                1 if terrain_category is not Ellipsis else 0,
                None if terrain_category is Ellipsis else terrain_category,
                _now(),
                run_id,
            ),
        )
    return get_run(store, run_id)
