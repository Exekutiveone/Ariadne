from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .label_ontology import ALL_CLASSES, MASK_VALUES, layer_of


class Coordinate(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lng: float = Field(ge=-180, le=180)


class Pause(BaseModel):
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(ge=0)
    note: str = ""

    @model_validator(mode="after")
    def chronological(self):
        if self.end_seconds < self.start_seconds:
            raise ValueError("Pausenende muss nach dem Pausenbeginn liegen")
        return self


class SurveyPayload(BaseModel):
    name: str = Field(min_length=3, max_length=120)
    start: Coordinate
    end: Coordinate
    route: list[Coordinate] = Field(min_length=2)
    movement_start: str | None = None
    movement_end: str | None = None
    pauses: list[Pause] = []
    notes: str = Field(default="", max_length=5000)


class VideoMeta(BaseModel):
    direction: Literal["A_TO_B", "B_TO_A"]
    orientation: Literal["PORTRAIT", "LANDSCAPE"]
    terrain_category: str | None = Field(default=None, min_length=1, max_length=120)
    # Das ganze Video zeigt nur nicht befahrbaren Grund (z. B. reine Graben-
    # oder Dickicht-Aufnahme). Anders als terrain_category (WAS fuer ein
    # Untergrund) ist das eine Aussage ueber Befahrbarkeit: sie geht direkt als
    # synthetische Vollnegativ-Frames ins Wegtraining, ohne dass jeder Frame von
    # Hand gelabelt werden muss.
    fully_not_traversable: bool = False

    @model_validator(mode="after")
    def normalize_terrain_category(self):
        if self.terrain_category is not None:
            value = self.terrain_category.strip()
            self.terrain_category = value or None
        return self


class VideoTerrainCategoryInput(BaseModel):
    """Teilaenderung der Videometadaten.

    Nur mitgeschickte Felder werden angefasst (ueber `model_fields_set` in
    main.py) — sonst wuerde eine reine Terrainkategorie-Aenderung
    `fully_not_traversable` unbeabsichtigt zuruecksetzen.
    """

    terrain_category: str | None = Field(default=None, max_length=120)
    fully_not_traversable: bool | None = None

    @model_validator(mode="after")
    def normalize_terrain_category(self):
        if self.terrain_category is not None:
            value = self.terrain_category.strip()
            self.terrain_category = value or None
        return self


class StoredVideo(VideoMeta):
    id: str
    original_name: str
    content_type: str
    size_bytes: int
    sha256: str


class MissionRecord(SurveyPayload):
    id: str
    status: Literal["READY_FOR_GOAL_2"]
    created_at: datetime
    videos: list[StoredVideo]
    schema_version: Literal["1.0"] = "1.0"


class GroundTruthMask(BaseModel):
    width: int = Field(ge=8, le=1024)
    height: int = Field(ge=8, le=2048)
    rle: list[int] = Field(min_length=2, max_length=1_000_000)

    @model_validator(mode="after")
    def valid_rle(self):
        if len(self.rle) % 2:
            raise ValueError("RLE muss aus Wert-/Längenpaaren bestehen")
        total = 0
        for index in range(0, len(self.rle), 2):
            value, length = self.rle[index], self.rle[index + 1]
            if value not in MASK_VALUES:
                raise ValueError(f"Ground-Truth-Wert muss einer von {sorted(MASK_VALUES)} sein")
            if length <= 0:
                raise ValueError("RLE-Längen müssen positiv sein")
            total += length
        if total != self.width * self.height:
            raise ValueError("RLE-Länge passt nicht zur Maskengröße")
        return self


class GroundTruthPolygon(BaseModel):
    """Eine markierte Fläche mit genau einer Klasse aus `label_ontology`.

    Alle Felder ausser `points` haben Vorgaben, die exakt den Zustand vor dem
    04.08.2026 beschreiben: eine befahrbare, von Hand gesetzte, sichere Fläche.
    Die 276 gespeicherten Ground-Truth-Dateien bleiben damit ohne Migration
    gültig und bedeuten weiterhin dasselbe.
    """

    id: str = Field(default="path-1", min_length=1, max_length=80, pattern=r"^[a-zA-Z0-9_-]+$")
    class_id: str = "traversable"
    points: list[tuple[float, float]] = Field(min_length=3, max_length=500)
    certainty: Literal["certain", "uncertain", "partially_occluded"] = "certain"
    origin: Literal["manual", "model_proposal", "manual_corrected", "human_confirmed"] = "manual"
    # Flächen, die aussehen wie das Gegenteil dessen, was sie sind: Schatten,
    # der wie ein Hindernis wirkt, Lichtfleck, der wie freie Fläche aussieht.
    # Genau die verbessern ein Modell am stärksten, deshalb sind sie auffindbar.
    hard_negative: bool = False
    note: str = Field(default="", max_length=500)
    # Zeitliche Verkettung: dieselbe Stelle ueber mehrere Frames hinweg.
    # Bewusst ein Feld am Polygon und kein eigenes Journal — die Angabe reist mit
    # dem Label mit, und das Training liest die Polygone ohnehin. Ein zweites
    # Verzeichnis waere eine zweite Wahrheit ueber dieselbe Sache.
    tracking_id: str | None = Field(default=None, min_length=1, max_length=80, pattern=r"^[a-zA-Z0-9_-]+$")
    carried_from_frame: int | None = Field(default=None, ge=0)
    # Geloeschte Polygone werden NICHT gespeichert: verschwindet eine
    # tracking_id im Folgeframe, ist das die Loeschung. Sie ist damit ableitbar
    # statt doppelt gefuehrt.
    edit: Literal["new", "carried_unchanged", "carried_adjusted", "corrected"] = "new"

    @model_validator(mode="after")
    def normalized_and_distinct(self):
        if self.class_id not in ALL_CLASSES:
            raise ValueError(f"Unbekannte Labelklasse: {self.class_id}")
        if any(not (0 <= x <= 1 and 0 <= y <= 1) for x, y in self.points):
            raise ValueError("Polygonpunkte müssen auf das Videobild normiert sein")
        if len({(round(x, 8), round(y, 8)) for x, y in self.points}) < 3:
            raise ValueError("Ein Polygon benötigt mindestens drei unterschiedliche Punkte")
        return self


class GroundTruthAnnotationInput(BaseModel):
    """Alle Labels eines Frames.

    `polygons` traegt Kernklassen, Hindernisse und Problemzonen; `roi` haelt den
    Auswertungsbereich getrennt davon. Die Trennung ist Absicht: der ROI
    schneidet nichts weg, er sagt dem Training nur, welche Pixel zaehlen. Das
    Originalbild bleibt unangetastet und die Regel spaeter aenderbar.
    """

    timestamp_ms: int = Field(ge=0)
    source_frame_hash: str | None = Field(default=None, min_length=12, max_length=128, pattern=r"^[a-fA-F0-9]+$")
    mask: GroundTruthMask | None = None
    polygons: list[GroundTruthPolygon] = Field(default_factory=list, max_length=120)
    roi: list[GroundTruthPolygon] = Field(default_factory=list, max_length=20)
    status: Literal["draft", "confirmed", "skipped"] = "draft"
    annotator: str = Field(default="human", min_length=1, max_length=80)
    notes: str = Field(default="", max_length=1000)
    # Aufloesung des Originalframes. Die Punkte sind normiert, aber ohne die
    # Quellgroesse laesst sich spaeter nicht mehr sagen, wie fein ein Label war.
    frame_width: int | None = Field(default=None, ge=1, le=16384)
    frame_height: int | None = Field(default=None, ge=1, le=16384)
    # Linear (der Reihe nach durchs Video) oder Shuffle (zufaellige Frames aus
    # mehreren Videos gemischt) — welcher Arbeitsmodus aktiv war, als dieser
    # Frame gelabelt wurde. Reine Metadaten, veraendert die Maske nicht.
    label_mode: Literal["linear", "shuffle"] = "linear"

    @model_validator(mode="after")
    def roi_holds_only_roi_classes(self):
        wrong = [item.class_id for item in self.roi if layer_of(item.class_id) != "roi"]
        if wrong:
            raise ValueError(
                f"Im Auswertungsbereich sind nur ROI-Klassen erlaubt, nicht: {', '.join(sorted(set(wrong)))}"
            )
        misplaced = [item.class_id for item in self.polygons if layer_of(item.class_id) == "roi"]
        if misplaced:
            raise ValueError("ROI-Klassen gehören in das Feld 'roi', nicht zu den Polygonen")
        return self


class RoiProfileInput(BaseModel):
    """Auswertungsbereich, der fuer ein ganzes Video gilt.

    Die Baender sind Anteile der Bildhoehe, nicht Pixel — die Aufloesung darf
    sich aendern, ohne das Profil zu entwerten.
    """

    top_ignore_fraction: float | None = Field(default=None, gt=0, lt=1)
    bottom_ignore_fraction: float | None = Field(default=None, gt=0, lt=1)
    roi: list[GroundTruthPolygon] = Field(default_factory=list, max_length=20)
    note: str = Field(default="", max_length=1000)
    annotator: str = Field(default="human", min_length=1, max_length=80)

    @model_validator(mode="after")
    def only_roi_classes_and_room_between_the_bands(self):
        wrong = [item.class_id for item in self.roi if layer_of(item.class_id) != "roi"]
        if wrong:
            raise ValueError(f"Im Profil sind nur ROI-Klassen erlaubt, nicht: {', '.join(sorted(set(wrong)))}")
        top = self.top_ignore_fraction or 0
        bottom = self.bottom_ignore_fraction or 0
        if top + bottom >= 1:
            raise ValueError("Oberes und unteres Band würden das ganze Bild ausschließen")
        return self


class OffPathIntervalInput(BaseModel):
    """Zeitspanne, waehrend des Anschauens markiert: in diesem Bereich zeigt
    KEIN einziger Frame einen befahrbaren Bereich — das Fahrzeug ist vom Weg
    abgekommen (Graben, dichtes Gebuesch, falsche Richtung ...).

    Anders als eine normale Ground Truth ist das nicht "kein Weg markiert",
    sondern ausdruecklich "es gibt hier keinen Weg". Das Training bestraft das
    Modell dafuer, hier trotzdem Wegflaeche zu erfinden (invented_path), und
    das Modell soll fuer Frames in diesem Bereich genau eine Aussage liefern:
    vom Weg abgekommen.
    """

    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    note: str = Field(default="", max_length=500)
    annotator: str = Field(default="human", min_length=1, max_length=80)

    @model_validator(mode="after")
    def chronological_and_not_too_short(self):
        if self.end_ms <= self.start_ms:
            raise ValueError("Das Intervallende muss nach dem Anfang liegen")
        if self.end_ms - self.start_ms < 200:
            raise ValueError("Ein Intervall muss mindestens 200 ms lang sein")
        return self


class PathRefinementInput(BaseModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    expected_kind: Literal["missed_label", "invented_path"]
    action: Literal["accept_model"] = "accept_model"


class TrajectoryInput(BaseModel):
    """Von Hand geplante Trajektorie fuer genau einen Frame.

    `origin` haelt fest, wie der Verlauf zustande kam: unveraendert vom Modell
    uebernommen, daraus nachgebessert oder komplett selbst gesetzt. Ohne diese
    Unterscheidung liesse sich spaeter nicht mehr sagen, was Handarbeit ist.
    """

    timestamp_ms: int = Field(ge=0)
    points: list[tuple[float, float]] = Field(min_length=2, max_length=400)
    corridor: Literal["mitte", "rechts", "links"] | None = None
    origin: Literal["model_proposal", "manual_edit", "manual"] = "manual_edit"
    note: str = Field(default="", max_length=1000)
    annotator: str = Field(default="human", min_length=1, max_length=80)

    @model_validator(mode="after")
    def normalized_and_descending(self):
        if any(not (0 <= x <= 1 and 0 <= y <= 1) for x, y in self.points):
            raise ValueError("Trajektorienpunkte müssen auf das Videobild normiert sein")
        if len({(round(x, 8), round(y, 8)) for x, y in self.points}) < 2:
            raise ValueError("Eine Trajektorie benötigt mindestens zwei unterschiedliche Punkte")
        return self


class RunRegistryUpdateInput(BaseModel):
    """Teilaenderung eines Runs.

    Alle Felder sind optional; nur mitgeschickte werden angefasst. Fuer
    `terrain_category` ist deshalb der Unterschied zwischen "nicht mitgeschickt"
    und "ausdruecklich auf null gesetzt" wichtig — er wird ueber
    `model_fields_set` ausgewertet, nicht ueber den Wert.
    """

    status: Literal["unlabeled", "queued_for_labeling", "labeled", "training_ready"] | None = None
    terrain_category: str | None = Field(default=None, max_length=120)
    note: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def normalize_terrain_category(self):
        if self.terrain_category is not None:
            value = self.terrain_category.strip()
            self.terrain_category = value or None
        return self


class TerrainTrainingInput(BaseModel):
    """Parameter eines Terrain-Trainingslaufs.

    Die Vorgabewerte spiegeln DEFAULT_FRAME_STRIDE und
    DEFAULT_CONFIDENCE_THRESHOLD aus `terrain_model`; dort liegt die fachliche
    Begründung. Hier stehen sie als Literale, weil `terrain_model` dieses Modul
    importiert und ein Rückimport zirkulär wäre.
    """

    frame_stride: int = Field(default=15, ge=1, le=600)
    confidence_threshold: float = Field(default=0.6, ge=0.05, le=0.99)


class TerrainVideoPredictionInput(BaseModel):
    frame_stride: int = Field(default=15, ge=1, le=600)
    confidence_threshold: float | None = Field(default=None, ge=0.05, le=0.99)


class CriticalFlagInput(BaseModel):
    """Meldung eines grob falschen Frames: Aufnahme abseits des befahrbaren
    Bereichs, in der trotzdem Weg erkannt wird."""

    severity: int = Field(default=3, ge=1, le=5)
    brush_mask: GroundTruthMask | None = None
    note: str = Field(default="", max_length=1000)
    annotator: str = Field(default="human", min_length=1, max_length=80)
