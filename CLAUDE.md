# ARIADNE — Projektkontext

Lokale Erfassung und Auswertung manueller Waldbegehungen. Missionen, Routen und
Originalvideos werden persistent gespeichert; daraus werden Vegetationsinstanzen,
Boden- und Befahrbarkeitsmasken sowie ein zeitlich stabilisierter ARGUS-Korridor
berechnet. Alles laeuft lokal auf CPU, ohne Cloud und ohne GPU.

Fachliche Ausgabesprache ist Deutsch — Fehlermeldungen, Labels und Berichte
werden dem Nutzer auf Deutsch angezeigt. Bezeichner im Code sind Englisch.

## Aufbau

- `backend/app/main.py` — nur FastAPI-Routing und HTTP-Fehlerabbildung, keine Fachlogik.
- `backend/app/storage.py` — `MissionStore`, Ablage unter `data/missions/<mission_id>`.
- `backend/app/segmentation.py`, `terrain.py` — Vegetationsinstanzen, Boden- und
  Befahrbarkeitsmasken, ARGUS-Korridor.
- `backend/app/path_model.py` — missionsspezifisches CPU-Wegmodell
  (`ariadne-cpu-path-rff`): Random-Feature-Ridge-Klassifikator ueber 22 Pixelmerkmalen.
- `backend/app/global_path_model.py`, `global_video_analysis.py` — missionsuebergreifendes
  Modell und persistente Vollvideo-Inferenz mit wiederaufnehmbaren Checkpoints.
- `backend/app/terrain_model.py` — videobasierte Terrainklassifizierung
  (`ariadne-cpu-terrain-rff`): ein Merkmalsvektor je Frame, Multiklassen-Ridge
  ueber Random-Feature-Projektionen, Laeufe unter `data/global_models/terrain_model`.
- `backend/app/corridor.py` — Fluchtpunkt-Geometrie und Korridor-Bewertung im
  Bildraum. Rein deterministisch, kein ML und keine Abhaengigkeit zu einem Modell;
  die Verdrahtung mit der Wegmaske steht in `global_path_model.py`.
- `backend/app/annotations.py`, `labeling.py` — manuelle Ground-Truth-Polygone.
- `src/` — React-Frontend. `AnalysisView.tsx` (Player mit Overlays) und
  `GroundTruthLabeler.tsx` (Polygonwerkzeug) sind die beiden grossen Komponenten.
  `TerrainModelPanel.tsx` haengt als eigener Abschnitt im `GlobalModelDashboard`.

## Befehle

```powershell
python -m pip install -r backend/requirements-dev.txt
npm install
npm run dev            # Backend auf 8000, Vite auf 5173, /api wird geproxyt
python -m pytest backend/tests -q
npm test
npx tsc -b
python -m ruff check . ; python -m ruff format .
```

Pytest **muss** als `python -m pytest` laufen, damit das Wurzelverzeichnis auf
`sys.path` liegt und `from backend.app...` in den Tests aufloest.

## Daten: was versioniert ist und was nicht

| Pfad | Groesse | In Git? |
|---|---|---|
| `data/missions/*/ground_truth`, `path_refinements` | 0,7 MB | **ja** — Handarbeit, nicht reproduzierbar |
| `data/missions/*/videos` | 3,1 GB | nein — bleiben dauerhaft lokal |
| `data/missions/*/derived` | 470 MB | nein — bei jedem Lauf neu berechnet |
| `data/runtime_cache`, `data/Mission1`, `data/Mission 2` | 3,1 GB | nein — Cache bzw. Duplikate |

**Entschieden (03.08.2026): Videos und Missionsdaten bleiben lokal.** Das Repo
ist ausschliesslich Dokumentation und Versionierung des Codes; die Anwendung
laeuft auf dem Rechner des Nutzers gegen die lokalen Daten. Es werden keine
Videos hochgeladen, auch nicht ueber LFS. Die LFS-Regeln in `.gitattributes`
bleiben als Absicherung stehen, greifen aber nicht, weil die Videopfade
ignoriert sind.

Wer das Repo klont, bekommt Code, Ground Truth und Refinements, aber keine
Videos — Missionen muessen lokal neu angelegt oder die Videodateien manuell
nach `data/missions/<mission_id>/videos/` kopiert werden.

## Konventionen und Fallstricke

- Formatierung und Linting ausschliesslich ueber `ruff` (Konfiguration in `pyproject.toml`,
  Zeilenlaenge 120). Backend-Code war frueher in zwei unterschiedlichen Stilen geschrieben;
  das ist vereinheitlicht und soll so bleiben.
- HTTP 500 gibt **keine** Exception-Details an den Client zurueck — Details gehoeren ins
  Log via `log.exception(...)`. HTTP 409/404 tragen bewusst fachliche deutsche Meldungen.
- Laufender Zustand (Checkpoints, Jobstatus) liegt unter `%LOCALAPPDATA%\Ariadne\runtime`,
  nicht im Projektordner — das Repo liegt in OneDrive und Sync-Sperren wuerden laufende
  Jobs abbrechen. Ueber `ARIADNE_RUNTIME_DIR` aenderbar, `ARIADNE_DATA_DIR` fuer die Daten.
- Entwickelt wird auf Python 3.14, die CI prueft gegen 3.12. Auf 3.14 werden
  Annotationen verzoegert ausgewertet (PEP 649), auf 3.12 nicht — Fehler in
  Annotationen fallen deshalb lokal nicht zwangslaeufig auf. `MissionStore.list`
  ueberschattet den Builtin `list`; `storage.py` braucht darum
  `from __future__ import annotations`.
- Ground-Truth-Polygonpunkte sind auf das Originalbild normiert, nie in Pixeln.
  Aenderungen an einem Frame duerfen nie einen anderen Frame veraendern.
- Aendert sich die Merkmalszahl in `_features`, werden alle gespeicherten Modelle
  ungueltig. `MODEL_SCHEMA_VERSION` anheben; `test_path_model_core.py` pinnt die Zahl.
- Train/Validation werden nach Frames getrennt und die Schwelle nur auf
  Validierungsframes gewaehlt. Diese Trennung nicht aufweichen.
- Beim Terrainmodell laeuft die Trennung eine Ebene hoeher: Train, Validierung
  und Test werden nach `video_id` gruppiert, nie nach einzelnen Frames. Frames
  desselben Videos sind fast identisch — ein Frame-Split waere ein Datenleck.
  Ebenso liegt die Terrainklasse nur am Video (`StoredVideo.terrain_category`);
  Frames erben sie und speichern nie ein eigenes Terrainlabel, damit ein Umlabeln
  des Videos sofort fuer alle seine Frames gilt.
- `TERRAIN_FEATURE_COUNT` ist in `test_terrain_model.py` gepinnt; aendert sich der
  Deskriptor, muss `TERRAIN_MODEL_SCHEMA_VERSION` steigen.
- Terrainlaeufe sind unveraenderlich: jeder Trainings- und jeder Videovorhersagelauf
  bekommt ein eigenes Verzeichnis, nur `current.json` zeigt um.
- Die Korridorpruefung bewertet **nur die Breite**. Wie weit voraus ein Korridor
  frei bleibt, wird bewusst nicht gemessen und darf auch nicht als Distanz
  berichtet werden. Ein Korridor wird nie still "frei": Rauschen und Zeilen
  ausserhalb des Bildes fuehren zu "unsicher", nicht zu "frei".
- `corridor.py` kennt weder Modell noch Mission und bekommt nur Masken. Diese
  Trennung erhalten — sie macht die Geometrie ohne Modell testbar.
- Die Ausgabe ist eine KI-gestuetzte Einschaetzung und ausdruecklich **keine
  sicherheitsrelevante Fahrfreigabe**. Diesen Vorbehalt in Berichten und UI beibehalten.
