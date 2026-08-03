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
- `backend/app/annotations.py`, `labeling.py` — manuelle Ground-Truth-Polygone.
- `src/` — React-Frontend. `AnalysisView.tsx` (Player mit Overlays) und
  `GroundTruthLabeler.tsx` (Polygonwerkzeug) sind die beiden grossen Komponenten.

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
| `data/missions/*/videos` | 3,1 GB | nein — siehe LFS-Entscheidung unten |
| `data/missions/*/derived` | 470 MB | nein — bei jedem Lauf neu berechnet |
| `data/runtime_cache`, `data/Mission1`, `data/Mission 2` | 3,1 GB | nein — Cache bzw. Duplikate |

**Offene Entscheidung:** `.gitattributes` richtet Git LFS fuer `.mov/.mp4/.npz`
bereits ein, aber 3,1 GB Video sprengen GitHubs Gratiskontingent (1 GB Storage,
1 GB Traffic pro Monat). Die Videos sind deshalb vorerst ignoriert. Vor dem
Aktivieren: LFS-Datenpaket buchen oder eigenes Remote verwenden.

## Konventionen und Fallstricke

- Formatierung und Linting ausschliesslich ueber `ruff` (Konfiguration in `pyproject.toml`,
  Zeilenlaenge 120). Backend-Code war frueher in zwei unterschiedlichen Stilen geschrieben;
  das ist vereinheitlicht und soll so bleiben.
- HTTP 500 gibt **keine** Exception-Details an den Client zurueck — Details gehoeren ins
  Log via `log.exception(...)`. HTTP 409/404 tragen bewusst fachliche deutsche Meldungen.
- Laufender Zustand (Checkpoints, Jobstatus) liegt unter `%LOCALAPPDATA%\Ariadne\runtime`,
  nicht im Projektordner — das Repo liegt in OneDrive und Sync-Sperren wuerden laufende
  Jobs abbrechen. Ueber `ARIADNE_RUNTIME_DIR` aenderbar, `ARIADNE_DATA_DIR` fuer die Daten.
- Ground-Truth-Polygonpunkte sind auf das Originalbild normiert, nie in Pixeln.
  Aenderungen an einem Frame duerfen nie einen anderen Frame veraendern.
- Aendert sich die Merkmalszahl in `_features`, werden alle gespeicherten Modelle
  ungueltig. `MODEL_SCHEMA_VERSION` anheben; `test_path_model_core.py` pinnt die Zahl.
- Train/Validation werden nach Frames getrennt und die Schwelle nur auf
  Validierungsframes gewaehlt. Diese Trennung nicht aufweichen.
- Die Ausgabe ist eine KI-gestuetzte Einschaetzung und ausdruecklich **keine
  sicherheitsrelevante Fahrfreigabe**. Diesen Vorbehalt in Berichten und UI beibehalten.
