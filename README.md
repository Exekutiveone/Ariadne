# ARIADNE – Survey- und Terrain-Player

Lokale, persistente Erfassung und Auswertung manueller Waldbegehungen. Die Anwendung speichert Mission, Route und Originalvideos, erkennt Vegetationsinstanzen und berechnet je Analyseframe eine Boden- und Befahrbarkeitsmaske samt zeitlich stabilisiertem ARGUS-Korridor.

## Start

```powershell
python -m pip install -r backend/requirements.txt
npm install
npm run dev
```

Danach `http://localhost:5173` öffnen. Vite leitet `/api` an das Backend auf Port 8000 weiter. Persistente Daten liegen standardmäßig unter `data/missions`; mit `ARIADNE_DATA_DIR` kann ein anderer Pfad gesetzt werden.

## Tests und Produktions-Build

```powershell
python -m pytest backend/tests
npm test
npm run build
```

Die stabile Übergabeschnittstelle für Goal 2 ist unter `GET /api/v1/missions`, `GET /api/v1/missions/{id}` und `GET /api/v1/missions/{id}/videos/{video_id}/content` verfügbar. Das Datenmodell ist in `backend/app/models.py` definiert.

## Goal-4-Terrainanalyse

Nach einer ausdrücklich gestarteten Auswertung kann im Analyseplayer zwischen Original, Boden, Befahrbarkeit und `Eigene Labels` gewechselt werden. Gespeicherte manuelle Labels lassen sich gezielt nach Frame auswählen und mit eigener Deckkraft anzeigen. In Boden- und Befahrbarkeitsansicht können sie zusätzlich über die KI-Maske gelegt werden; die weiße Polygonkontur unterscheidet Ground Truth und automatische Auswertung. Der davon getrennte Ground-Truth-Arbeitsbereich bleibt weiterhin rein manuell. Grün, Gelb, Rot und Grau der späteren KI-Auswertung werden aus dem jeweiligen Videoframe berechnet; repräsentative Quell- und Overlayframes werden mit dem Analyselauf gespeichert.

### Ground Truth präzise markieren

1. Bei der Mission `Befahrbaren Weg labeln` öffnen. Dieser Arbeitsbereich lädt ausschließlich Originalvideo-Metadaten und vorhandene manuelle Polygone.
2. Entweder eine Schrittweite (`jeden 5.`, `10.`, `20.` Frame) oder eine gewünschte Gesamtzahl gleichmäßig verteilter Frames wählen.
3. Mit mindestens drei Punkten den befahrbaren Bereich markieren. Danach lassen sich einzelne Punkte verschieben, ergänzen oder entfernen sowie das gesamte Polygon verschieben.
4. Mit Mausrad oder `+`/`−` zoomen und mit dem Werkzeug `Bild verschieben` den Ausschnitt bewegen. Die Polygonkoordinaten bleiben relativ zum Originalbild gespeichert.
5. Beim Wechsel zum nächsten ausgewählten Frame bleibt das aktuelle Polygon automatisch als editierbare Vorlage liegen. Änderungen werden ausschließlich im neuen Frame gespeichert; das vorherige Label bleibt unverändert.
6. Nicht relevante Frames mit `Frame überspringen` auslassen. Undo, Redo sowie das Löschen des aktuellen Polygons stehen jederzeit vor dem Speichern zur Verfügung.

Alle gespeicherten Entwürfe, bestätigten Masken und übersprungenen Frames erscheinen in der Liste `Gespeicherte Polygonmasken`. Ein Klick öffnet den exakten Originalframe und legt die gespeicherte Maske wieder editierbar über das Video. Polygonpunkte lassen sich dabei immer direkt ziehen, unabhängig vom aktuell gewählten Werkzeug.

Während der Markierungsrunde werden keine vollständige Befahrbarkeitsanalyse oder Trajektorie erzeugt. Nur wenn bereits ein Wegmodell trainiert wurde und der Nutzer `KI-Wegmaske anzeigen` aktiviert, wird eine getrennte türkisfarbene Vergleichsmaske berechnet. Erst `Markierung abschließen & Auswertung starten` startet die übrige Verarbeitung ausdrücklich.

Entwürfe, bestätigte Polygone und übersprungene Frames werden quellgebunden unter `data/missions/<mission_id>/ground_truth/<video_id>/<frame_index>.json` gespeichert. Jeder Polygonpunkt liegt normiert auf dem Originalbild. Frameindex und Zeitstempel werden serverseitig gegen FPS und Gesamtframezahl des Originalvideos geprüft, damit Änderungen an einem Frame niemals vorherige Frames verändern.

### CPU-Wegerkennung trainieren

Im Ground-Truth-Bereich startet `Weg-KI auf ... Labelframes trainieren` ein lokales, nichtlineares Pixelmodell ohne Cloud und GPU. Pixel innerhalb bestätigter Polygone sind positive Wegbeispiele; Pixel außerhalb derselben Polygone sind negative Beispiele. Training und Validierung verwenden getrennte Frames. Die Schwelle wird ausschließlich auf den Validierungsframes so gewählt, dass übersehene gelabelte Wegfläche und fälschlich erfundene Wegfläche gleich stark bestraft werden.

Das Modell, der maschinenlesbare Ergebnisbericht und grün/rot/gelb markierte Evidenzframes liegen unter `data/missions/<mission_id>/derived/path_model_runs/<run_id>`. Diese erste missionsspezifische CPU-Baseline ist keine sicherheitsrelevante Fahrfreigabe.

Sobald ein Modell vorhanden ist, kann im Labeling-Arbeitsbereich `KI-Wegmaske anzeigen` aktiviert werden. Die türkisfarbene Maske wird aus dem jeweils sichtbaren Originalframe berechnet und bleibt beim Zoomen und Verschieben synchron über dem Bild. Ihre Deckkraft ist unabhängig von der grünen manuellen Ground-Truth-Maske einstellbar.

Für einen Feedback-Loop kann bei einem gelabelten Frame `Refinement starten` aktiviert werden. Ein Klick auf eine rote oder gelbe zusammenhängende Fehlerfläche wählt genau diese Region aus. Mit `KI hatte hier recht` wird die Korrektur getrennt vom ursprünglichen Polygon unter `path_refinements/<video_id>/<frame_index>.json` gespeichert, sofort in der Framebewertung berücksichtigt und beim nächsten Wegmodell-Training in die effektive Ground Truth übernommen.

Das `KI-Modellzentrum` auf der Hauptseite aggregiert alle Missionen mit bestätigten Labels. `Auf allen Labels trainieren` erzeugt daraus ein missionsübergreifendes CPU-Basismodell inklusive aller gespeicherten Refinements. Globale Läufe und Evidenz werden unter `data/global_models/path_model` gespeichert und überschreiben keine missionsspezifischen Modelle.

Im globalen Analyseplayer können anschließend Mission und Originalvideo ausgewählt werden. `Video vollständig analysieren` startet einen persistenten Hintergrundprozess, der jeden Originalframe einmal mit dem aktiven globalen Modell berechnet. Fortschritt, Framezahl, verstrichene Zeit und geschätzte Restzeit werden angezeigt. Nach Abschluss spielt der Browser das Originalvideo normal ab und legt die vorberechneten Masken synchron darüber. Ungelabelte Frames zeigen Türkis; gelabelte Frames zusätzlich Grün/Rot/Gelb und den individuellen Vergleichsscore.

Status, wiederaufnehmbare 100-Frame-Checkpoints und fertige Videoanalyse-Caches liegen standardmäßig unter `%LOCALAPPDATA%\Ariadne\runtime`; dadurch sind laufende Jobs nicht von OneDrive- oder Netzlaufwerk-Sperren abhängig. Mit `ARIADNE_RUNTIME_DIR` kann dieser Pfad explizit geändert werden. Atomare Statusupdates werden bei kurzzeitigen Sperren automatisch wiederholt und dürfen die Inferenz nicht abbrechen.

### Portable GitHub-Sandbox

Die operative Sandbox ist bewusst versionierbar: `data/missions` enthält Missionen, Originalvideos, Ground Truth, Refinements und missionsspezifische Modelle; `data/global_models` enthält das globale Modell; abgeschlossene Frameanalysen werden zusätzlich nach `data/runtime_cache` exportiert. Laufende Checkpoints bleiben zur Vermeidung von Sync-Sperren lokal und sind nach Abschluss nicht mehr erforderlich.

Große Originalvideos und NPZ-Modellartefakte werden über die Regeln in `.gitattributes` mit Git LFS gespeichert. Nach einem Clone muss Git LFS installiert sein und `git lfs pull` ausgeführt werden. Die ursprünglichen Importordner `data/Mission1` und `data/Mission 2` bleiben ausgeschlossen, weil sie bytegleiche Duplikate der bereits unter `data/missions/<id>/videos` gespeicherten Anwendungsdaten enthalten.

Für längere Optimierung steht `Training im Hintergrund` zur Verfügung. Der Schnelltest prüft einen Kandidaten; das Nachtprofil vergleicht bis zum gewählten Zeitlimit mehrere CPU-Konfigurationen und aktiviert anschließend automatisch den besten Lauf. Trainingsstatus und Protokolle werden persistent unter `derived/path_training_job.json` beziehungsweise `derived/path_training_logs` gespeichert. Der Browser darf geschlossen werden, der lokale Backend-Server und der Rechner müssen jedoch eingeschaltet bleiben.

Die ARGUS-Geometrie ist vor einer Auswertung über Umgebungsvariablen konfigurierbar:

```powershell
$env:ARIADNE_ARGUS_WIDTH_M='0.35'
$env:ARIADNE_ARGUS_SAFETY_MARGIN_M='0.20'
$env:ARIADNE_TERRAIN_NEAR_FIELD_WIDTH_M='3.2'
$env:ARIADNE_TERRAIN_METRIC_CALIBRATION='perspective_estimate'
```

Ohne explizite Konfiguration nutzt die lokale Demo dokumentierte Arbeitsannahmen. Die Ausgabe ist eine KI-gestützte Einschätzung und keine sicherheitsrelevante Fahrfreigabe.
