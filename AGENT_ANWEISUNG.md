# Arbeitsanweisung: Abgestufte Befahrbarkeits-Klassifikation + Trajektorienplanung

Auftrag für einen umsetzenden Agenten im Repo ARIADNE. Lies zuerst `CLAUDE.md` vollständig — alle dortigen Konventionen gelten uneingeschränkt. Diese Anweisung wurde mit dem Nutzer abgestimmt; die getroffenen Entscheidungen sind bindend und nicht neu zu verhandeln.

## Ziel

Die CPU-Wegerkennung (`ariadne-cpu-path-rff`) gibt heute eine binäre Maske aus (Weg / kein Weg). Sie soll eine **abgestufte Farbklassifikation** liefern und eine **Fahrtrajektorie** vorschlagen:

- **3 Grüntöne**: je dunkler, desto sicherer befahrbar.
- **Orange**: potenziell befahrbar mit Risiko (Unsicherheitsband um den Schwellwert — Weg A, KEIN neues Labeln, KEIN Mehrklassen-Umbau des Modells).
- **Rot**: Problemzonen — nur zusammenhängende Hindernisflächen im oder direkt am Fahrbereich. Himmel und ferne Umgebung bleiben transparent.
- **Trajektorie**: eine geglättete Linie, wie das Fahrzeug fahren könnte, abgeleitet aus der abgestuften KI-Maske.

Die bestehende binäre Maske und der Refinement-Workflow bleiben unverändert erhalten — **alle Änderungen sind additiv** (Antwortschema erweitern, nichts ersetzen). Gespeicherte Modelle (`model.npz`) bleiben gültig; `_features` und `MODEL_SCHEMA_VERSION` werden NICHT angefasst.

## Klassen-Ontologie der neuen Abstufungsmaske (`grade_mask`)

RLE-kodiert wie die bestehende Maske (`_encode_binary_rle` ist bereits generisch für Mehrwertmasken).

| Wert | Schlüssel | Bedeutung (deutsches UI-Label) | Farbe |
|---|---|---|---|
| 0 | `unrated` | Nicht bewertet / Umgebung (transparent) | `#00000000` |
| 1 | `safe` | Sicher befahrbar | `#1e8c46` |
| 2 | `good` | Gut befahrbar | `#55d96f` |
| 3 | `marginal` | Knapp befahrbar | `#a3ecb4` |
| 4 | `risky` | Potenziell befahrbar, mit Risiko | `#f08c3a` |
| 5 | `problem` | Problemzone / Hindernis | `#e05b52` |

Diese Hexwerte sind vorgegeben (abgestimmt auf die bestehende UI-Palette). Legende und Ontologie in der API-Antwort mitliefern (deutsche Labels, englische Schlüssel — wie `GROUND_TRUTH_ONTOLOGY` in `annotations.py`).

## Abstufungslogik (Startwerte, visuell zu verifizieren)

Die RFF-Regression zielt auf Targets ±1; der Schwellwert `t` wird auf Validierungsframes gewählt. Definiere den normierten Abstand `m = (s − t) / max(ε, 1 − t)` pro Pixel-Score `s`.

- Innerhalb der bereinigten Binärmaske (`prediction == 1`): `m ≥ 0.6` → Wert 1 (dunkelgrün); `0.25 ≤ m < 0.6` → Wert 2; sonst → Wert 3.
- Außerhalb (`prediction == 0`): `m ≥ −0.2` → Wert 4 (orange, Unsicherheitsband); sonst Kandidat „sicher nicht befahrbar".
- **Rot (Wert 5)** nur für sicher-nicht-befahrbare, zusammenhängende Komponenten (connected components), die den befahrbaren Bereich berühren oder ihm nahe sind (z. B. Dilatation der Grünfläche als Nachbarschaftstest) und eine Mindestgröße haben. Alle übrigen Negativ-Pixel → Wert 0 (transparent).
- Auf die Stufengrenzen leichte morphologische Glättung anwenden, damit das Overlay im Video nicht flimmert (analog `_clean_prediction`).

Die Bandgrenzen (0.6 / 0.25 / −0.2) und die Mindestgröße sind Startwerte: an den Evidenzframes echter Missionen visuell prüfen und bei Bedarf anpassen. Gewählte Endwerte als Konstanten mit Kommentar dokumentieren und in die API-Antwort schreiben (`grading`-Block mit Bandgrenzen), damit Ergebnisse reproduzierbar sind.

## Umsetzungsort im Backend

- Gemeinsame Hilfsfunktion in `backend/app/path_model.py` (z. B. `_grade_prediction(scores, prediction, threshold, shape)`), genutzt von:
  - `predict_path_frame` (Missionsmodell),
  - dem Global-Predict in `backend/app/global_path_model.py`,
  - der Vollvideo-Inferenz in `backend/app/global_video_analysis.py` (siehe Phase 3).
- Antwort additiv erweitern: `grade_mask` (RLE), `grade_ontology`, `grading` (Bandgrenzen). Bestehende Felder unverändert.
- Vorbehalt beibehalten und auf die neue Darstellung ausweiten: Die Ausgabe ist eine KI-Einschätzung und **keine sicherheitsrelevante Fahrfreigabe** — in Legendentexten und Berichten explizit nennen.

## Umsetzungsort im Frontend

Die Abstufung wird an **allen drei** Stellen angezeigt:

1. `src/GroundTruthLabeler.tsx` — die eingeblendete KI-Maske (`showAiMask` / `predictPathFrame`) abgestuft rendern.
2. `src/GlobalModelDashboard.tsx` — Frameanalyse des globalen Modells (`predictGlobalPathFrame`) abgestuft rendern; bei Vollvideo-Ergebnissen die gespeicherten `grade_mask`-Frames abspielen.
3. `src/AnalysisView.tsx` — neuer Overlay-Modus im Player (neben Original/Boden/Befahrbarkeit/Annotation), der die abgestufte KI-Maske und die Trajektorienlinie zeigt.

Jeweils: Farbwerte exakt wie oben, Legende mit deutschen Labels, Deckkraft-Regler wiederverwenden wo vorhanden. `types.ts` additiv erweitern (`grade_mask?`, `grade_ontology?`, `grading?`, `trajectory?`). Rendern über die bestehende `renderRleMask`-Infrastruktur.

## Trajektorie (wird in diesem Auftrag mit umgesetzt)

- Vorbild ist die Centerline-/Korridor-Logik in `backend/app/terrain.py` (`_corridor`): Diese Konzepte auf die **KI-Abstufungsmaske** übertragen, nicht die heuristische Terrain-Maske wiederverwenden.
- Fahrbare Grundfläche = Stufen 1–3 (Grüntöne); Orange zählt NICHT als fahrbar. Größte zusammenhängende, von der unteren Bildkante erreichbare Komponente wählen, erodieren, zeilenweise Mittellinie bestimmen, glätten (z. B. gleitender Mittelwert), als normierte Punktliste `trajectory.centerline` ausgeben.
- Status-Gating wie beim ARGUS-Korridor: bei zu schmaler Fläche, geringem Dunkelgrün-Anteil unter der Linie oder abgerissener Komponente `status: "uncertain"` bzw. `"unavailable"` mit deutschen `reasons` liefern statt einer erzwungenen Linie.
- Frontend: Linie als Polyline über dem Overlay (Farbe nach Status, wie beim bestehenden Korridor), abschaltbar per Toggle. Punktkoordinaten normiert auf das Originalbild (Repo-Konvention).

## Vollvideo-Analyse

`global_video_analysis.py` speichert zusätzlich pro Frame `grade_mask` (und `trajectory`, sofern berechnet). Wiederaufnehmbarkeit der Checkpoints darf nicht brechen: Bereits vorhandene Teilergebnisse ohne `grade_mask` müssen weiter ladbar sein (Feld optional behandeln oder Ergebnis-Schema-Version anheben und sauber migrieren — begründet entscheiden). Laufender Zustand bleibt unter `ARIADNE_RUNTIME_DIR` (nie im OneDrive-Projektordner).

## Vorgehen: 3 Phasen mit Zwischenstopps

Nach **jeder** Phase: anhalten, kompakten Ergebnisbericht auf Deutsch liefern (was geändert, welche Tests grün, offene Punkte) und auf Freigabe des Nutzers warten. Nicht eigenmächtig in die nächste Phase gehen.

1. **Phase 1 — Backend-Abstufung (Einzelframe):** `_grade_prediction` + Erweiterung von `predict_path_frame` und Global-Predict, inkl. neuer Unit-Tests.
2. **Phase 2 — Frontend-Anzeige:** alle drei Views, Legenden, Typen, Frontend-Tests.
3. **Phase 3 — Vollvideo + Trajektorie:** Vollvideo-Integration, Trajektorien-Backend + -Anzeige, Tests.

## Qualitätsanforderungen (verpflichtend, je Phase)

- Neue Unit-Tests: Abstufungslogik (synthetische Scores → erwartete Stufen, Rot-Problemzonen-Logik, RLE-Roundtrip) und Trajektorie (synthetische Masken → Centerline/Status). Muster: `backend/tests/test_path_model_core.py`.
- Alles grün, ohne Ausnahme: `python -m pytest backend/tests -q` (exakt so aufrufen, siehe CLAUDE.md), `npm test`, `npx tsc -b`, `python -m ruff check .` und `python -m ruff format .`.
- Nichts an Train/Validation-Split, `_features`, Threshold-Wahl oder gespeicherten Modellformaten ändern. Kein Training nötig — reine Inferenz-/Darstellungserweiterung.
- Deutsche Nutzertexte, englische Bezeichner. HTTP-500 ohne Exception-Details an den Client.

## Ausdrücklich NICHT Teil dieses Auftrags

- Weg B (echte Mehrklassen-Labels „riskant"/„Hindernis", Labeler-Umbau, Mehrklassen-Training) — bewusst verschoben.
- Änderungen an Segmentierung, Terrain-Heuristik, Missionsverwaltung oder Datenablage-Richtlinien (Videos bleiben lokal, siehe CLAUDE.md).
