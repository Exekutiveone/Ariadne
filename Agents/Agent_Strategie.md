# AIGIS — Datenkonzept Bilderkennung & Autonomie

### Wahrnehmungs- und Autonomiemodul zwischen ARGUS (Fahrzeug) und Athene (Analytics-Plattform)

_Stand: 03.08.2026 — strukturiert nach Umsetzungshorizont_

Dieses Dokument ist die langfristige Referenz für das AIGIS-Konzept. Es ist in zwei Teile geteilt:

- **Teil A — JETZT:** Scope der aktuellen Entwicklungsphase. Alles hier ist für den Code-Agent relevant und darf umgesetzt, erweitert und refactored werden.
- **Teil B — ZUKUNFT:** Bewusst verschobene Bausteine. Diese dienen als Architektur-Kontext (nichts bauen, was ihnen später im Weg steht), werden aber **nicht implementiert**, bis sie explizit in Teil A wandern.

---

## Grundlagen (gelten für beide Teile)

### Zweck

AIGIS ist der Arbeitstitel für das Bilderkennungsmodul von ARGUS und langfristig für die Schnittstelle ARGUS → Athene. Die Bilderkennung wird **als eigenständiges Modul unabhängig vom Fahrzeug** entwickelt:

1. Training findet off-device statt, nicht auf dem Fahrzeug.
2. Die Bilderkennung ist der erste Baustein der Kette — ohne sie kann das Gesamtsystem nicht arbeiten.
3. Das Fahrzeug fährt aktuell noch nicht zuverlässig; die Modulentwicklung darf davon nicht blockiert werden.

### Fahrzeuggeometrie (Konstanten für alle geometrischen Berechnungen)

| Parameter                | Wert     | Charakter                 |
| ------------------------ | -------- | ------------------------- |
| Fahrzeugbreite           | 30–40 cm | fix                       |
| Lenk-/Sicherheitsbereich | + 10 cm  | fix erforderlich          |
| Wendekreis               | ~1 m     | weiches Kriterium         |
| Gewichtsklasse           | ~40 kg   | Kontext für Befahrbarkeit |

### Kamerasetup ARGUS (Zielzustand)

- Drei RGB-Kameras (2× Global Shutter, 1× Rolling Shutter), dauerhafte Videoaufnahme.
- Eine Kamera frontal (0°) — Fahrzeugsteuerung.
- Zwei Kameras nach oben gewinkelt — spätere Vegetationserkennung.

### Aktuelle Datenbasis

- ~45 Min. Handy-Video (davon ~20 Min. gelabelt), nicht georeferenziert, nach _unten_ geneigt.
- **Bekannter Domain Gap:** Handy-Optik/-Winkel ≠ spätere ARGUS-Kameras. Bei allem, was jetzt trainiert wird, mitdenken; nichts hart auf die Handy-Perspektive optimieren.

---

# TEIL A — JETZT (Scope für den Agent)

## A.1 Untergrund-/Terrainklassifikation (Wahrnehmungs-Ebene 1)

Grundlegendste inhaltliche Erkennungsebene. Klassenliste (offen erweiterbar):

- Asphaltierte Straße
- Schotterweg
- Mischformen (z. B. Schotter + Wiese)
- Feld-/Landwege mit Fahrspuren (niedergefahrene Spuren mit Mittelstreifen)
- Walduntergrund
- Wiese flach (problemlos befahrbar)
- Wiese hoch (erheblich erschwerte Fahrt)
- Sammelkategorien für den Einstieg: **Mischkategorie** (Mehrfachauswahl beteiligter Untergründe) und **Klar definierter Weg**

Wichtige Trennung: _semantische Klasse_ (was ist es?) ≠ _Befahrbarkeit_ (kann dieses Fahrzeug dort fahren?). Beides wird getrennt modelliert. Aktueller Modellstand: binäre Befahrbarkeits-Segmentierung — Ausbau Richtung dreistufig (befahrbar / unsicher / nicht befahrbar) und Richtung Untergrundklassen.

## A.2 Befahrbarkeits-Segmentierung inkl. Grauzone

Die grobe Einteilung befahrbar / nicht befahrbar ist machbar — der kritische Teil ist der **Zwischenraum**. Genau an der Grauzone scheitern UGVs, nicht an Betonwegen oder Pfützen. Befahrbarkeit ist keine reine Bildeigenschaft (hängt von Masse, Traktion, Bodenfeuchte ab). Ziel dieser Phase: die Grauzone als **eigene Klasse** etablieren und gezielt labeln.

## A.3 Korridor-Bewertung im Bildraum (Wahrnehmungs-Ebene 3)

Deterministische Geometrie auf der Segmentierungsmaske — **kein ML**:

- Drei Standard-Korridore: **Mitte** (Default bei schmalen Wald-/Feldwegen), **Rechts** (Rechtsfahrgebot: Radwege, landwirtschaftliche Wege), **Links** (Ausweichoption).
- Prüfung: Ist ein durchgehend ausreichend breiter Streifen (Fahrzeugbreite + 10 cm, perspektivisch Richtung Fluchtpunkt skaliert) im Korridor frei?
- Längsdimension bewusst irrelevant — nur die Breitenprüfung zählt.
- Output pro Frame: Status je Korridor (frei / blockiert / unsicher).

## A.4 Räumliche Bildzerlegung — Vorstufe (aus Ebene 2)

Jetzt schon sinnvoll als billiger Vorfilter:

- **Fluchtpunkt-Geometrie:** Linien von den unteren Bildrändern zum Fluchtpunkt grenzen den sicher irrelevanten oberen Bildbereich (Himmel/Ferne) vorab ab.
- **Irrelevanz-Zone (oben):** oberhalb des Fluchtpunkts keine Auswertung versuchen.

Die volle Zonen-Zerlegung (Nahbereichs-Box, Seitenvegetation als eigene Klasse, Weg-Fallunterscheidungen) → Teil B.

## A.5 Daten-Infrastruktur: Run-Registry

Verwaltung der Videodaten als "Runs" (SQLite + lokale Weboberfläche):

- Automatischer Scan des Video-Ordners; neue Aufnahmen werden als Runs angelegt.
- Pro Run: Status (`unlabeled` / `queued_for_labeling` / `labeled` / `training_ready`), vorherrschender Untergrund (Klassenliste aus A.1 inkl. Sammelkategorien), Freitextnotiz.
- Die Registry ist die **eine Wahrheitsquelle** für die Kette Run → Frames → Masken → Training.

## A.6 Labeling-Workflow

- **Sparse-Annotation:** nicht jeder Frame, sondern z. B. jeder 50. (Video-Frames sind nahezu redundant).
- **Machine-assisted Labeling:** externes Tool (X-AnyLabeling mit SAM) statt Eigenentwicklung — grob klicken, SAM zieht die Maske, korrigieren. Wenige hundert gut gelabelte Frames reichen für ein Finetuning; kein GB/TB-Ziel mehr.
- **Rückimport:** gelabelte Masken werden per Skript ihrem Run zugeordnet, Status springt automatisch weiter.

## A.7 Active Learning

Statt stur jeden N-ten Frame zu labeln: das aktuelle Modell über ungelabelte Daten laufen lassen und gezielt die Frames in die Labeling-Queue schreiben, bei denen es unsicher ist (z. B. hoher Anteil Pixel nahe Entscheidungsgrenze). Handarbeit fließt dorthin, wo sie den größten Lerneffekt hat.

## A.8 Trainings-Pipeline (portabel)

- Kompaktes Segmentierungsmodell (z. B. U-Net mit MobileNetV3-Backbone), Finetuning statt Training from scratch.
- **Containerisiert von Anfang an** (Docker, config-gesteuert): lokal und Cloud identisch lauffähig — "wo trainiere ich" ist eine Konfigurations-, keine Architekturentscheidung.
- **Smoke-Test-Pflicht:** Jede Pipeline-Änderung muss lokal auf ~20 Frames / CPU durchlaufen, bevor Cloud-Compute angefasst wird.
- Compute-Strategie: lokaler Desktop = Labeling- und Inferenz-Station; Training in der Cloud (RunPod/Lambda, später AWS). Keine GPU-Hardware kaufen.

## A.9 Inferenz-Performance

Aktuell ~2 s/Frame. Quick-Win-Kandidat: ONNX-Export + lokale GPU (Pascal, 4 GB) → erwarteter Faktor 5–10. Relevant für Active-Learning-Durchläufe über ungelabelte Daten.

## A.10 Stuck-Event-Datenbank (nur Schema + Erfassung, keine Auswertung)

Vorbereitung für das Feststeck-Protokoll aus Teil B: SQLite-Schema + simples Erfassungsformular für Stuck-Events — 9 Fotos (vorne, hinten, links, rechts, oben, 4× Rad), Schadensbericht (mechanischer Schaden ja/nein), Zeitstempel, Untergrundklasse, Freitext. Muss ab dem ersten echten ARGUS-Fahrtag existieren; nachträglich rekonstruieren geht nicht. Auswertung/Training darauf → Teil B.

---

# TEIL B — ZUKUNFT (bewusst verschoben, nicht implementieren)

> Kontext für Architekturentscheidungen. Nichts in Teil A darf diese Bausteine strukturell verbauen — aber sie werden erst umgesetzt, wenn sie explizit nach Teil A wandern. Auslöser sind jeweils vermerkt.

## B.1 Bildqualitäts-Gate (Ebene 0) — _wartet auf: echte ARGUS-Kameras_

Gatekeeper vor jeder Auswertung: verwackelte Frames, starke Sonneneinstrahlung/Blendung, verschmutzte oder beschädigte Linse erkennen. Mit Handy-Daten nicht sinnvoll trainierbar — die Störmuster der echten Kameras sind andere.

## B.2 Volle räumliche Zonen-Zerlegung (Ebene 2) — _wartet auf: ARGUS-Kameraperspektive_

- **Nahbereichs-Box (unten):** v. a. bei Fisheye/nach unten geneigten Kameras; für Trajektorienplanung irrelevant, aber präzise Ground-Truth des aktuellen Standorts (Feststeck-Kontext).
- **Weg-Fallunterscheidung:** (a) klassischer Weg, (b) großflächiges Gelände ohne Weg, (c) Gelände mit vom Fahrzeug selbst geplantem Pfad.
- **Seitenvegetation als eigene Klasse:** Für die Fahrt nur Border; für Athene (Vegetationsmonitoring) hochrelevant. Gleiche Pixel, zwei Konsumenten.

## B.3 Szenen-Anomalien (Ebene 4) — _wartet auf: stabile Ebene-1-Basis + mehr Daten_

- Statische Hindernisse: Baumstämme, Objekte; feste Gegenstände > ~10 cm als Kollisionsrisiko-Schwelle.
- Untergrund-Störungen im Weg: Pfützen, Vegetation _innerhalb_ des Wegs.
- **Menschen als eigene Störklasse** — dynamisch, sicherheitskritisch, eigene Reaktionsanforderungen.

## B.4 Kartenbildung — _wartet auf: fahrendes Fahrzeug_

1. **Stufe 1 — Lineare Karte:** Fahrt als Linie, Erkennungen sequenziell abgetragen (1D-Log; braucht nur Zeitstempel — frühester Kandidat für Teil A).
2. **Stufe 2 — Geometrische Wegverfolgung:** Wegform rein visuell rekonstruieren (visuelle Odometrie). Entschieden: wird gebaut, **nicht** übersprungen — GPS-Ausfall ist im Wald fast der Normalfall (Abschattung/Multipath unter Kronendach, Fehler 10–30 m üblich).
3. **Stufe 3 — Georeferenzierung.** Zielarchitektur: Fusion — GNSS als Anker wo verfügbar, dazwischen relative visuelle Verfolgung.

## B.5 Sensorfusion — _wartet auf: fahrendes Fahrzeug mit Sensorik_

- IMU/Telemetrie (relative Bewegung), Radodometrie, Magnetometer (absolute Ausrichtung, abgeschirmt).
- Einordnung: Magnetometer ist der störanfälligste Sensor (dynamische Hard-/Soft-Iron-Störungen durch Motorstrom; Umgebung: Zäune, Leitungen, mineralhaltiger Boden) — zuverlässig erst in Fusion mit Gyro (Madgwick/Kalman). Radodometrie: Schlupf-Problem im Wald. Diese Einzelschwächen sind das stärkste Argument für die Kamera als gleichberechtigten Fusionspartner.

## B.6 Simulation & Abgleich — _wartet auf: Karte (B.4)_

Virtueller ARGUS in der Karte: vermessene Kamera-Geometrie als Sichtfeld-Kegel, Fahrzeuggeometrie als Footprint. Trajektorienplanung damit auf zwei Ebenen — reaktiv im Bildraum (A.3) und deliberativ im Kartenraum — mit gegenseitiger Validierung. Widersprüche zwischen beiden Ebenen zeigen Erkennungsfehler oder reale Weltveränderungen an.

## B.7 Feststeck-Protokoll & Traversability-Learning — _wartet auf: erste Stuck-Events (Schema aus A.10 liegt bereit)_

- Automatische Stuck-Detektion (Räder drehen, Position ändert sich nicht).
- Operator-Messprotokoll → Datenbank (A.10). Doppelter Nutzen: Hardware-Verbesserung + Ground-Truth „wirklich nicht befahrbar".
- Stuck-Events sind die einzigen Labels, die kein Mensch vergeben kann — Befahrbarkeit für _dieses_ Fahrzeug lernt nur das Fahrzeug durch Scheitern (_self-supervised traversability learning_, vgl. BADGR).
- **Rückwirkendes Labeln:** die letzten N Sekunden Video _vor_ dem Feststecken sind das Trainingsmaterial (Anfahrtsperspektive aus 2–5 m); das Ereignis ist nur das Label.

## B.8 Verhaltensebene — _wartet auf: autonome Fahrversuche_

- **Recovery-Skills als diskrete Bibliothek:** Rück-und-Anlauf (hohes Gras), Reifen freilenken (Pendeln), erweiterbar. Lernaufgabe schrumpft von End-to-End-Motorsteuerung auf Skill-Auswahl (Klassifikation). Jedes Stuck-Event liefert doppelt Daten: „Untergrund X → festgesteckt" + „Skill Y gelöst / nicht gelöst".
- **Fahrprofile pro Streckentyp:** Geschwindigkeit/Aggressivität als Parameter-Sets, gekoppelt an Ebene-1-Klassen.
- **Missionsparameter** als oberste Vorgabe (Route, Ziel, Monitoring-Auftrag).

## B.9 Systemarchitektur (Zielbild) — _wartet auf: mehr als ein Modul auf dem Fahrzeug_

- **State Machine / Behavior Tree** steuert die Fahrt; die KI gibt nur einen angestrebten Trajektorienpunkt vor; kleine deterministische Regler (Pure Pursuit/PID/MPC) fahren dahin.
- **Master-Router:** erkennt Systemzustand (normale Fahrt, Grauzone, festgesteckt, Mensch erkannt) und aktiviert das passende Spezialmodul. Anfangs regelbasiert (if-Logik), später ggf. gelernt. Mehrere kleine Spezialmodelle statt eines großen (auf Edge-Hardware auch speichertechnisch besser; Modellwechsel zur Laufzeit unproblematisch).
- Referenzrahmen: ROS 2 / Nav2 (hierarchischer Stack inkl. Behavior Trees).
- Hauptgewinn: **Testbarkeit & Sicherheit** — jedes Modul isoliert testbar; ein schlechtes Wahrnehmungsmodell kann den deterministischen Not-Stopp nicht kompromittieren. Voraussetzung für Abnahme/Versicherung im öffentlichen Wald.
- Bekannter Preis: **Schnittstellen** (Einheiten, Koordinatensysteme, Timing) — früh und schriftlich definieren, sobald Modul Nr. 2 existiert.

## B.10 Edge-Deployment — _wartet auf: Zielhardware-Entscheidung (Hailo-8L vs. Jetson)_

Zielpfad: Cloud → lokaler PC → Edge. Hailo-8L erzwingt INT8-Quantisierung und begrenzte Operationen; Jetson ist flexibler (CUDA), aber teurer/stromhungriger. **Vor** dem nächsten größeren Training: Export-Kette (PyTorch → ONNX → Zielformat) einmal komplett mit dem aktuellen Modell durchtesten — Inkompatibilitäten sofort finden, nicht nach Monaten Training.

---

## Gesamtbild: Der Stack

```
Missionsparameter                                    [B.8]
      │
Master-Router (Zustandserkennung)                    [B.9]
      │
Verhaltensebene (Fahrprofile, Recovery-Skills)       [B.8]
      │
Planung (Kartenraum: Simulation, virtueller ARGUS)   [B.6]
      │                          ▲
Kartierung (linear → geometrisch → georef.)          [B.4]
      │                          │ Fusion: IMU, Odometrie,
Wahrnehmung (Ebenen 0–4)         │ Magnetometer, GNSS [B.5]
  ├── Ebene 0 Qualitäts-Gate     [B.1]
  ├── Ebene 1 Untergrund         [A.1]  ← JETZT
  ├── Ebene 2 Zonen              [A.4 Vorstufe / B.2]
  ├── Ebene 3 Korridore          [A.3]  ← JETZT
  └── Ebene 4 Szenen-Anomalien   [B.3]
      │
State Machine / determ. Fahrregelung → Motoren       [B.9]

Quer durch alles: Feedback-Schleife                  [A.5–A.10, B.7]
(Registry, Sparse-Labeling, SAM-Assist,
 Active Learning, Stuck-Protokoll)
```
