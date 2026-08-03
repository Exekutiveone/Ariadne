# Missionsübergreifende Evaluation des CPU-Wegmodells

Erzeugt: 2026-08-03T19:40:49+00:00

## Datenbasis

| Mission | Bestätigte Frames | In dieser Evaluation |
|---|---|---|
| Misson 3 | 9 | nein |
| Misson  2 | 99 | ja |
| Mission 1 | 152 | ja |

Jeder Lauf trainiert auf **allen** bestätigten Frames einer Mission und misst auf
**allen** bestätigten Frames der jeweils anderen. Die Entscheidungsschwelle wird
ausschließlich auf zurückgehaltenen Frames der Trainingsmission gewählt; die
Evaluationsframes fließen an keiner Stelle in Training oder Schwellenwahl ein.

## Ergebnisse

| Lauf | Training | Evaluation | Positionsmerkmale | Merkmale | IoU | Precision | Recall | Eval-Frames |
|---|---|---|---|---|---|---|---|---|
| A | Mission 1 | Misson  2 | ja | 22 | 0.728 | 0.741 | 0.976 | 99 |
| B | Misson  2 | Mission 1 | ja | 22 | 0.702 | 0.876 | 0.779 | 152 |
| C | Mission 1 | Misson  2 | nein | 14 | 0.579 | 0.657 | 0.830 | 99 |
| D | Misson  2 | Mission 1 | nein | 14 | 0.440 | 0.737 | 0.522 | 152 |

## Vergleichswert: bisherige In-Mission-Metrik

| Quelle | Training | Validierung | IoU | Precision | Recall |
|---|---|---|---|---|---|
| Aktives globales Modell (global-path-20260803T193255Z-8b5ded24) | 203 Frames beider Missionen | 48 Frames derselben Missionen | 0.796 | 0.850 | 0.926 |

Diese Zahl hält jeden 5. Frame derselben Missionen zurück. Benachbarte Videoframes
sind visuell hochkorreliert, deshalb ist sie systematisch optimistisch und keine
Aussage über unbekanntes Gelände.

## Evidenzframes

Je Lauf die drei schlechtesten und zwei besten Frames nach IoU, links das
Originalbild, rechts das Overlay: grün korrekt, rot übersehene Wegfläche,
gelb fälschlich erkannte Wegfläche.

### A — Mission 1 → Misson  2

- `A-0-worst-dcdd9bf1-0015478.jpg` — worst, IoU 0.328, Frame 15479
- `A-1-worst-dcdd9bf1-0016422.jpg` — worst, IoU 0.402, Frame 16423
- `A-2-worst-dcdd9bf1-0011514.jpg` — worst, IoU 0.412, Frame 11515
- `A-3-best-dcdd9bf1-0000944.jpg` — best, IoU 0.938, Frame 945
- `A-4-best-dcdd9bf1-0001133.jpg` — best, IoU 0.924, Frame 1134

### B — Misson  2 → Mission 1

- `B-0-worst-8eba07e3-0001650.jpg` — worst, IoU 0.018, Frame 1651
- `B-1-worst-865b5ff9-0001650.jpg` — worst, IoU 0.052, Frame 1651
- `B-2-worst-865b5ff9-0001600.jpg` — worst, IoU 0.055, Frame 1601
- `B-3-best-66ae6f3a-0000450.jpg` — best, IoU 0.932, Frame 451
- `B-4-best-66ae6f3a-0000300.jpg` — best, IoU 0.910, Frame 301

### C — Mission 1 → Misson  2

- `C-0-worst-dcdd9bf1-0015478.jpg` — worst, IoU 0.201, Frame 15479
- `C-1-worst-dcdd9bf1-0011514.jpg` — worst, IoU 0.245, Frame 11515
- `C-2-worst-dcdd9bf1-0010759.jpg` — worst, IoU 0.254, Frame 10760
- `C-3-best-dcdd9bf1-0009815.jpg` — best, IoU 0.910, Frame 9816
- `C-4-best-dcdd9bf1-0017554.jpg` — best, IoU 0.905, Frame 17555

### D — Misson  2 → Mission 1

- `D-0-worst-865b5ff9-0000750.jpg` — worst, IoU 0.000, Frame 751
- `D-1-worst-865b5ff9-0000800.jpg` — worst, IoU 0.000, Frame 801
- `D-2-worst-865b5ff9-0001150.jpg` — worst, IoU 0.000, Frame 1151
- `D-3-best-66ae6f3a-0000450.jpg` — best, IoU 0.915, Frame 451
- `D-4-best-865b5ff9-0000030.jpg` — best, IoU 0.844, Frame 31

## Vorbehalt

Die Ausgabe ist eine KI-gestützte Einschätzung und keine sicherheitsrelevante
Fahrfreigabe.
