"""Eine Wahrheitsquelle fuer alle Labelklassen.

Vier Ebenen, bewusst getrennt, weil sie im Training unterschiedlich benutzt
werden:

1. **Kernklassen** — genau eine je Bodenflaeche. Sie sind das Ziel des
   Wegmodells und tragen als einzige einen Maskenwert.
2. **Hindernisse** — *warum* etwas nicht befahrbar ist. Liegen ueber der
   Bodenflaeche, nicht daneben.
3. **Problemzonen** — Uebergaenge und Stoerungen, die eine Flaeche unsicher
   machen, ohne ein Hindernis zu sein (hohes Gras, Matsch, Schatten, Unschaerfe).
4. **ROI** — welche Bildbereiche ueberhaupt ausgewertet werden. Eine eigene
   Ebene statt eines Zuschnitts: das Originalbild bleibt vollstaendig, die Regel
   ist spaeter aenderbar, ohne dass Daten verloren gehen.

Bestandsschutz: die 276 vorhandenen Ground-Truth-Dateien tragen alle
`class_id: "traversable"` ohne Metadaten. Sie bleiben ohne Migration gueltig —
`traversable` ist weiter eine Kernklasse mit Maskenwert 1, und alle neuen Felder
haben Vorgaben, die genau den alten Zustand beschreiben. Neue Maskenwerte werden
angehaengt statt umnummeriert, damit gespeicherte Rastermasken gueltig bleiben.
"""

LABEL_SCHEMA_VERSION = "3.0"

# Maskenwert 0 heisst "nicht markiert" und wird im Training ignoriert — nicht
# "nicht befahrbar". Diese Unterscheidung ist dieselbe wie bei der Abstufung des
# Wegmodells (siehe corridor.py) und aus demselben Grund wichtig.
UNLABELLED = {"key": "unlabelled", "value": 0, "label": "Nicht markiert / im Training ignorieren", "color": "#00000000"}

# Ebene 1 --------------------------------------------------------------------
# Werte 1..3 stammen aus der Ontologie vor dem 04.08.2026 und bleiben unveraendert.
# "restricted" ist neu und haengt sich als 4 an, statt die Reihenfolge zu drehen.
CORE_CLASSES = {
    "traversable": {
        "value": 1,
        "label": "Befahrbarer Boden",
        "color": "#55d96f",
        "description": "Fester Untergrund, auf dem das Fahrzeug ohne Einschränkung fahren kann.",
    },
    "restricted": {
        "value": 4,
        "label": "Eingeschränkt befahrbar",
        "color": "#e4c264",
        "description": "Befahrbar, aber mit Risiko oder nur langsam — etwa Matsch, loser Schotter, hohes Gras.",
    },
    "not_traversable": {
        "value": 2,
        "label": "Nicht befahrbar",
        "color": "#e05b52",
        "description": "Sicher nicht befahrbar, unabhängig von der Ursache.",
    },
    "unknown": {
        "value": 3,
        "label": "Nicht bewertbar / verdeckt",
        "color": "#737c78",
        "description": "Keine Aussage möglich: verdeckt, überbelichtet, zu unscharf.",
    },
}

# Ebene 2 --------------------------------------------------------------------
# Hindernisse tragen keinen eigenen Maskenwert: sie erklaeren eine Flaeche, sie
# ersetzen sie nicht. Ein Baum steht auf nicht befahrbarem Boden — beides wird
# markiert, und das Modell kann spaeter lernen, woran es lag.
OBSTACLE_CLASSES = {
    "tree": {"label": "Baum", "color": "#2f7d4f"},
    "shrub": {"label": "Busch / dichtes Gebüsch", "color": "#4f9c5f"},
    "log": {"label": "Großer Ast / Baumstamm", "color": "#8a6a3d"},
    "rock": {"label": "Stein", "color": "#8f9499"},
    "ditch": {"label": "Graben", "color": "#5b4a7a"},
    "water": {"label": "Wasser / Pfütze", "color": "#3f7fbf"},
    "barrier": {"label": "Zaun / Pfosten / harte Barriere", "color": "#c94f8a"},
}

# Ebene 3 --------------------------------------------------------------------
# Diese Flaechen sind im Einsatz die schwierigen. Sie werden selten "klar
# befahrbar" sein, sondern meist zusammen mit der Kernklasse "restricted"
# markiert.
ZONE_CLASSES = {
    "tall_grass": {"label": "Hohes Gras", "color": "#9bbf5a"},
    "leaves": {"label": "Laub", "color": "#bf8f4a"},
    "mud": {"label": "Matsch", "color": "#7a5c3d"},
    "loose_gravel": {"label": "Loser Schotter", "color": "#a8a49a"},
    "mixed_surface": {"label": "Gemischter Untergrund", "color": "#b0975f"},
    "narrow_passage": {"label": "Engstelle", "color": "#d98f3f"},
    "partial_occlusion": {"label": "Teilverdeckung", "color": "#6f7a86"},
    "shadow": {"label": "Schatten", "color": "#4a5158"},
    "blur": {"label": "Unscharfer Bereich", "color": "#6a6f74"},
}

# Ebene 4 --------------------------------------------------------------------
ROI_CLASSES = {
    "roi_interest": {"label": "Interessiert", "color": "#67e6f1"},
    "roi_ignore": {"label": "Nicht interessiert / ignorieren", "color": "#3a4149"},
    "roi_uncertain": {"label": "ROI unsicher", "color": "#c9a94f"},
}

# Metadaten je Polygon -------------------------------------------------------
# Ohne diese Angaben laesst sich spaeter nicht mehr sagen, was Handarbeit war
# und was ein uebernommener Vorschlag — genau die Unterscheidung, die Refinement
# und Active Learning brauchen.
CERTAINTY_LEVELS = {
    "certain": {"label": "Sicher"},
    "uncertain": {"label": "Unsicher"},
    "partially_occluded": {"label": "Teilweise verdeckt"},
}
ORIGIN_LEVELS = {
    "manual": {"label": "Von Hand gesetzt"},
    "model_proposal": {"label": "Automatisch vorgeschlagen"},
    "manual_corrected": {"label": "Vorschlag von Hand korrigiert"},
    "human_confirmed": {"label": "Vom Menschen bestätigt"},
}

LAYERS = {
    "core": {"label": "Kernklasse", "classes": CORE_CLASSES, "exclusive": True},
    "obstacle": {"label": "Hindernis", "classes": OBSTACLE_CLASSES, "exclusive": False},
    "zone": {"label": "Problemzone", "classes": ZONE_CLASSES, "exclusive": False},
    "roi": {"label": "Auswertungsbereich", "classes": ROI_CLASSES, "exclusive": False},
}

CLASS_LAYER = {
    **{key: "core" for key in CORE_CLASSES},
    **{key: "obstacle" for key in OBSTACLE_CLASSES},
    **{key: "zone" for key in ZONE_CLASSES},
    **{key: "roi" for key in ROI_CLASSES},
}
ALL_CLASSES = tuple(CLASS_LAYER)
# Nur Kernklassen landen in der Rastermaske; die Werte 0..4 sind damit belegt.
MASK_VALUES = {UNLABELLED["value"], *(item["value"] for item in CORE_CLASSES.values())}


def layer_of(class_id: str) -> str:
    if class_id not in CLASS_LAYER:
        raise ValueError(f"Unbekannte Labelklasse: {class_id}")
    return CLASS_LAYER[class_id]


def mask_value(class_id: str) -> int | None:
    """Maskenwert einer Kernklasse; None fuer alle anderen Ebenen."""
    return CORE_CLASSES[class_id]["value"] if class_id in CORE_CLASSES else None


def describe(class_id: str) -> dict:
    layer = layer_of(class_id)
    entry = LAYERS[layer]["classes"][class_id]
    return {
        "class_id": class_id,
        "layer": layer,
        "label": entry["label"],
        "color": entry["color"],
        "value": entry.get("value"),
        "description": entry.get("description", ""),
    }


def ontology_document() -> dict:
    """Vollstaendige Ontologie fuer API-Antworten und die Oberflaeche.

    Die Oberflaeche baut ihre Auswahl daraus, damit Klassenliste und Farben
    nicht an zwei Stellen gepflegt werden muessen.
    """
    return {
        "schema_version": LABEL_SCHEMA_VERSION,
        "unlabelled": UNLABELLED,
        "layers": [
            {
                "layer": key,
                "label": entry["label"],
                "exclusive": entry["exclusive"],
                "classes": [describe(class_id) for class_id in entry["classes"]],
            }
            for key, entry in LAYERS.items()
        ],
        "certainty": [{"value": key, **item} for key, item in CERTAINTY_LEVELS.items()],
        "origin": [{"value": key, **item} for key, item in ORIGIN_LEVELS.items()],
        "notes": [
            "Maskenwert 0 heisst 'nicht markiert' und wird im Training ignoriert, nicht 'nicht befahrbar'.",
            "Hindernisse und Problemzonen tragen keinen Maskenwert: sie erklaeren eine Fläche, statt sie zu ersetzen.",
            "Der Auswertungsbereich schneidet das Bild nicht zu — das Original bleibt vollständig erhalten.",
        ],
    }
