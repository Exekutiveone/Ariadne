import numpy as np
import pytest
from pydantic import ValidationError

from backend.app.label_ontology import (
    ALL_CLASSES,
    CORE_CLASSES,
    MASK_VALUES,
    OBSTACLE_CLASSES,
    ROI_CLASSES,
    ZONE_CLASSES,
    layer_of,
    mask_value,
    ontology_document,
)
from backend.app.models import GroundTruthAnnotationInput, GroundTruthPolygon
from backend.app.path_masks import polygon_mask

SQUARE = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]


def test_the_four_core_classes_are_the_ones_we_agreed_on():
    assert set(CORE_CLASSES) == {"traversable", "restricted", "not_traversable", "unknown"}


def test_existing_mask_values_were_appended_to_not_renumbered():
    """Die Werte 1..3 stammen aus der Ontologie vor der Erweiterung.

    Waeren sie umnummeriert worden, haetten alle gespeicherten Rastermasken
    stillschweigend eine andere Bedeutung bekommen.
    """
    assert CORE_CLASSES["traversable"]["value"] == 1
    assert CORE_CLASSES["not_traversable"]["value"] == 2
    assert CORE_CLASSES["unknown"]["value"] == 3
    assert CORE_CLASSES["restricted"]["value"] == 4
    assert MASK_VALUES == {0, 1, 2, 3, 4}


def test_only_core_classes_carry_a_mask_value():
    # Hindernisse und Zonen erklaeren eine Flaeche, sie ersetzen sie nicht.
    for class_id in CORE_CLASSES:
        assert mask_value(class_id) is not None
    for class_id in (*OBSTACLE_CLASSES, *ZONE_CLASSES, *ROI_CLASSES):
        assert mask_value(class_id) is None


def test_every_class_belongs_to_exactly_one_layer():
    assert len(ALL_CLASSES) == len(set(ALL_CLASSES))
    for class_id in ALL_CLASSES:
        assert layer_of(class_id) in {"core", "obstacle", "zone", "roi"}
    with pytest.raises(ValueError, match="Unbekannte Labelklasse"):
        layer_of("gibt-es-nicht")


def test_the_ontology_document_carries_everything_the_ui_needs():
    document = ontology_document()
    layers = {item["layer"]: item for item in document["layers"]}
    assert layers["core"]["exclusive"] is True
    assert layers["obstacle"]["exclusive"] is False
    for entry in layers["core"]["classes"]:
        assert entry["label"] and entry["color"].startswith("#")
    assert {item["value"] for item in document["certainty"]} == {"certain", "uncertain", "partially_occluded"}
    assert any("nicht markiert" in note for note in document["notes"])


def test_a_polygon_without_metadata_still_means_what_it_meant_before():
    """Bestandsschutz fuer die 276 gespeicherten Ground-Truth-Dateien."""
    polygon = GroundTruthPolygon(points=SQUARE)
    assert polygon.class_id == "traversable"
    assert polygon.certainty == "certain"
    assert polygon.origin == "manual"
    assert polygon.hard_negative is False


def test_an_unknown_class_is_rejected_instead_of_stored():
    with pytest.raises(ValidationError, match="Unbekannte Labelklasse"):
        GroundTruthPolygon(points=SQUARE, class_id="baum")  # deutsch statt Schluessel


def test_obstacles_and_zones_do_not_become_path_training_data():
    """Ohne Klassenfilter waere jedes Baumpolygon ein Trainingsbeispiel fuer Weg."""
    record = {
        "polygons": [
            {"class_id": "tree", "points": SQUARE},
            {"class_id": "mud", "points": SQUARE},
            {"class_id": "not_traversable", "points": SQUARE},
        ]
    }
    assert polygon_mask(record, 40, 40).sum() == 0

    record["polygons"].append({"class_id": "traversable", "points": SQUARE})
    assert polygon_mask(record, 40, 40).sum() > 0


def test_polygons_from_before_the_ontology_still_count_as_path():
    # Kein class_id im gespeicherten Datensatz: damals gab es nur "befahrbar".
    legacy = {"polygons": [{"id": "path-1", "points": SQUARE}]}
    assert polygon_mask(legacy, 40, 40).sum() > 0


def test_the_roi_is_its_own_layer_and_cannot_be_mixed_in():
    payload = GroundTruthAnnotationInput(
        timestamp_ms=0,
        polygons=[GroundTruthPolygon(points=SQUARE, class_id="traversable")],
        roi=[GroundTruthPolygon(points=SQUARE, class_id="roi_ignore")],
    )
    assert payload.roi[0].class_id == "roi_ignore"

    with pytest.raises(ValidationError, match="gehören in das Feld 'roi'"):
        GroundTruthAnnotationInput(timestamp_ms=0, polygons=[GroundTruthPolygon(points=SQUARE, class_id="roi_ignore")])
    with pytest.raises(ValidationError, match="nur ROI-Klassen erlaubt"):
        GroundTruthAnnotationInput(timestamp_ms=0, roi=[GroundTruthPolygon(points=SQUARE, class_id="tree")])


def test_a_restricted_area_is_a_valid_mask_value():
    # Werte 0..4, damit "eingeschraenkt befahrbar" auch als Raster speicherbar ist.
    from backend.app.models import GroundTruthMask

    mask = GroundTruthMask(width=8, height=8, rle=[4, 32, 1, 32])
    assert np.array(mask.rle[::2]).max() == 4
