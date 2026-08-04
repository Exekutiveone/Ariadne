"""Kleine, reversible Projektliste fuer frei angelegte Problemzonen-Gründe.

Sie ergänzt die fest verdrahteten Standardgründe, ohne die Label-Ontologie oder
deren Maskenwerte zu verändern. Nutzungen werden pro Projekt gezählt, damit die
häufigsten Begriffe im Labeler zuerst erscheinen.
"""
import json
import os
from pathlib import Path

DEFAULT_REASONS = [
    {"value": "hohes_gras", "label": "Hohes Gras / Bewuchs", "uses": 0},
    {"value": "laub_teilverdeckt", "label": "Laub oder teilweise verdeckt", "uses": 0},
    {"value": "unebenheit_stufe", "label": "Unebenheit oder Stufe", "uses": 0},
    {"value": "engstelle", "label": "Mögliche Engstelle", "uses": 0},
    {"value": "sicht_unsicherheit", "label": "Sicht oder Einschätzung unsicher", "uses": 0},
    {"value": "sonstiges", "label": "Sonstiges — siehe Notiz", "uses": 0},
]

def _path(root: Path) -> Path: return root / "problem_reasons.json"

def list_reasons(root: Path):
    try: stored = json.loads(_path(root).read_text(encoding="utf-8"))
    except (OSError, ValueError): stored = []
    by_value = {item["value"]: item for item in DEFAULT_REASONS}
    for item in stored:
        if isinstance(item, dict) and item.get("value") and item.get("label"):
            by_value[item["value"]] = {"value": item["value"], "label": item["label"], "uses": int(item.get("uses", 0))}
    return sorted(by_value.values(), key=lambda item: (-item["uses"], item["label"].casefold()))

def add_reason(root: Path, label: str):
    clean = " ".join(label.split()).strip()
    if not clean: raise ValueError("Ein Grund darf nicht leer sein")
    if len(clean) > 80: raise ValueError("Ein Grund darf höchstens 80 Zeichen haben")
    items = list_reasons(root)
    if any(item["label"].casefold() == clean.casefold() for item in items):
        raise ValueError("Dieser Grund existiert bereits")
    value = "custom_" + "".join(ch.lower() if ch.isalnum() else "_" for ch in clean).strip("_")[:48]
    index = 2
    existing = {item["value"] for item in items}
    base = value
    while value in existing:
        value = f"{base}_{index}"; index += 1
    custom = [item for item in items if item["value"].startswith("custom_")]
    custom.append({"value": value, "label": clean, "uses": 0})
    _write(root, custom)
    return {"value": value, "label": clean, "uses": 0}

def use_reason(root: Path, value: str):
    items = list_reasons(root)
    for item in items:
        if item["value"] == value: item["uses"] += 1
    _write(root, items)

def _write(root: Path, items: list):
    path = _path(root); tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)
