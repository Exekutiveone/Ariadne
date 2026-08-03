"""Ablageort für laufenden Zustand (Jobstatus, Checkpoints).

Liegt bewusst ausserhalb des Projektordners: das Repo liegt in OneDrive und
Sync-Sperren würden laufende Jobs abbrechen (siehe CLAUDE.md). Über
ARIADNE_RUNTIME_DIR explizit änderbar.
"""

import os
import tempfile
from pathlib import Path


def runtime_root() -> Path:
    configured = os.getenv("ARIADNE_RUNTIME_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    local_app_data = os.getenv("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "Ariadne" / "runtime"
    return Path(tempfile.gettempdir()) / "ariadne-runtime"
