"""Project-anchored locations for saved models and datasets.

Relative paths such as ``models/viola_jones/cascade.xml`` used to be resolved
against the folder the app was started from, so starting it from anywhere else
"lost" every saved model. Everything now resolves against the project folder.
"""

from __future__ import annotations

from pathlib import Path

# forensics_app/paths.py -> forensics_app/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATASETS_DIR = PROJECT_ROOT / "datasets"


def resolve_path(path: str | Path) -> Path:
    """Return an absolute path for a user-supplied or default path.

    Absolute paths are returned unchanged. A relative path that exists from the
    current folder is kept (compatibility with launching from the project
    folder); any other relative path is resolved against the project folder.
    """
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    return PROJECT_ROOT / candidate
