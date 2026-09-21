"""Resolve project paths (configs/paths.yaml + MEDPLIB_BT_* env overrides) and expose MedPLIB."""
import os
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CFG_FILE = PROJECT_ROOT / "configs" / "paths.yaml"


def _load():
    with open(_CFG_FILE) as f:
        cfg = yaml.safe_load(f) or {}
    out = {}
    for k, v in cfg.items():
        v = os.environ.get(f"MEDPLIB_BT_{k.upper()}", v)
        p = Path(str(v))
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        out[k] = p
    return out


PATHS = _load()


def get(key: str) -> Path:
    return PATHS[key]


def add_medplib_to_syspath():
    """Make `model.*`, `utils.*`, `datasets.*` from the vendored MedPLIB importable.

    MedPLIB has a top-level package called `datasets` that shadows HF datasets; nothing in
    this project imports HF datasets, so that is acceptable.
    """
    root = str(PATHS["medplib_root"])
    if root not in sys.path:
        sys.path.insert(0, root)
    return root
