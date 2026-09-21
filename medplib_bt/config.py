"""YAML experiment configs.

* ``model:`` / ``data:`` values that are strings are treated as paths to the model / data
  config files and replaced by their contents;
* ``${paths.<key>}`` placeholders are filled from configs/paths.yaml (see paths.py);
* ``--set a.b.c=value`` command line overrides (value parsed as YAML) are applied last.
"""
import copy
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

from . import paths

_PLACEHOLDER = re.compile(r"\$\{paths\.([a-zA-Z0-9_]+)\}")


def _interpolate(obj):
    if isinstance(obj, str):
        return _PLACEHOLDER.sub(lambda m: str(paths.get(m.group(1))), obj)
    if isinstance(obj, list):
        return [_interpolate(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _interpolate(v) for k, v in obj.items()}
    return obj


def load_yaml(path) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = paths.PROJECT_ROOT / path
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_file"] = str(path)
    return cfg


def _deep_update(base: Dict, upd: Dict) -> Dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def set_by_path(cfg: Dict, dotted: str, value):
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def apply_overrides(cfg: Dict, overrides: Iterable[str]) -> Dict:
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"override must look like key.sub=value, got {ov!r}")
        k, v = ov.split("=", 1)
        set_by_path(cfg, k.strip(), yaml.safe_load(v))
    return cfg


def load_config(path, overrides: Iterable[str] = ()) -> Dict[str, Any]:
    """Load an experiment config, resolve model/data references, interpolate paths, apply
    overrides. Nested overrides for the referenced configs use their key, e.g.
    ``--set model.lora.r=8`` or ``--set data.task_mix.joint=1.0``."""
    cfg = load_yaml(path)
    for ref in ("model", "data"):
        if isinstance(cfg.get(ref), str):
            cfg[ref] = load_yaml(cfg[ref])
        elif isinstance(cfg.get(ref), dict) and isinstance(cfg[ref].get("_base"), str):
            base = load_yaml(cfg[ref].pop("_base"))
            cfg[ref] = _deep_update(base, cfg[ref])
    cfg = apply_overrides(cfg, overrides)
    return _interpolate(cfg)


def dump(cfg: Dict[str, Any]) -> str:
    return yaml.safe_dump(copy.deepcopy(cfg), sort_keys=False, allow_unicode=True)
