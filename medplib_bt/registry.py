"""Minimal name -> builder registries (models, datasets), mmdet/detectron style.

A config dict selects a builder with its ``type`` key::

    @MODELS.register("medplib_moe")
    def build_medplib(cfg, **kwargs): ...

    tokenizer, model = MODELS.build(cfg["model"], dtype=torch.bfloat16)
"""
from typing import Any, Callable, Dict


class Registry:
    def __init__(self, name: str):
        self.name = name
        self._entries: Dict[str, Callable] = {}

    def register(self, key: str):
        def deco(fn):
            if key in self._entries:
                raise KeyError(f"{self.name} registry already has '{key}'")
            self._entries[key] = fn
            return fn
        return deco

    def get(self, key: str) -> Callable:
        if key not in self._entries:
            raise KeyError(f"unknown {self.name} type '{key}'; registered: {sorted(self._entries)}")
        return self._entries[key]

    def build(self, cfg: Dict[str, Any], **kwargs):
        cfg = dict(cfg)
        typ = cfg.pop("type")
        return self.get(typ)(cfg, **kwargs)

    def keys(self):
        return sorted(self._entries)


MODELS = Registry("model")
DATASETS = Registry("dataset")
