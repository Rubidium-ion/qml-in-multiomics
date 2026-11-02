"""Modality registry: name -> adapter class.

Kept deliberately tiny. Adapters self-register on import of qmlmo.modalities,
so a config file can name a modality as a string without importing anything.
"""
from __future__ import annotations

from typing import Any

_REGISTRY: dict[str, type] = {}


def register_modality(cls: type) -> type:
    """Class decorator. Requires a non-empty, unique `.name`."""
    name = getattr(cls, "name", None)
    if not name or name == "base":
        raise ValueError(f"{cls.__name__} must define a unique `name`")
    if name in _REGISTRY and _REGISTRY[name] is not cls:
        raise ValueError(f"modality {name!r} already registered by {_REGISTRY[name].__name__}")
    _REGISTRY[name] = cls
    return cls


def get_modality(name: str) -> type:
    if name not in _REGISTRY:
        raise KeyError(f"unknown modality {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_modalities() -> list[str]:
    return sorted(_REGISTRY)


def build_modality(name: str, **params: Any):
    return get_modality(name)(**params)
