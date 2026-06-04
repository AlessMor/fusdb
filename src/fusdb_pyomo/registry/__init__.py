"""Registry exports and physical constants."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .tag_registry import TAGS, TagRegistry
from .variable_registry import VARIABLES, VariableRegistry, VariableSpec
from .relation_registry import RELATIONS, RelationRegistry
from .species_registry import SPECIES, SpeciesRegistry, SpeciesSpec
from .unitregistry import convert_value, unit_registry
from .reactivity_config import REACTIVITY_TABLES, ReactivityTableConfig

_CONSTANTS_PATH = Path(__file__).with_name("constants.yaml")
with _CONSTANTS_PATH.open("r", encoding="utf-8") as handle:
    _CONSTANTS = yaml.safe_load(handle) or {}

globals().update(_CONSTANTS)


def __getattr__(name: str) -> Any:
    try:
        return _CONSTANTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
