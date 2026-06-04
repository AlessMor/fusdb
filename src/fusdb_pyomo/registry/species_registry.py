"""Species registry."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class SpeciesSpec:
    """Metadata for one ion species."""

    key: str
    full_name: str = ""
    atomic_symbol: str = ""
    atomic_number: int | None = None
    atomic_mass: float | None = None
    isotopic_mass_u: float | None = None


class SpeciesRegistry:
    """Registry of isotope/species metadata."""

    def __init__(self, specs: Mapping[str, SpeciesSpec]) -> None:
        self._specs = MappingProxyType(dict(specs))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SpeciesRegistry":
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        specs: dict[str, SpeciesSpec] = {}
        for key, entry in raw.items():
            entry = entry or {}
            specs[str(key)] = SpeciesSpec(
                key=str(key),
                full_name=str(entry.get("full_name", "")),
                atomic_symbol=str(entry.get("atomic_symbol", "")),
                atomic_number=entry.get("atomic_number"),
                atomic_mass=entry.get("atomic_mass"),
                isotopic_mass_u=entry.get("isotopic_mass_u"),
            )
        return cls(specs)

    def get(self, key: str) -> SpeciesSpec:
        return self._specs[str(key)]

    def __getitem__(self, key: str) -> SpeciesSpec:
        return self.get(key)

    def __contains__(self, key: object) -> bool:
        return str(key) in self._specs

    def __iter__(self):
        return iter(self._specs.values())


_DEFAULT_PATH = Path(__file__).with_name("allowed_species.yaml")
SPECIES = SpeciesRegistry.from_yaml(_DEFAULT_PATH) if _DEFAULT_PATH.exists() else SpeciesRegistry({})
