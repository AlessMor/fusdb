"""Species registry.

``species.yaml`` nests isotopes under their element; this module
flattens that at import into a single dict of fully-resolved specs, so a lookup
stays one ``dict`` access with every inherited field already merged.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class SpeciesSpec:
    """Metadata for one species -- an element, or one isotope of an element.

    Isotope specs are resolved: every field the isotope did not state has
    already been inherited from its element.
    """

    key: str
    element: str
    aliases: tuple[str, ...] = ()
    full_name: str = ""
    atomic_number: int | None = None
    atomic_mass: float | None = None
    isotopic_mass_u: float | None = None
    atomic_data: tuple[str, ...] = ()

    @property
    def symbol(self) -> str:
        """The canonical spelling of this species (its registry key)."""
        return self.key

    @property
    def is_element(self) -> bool:
        """Whether this is the element row rather than one of its isotopes."""
        return self.key == self.element


def _spec_from_entry(key: str, entry: Mapping[str, Any], element: str) -> SpeciesSpec:
    symbol = entry.get("symbol", key)
    aliases = (symbol,) if isinstance(symbol, str) else tuple(str(item) for item in symbol)
    if aliases[0] != key:
        raise ValueError(f"Species {key!r} must list its own key first in `symbol`, got {aliases[0]!r}.")
    return SpeciesSpec(
        key=key,
        element=element,
        aliases=aliases,
        full_name=str(entry.get("full_name", "")),
        atomic_number=entry.get("atomic_number"),
        atomic_mass=entry.get("atomic_mass"),
        isotopic_mass_u=entry.get("isotopic_mass_u"),
        atomic_data=tuple(str(item) for item in entry.get("atomic_data", ()) or ()),
    )


class SpeciesRegistry:
    """Registry of element/isotope metadata, flattened and alias-resolved."""

    def __init__(self, specs: Mapping[str, SpeciesSpec]) -> None:
        self._specs = MappingProxyType(dict(specs))
        index: dict[str, SpeciesSpec] = {}
        for spec in self._specs.values():
            for alias in spec.aliases:
                if alias in index and index[alias] is not spec:
                    raise ValueError(f"Species alias {alias!r} is claimed by both {index[alias].key!r} and {spec.key!r}.")
                index[alias] = spec
        self._index = MappingProxyType(index)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SpeciesRegistry":
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        specs: dict[str, SpeciesSpec] = {}
        for key, entry in raw.items():
            entry = entry or {}
            element_key = str(key)
            element = _spec_from_entry(element_key, entry, element_key)
            specs[element_key] = element
            for isotope_key, isotope_entry in (entry.get("isotopes", {}) or {}).items():
                isotope_entry = isotope_entry or {}
                # Inherit the element's resolved fields, then let the isotope
                # override any of them.  `symbol` and `isotopes` are row
                # identity, not physical properties, so they never inherit --
                # were `symbol` inherited, T would answer to H.
                if str(isotope_key) in specs:
                    raise ValueError(f"Species {isotope_key!r} is declared twice.")
                stated = _spec_from_entry(str(isotope_key), isotope_entry, element_key)
                inherited = {
                    field: getattr(element, field)
                    for field in ("full_name", "atomic_number", "atomic_mass", "isotopic_mass_u", "atomic_data")
                    if field not in isotope_entry
                }
                specs[str(isotope_key)] = replace(stated, **inherited)
        return cls(specs)

    def get(self, key: str) -> SpeciesSpec:
        return self._index[str(key)]

    def __getitem__(self, key: str) -> SpeciesSpec:
        return self.get(key)

    def __contains__(self, key: object) -> bool:
        return str(key) in self._index

    def __iter__(self):
        return iter(self._specs.values())

    def elements(self):
        """The element rows, in registry order."""
        return tuple(spec for spec in self if spec.is_element)

    def with_atomic_data(self, stem: str) -> tuple[str, ...]:
        """Element symbols carrying the ``stem`` dataset, in registry order.

        ``stem`` is the ``{datatype}_{source}`` part of a dataset id, e.g.
        ``coolingcurve_radas_coronal``.  Isotopes are excluded: they inherit
        their element's atomic data, so listing them would double-count.
        """
        return tuple(spec.key for spec in self.elements() if stem in spec.atomic_data)


_DEFAULT_PATH = Path(__file__).with_name("species.yaml")
SPECIES = SpeciesRegistry.from_yaml(_DEFAULT_PATH) if _DEFAULT_PATH.exists() else SpeciesRegistry({})
