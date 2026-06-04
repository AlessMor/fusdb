"""Variable registry loaded from ``variables.yaml``."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from ..utils import parse_constraint_specs, parse_domain, unique_preserve_order, validate_solver_domain


@dataclass(frozen=True, slots=True)
class VariableSpec:
    """Registry metadata for one canonical variable."""

    name: str
    aliases: tuple[str, ...] = ()
    unit: str = "dimensionless"
    shape: int = 0
    domain: tuple[float | None, float | None, bool, bool] = (None, None, True, True)
    solver_domain: tuple[float | None, float | None, bool, bool] = (None, None, True, True)
    constraints: tuple[tuple[str, bool], ...] = ()
    description: str = ""
    rel_tol: float = 0.01
    average_variable: str | None = None
    default_relation: tuple[str, ...] = ()


class VariableRegistry:
    """Registry of allowed variables and aliases.

    The registry only stores metadata. Values belong to ``Variable`` objects.
    """

    def __init__(self, specs: Iterable[VariableSpec], *, rel_tol_default: float = 0.01) -> None:
        self.rel_tol_default = float(rel_tol_default)
        by_name: dict[str, VariableSpec] = {}
        alias_to_name: dict[str, str] = {}
        for spec in specs:
            if spec.name in by_name:
                raise ValueError(f"Duplicate variable {spec.name!r}.")
            by_name[spec.name] = spec
            alias_to_name[spec.name] = spec.name
            for alias in spec.aliases:
                if alias in alias_to_name:
                    raise ValueError(f"Alias {alias!r} is ambiguous.")
                alias_to_name[alias] = spec.name
        self._specs = MappingProxyType(by_name)
        self._alias_to_name = MappingProxyType(alias_to_name)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "VariableRegistry":
        """Load a registry from YAML."""
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        defaults = raw.pop("defaults", {}) if isinstance(raw, dict) else {}
        rel_tol_default = float(defaults.get("rel_tol", 0.01)) if isinstance(defaults, Mapping) else 0.01
        specs: list[VariableSpec] = []
        for name, entry in raw.items():
            if not isinstance(entry, Mapping):
                raise TypeError(f"Variable {name!r} must be a mapping.")
            aliases = unique_preserve_order(entry.get("aliases", ()) or ())
            unit = str(entry.get("default_unit", entry.get("unit", "dimensionless")))
            shape = int(entry.get("shape", entry.get("ndim", 0)) or 0)
            domain = parse_domain(entry.get("domain"))
            solver_domain = parse_domain(entry.get("solver_domain", entry.get("domain")))
            validate_solver_domain(str(name), domain, solver_domain)
            constraints = parse_constraint_specs(entry.get("constraints"))
            description = entry.get("description", "")
            if isinstance(description, list):
                description = " ".join(str(item) for item in description)
            rel_tol = float(entry.get("rel_tol", entry.get("rel_tol_defaultpervar", rel_tol_default)))
            default_relation = entry.get("default_relation", ()) or ()
            if isinstance(default_relation, str):
                default_relation = (default_relation,)
            average_variable = entry.get("average_variable")
            specs.append(
                VariableSpec(
                    name=str(name),
                    aliases=aliases,
                    unit=unit,
                    shape=shape,
                    domain=domain,
                    solver_domain=solver_domain,
                    constraints=constraints,
                    description=str(description),
                    rel_tol=rel_tol,
                    average_variable=None if average_variable is None else str(average_variable),
                    default_relation=tuple(str(item) for item in default_relation),
                )
            )
        return cls(specs, rel_tol_default=rel_tol_default)

    def resolve(self, name: str) -> str:
        """Resolve a canonical name or alias."""
        try:
            return self._alias_to_name[str(name)]
        except KeyError as exc:
            raise KeyError(f"Unknown variable {name!r}.") from exc

    def get(self, name: str) -> VariableSpec:
        """Return one variable spec by name or alias."""
        return self._specs[self.resolve(name)]

    def __getitem__(self, name: str) -> VariableSpec:
        return self.get(name)

    def __contains__(self, name: object) -> bool:
        return str(name) in self._alias_to_name

    def __iter__(self):
        return iter(self._specs.values())

    def __len__(self) -> int:
        return len(self._specs)


_DEFAULT_PATH = Path(__file__).with_name("variables.yaml")
VARIABLES = VariableRegistry.from_yaml(_DEFAULT_PATH) if _DEFAULT_PATH.exists() else VariableRegistry(())
