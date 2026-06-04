"""Relation registry and relation module discovery."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable

from ..relation import REGISTERED_RELATIONS, Relation
from ..utils import normalize_tags
from .tag_registry import TAGS, TagRegistry
from .variable_registry import VARIABLES, VariableRegistry


class RelationRegistry:
    """Registry of decorated relations."""

    def __init__(self, relations: Iterable[Relation] = ()) -> None:
        by_name: dict[str, Relation] = {}
        for rel in relations:
            if rel.name in by_name:
                raise ValueError(f"Duplicate relation {rel.name!r}.")
            by_name[rel.name] = rel
        self._relations = MappingProxyType(by_name)

    @classmethod
    def discover(cls) -> "RelationRegistry":
        """Import all modules under ``fusdb_pyomo.relations`` and collect decorators."""
        package_root = Path(__file__).resolve().parents[1]
        relations_root = package_root / "relations"
        for path in sorted(relations_root.rglob("*.py")):
            if path.name == "__init__.py" or "__pycache__" in path.parts:
                continue
            rel = path.relative_to(package_root).with_suffix("")
            module = f"fusdb_pyomo.{'.'.join(rel.parts)}"
            importlib.import_module(module)
        return cls(REGISTERED_RELATIONS.values())

    def get(self, name: str) -> Relation:
        """Return one relation by name."""
        return self._relations[str(name)]

    def get_filtered_relations(
        self,
        *,
        names: Iterable[str] | None = None,
        tags: Iterable[str] | None = None,
        variables: Iterable[str] | None = None,
        exclude: Iterable[str] | None = None,
        order: Iterable[str] | None = None,
        variable_registry: VariableRegistry = VARIABLES,
        tag_registry: TagRegistry = TAGS,
    ) -> tuple[Relation, ...]:
        """Return selected relations.

        Selection order is deterministic: tag/default filtering, explicit includes,
        explicit excludes, then explicit ordering. Exclusion always wins.
        """
        exclude_set = {str(item) for item in (exclude or ())}
        include_names = [str(item) for item in (names or ())]
        reactor_tags = normalize_tags(tags)

        selected: list[Relation] = []
        for rel in self._relations.values():
            if tag_registry.relation_matches(rel.tags, reactor_tags):
                selected.append(rel)

        # Variable default_relation limits duplicate output producers when present.
        allowed_by_output: dict[str, set[str]] = {}
        for spec in variable_registry:
            if spec.default_relation:
                allowed_by_output[spec.name] = set(spec.default_relation)
        if allowed_by_output:
            filtered: list[Relation] = []
            for rel in selected:
                defaults = [allowed_by_output[out] for out in rel.output_names if out in allowed_by_output]
                if defaults and rel.name not in set().union(*defaults):
                    continue
                filtered.append(rel)
            selected = filtered

        selected_by_name = {rel.name: rel for rel in selected}
        for name in include_names:
            if name not in self._relations:
                raise KeyError(f"Unknown relation {name!r}.")
            selected_by_name.setdefault(name, self._relations[name])
        for name in exclude_set:
            selected_by_name.pop(name, None)

        if variables is not None:
            needed = {variable_registry.resolve(name) for name in variables}
            selected_by_name = {
                name: rel
                for name, rel in selected_by_name.items()
                if any(var in needed for var in rel.variables)
            }

        selected = list(selected_by_name.values())
        if order:
            ordered: list[Relation] = []
            remaining = {rel.name: rel for rel in selected}
            for name in order:
                text = str(name)
                if text not in remaining:
                    raise ValueError(f"relations.order references inactive relation {text!r}.")
                ordered.append(remaining.pop(text))
            ordered.extend(remaining.values())
            selected = ordered
        return tuple(selected)

    def producers(self, variable: str, *, variable_registry: VariableRegistry = VARIABLES) -> tuple[Relation, ...]:
        """Return relations that declare ``variable`` as an output."""
        name = variable_registry.resolve(variable)
        return tuple(rel for rel in self._relations.values() if name in rel.output_names)

    def __contains__(self, name: object) -> bool:
        return str(name) in self._relations

    def __iter__(self):
        return iter(self._relations.values())

    def __len__(self) -> int:
        return len(self._relations)


class LazyRelationRegistry:
    """Tiny lazy proxy so relation modules are imported only when needed."""

    def __init__(self) -> None:
        self._registry: RelationRegistry | None = None

    def _get(self) -> RelationRegistry:
        if self._registry is None:
            self._registry = RelationRegistry.discover()
        return self._registry

    def __getattr__(self, name: str) -> Any:
        return getattr(self._get(), name)

    def __iter__(self):
        return iter(self._get())

    def __len__(self) -> int:
        return len(self._get())

    def __contains__(self, name: object) -> bool:
        return name in self._get()


RELATIONS = LazyRelationRegistry()
