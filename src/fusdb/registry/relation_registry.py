"""Relation registry and relation module discovery."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ..relation import REGISTERED_RELATIONS, Relation, canonicalize_relation
from ..utils import normalize_tags
from .tag_registry import TAGS, TagRegistry
from .variable_registry import VARIABLES, VariableRegistry

logger = logging.getLogger(__name__)


class RelationRegistry:
    """Registry of decorated relations."""

    def __init__(
        self,
        relations: Iterable[Relation] = (),
        *,
        variable_registry: VariableRegistry = VARIABLES,
        tag_registry: TagRegistry = TAGS,
    ) -> None:
        by_name: dict[str, Relation] = {}
        by_function: dict[str, Relation] = {}
        owners: dict[str, Relation] = {}
        for rel in relations:
            for identifier, kind in ((rel.name, "name"), (rel.function_name, "function name")):
                owner = owners.get(identifier)
                if owner is not None and owner is not rel:
                    raise ValueError(
                        f"Relation identifier {identifier!r} ({kind} of {rel.name!r}) collides with "
                        f"relation {owner.name!r}; relation names and function names must be unique."
                    )
                owners[identifier] = rel
            unknown = tuple(tag for tag in normalize_tags(rel.tags) if tag not in tag_registry.allowed)
            if unknown:
                raise ValueError(
                    f"Relation {rel.name!r} declares tag(s) "
                    f"{', '.join(repr(tag) for tag in unknown)} that are not in allowed_tags.yaml. "
                    "Add each to a group there: a selection group (device, confinement_mode, "
                    "internal) if it should filter which reactors see the relation, or "
                    "'descriptive' if it is documentation or a code-read role marker."
                )
            canonical = canonicalize_relation(rel, variable_registry)
            by_name[rel.name] = canonical
            by_function[rel.function_name] = canonical
        self._relations = MappingProxyType(by_name)
        self._by_function = MappingProxyType(by_function)

    @classmethod
    def discover(cls, *, variable_registry: VariableRegistry = VARIABLES) -> "RelationRegistry":
        package_root = Path(__file__).resolve().parents[1]
        relations_root = package_root / "relations"
        for path in sorted(relations_root.rglob("*.py")):
            if path.name == "__init__.py" or "__pycache__" in path.parts:
                continue
            rel = path.relative_to(package_root).with_suffix("")
            importlib.import_module(f"fusdb.{'.'.join(rel.parts)}")
        return cls(REGISTERED_RELATIONS.values(), variable_registry=variable_registry)

    def get(self, name: str) -> Relation:
        return self._resolve(name)

    def _resolve(self, name: str) -> Relation:
        text = str(name)
        if text in self._relations:
            return self._relations[text]
        if text in self._by_function:
            return self._by_function[text]
        raise KeyError(f"Unknown relation {name!r}.")

    def _canonical_name(self, name: Any) -> str:
        try:
            return self._resolve(name).name
        except KeyError:
            return str(name)

    def _effective_default_relations(
        self,
        variable_registry: VariableRegistry,
        overrides: Mapping[str, Iterable[str]] | None,
    ) -> tuple[dict[str, set[str]], dict[str, tuple[str, ...]]]:
        """Return effective provider gates and canonical scenario overrides.

        A scenario override replaces the registry preference for that variable.
        Selecting a multi-output relation is atomic: unless another output has
        its own explicit override, that relation also replaces the registry
        default gate for every side output it produces. Explicit empty overrides
        deliberately remove the gate and therefore opt that side output out of
        atomic propagation.
        """
        allowed: dict[str, set[str]] = {
            spec.name: {self._canonical_name(name) for name in spec.default_relation}
            for spec in variable_registry
            if spec.default_relation
        }
        explicit: dict[str, tuple[str, ...]] = {}
        for raw_name, raw_relations in (overrides or {}).items():
            name = variable_registry.resolve(raw_name)
            relations = tuple(self._canonical_name(item) for item in raw_relations)
            explicit[name] = relations
            if relations:
                allowed[name] = set(relations)
            else:
                allowed.pop(name, None)

        for name, relation_names in explicit.items():
            for relation_name in relation_names:
                rel = self._resolve(relation_name)
                if name not in rel.output_names:
                    raise ValueError(
                        f"Variable {name!r} selects default_relation {relation_name!r}, "
                        f"but that relation does not produce {name!r}."
                    )
                for out in rel.output_names:
                    if out == name:
                        continue
                    if out in explicit:
                        other = explicit[out]
                        if other and rel.name not in other:
                            raise ValueError(
                                f"Relation {rel.name!r} is selected for {name!r} and also produces {out!r}, "
                                f"but {out!r} explicitly selects {list(other)!r}. Multi-output relations "
                                "are atomic; choose compatible providers."
                            )
                        # Empty explicit override means no provider gate for this side output.
                        continue
                    allowed[out] = {rel.name}
        return allowed, explicit

    def get_filtered_relations(
        self,
        *,
        names: Iterable[str] | None = None,
        tags: Iterable[str] | None = None,
        variables: Iterable[str] | None = None,
        exclude: Iterable[str] | None = None,
        order: Iterable[str] | None = None,
        default_relations: Mapping[str, Iterable[str]] | None = None,
        variable_registry: VariableRegistry = VARIABLES,
        tag_registry: TagRegistry = TAGS,
    ) -> tuple[Relation, ...]:
        """Return selected relations after tag/default/include/exclude/order filtering."""
        exclude_set = {self._canonical_name(item) for item in (exclude or ())}
        include_names = [str(item) for item in (names or ())]
        reactor_tags = normalize_tags(tags)

        selected = [rel for rel in self._relations.values() if tag_registry.relation_matches(rel.tags, reactor_tags)]
        allowed_by_output, _explicit_overrides = self._effective_default_relations(
            variable_registry, default_relations
        )
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
            rel = self._resolve(name)
            if rel.name not in selected_by_name:
                overridden = [
                    out for out in rel.output_names
                    if out in allowed_by_output and rel.name not in allowed_by_output[out]
                ]
                if overridden:
                    logger.warning(
                        "Included relation %r is not a preferred provider of %s; it overrides "
                        "the effective default_relation set (%s). This is allowed, but the "
                        "preferred provider(s) remain active alongside it unless excluded.",
                        rel.name,
                        ", ".join(repr(out) for out in overridden),
                        "; ".join(f"{out}: {sorted(allowed_by_output[out])}" for out in overridden),
                    )
            selected_by_name.setdefault(rel.name, rel)

        for item in (exclude or ()):
            try:
                canonical = self._resolve(item).name
            except KeyError:
                logger.warning(
                    "relations.exclude names %r, which is not a known relation "
                    "(neither a relation name nor a decorated function name); nothing excluded.",
                    str(item),
                )
                continue
            if canonical not in selected_by_name:
                logger.warning(
                    "relations.exclude names %r, which was not active anyway "
                    "(not selected by tags/default_relation, or already excluded); nothing to remove.",
                    canonical,
                )
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
                canonical = self._canonical_name(name)
                if canonical not in remaining:
                    raise ValueError(f"relations.order references inactive relation {str(name)!r}.")
                ordered.append(remaining.pop(canonical))
            ordered.extend(remaining.values())
            selected = ordered
        return tuple(selected)

    def producers(self, variable: str, *, variable_registry: VariableRegistry = VARIABLES) -> tuple[Relation, ...]:
        name = variable_registry.resolve(variable)
        return tuple(rel for rel in self._relations.values() if name in rel.output_names)

    def __contains__(self, name: object) -> bool:
        text = str(name)
        return text in self._relations or text in self._by_function

    def __iter__(self):
        return iter(self._relations.values())

    def __len__(self) -> int:
        return len(self._relations)


class LazyRelationRegistry:
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
