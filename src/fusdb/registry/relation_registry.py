"""Relation registry and relation module discovery."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable

from ..relation import REGISTERED_RELATIONS, Relation, canonicalize_relation
from ..utils import normalize_tags
from .tag_registry import TAGS, TagRegistry
from .variable_registry import VARIABLES, VariableRegistry

logger = logging.getLogger(__name__)


class RelationRegistry:
    """Registry of decorated relations.

    Relations are validated against ``variable_registry`` at build time, so
    alias-degenerate relations (declared outputs that resolve to one of their own
    inputs) are rejected here rather than silently dropped when a RelationSystem
    is later compiled.
    """

    def __init__(
        self,
        relations: Iterable[Relation] = (),
        *,
        variable_registry: VariableRegistry = VARIABLES,
        tag_registry: TagRegistry = TAGS,
    ) -> None:
        by_name: dict[str, Relation] = {}
        by_function: dict[str, Relation] = {}
        # A relation is addressable by either its user-facing ``name`` or its
        # decorated ``function_name``.  For that dual addressing to be
        # unambiguous, every identifier must resolve to exactly one relation:
        # names unique, function names unique, and no name colliding with a
        # different relation's function name.  (A relation decorated without an
        # explicit name has ``name == function_name``; that is the same owner
        # registering both identifiers, not a collision.)
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
            # Every relation tag must be declared in allowed_tags.yaml.  An
            # undeclared tag is silently ignored by ``relation_matches``, so a
            # typo (``profile_shapes`` for ``profile_shape``) does not fail --
            # it makes the relation globally active, and the damage shows up
            # later as physics drift on an unrelated reactor.  Declared-but-
            # non-filtering tags belong in the ``descriptive`` group.
            unknown = tuple(tag for tag in normalize_tags(rel.tags) if tag not in tag_registry.allowed)
            if unknown:
                raise ValueError(
                    f"Relation {rel.name!r} declares tag(s) "
                    f"{', '.join(repr(tag) for tag in unknown)} that are not in allowed_tags.yaml. "
                    "Add each to a group there: a selection group (device, confinement_mode, "
                    "internal) if it should filter which reactors see the relation, or "
                    "'descriptive' if it is documentation or a code-read role marker."
                )
            # Validate against the variable registry (rejects alias-degenerate
            # relations) and store the canonicalized relation, so every relation
            # leaving the registry already uses canonical variable names.
            # Name-based filtering is unaffected: canonicalization changes
            # variable names, never ``rel.name``.
            canonical = canonicalize_relation(rel, variable_registry)
            by_name[rel.name] = canonical
            by_function[rel.function_name] = canonical
        self._relations = MappingProxyType(by_name)
        self._by_function = MappingProxyType(by_function)

    @classmethod
    def discover(cls, *, variable_registry: VariableRegistry = VARIABLES) -> "RelationRegistry":
        """Import all modules under ``fusdb.relations`` and collect decorators."""
        package_root = Path(__file__).resolve().parents[1]
        relations_root = package_root / "relations"
        for path in sorted(relations_root.rglob("*.py")):
            if path.name == "__init__.py" or "__pycache__" in path.parts:
                continue
            rel = path.relative_to(package_root).with_suffix("")
            module = f"fusdb.{'.'.join(rel.parts)}"
            importlib.import_module(module)
        return cls(REGISTERED_RELATIONS.values(), variable_registry=variable_registry)

    def get(self, name: str) -> Relation:
        """Return one relation by name or decorated function name."""
        return self._resolve(name)

    def _resolve(self, name: str) -> Relation:
        """Resolve a relation by user-facing name or decorated function name."""
        text = str(name)
        if text in self._relations:
            return self._relations[text]
        if text in self._by_function:
            return self._by_function[text]
        raise KeyError(f"Unknown relation {name!r}.")

    def _canonical_name(self, name: Any) -> str:
        """Return the user-facing name for an identifier (name or function name).

        Unknown identifiers are returned unchanged so callers keep their own
        not-found handling (``exclude`` silently ignores misses; ``order``
        raises an inactive-relation error).
        """
        try:
            return self._resolve(name).name
        except KeyError:
            return str(name)

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
        exclude_set = {self._canonical_name(item) for item in (exclude or ())}
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
            rel = self._resolve(name)
            # An include that is absent from a ``default_relation`` list for one
            # of its own outputs is overriding the registry's preferred provider
            # set for that variable.  That is a legitimate thing to do -- it is
            # how a fixture selects a convention -- but it is a modelling
            # decision, so it must not be silent.
            if rel.name not in selected_by_name:
                overridden = [
                    out for out in rel.output_names
                    if out in allowed_by_output and rel.name not in allowed_by_output[out]
                ]
                if overridden:
                    logger.warning(
                        "Included relation %r is not a preferred provider of %s; it overrides "
                        "the default_relation set declared in variables.yaml (%s). This is "
                        "allowed, but the preferred provider(s) remain active alongside it "
                        "unless excluded.",
                        rel.name,
                        ", ".join(repr(out) for out in overridden),
                        "; ".join(
                            f"{out}: {sorted(allowed_by_output[out])}" for out in overridden
                        ),
                    )
            selected_by_name.setdefault(rel.name, rel)
        # Exclusion is silent by design when it matches nothing, which hides both
        # a misspelled name and a relation that some earlier gate had already
        # dropped -- in either case the author believes they removed something
        # and did not.
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
        """Return relations that declare ``variable`` as an output."""
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
