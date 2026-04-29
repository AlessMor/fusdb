"""Index-backed relation-variable graph used by RelationSystem."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from scipy.sparse import coo_matrix

from .registry import allowed_variable_ndim
from .variable_class import Variable


@dataclass
class SystemGraph:
    """Store relation-variable topology through dense indices.

    Args:
        relations: Ordered relation objects.
        variables: Variable objects indexed by dense variable id.
        relation_id_by_object: Relation object to dense relation id.
        variable_id_by_name: Canonical variable name to dense variable id.
        relation_variable_ids: Relation id to connected variable ids.
        variable_relation_ids: Variable id to connected relation ids.
        duplicate_relation_names: Relation-name duplicates encountered while building topology.
    """

    relations: list[object]
    variables: list[Variable]
    relation_id_by_object: dict[object, int]
    variable_id_by_name: dict[str, int]
    relation_variable_ids: tuple[tuple[int, ...], ...]
    variable_relation_ids: tuple[tuple[int, ...], ...]
    duplicate_relation_names: dict[str, list[int]]

    def variable_names(self) -> list[str]:
        """Return ordered canonical variable names.

        Returns:
            Ordered canonical variable names.
        """
        return [var.name for var in self.variables]

    def variable(self, name: str) -> Variable | None:
        """Return the Variable object for one canonical name.

        Args:
            name: Canonical variable name.

        Returns:
            Variable object or ``None`` when the name is unknown.
        """
        var_id = self.variable_id_by_name.get(name)
        if var_id is None:
            return None
        return self.variables[var_id]

    def variables_dict(self) -> dict[str, Variable]:
        """Return Variable objects keyed by canonical name.

        Returns:
            Variable mapping for every variable node in the graph.
        """
        return {var.name: var for var in self.variables}

    def has_variable(self, name: str) -> bool:
        """Return whether a variable node exists for one name.

        Args:
            name: Canonical variable name.

        Returns:
            ``True`` when the graph has a variable node for ``name``.
        """
        return name in self.variable_id_by_name

    def relation_variable_names(self, relation: object) -> tuple[str, ...]:
        """Return variable names connected to one relation.

        Args:
            relation: Relation object.

        Returns:
            Connected canonical variable names.
        """
        rel_id = self.relation_id_by_object.get(relation)
        if rel_id is None:
            return ()
        return tuple(
            self.variables[var_id].name
            for var_id in self.relation_variable_ids[rel_id]
        )

    def variable_relations(self, name: str) -> tuple[object, ...]:
        """Return relation objects connected to one variable.

        Args:
            name: Canonical variable name.

        Returns:
            Connected relation objects.
        """
        var_id = self.variable_id_by_name.get(name)
        if var_id is None:
            return ()
        return tuple(
            self.relations[rel_id]
            for rel_id in self.variable_relation_ids[var_id]
        )

    def variable_ndim(self, name: str) -> int:
        """Return variable dimensionality with registry fallback.

        Args:
            name: Canonical variable name.

        Returns:
            Variable dimensionality.
        """
        var_id = self.variable_id_by_name.get(name)
        if var_id is not None:
            return int(self.variables[var_id].ndim)
        return int(allowed_variable_ndim(name))

    def build_scalar_incidence(
        self,
        relations: Iterable[object],
        variables: Iterable[str],
        *,
        scalar_only: bool = True,
    ) -> tuple[list[object], list[str], coo_matrix]:
        """Build one relation-variable sparse incidence matrix.

        Args:
            relations: Candidate relation objects.
            variables: Candidate variable names.
            scalar_only: Exclude profile variables when ``True``.

        Returns:
            Tuple ``(relations, variables, matrix)`` with unused rows/columns removed.
        """
        # Normalize subsets through graph ids so output order follows SystemGraph order.
        relation_ids = [
            rel_id
            for relation in relations
            for rel_id in (self.relation_id_by_object.get(relation),)
            if rel_id is not None
        ]
        variable_ids = [
            var_id
            for name in dict.fromkeys(variables)
            for var_id in (self.variable_id_by_name.get(name),)
            if var_id is not None and (not scalar_only or self.variables[var_id].ndim == 0)
        ]
        if not relation_ids or not variable_ids:
            return [], [], coo_matrix((0, 0), dtype=float)

        # Keep only variables that actually appear in the active relation subset.
        variable_id_set = set(variable_ids)
        used_variable_ids: set[int] = set()
        active_relation_ids: list[int] = []
        for rel_id in relation_ids:
            linked = [
                var_id
                for var_id in self.relation_variable_ids[rel_id]
                if var_id in variable_id_set
            ]
            if not linked:
                continue
            active_relation_ids.append(rel_id)
            used_variable_ids.update(linked)
        active_variable_ids = [var_id for var_id in variable_ids if var_id in used_variable_ids]
        if not active_relation_ids or not active_variable_ids:
            return [], [], coo_matrix((0, 0), dtype=float)

        # Materialize the compact row/column matrix.
        col_by_var_id = {
            var_id: col_idx
            for col_idx, var_id in enumerate(active_variable_ids)
        }
        rows: list[int] = []
        cols: list[int] = []
        for row_idx, rel_id in enumerate(active_relation_ids):
            for var_id in dict.fromkeys(self.relation_variable_ids[rel_id]):
                col_idx = col_by_var_id.get(var_id)
                if col_idx is None:
                    continue
                rows.append(row_idx)
                cols.append(col_idx)

        relations_out = [self.relations[rel_id] for rel_id in active_relation_ids]
        variables_out = [self.variables[var_id].name for var_id in active_variable_ids]
        matrix = coo_matrix(
            ([1.0] * len(rows), (rows, cols)),
            shape=(len(relations_out), len(variables_out)),
            dtype=float,
        )
        return relations_out, variables_out, matrix

    def connected_components(
        self,
        relations: Iterable[object],
        variables: Iterable[str],
    ) -> list[tuple[list[object], list[str]]]:
        """Return connected relation-variable components limited to subsets.

        Args:
            relations: Candidate relation objects.
            variables: Candidate variable names.

        Returns:
            Components as ordered ``(relations, variable_names)`` pairs.
        """
        # Convert caller subsets into dense ids before walking adjacency.
        relation_ids = {
            rel_id
            for relation in relations
            for rel_id in (self.relation_id_by_object.get(relation),)
            if rel_id is not None
        }
        variable_ids = {
            var_id
            for name in variables
            for var_id in (self.variable_id_by_name.get(name),)
            if var_id is not None
        }
        if not relation_ids or not variable_ids:
            return []

        # Traverse the bipartite relation-variable graph without materializing a graph object.
        components: list[tuple[list[object], list[str]]] = []
        visited_relations: set[int] = set()
        visited_variables: set[int] = set()
        for start_rel_id in sorted(relation_ids):
            if start_rel_id in visited_relations:
                continue

            rel_stack = [start_rel_id]
            var_stack: list[int] = []
            visited_relations.add(start_rel_id)
            component_relation_ids: set[int] = set()
            component_variable_ids: set[int] = set()

            while rel_stack or var_stack:
                while rel_stack:
                    rel_id = rel_stack.pop()
                    component_relation_ids.add(rel_id)
                    for var_id in self.relation_variable_ids[rel_id]:
                        if var_id not in variable_ids or var_id in visited_variables:
                            continue
                        visited_variables.add(var_id)
                        var_stack.append(var_id)

                while var_stack:
                    var_id = var_stack.pop()
                    component_variable_ids.add(var_id)
                    for rel_id in self.variable_relation_ids[var_id]:
                        if rel_id not in relation_ids or rel_id in visited_relations:
                            continue
                        visited_relations.add(rel_id)
                        rel_stack.append(rel_id)

            comp_relations = [
                self.relations[rel_id]
                for rel_id in sorted(component_relation_ids)
            ]
            comp_variables = [
                self.variables[var_id].name
                for var_id in sorted(component_variable_ids)
            ]
            if comp_relations and comp_variables:
                components.append((comp_relations, comp_variables))
        return components

    def add_variable(self, name: str, variable: Variable) -> None:
        """Register one runtime-created variable while preserving graph indices.

        Args:
            name: Canonical variable name.
            variable: Variable object to attach.

        Returns:
            None.
        """
        # Replace an existing object payload when the graph already knows the name.
        var_id = self.variable_id_by_name.get(name)
        if var_id is not None:
            self.variables[var_id] = variable
            return

        # Add a new isolated node for truly runtime-created variables.
        self.variable_id_by_name[name] = len(self.variables)
        self.variables.append(variable)
        self.variable_relation_ids = (*self.variable_relation_ids, ())

    @classmethod
    def build(
        cls,
        *,
        relations: Iterable[object],
        variables: Iterable[Variable],
    ) -> "SystemGraph":
        """Build one ordered, index-backed relation-variable graph.

        Args:
            relations: Final relation objects to index.
            variables: Final variable objects to index.

        Returns:
            One graph object containing indexed topology and object payloads.
        """
        relation_candidates = list(relations)
        input_variables = list(variables)

        # Intern variables as actual Variable objects; relation-discovered variables get empty objects.
        graph_variables: list[Variable] = []
        variable_id_by_name: dict[str, int] = {}
        for var in input_variables:
            existing = variable_id_by_name.get(var.name)
            if existing is not None:
                graph_variables[existing] = var
                continue
            variable_id_by_name[var.name] = len(graph_variables)
            graph_variables.append(var)

        def variable_id(name: str) -> int:
            """Return existing variable id or create an empty Variable object."""
            existing = variable_id_by_name.get(name)
            if existing is not None:
                return existing
            variable_id_by_name[name] = len(graph_variables)
            graph_variables.append(
                Variable.make(
                    name=name,
                    ndim=allowed_variable_ndim(name),
                )
            )
            return variable_id_by_name[name]

        # Build indexed relation-variable adjacency directly from Relation symbols.
        relation_id_by_object: dict[object, int] = {}
        relation_variable_ids: list[tuple[int, ...]] = []
        variable_relation_id_sets: list[set[int]] = [set() for _ in graph_variables]
        seen_names: set[str] = set()
        duplicate_relation_names: dict[str, list[int]] = {}
        for rel in relation_candidates:
            rel_id = len(relation_variable_ids)
            rel_name = getattr(rel, "name", None) or f"relation_{rel_id}"
            if rel_name in seen_names:
                duplicate_relation_names.setdefault(rel_name, []).append(rel_id)
            else:
                seen_names.add(rel_name)

            ids = tuple(
                variable_id(str(name))
                for name in getattr(rel, "symbols", {})
                if name is not None
            )
            while len(variable_relation_id_sets) < len(graph_variables):
                variable_relation_id_sets.append(set())
            for var_id in ids:
                variable_relation_id_sets[var_id].add(rel_id)
            relation_id_by_object[rel] = rel_id
            relation_variable_ids.append(ids)

        return cls(
            relations=relation_candidates,
            variables=graph_variables,
            relation_id_by_object=relation_id_by_object,
            variable_id_by_name=variable_id_by_name,
            relation_variable_ids=tuple(relation_variable_ids),
            variable_relation_ids=tuple(tuple(sorted(ids)) for ids in variable_relation_id_sets),
            duplicate_relation_names=duplicate_relation_names,
        )
