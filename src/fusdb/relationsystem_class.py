"""RelationSystem class to solve interconnected relations."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, Mapping
import logging
import math
import numpy as np
import sympy as sp

from .structural import structural_decomposition
from .system_graph import SystemGraph
from .variable_class import Variable
from .relation_class import get_filtered_relations
from .utils import normalize_tag
from .utils import (
    all_tolerances,
    as_profile_array,
    brent_root,
    relative_change,
    safe_float,
    within_tolerance,
    scalarize_mapping,
    scalarize_value,
)

logger = logging.getLogger(__name__)


@dataclass
class RelationSystem:
    """Collection of relations and variables with a solver."""
    relations: list
    variables: list[Variable]
    mode: str | None = "overwrite"
    verbose: bool = False
    n_max: int = 4
    max_passes: int = 8
    default_rel_tol: float = 0.01
    reactor_tags: Iterable[str] = field(default_factory=tuple)
    solving_order: Iterable[str] = field(default_factory=tuple)
    variable_methods: Iterable[str] = field(default_factory=tuple)
    default_relations: list = field(default_factory=list)
    _log: logging.Logger = field(init=False, repr=False)
    graph: SystemGraph = field(init=False, repr=False)
    variable_bounds: dict[str, tuple[float | None, float | None]] = field(init=False, default_factory=dict, repr=False)
    compiled_constraints: dict[str, tuple[tuple[str, ...], object | None]] = field(
        init=False, default_factory=dict, repr=False
    )
    warned: dict[str, set[str]] = field(init=False, default_factory=dict, repr=False)
    last_result: dict[str, object] = field(init=False, default_factory=dict)

    # Runtime scratch kept only while preserving the current solver algorithm.
    _overrides: dict[str, object] = field(init=False, default_factory=dict, repr=False)
    _bundle_violation_applied: set[object] = field(init=False, default_factory=set, repr=False)
    _relation_status_cache: dict[object, tuple[str, float | None]] = field(init=False, default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """Initialize setup, solve state, and evaluation metadata."""
        # Build one child logger and align its threshold with instance verbosity.
        self._log = logger.getChild(self.__class__.__name__)
        self._log.setLevel(logging.INFO if self.verbose else logging.WARNING)

        # Normalize missing mode values, then reject unsupported solver modes early.
        self.mode = "overwrite" if self.mode is None else str(self.mode)
        if self.mode not in ("overwrite", "check"):
            raise ValueError("RelationSystem mode must be 'overwrite' or 'check'.")

        # Canonicalize inputs before building relation-variable topology.
        # Relations
        if self.relations is None:
            self.relations = []
        elif not isinstance(self.relations, list):
            raise TypeError("RelationSystem relations must be a list of Relation objects")
        # Variables
        if self.variables is None:
            self.variables = []
        elif not isinstance(self.variables, list):
            raise TypeError("RelationSystem variables must be a list of Variable objects")
        # Reset solver/runtime mutable state that is not part of topology.
        self._overrides = {}
        self.warned = {
            "inconsistency": set(),
        }
        self.last_result = {
            "stop_reason": None,
            "final_check": {},
            "metrics": {},
            "violated_relations": set(),
        }
        self._bundle_violation_applied = set()
        self._relation_status_cache = {}

        # Resolve relation inputs here so SystemGraph only indexes finalized topology.
        relation_candidates = list(self.relations)
        default_relation_candidates = list(self.default_relations)
        resolved_variable_methods = tuple(name for name in self.variable_methods if name)
        if not resolved_variable_methods:
            resolved_variable_methods = tuple(
                var.method
                for var in self.variables
                if var.method
            )
        if relation_candidates:
            seen_relation_ids = {id(rel) for rel in relation_candidates}
            for rel in default_relation_candidates:
                if id(rel) in seen_relation_ids:
                    continue
                relation_candidates.append(rel)
                seen_relation_ids.add(id(rel))
        elif self.reactor_tags or resolved_variable_methods or default_relation_candidates:
            relation_candidates = list(
                get_filtered_relations(
                    self.reactor_tags,
                    resolved_variable_methods,
                    extra_relations=default_relation_candidates,
                )
            )

        solving_order_items = [str(item) for item in self.solving_order if str(item)]
        if solving_order_items:
            exact_name_items = {
                item
                for item in solving_order_items
                if any(getattr(rel, "name", None) == item for rel in relation_candidates)
            }
            explicitly_named = {
                rel
                for rel in relation_candidates
                if getattr(rel, "name", None) in exact_name_items
            }
            ordered_relation_objects: list[object] = []
            seen: set[object] = set()
            for item in solving_order_items:
                exact_matches = [
                    rel
                    for rel in relation_candidates
                    if getattr(rel, "name", None) == item
                ]
                if exact_matches:
                    for rel in exact_matches:
                        if rel in seen:
                            continue
                        ordered_relation_objects.append(rel)
                        seen.add(rel)
                    continue

                # Treat non-relation-name items as normalized tag/domain groups.
                target_tag = normalize_tag(item)
                for rel in relation_candidates:
                    if rel in seen or rel in explicitly_named:
                        continue
                    relation_tags = {
                        normalize_tag(tag)
                        for tag in (getattr(rel, "tags", ()) or ())
                    }
                    if target_tag not in relation_tags:
                        continue
                    ordered_relation_objects.append(rel)
                    seen.add(rel)
            relation_candidates = ordered_relation_objects

        self.graph = SystemGraph.build(
            relations=relation_candidates,
            variables=self.variables,
        )
        self.relations = self.graph.relations
        self._log.debug(
            "RelationSystem.__post_init__: mode=%s, n_relations=%s, n_variables=%s",
            self.mode,
            len(self.relations),
            len(self.graph.variables),
        )

        if self.graph.duplicate_relation_names:
            duplicate_names = sorted(self.graph.duplicate_relation_names)
            sample = ", ".join(duplicate_names[:8])
            if len(duplicate_names) > 8:
                sample = f"{sample}, ..."
            duplicate_total = sum(len(indices) for indices in self.graph.duplicate_relation_names.values())
            self._log.info(
                "Detected %d duplicate relation entries across %d names; names are not unique keys. Examples: %s",
                duplicate_total,
                len(duplicate_names),
                sample,
            )

        self._log.debug("Total variables in system: %s", len(self.graph.variable_names()))

        # Step 4: pre-compile object-owned constraints and infer static bounds once.
        all_constraints: set[str] = set()
        var_with_constraints = 0
        for var in self.graph.variables:
            constraints = tuple(getattr(var, "constraints", ()) or ())
            if constraints:
                var_with_constraints += 1
            all_constraints.update(constraints)

        rel_with_constraints = 0
        for rel in self.relations:
            constraints = tuple(getattr(rel, "constraints", ()) or ())
            if constraints:
                rel_with_constraints += 1
            all_constraints.update(constraints)

        for constraint in all_constraints:
            self._compile_constraint(constraint)

        self.variable_bounds = {name: self._compute_var_bounds(name) for name in self.graph.variable_names()}

        self._log.debug(
            "Built constraints: %s variables with constraints, %s relations with constraints",
            var_with_constraints,
            rel_with_constraints,
        )
        
        if self.mode == "overwrite":
            self._start_pass()





    def _values_equal_for_var(
        self,
        var: Variable,
        left: object,
        right: object,
    ) -> bool:
        """Return whether two values should be treated as unchanged for a variable."""
        rel_tol = float(var.rel_tol if var.rel_tol is not None else self.default_rel_tol)
        if left is None or right is None:
            return left is right

        if var.ndim == 1:
            y_left = as_profile_array(left)
            y_right = as_profile_array(right)
            if y_left is None or y_right is None:
                return False
            return bool(all_tolerances(y_left, y_right, rel_tol=rel_tol))

        lv = safe_float(left)
        rv = safe_float(right)
        if lv is not None and rv is not None:
            return all_tolerances(lv, rv, rel_tol=rel_tol)
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            return all_tolerances(left, right, rel_tol=rel_tol)

        if isinstance(left, sp.Basic) or isinstance(right, sp.Basic):
            try:
                if sp.simplify(left - right) == 0:
                    return True
            except Exception:
                pass
            lv = safe_float(left.evalf() if isinstance(left, sp.Basic) else left)
            rv = safe_float(right.evalf() if isinstance(right, sp.Basic) else right)
            if lv is not None and rv is not None:
                return all_tolerances(lv, rv, rel_tol=rel_tol)

        try:
            return bool(left == right)
        except Exception:
            return False

    def _structural_closure_blocks(
        self,
        values: dict[str, object],
        *,
        width: int,
    ) -> list[tuple[list[object], list[str]]]:
        """Return coupled solve blocks from structural decomposition.

        Args:
            values: Current effective values map.
            width: Current n-sweep coupled width.

        Returns:
            ``[(relations, unknowns), ...]`` block candidates.
        """
        # Gather unresolved scalar unknowns that can still be adjusted.
        unknown = {
            name
            for name in self.graph.variable_names()
            if self.graph.variable_ndim(name) == 0
            and self._is_adjustable_variable(name)
            and values.get(name) is None
        }
        if len(unknown) < 2:
            return []

        # Keep only closed scalar relations touching current unknowns.
        candidate_relations: list[object] = []
        for rel in self.relations:
            if not self._is_block_relation_candidate(rel):
                continue
            rel_vars = tuple(
                name
                for name in self.graph.relation_variable_names(rel)
                if self.graph.variable_ndim(name) == 0
            )
            if not rel_vars or not any(name in unknown for name in rel_vars):
                continue
            if any(values.get(name) is None and name not in unknown for name in rel_vars):
                continue
            candidate_relations.append(rel)
        if not candidate_relations:
            return []

        # Decompose the scalar incidence matrix and extract well-constrained blocks.
        candidate_vars = [name for name in self.graph.variable_names() if name in unknown]
        rels, vars_, matrix = self.graph.build_scalar_incidence(candidate_relations, candidate_vars)
        decomposition = structural_decomposition(
            relations=rels,
            variables=vars_,
            matrix=matrix,
        )
        if matrix.shape[0] != 0 and matrix.shape[1] != 0:
            self.last_result["metrics"]["structural_decompositions"] = int(
                self.last_result["metrics"].get("structural_decompositions", 0)
            ) + 1

        blocks: list[tuple[list[object], list[str]]] = []
        for row_block, col_block in decomposition["blocks"]:
            block_unknowns = sorted(
                name
                for idx, name in enumerate(vars_)
                if idx in col_block and name in unknown
            )
            if len(block_unknowns) < 2 or len(block_unknowns) > width:
                continue
            block_relations = [rels[idx] for idx in row_block]
            blocks.append((block_relations, block_unknowns))
        if blocks:
            return blocks

        # Fall back to canonical graph-connected components when DM blocks are empty.
        for block_relations, block_variables in self.graph.connected_components(rels, vars_):
            block_unknowns = sorted(
                (name for name in block_variables if name in unknown),
                key=lambda name: (self.graph.variable_id_by_name.get(name, 10**9), str(name)),
            )
            if len(block_unknowns) < 2 or len(block_unknowns) > width:
                continue
            equation_count = sum(
                max(len(getattr(rel, "outputs", ()) or ()), 1)
                for rel in block_relations
            )
            if equation_count < len(block_unknowns):
                continue
            if any(
                not any(name in self.graph.relation_variable_names(rel) for rel in block_relations)
                for name in block_unknowns
            ):
                continue
            blocks.append((list(block_relations), block_unknowns))
        return blocks

    def _structural_reconciliation_components(
        self,
        values: dict[str, object],
        violated_rels: set[object],
    ) -> list[tuple[list[object], list[str]]]:
        """Return violated structural components for reconciliation.

        Args:
            values: Current effective values map.
            violated_rels: Current violated decidable relation set.

        Returns:
            ``[(relations, variables), ...]`` reconciliation components.
        """
        if not violated_rels:
            return []

        # Keep only decidable scalar relations with adjustable scalar variables.
        decidable_relations = [
            rel
            for rel in self.relations
            if self._relation_status(rel, values)[0] != "UNDECIDABLE"
            and self._is_block_relation_candidate(rel, allow_profiles=True)
        ]
        adjustable = [
            name
            for name in self.graph.variable_names()
            if self._is_adjustable_variable(name)
            and (
                safe_float(values.get(name)) is not None
                or as_profile_array(values.get(name)) is not None
            )
        ]
        rels, vars_, matrix = self.graph.build_scalar_incidence(
            decidable_relations,
            adjustable,
            scalar_only=False,
        )
        if matrix.shape[0] == 0 or matrix.shape[1] == 0:
            return []

        decomposition = structural_decomposition(
            relations=rels,
            variables=vars_,
            matrix=matrix,
        )
        if matrix.shape[0] != 0 and matrix.shape[1] != 0:
            self.last_result["metrics"]["structural_decompositions"] = int(
                self.last_result["metrics"].get("structural_decompositions", 0)
            ) + 1
        violated_lookup = {rel for rel in violated_rels}

        # Use structural blocks first and keep only violated ones.
        components: list[tuple[list[object], list[str]]] = []
        covered_violated: set[object] = set()
        for row_block, col_block in decomposition["blocks"]:
            block_rels = [rels[idx] for idx in row_block]
            if not any(rel in violated_lookup for rel in block_rels):
                continue
            covered_violated.update(rel for rel in block_rels if rel in violated_lookup)
            violated_outputs = {
                output
                for rel in block_rels
                if rel in violated_lookup
                for output in tuple(getattr(rel, "outputs", ()) or ())
            }
            ranked_vars = sorted(
                (vars_[idx] for idx in col_block),
                key=lambda name: (
                    0 if name in violated_outputs else 1,
                    -sum(
                        1
                        for rel in self.graph.variable_relations(name)
                        if rel in violated_lookup
                    ),
                    -len(self.graph.variable_relations(name)),
                    name,
                ),
            )
            must_have = [name for name in ranked_vars if name in violated_outputs]
            width_limit = max(int(self.n_max), len(must_have))
            selected = list(must_have)
            if len(selected) < width_limit:
                selected.extend(
                    name
                    for name in ranked_vars
                    if name not in set(selected)
                )
                selected = selected[:width_limit]
            if selected:
                components.append((block_rels, sorted(selected)))

        uncovered_violated = set(violated_rels) - covered_violated
        if uncovered_violated:
            for block_rels, block_vars in self.graph.connected_components(rels, vars_):
                if not any(rel in uncovered_violated for rel in block_rels):
                    continue
                violated_outputs = {
                    output
                    for rel in block_rels
                    if rel in violated_lookup
                    for output in tuple(getattr(rel, "outputs", ()) or ())
                }
                ranked_vars = sorted(
                    block_vars,
                    key=lambda name: (
                        0 if name in violated_outputs else 1,
                        -sum(
                            1
                            for rel in self.graph.variable_relations(name)
                            if rel in violated_lookup
                        ),
                        -len(self.graph.variable_relations(name)),
                        name,
                    ),
                )
                must_have = [name for name in ranked_vars if name in violated_outputs]
                width_limit = max(int(self.n_max), len(must_have))
                selected = list(must_have)
                if len(selected) < width_limit:
                    selected.extend(
                        name
                        for name in ranked_vars
                        if name not in set(selected)
                    )
                    selected = selected[:width_limit]
                if selected:
                    components.append((block_rels, sorted(selected)))

        # Keep deterministic order and remove duplicates.
        dedup: list[tuple[list[object], list[str]]] = []
        seen: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
        for rel_block, var_block in components:
            rel_key = tuple(sorted(getattr(rel, "name", rel.outputs[0]) for rel in rel_block))
            var_key = tuple(sorted(var_block))
            key = (rel_key, var_key)
            if key in seen:
                continue
            seen.add(key)
            dedup.append((rel_block, list(var_key)))
        return dedup

    def _normalized_residual_norm(
        self,
        values: dict[str, object],
        relations: Iterable[object],
    ) -> float:
        """Return one normalized residual norm for relation quality ranking.

        Args:
            values: Current effective values map.
            relations: Relations to score.

        Returns:
            Non-negative normalized residual norm.
        """
        residuals: list[float] = []
        for rel in relations:
            status, residual = self._relation_status(rel, values)
            if status == "UNDECIDABLE":
                continue
            if residual is None:
                residuals.append(1.0)
                continue
            residuals.append(float(residual))
        if not residuals:
            return 0.0
        arr = np.asarray(residuals, dtype=float)
        return float(np.linalg.norm(arr) / max(len(residuals), 1))

    def _structural_summary(self, values: dict[str, object]) -> dict[str, object]:
        """Return one compact structural decomposition summary for diagnostics.

        Args:
            values: Current effective values map.

        Returns:
            Compact structural summary payload.
        """
        # Use scalar relations/variables so decomposition reflects solve-relevant topology.
        scalar_vars = [name for name in self.graph.variable_names() if self.graph.variable_ndim(name) == 0]
        scalar_relations = [
            rel
            for rel in self.relations
            if any(self.graph.variable_ndim(name) == 0 for name in self.graph.relation_variable_names(rel))
        ]
        rels, vars_, matrix = self.graph.build_scalar_incidence(scalar_relations, scalar_vars)
        decomposition = structural_decomposition(
            relations=rels,
            variables=vars_,
            matrix=matrix,
        )
        if matrix.shape[0] != 0 and matrix.shape[1] != 0:
            self.last_result["metrics"]["structural_decompositions"] = int(
                self.last_result["metrics"].get("structural_decompositions", 0)
            ) + 1

        # Map partition indices to readable relation/variable names.
        row_partitions = decomposition["row_partitions"]
        col_partitions = decomposition["col_partitions"]
        under_rels = [getattr(rels[idx], "name", rels[idx].outputs[0]) for idx in row_partitions["under"]]
        well_rels = [getattr(rels[idx], "name", rels[idx].outputs[0]) for idx in row_partitions["well"]]
        over_rels = [getattr(rels[idx], "name", rels[idx].outputs[0]) for idx in row_partitions["over"]]
        under_vars = [vars_[idx] for idx in col_partitions["under"]]
        well_vars = [vars_[idx] for idx in col_partitions["well"]]
        over_vars = [vars_[idx] for idx in col_partitions["over"]]

        violated = self._violated_decidable_relations(values, rels)
        violated_block_count = sum(
            1
            for row_block, _ in decomposition["blocks"]
            if any(rels[idx] in violated for idx in row_block)
        )

        return {
            "underconstrained_relations": under_rels,
            "wellconstrained_relations": well_rels,
            "overconstrained_relations": over_rels,
            "underconstrained_variables": under_vars,
            "wellconstrained_variables": well_vars,
            "overconstrained_variables": over_vars,
            "wellconstrained_block_count": int(len(decomposition["blocks"])),
            "violated_block_count": int(violated_block_count),
        }

    def _closure_structural_plan(self, values: dict[str, object]) -> dict[str, object]:
        """Return structural closure plan with writer multiplicity classification.

        Args:
            values: Current effective values map.

        Returns:
            Plan containing partitioned relations and output writer counts.
        """
        # Build one scalar candidate relation set for block-level closure decisions.
        candidate_relations: list[object] = []
        for rel in self.relations:
            if not self._is_block_relation_candidate(rel):
                continue
            rel_vars = tuple(
                name
                for name in self.graph.relation_variable_names(rel)
                if self.graph.variable_ndim(name) == 0
            )
            if not rel_vars:
                continue
            if any(
                values.get(name) is None and not self._is_adjustable_variable(name)
                for name in rel_vars
            ):
                continue
            candidate_relations.append(rel)

        candidate_vars = list(
            dict.fromkeys(
                output_name
                for rel in candidate_relations
                for output_name in tuple(getattr(rel, "outputs", ()) or ())
                if self.graph.variable_ndim(output_name) == 0 and self._is_adjustable_variable(output_name)
            )
        )
        rels, vars_, matrix = self.graph.build_scalar_incidence(candidate_relations, candidate_vars)
        decomposition = structural_decomposition(
            relations=rels,
            variables=vars_,
            matrix=matrix,
        )
        if matrix.shape[0] != 0 and matrix.shape[1] != 0:
            self.last_result["metrics"]["structural_decompositions"] = int(
                self.last_result["metrics"].get("structural_decompositions", 0)
            ) + 1

        # Classify relations by DM partition so closure/reconciliation routing is explicit.
        row_partitions = decomposition.get("row_partitions", {})
        under_relations = {
            rels[idx]
            for idx in row_partitions.get("under", ())
            if 0 <= idx < len(rels)
        }
        well_relations = {
            rels[idx]
            for idx in row_partitions.get("well", ())
            if 0 <= idx < len(rels)
        }
        over_relations = {
            rels[idx]
            for idx in row_partitions.get("over", ())
            if 0 <= idx < len(rels)
        }

        # Count active writers per scalar output to detect multi-writer progress paths.
        writer_counts: dict[str, int] = {}
        for rel in candidate_relations:
            outputs = tuple(getattr(rel, "outputs", ()) or ())
            if not outputs:
                continue
            if len(outputs) > 1:
                if any(values.get(name) is None for name in tuple(getattr(rel, "inputs", ()) or ())):
                    continue
                for output_name in outputs:
                    if self.graph.variable_ndim(output_name) != 0 or not self._is_adjustable_variable(output_name):
                        continue
                    writer_counts[output_name] = writer_counts.get(output_name, 0) + 1
                continue

            target_name = outputs[0]
            if self.graph.variable_ndim(target_name) != 0 or not self._is_adjustable_variable(target_name):
                continue
            input_names = rel.input_names(target_name)
            if any(self._resolve_value_for_name(name, values) is None for name in input_names):
                continue
            writer_counts[target_name] = writer_counts.get(target_name, 0) + 1

        multiwriter_relations: set[object] = set()
        for rel in candidate_relations:
            outputs = tuple(getattr(rel, "outputs", ()) or ())
            if any(writer_counts.get(output_name, 0) > 1 for output_name in outputs):
                multiwriter_relations.add(rel)

        return {
            "candidate_relations": candidate_relations,
            "under_relations": under_relations,
            "well_relations": well_relations,
            "over_relations": over_relations,
            "multiwriter_relations": multiwriter_relations,
            "writer_counts": writer_counts,
        }

    def _to_scalar_values(self, values: dict[str, object]) -> dict[str, object]:
        """Return values where profile payloads are reduced to profile means."""
        return scalarize_mapping(values, ndim_lookup=self.graph.variable_ndim)

    def _normalize_runtime_value(self, name: str, value: object) -> object:
        """Validate one runtime value against the profile-array contract."""
        if value is None or self.graph.variable_ndim(name) != 1:
            return value
        arr = as_profile_array(value)
        if arr is not None:
            return arr
        scalar = safe_float(value)
        if scalar is None:
            raise TypeError(
                f"Profile variable '{name}' must be provided as a scalar or 1D numpy.ndarray."
            )
        var = self.graph.variable(name)
        profile_size = 51 if var is None else int(var.profile_size)
        if profile_size < 1:
            raise ValueError(f"Profile variable '{name}' requires profile_size >= 1.")
        return np.full(profile_size, scalar, dtype=float)

    def _derive_missing_value(
        self,
        name: str,
        values: Mapping[str, object],
    ) -> object | None:
        """Derive one missing value explicitly from profile/average counterparts."""
        from .registry import canonical_variable_name

        ndim = self.graph.variable_ndim(name)
        if ndim == 1:
            avg_name = canonical_variable_name(f"{name}_avg")
            source_scalar = safe_float(scalarize_value(values.get(avg_name)))
            if source_scalar is not None:
                return self._normalize_runtime_value(name, source_scalar)
            return None

        if not name.endswith("_avg"):
            return None
        profile_name = canonical_variable_name(name[:-4])
        if self.graph.variable_ndim(profile_name) == 1:
            source = values.get(profile_name)
            reduced = safe_float(scalarize_value(source))
            if reduced is not None:
                return reduced
        return None

    def _resolve_value_for_name(
        self,
        name: str,
        values: Mapping[str, object],
    ) -> object | None:
        """Return direct value or explicit profile/average fallback for one name."""
        direct = values.get(name)
        if direct is not None:
            return self._normalize_runtime_value(name, direct)
        return self._derive_missing_value(name, values)

    def _apply_explicit_fallbacks(self, values: dict[str, object]) -> dict[str, object]:
        """Fill missing values using explicit profile/average fallback rules."""
        merged = dict(values)
        changed = True
        while changed:
            changed = False
            for name in self.graph.variable_names():
                if merged.get(name) is not None:
                    continue
                derived = self._derive_missing_value(name, merged)
                if derived is None:
                    continue
                merged[name] = derived
                changed = True
        return merged

    @property
    def variables_dict(self) -> dict[str, Variable]:
        """Return current variables keyed by name (dict graph is source of truth)."""
        vars_map: dict[str, Variable] = {}
        for name in self.graph.variable_names():
            var = self.graph.variable(name)
            if var is not None:
                vars_map[name] = var
        return vars_map
    
    def _values_dict(self) -> dict[str, object]:
        """Return mapping of effective values."""
        values: dict[str, object] = {}
        for name in self.graph.variable_names():
            var = self.graph.variable(name)
            if var is None:
                continue
            value = var.current_value if var.current_value is not None else var.input_value
            if value is not None:
                values[name] = value
        return values

    def _compile_constraint(self, constraint: str) -> tuple[tuple[str, ...], object | None]:
        cached = self.compiled_constraints.get(constraint)
        if cached is not None:
            return cached
        try:
            expr = sp.sympify(constraint)
        except Exception:
            cached = ((), None)
            self.compiled_constraints[constraint] = cached
            return cached
        if not hasattr(expr, "free_symbols"):
            try:
                cached = ((), bool(expr))
            except Exception:
                cached = ((), None)
            self.compiled_constraints[constraint] = cached
            return cached
        symbols = tuple(sorted(expr.free_symbols, key=lambda s: s.name))
        if not symbols:
            try:
                cached = ((), bool(expr))
            except Exception:
                cached = ((), None)
            self.compiled_constraints[constraint] = cached
            return cached
        names = tuple(sym.name for sym in symbols)
        try:
            func = sp.lambdify(symbols, expr, modules=["numpy"])
        except Exception:
            func = None
        cached = (names, func)
        self.compiled_constraints[constraint] = cached
        return cached

    def _constraint_result(
        self,
        constraint: str,
        values_scalar: dict[str, object],
    ) -> bool | None:
        cached = self.compiled_constraints.get(constraint)
        if cached is None:
            return None
        names, func = cached
        if func is None:
            return None
        if not names:
            try:
                return bool(func)
            except Exception:
                return None
        args: list[object] = []
        for name in names:
            val = values_scalar.get(name)
            if val is None:
                return None
            args.append(val)
        try:
            return bool(func(*args))
        except Exception:
            return None

    def _constraint_violations(
        self,
        constraints: Iterable[str],
        values_scalar: dict[str, object],
    ) -> list[str]:
        return [
            constraint
            for constraint in constraints
            if self._constraint_result(constraint, values_scalar) is False
        ]

    def _accept_candidate_values(
        self,
        solved: dict[str, object],
        *,
        rels: list[object],
        reason: str,
        relation: str | list[str] | None,
        values_map: dict[str, object] | None = None,
        warn_input: bool = False,
        check_violation_increase: bool = False,
    ) -> bool:
        """Validate and commit candidate values in one place."""
        if not solved:
            return False

        # Ensure candidate variables own registry constraints before validation.
        for name in solved:
            if self.graph.has_variable(name):
                continue
            var = Variable.make(name=name, ndim=self.graph.variable_ndim(name))
            self.graph.add_variable(name, var)
            for constraint in tuple(getattr(var, "constraints", ()) or ()):
                self._compile_constraint(constraint)
            self.variable_bounds[name] = self._compute_var_bounds(name)

        base_values = dict(values_map or self._values_dict())
        merged = dict(base_values)
        merged.update(solved)
        for name in solved:
            if self._constraints_violated(merged, names=[name]):
                return False
        for rel in rels:
            if self._constraints_violated(merged, rel=rel, names=()):
                return False

        if check_violation_increase:
            if len(solved) != 1:
                return False
            name, value = next(iter(solved.items()))
            value_scalar = safe_float(value)
            if value_scalar is None and as_profile_array(value) is None:
                return False
            if value_scalar is not None:
                rels_to_check = self.graph.variable_relations(name)
                if rels_to_check:
                    base_violations = len(
                        self._violated_relations(base_values, list(rels_to_check))
                    )
                    candidate_values = dict(base_values)
                    candidate_values[name] = value_scalar
                    candidate_violations = len(
                        self._violated_relations(candidate_values, list(rels_to_check))
                    )
                    if candidate_violations > base_violations:
                        self._log.debug(
                            "    ✗ Candidate for %s increases violations",
                            name,
                        )
                        return False

        accepted: dict[str, object] = {}
        for name, value in solved.items():
            var = self.graph.variable(name)
            if var is not None and var.fixed:
                return False

            is_profile = self.graph.variable_ndim(name) == 1
            if is_profile:
                accepted[name] = value
                continue

            value_scalar = safe_float(value)
            if value_scalar is None:
                return False

            cur_value = base_values.get(name)
            cur_scalar = safe_float(cur_value) if cur_value is not None else None
            if warn_input and cur_scalar is not None:
                input_value = var.input_value if var is not None else None
                if input_value is not None and name not in self.warned["inconsistency"]:
                    input_scalar = safe_float(input_value)
                    if input_scalar is not None:
                        var_tol = self.graph.variable(name)
                        rel_tol = var_tol.rel_tol if var_tol is not None and var_tol.rel_tol is not None else self.default_rel_tol
                        if not within_tolerance(input_scalar, value_scalar, rel_tol=float(rel_tol)):
                            rel_name = (
                                relation
                                if isinstance(relation, str)
                                else (relation[0] if relation else "unknown")
                            )
                            self._log.warning(
                                "Inconsistency: relation '%s' computed %s = %.3g, but input specifies %s = %.3g",
                                rel_name,
                                name,
                                value_scalar,
                                name,
                                input_scalar,
                            )
                            self.warned["inconsistency"].add(name)

            if cur_scalar is not None:
                var_tol = self.graph.variable(name)
                rel_tol = var_tol.rel_tol if var_tol is not None and var_tol.rel_tol is not None else self.default_rel_tol
                if within_tolerance(cur_scalar, value_scalar, rel_tol=float(rel_tol)):
                    continue

            accepted[name] = value

        if not accepted:
            return False

        for name, value in accepted.items():
            self._set_value(
                name,
                value,
                reason=reason,
                force=True,
            )
        return True

    def _set_value(
        self,
        name: str,
        value: object,
        *,
        reason: str | None = None,
        force: bool = False,
    ) -> bool:
        """Store a solved value in values map and refresh relation status cache."""
        self._log.debug(
            "_set_value(%s=%s, pass_mode=%s, reason=%s)",
            name,
            value,
            self.mode == "overwrite",
            reason,
        )
        var = self.graph.variable(name)
        if var is None:
            self._log.debug("Creating new variable: %s", name)
            var = Variable.make(name=name, ndim=self.graph.variable_ndim(name))
            self.graph.add_variable(name, var)
            for constraint in tuple(getattr(var, "constraints", ()) or ()):
                self._compile_constraint(constraint)
            self.variable_bounds[name] = self._compute_var_bounds(name)
        if (
            not force
            and var.current_value is not None
            and self._values_equal_for_var(var, var.current_value, value)
        ):
            self._log.debug("  No change for %s", name)
            return False

        changed = var.add_value(
            value,
            reason=reason,
            as_input=reason in ("input", "default"),
        )
        if not changed:
            self._log.debug("  No change for %s", name)
            return False
        self._relation_status_cache = {}
        self.last_result["metrics"]["new_assignments_total"] = int(
            self.last_result["metrics"].get("new_assignments_total", 0)
        ) + 1
        return True
    
    def _residual(self, rel: object, values: dict[str, object], *, scaled: bool = False) -> float | None:
        values_scalar = self._to_scalar_values(values)
        try:
            expected = rel.evaluate(values)
        except Exception:
            return None
        expected_scalar = safe_float(expected)
        if expected_scalar is None:
            return None
        target_name = rel.outputs[0]
        actual_scalar = safe_float(values_scalar.get(target_name))
        if actual_scalar is None:
            return None
        residual = actual_scalar - expected_scalar
        if not scaled:
            return residual
        scale = max(abs(expected_scalar), abs(actual_scalar), 1.0)
        return residual / scale

    def _residual_derivative(
        self,
        rel: object,
        name: str,
        values: dict[str, object],
        *,
        current: float | None = None,
    ) -> float | None:
        values_scalar = self._to_scalar_values(values)
        if name == rel.outputs[0]:
            return 1.0
        if self.graph.variable_ndim(name) == 1:
            return None
        if current is None:
            current = safe_float(values_scalar.get(name))
        if current is None:
            return None
        step = 1e-6 * max(abs(current), 1.0)
        v_plus = dict(values)
        v_minus = dict(values)
        v_plus[name] = current + step
        v_minus[name] = current - step
        try:
            f_plus = safe_float(rel.evaluate(v_plus))
        except Exception:
            f_plus = None
        try:
            f_minus = safe_float(rel.evaluate(v_minus))
        except Exception:
            f_minus = None
        if f_plus is None or f_minus is None:
            return None
        dfdx = (f_plus - f_minus) / (2 * step)
        if dfdx == 0 or not math.isfinite(dfdx):
            return None
        return -dfdx

    def _constraints_violated(
        self,
        values: dict[str, object],
        *,
        rel: object | None = None,
        names: Iterable[str] | None = None,
    ) -> bool:
        values_scalar = self._to_scalar_values(values)
        if rel is not None:
            if self._constraint_violations(tuple(getattr(rel, "constraints", ()) or ()), values_scalar):
                return True
        if names is None:
            names = self.graph.relation_variable_names(rel) if rel is not None else ()
        for name in names:
            var = self.graph.variable(name)
            constraints = tuple(getattr(var, "constraints", ()) or ()) if var is not None else ()
            if self._constraint_violations(constraints, values_scalar):
                return True
        return False

    def _numeric_inverse_single_scalar(
        self,
        rel: object,
        unknown: str,
        values_map: dict[str, object],
    ) -> float | None:
        """Solve one scalar unknown by root-finding relation residual.

        Args:
            rel: Relation being inverted.
            unknown: Scalar relation input to solve.
            values_map: Current known values.

        Returns:
            Solved scalar value, or ``None`` when no reliable inverse is found.
        """
        if self.graph.variable_ndim(unknown) == 1:
            return None
        target_name = rel.outputs[0]
        if unknown == target_name:
            return None
        if target_name not in values_map:
            return None

        target = safe_float(values_map.get(target_name))
        if target is None:
            return None
        input_names = rel.input_names(target_name)
        if any(name != unknown and values_map.get(name) is None for name in input_names):
            return None

        unknown_var = self.graph.variable(unknown)
        x_rel_tol = float(unknown_var.rel_tol if unknown_var is not None and unknown_var.rel_tol is not None else self.default_rel_tol)

        def _residual(x: float) -> float | None:
            merged = dict(values_map)
            merged[unknown] = x
            try:
                expected = rel.evaluate(merged)
            except Exception:
                return None
            expected_scalar = safe_float(expected)
            if expected_scalar is None:
                return None
            return target - expected_scalar

        def _is_solution(x: float, fx: float | None = None) -> bool:
            merged = dict(values_map)
            merged[unknown] = x
            try:
                expected = rel.evaluate(merged)
            except Exception:
                return False
            expected_scalar = safe_float(expected)
            if expected_scalar is None:
                return False
            # Root acceptance needs a small-value scale, otherwise target values
            # like delta ~= 0.003 accept the lower bound 0 as "close enough".
            residual = target - expected_scalar if fx is None else fx
            scale = max(abs(target), abs(expected_scalar), 1e-12)
            return abs(residual) <= x_rel_tol * scale

        lower, upper = self.variable_bounds.get(unknown, (None, None))
        if lower is not None and upper is not None and lower >= upper:
            return None

        guess_candidates: list[float] = []
        current = safe_float(values_map.get(unknown))
        if current is not None:
            guess_candidates.append(current)
        if unknown_var is not None:
            input_guess = safe_float(unknown_var.input_value)
            if input_guess is not None:
                guess_candidates.append(input_guess)
        if rel.initial_guesses and unknown in rel.initial_guesses:
            try:
                rel_guess = safe_float(rel.initial_guesses[unknown](values_map))
            except Exception:
                rel_guess = None
            if rel_guess is not None:
                guess_candidates.append(rel_guess)
        if lower is not None and upper is not None and math.isfinite(lower) and math.isfinite(upper):
            guess_candidates.append(0.5 * (lower + upper))
        guess_candidates.extend([1.0, 0.0, -1.0])

        def _clip(x: float) -> float:
            if lower is not None:
                x = max(x, lower)
            if upper is not None:
                x = min(x, upper)
            return x

        bracket: tuple[float, float, float, float] | None = None

        if lower is not None and upper is not None and math.isfinite(lower) and math.isfinite(upper):
            fa = _residual(lower)
            fb = _residual(upper)
            if fa is not None and fb is not None:
                if _is_solution(lower, fa):
                    return lower
                if _is_solution(upper, fb):
                    return upper
                if fa * fb < 0.0:
                    bracket = (lower, upper, fa, fb)
            if bracket is None:
                n_scan = 41
                xs = np.linspace(lower, upper, n_scan, dtype=float)
                prev_x = float(xs[0])
                prev_f = _residual(prev_x)
                if prev_f is not None and _is_solution(prev_x, prev_f):
                    return prev_x
                for x in xs[1:]:
                    x_val = float(x)
                    f_val = _residual(x_val)
                    if f_val is None:
                        prev_x, prev_f = x_val, f_val
                        continue
                    if _is_solution(x_val, f_val):
                        return x_val
                    if prev_f is not None and prev_f * f_val < 0.0:
                        bracket = (prev_x, x_val, prev_f, f_val)
                        break
                    prev_x, prev_f = x_val, f_val

        if bracket is None:
            span_factors = (1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 1e4)
            for guess in guess_candidates:
                if guess is None or not math.isfinite(guess):
                    continue
                center = _clip(float(guess))
                for factor in span_factors:
                    span = factor * max(abs(center), 1.0)
                    a = _clip(center - span)
                    b = _clip(center + span)
                    if not (math.isfinite(a) and math.isfinite(b)) or a >= b:
                        continue
                    fa = _residual(a)
                    fb = _residual(b)
                    if fa is None or fb is None:
                        continue
                    if _is_solution(a, fa):
                        return a
                    if _is_solution(b, fb):
                        return b
                    if fa * fb < 0.0:
                        bracket = (a, b, fa, fb)
                        break
                if bracket is not None:
                    break

        if bracket is None:
            return None

        a, b, fa, fb = bracket
        root = brent_root(
            _residual,
            a,
            b,
            fa,
            fb,
            rel_tol=float(x_rel_tol),
            max_iter=100,
        )
        if root is None:
            return None
        return root if _is_solution(root) else None

    def _solve_for_value(
        self,
        rel: object,
        name: str,
        values_map: dict[str, object],
        *,
        prefer_eval_output: bool = False,
    ) -> object | None:
        # Resolve relation target and declared input order once.
        working_values = self._apply_explicit_fallbacks(dict(values_map))
        target_name = rel.outputs[0]
        input_names = rel.input_names(target_name)

        # Respect explicit forward evaluation when requested.
        if (
            prefer_eval_output
            and target_name is not None
            and name == target_name
            and all(working_values.get(n) is not None for n in input_names)
        ):
            try:
                return rel.evaluate(working_values)
            except Exception:
                return None

        # Read solve_for policy flags for this direction.
        spec = (getattr(rel, "solve_for", {}) or {}).get(name)
        if isinstance(spec, bool):
            enabled, has_explicit_solver = bool(spec), False
        elif callable(spec):
            enabled, has_explicit_solver = True, True
        elif isinstance(spec, Mapping):
            enabled = bool(spec.get("enabled", True))
            has_explicit_solver = callable(spec.get("fn"))
        else:
            enabled, has_explicit_solver = True, False
        if not enabled:
            return None

        # Detect profile-dependent inputs once for fallback routing.
        has_profile_input = any(
            self.graph.variable_ndim(input_name) == 1
            for input_name in input_names
        )
        solved = None

        # Use user-defined explicit solve_for callables when present.
        if has_explicit_solver:
            try:
                solved = rel.solve_for_value(name, working_values)
            except Exception:
                solved = None

        # For non-output scalar unknowns, prefer numeric inversion first.
        if (
            solved is None
            and target_name is not None
            and name != target_name
            and self.graph.variable_ndim(name) == 0
        ):
            solved = self._numeric_inverse_single_scalar(rel, name, working_values)

        # Fall back to symbolic inversion only for cheap scalar cases.
        try_symbolic_inverse = False
        if (
            solved is None
            and not has_explicit_solver
            and target_name is not None
            and name != target_name
            and not has_profile_input
        ):
            expr = getattr(rel, "sympy_expression", None)
            symbols = getattr(rel, "symbols", {}) or {}
            sym_unknown = symbols.get(name)
            try_symbolic_inverse = expr is not None and sym_unknown is not None

            # Skip large symbolic systems where solve() is usually expensive.
            if try_symbolic_inverse and len(symbols) > 6:
                try_symbolic_inverse = False
            try:
                if try_symbolic_inverse and int(sp.count_ops(expr, visual=False)) > 80:
                    try_symbolic_inverse = False
            except Exception:
                try_symbolic_inverse = False

            # Skip fractional power inversions on the unknown (common slow path).
            try:
                if try_symbolic_inverse:
                    for power in expr.atoms(sp.Pow):
                        if not power.base.has(sym_unknown):
                            continue
                        exponent = power.exp
                        if exponent.is_number and not exponent.is_integer:
                            try_symbolic_inverse = False
                            break
            except Exception:
                try_symbolic_inverse = False

        if (
            solved is None
            and not has_explicit_solver
            and target_name is not None
            and name != target_name
            and not has_profile_input
            and try_symbolic_inverse
        ):
            values_scalar = self._to_scalar_values(working_values)
            try:
                solved = rel.solve_for_value(name, values_scalar)
            except Exception:
                solved = None

        # Finally compute forward output directly when solving for relation output.
        if solved is None and target_name is not None and name == target_name:
            try:
                solved = rel.evaluate(working_values)
            except Exception:
                return None
        return solved


    def _compute_var_bounds(self, name: str) -> tuple[float | None, float | None]:
        lower: float | None = None
        upper: float | None = None
        var = self.graph.variable(name)
        constraints = tuple(getattr(var, "constraints", ()) or ()) if var is not None else ()

        for constraint in constraints:
            try:
                expr = sp.sympify(constraint)
            except Exception:
                continue
            for arg in (expr.args if isinstance(expr, sp.And) else (expr,)):
                if not hasattr(arg, "rel_op"):
                    continue
                lhs, rhs = arg.lhs, arg.rhs
                lhs_is_name = isinstance(lhs, sp.Symbol) and lhs.name == name
                rhs_is_name = isinstance(rhs, sp.Symbol) and rhs.name == name
                try:
                    lhs_num = float(lhs) if getattr(lhs, "is_number", False) else None
                    rhs_num = float(rhs) if getattr(rhs, "is_number", False) else None
                except Exception:
                    lhs_num = None
                    rhs_num = None
                op = arg.rel_op
                if op in (">", ">="):
                    if lhs_is_name and rhs_num is not None:
                        lower = rhs_num if lower is None else max(lower, rhs_num)
                    elif rhs_is_name and lhs_num is not None:
                        upper = lhs_num if upper is None else min(upper, lhs_num)
                elif op in ("<", "<="):
                    if lhs_is_name and rhs_num is not None:
                        upper = rhs_num if upper is None else min(upper, rhs_num)
                    elif rhs_is_name and lhs_num is not None:
                        lower = lhs_num if lower is None else max(lower, lhs_num)
        return lower, upper

    def _constraint_residuals(
        self,
        constraints: tuple[str, ...],
        values: dict[str, object],
        *,
        penalty: float,
        values_scalar: dict[str, object] | None = None,
    ) -> list[float]:
        """Return per-constraint penalty residuals (0 if satisfied).

        Args:
            constraints: Constraint expressions to evaluate.
            values: Runtime values mapping.
            penalty: Residual penalty for undecidable/violated constraints.
            values_scalar: Optional pre-scalarized values cache for this call.

        Returns:
            Penalty residual terms, one per constraint.
        """
        if not constraints:
            return []
        scalar_values = values_scalar if values_scalar is not None else self._to_scalar_values(values)
        residuals: list[float] = []
        for constraint in constraints:
            ok = self._constraint_result(constraint, scalar_values)
            if ok is None:
                residuals.append(penalty)
                continue
            residuals.append(0.0 if ok else penalty)
        return residuals

    def _is_block_relation_candidate(
        self,
        rel: object,
        *,
        allow_profiles: bool = False,
    ) -> bool:
        """Return whether one relation is suitable for least-squares block solving.

        Args:
            rel: Relation object to evaluate.

        Returns:
            ``True`` when the relation should be included in block solves.
        """
        # Keep closure block solving scalar-only, but allow profile relations in reconciliation.
        rel_symbols = self.graph.relation_variable_names(rel)
        if not allow_profiles and any(self.graph.variable_ndim(name) == 1 for name in rel_symbols):
            return False

        # Skip implicit multi-output fixed-point bundles (outputs also in inputs).
        # These are expensive and destabilize closure/reconciliation LS loops.
        outputs = tuple(getattr(rel, "outputs", ()) or ())
        inputs = tuple(getattr(rel, "inputs", ()) or ())
        if len(outputs) > 1 and set(outputs).intersection(inputs):
            return False
        return True

    def _least_squares_block(
        self,
        relations: list[object],
        unknowns: list[str],
        values_map: dict[str, object],
        *,
        allow_profiles: bool,
        require_full_rank: bool,
        move_reference: dict[str, object] | None = None,
    ) -> dict[str, object] | None:
        """Solve one relation block with one shared least-squares core."""
        try:
            from scipy.optimize import least_squares
        except Exception:
            return None
        if any(
            not self._is_block_relation_candidate(rel, allow_profiles=allow_profiles)
            for rel in relations
        ):
            return None
        if not unknowns:
            return None

        # Build one dof map: scalar unknowns are absolute, profile unknowns are additive offsets.
        scalar_unknowns: list[str] = []
        profile_unknowns: list[str] = []
        profile_base: dict[str, np.ndarray] = {}
        for name in unknowns:
            if self.graph.variable_ndim(name) == 1:
                if not allow_profiles:
                    continue
                arr = as_profile_array(values_map.get(name))
                if arr is None:
                    derived = self._derive_missing_value(name, values_map)
                    arr = as_profile_array(derived)
                if arr is None:
                    continue
                profile_unknowns.append(name)
                profile_base[name] = arr
                continue
            scalar_unknowns.append(name)
        dof_names = scalar_unknowns + profile_unknowns
        if not dof_names:
            return None

        # Keep only relations closed by known values, unknown dofs, or explicit avg/profile fallbacks.
        unknown_set = set(dof_names)
        active_relations: list[object] = []
        relation_row_count = 0
        for rel in relations:
            rel_symbols = self.graph.relation_variable_names(rel)
            closed = True
            for symbol_name in rel_symbols:
                if symbol_name in unknown_set:
                    continue
                if self._resolve_value_for_name(symbol_name, values_map) is None:
                    closed = False
                    break
            if not closed:
                continue
            active_relations.append(rel)
            relation_row_count += max(len(getattr(rel, "outputs", ()) or ()), 1)
        if not active_relations:
            return None

        # Skip underdetermined scalar-only blocks in closure mode.
        if require_full_rank and not profile_unknowns and relation_row_count < len(scalar_unknowns):
            return None
        if any(
            not any(name in self.graph.relation_variable_names(rel) for rel in active_relations)
            for name in dof_names
        ):
            return None

        penalty = 1e3

        def _to_profile_pair(actual: object, expected: object) -> tuple[np.ndarray | None, np.ndarray | None]:
            """Convert scalar/profile payloads to comparable profile arrays."""
            arr_actual = as_profile_array(actual)
            arr_expected = as_profile_array(expected)
            if arr_actual is not None and arr_expected is not None:
                if arr_actual.shape != arr_expected.shape:
                    return None, None
                return arr_actual, arr_expected
            if arr_actual is not None:
                scalar = safe_float(scalarize_value(expected))
                if scalar is None:
                    return None, None
                return arr_actual, np.full(arr_actual.shape, scalar, dtype=float)
            if arr_expected is not None:
                scalar = safe_float(scalarize_value(actual))
                if scalar is None:
                    return None, None
                return np.full(arr_expected.shape, scalar, dtype=float), arr_expected
            return None, None

        def _normalized_output_residual(actual: object, expected: object) -> float | None:
            """Return one normalized residual term for scalar/profile comparisons."""
            arr_actual, arr_expected = _to_profile_pair(actual, expected)
            if arr_actual is not None and arr_expected is not None:
                scale = np.maximum(np.maximum(np.abs(arr_actual), np.abs(arr_expected)), 1.0)
                delta = (arr_actual - arr_expected) / scale
                return float(np.linalg.norm(delta) / max(delta.size, 1))

            actual_scalar = safe_float(scalarize_value(actual))
            expected_scalar = safe_float(scalarize_value(expected))
            if actual_scalar is None or expected_scalar is None:
                return None
            scale = max(abs(actual_scalar), abs(expected_scalar), 1.0)
            return float((actual_scalar - expected_scalar) / scale)

        def _merged_from_x(x: list[float]) -> dict[str, object]:
            """Build one merged values map from scalar/profile dofs."""
            merged = dict(values_map)
            for idx, name in enumerate(scalar_unknowns):
                merged[name] = float(x[idx])
            for idx, name in enumerate(profile_unknowns):
                delta = float(x[len(scalar_unknowns) + idx])
                merged[name] = profile_base[name] + delta
            return merged

        def _relation_residual_terms(rel: object, merged: dict[str, object]) -> list[float]:
            """Return normalized residual terms for one relation."""
            rel_values = dict(merged)
            for symbol_name in self.graph.relation_variable_names(rel):
                if rel_values.get(symbol_name) is not None:
                    continue
                derived = self._derive_missing_value(symbol_name, rel_values)
                if derived is None:
                    return [penalty]
                rel_values[symbol_name] = derived

            terms: list[float] = []
            if len(rel.outputs) > 1:
                try:
                    solved = rel.apply(rel_values)
                except Exception:
                    return [penalty]
                if not solved:
                    return [penalty]
                for output_name in rel.outputs:
                    residual_term = _normalized_output_residual(
                        rel_values.get(output_name),
                        solved.get(output_name),
                    )
                    if residual_term is None:
                        return [penalty]
                    terms.append(residual_term)
                return terms

            target_name = rel.outputs[0]
            try:
                expected = rel.evaluate(rel_values)
            except Exception:
                return [penalty]
            residual_term = _normalized_output_residual(
                rel_values.get(target_name),
                expected,
            )
            if residual_term is None:
                return [penalty]
            return [residual_term]

        def F(x: list[float]) -> list[float]:
            """Return stacked residual vector for least-squares optimization."""
            merged = _merged_from_x(x)
            for rel in active_relations:
                for symbol_name in self.graph.relation_variable_names(rel):
                    if merged.get(symbol_name) is not None:
                        continue
                    derived = self._derive_missing_value(symbol_name, merged)
                    if derived is not None:
                        merged[symbol_name] = derived

            values_scalar = self._to_scalar_values(merged)
            residuals: list[float] = []
            for rel in active_relations:
                residuals.extend(_relation_residual_terms(rel, merged))
            for name in dof_names:
                var = self.graph.variable(name)
                variable_constraints = tuple(getattr(var, "constraints", ()) or ()) if var is not None else ()
                residuals += self._constraint_residuals(
                    variable_constraints,
                    merged,
                    penalty=penalty,
                    values_scalar=values_scalar,
                )
            for rel in active_relations:
                residuals += self._constraint_residuals(
                    tuple(getattr(rel, "constraints", ()) or ()),
                    merged,
                    penalty=penalty,
                    values_scalar=values_scalar,
                )
            if move_reference:
                movement_scale = 1e-3
                for idx, name in enumerate(scalar_unknowns):
                    target = float(x[idx])
                    ref_scalar = safe_float(move_reference.get(name)) if move_reference else None
                    if ref_scalar is None:
                        ref_scalar = safe_float(values_map.get(name))
                    if ref_scalar is None:
                        continue
                    residuals.append(movement_scale * relative_change(ref_scalar, target))
                for idx, name in enumerate(profile_unknowns):
                    base = profile_base.get(name)
                    if base is None:
                        continue
                    new_arr = base + float(x[len(scalar_unknowns) + idx])
                    scale = max(abs(float(np.mean(base))), 1.0)
                    delta = (new_arr - base) / scale
                    residuals.append(movement_scale * float(np.linalg.norm(delta) / max(delta.size, 1)))

            return residuals if residuals else [penalty]

        # Build x0 and bounds for scalar/profile dofs.
        lower: list[float] = []
        upper: list[float] = []
        base: list[float] = []
        for name in scalar_unknowns:
            guess_scalar = safe_float(values_map.get(name))
            if guess_scalar is None:
                for rel in active_relations:
                    guess_fn = rel.initial_guesses.get(name) if rel.initial_guesses else None
                    if guess_fn is None:
                        continue
                    try:
                        guess_scalar = safe_float(guess_fn(values_map))
                    except Exception:
                        guess_scalar = None
                    if guess_scalar is not None:
                        break
            base.append(guess_scalar if guess_scalar is not None else 1.0)
            lo, hi = self.variable_bounds.get(name, (None, None))
            lower.append(-math.inf if lo is None else float(lo))
            upper.append(math.inf if hi is None else float(hi))
        for _ in profile_unknowns:
            base.append(0.0)
            lower.append(-math.inf)
            upper.append(math.inf)

        # Run multi-start for scalar blocks; one start is enough for mixed profile blocks.
        start_scales = (1e-3, 1e-1, 1.0, 1e1, 1e3) if not profile_unknowns else (1.0,)
        best_x: np.ndarray | None = None
        best_cost: float | None = None
        for scale in start_scales:
            x0: list[float] = []
            for idx, guess in enumerate(base):
                if idx < len(scalar_unknowns):
                    val = guess * scale
                else:
                    val = guess
                lo = lower[idx]
                hi = upper[idx]
                if not math.isfinite(val):
                    val = guess
                if math.isfinite(lo):
                    val = max(val, lo)
                if math.isfinite(hi):
                    val = min(val, hi)
                x0.append(float(val))

            try:
                result = least_squares(
                    F,
                    x0,
                    method="trf",
                    loss="soft_l1",
                    x_scale="jac",
                    max_nfev=200,
                    bounds=(lower, upper),
                )
            except Exception:
                continue
            if not result.success:
                continue

            if require_full_rank and not profile_unknowns and scalar_unknowns:
                try:
                    jacobian = np.asarray(result.jac, dtype=float)
                except Exception:
                    continue
                if jacobian.ndim != 2 or jacobian.shape[1] < len(scalar_unknowns):
                    continue
                relation_jacobian = jacobian[: min(relation_row_count, jacobian.shape[0]), : len(scalar_unknowns)]
                try:
                    relation_rank = int(np.linalg.matrix_rank(relation_jacobian))
                except Exception:
                    continue
                if relation_rank < len(scalar_unknowns):
                    continue

            if best_cost is None or float(result.cost) < best_cost:
                best_cost = float(result.cost)
                best_x = np.asarray(result.x, dtype=float)

        if best_x is None:
            return None

        merged = _merged_from_x(best_x.tolist())
        solved: dict[str, object] = {name: merged.get(name) for name in dof_names}
        return solved if solved else None

    def _solve_block(
        self,
        relations: list[object],
        unknowns: list[str],
        values_map: dict[str, object],
        *,
        move_reference: dict[str, float] | None = None,
    ) -> dict[str, object] | None:
        """Solve one relation block for unknowns (square or rectangular)."""
        if not relations or not unknowns:
            return None
        allow_profiles = any(self.graph.variable_ndim(name) == 1 for name in unknowns) or any(
            any(self.graph.variable_ndim(name) == 1 for name in self.graph.relation_variable_names(rel))
            for rel in relations
        )
        if any(
            not self._is_block_relation_candidate(rel, allow_profiles=allow_profiles)
            for rel in relations
        ):
            return None

        if len(unknowns) == 1 and len(relations) == 1 and self.graph.variable_ndim(unknowns[0]) == 0:
            rel = relations[0]
            unknown = unknowns[0]
            solved = self._solve_for_value(rel, unknown, values_map, prefer_eval_output=True)
            return None if solved is None else {unknown: solved}

        return self._least_squares_block(
            relations,
            unknowns,
            values_map,
            allow_profiles=allow_profiles,
            require_full_rank=not allow_profiles,
            move_reference=move_reference,
        )

    def _start_pass(self) -> None:
        """Start a new override pass and seed inputs. Args: none. Returns: None."""
        self._relation_status_cache = {}
        
        for name in self.graph.variable_names():
            var = self.graph.variable(name)
            if var is None or var.input_source is None:
                continue
            
            override = self._overrides.get(name)
            base = override if override is not None else var.input_value
            if base is None:
                continue
            
            reason = "override" if override is not None else "input"
            self._set_value(name, base, reason=reason)

    def _is_adjustable_variable(self, name: str) -> bool:
        """Return whether a variable can be adjusted by solver reconciliation."""
        var = self.graph.variable(name)
        if var is None:
            return True
        return not var.fixed

    def _violated_decidable_relations(
        self,
        values: dict[str, object],
        rels: Iterable[object] | None = None,
    ) -> set[object]:
        """Return violated relations that are decidable under current values."""
        rel_list = list(self.relations) if rels is None else list(rels)
        violated: set[object] = set()
        for rel in rel_list:
            status, _ = self._relation_status(rel, values)
            if status == "VIOLATED":
                violated.add(rel)
        return violated

    def _run_relation_step(
        self,
        rel: object,
    ) -> bool:
        """Attempt one forward/inverse relation step."""
        # Resolve direct values first, then explicit profile/average fallbacks.
        runtime_values = self._values_dict()
        rel_values: dict[str, object] = {}
        rel_vars = self.graph.relation_variable_names(rel)
        for name in rel_vars:
            resolved = self._resolve_value_for_name(name, runtime_values)
            if resolved is None:
                rel_values[name] = None
                continue
            rel_values[name] = resolved
            if runtime_values.get(name) is None:
                self._set_value(name, resolved, reason="derived_relation_input")
                runtime_values[name] = resolved
        missing = [name for name in rel_vars if rel_values.get(name) is None]
        target_name = rel.outputs[0]
        relation_applied = False

        if len(rel.outputs) > 1:
            missing_inputs = [name for name in rel.inputs if rel_values.get(name) is None]
            missing_outputs = [name for name in rel.outputs if rel_values.get(name) is None]
            self_updates_outputs = bool(set(rel.outputs).intersection(rel.inputs))
            should_apply_bundle = False
            if not missing_inputs and missing_outputs:
                should_apply_bundle = True
            elif not missing_inputs and not missing_outputs:
                status, _residual = self._relation_status(rel, rel_values)
                should_apply_bundle = status == "VIOLATED"
                if (
                    should_apply_bundle
                    and self_updates_outputs
                    and rel in self._bundle_violation_applied
                ):
                    should_apply_bundle = False

            # Recompute multi-output bundles when they can fill missing outputs or are violated.
            if should_apply_bundle:
                try:
                    solved_bundle = rel.apply(rel_values)
                except Exception:
                    solved_bundle = None
                if solved_bundle:
                    candidate = {
                        name: solved_bundle[name]
                        for name in rel.outputs
                        if name in solved_bundle
                    }
                    if self_updates_outputs and not missing_outputs and candidate:
                        # For implicit fixed-point bundles (outputs also used as inputs),
                        # accept only strict lexicographic global improvements.
                        base_values = self._values_dict()
                        baseline_objective = (
                            len(self._violated_decidable_relations(base_values)),
                            self._normalized_residual_norm(base_values, self.relations),
                            0.0,
                        )
                        candidate_values = dict(base_values)
                        candidate_values.update(candidate)
                        movement = 0.0
                        for name, value in candidate.items():
                            old_value = base_values.get(name)
                            if self.graph.variable_ndim(name) == 1:
                                old_arr = as_profile_array(old_value)
                                new_arr = as_profile_array(value)
                                if old_arr is None or new_arr is None or old_arr.shape != new_arr.shape:
                                    continue
                                scale = max(abs(float(np.mean(old_arr))), 1.0)
                                delta = (new_arr - old_arr) / scale
                                movement += float(np.linalg.norm(delta) / max(delta.size, 1)) ** 2
                                continue
                            old_scalar = safe_float(old_value)
                            new_scalar = safe_float(value)
                            if old_scalar is None or new_scalar is None:
                                continue
                            movement += relative_change(old_scalar, new_scalar) ** 2
                        objective = (
                            len(self._violated_decidable_relations(candidate_values)),
                            self._normalized_residual_norm(candidate_values, self.relations),
                            movement,
                        )
                        if objective >= baseline_objective:
                            return False
                    relation_applied = self._accept_candidate_values(
                        candidate,
                        rels=[rel],
                        reason="relation",
                        relation=rel.name,
                        warn_input=not self_updates_outputs,
                        check_violation_increase=False,
                    )
                    if relation_applied and self_updates_outputs and not missing_outputs:
                        self._bundle_violation_applied.add(rel)
            return relation_applied

        if missing:
            if len(missing) != 1:
                return False
            missing_var = missing[0]
            if target_name is not None and missing_var == target_name:
                try:
                    solved_value = rel.evaluate(rel_values)
                except Exception:
                    solved_value = None
                if solved_value is not None:
                    relation_applied = self._accept_candidate_values(
                        {target_name: solved_value},
                        rels=[rel],
                        reason="relation",
                        relation=rel.name,
                        warn_input=True,
                        check_violation_increase=False,
                    )
            elif not (
                target_name is not None
                and missing_var != target_name
                and rel_values.get(target_name) is None
            ):
                known_values = {
                    name: value
                    for name, value in rel_values.items()
                    if name != missing_var and value is not None
                }
                try:
                    solved_value = self._solve_for_value(rel, missing_var, known_values)
                except Exception:
                    solved_value = None
                solved_scalar = safe_float(solved_value)
                if solved_scalar is not None:
                    relation_applied = self._accept_candidate_values(
                        {missing_var: solved_scalar},
                        rels=[rel],
                        reason="relation_inverse",
                        relation=rel.name,
                        warn_input=False,
                        # Keep one-unknown inverse fills for currently-missing variables.
                        # These fills should expose inconsistency instead of being blocked.
                        check_violation_increase=False,
                    )
            return relation_applied

        if target_name is None:
            return False

        try:
            solved_value = rel.evaluate(rel_values)
        except Exception:
            solved_value = None
        if solved_value is None:
            return False
        return self._accept_candidate_values(
            {target_name: solved_value},
            rels=[rel],
            reason="relation",
            relation=rel.name,
            warn_input=True,
            check_violation_increase=True,
        )

    def _run_closure_phase(self) -> dict[str, int]:
        """Run fixed-point closure with explicit n=0..n_max sweeps."""
        stats = {
            "new_assignments": 0,
            "block_solves": 0,
            "immediate_reconciliation": 0,
            "verification_passes": 0,
            "closure_sweeps": 0,
            "closure_cycle_breaks": 0,
            "underconstrained_relations": 0,
            "overconstrained_relations": 0,
            "multiwriter_relations": 0,
        }
        self._bundle_violation_applied = set()
        self._relation_status_cache = {}
        seen_sweep_fingerprints: set[tuple[tuple[tuple[str, float | None], ...], tuple[str, ...]]] = set()
        max_closure_sweeps = max(8, 2 * max(int(self.max_passes), 1))

        while True:
            stats["closure_sweeps"] += 1
            assignments_before = int(self.last_result["metrics"].get("new_assignments_total", 0))
            sweep_progress = False

            values = self._values_dict()
            structural_plan = self._closure_structural_plan(values)
            over_relations = structural_plan["over_relations"]
            multiwriter_relations = structural_plan["multiwriter_relations"]
            stats["underconstrained_relations"] += int(len(structural_plan["under_relations"]))
            stats["overconstrained_relations"] += int(len(over_relations))
            stats["multiwriter_relations"] += int(len(multiwriter_relations))

            # n=1: apply one-missing-variable closure only on single-writer output paths.
            n1_progress = False
            for rel in self.relations:
                if rel in multiwriter_relations:
                    outputs = tuple(getattr(rel, "outputs", ()) or ())
                    # Allow one initial seed for missing multi-writer outputs,
                    # but avoid overwrite ping-pong once values are present.
                    current_values = self._values_dict()
                    if any(name in current_values for name in outputs):
                        continue
                if self._run_relation_step(rel):
                    n1_progress = True
            if n1_progress:
                sweep_progress = True
                values = self._values_dict()

            # n=2..n_max: solve coupled structural blocks by increasing width.
            for width in range(2, max(int(self.n_max), 1) + 1):
                values = self._values_dict()
                blocks = self._structural_closure_blocks(values, width=width)
                if not blocks:
                    continue
                for rel_subset, unknowns in blocks:
                    if len(unknowns) < 2 or len(unknowns) > width:
                        continue
                    solved = self._solve_block(rel_subset, unknowns, values)
                    if not solved:
                        continue
                    if self._accept_candidate_values(
                        solved,
                        rels=rel_subset,
                        reason="solve_component",
                        relation=[rel.name for rel in rel_subset],
                        values_map=values,
                        warn_input=False,
                        check_violation_increase=False,
                    ):
                        sweep_progress = True
                        stats["block_solves"] += 1
                        values = self._values_dict()
                self.last_result["metrics"]["closure_structural_blocks"] = int(
                    self.last_result["metrics"].get("closure_structural_blocks", 0)
                ) + int(len(blocks))

            # Overconstrained/multi-writer violated blocks go straight to reconciliation.
            values = self._values_dict()
            conflicted_violated = {
                rel
                for rel in self._violated_decidable_relations(values)
                if rel in over_relations or rel in multiwriter_relations
            }
            if conflicted_violated and int(self.max_passes) > 0:
                immediate = self._run_reconciliation_phase(
                    values,
                    conflicted_violated,
                    focus_relations=conflicted_violated,
                    allow_direct=False,
                    max_rounds_override=1,
                    restart_pass=False,
                )
                if immediate > 0:
                    stats["immediate_reconciliation"] += int(immediate)
                    sweep_progress = True

            # n=0: verify all decidable relations after n=1..n_max updates.
            values = self._values_dict()
            self.last_result["violated_relations"] = self._violated_relations(values)
            stats["verification_passes"] += 1

            assignments_after = int(self.last_result["metrics"].get("new_assignments_total", 0))
            delta = max(assignments_after - assignments_before, 0)
            stats["new_assignments"] += delta
            sweep_fingerprint = self._solve_fingerprint(
                values,
                self._violated_decidable_relations(values),
            )
            if sweep_fingerprint in seen_sweep_fingerprints:
                stats["closure_cycle_breaks"] += 1
                break
            seen_sweep_fingerprints.add(sweep_fingerprint)
            if stats["closure_sweeps"] >= max_closure_sweeps:
                stats["closure_cycle_breaks"] += 1
                break
            if delta == 0 and not sweep_progress:
                break

        return stats

    def _run_reconciliation_phase(
        self,
        values: dict[str, object],
        violated_decidable: set[object],
        *,
        focus_relations: set[object] | None = None,
        allow_direct: bool = True,
        max_rounds_override: int | None = None,
        restart_pass: bool = True,
    ) -> int:
        """Run lexicographic reconciliation on violated decidable components."""
        if not violated_decidable:
            return 0

        # Objective priority: violated count, residual norm, movement.
        successes = 0
        if max_rounds_override is not None:
            max_rounds = max(int(max_rounds_override), 1)
        else:
            max_rounds = min(max(len(violated_decidable), 1), max(int(self.max_passes), 1))
        focus_lookup = set(focus_relations) if focus_relations else None

        for _round_idx in range(max_rounds):
            values = self._values_dict()
            self._relation_status_cache = {}
            current_violated = self._violated_decidable_relations(values)
            if focus_lookup is not None:
                current_violated = {rel for rel in current_violated if rel in focus_lookup}
            if not current_violated:
                break

            baseline_count = len(current_violated)
            baseline_norm = self._normalized_residual_norm(values, self.relations)
            baseline_objective = (baseline_count, baseline_norm, 0.0)

            # Step 1: attempt one-variable direct culprit correction first.
            best_direct: tuple[tuple[int, float, float], object, dict[str, float], str] | None = None
            if allow_direct:
                for rel in sorted(current_violated, key=lambda item: getattr(item, "name", item.outputs[0])):
                    self.last_result["metrics"]["reconciliation_direct_attempts"] = int(
                        self.last_result["metrics"].get("reconciliation_direct_attempts", 0)
                    ) + 1
                    culprit = self._culprit_for_relation(rel, values)
                    if culprit is None:
                        continue
                    name, _change, target = culprit
                    if not self._is_adjustable_variable(name) or self.graph.variable_ndim(name) == 1:
                        continue

                    current_scalar = safe_float(values.get(name))
                    target_scalar = safe_float(target)
                    if current_scalar is None or target_scalar is None:
                        continue
                    var_tol = self.graph.variable(name)
                    rel_tol = var_tol.rel_tol if var_tol is not None and var_tol.rel_tol is not None else self.default_rel_tol
                    if within_tolerance(current_scalar, target_scalar, rel_tol=float(rel_tol)):
                        continue

                    solved = {name: target_scalar}
                    candidate_values = dict(values)
                    candidate_values.update(solved)
                    if self._constraints_violated(candidate_values, rel=rel, names=[name]):
                        continue

                    objective = (
                        len(self._violated_decidable_relations(candidate_values)),
                        self._normalized_residual_norm(candidate_values, self.relations),
                        relative_change(current_scalar, target_scalar) ** 2,
                    )
                    if objective >= baseline_objective:
                        continue
                    if best_direct is None or objective < best_direct[0]:
                        best_direct = (
                            objective,
                            rel,
                            solved,
                            str(getattr(rel, "name", name)),
                        )

            if best_direct is not None:
                objective, rel, solved, relation_name = best_direct
                accepted = self._accept_candidate_values(
                    solved,
                    rels=[rel],
                    reason="reconcile",
                    relation=relation_name,
                    values_map=values,
                    warn_input=False,
                    check_violation_increase=False,
                )
                if not accepted:
                    break
                name, value = next(iter(solved.items()))
                self._overrides[name] = value
                successes += 1
                self._log.info(
                    (
                        "Reconciliation direct step adjusted %s to %.6g "
                        "(violated=%s, residual_norm=%.3g, movement=%.3g)."
                    ),
                    name,
                    float(value),
                    objective[0],
                    objective[1],
                    objective[2],
                )
                continue

            # Step 2: if direct step fails, run structural block reconciliation.
            components = self._structural_reconciliation_components(values, current_violated)
            self.last_result["metrics"]["reconciliation_structural_components"] = int(
                self.last_result["metrics"].get("reconciliation_structural_components", 0)
            ) + int(len(components))
            best_block: tuple[tuple[int, float, float], list[object], dict[str, object]] | None = None

            for rels, unknowns in components[: max(int(self.n_max), 1) * 2]:
                self.last_result["metrics"]["reconciliation_block_attempts"] = int(
                    self.last_result["metrics"].get("reconciliation_block_attempts", 0)
                ) + 1
                baseline: dict[str, object] = {}
                for name in unknowns:
                    if self.graph.variable_ndim(name) == 1:
                        arr = as_profile_array(values.get(name))
                        if arr is not None:
                            baseline[name] = arr
                        continue
                    scalar = safe_float(values.get(name))
                    if scalar is not None:
                        baseline[name] = scalar
                if len(baseline) < 1:
                    continue
                solved = self._solve_block(
                    rels,
                    sorted(baseline),
                    values,
                    move_reference=baseline,
                )
                if not solved:
                    continue
                solved_values: dict[str, object] = {}
                for name, value in solved.items():
                    if self.graph.variable_ndim(name) == 1:
                        arr = as_profile_array(value)
                        if arr is not None:
                            solved_values[name] = arr
                        continue
                    scalar = safe_float(value)
                    if scalar is not None:
                        solved_values[name] = scalar
                if not solved_values:
                    continue

                candidate_values = dict(values)
                candidate_values.update(solved_values)
                if any(
                    self._constraints_violated(candidate_values, rel=rel, names=solved_values.keys())
                    for rel in rels
                ):
                    continue

                movement = 0.0
                for name, new_value in solved_values.items():
                    old_value = baseline.get(name)
                    if old_value is None:
                        continue
                    if self.graph.variable_ndim(name) == 1:
                        old_arr = as_profile_array(old_value)
                        new_arr = as_profile_array(new_value)
                        if old_arr is None or new_arr is None or old_arr.shape != new_arr.shape:
                            continue
                        scale = max(abs(float(np.mean(old_arr))), 1.0)
                        delta = (new_arr - old_arr) / scale
                        move_score = float(np.linalg.norm(delta) / max(delta.size, 1))
                    else:
                        old_scalar = safe_float(old_value)
                        new_scalar = safe_float(new_value)
                        if old_scalar is None or new_scalar is None:
                            continue
                        move_score = relative_change(old_scalar, new_scalar)
                    movement += move_score**2
                objective = (
                    len(self._violated_decidable_relations(candidate_values)),
                    self._normalized_residual_norm(candidate_values, self.relations),
                    movement,
                )
                if objective >= baseline_objective:
                    continue
                if best_block is None or objective < best_block[0]:
                    best_block = (
                        objective,
                        rels,
                        solved_values,
                    )

            if best_block is None:
                break

            objective, rels, solved_values = best_block
            accepted = self._accept_candidate_values(
                solved_values,
                rels=rels,
                reason="reconcile",
                relation=[getattr(rel, "name", rel.outputs[0]) for rel in rels],
                values_map=values,
                warn_input=False,
                check_violation_increase=False,
            )
            if not accepted:
                break
            for name, value in solved_values.items():
                self._overrides[name] = value
            successes += 1
            self._log.info(
                (
                    "Reconciliation block step adjusted %s vars "
                    "(violated=%s, residual_norm=%.3g, movement=%.3g)."
                ),
                len(solved_values),
                objective[0],
                objective[1],
                objective[2],
            )

        if successes and restart_pass:
            self._start_pass()
        return successes

    def _solve_fingerprint(
        self,
        values: dict[str, object],
        violated_decidable: set[object],
    ) -> tuple[tuple[tuple[str, float | None], ...], tuple[str, ...]]:
        """Return compact solver fingerprint for cycle detection."""
        adjustable_values: list[tuple[str, float | None]] = []
        for name in self.graph.variable_names():
            if not self._is_adjustable_variable(name):
                continue
            scalar = safe_float(scalarize_value(values.get(name)))
            rounded = None if scalar is None else float(f"{scalar:.12g}")
            adjustable_values.append((name, rounded))
        violated_names = tuple(
            sorted(
                getattr(rel, "name", rel.outputs[0])
                for rel in violated_decidable
            )
        )
        return tuple(adjustable_values), violated_names

    def _relation_status_rows(
        self,
        values: dict[str, object],
        rels: Iterable[object] | None = None,
    ) -> list[tuple[object, str, str, float | None]]:
        """Return one deterministic relation status table for a values mapping."""
        rows: list[tuple[object, str, str, float | None]] = []
        rel_list = list(self.relations) if rels is None else list(rels)
        for rel in rel_list:
            rel_name = getattr(rel, "name", rel.outputs[0])
            status, residual = self._relation_status(rel, values)
            rows.append((rel, rel_name, status, residual))
        return rows

    def _run_final_network_verification(self, values: dict[str, object]) -> dict[str, object]:
        """Verify every relation on final values and persist terminal status."""
        relation_status: list[tuple[str, str, float | None]] = []
        violated: set[object] = set()
        sat_count = 0
        violated_count = 0
        undecidable_count = 0

        for rel, rel_name, status, residual in self._relation_status_rows(values):
            relation_status.append((rel_name, status, residual))
            if status == "SAT":
                sat_count += 1
                continue
            if status == "VIOLATED":
                violated.add(rel)
                violated_count += 1
                continue
            undecidable_count += 1

        summary = {
            "relation_status": relation_status,
            "relations_checked": len(self.relations),
            "sat_count": sat_count,
            "violated_count": violated_count,
            "undecidable_count": undecidable_count,
            "all_satisfied": violated_count == 0 and undecidable_count == 0,
        }
        self.last_result["violated_relations"] = violated
        self.last_result["final_check"] = summary
        self.last_result["metrics"]["final_check_relations"] = len(self.relations)
        self.last_result["metrics"]["final_check_sat_count"] = sat_count
        self.last_result["metrics"]["final_check_violated_count"] = violated_count
        self.last_result["metrics"]["final_check_undecidable_count"] = undecidable_count
        self.last_result["metrics"]["final_check_all_satisfied"] = bool(summary["all_satisfied"])
        return summary

    def solve(self) -> None:
        """Solve the relation system in-place using closure and reconciliation phases.

        Args:
            None.

        Returns:
            None.
        """
        self.last_result["stop_reason"] = None
        self.last_result["metrics"] = {
            "new_assignments_total": 0,
            "iterations": 0,
            "new_assignments": 0,
            "block_solves": 0,
            "closure_immediate_reconciliation": 0,
            "closure_verification_passes": 0,
            "closure_sweeps": 0,
            "closure_cycle_breaks": 0,
            "closure_structural_blocks": 0,
            "closure_underconstrained_relations": 0,
            "closure_overconstrained_relations": 0,
            "closure_multiwriter_relations": 0,
            "structural_decompositions": 0,
            "reconciliation_success": 0,
            "reconciliation_direct_attempts": 0,
            "reconciliation_block_attempts": 0,
            "reconciliation_structural_components": 0,
            "violated_decidable_count": 0,
            "final_check_relations": 0,
            "final_check_sat_count": 0,
            "final_check_violated_count": 0,
            "final_check_undecidable_count": 0,
            "final_check_all_satisfied": False,
        }

        # Check mode runs one validation pass on current values without any writes.
        if self.mode == "check":
            self._relation_status_cache = {}
            final_values = self._values_dict()
            final_check = self._run_final_network_verification(final_values)
            self.last_result["metrics"]["violated_decidable_count"] = len(
                self._violated_decidable_relations(final_values)
            )
            if int(final_check["violated_count"]) > 0:
                self.last_result["stop_reason"] = "final_check_violated"
            elif int(final_check["undecidable_count"]) > 0:
                self.last_result["stop_reason"] = "final_check_undecidable"
            else:
                self.last_result["stop_reason"] = "converged"
            return

        seen_fingerprints: set[tuple[tuple[tuple[str, float | None], ...], tuple[str, ...]]] = set()
        previous_violated_count: int | None = None
        reconciliation_attempts = 0
        max_iterations = max(8, 4 * max(int(self.max_passes), 1))

        while True:
            self.last_result["metrics"]["iterations"] = int(self.last_result["metrics"]["iterations"]) + 1
            self._relation_status_cache = {}

            closure = self._run_closure_phase()
            self.last_result["metrics"]["new_assignments"] = closure["new_assignments"]
            self.last_result["metrics"]["block_solves"] = closure["block_solves"]
            self.last_result["metrics"]["closure_immediate_reconciliation"] = int(
                self.last_result["metrics"].get("closure_immediate_reconciliation", 0)
            ) + int(closure.get("immediate_reconciliation", 0))
            self.last_result["metrics"]["closure_verification_passes"] = int(
                self.last_result["metrics"].get("closure_verification_passes", 0)
            ) + int(closure["verification_passes"])
            self.last_result["metrics"]["closure_sweeps"] = int(
                self.last_result["metrics"].get("closure_sweeps", 0)
            ) + int(closure["closure_sweeps"])
            self.last_result["metrics"]["closure_cycle_breaks"] = int(
                self.last_result["metrics"].get("closure_cycle_breaks", 0)
            ) + int(closure["closure_cycle_breaks"])
            self.last_result["metrics"]["closure_underconstrained_relations"] = int(
                self.last_result["metrics"].get("closure_underconstrained_relations", 0)
            ) + int(closure.get("underconstrained_relations", 0))
            self.last_result["metrics"]["closure_overconstrained_relations"] = int(
                self.last_result["metrics"].get("closure_overconstrained_relations", 0)
            ) + int(closure.get("overconstrained_relations", 0))
            self.last_result["metrics"]["closure_multiwriter_relations"] = int(
                self.last_result["metrics"].get("closure_multiwriter_relations", 0)
            ) + int(closure.get("multiwriter_relations", 0))

            values = self._values_dict()
            violated_decidable = self._violated_decidable_relations(values)
            violated_count = len(violated_decidable)
            self.last_result["violated_relations"] = self._violated_relations(values)
            self.last_result["metrics"]["violated_decidable_count"] = violated_count

            reconciliation_success = 0
            if violated_count and reconciliation_attempts < self.max_passes:
                reconciliation_attempts += 1
                reconciliation_success = self._run_reconciliation_phase(values, violated_decidable)
            self.last_result["metrics"]["reconciliation_success"] = reconciliation_success

            values = self._values_dict()
            violated_decidable = self._violated_decidable_relations(values)
            violated_count = len(violated_decidable)
            self.last_result["violated_relations"] = self._violated_relations(values)
            self.last_result["metrics"]["violated_decidable_count"] = violated_count
            decreased_violations = (
                previous_violated_count is not None and violated_count < previous_violated_count
            )

            if violated_count == 0 and reconciliation_success == 0:
                self.last_result["stop_reason"] = "converged"
                break

            stalled = (
                closure["new_assignments"] == 0
                and closure["block_solves"] == 0
                and closure.get("immediate_reconciliation", 0) == 0
                and reconciliation_success == 0
                and not decreased_violations
            )
            if stalled:
                if violated_count > 0 and reconciliation_attempts >= self.max_passes:
                    self.last_result["stop_reason"] = "max_passes_reached"
                elif violated_count > 0:
                    self.last_result["stop_reason"] = "stalled"
                else:
                    self.last_result["stop_reason"] = "converged"
                break

            fingerprint = self._solve_fingerprint(values, violated_decidable)
            if fingerprint in seen_fingerprints:
                self.last_result["stop_reason"] = "cycle_detected"
                break
            seen_fingerprints.add(fingerprint)

            if int(self.last_result["metrics"]["iterations"]) >= max_iterations:
                if violated_count > 0:
                    self.last_result["stop_reason"] = "max_passes_reached"
                else:
                    self.last_result["stop_reason"] = "converged"
                break

            previous_violated_count = violated_count

        final_values = self._values_dict()
        final_check = self._run_final_network_verification(final_values)

        # Refine converged stop reasons when final audit still finds unresolved status.
        if self.last_result["stop_reason"] == "converged":
            if int(final_check["violated_count"]) > 0:
                self.last_result["stop_reason"] = "final_check_violated"
            elif int(final_check["undecidable_count"]) > 0:
                self.last_result["stop_reason"] = "final_check_undecidable"

        if self.last_result["stop_reason"] in ("stalled", "max_passes_reached") and self.last_result["violated_relations"]:
            self._log.warning(
                "Inconsistency cannot be repaired with current adjustable variables; unresolved violated relations remain.",
            )

    def evaluate(
        self,
        values: dict[str, object],
        *,
        chunk_size: int | None = None,
    ) -> dict[str, object]:
        """Evaluate relations on scalar or grid inputs using dense matrix state.

        Args:
            values: Variable mapping with scalar, array, or profile payloads.
            chunk_size: Optional max row chunk for row-wise fallback evaluation.

        Returns:
            Updated values mapping containing computed outputs.
        """
        # Step 0: copy caller payload and check geometry-volume consistency once.
        evaluated = self._apply_explicit_fallbacks(
            {
                name: self._normalize_runtime_value(name, value)
                for name, value in dict(values).items()
            }
        )
        scalar_names = [name for name in self.graph.variable_names() if self.graph.variable_ndim(name) == 0]
        var_index = {name: idx for idx, name in enumerate(scalar_names)}

        # Step 1: register extra scalar keys that appear only in runtime payload.
        for name, value in evaluated.items():
            if name in var_index:
                continue
            if value is None or self.graph.variable_ndim(name) == 1:
                continue
            var_index[name] = len(scalar_names)
            scalar_names.append(name)

        # Step 2: infer one common broadcast shape for matrix evaluation.
        candidate_arrays: list[np.ndarray] = []
        for name in scalar_names:
            raw = evaluated.get(name)
            if raw is None or self.graph.variable_ndim(name) == 1:
                continue
            try:
                arr = np.asarray(raw, dtype=float)
            except Exception:
                continue
            if arr.shape != ():
                candidate_arrays.append(arr)

        if candidate_arrays:
            target_shape = np.broadcast_arrays(*candidate_arrays)[0].shape
            n_points = int(np.prod(target_shape))
        else:
            target_shape = ()
            n_points = 1

        n_vars = len(scalar_names)
        state = np.full((n_points, n_vars), np.nan, dtype=float)
        known = np.zeros((n_points, n_vars), dtype=bool)

        # Step 3: seed dense matrix columns with known scalar values.
        for name, idx in var_index.items():
            raw = evaluated.get(name)
            if raw is None or self.graph.variable_ndim(name) == 1:
                continue
            try:
                arr = np.asarray(raw, dtype=float)
            except Exception:
                continue
            if arr.shape == ():
                state[:, idx] = float(arr)
                known[:, idx] = True
                continue
            try:
                flat = np.broadcast_to(arr, target_shape).reshape(-1)
            except Exception:
                continue
            state[:, idx] = flat
            known[:, idx] = True

        def _profile_to_scalar(value: object) -> float | None:
            """Convert scalar/profile payloads to one scalar for matrix evaluation.

            Args:
                value: Candidate scalar/profile payload.

            Returns:
                One finite scalar when conversion succeeds, else None.
            """
            return safe_float(scalarize_value(value))

        # Step 4: iterate relation passes until no additional writes are possible.
        eval_plan: list[tuple[object, int | None, tuple[str, ...], tuple[int | None, ...]]] = []
        for rel in self.relations:
            if len(rel.outputs) > 1:
                continue
            target_name = rel.outputs[0]
            input_names = tuple(rel.input_names(target_name))
            output_idx = var_index.get(target_name)
            input_idx = tuple(var_index.get(name) for name in input_names)
            eval_plan.append((rel, output_idx, input_names, input_idx))
        max_iter = max(6, len(self.relations) + 1)
        row_step = max(1, chunk_size or n_points)
        for _ in range(max_iter):
            progress = False

            # Step 4.0) Resolve multi-output forward relations first.
            bundle_progress = False
            for rel in self.relations:
                if len(rel.outputs) <= 1:
                    continue

                input_values: dict[str, object] = {}
                valid = True
                for in_name in rel.inputs:
                    value = self._resolve_value_for_name(in_name, evaluated)
                    if value is None:
                        idx = var_index.get(in_name)
                        if idx is None:
                            valid = False
                            break
                        if n_points == 1:
                            if known[0, idx]:
                                value = float(state[0, idx])
                            else:
                                valid = False
                                break
                        else:
                            if np.all(known[:, idx]):
                                value = state[:, idx].copy()
                            else:
                                valid = False
                                break
                    input_values[in_name] = value
                if not valid:
                    continue

                output_names = tuple(rel.outputs)
                if n_points == 1:
                    try:
                        result = rel.apply(input_values)
                    except Exception:
                        result = None
                    if not result:
                        continue
                    wrote_any = False
                    for out_name in output_names:
                        if out_name not in result:
                            continue
                        value = result[out_name]
                        if self.graph.variable_ndim(out_name) == 1:
                            evaluated[out_name] = self._normalize_runtime_value(out_name, value)
                            wrote_any = True
                            continue
                        scalar = _profile_to_scalar(value)
                        if scalar is None or not np.isfinite(scalar):
                            continue
                        out_idx = var_index.get(out_name)
                        if out_idx is None:
                            evaluated[out_name] = scalar
                        else:
                            state[0, out_idx] = scalar
                            known[0, out_idx] = True
                        wrote_any = True
                    if wrote_any:
                        bundle_progress = True
                    continue

                out_buffers = {
                    out_name: np.full(n_points, np.nan, dtype=float)
                    for out_name in output_names
                }
                for row in range(n_points):
                    kwargs_row: dict[str, object] = {}
                    for in_name, value in input_values.items():
                        if isinstance(value, np.ndarray):
                            arr = np.asarray(value)
                            if arr.shape == (n_points,):
                                kwargs_row[in_name] = float(arr[row])
                            elif arr.shape == target_shape:
                                kwargs_row[in_name] = float(arr.reshape(-1)[row])
                            else:
                                kwargs_row[in_name] = arr
                        else:
                            kwargs_row[in_name] = value
                    try:
                        result = rel.apply(kwargs_row)
                    except Exception:
                        result = None
                    if not result:
                        continue
                    for out_name in output_names:
                        if out_name not in result:
                            continue
                        scalar = _profile_to_scalar(result[out_name])
                        if scalar is not None and np.isfinite(scalar):
                            out_buffers[out_name][row] = scalar

                for out_name, out in out_buffers.items():
                    finite = np.isfinite(out)
                    if not np.any(finite):
                        continue
                    if self.graph.variable_ndim(out_name) == 1:
                        evaluated[out_name] = self._normalize_runtime_value(out_name, out)
                    else:
                        out_idx = var_index.get(out_name)
                        if out_idx is None:
                            evaluated[out_name] = out
                        else:
                            state[finite, out_idx] = out[finite]
                            known[finite, out_idx] = True
                    bundle_progress = True

            if bundle_progress:
                progress = True

            # Step 4.1) Resolve profile-target single-output relations.
            profile_progress = False
            for rel in self.relations:
                if len(rel.outputs) > 1:
                    continue
                target_name = rel.outputs[0]
                if self.graph.variable_ndim(target_name) != 1:
                    continue
                if evaluated.get(target_name) is not None:
                    continue

                input_values: dict[str, object] = {}
                valid = True
                for in_name in rel.input_names(target_name):
                    value = self._resolve_value_for_name(in_name, evaluated)
                    if value is None:
                        idx = var_index.get(in_name)
                        if idx is None:
                            valid = False
                            break
                        if n_points == 1:
                            if known[0, idx]:
                                value = float(state[0, idx])
                            else:
                                valid = False
                                break
                        else:
                            if np.all(known[:, idx]):
                                value = state[:, idx].copy()
                            else:
                                valid = False
                                break
                    input_values[in_name] = value
                if not valid:
                    continue

                if n_points == 1:
                    try:
                        result = rel.evaluate(input_values)
                    except Exception:
                        result = None
                    if result is None:
                        continue
                    evaluated[target_name] = self._normalize_runtime_value(target_name, result)
                    profile_progress = True
                    continue

                out = np.full(n_points, np.nan, dtype=float)
                for row in range(n_points):
                    kwargs_row: dict[str, object] = {}
                    for in_name, value in input_values.items():
                        if isinstance(value, np.ndarray):
                            arr = np.asarray(value)
                            if arr.shape == (n_points,):
                                kwargs_row[in_name] = float(arr[row])
                            elif arr.shape == target_shape:
                                kwargs_row[in_name] = float(arr.reshape(-1)[row])
                            else:
                                kwargs_row[in_name] = arr
                        else:
                            kwargs_row[in_name] = value
                    try:
                        result = rel.evaluate(kwargs_row)
                    except Exception:
                        result = None
                    scalar = _profile_to_scalar(result)
                    if scalar is not None and np.isfinite(scalar):
                        out[row] = scalar
                if np.any(np.isfinite(out)):
                    evaluated[target_name] = self._normalize_runtime_value(target_name, out)
                    profile_progress = True

            if profile_progress:
                progress = True

            # Step 4.2) Resolve scalar-target relations using vectorized first, row-wise fallback.
            for rel, output_idx0, input_names, input_idx0 in eval_plan:
                target_name = rel.outputs[0]
                out_idx = var_index.get(target_name)
                if out_idx is None:
                    continue
                if output_idx0 is not None and output_idx0 != out_idx:
                    out_idx = var_index.get(target_name)
                pending = ~known[:, out_idx]
                if not np.any(pending):
                    continue

                scalar_inputs: list[tuple[str, int]] = []
                const_inputs: dict[str, object] = {}
                valid = True
                has_profile_const = False
                for in_name, in_idx in zip(input_names, input_idx0):
                    idx = in_idx if in_idx is not None else var_index.get(in_name)
                    if idx is not None:
                        pending &= known[:, idx]
                        scalar_inputs.append((in_name, idx))
                        continue

                    raw = self._resolve_value_for_name(in_name, evaluated)
                    if raw is None:
                        valid = False
                        break
                    if self.graph.variable_ndim(in_name) == 1:
                        if isinstance(raw, np.ndarray):
                            arr = np.asarray(raw)
                            if arr.shape == target_shape:
                                const_inputs[in_name] = arr.reshape(-1)
                            else:
                                const_inputs[in_name] = arr
                        else:
                            const_inputs[in_name] = raw
                        has_profile_const = True
                    elif isinstance(raw, np.ndarray):
                        try:
                            const_inputs[in_name] = np.broadcast_to(
                                np.asarray(raw, dtype=float), target_shape
                            ).reshape(-1)
                        except Exception:
                            valid = False
                            break
                    else:
                        const_inputs[in_name] = raw
                if not valid:
                    continue

                rows = np.flatnonzero(pending)
                if rows.size == 0:
                    continue

                wrote = False
                for start in range(0, rows.size, row_step):
                    row_ids = rows[start : start + row_step]
                    if not has_profile_const:
                        kwargs: dict[str, object] = {}
                        for in_name, idx in scalar_inputs:
                            kwargs[in_name] = state[row_ids, idx]
                        for in_name, value in const_inputs.items():
                            kwargs[in_name] = value[row_ids] if isinstance(value, np.ndarray) else value

                        try:
                            result = rel.evaluate(kwargs)
                            out = np.asarray(result, dtype=float)
                            if out.shape == ():
                                out = np.full(row_ids.size, float(out), dtype=float)
                            else:
                                out = np.broadcast_to(out, (row_ids.size,)).astype(float, copy=False)
                            state[row_ids, out_idx] = out
                            known[row_ids, out_idx] = True
                            wrote = True
                            continue
                        except Exception:
                            pass

                    out = np.full(row_ids.size, np.nan, dtype=float)
                    for i, row in enumerate(row_ids):
                        kwargs_row: dict[str, object] = {}
                        for in_name, idx in scalar_inputs:
                            kwargs_row[in_name] = float(state[row, idx])
                        for in_name, value in const_inputs.items():
                            if isinstance(value, np.ndarray):
                                if value.shape == (n_points,):
                                    kwargs_row[in_name] = float(value[row])
                                else:
                                    kwargs_row[in_name] = value
                            else:
                                kwargs_row[in_name] = value
                        try:
                            out[i] = float(rel.evaluate(kwargs_row))
                        except Exception:
                            out[i] = np.nan
                    finite = np.isfinite(out)
                    if np.any(finite):
                        state[row_ids[finite], out_idx] = out[finite]
                        known[row_ids[finite], out_idx] = True
                        wrote = True
                if wrote:
                    progress = True
            if not progress:
                break

        # Step 5: materialize matrix columns back into output mapping.
        for name, idx in var_index.items():
            if not known[:, idx].any() and name not in evaluated:
                continue
            if n_points == 1:
                evaluated[name] = None if not known[0, idx] else float(state[0, idx])
            else:
                evaluated[name] = state[:, idx].reshape(target_shape)

        return evaluated

    def _violated_relations(self, values: dict[str, object], rels: list | None = None) -> set[object]:
        """Return the set of violated relations for the given values."""
        return {
            rel
            for rel, _name, status, _residual in self._relation_status_rows(values, rels)
            if status == "VIOLATED"
        }

    def _relation_status(self, rel: object, values: dict[str, object]) -> tuple[str, float | None]:
        """Return (status, residual) for a relation given values."""
        cache = self._relation_status_cache
        rel_symbols = self.graph.relation_variable_names(rel)

        # Resolve explicit profile/average fallbacks before status evaluation.
        resolved_values = dict(values)
        for name in rel_symbols:
            if resolved_values.get(name) is not None:
                continue
            derived = self._derive_missing_value(name, resolved_values)
            if derived is not None:
                resolved_values[name] = derived

        # Build one compact value token per relation variable for cache invalidation.
        cache_tokens: list[tuple[str, tuple[object, ...]]] = []
        for name in rel_symbols:
            value = resolved_values.get(name)
            if value is None:
                token = ("none",)
            else:
                scalar = safe_float(value)
                if scalar is not None:
                    token = ("scalar", float(f"{scalar:.12g}"))
                elif isinstance(value, np.ndarray):
                    shape = tuple(int(dim) for dim in value.shape)
                    size = int(value.size)
                    if size == 0:
                        token = ("array", shape, size, id(value))
                    else:
                        first = safe_float(value.reshape(-1)[0])
                        last = safe_float(value.reshape(-1)[-1])
                        token = ("array", shape, size, id(value), first, last)
                else:
                    token = ("obj", id(value))
            cache_tokens.append((name, token))

        # Reuse relation status results when the relation inputs are unchanged.
        cache_key = (rel, tuple(cache_tokens))
        if cache_key in cache:
            cached = cache.get(cache_key)
            if isinstance(cached, tuple) and len(cached) == 2:
                return cached

        values_scalar = self._to_scalar_values(resolved_values)
        if any(values_scalar.get(name) is None for name in rel_symbols):
            result = ("UNDECIDABLE", None)
            cache[cache_key] = result
            return result

        for constraint in tuple(getattr(rel, "constraints", ()) or ()):
            result = self._constraint_result(constraint, values_scalar)
            if result is None:
                status_result = ("UNDECIDABLE", None)
                cache[cache_key] = status_result
                return status_result
            if result is False:
                status_result = ("VIOLATED", None)
                cache[cache_key] = status_result
                return status_result

        if len(rel.outputs) > 1:
            try:
                solved = rel.apply(resolved_values)
            except Exception:
                status_result = ("UNDECIDABLE", None)
                cache[cache_key] = status_result
                return status_result
            if not solved:
                status_result = ("UNDECIDABLE", None)
                cache[cache_key] = status_result
                return status_result

            residuals: list[float] = []
            for output_name in rel.outputs:
                expected_scalar = safe_float(scalarize_value(solved.get(output_name)))
                actual_scalar = safe_float(values_scalar.get(output_name))
                if expected_scalar is None or actual_scalar is None:
                    status_result = ("UNDECIDABLE", None)
                    cache[cache_key] = status_result
                    return status_result
                residual = actual_scalar - expected_scalar
                residuals.append(residual)
                var_tol = self.graph.variable(output_name)
                rel_tol = var_tol.rel_tol if var_tol is not None and var_tol.rel_tol is not None else self.default_rel_tol
                if not within_tolerance(actual_scalar, expected_scalar, rel_tol=float(rel_tol)):
                    status_result = ("VIOLATED", max(abs(item) for item in residuals))
                    cache[cache_key] = status_result
                    return status_result
            status_result = ("SAT", max((abs(item) for item in residuals), default=0.0))
            cache[cache_key] = status_result
            return status_result

        try:
            expected_scalar = safe_float(scalarize_value(rel.evaluate(resolved_values)))
        except Exception:
            expected_scalar = None
        if expected_scalar is None:
            status_result = ("UNDECIDABLE", None)
            cache[cache_key] = status_result
            return status_result
        target_name = rel.outputs[0]
        actual_scalar = safe_float(values_scalar.get(target_name))
        if actual_scalar is None:
            status_result = ("UNDECIDABLE", None)
            cache[cache_key] = status_result
            return status_result
        residual = actual_scalar - expected_scalar
        var_tol = self.graph.variable(target_name)
        rel_tol = var_tol.rel_tol if var_tol is not None and var_tol.rel_tol is not None else self.default_rel_tol
        status = (
            "SAT"
            if within_tolerance(actual_scalar, expected_scalar, rel_tol=float(rel_tol))
            else "VIOLATED"
        )
        status_result = (status, residual)
        cache[cache_key] = status_result
        return status_result


    def _culprit_for_relation(
        self,
        rel: object,
        values: dict[str, object],
    ) -> tuple[str, float, float] | None:
        """
        Identify a variable adjustment that can satisfy one violated relation.

        Returns:
            Tuple of (variable_name, relative_change, target_value) for the best culprit,
            or None if no suitable culprit found.
        """
        rel_vars = self.graph.relation_variable_names(rel)
        if any(values.get(name) is None for name in rel_vars):
            self._log.debug(
                "culprit_for_relation(%s): missing values for some variables",
                rel.name,
            )
            return None

        best: tuple[str, float, float] | None = None
        best_key: tuple[float, str] | None = None
        try:
            exp_scalar = safe_float(rel.evaluate(values))
        except Exception:
            exp_scalar = None
        if exp_scalar is None:
            return None
        target_name = rel.outputs[0]
        act_scalar = safe_float(values.get(target_name))
        if act_scalar is None:
            return None
        residual = act_scalar - exp_scalar
        var_tol = self.graph.variable(target_name)
        rel_tol = var_tol.rel_tol if var_tol is not None and var_tol.rel_tol is not None else self.default_rel_tol
        if within_tolerance(act_scalar, exp_scalar, rel_tol=float(rel_tol)):
            return None

        # Rank by smallest relative movement in current runtime values.
        for name in rel_vars:
            var = self.graph.variable(name)
            if var is None or var.fixed:
                continue

            current = safe_float(values.get(name))
            if current is None:
                continue

            target_scalar = None
            if name == target_name:
                target_scalar = exp_scalar
            elif name in rel.inputs:
                # Use one Newton-like step from local residual derivative.
                dres = self._residual_derivative(rel, name, values, current=current)
                if dres is None:
                    continue
                target_scalar = current - residual / dres

            if target_scalar is None:
                continue

            merged = {**values, name: target_scalar}
            if self._constraints_violated(merged, rel=rel, names=[name]):
                continue

            var_tol_inner = self.graph.variable(name)
            rel_tol_inner = var_tol_inner.rel_tol if var_tol_inner is not None and var_tol_inner.rel_tol is not None else self.default_rel_tol
            if within_tolerance(current, target_scalar, rel_tol=float(rel_tol_inner)):
                continue

            change = relative_change(current, target_scalar)
            key = (change, name)
            if best is None or best_key is None or key < best_key:
                best = (name, change, target_scalar)
                best_key = key
                self._log.debug(
                    "  New best culprit: %s, change=%.6g, target=%.6g",
                    name,
                    change,
                    target_scalar,
                )

        if best:
            self._log.debug(
                "culprit_for_relation(%s): best=%s, change=%.6g",
                rel.name,
                best[0],
                best[1],
            )
        else:
            self._log.debug("culprit_for_relation(%s): no culprit found", rel.name)
        return best

    def diagnose(self, values_override: dict[str, object] | None = None) -> dict[str, object]:
        """Return consolidated diagnostics for relations, variables, and culprits.

        Args:
            values_override: Optional values mapping overriding current runtime values.

        Returns:
            Diagnostics dictionary.
        """
        # Step 1: use provided values override or current effective values.
        values = values_override or self._values_dict()
        self._relation_status_cache = {}

        # Step 2: evaluate relation status and collect likely culprit suggestions.
        relation_results: list[tuple[str, str, float | None]] = []
        culprits: dict[str, tuple[str, float, float]] = {}
        violated_relations: list[str] = []
        for rel, rel_name, status, residual in self._relation_status_rows(values):
            relation_results.append((rel_name, status, residual))
            if status == "VIOLATED":
                violated_relations.append(rel_name)
                culprit = self._culprit_for_relation(rel, values)
                if culprit is not None:
                    culprits[rel_name] = culprit

        # Step 2.1: aggregate culprit votes per variable for variable-level diagnostics.
        culprit_votes: dict[str, int] = {}
        for culprit_name, _change, _target in culprits.values():
            culprit_votes[culprit_name] = culprit_votes.get(culprit_name, 0) + 1

        # Step 3: evaluate variable consistency against input/current values.
        variable_issues: list[tuple[str, str, int | None]] = []
        for name in self.graph.variable_names():
            var = self.graph.variable(name)
            if var is None:
                continue
            if var.ndim == 1:
                variable_issues.append((name, "UNDETERMINABLE", None))
                continue
            input_val = var.input_value
            current_val = var.current_value
            if input_val is None or current_val is None:
                variable_issues.append((name, "UNDETERMINABLE", None))
                continue
            base = safe_float(input_val)
            cur = safe_float(current_val)
            if base is None or cur is None:
                variable_issues.append((name, "UNDETERMINABLE", None))
                continue
            # Mark as inconsistent when runtime drift from input exceeds tolerance.
            var_tol = self.graph.variable(name)
            rel_tol = var_tol.rel_tol if var_tol is not None and var_tol.rel_tol is not None else self.default_rel_tol
            inconsistent_by_drift = not within_tolerance(base, cur, rel_tol=float(rel_tol))
            # Mark as inconsistent when this variable is a repeated culprit for violated relations.
            culprit_hits = culprit_votes.get(name, 0)
            if inconsistent_by_drift:
                variable_issues.append((name, "INCONSISTENT", max(culprit_hits, 1)))
                continue
            if culprit_hits > 0:
                variable_issues.append((name, "INCONSISTENT", culprit_hits))
                continue
            variable_issues.append((name, "CONSISTENT", None))

        # Step 4: include one compact structural decomposition summary.
        structural_summary = self._structural_summary(values)

        # Step 5: return one consolidated diagnostic payload.
        return {
            "relation_status": relation_results,
            "violated_relations": violated_relations,
            "likely_culprits": culprits,
            "variable_issues": variable_issues,
            "structural_summary": structural_summary,
        }
