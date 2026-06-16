"""RelationSystem container, graph compiler and mode dispatcher."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, NamedTuple

import networkx as nx
import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from .relation import Relation, constraint_from_expression
from .registry import VARIABLES, VariableRegistry
from .utils import coerce_to_shape, domain_bounds_for_solver, parse_constraint_specs, scipy_bounds, value_in_domain
from .variable import Variable

DEFAULT_PROFILE_SIZE = 46


class Span(NamedTuple):
    """Packing span for one free variable in the solver vector.

    Fields mirror the previous 5-tuple ``(name, start, stop, offsets, scales)``
    so existing positional unpacking continues to work, while named access
    (``span.offsets``) keeps the many span signatures readable.
    """

    name: str
    start: int
    stop: int
    offsets: np.ndarray
    scales: np.ndarray


class RelationSystem:
    """Variables and relations compiled into one numeric system.

    Args:
        variables: Scenario variables.
        relations: Post-filter relation candidates.
        constraints: System-level constraints.
        name: Diagnostic system name.
        verbose: Whether to include verbose diagnostics.
        variable_registry: Variable registry.
        targets: Requested variables that anchor graph components.
        solve_for: Variables explicitly requested as solver outputs.
    """

    def __init__(
        self,
        variables: Iterable[Variable],
        relations: Iterable[Relation],
        *,
        constraints: Any = None,
        name: str | None = None,
        verbose: bool = False,
        variable_registry: VariableRegistry = VARIABLES,
        targets: Iterable[str] | None = None,
        solve_for: Iterable[str] | None = None,
    ) -> None:
        self.name = str(name or "relation_system")
        self.verbose = bool(verbose)
        self.variable_registry = variable_registry
        self.zero_tol = 1e-12
        self.targets = self._resolve_names(targets or ())
        self.solve_for = self._resolve_names(solve_for or ())
        self.variables = list(variables)
        self.variables_by_name: dict[str, Variable] = {}
        for var in self.variables:
            if var.name in self.variables_by_name:
                raise ValueError(f"Duplicate variable {var.name!r}.")
            self.variables_by_name[var.name] = var

        # Compilation runs as an ordered sequence of phases; each reads the
        # state left by earlier phases and writes its own attributes.
        self._build_candidate_relations(relations, constraints)
        self._infer_profile_size()
        self._ensure_rho_grid()
        self._broadcast_profiles()
        self._build_profile_shape_controls()
        self._compile_active_relations()
        self._register_profile_generators()
        self._append_active_guards()

        # Do not write relation-derived outputs into Variable.value during compilation.
        # Missing outputs are completed in local value maps during residual evaluation
        # and are stored only after a solve/ordered run finishes.
        self._refresh_scales()

    def _build_candidate_relations(self, relations: Iterable[Relation], constraints: Any) -> None:
        """Resolve relation variable names and create system constraints.

        A relation whose declared output collapses onto one of its own inputs
        after alias resolution (for example ``n_e_avg = n_avg`` when
        ``n_e_avg`` is an alias of ``n_avg``) is a tautology for this
        registry: it determines nothing, and the acausal seeding would
        otherwise "solve" the identity to an arbitrary grid value.
        """
        self.alias_degenerate_reasons: dict[str, str] = {}
        self.candidate_primary_relations = []
        for rel in relations:
            resolved = self._resolve_relation_names(rel)
            if resolved.implicit and not rel.implicit:
                self.alias_degenerate_reasons[resolved.name] = (
                    "inactive_alias_degenerate: declared outputs ("
                    + ", ".join(sorted(rel.outputs))
                    + ") resolve to the same canonical variable as an input"
                )
                continue
            self.candidate_primary_relations.append(resolved)
        self.system_constraint_relations = [
            self._resolve_relation_names(
                constraint_from_expression(text, name=f"system_constraint_{index}", enforce=enforce, source_kind="system", source_name=self.name)
            )
            for index, (text, enforce) in enumerate(parse_constraint_specs(constraints))
        ]

    def _infer_profile_size(self) -> None:
        """Infer one shared profile grid size from the supplied profile metadata."""
        profile_sizes: set[int] = set()
        for var in self.variables:
            if var.shape != 1:
                continue
            if var.size is not None:
                profile_sizes.add(int(var.size))
            elif isinstance(var.input_value, np.ndarray) and var.input_value.ndim == 1:
                profile_sizes.add(int(var.input_value.shape[0]))
        if len(profile_sizes) > 1:
            raise ValueError(f"Profile sizes are incompatible: {sorted(profile_sizes)}.")
        self.profile_size = next(iter(profile_sizes), DEFAULT_PROFILE_SIZE)

    def _ensure_rho_grid(self) -> None:
        """Ensure a canonical fixed ``rho`` grid exists.

        Created whenever profiles or rho-dependent relations are present.
        """
        if "rho" in self.variable_registry:
            uses_rho = any("rho" in rel.variables for rel in self.candidate_primary_relations)
            has_profile = any(var.shape == 1 for var in self.variables_by_name.values())
            if uses_rho or has_profile:
                rho_value = np.linspace(0.0, 1.0, self.profile_size)
                if "rho" not in self.variables_by_name:
                    rho = Variable("rho", value=rho_value, size=self.profile_size, fixed=True)
                    self.variables.append(rho)
                    self.variables_by_name["rho"] = rho
                else:
                    rho = self.variables_by_name["rho"]
                    if rho.input_value is None:
                        rho.size = self.profile_size
                        rho.set_input(rho_value)
                    rho.fixed = True

    def _broadcast_profiles(self) -> None:
        """Broadcast scalar profile data onto the shared grid.

        Also validates explicitly supplied profile lengths.
        """
        for var in self.variables_by_name.values():
            if var.shape != 1:
                continue
            if var.size is None:
                var.size = self.profile_size
            if var.input_value is not None:
                arr = np.asarray(var.input_value, dtype=float)
                if arr.ndim == 0:
                    var.input_value = np.full(var.size, float(arr))
                elif arr.ndim == 1 and arr.shape[0] != var.size:
                    raise ValueError(f"Profile {var.name!r} has length {arr.shape[0]}, expected {var.size}.")
            if var.value is not None:
                arr = np.asarray(var.value, dtype=float)
                if arr.ndim == 0:
                    var.value = np.full(var.size, float(arr))
                elif arr.ndim == 1 and arr.shape[0] != var.size:
                    raise ValueError(f"Profile {var.name!r} current length {arr.shape[0]}, expected {var.size}.")

    def _build_profile_shape_controls(self) -> None:
        """Convert supplied fixed profiles into fixed shapes plus scalar averages.

        Builds the shape/average controls used when the registry exposes an
        average variable for a supplied profile.
        """
        self.profile_shape_by_name: dict[str, np.ndarray] = {}
        self.profile_average_by_name: dict[str, str] = {}
        self.profile_source_by_name: dict[str, str] = {}
        self.fixed_supplied_profile_names: set[str] = set()
        for name, var in list(self.variables_by_name.items()):
            if name == "rho" or var.shape != 1 or var.input_value is None:
                continue
            avg_name = self._profile_average_name(name)
            if avg_name is None:
                continue
            avg_var = self._ensure_variable_exists(avg_name)
            arr = np.asarray(var.input_value, dtype=float)
            if arr.ndim == 0:
                arr = np.full(self.profile_size, float(arr))
            if arr.ndim != 1:
                raise ValueError(f"Profile {name!r} must be one-dimensional.")
            avg = self._profile_average(arr)
            if not np.isfinite(avg) or abs(avg) <= 1.0e-300:
                shape = np.ones_like(arr, dtype=float)
                avg = 0.0
            else:
                shape = arr / avg
                shape_avg = self._profile_average(shape)
                if np.isfinite(shape_avg) and abs(shape_avg) > 1.0e-300:
                    shape = shape / shape_avg
            self.profile_average_by_name[name] = avg_name
            self.profile_source_by_name[name] = "fixed_supplied_profile" if var.fixed else "supplied_profile"
            created_average_input = avg_var.input_value is None
            if created_average_input:
                avg_var.set_input(self._public_value(avg_name, avg))
            if var.fixed:
                self.profile_shape_by_name[name] = shape.astype(float, copy=False)
                self.fixed_supplied_profile_names.add(name)
                if created_average_input:
                    avg_var.fixed = True

    def _compile_active_relations(self) -> None:
        """Classify usable relations and compute the block/derived-variable plan.

        Activates the required defaults, partitions the system into decidable
        and undecidable variables, and writes the active relation set, block
        cores, derived providers and compiler report consumed by modes.
        """
        for name in sorted(self.targets | self.solve_for):
            self._ensure_variable_exists(name)
        requested = set(self.targets) | set(self.solve_for)
        supplied = {name for name, var in self.variables_by_name.items() if var.input_value is not None}
        inactive: dict[str, str] = dict(self.alias_degenerate_reasons)
        usable: list[Relation] = []
        for rel in self.candidate_primary_relations:
            if rel.dependency == "generated_profile" and rel.output_names and all(
                (var := self.variables_by_name.get(out)) is not None and var.fixed and var.input_value is not None
                for out in rel.output_names
            ):
                inactive[rel.name] = "inactive_profile_supplied_fixed"
                continue
            usable.append(rel)
            for rel_name in rel.variables:
                self._ensure_variable_exists(rel_name)
        non_default = [rel for rel in usable if not self._is_default_relation(rel)]
        defaults = [rel for rel in usable if self._is_default_relation(rel)]
        self.default_provider_by_output = {}
        for rel in sorted(defaults, key=lambda item: item.name):
            for out in rel.output_names:
                self.default_provider_by_output.setdefault(out, rel)
        non_default_profile_outputs = {
            out for rel in non_default for out in rel.output_names
            if out in self.variable_registry and self.variable_registry.get(out).shape == 1
        }
        forward = set(self._forward_decision_rounds(non_default)[0])
        active_defaults: list[Relation] = [
            rel for rel in sorted(defaults, key=lambda item: (len(item.variables), item.name))
            if any(
                out in self.variable_registry and self.variable_registry.get(out).shape == 1
                and out not in non_default_profile_outputs
                for out in rel.output_names
            )
        ]
        if active_defaults:
            forward = set(self._forward_decision_rounds(non_default + active_defaults)[0])
        changed = True
        while changed:
            changed = False
            for rel in sorted(defaults, key=lambda item: (len(item.variables), item.name)):
                if rel in active_defaults:
                    continue
                if any(out not in forward for out in rel.output_names) and all(inp in forward for inp in rel.input_names):
                    active_defaults.append(rel)
                    forward = set(self._forward_decision_rounds(non_default + active_defaults)[0])
                    changed = True
        pool = non_default + active_defaults
        partition = self._structural_partition(pool, forward)
        block_decidable = set(partition["determined_variables"])
        decidable = supplied | forward | block_decidable
        undecidable = set(partition["underdetermined_variables"]) - decidable
        self.underdetermined_requests = sorted(undecidable & requested)
        self.structural_blocks = list(partition["blocks"])
        active: list[Relation] = []
        for rel in pool:
            undec = sorted(set(rel.variables) & undecidable)
            if undec:
                inactive[rel.name] = "inactive_undecidable: cannot determine " + ", ".join(undec)
            else:
                active.append(rel)
        for rel in defaults:
            if rel not in active_defaults and rel.name not in inactive:
                inactive.setdefault(rel.name, "inactive_default_not_needed")
        self.primary_relations = active
        self.relations = list(active)
        self.active_variable_names = {name for rel in active for name in rel.variables} | set(self.targets) | set(self.solve_for)
        for name in sorted(self.active_variable_names):
            self._ensure_variable_exists(name)
        produced = {out for rel in active for out in rel.output_names if not rel.implicit}
        self.block_core_names = {
            name for name in self.active_variable_names
            if name in decidable
            and name not in supplied
            and name not in produced
            and not self.variables_by_name[name].fixed
        }
        _, forward_decider = self._forward_decision_rounds(active, extra_known=self.block_core_names)
        self.derived_provider_by_output = {}
        self.derived_variable_names = set()
        for name in sorted(self.active_variable_names):
            var = self.variables_by_name[name]
            if var.fixed or var.input_value is not None or name in self.block_core_names:
                continue
            selected = forward_decider.get(name)
            if selected is None or selected not in active:
                continue
            self.derived_provider_by_output[name] = selected
            self.derived_variable_names.add(name)
        self.blocked_relation_reasons = inactive
        self.compiler_report = {
            "activation_semantics": "decidability_closure",
            "supplied_variables": tuple(sorted(supplied)),
            "active_variables": tuple(sorted(self.active_variable_names)),
            "derived_variables": tuple(sorted(self.derived_variable_names)),
            "default_provider_outputs": {name: rel.name for name, rel in sorted(self.default_provider_by_output.items())},
            "active_relations": tuple(rel.name for rel in active),
            "inactive_relations": dict(sorted(inactive.items())),
            "relation_to_vars": self._incidence_views()[0],
            "structural_determinacy": {
                "determined_missing_variables": tuple(sorted(partition["determined_variables"])),
                "undecidable_variables": tuple(sorted(undecidable)),
                "deficiencies": partition["deficiencies"],
                "undecidable_requests": tuple(self.underdetermined_requests),
                "blocks": tuple(self.structural_blocks),
            },
        }

    def _register_profile_generators(self) -> None:
        """Register explicit lower-dimensional profile generators as providers.

        Activates any required scalar-average controls and refreshes the
        profile-related compiler report views.
        """
        for rel in list(getattr(self, "relations", ())):
            profile_outputs = [
                out for out in rel.output_names
                if out in self.variable_registry and self.variable_registry.get(out).shape == 1 and out != "rho"
            ]
            if not profile_outputs or rel.implicit:
                continue
            lower_dimensional = True
            for inp in rel.input_names:
                if inp == "rho":
                    continue
                if inp not in self.variable_registry or self.variable_registry.get(inp).shape == 1:
                    lower_dimensional = False
                    break
            if not lower_dimensional:
                continue
            for out in rel.output_names:
                if out not in self.variable_registry or self.variable_registry.get(out).shape != 1 or out == "rho":
                    continue
                var = self.variables_by_name.get(out)
                if var is not None and var.fixed:
                    continue
                avg_name = self._profile_average_name(out)
                if avg_name is not None:
                    self._ensure_variable_exists(avg_name)
                    self.profile_average_by_name.setdefault(out, avg_name)
                    self.active_variable_names.add(avg_name)
                self.derived_provider_by_output[out] = rel
                self.derived_variable_names.add(out)
        self.compiler_report["profile_average_by_name"] = dict(sorted(self.profile_average_by_name.items()))
        self.compiler_report["profile_source_by_name"] = dict(sorted(self.profile_source_by_name.items()))
        self.compiler_report["active_variables"] = tuple(sorted(self.active_variable_names))
        self.compiler_report["derived_variables"] = tuple(sorted(self.derived_variable_names))

    def _append_active_guards(self) -> None:
        """Append active relation guards and build the relation name index.

        Guards come from relation-local, variable-local, and system-level
        constraints whose variables are all active.
        """
        active_names = {rel.name for rel in self.relations}
        active_vars = set(self.active_variable_names)
        for rel in list(self.primary_relations):
            for guard in rel.constraint_relations:
                guard = self._resolve_relation_names(guard)
                if guard.name not in active_names and set(guard.variables) <= active_vars:
                    self.relations.append(guard)
                    active_names.add(guard.name)
        for name in sorted(active_vars):
            for guard in self.variables_by_name[name].relations:
                guard = self._resolve_relation_names(guard)
                if guard.name not in active_names and set(guard.variables) <= active_vars:
                    self.relations.append(guard)
                    active_names.add(guard.name)
        for guard in self.system_constraint_relations:
            if guard.name not in active_names and set(guard.variables) <= active_vars:
                self.relations.append(guard)
                active_names.add(guard.name)
        self.relations_by_name = {
            rel.name: rel for rel in [*self.candidate_primary_relations, *self.system_constraint_relations, *self.relations]
        }

    def _incidence_views(self) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
        """Return ``(relation_to_vars, var_to_relations)`` over candidate relations.

        Derived on demand: the relations themselves are the single source of the
        variable<->relation incidence, so these are kept only as report/graph
        views rather than stored parallel dicts.
        """
        relation_to_vars = {rel.name: rel.variables for rel in self.candidate_primary_relations}
        var_to_relations: dict[str, list[str]] = {}
        for rel in self.candidate_primary_relations:
            for var in rel.variables:
                var_to_relations.setdefault(var, []).append(rel.name)
        return relation_to_vars, {name: tuple(rels) for name, rels in var_to_relations.items()}

    @property
    def graph(self) -> dict[str, Any]:
        """Compiled variable-relation graph used by execution modes.

        The graph is structural only. Mode implementations may use it for
        propagation, block solving, residual compilation and diagnostics, but
        RelationSystem remains only the owner of variables, relations and the
        registry.
        """
        relation_to_vars, var_to_relations = self._incidence_views()
        return {
            "variables": tuple(sorted(self.variables_by_name)),
            "relations": tuple(rel.name for rel in self.relations),
            "active_relations": tuple(rel.name for rel in self.relations),
            "enforced_relations": tuple(rel.name for rel in self.relations if rel.enforce),
            "relation_to_vars": relation_to_vars,
            "var_to_relations": var_to_relations,
            "derived_provider_by_output": {name: rel.name for name, rel in self.derived_provider_by_output.items()},
            "derived_variables": tuple(sorted(self.derived_variable_names)),
            "active_variables": tuple(sorted(self.active_variable_names)),
            "blocked_relations": dict(getattr(self, "blocked_relation_reasons", {})),
        }

    def verify(self, **options: Any) -> dict[str, Any]:
        return self.run("verify", **options)

    def reconcile(self, **options: Any) -> dict[str, Any]:
        return self.run("reconcile", **options)

    def optimize(self, **options: Any) -> dict[str, Any]:
        return self.run("optimize", **options)

    def ordered(self, **options: Any) -> dict[str, Any]:
        return self.run("ordered", **options)

    def run(self, mode: str = "verify", **options: Any) -> dict[str, Any]:
        """Dispatch to an isolated execution mode."""
        from .modes import get_mode

        return get_mode(mode)(self, **options)

    def _profile_average_name(self, name: str) -> str | None:
        """Return the scalar-average variable controlling a profile, or None.

        Resolves the registry ``average_variable`` metadata, falling back to the
        ``<name>_avg`` alias convention.
        """
        if name not in self.variable_registry:
            return None
        spec = self.variable_registry.get(name)
        if getattr(spec, "average_variable", None):
            candidate = str(spec.average_variable)
            return self.variable_registry.resolve(candidate) if candidate in self.variable_registry else candidate
        alias = f"{name}_avg"
        if alias in self.variable_registry:
            return self.variable_registry.resolve(alias)
        return None

    def _profile_average(self, value: Any) -> float:
        """Return the rho-weighted grid average of a profile-like value.

        Uses the trapezoidal average over the canonical ``rho`` grid when it is
        available, otherwise the arithmetic mean.  Scalars return themselves and
        empty profiles return zero.
        """
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return float(arr)
        if arr.size == 0:
            return 0.0
        rho_var = self.variables_by_name.get("rho")
        if rho_var is not None and rho_var.input_value is not None and rho_var.value is not None:
            rho = np.asarray(rho_var.value, dtype=float)
            if rho.ndim == 1 and rho.size == arr.size and rho.size > 1:
                width = float(rho[-1] - rho[0])
                if width > 0.0:
                    return float(trapezoid(arr, x=rho) / width)
        return float(np.mean(arr))

    def _with_profile_values(self, values: dict[str, Any], *, strict: bool) -> dict[str, Any]:
        """Rebuild profile arrays from scalar averages and stored shapes.

        Args:
            values: Current scalar/profile namespace.
            strict: Whether missing average controls should raise.

        Returns:
            Namespace with shape-controlled profiles updated.
        """
        if not getattr(self, "profile_shape_by_name", None):
            return values
        out = dict(values)
        for name, shape in self.profile_shape_by_name.items():
            if name in getattr(self, "fixed_supplied_profile_names", set()):
                fixed_var = self.variables_by_name.get(name)
                if fixed_var is not None and fixed_var.input_value is not None:
                    out[name] = self._solver_value(name, fixed_var.input_value)
                continue
            avg_name = self.profile_average_by_name.get(name)
            if avg_name is None:
                continue
            if avg_name not in out or out[avg_name] is None:
                if strict:
                    raise ValueError(f"Could not rebuild profile {name!r}; missing average {avg_name!r}.")
                continue
            avg = float(np.asarray(out[avg_name], dtype=float).reshape(-1)[0])
            out[name] = self._solver_value(name, avg * shape)
        return out

    def _relation_is_residual_relation(self, rel: Relation) -> bool:
        """Return whether a relation belongs in the nonlinear residual core.

        Explicit-output relations whose outputs are uniquely owned non-fixed
        derived variables are evaluated structurally by ``_with_derived_values``
        on every candidate value map.  They are still verified during final
        certification, but they are not soft least-squares residuals.

        Relations remain residual relations when they are outputless, implicit,
        have fixed/non-derived outputs, have ambiguous producers, or are guard /
        constraint relations.
        """
        if rel.implicit or not rel.output_names:
            return True
        providers = getattr(self, "derived_provider_by_output", {})
        if not providers:
            return True
        # A relation may have multiple outputs. It is structural only if every
        # declared output is owned by this same provider.  Partial ownership must
        # stay in the residual core so final values cannot silently ignore one
        # side of the equation.
        return not all(providers.get(out) is rel for out in rel.output_names)

    def _forward_decision_rounds(self, relations: list[Relation], extra_known: Iterable[str] = ()) -> tuple[dict[str, int], dict[str, "Relation"]]:
        """Return forward-closure rounds and per-variable forward providers.

        ``extra_known`` seeds additional variables as decided at round 0 -- used
        to treat block cores as available so the block-downstream variables get
        forward providers.

        Supplied variables are decided at round 0.  Each round, real relations
        are exhausted before defaults, so a variable is owned by a default only
        when no real relation can decide it.  A relation decides a variable
        either forward (all inputs known, every output decided) or acausally
        (a single remaining variable from the rest).  The returned ``decider``
        maps a variable to the relation that first decided it as one of its
        declared outputs -- the forward provider for completion.  Variables
        decided only acausally (as an input, e.g. ``a`` from ``A = R/a``) or
        never reached have no ``decider`` entry and are packed as free
        variables for the global solve.
        """
        rounds: dict[str, int] = {
            name: 0 for name, var in self.variables_by_name.items() if var.input_value is not None
        }
        for name in extra_known:
            rounds.setdefault(name, 0)
        decider: dict[str, Relation] = {}
        non_default = [rel for rel in relations if not self._is_default_relation(rel)]
        defaults = [rel for rel in relations if self._is_default_relation(rel)]
        round_no = 0
        changed = True
        while changed:
            changed = False
            round_no += 1

            # Forward decisions take priority and run to a fixed point first, so
            # a variable that some relation can compute as an output (n_D from
            # n_i and f_D) is owned by that producer rather than being decided
            # acausally as an input of another relation (Integrated D fraction).
            # Real relations are exhausted before defaults within each sweep.
            forward_changed = True
            while forward_changed:
                forward_changed = False
                for group in (non_default, defaults):
                    for rel in group:
                        if rel.output_names and all(inp in rounds for inp in rel.input_names):
                            for out in rel.output_names:
                                if out not in rounds:
                                    rounds[out] = round_no
                                    decider.setdefault(out, rel)
                                    forward_changed = changed = True

            # Acausal fallback: a single remaining variable is decided from the
            # rest.  Applied only after forward decisions stall; control returns
            # to the forward sweep so any newly available output is owned by its
            # producer.
            for group in (non_default, defaults):
                for rel in group:
                    undecided = [v for v in rel.variables if v not in rounds]
                    if len(undecided) == 1 and undecided[0] not in rounds:
                        v = undecided[0]
                        rounds[v] = round_no
                        if v in rel.output_names:
                            decider.setdefault(v, rel)
                        changed = True
        return rounds, decider

    def _structural_partition(self, relations: list[Relation], known: set[str]) -> dict[str, Any]:
        """Split missing variables by structural determinacy (Dulmage-Mendelsohn).

        Equality relations and missing variables form a bipartite graph in
        scalar dimensions.  A missing variable is underdetermined when it is
        reachable by an alternating path from an unmatched variable of a
        maximum matching; such variables keep a leftover degree of freedom in
        every maximum matching and can never be computed from the supplied
        values.  All other missing variables are structurally determined.

        This is structural information only: a structurally determined block
        may still fail to solve numerically, and independent verification
        remains the final arbiter.

        Args:
            relations: Active relations, including activated default providers.
            known: Variable names that already have a value.

        Returns:
            Mapping with ``determined_variables``, ``underdetermined_variables``
            and per-group ``deficiencies`` diagnostics.
        """
        unknowns = sorted({name for rel in relations for name in rel.variables if name not in known})
        for name in unknowns:
            self._ensure_variable_exists(name)
        result: dict[str, Any] = {
            "determined_variables": set(unknowns),
            "underdetermined_variables": set(),
            "deficiencies": [],
            "blocks": [],
        }
        if not unknowns:
            return result

        col_span: dict[str, tuple[int, int]] = {}
        n_cols = 0
        for name in unknowns:
            dim = self._variable_dim(name)
            col_span[name] = (n_cols, n_cols + dim)
            n_cols += dim

        # One scalar row per residual dimension; inequalities determine nothing.
        row_adj: list[list[int]] = []
        row_relation: list[str] = []
        for rel in relations:
            if not rel.outputs and rel.op != "==":
                continue
            cols = [c for name in rel.variables if name in col_span for c in range(*col_span[name])]
            if not cols:
                continue
            for _ in range(max(1, self._relation_row_dim(rel))):
                row_adj.append(cols)
                row_relation.append(rel.name)

        match_col = np.full(max(len(row_adj), 1), -1, dtype=int)
        if row_adj:
            matrix = lil_matrix((len(row_adj), n_cols), dtype=bool)
            for r, cols in enumerate(row_adj):
                matrix[r, cols] = True
            match_col = np.asarray(maximum_bipartite_matching(matrix.tocsr(), perm_type="column"), dtype=int)
        match_row = np.full(n_cols, -1, dtype=int)
        for r in range(len(row_adj)):
            if match_col[r] >= 0:
                match_row[int(match_col[r])] = r

        # Alternating reachability from unmatched columns over a directed graph
        # (column -> incident row, row -> its matched column).  Columns it reaches
        # are exactly the underdetermined ones; rows it reaches are the relations
        # participating in the deficiency.
        reach = nx.DiGraph()
        for r, cols in enumerate(row_adj):
            for c in cols:
                reach.add_edge(("c", c), ("r", r))
            mc = int(match_col[r])
            if mc >= 0:
                reach.add_edge(("r", r), ("c", mc))
        reached: set[tuple[str, int]] = set()
        for c in range(n_cols):
            if match_row[c] < 0:
                reached.add(("c", c))
                if ("c", c) in reach:
                    reached |= nx.descendants(reach, ("c", c))
        under_cols = {c for kind, c in reached if kind == "c"}
        under_rows = {r for kind, r in reached if kind == "r"}

        name_of_col = {c: name for name, (start, stop) in col_span.items() for c in range(start, stop)}
        under_names = {name for name, (start, stop) in col_span.items() if any(c in under_cols for c in range(start, stop))}
        result["determined_variables"] = set(unknowns) - under_names
        result["underdetermined_variables"] = under_names
        result["blocks"] = self._structural_block_plan(row_adj, match_row, under_cols, name_of_col)

        # Group the underdetermined part into connected deficiencies on the
        # bipartite (column, row) incidence; each group needs (cols - rows) more
        # supplied values among its variables.
        deficiency_graph = nx.Graph()
        deficiency_graph.add_nodes_from(("c", c) for c in under_cols)
        for r in under_rows:
            for c in row_adj[r]:
                if c in under_cols:
                    deficiency_graph.add_edge(("c", c), ("r", r))
        deficiencies: list[dict[str, Any]] = []
        for comp in nx.connected_components(deficiency_graph):
            comp_cols = [c for kind, c in comp if kind == "c"]
            comp_rows = [r for kind, r in comp if kind == "r"]
            deficiencies.append(
                {
                    "variables": sorted({name_of_col[c] for c in comp_cols}),
                    "relations": sorted({row_relation[r] for r in comp_rows}),
                    "missing_values": int(len(comp_cols) - len(comp_rows)),
                }
            )
        result["deficiencies"] = sorted(deficiencies, key=lambda item: tuple(item["variables"]))
        return result

    def _structural_block_plan(
        self,
        row_adj: list[list[int]],
        match_row: np.ndarray,
        under_cols: set[int],
        name_of_col: Mapping[int, str],
    ) -> list[tuple[str, ...]]:
        """Return determined-variable blocks in dependency order.

        The strongly connected components of the matched dependency digraph
        are the simultaneous solve blocks (1x1 direct steps, 2x2, 3x3, ...),
        and their topological order is the solve order.  Components that share
        a variable (profile columns can split across components) are merged
        and keep the earliest position.
        """
        determined = [c for c in range(len(match_row)) if c not in under_cols and match_row[c] >= 0]
        if not determined:
            return []
        determined_set = set(determined)
        # Dependency digraph over determined columns: c2 -> c when c2 appears in
        # the relation matched to c (so c depends on c2).  Strongly connected
        # components are the simultaneous solve blocks; the condensation gives
        # their dependency order.
        digraph = nx.DiGraph()
        digraph.add_nodes_from(determined)
        for c in determined:
            for c2 in row_adj[int(match_row[c])]:
                if c2 != c and c2 in determined_set:
                    digraph.add_edge(c2, c)
        condensation = nx.condensation(digraph)

        # Components -> variable-name groups, merging groups that share a name
        # (profile columns can split across components) and ordering by first
        # topological appearance.
        parent: dict[str, str] = {}

        def find(name: str) -> str:
            while parent[name] != name:
                parent[name] = parent[parent[name]]
                name = parent[name]
            return name

        def component_names(comp: int) -> list[str]:
            return sorted({name_of_col[col] for col in condensation.nodes[comp]["members"]})

        first_rank: dict[str, int] = {}
        for rank, comp in enumerate(nx.lexicographical_topological_sort(condensation, key=component_names)):
            names = component_names(comp)
            for name in names:
                parent.setdefault(name, name)
                first_rank.setdefault(name, rank)
            for left, right in zip(names, names[1:]):
                parent[find(right)] = find(left)
        groups: dict[str, set[str]] = {}
        for name in parent:
            groups.setdefault(find(name), set()).add(name)
        ordered = sorted(groups.values(), key=lambda group: min(first_rank[name] for name in group))
        return [tuple(sorted(group)) for group in ordered]

    def _initial_values_from_graph(
        self,
        *,
        residual_tol: float = 1.0,
        max_passes: int = 50,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Fill solver start values by direct propagation from supplied values.

        Iteratively solves every relation that has exactly one missing variable
        (the 1x1 / acausal step), to a fixed point.  These are exact values, not
        movement references.  Variables that remain missing are the free unknowns
        of larger coupled blocks (block cores); they are packed directly and
        determined by the simultaneous reconcile against their block's supplied
        anchor, so no separate block solver is needed here.
        """
        values = self._values_from_variables(for_solver=True, skip_missing=True, complete=False, use_input_values=True)
        values = self._with_profile_values(dict(values), strict=False)
        original = set(values)
        info: dict[str, Any] = {}
        residual_tol = float(residual_tol)
        # Stage 1: direct 1x1/acausal propagation to a fixed point.
        for _direct_pass in range(max_passes):
            if not self._compute_direct_outputs(values, info, original):
                break
        # Stage 2: solve the determined blocks (2x2 ... N x N) for their cores.
        progress = True
        while progress:
            progress = False
            for block in getattr(self, "structural_blocks", ()):
                if self._compute_planned_block(block, values, info, original, residual_tol=residual_tol):
                    progress = True
                    for _direct_pass in range(max_passes):
                        if not self._compute_direct_outputs(values, info, original):
                            break
        merged = tuple(
            name
            for block in getattr(self, "structural_blocks", ())
            for name in block
            if (name not in values or values[name] is None) and name not in original
        )
        if merged and self._compute_planned_block(merged, values, info, original, residual_tol=residual_tol):
            for _direct_pass in range(max_passes):
                if not self._compute_direct_outputs(values, info, original):
                    break
        initial_values = {name: values[name] for name in values if name in info}
        return initial_values, info


    def _initial_direct_relation_pool(self) -> list[Relation]:
        """Relations allowed for direct output initial computation.

        The global reconcile still uses ``self.relations``.  For initial guesses
        only, inactive weak/default providers may fill missing values when their
        inputs are already known.  This makes defaults useful as x0 generators
        without adding them as extra enforced residuals or movement references.
        """
        by_name = {rel.name: rel for rel in self.relations}
        for rel in self.candidate_primary_relations:
            if rel.name in by_name:
                continue
            if self._is_default_relation(rel):
                by_name[rel.name] = rel
        return list(by_name.values())

    def _compute_direct_outputs(self, values: dict[str, Any], info: dict[str, Any], original: set[str]) -> bool:
        """Initial computation declared outputs whose inputs are already known.

        This initial computation stage is only allowed to fill missing values.  It must not
        override user-supplied values, even for x0.  Supplied non-fixed outputs
        are handled by runtime structural closure and movement penalties, not by
        the initial guess map.  This keeps ``initial_values`` a pure provenance record of
        direct/block/default computations for variables that were actually
        missing.
        """
        progress = False
        for rel in self._initial_direct_relation_pool():
            if rel.output_names and all(inp in values and values[inp] is not None for inp in rel.input_names):
                try:
                    eval_values = self._relation_evaluation_values(rel, values)
                    mapped = rel.output_map(rel.evaluate(eval_values))
                except Exception:
                    mapped = {}
                for name in rel.output_names:
                    if name not in mapped or name not in self.variable_registry:
                        continue
                    var = self.variables_by_name.get(name)
                    if var is not None and var.fixed:
                        continue
                    if name in values and values[name] is not None:
                        continue
                    try:
                        value = self._solver_value(name, mapped[name])
                        if not self._candidate_value_is_valid(name, value):
                            continue
                    except Exception:
                        continue
                    values[name] = value
                    info[name] = {
                        "source": "direct_output",
                        "relation": rel.name,
                        "block_size": 0,
                        "enforced": bool(rel.enforce),
                    }
                    progress = True

            # Acausal one-missing-variable initial computation.  This is still
            # a direct computation step, not a persistent stored state: it fills
            # the initial guess namespace only when exactly one variable in
            # an active relation is missing and the relation object can solve it
            # while verifying the canonical relation.  It is generic and uses no
            # variable-name assumptions.  Implicit relations are excluded: a
            # variable appearing on both sides cannot be trusted to a blind
            # inverse scan (flat residuals would accept arbitrary values).
            if rel.implicit:
                continue
            missing = [name for name in rel.variables if name not in values or values[name] is None]
            if len(missing) != 1:
                continue
            name = missing[0]
            if name not in self.variable_registry:
                continue
            var = self.variables_by_name.get(name)
            if var is not None and var.fixed:
                continue
            if name in original:
                continue
            known = {vname: values[vname] for vname in rel.variables if vname != name and vname in values and values[vname] is not None}
            if len(known) != len(rel.variables) - 1:
                continue
            try:
                value = self._solver_value(name, rel(**known))
                if not self._candidate_value_is_valid(name, value):
                    continue
            except Exception:
                continue
            values[name] = value
            info[name] = {
                "source": "direct_inverse",
                "relation": rel.name,
                "block_size": 1,
                "enforced": bool(rel.enforce),
            }
            progress = True
        return progress

    def _compute_planned_block(
        self,
        block: tuple[str, ...],
        values: dict[str, Any],
        info: dict[str, Any],
        original: set[str],
        *,
        residual_tol: float,
    ) -> bool:
        """Solve one structurally planned block as initial guesses.

        The block is first extended with every variable producible from it
        (generated profiles, reactivities, rates), so relations that pin the
        block only through those intermediates participate in the solve.
        """
        unknowns = tuple(name for name in block if name not in values or values[name] is None)
        if not unknowns or any(name in original for name in unknowns):
            return False
        for name in unknowns:
            var = self.variables_by_name.get(name)
            if var is None or var.fixed:
                return False
        extended, rels = self._block_closure(unknowns, values)
        if not rels:
            return False
        solved = self._solve_initial_block(extended, rels, values, residual_tol=residual_tol)
        if solved is None:
            return False
        for name, value in solved["values"].items():
            values[name] = value
            info[name] = {
                "source": "scalar_block",
                "relations": [rel.name for rel in rels if rel.enforce],
                "block_size": len(unknowns),
                "max_abs_residual": solved["max_abs_residual"],
                "nfev": solved["nfev"],
            }
        return True

    def _block_closure(self, unknowns: tuple[str, ...], values: Mapping[str, Any]) -> tuple[tuple[str, ...], list[Relation]]:
        """Extend a planned block with variables producible from it.

        Returns the extended unknown set and the participating relations:
        every active relation whose missing variables are covered by the
        extended set.  Non-enforced relations participate as value providers
        only; enforced relations supply the residual rows.
        """
        available = {name for name, value in values.items() if value is not None}
        extended = set(unknowns)
        changed = True
        while changed:
            changed = False
            for rel in self.relations:
                if rel.implicit or not rel.output_names:
                    continue
                if not all(inp in available or inp in extended for inp in rel.input_names):
                    continue
                for out in rel.output_names:
                    if out in available or out in extended or out not in self.variable_registry:
                        continue
                    var = self.variables_by_name.get(out)
                    if var is not None and var.fixed:
                        continue
                    extended.add(out)
                    changed = True
        rels: list[Relation] = []
        for rel in self.relations:
            missing = {name for name in rel.variables if name not in available}
            if not missing or not missing <= extended:
                continue
            rels.append(rel)
        return tuple(sorted(extended)), rels

    def _solve_initial_block(
        self,
        unknowns: tuple[str, ...],
        rels: list[Relation],
        values: Mapping[str, Any],
        *,
        residual_tol: float,
    ) -> dict[str, Any] | None:
        """Solve one small scalar initial-computation block.

        Unknowns that are declared outputs of a block relation are recomputed
        from that relation inside the residual, so the numerical search runs
        only over the remaining core unknowns.  Core starts come from
        supplied/current values when present, otherwise from the magnitudes
        of the known values around the block, refined by a coarse log-grid
        scan.  Solver domains constrain the search; they never provide a
        start on their own.
        """
        for name in unknowns:
            if name not in self.variables_by_name:
                return None
            if self.variables_by_name[name].fixed:
                return None

        producers = self._block_producers(unknowns, rels, values)
        core = [name for name in unknowns if name not in producers]
        if not core:
            # A fully produced cycle still needs one numerical degree of freedom.
            core = [unknowns[0]]
            producers.pop(unknowns[0], None)
        # Profiles are recomputed from their producers; the numerical core must
        # stay scalar so the search dimension never explodes pointwise.
        for name in core:
            if self.variable_registry.get(name).shape == 1:
                return None
        enforced_rows = sum(max(1, self._relation_row_dim(rel)) for rel in rels if rel.enforce)
        if enforced_rows < len(core):
            return None

        def namespace_from(core_values: Mapping[str, float]) -> dict[str, Any]:
            ns = dict(values)
            ns.update(core_values)
            for produced, rel in producers.items():
                try:
                    mapped = rel.output_map(rel.evaluate(self._relation_evaluation_values(rel, ns)))
                    if mapped.get(produced) is not None:
                        ns[produced] = self._solver_value(produced, mapped[produced])
                except Exception:
                    continue
            return self._complete_values(ns, strict=False)

        def residual_from(core_values: Mapping[str, float]) -> np.ndarray:
            ns = namespace_from(core_values)
            blocks: list[np.ndarray] = []
            for rel in rels:
                if not rel.enforce:
                    continue
                if any(name not in ns or ns[name] is None for name in rel.variables):
                    blocks.append(np.asarray([1.0e6], dtype=float))
                    continue
                try:
                    eval_values = self._relation_evaluation_values(rel, ns)
                    blocks.append(self._residual_vector(rel, eval_values, safe=True))
                except Exception:
                    blocks.append(np.asarray([1.0e6], dtype=float))
            out = np.concatenate([block.reshape(-1) for block in blocks if block.size]) if blocks else np.empty(0, dtype=float)
            return np.nan_to_num(out, nan=1.0e6, posinf=1.0e6, neginf=-1.0e6)

        def score(core_values: Mapping[str, float]) -> float:
            residual = residual_from(core_values)
            return float(np.max(np.abs(residual))) if residual.size else np.inf

        bounds_by_name: dict[str, tuple[float, float]] = {}
        starts: dict[str, float] = {}
        for name in core:
            var = self.variables_by_name[name]
            lb, ub = scipy_bounds(self.variable_registry.get(name).solver_domain, zero_tol=self.zero_tol)
            bounds_by_name[name] = (lb, ub)
            try:
                starts[name] = float(self._initial_value(var))
            except Exception:
                start = self._block_start_from_knowns(rels, values, lb, ub)
                if start is None:
                    return None
                starts[name] = start

        # Coordinate-wise log-grid refinement of the starts.  One sweep is
        # exact for a single core unknown; two sweeps untangle coupled cores.
        for _sweep in range(1 if len(core) == 1 else 2):
            for name in core:
                lb, ub = bounds_by_name[name]
                best, best_score = starts[name], score(starts)
                for point in self._scalar_start_grid(lb, ub):
                    point_score = score({**starts, name: point})
                    if point_score < best_score:
                        best, best_score = point, point_score
                starts[name] = best

        # An unconstrained core direction means the block residual does not
        # determine the value: accepting it would seed an arbitrary number.
        # The direction is flat when widely separated grid points give the
        # same in-tolerance score; weak but nonzero dependence is kept.
        for name in core:
            lb, ub = bounds_by_name[name]
            grid = self._scalar_start_grid(lb, ub)
            if len(grid) < 3:
                continue
            probes = [score({**starts, name: point}) for point in (grid[0], grid[len(grid) // 2], grid[-1])]
            if max(probes) - min(probes) <= 1e-9 and min(probes) <= residual_tol:
                return None

        offsets: list[float] = []
        scales: list[float] = []
        lower: list[float] = []
        upper: list[float] = []
        transforms: list[str] = []
        for name in core:
            var = self.variables_by_name[name]
            lb, ub = bounds_by_name[name]
            init = min(max(starts[name], lb), ub) if np.isfinite(lb) or np.isfinite(ub) else starts[name]
            scale, offset, lo, hi, transform = self._pack_scalar(name, var, init, lb, ub, scale_ref=init)
            offsets.append(offset)
            scales.append(scale)
            lower.append(lo)
            upper.append(hi)
            transforms.append(transform)

        def core_values_from(x: np.ndarray) -> dict[str, float]:
            arr = np.asarray(x, dtype=float)
            out: dict[str, float] = {}
            for idx, name in enumerate(core):
                if transforms[idx] == "log":
                    out[name] = float(offsets[idx] * np.exp(arr[idx]))
                else:
                    out[name] = float(offsets[idx] + scales[idx] * arr[idx])
            return out

        def residual(x: np.ndarray) -> np.ndarray:
            return residual_from(core_values_from(x))

        x0 = np.zeros(len(core), dtype=float)
        try:
            probe = residual(x0)
            if probe.size < len(core):
                return None
            sol = least_squares(
                residual,
                x0,
                bounds=(np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)),
                method="trf",
                x_scale=np.ones_like(x0),
                max_nfev=80,
                xtol=1e-10,
                ftol=1e-10,
                gtol=1e-10,
            )
        except Exception:
            return None
        final_residual = residual(sol.x)
        max_abs = float(np.max(np.abs(final_residual))) if final_residual.size else 0.0
        if not np.isfinite(max_abs) or max_abs > residual_tol:
            return None
        ns = namespace_from(core_values_from(sol.x))
        solved: dict[str, Any] = {}
        for name in unknowns:
            if name not in ns or ns[name] is None:
                return None
            value = ns[name]
            if not self._candidate_value_is_valid(name, value):
                return None
            solved[name] = self._solver_value(name, value)
        return {"values": solved, "max_abs_residual": max_abs, "nfev": int(getattr(sol, "nfev", -1))}

    def _block_producers(self, unknowns: tuple[str, ...], rels: list[Relation], values: Mapping[str, Any]) -> dict[str, Relation]:
        """Return produced-unknown -> relation, in evaluation order.

        A block unknown is produced when one block relation declares it as an
        output and every input of that relation is either known or another
        block unknown.  Producible unknowns that cannot be ordered, because
        they form a cyclic chain, stay in the numerical core.
        """
        unknown_set = set(unknowns)
        available = {name for name, value in values.items() if value is not None}
        producible: set[str] = set()
        for rel in rels:
            if rel.implicit or not rel.output_names:
                continue
            producible.update(out for out in rel.output_names if out in unknown_set)
        # Unknowns no relation can produce are the numerical core seeds.
        available.update(name for name in unknown_set if name not in producible)

        # Greedy topological selection: an unknown is assigned the first
        # relation whose inputs are already available, so purely cyclic
        # alternatives (for example quasineutrality pairs) never deadlock the
        # ordering.  Defaults come first: they carry the weak-assumption
        # value, while enforced alternatives stay in the residual rows.
        ordered_rels = sorted(rels, key=lambda rel: not self._is_default_relation(rel))
        ordered: dict[str, Relation] = {}
        changed = True
        while changed:
            changed = False
            for rel in ordered_rels:
                if rel.implicit or not rel.output_names:
                    continue
                if not all(inp in available for inp in rel.input_names):
                    continue
                for out in rel.output_names:
                    if out in unknown_set and out not in available:
                        ordered[out] = rel
                        available.add(out)
                        changed = True
        return ordered

    def _block_start_from_knowns(self, rels: list[Relation], values: Mapping[str, Any], lb: float, ub: float) -> float | None:
        """Return a start from the magnitudes of the block's known values.

        This mirrors the standalone relation inverse-solve heuristic: the
        geometric mean of the finite positive known values touching the block,
        clipped into the solver interval.  Bounds only clip; they never
        generate the start themselves.
        """
        magnitudes: list[float] = []
        for rel in rels:
            for name in rel.variables:
                value = values.get(name)
                if value is None:
                    continue
                arr = np.asarray(value, dtype=float).reshape(-1)
                magnitudes.extend(float(item) for item in arr if np.isfinite(item) and item > 0.0)
        if not magnitudes:
            return None
        start = float(np.exp(np.mean(np.log(np.asarray(magnitudes, dtype=float)))))
        if np.isfinite(lb) and start < lb:
            start = float(lb)
        if np.isfinite(ub) and start > ub:
            start = float(ub)
        return start

    def _scalar_start_grid(self, lb: float, ub: float) -> list[float]:
        """Return signed log-spaced start candidates inside the solver interval."""
        points: set[float] = set()
        for exponent in range(-30, 31, 2):
            for sign in (1.0, -1.0):
                value = sign * 10.0**exponent
                if (not np.isfinite(lb) or value >= lb) and (not np.isfinite(ub) or value <= ub):
                    points.add(value)
        return sorted(points)

    def _candidate_value_is_valid(self, name: str, value: Any) -> bool:
        """Return whether a prospective candidate value is finite and inside bounds."""
        try:
            arr = np.asarray(value, dtype=float)
            if arr.size == 0 or not np.all(np.isfinite(arr)):
                return False
            public = self._public_value(name, value)
            spec = self.variable_registry.get(name)
            return value_in_domain(public, spec.domain, zero_tol=0.0)
        except Exception:
            return False

    def _use_log_transform(self, name: str, var: Variable, init: float, lower: float) -> bool:
        """Return whether to pack a scalar variable logarithmically.

        The decision is purely structural/numerical: scalar variable, positive
        solver lower bound, positive finite initial value.  No variable-name or
        physics-category assumptions are used.
        """
        return bool(
            var.shape == 0
            and np.isfinite(lower)
            and lower > 0.0
            and np.isfinite(init)
            and init > 0.0
        )

    def _pack_scalar(
        self, name: str, var: Variable, init: float, lb: float, ub: float, *, scale_ref: Any, allow_log: bool = True
    ) -> tuple[float, float, float, float, str]:
        """Map one scalar to a solver coordinate ``(scale, offset, lower, upper, transform)``.

        Positive-bounded scalars pack logarithmically; others linearly with a
        tolerance/reference ``scale``.  ``allow_log=False`` forces linear packing.
        """
        scale = self._variable_scale(name, scale_ref)
        if allow_log and self._use_log_transform(name, var, init, lb):
            lower = np.log(lb / init) if np.isfinite(lb) and lb > 0.0 else -np.inf
            upper = np.log(ub / init) if np.isfinite(ub) and ub > 0.0 else np.inf
            return scale, init, lower, upper, "log"
        lower = (lb - init) / scale if np.isfinite(lb) else -np.inf
        upper = (ub - init) / scale if np.isfinite(ub) else np.inf
        return scale, init, lower, upper, "linear"

    def _pack_free_variables(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[Span]]:
        """Pack active non-fixed variables into one scaled vector.

        Returns:
            ``(x0, lower, upper, x_scale, spans)``.
        """
        x0: list[float] = []
        lower: list[float] = []
        upper: list[float] = []
        x_scale: list[float] = []
        spans: list[Span] = []
        transforms: dict[str, str] = {}
        self._uninitialized_free_variables = []
        for name in sorted(self.active_variable_names):
            var = self.variables_by_name[name]
            if name in getattr(self, "derived_variable_names", set()):
                continue
            if var.fixed:
                if var.input_value is None:
                    raise ValueError(f"Fixed variable {name!r} has no value.")
                continue
            spec = self.variable_registry.get(name)
            lb, ub = scipy_bounds(spec.solver_domain, zero_tol=self.zero_tol)
            size = self._variable_dim(name)
            start = len(x0)
            offsets: list[float] = []
            scales: list[float] = []
            try:
                initial_elements = [
                    float(self._initial_value(var, index=i if var.shape == 1 else None))
                    for i in range(size)
                ]
            except Exception:
                self._uninitialized_free_variables.append(name)
                continue
            for i, init in enumerate(initial_elements):
                ref = self._reference_for_movement(var, init, index=i if var.shape == 1 else None)
                scale, offset, lo, hi, transform = self._pack_scalar(name, var, init, lb, ub, scale_ref=ref)
                x0.append(0.0)
                lower.append(lo)
                upper.append(hi)
                x_scale.append(1.0)
                offsets.append(offset)
                scales.append(scale)
                if transform == "log":
                    transforms[name] = "log"
            spans.append(Span(name, start, len(x0), np.asarray(offsets, dtype=float), np.asarray(scales, dtype=float)))
        self._span_transforms = transforms
        return np.asarray(x0), np.asarray(lower), np.asarray(upper), np.asarray(x_scale), spans

    def _required_uninitialized_free_variables(self) -> list[str]:
        """Return uninitialized free variables required by enforced relations."""
        uninitialized = set(getattr(self, "_uninitialized_free_variables", ()))
        if not uninitialized:
            return []
        required: list[str] = []
        for name in sorted(uninitialized):
            for rel in self.relations:
                if rel.enforce and name in rel.variables:
                    required.append(name)
                    break
        return required

    def _record_uninitialized_failure(self, result: dict[str, Any]) -> bool:
        """Fill ``result`` and return True when an enforced relation needs an
        unsupplied, ungenerated free variable; the mode then returns ``result``."""
        required = self._required_uninitialized_free_variables()
        if not required:
            return False
        result["errors"].append(
            "Uninitialized required free variables: " + ", ".join(required)
            + ". These variables were not supplied and were not generated by active relations."
        )
        result["termination"] = "initialization failed"
        result["uninitialized_free_variables"] = list(getattr(self, "_uninitialized_free_variables", []))
        return True

    def _reject_unknown_options(self, result: dict[str, Any], unknown: Mapping[str, Any]) -> bool:
        """Record unknown mode option names on ``result``; True if any were given."""
        if not unknown:
            return False
        result["errors"].append(f"Unknown {result['mode']} option(s): " + ", ".join(sorted(unknown)))
        result["termination"] = "invalid options"
        return True

    def _values_from_vector(self, x: np.ndarray, spans: list[Span]) -> dict[str, Any]:
        """Reconstruct solver values from a packed vector.

        Args:
            x: Solver vector.
            spans: Variable packing spans.

        Returns:
            Variable value mapping.
        """
        values = self._values_from_variables(for_solver=True, skip_missing=True, complete=False, use_input_values=True)
        for name, start, stop, offsets, scales in spans:
            var = self.variables_by_name[name]
            local_x = np.asarray(x[start:stop], dtype=float)
            if getattr(self, "_span_transforms", {}).get(name) == "log":
                actual = offsets * np.exp(local_x)
            else:
                actual = offsets + scales * local_x
            values[name] = actual.copy() if var.shape == 1 else float(actual[0])
        return self._complete_values(values, strict=False)

    def _values_from_variables(
        self,
        *,
        for_solver: bool,
        skip_missing: bool = False,
        complete: bool = True,
        use_input_values: bool = False,
    ) -> dict[str, Any]:
        """Return a value map from variable state.

        Args:
            for_solver: Whether to convert values to solver units/shapes.
            skip_missing: Whether to omit missing variables from the map.
            complete: Whether to run structural completion on solver values.
            use_input_values: Whether to read immutable input values instead of current values.

        Returns:
            Variable value map in public or solver form.
        """
        values: dict[str, Any] = {}
        for name, var in self.variables_by_name.items():
            value = var.input_value if use_input_values else var.value
            if value is None:
                if not skip_missing and name not in getattr(self, "derived_variable_names", set()):
                    values[name] = None
                continue
            values[name] = self._solver_value(name, value) if for_solver else value
        if for_solver and complete:
            values = self._complete_values(values, strict=False)
        return values

    def _complete_values(self, values: dict[str, Any], *, strict: bool) -> dict[str, Any]:
        """Apply profile-shape reconstruction and derived providers.

        Args:
            values: Current namespace.
            strict: Whether unresolved providers should raise.

        Returns:
            Completed solver namespace.
        """
        out = self._with_profile_values(values, strict=strict)
        out = self._with_derived_values(out, strict=strict)
        return out

    def _with_derived_values(self, values: dict[str, Any], *, strict: bool) -> dict[str, Any]:
        """Complete a namespace using graph providers and defaults.

        Runtime completion is the only place where missing variables may be
        filled.  Values come from explicit active providers or from relations
        tagged as defaults.  No variable-name heuristic is used.

        Defaults are fallback providers: they fill only missing variables and do
        not override supplied values or values produced by explicit providers.
        This lets optional branches such as minority-fuel reactions remain
        well-defined without inventing variables from names.
        """
        explicit_providers = getattr(self, "derived_provider_by_output", {})
        default_providers = getattr(self, "default_provider_by_output", {})
        if not explicit_providers and not default_providers:
            return values

        out = dict(values)
        active_vars = set(getattr(self, "active_variable_names", ()))
        unresolved = set(explicit_providers)

        def missing(name: str) -> bool:
            return name not in out or out[name] is None

        def assign_from_relation(rel: Relation, *, only_missing: bool) -> bool:
            if any(missing(inp) for inp in rel.input_names):
                return False
            try:
                eval_values = self._relation_evaluation_values(rel, out)
                mapped = rel.output_map(rel.evaluate(eval_values))
            except Exception:
                if strict:
                    raise
                return False
            changed_local = False
            for out_name, out_value in mapped.items():
                if out_name not in self.variable_registry:
                    continue
                if out_name not in active_vars and out_name not in explicit_providers:
                    continue
                if only_missing and not missing(out_name):
                    continue
                try:
                    value = self._solver_value(out_name, out_value)
                except Exception:
                    if strict:
                        raise
                    continue
                old_missing = missing(out_name)
                old_value = out.get(out_name, None)
                out[out_name] = value
                unresolved.discard(out_name)
                if old_missing:
                    changed_local = True
                else:
                    try:
                        old_arr = np.asarray(old_value, dtype=float)
                        new_arr = np.asarray(value, dtype=float)
                        if old_arr.shape != new_arr.shape or not np.allclose(old_arr, new_arr, rtol=0.0, atol=1.0e-300):
                            changed_local = True
                    except Exception:
                        # Non-numeric values should be rare; treat an object
                        # inequality as progress only when Python can decide it.
                        if old_value != value:
                            changed_local = True
            return changed_local

        # Providers are evaluated in dependency order, so the acyclic majority
        # resolves in the first pass.  A few extra passes only matter for the
        # rare mutually-defined pair (for example a quasineutrality n_e<->n_i
        # cycle); the loop stops as soon as a pass changes nothing.
        plan = self._completion_plan()
        for _pass in range(6):
            changed = False
            for rel, only_missing in plan:
                if assign_from_relation(rel, only_missing=only_missing):
                    changed = True
            if not changed:
                break

        if strict:
            missing_explicit = {
                name: [inp for inp in explicit_providers[name].input_names if missing(inp)]
                for name in unresolved
                if missing(name)
            }
            missing_active = sorted(name for name in active_vars if missing(name))
            if missing_explicit or missing_active:
                raise ValueError(
                    "Could not complete active graph; "
                    f"missing explicit providers={missing_explicit}, "
                    f"missing active variables={missing_active}"
                )
        return out

    def _provider_graph(self) -> nx.DiGraph:
        """Directed provider graph over variable names, computed once.

        Each completed variable has one selected provider relation (an explicit
        derived provider wins over a default), carried on the node together with
        its ``only_missing`` flag, with an ``input -> output`` edge for every
        provider input.  Profile variables also get an ``average -> profile``
        edge so the scalar average control is a structural ancestor even for a
        supplied profile that has no provider relation.  Both the completion
        order (topological) and the Jacobian sparsity (ancestors) read this one
        graph instead of re-walking the providers by hand.
        """
        cached = getattr(self, "_provider_graph_cache", None)
        if cached is not None:
            return cached
        graph = nx.DiGraph()
        # One provider per variable; explicit ownership wins over a default.
        provider_of: dict[str, tuple[Relation, bool]] = {}
        for name, rel in getattr(self, "default_provider_by_output", {}).items():
            provider_of[name] = (rel, True)
        for name, rel in getattr(self, "derived_provider_by_output", {}).items():
            provider_of[name] = (rel, False)
        for out, (rel, only_missing) in provider_of.items():
            graph.add_node(out, provider=rel, only_missing=only_missing)
            for inp in rel.input_names:
                if inp != out:
                    graph.add_edge(inp, out)
        for profile, avg in getattr(self, "profile_average_by_name", {}).items():
            if avg != profile:
                graph.add_edge(avg, profile)
        self._provider_graph_cache = graph
        return graph

    def _completion_plan(self) -> list[tuple[Relation, bool]]:
        """Return derived/default providers in dependency order, computed once.

        A topological order over the provider graph lets ``_with_derived_values``
        complete the namespace in one linear pass instead of iterating a fixed
        point on every residual evaluation.  Condensation tolerates the rare
        provider cycle (e.g. a quasineutrality ``n_e<->n_i`` pair); within a
        component variables are emitted in name order.  Explicit providers
        recompute their output (``only_missing`` False); defaults only fill a
        still-missing output.
        """
        cached = getattr(self, "_completion_plan_cache", None)
        if cached is not None:
            return cached
        graph = self._provider_graph()
        condensation = nx.condensation(graph)
        ordered: list[tuple[Relation, bool]] = []
        placed: set[str] = set()
        for comp in nx.lexicographical_topological_sort(condensation, key=lambda c: min(condensation.nodes[c]["members"])):
            for name in sorted(condensation.nodes[comp]["members"]):
                rel = graph.nodes[name].get("provider")
                if rel is None or rel.name in placed:
                    continue
                placed.add(rel.name)
                ordered.append((rel, graph.nodes[name]["only_missing"]))
        self._completion_plan_cache = ordered
        return ordered

    def _evaluate_relation_residuals(self, values: dict[str, Any], *, strict: bool, solver_residuals: bool) -> tuple[dict[str, dict[str, Any]], np.ndarray, list[str], list[str]]:
        """Evaluate active relation residuals.

        During optimizer calls (``solver_residuals=True``), only numerical
        residuals are computed.  Full ``verify_status`` dictionaries are built
        only for independent verification after the solve.  This avoids calling
        relation functions twice for every optimizer residual evaluation.
        """
        status: dict[str, dict[str, Any]] = {}
        blocks: list[np.ndarray] = []
        errors: list[str] = []
        warnings: list[str] = []
        for rel in self.relations:
            if not self._relation_is_residual_relation(rel):
                # Structural provider: its outputs were recomputed by
                # _with_derived_values.  During optimization it contributes no
                # residual row.  During certification it is still checked as an
                # enforced relation on the completed value map.
                missing = [name for name in rel.variables if name not in values or values[name] is None]
                if missing:
                    message = f"Relation {rel.name!r} missing variables {missing}."
                    if not solver_residuals:
                        status[rel.name] = {
                            "relation": rel.name,
                            "verified": False,
                            "missing": missing,
                            "errors": [message],
                            "warnings": [],
                            "enforced": rel.enforce,
                            "source": "derived_provider",
                        }
                        if rel.enforce:
                            errors.append(message)
                    continue
                if not solver_residuals:
                    try:
                        eval_values = self._relation_evaluation_values(rel, values)
                        rel_status = self._verify_status(rel, eval_values)
                        rel_status["source"] = "derived_provider"
                        # Keep provider residuals out of the global certificate
                        # residual vector; source/status still records failures.
                        status[rel.name] = rel_status
                        warnings.extend(rel_status.get("warnings", []))
                        if rel.enforce and rel_status.get("errors"):
                            errors.extend(f"{rel.name}: {err}" for err in rel_status["errors"])
                    except Exception as exc:
                        status[rel.name] = {
                            "relation": rel.name,
                            "verified": False,
                            "errors": [str(exc)],
                            "warnings": [],
                            "enforced": rel.enforce,
                            "source": "derived_provider",
                        }
                        if rel.enforce:
                            errors.append(f"{rel.name}: {exc}")
                continue
            missing = [name for name in rel.variables if name not in values or values[name] is None]
            if missing:
                message = f"Relation {rel.name!r} missing variables {missing}."
                if strict and rel.enforce:
                    errors.append(message)
                    blocks.append(np.asarray([1.0e12]))
                if not solver_residuals:
                    status[rel.name] = {"relation": rel.name, "verified": False, "missing": missing, "errors": [message], "warnings": [], "enforced": rel.enforce}
                continue
            eval_values = self._relation_evaluation_values(rel, values)
            if solver_residuals:
                residual = self._residual_vector(rel, eval_values, safe=True)
                if rel.enforce:
                    blocks.append(residual)
                continue
            rel_status = self._verify_status(rel, eval_values)
            residual = self._residual_vector(rel, eval_values, safe=False)
            status[rel.name] = rel_status
            warnings.extend(rel_status.get("warnings", []))
            if rel.enforce:
                blocks.append(residual)
                if rel_status.get("errors"):
                    errors.extend(f"{rel.name}: {err}" for err in rel_status["errors"])
            elif not rel_status["verified"]:
                if self._is_default_relation(rel):
                    warnings.append(f"{rel.name}: weak default not satisfied after reconciliation")
                else:
                    errors.append(f"{rel.name}: check-only applicability failed")
        residuals = np.concatenate([block.reshape(-1) for block in blocks if block.size]) if blocks else np.empty(0, dtype=float)
        if solver_residuals and not np.all(np.isfinite(residuals)):
            residuals = np.nan_to_num(residuals, nan=1.0e12, posinf=1.0e12, neginf=-1.0e12)
        return status, residuals, errors, warnings

    def _sparsity_variable_names(self, name: str) -> set[str]:
        """Return variables that can affect one variable through the providers.

        These are ``name`` plus its ancestors in the provider graph (the same
        provider/average edges completion uses).  Conservative over-inclusion is
        always safe for sparse differencing; a missed dependency is what would
        corrupt the Jacobian.
        """
        graph = self._provider_graph()
        if name not in graph:
            return {name}
        return {name} | nx.ancestors(graph, name)

    def _build_jac_sparsity(self, spans: list[Span], reference: Mapping[str, Any] | None = None):
        """Build conservative residual-variable sparsity for SciPy coloring.

        The matrix rows must match the complete least-squares residual vector:
        enforced relation residuals followed by movement-penalty residuals.  The
        previous implementation omitted movement rows, which disabled sparse
        finite-difference coloring whenever movement penalties were enabled.
        """
        if not spans:
            return None
        span_by_name = {name: (start, stop) for name, start, stop, _offsets, _scales in spans}
        packed_names = set(span_by_name)
        values = self._values_from_vector(np.zeros(spans[-1][2], dtype=float), spans)

        row_specs: list[tuple[int, set[str]]] = []
        for rel in self.relations:
            if not self._relation_is_residual_relation(rel) or not rel.enforce:
                continue
            missing = [name for name in rel.variables if name not in values or values[name] is None]
            if missing:
                rdim = 1
            else:
                try:
                    eval_values = self._relation_evaluation_values(rel, values)
                    rdim = int(self._residual_vector(rel, eval_values, safe=True).size)
                except Exception:
                    rdim = max(1, self._relation_row_dim(rel))
            if rdim <= 0:
                continue
            names: set[str] = set()
            for var_name in rel.variables:
                names.update(self._sparsity_variable_names(var_name))
            row_specs.append((rdim, names))

        # Domain residual rows are part of the hard feasibility system.
        # They are zero inside physical domains and positive in tolerance units
        # outside them.  Derived outputs may violate domains even though they are
        # not packed directly, so dependencies recurse through structural
        # providers.
        for name in sorted(getattr(self, "active_variable_names", set())):
            if name not in values or values[name] is None or name not in self.variable_registry:
                continue
            spec = self.variable_registry.get(name)
            lower, upper, _li, _ui = spec.domain
            sides = int(lower is not None) + int(upper is not None)
            if sides <= 0:
                continue
            try:
                dim = int(np.asarray(values[name], dtype=float).reshape(-1).size)
            except Exception:
                continue
            if dim > 0:
                row_specs.append((dim * sides, self._sparsity_variable_names(name)))

        if reference is not None:
            # Movement residuals for directly packed variables: identity rows.
            for name, start, stop, _offsets, _scales in spans:
                if name not in values or name not in reference or reference[name] is None:
                    continue
                row_specs.append((int(stop - start), {name}))
            # Movement residuals for supplied variables derived from explicit
            # equations (for example profile fits) depend on the packed inputs
            # recursively reaching that derived variable.
            for name in sorted(getattr(self, "derived_variable_names", set()) - packed_names):
                var = self.variables_by_name.get(name)
                if var is None or var.input_value is None or name not in values or values[name] is None:
                    continue
                try:
                    rdim = int(np.asarray(values[name], dtype=float).reshape(-1).size)
                except Exception:
                    continue
                if rdim <= 0:
                    continue
                row_specs.append((rdim, self._sparsity_variable_names(name)))

        total_rows = sum(rdim for rdim, _names in row_specs)
        if total_rows <= 0:
            return None
        matrix = lil_matrix((total_rows, spans[-1][2]), dtype=bool)
        row = 0
        for rdim, names in row_specs:
            for var_name in names:
                if var_name in span_by_name:
                    start, stop = span_by_name[var_name]
                    matrix[row:row + rdim, start:stop] = True
            row += rdim
        return matrix.tocsr()

    def _domain_residuals(self, values: Mapping[str, Any]) -> np.ndarray:
        """Return physical-domain violation residuals in tolerance units.

        Domains are hard feasibility constraints.  This residual is zero inside
        the physical domain and positive/negative outside it, normalized by the
        variable tolerance width.  It is used only by reconcile/optimize solver
        objectives; final success is still decided by ``_domain_errors`` and
        canonical relation verification.
        """
        rows: list[np.ndarray] = []
        for name in sorted(getattr(self, "active_variable_names", set())):
            if name not in values or values[name] is None or name not in self.variable_registry:
                continue
            spec = self.variable_registry.get(name)
            lower, upper, lower_inc, upper_inc = spec.domain
            if lower is None and upper is None:
                continue
            try:
                arr = np.asarray(values[name], dtype=float).reshape(-1)
            except Exception:
                rows.append(np.asarray([1.0e12], dtype=float))
                continue
            if not np.all(np.isfinite(arr)):
                rows.append(np.full(max(1, arr.size), 1.0e12, dtype=float))
                continue
            tol = np.maximum(self._variable_tolerance_width(name, np.maximum(np.abs(arr), self._tolerance_scale_floor(name))), 1.0e-300)
            if lower is not None:
                boundary = float(lower) + (self.zero_tol if not lower_inc else 0.0)
                rows.append(np.maximum(boundary - arr, 0.0) / tol)
            if upper is not None:
                boundary = float(upper) - (self.zero_tol if not upper_inc else 0.0)
                rows.append(np.maximum(arr - boundary, 0.0) / tol)
        return np.concatenate(rows) if rows else np.empty(0, dtype=float)

    def _movement_residuals(self, values: Mapping[str, Any], reference: Mapping[str, Any], spans: list[Span]) -> np.ndarray:
        """Return normalized movement from reference values.

        Args:
            values: Current values.
            reference: Reference values.
            spans: Packed variable spans.

        Returns:
            One-dimensional residual vector.
        """
        rows: list[np.ndarray] = []
        packed = set()
        for name, _start, _stop, _offsets, scales in spans:
            packed.add(name)
            if name not in values or name not in reference or reference[name] is None:
                continue
            delta = np.asarray(values[name], dtype=float).reshape(-1) - np.asarray(reference[name], dtype=float).reshape(-1)
            ref = np.asarray(reference[name], dtype=float).reshape(-1)
            tol = self._variable_movement_tolerance(name, ref)
            rows.append(delta / np.maximum(np.broadcast_to(tol, delta.shape), 1.0e-300))
        # A supplied variable can be derived from an explicit output relation
        # instead of being packed directly, e.g. a profile generated from
        # average+peaking. Its original supplied value is still soft evidence.
        for name in sorted(getattr(self, "derived_variable_names", set()) - packed):
            var = self.variables_by_name.get(name)
            if var is None or var.input_value is None or name not in values or values[name] is None:
                continue
            delta = np.asarray(values[name], dtype=float).reshape(-1) - np.asarray(self._solver_value(name, var.input_value), dtype=float).reshape(-1)
            tol = self._variable_movement_tolerance(name, var.input_value)
            rows.append(delta / np.maximum(tol, 1.0e-300))
        return np.concatenate(rows) if rows else np.empty(0, dtype=float)

    def _fixed_value_errors(self, values: Mapping[str, Any]) -> list[str]:
        """Return errors for fixed variables changed in a candidate value map."""
        errors: list[str] = []
        for name, var in self.variables_by_name.items():
            if not var.fixed or var.input_value is None or name not in values or values[name] is None:
                continue
            try:
                old = np.asarray(self._solver_value(name, var.input_value), dtype=float).reshape(-1)
                new = np.asarray(values[name], dtype=float).reshape(-1)
                if old.shape != new.shape or not np.allclose(old, new, rtol=0.0, atol=max(self.zero_tol, 1e-10 * max(1.0, float(np.max(np.abs(old))) if old.size else 1.0))):
                    errors.append(f"Fixed variable {name!r} changed during candidate solve.")
            except Exception as exc:
                errors.append(f"Could not validate fixed variable {name!r}: {exc}")
        return errors

    def _domain_errors(self, values: Mapping[str, Any]) -> list[str]:
        """Return variable-domain errors for a candidate value map."""
        errors: list[str] = []
        for name, value in values.items():
            if name not in self.variable_registry or value is None:
                continue
            spec = self.variable_registry.get(name)
            if spec.domain is None:
                continue
            try:
                # Validate the raw candidate value against the physical
                # domain.  Do not call _public_value here, because that may
                # project a violating value back onto a solver boundary and hide
                # an invalid reconciliation candidate.
                if not value_in_domain(value, spec.domain, zero_tol=0.0):
                    errors.append(f"Variable {name!r} violates domain {spec.domain!r}.")
            except Exception as exc:
                errors.append(f"Could not validate domain for variable {name!r}: {exc}")
        return errors

    def _store_solved_values(self, values: Mapping[str, Any]) -> None:
        """Overwrite current public values from a solver-domain value map.

        Inputs are not modified. Fixed variables keep their input-only state.
        """
        names = sorted((set(self.active_variable_names) | set(values)) & set(self.variables_by_name))
        for name in names:
            var = self.variables_by_name[name]
            if var.fixed or name not in values or values[name] is None:
                continue
            try:
                var.set_value(self._public_value(name, values[name]))
            except Exception:
                continue
        # Keep profile-average controls consistent with the stored profiles.
        # A later system built from these stored values would otherwise invent
        # the missing average input itself, so re-running a mode on the solved
        # state would appear to create new values.
        for name in names:
            var = self.variables_by_name[name]
            if var.shape != 1 or name == "rho" or var.value is None:
                continue
            avg_name = self.profile_average_by_name.get(name) or self._profile_average_name(name)
            if avg_name is None or avg_name not in self.variables_by_name:
                continue
            avg_var = self.variables_by_name[avg_name]
            if avg_var.fixed or avg_var.value is not None:
                continue
            try:
                average = self._profile_average(self._solver_value(name, var.value))
                avg_var.set_value(self._public_value(avg_name, average))
            except Exception:
                continue

    def _coerce_to_registry_shape(self, name: str, value: Any) -> Any:
        """Return ``value`` with the shape declared for ``name`` in the registry.

        Scalar registry variables must remain scalar.  A profile-shaped value for
        a scalar variable is a relation/planning error and is rejected instead of
        being displayed or stored as a fake scalar result.  Profile variables may
        receive a scalar, which is broadcast to the active profile grid.
        """
        if name not in self.variable_registry:
            return value
        spec = self.variable_registry.get(name)
        size: int | None = None
        if spec.shape == 1:
            var = self.variables_by_name.get(name)
            size = int((var.size if var is not None else None) or self.profile_size)
        coerced, _size = coerce_to_shape(
            name, value, is_profile=spec.shape == 1, size=size, squeeze_scalar=True, reject_nan=True
        )
        return coerced

    def _solver_value(self, name: str, value: Any) -> Any:
        """Convert a public value to canonical solver shape.

        Values lying exactly on a physical-domain boundary are projected onto
        the corresponding solver-domain boundary; this is the inverse of
        ``_public_value``, so a profile edge stored publicly as ``T = 0`` is
        evaluated at the numerically safe solver bound (for example 1e-12)
        instead of hitting singular physics formulas.  Only values within
        ``zero_tol`` of the boundary are projected: interior values and real
        domain violations are never clipped.
        """
        shaped = self._coerce_to_registry_shape(name, value)
        arr = np.asarray(shaped, dtype=float)
        out = arr.astype(float, copy=True)
        if name in self.variable_registry:
            spec = self.variable_registry.get(name)
            d_lo, d_hi, d_lo_inc, d_hi_inc = spec.domain
            s_lo, s_hi = scipy_bounds(spec.solver_domain, zero_tol=self.zero_tol)
            if d_lo is not None:
                lo = float(d_lo)
                if np.isfinite(s_lo) and s_lo > lo:
                    target = float(s_lo)
                elif not d_lo_inc:
                    target = lo + self.zero_tol
                else:
                    target = None
                if target is not None:
                    out = np.where((out >= lo - self.zero_tol) & (out <= lo + self.zero_tol), target, out)
            if d_hi is not None:
                hi = float(d_hi)
                if np.isfinite(s_hi) and s_hi < hi:
                    target = float(s_hi)
                elif not d_hi_inc:
                    target = hi - self.zero_tol
                else:
                    target = None
                if target is not None:
                    out = np.where((out >= hi - self.zero_tol) & (out <= hi + self.zero_tol), target, out)
        return float(out) if out.ndim == 0 else out

    def _check_solver_domain_value(self, name: str, value: Any) -> None:
        """Raise if ``value`` is outside the variable solver domain."""
        spec = self.variable_registry.get(name)
        lb, ub = scipy_bounds(spec.solver_domain, zero_tol=self.zero_tol)
        arr = np.asarray(self._solver_value(name, value), dtype=float)
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Variable {name!r} initial value is not finite.")
        if np.isfinite(lb) and np.any(arr < lb):
            raise ValueError(f"Variable {name!r} initial value is below solver_domain lower bound {lb}.")
        if np.isfinite(ub) and np.any(arr > ub):
            raise ValueError(f"Variable {name!r} initial value is above solver_domain upper bound {ub}.")

    def _relation_evaluation_values(self, rel: Relation, values: Mapping[str, Any]) -> dict[str, Any]:
        """Return a solver-safe namespace for one relation evaluation.

        Args:
            rel: Relation about to be evaluated.
            values: Current public or solver namespace.

        Returns:
            Copy of ``values`` with registry variables coerced to canonical
            solver shapes.  No solver-domain clipping is performed here.
        """
        out = dict(values)
        # Relation inputs are coerced to canonical solver shape/unit only.
        # Domains and solver domains are checked by residuals, bounds, and final
        # verification; they are not algebraic projections.
        for name in set(rel.input_names) | set(rel.constant_names):
            if name in out and out[name] is not None and name in self.variable_registry:
                out[name] = self._solver_value(name, out[name])
        return out

    def _residual_vector(self, rel: Relation, eval_values: Mapping[str, Any], *, safe: bool) -> np.ndarray:
        """Return one relation's scaled residual vector using system tolerances."""
        return rel.residual_vector(eval_values, scales=self.variable_scales, rel_tols=self.variable_tolerances, abs_tols=self.variable_abs_tolerances, safe=safe)

    def _verify_status(self, rel: Relation, eval_values: Mapping[str, Any]) -> dict[str, Any]:
        """Return one relation's verify status using system tolerances."""
        return rel.verify_status(eval_values, scales=self.variable_scales, rel_tols=self.variable_tolerances, abs_tols=self.variable_abs_tolerances)

    def _public_value(self, name: str, value: Any) -> Any:
        """Project solver-boundary values to physical-domain boundary values.

        Args:
            name: Variable name.
            value: Solver value.

        Returns:
            Public value inside the physical domain.
        """
        spec = self.variable_registry.get(name)
        d_lo, d_hi, d_lo_inc, d_hi_inc = spec.domain
        s_lo, s_hi = domain_bounds_for_solver(spec.solver_domain, zero_tol=self.zero_tol)
        arr = np.asarray(self._coerce_to_registry_shape(name, value), dtype=float).copy()
        if s_lo is not None and d_lo is not None and d_lo_inc and not np.isclose(float(s_lo), float(d_lo), rtol=0.0, atol=self.zero_tol):
            arr = np.where(np.isclose(arr, s_lo, rtol=0.0, atol=max(self.zero_tol, abs(s_lo) * 1e-10)), d_lo, arr)
        if s_hi is not None and d_hi is not None and d_hi_inc and not np.isclose(float(s_hi), float(d_hi), rtol=0.0, atol=self.zero_tol):
            arr = np.where(np.isclose(arr, s_hi, rtol=0.0, atol=max(self.zero_tol, abs(s_hi) * 1e-10)), d_hi, arr)
        return float(arr) if arr.ndim == 0 else arr

    def _refresh_scales(self) -> None:
        """Refresh variable scales and tolerances used by residuals.

        Domains and solver domains are admissible-value constraints, not
        numerical scales.  The finite scale floor comes from abs_tol / rel_tol,
        while current/reference magnitudes provide relative scaling.
        """
        self.variable_tolerances = {
            name: float(var.rel_tol or self.variable_registry.rel_tol_default)
            for name, var in self.variables_by_name.items()
        }
        self.variable_abs_tolerances = {
            name: float(getattr(var, "abs_tol", self.variable_registry.get(name).abs_tol))
            for name, var in self.variables_by_name.items()
        }
        self.variable_scales = {
            name: self._variable_scale(name, self._current_reference(name))
            for name in self.variables_by_name
        }

    def _current_reference(self, name: str) -> float:
        var = self.variables_by_name[name]
        for value in (var.value, var.input_value):
            if value is not None:
                arr = np.asarray(value, dtype=float).reshape(-1)
                finite = arr[np.isfinite(arr)]
                if finite.size:
                    return float(np.max(np.abs(finite)))
        return 0.0

    def _tolerance_scale_floor(self, name: str) -> float:
        """Return the finite scale floor implied by abs_tol / rel_tol."""
        try:
            spec = self.variable_registry.get(name)
            rel_tol = float(self.variables_by_name.get(name).rel_tol if name in self.variables_by_name else spec.rel_tol)
            abs_tol = float(getattr(self.variables_by_name.get(name), "abs_tol", spec.abs_tol) if name in self.variables_by_name else spec.abs_tol)
        except Exception:
            rel_tol = float(self.variable_registry.rel_tol_default)
            abs_tol = 0.0
        if rel_tol > 0.0 and abs_tol > 0.0:
            return max(abs_tol / rel_tol, 1.0e-300)
        if abs_tol > 0.0:
            return max(abs_tol, 1.0e-300)
        return 1.0e-300

    def _finite_magnitudes(self, *values: Any) -> list[float]:
        """Return finite absolute magnitudes from scalars or arrays."""
        out: list[float] = []
        for value in values:
            if value is None:
                continue
            try:
                arr = np.asarray(value, dtype=float).reshape(-1)
            except Exception:
                continue
            finite = arr[np.isfinite(arr)]
            if finite.size:
                out.append(float(np.max(np.abs(finite))))
        return out

    def _variable_scale(self, name: str, *values: Any) -> float:
        """Return the residual/movement scale for one variable.

        The scale is the largest of the abs_tol / rel_tol floor and the finite
        magnitudes of the supplied reference values.  It intentionally ignores
        physical and solver-domain bounds, including unbounded or artificial
        large bounds.  Both relation-residual scaling and movement-penalty
        scaling use this same definition.
        """
        return max([self._tolerance_scale_floor(name), *self._finite_magnitudes(*values)])

    def _variable_tolerance_width(self, name: str, scale: Any) -> np.ndarray:
        """Return physical tolerance width for a variable and scale."""
        rel_tol = float(self.variable_tolerances.get(name, self.variable_registry.rel_tol_default))
        abs_tol = float(self.variable_abs_tolerances.get(name, 0.0))
        scl = np.maximum(np.asarray(scale, dtype=float), 1.0e-300)
        return np.maximum(abs_tol, rel_tol * scl)

    def _variable_movement_tolerance(self, name: str, reference: Any) -> np.ndarray:
        """Return tolerance-width denominator for movement from supplied reference."""
        ref_scale = self._variable_scale(name, reference)
        return self._variable_tolerance_width(name, ref_scale)

    def _reference_for_movement(self, var: Variable, fallback: Any, index: int | None = None) -> Any:
        """Return supplied reference value for movement scaling, or fallback for missing variables."""
        if var.input_value is not None:
            try:
                arr = np.asarray(self._solver_value(var.name, var.input_value), dtype=float).reshape(-1)
                if arr.size:
                    return float(arr[min(index or 0, arr.size - 1)])
            except Exception:
                pass
        return fallback

    def _initial_value(self, var: Variable, index: int | None = None) -> float:
        """Return an initial value for one variable element.

        Initial values may come only from user input or relation-generated guesses.
        Solver domains are constraints, not value providers.
        """
        initial_values = getattr(self, "_initial_guesses", {})

        # Relation-generated values are x0 hints, not movement references.  They
        # may override supplied non-fixed values for initialization only; the
        # original public value remains in the movement reference map.
        if var.name in initial_values and initial_values[var.name] is not None:
            solver_value = self._solver_value(var.name, initial_values[var.name])
            self._check_solver_domain_value(var.name, solver_value)
            arr = np.asarray(solver_value, dtype=float).reshape(-1)
            if arr.size:
                return float(arr[min(index or 0, arr.size - 1)])

        if var.input_value is not None:
            solver_value = self._solver_value(var.name, var.input_value)
            self._check_solver_domain_value(var.name, solver_value)
            arr = np.asarray(solver_value, dtype=float).reshape(-1)
            if arr.size:
                return float(arr[min(index or 0, arr.size - 1)])

        # A block core (the free unknown of a determined block, for example V_p
        # inverted from the supplied P_fus) is not forward-reachable, so it has
        # no seed.  It is determined by the global solve against its block's
        # supplied anchor; the start here is only a numerical initial point, not
        # an invented physical value, so a determined block converges to the
        # same unique answer regardless.  The magnitude comes from the variable
        # tolerance scale, which the log transform then explores.
        if var.name in getattr(self, "block_core_names", set()):
            return float(self._tolerance_scale_floor(var.name))

        raise ValueError(
            f"No initial value for variable {var.name!r}: it was not supplied "
            "and was not generated by an active relation."
        )

    def _relation_row_dim(self, rel: Relation) -> int:
        """Return the number of scalar comparison rows the relation produces.

        Output relations contribute one comparison per output dimension.
        Outputless residual relations contribute one row, vectorized over the
        profile grid when they touch profile variables.
        """
        if rel.output_names:
            return sum(self._variable_dim(name) for name in rel.output_names if name in self.variable_registry)
        return max([1, *(self._variable_dim(name) for name in rel.variables if name in self.variable_registry and self.variable_registry.get(name).shape == 1)])

    def _variable_dim(self, name: str) -> int:
        spec = self.variable_registry.get(name)
        if spec.shape != 1:
            return 1
        return int(self.variables_by_name.get(name).size or self.profile_size)

    def _resolve_relation_names(self, rel: Relation) -> Relation:
        inputs = tuple(self.variable_registry.resolve(name) if name in self.variable_registry else name for name in rel.input_names)
        outputs = tuple(self.variable_registry.resolve(name) if name in self.variable_registry else name for name in rel.output_names)
        if inputs == rel.input_names and outputs == rel.output_names:
            return rel
        return Relation(name=rel.name, func=rel.func, input_names=inputs, outputs=outputs, op=rel.op, rhs=rel.rhs, tags=rel.tags, enforce=rel.enforce, constraints=rel.constraints, source_kind=rel.source_kind, source_name=rel.source_name, constant_names=rel.constant_names, dependency=rel.dependency, function_name=rel.function_name, argument_names=rel.argument_names)

    def _resolve_names(self, names: Iterable[str]) -> set[str]:
        out: set[str] = set()
        for name in names:
            text = str(name)
            out.add(self.variable_registry.resolve(text) if text in self.variable_registry else text)
        return out

    def _ensure_variable_exists(self, raw_name: str) -> Variable:
        name = self.variable_registry.resolve(raw_name) if raw_name in self.variable_registry else str(raw_name)
        if name in self.variables_by_name:
            return self.variables_by_name[name]
        if name not in self.variable_registry:
            raise ValueError(f"Relation requires unknown variable {name!r}.")
        spec = self.variable_registry.get(name)
        var = Variable(name, size=self.profile_size if spec.shape == 1 else None)
        self.variables.append(var)
        self.variables_by_name[name] = var
        return var

    def _is_default_relation(self, rel: Relation) -> bool:
        return "default" in set(rel.tags) or str(rel.source_kind).startswith("default")

    def _ordered_single_relation(self, item: Any) -> Relation:
        if isinstance(item, Relation):
            return item
        name = str(item)
        if name not in self.relations_by_name:
            raise KeyError(f"Unknown ordered relation {name!r}.")
        return self.relations_by_name[name]

    def _solve_ordered_block(self, rels: list[Relation], values: dict[str, Any], result: dict[str, Any]) -> bool:
        unknowns: list[str] = []
        for rel in rels:
            for name in rel.variables:
                self._ensure_variable_exists(name)
                if name not in values or values[name] is None:
                    if self.variables_by_name[name].fixed:
                        result["errors"].append(f"Fixed variable {name!r} in ordered block has no value.")
                        return False
                    if name not in unknowns:
                        unknowns.append(name)
        if not unknowns:
            return all(self._verify_status(rel, self._relation_evaluation_values(rel, values))["verified"] for rel in rels)

        # Primary path: delegate to the shared block solver used by reconcile's
        # initial computation (``_solve_initial_block``), which packs positive
        # unknowns logarithmically and refines starts with a coarse log-grid scan.
        # Using the same routine for ordered blocks and reconcile means an ordered
        # 2x2 block (e.g. solving tau_E/P_loss from the energy-confinement scaling
        # and the W_th = P_loss * tau_E balance) converges wherever reconcile does.
        # The local linear solve below is kept only as a fallback for structural
        # cases the shared solver declines (e.g. a profile-valued numerical core).
        block = self._solve_initial_block(tuple(unknowns), rels, values, residual_tol=1.0)
        if block is not None:
            for name, value in block["values"].items():
                values[name] = value
                self.variables_by_name[name].set_value(self._public_value(name, value))
            if all(self._verify_status(rel, self._relation_evaluation_values(rel, values))["verified"] for rel in rels):
                return True
            result["errors"].append("Ordered solve block did not verify.")
            return False

        spans: list[Span] = []
        x0: list[float] = []
        lower: list[float] = []
        upper: list[float] = []
        for name in unknowns:
            var = self.variables_by_name[name]
            lb, ub = scipy_bounds(self.variable_registry.get(name).solver_domain, zero_tol=self.zero_tol)
            size = self._variable_dim(name)
            start = len(x0)
            offsets = []
            scales = []
            for i in range(size):
                try:
                    init = float(self._initial_value(var, index=i if var.shape == 1 else None))
                except ValueError:
                    known_start = self._block_start_from_knowns(rels, values, lb, ub)
                    if known_start is None:
                        result["errors"].append(f"No initial value available for {name!r} in ordered block.")
                        return False
                    init = known_start
                ref = self._reference_for_movement(var, init, index=i if var.shape == 1 else None)
                scale, offset, lo, hi, _transform = self._pack_scalar(name, var, init, lb, ub, scale_ref=ref, allow_log=False)
                x0.append(0.0)
                lower.append(lo)
                upper.append(hi)
                offsets.append(offset)
                scales.append(scale)
            spans.append(Span(name, start, len(x0), np.asarray(offsets), np.asarray(scales)))

        def block_values(x: np.ndarray) -> dict[str, Any]:
            out = dict(values)
            for name, start, stop, offsets, scales in spans:
                var = self.variables_by_name[name]
                actual = offsets + scales * x[start:stop]
                out[name] = actual.copy() if var.shape == 1 else float(actual[0])
            return out

        def residual(x: np.ndarray) -> np.ndarray:
            local = block_values(x)
            blocks = [self._residual_vector(rel, self._relation_evaluation_values(rel, local), safe=True) for rel in rels if rel.enforce]
            return np.concatenate(blocks) if blocks else np.empty(0)

        try:
            probe = residual(np.asarray(x0))
            if probe.size < len(x0):
                result["errors"].append(f"Ordered solve block is underdetermined: {probe.size} residuals for {len(x0)} unknowns {unknowns}.")
                return False
            sol = least_squares(residual, np.asarray(x0), bounds=(np.asarray(lower), np.asarray(upper)), method="trf", max_nfev=200)
        except Exception as exc:
            result["errors"].append(f"Ordered solve block failed: {exc}")
            return False
        solved = block_values(sol.x)
        if residual(sol.x).size and float(np.max(np.abs(residual(sol.x)))) > 1e-6:
            result["errors"].append("Ordered solve block did not verify.")
            return False
        for name in unknowns:
            values[name] = solved[name]
            self.variables_by_name[name].set_value(self._public_value(name, solved[name]))
        return True

    def _blocking_compiler_issues(self) -> list[str]:
        """Return inactive relations that should make the system fail.

        Blocked optional relations are diagnostics, alternative decompositions,
        or relations whose inputs are not available in the reactor record.  They
        should be reported, but they should not make a reconcile fail when all
        active enforced relations verify.  Only relations needed to satisfy an
        explicit target/solve_for request are blocking.
        """
        requested = set(self.targets) | set(self.solve_for)
        if not requested:
            return []
        out = [
            f"requested variable {name!r} is structurally underdetermined"
            for name in getattr(self, "underdetermined_requests", ())
        ]
        # Benign inactivations (the relation was a fallback that was not needed,
        # or an authoritative supplied-profile measurement) never block.  Any
        # other reason that reaches a requested variable is reported.
        benign = {"inactive_default_not_needed", "inactive_profile_supplied_fixed"}
        for name, reason in self.blocked_relation_reasons.items():
            if reason in benign:
                continue
            rel = self.relations_by_name.get(name)
            if rel is not None and requested.intersection(set(rel.variables)):
                out.append(f"blocked relation {name!r} touches a requested variable ({reason})")
        return out

    def _classify_variables(self, relation_status: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for name, var in self.variables_by_name.items():
            out[name] = {
                "has_input": var.input_value is not None,
                "value_present": var.value is not None,
                "fixed": bool(var.fixed),
            }
        return out

    def _rank_input_culprits(self, relation_status: Mapping[str, Mapping[str, Any]], variable_status: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
        counts: dict[str, int] = {}
        for rel in self.relations:
            status = relation_status.get(rel.name, {})
            if status.get("verified", True):
                continue
            for name in rel.variables:
                var = self.variables_by_name.get(name)
                if var is not None and not var.fixed:
                    counts[name] = counts.get(name, 0) + 1
        return [{"name": name, "count": count} for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]

    def _new_result(self, mode: str) -> dict[str, Any]:
        return {"mode": mode, "success": False, "termination": "not run", "errors": [], "warnings": [], "compiler_report": getattr(self, "compiler_report", {})}

    def _result_from_certificate(
        self,
        mode: str,
        certificate: Mapping[str, Any],
        *,
        termination: str,
        solver: Mapping[str, Any] | None = None,
        include_values: bool = False,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a mode result dict from a verification certificate.

        Shared by verify/reconcile/optimize so the common result shape (status,
        residuals, certificate, graph and compiler views) is assembled in one
        place.  ``solver`` and ``values`` are added only when supplied, and
        ``extra`` overlays any mode-specific keys.
        """
        result = self._new_result(mode)
        result.update(
            {
                "relation_status": certificate["relation_status"],
                "residuals": certificate["residuals"].tolist(),
                "errors": certificate["errors"],
                "warnings": certificate["warnings"],
                "variable_status": self._classify_variables(certificate["relation_status"]),
                "termination": termination,
                "success": bool(certificate["verified"]),
                "verified": bool(certificate["verified"]),
                "certificate": {k: v for k, v in certificate.items() if k not in {"residuals", "values"}},
                "variables": self.variables_by_name,
                "relations": self.primary_relations,
                "graph": self.graph,
                "compiler_report": self.compiler_report,
            }
        )
        if include_values:
            result["values"] = certificate["values"]
        if solver is not None:
            result["solver"] = dict(solver)
        if extra:
            result.update(extra)
        return result

    def _solver_report(self, **overrides: Any) -> dict[str, Any]:
        """Return a solver-metadata block with neutral defaults.

        Modes override only the fields they have; the no-solve paths keep the
        same key set without hand-writing a full block of zeros.
        """
        report: dict[str, Any] = {
            "backend": "none",
            "success": True,
            "status": 0,
            "cost": 0.0,
            "optimality": 0.0,
            "nfev": 0,
            "message": "",
            "residual_calls": 0,
            "residual_eval_time_s": 0.0,
            "residual_size": 0,
            "solver_dim": 0,
            "jac_sparsity_used": False,
            "jac_sparsity_shape": None,
            "residual_eval_mean_ms": 0.0,
            "relation_weight": 0.0,
            "relation_weight_schedule": [],
            "phase_schedule": [],
            "stage_history": [],
            "movement_weight": 0.0,
            "initial_guess_variables": 0,
        }
        report.update(overrides)
        return report
