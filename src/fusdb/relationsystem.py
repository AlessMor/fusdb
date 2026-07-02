"""RelationSystem container, graph compiler and mode dispatcher."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import networkx as nx
import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .relation import Relation, canonicalize_relation, canonicalize_relation_names, constraint_from_expression
from .registry import VARIABLES
from .utils import ZERO_TOL, parse_constraint_specs, scipy_bounds, value_in_domain
from .variable import Variable


# ── Pure structural helpers ────────────────────────────────────────────────
# Module-level functions with explicit inputs: they read nothing off a
# RelationSystem, so the compile pass's dataflow stays visible at call sites.


def _is_default_relation(rel: Relation) -> bool:
    """Whether a relation is a weak default (fallback provider / x0 seed)."""
    return "default" in set(rel.tags) or str(rel.source_kind).startswith("default")


def _scalar_start_grid(lb: float, ub: float) -> list[float]:
    """Return signed log-spaced start candidates inside the solver interval."""
    points: set[float] = set()
    for exponent in range(-30, 31, 2):
        for sign in (1.0, -1.0):
            value = sign * 10.0**exponent
            if (not np.isfinite(lb) or value >= lb) and (not np.isfinite(ub) or value <= ub):
                points.add(value)
    return sorted(points)


def _forward_decision_rounds(
    relations: list[Relation], supplied: Iterable[str], extra_known: Iterable[str] = ()
) -> tuple[dict[str, int], dict[str, Relation]]:
    """Return forward-closure rounds and per-variable forward providers.

    ``supplied`` names are decided at round 0; ``extra_known`` seeds additional
    variables as decided at round 0 -- used to treat block cores as available
    so the block-downstream variables get forward providers.

    Each round, real relations are exhausted before defaults, so a variable is
    owned by a default only when no real relation can decide it.  A relation
    decides a variable either forward (all inputs known, every output decided)
    or acausally (a single remaining variable from the rest).  The returned
    ``decider`` maps a variable to the relation that first decided it as one of
    its declared outputs -- the forward provider for completion.  Variables
    decided only acausally (as an input, e.g. ``a`` from ``A = R/a``) or never
    reached have no ``decider`` entry and are packed as free variables for the
    global solve.
    """
    rounds: dict[str, int] = {name: 0 for name in supplied}
    for name in extra_known:
        rounds.setdefault(name, 0)
    decider: dict[str, Relation] = {}
    non_default = [rel for rel in relations if not _is_default_relation(rel)]
    defaults = [rel for rel in relations if _is_default_relation(rel)]
    # Pure-input variables: produced by no relation here and referenced only
    # by outputless (constraint) relations.  The acausal fallback must not
    # solve such a variable as "the last unknown" of a constraint (e.g. an
    # unsupplied tau_p that only the particle balances reference) -- it is a
    # free parameter, not a value the constraint determines.  Leaving it
    # undecided lets the structural partition mark it underdetermined and
    # deactivate the constraints that need it.
    produced_any = {out for rel in relations for out in rel.output_names}
    ref_rels: dict[str, list[Relation]] = {}
    for rel in relations:
        for v in rel.variables:
            ref_rels.setdefault(v, []).append(rel)
    pure_input = {
        v for v, rels in ref_rels.items()
        if v not in produced_any and all((not r.outputs and r.op == "==") for r in rels)
    }
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
                if len(undecided) == 1 and undecided[0] not in rounds and undecided[0] not in pure_input:
                    v = undecided[0]
                    rounds[v] = round_no
                    if v in rel.output_names:
                        decider.setdefault(v, rel)
                    changed = True
    return rounds, decider


def _structural_block_plan(
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


class RelationSystem:
    """Variables and relations compiled into one numeric system.

    Execution modes (:mod:`fusdb.modes`) drive a compiled system through this
    public interface and own their own algorithm and result shape:

    - :meth:`compile` -- build/prune the active system (``run`` calls this).
    - :meth:`pack` / :meth:`unpack` -- free variables <-> solver vector.
      ``pack`` stores the packed layout (``packed_specs``); the residual
      helpers read it.
    - :meth:`values` / :meth:`complete` -- read the value namespace; close it.
    - :meth:`solver_residual_vector` / :meth:`domain_residuals` /
      :meth:`movement_residuals` -- the residual blocks (modes weight them);
      :meth:`certify_relations` -- the certification statuses.
    - :meth:`store` -- write solved values back into the variables.
    - :meth:`initial_values_from_graph`, :meth:`build_jac_sparsity`,
      :meth:`movement_weights`, :meth:`refresh_scales` -- solve
      setup/initialization helpers.

    Args:
        variables: Scenario variables.
        relations: Post-filter relation candidates.
        constraints: System-level constraints.
        name: Diagnostic system name.
    """

    def __init__(
        self,
        variables: Iterable[Variable],
        relations: Iterable[Relation],
        *,
        constraints: Any = None,
        name: str | None = None,
    ) -> None:
        self.name = str(name or "relation_system")
        # Result of the most recent run(); read by variables_table for header
        # colouring. None until a mode has been dispatched.
        self.last_result: dict[str, Any] | None = None
        # Single read point for registry metadata; Variable/Relation are bound
        # to the same shared global registry.
        self.variable_registry = VARIABLES
        # Variables found unevaluable by the prune pass (no value, no producer,
        # no anchored block); relations that need them are deactivated.
        self._unevaluable_names: set[str] = set()
        # Packed free-variable layout, one record per packed variable
        # ``(name, start, stop, offsets, scales, shape, transform)``; written by
        # pack() together with the runtime completion plan it caches.
        self.packed_specs: list[tuple[str, int, int, np.ndarray, np.ndarray, int, str | None]] = []
        self.packed_dim: int = 0

        # The single variable container; insertion-ordered.
        self.variables_by_name: dict[str, Variable] = {}
        for var in variables:
            if var.name in self.variables_by_name:
                raise ValueError(f"Duplicate variable {var.name!r}.")
            self.variables_by_name[var.name] = var

        # Construction phases: each reads the state left by earlier phases
        # and writes its own attributes onto self.
        self._canonicalize_candidates(relations, constraints)
        self._infer_profile_size()
        self._materialize_rho_grid()
        self._broadcast_profile_values()
        self._split_supplied_profiles()

        # Declare the remaining lifecycle state, then build the static
        # active set only (no runtime initialization/pruning).  run()
        # dispatches the default seeding and the initial-value computation used
        # to prune unevaluable variables, so constructing a RelationSystem does
        # no solve-oriented initialization and writes no relation-derived
        # outputs into Variable.value; missing outputs are completed in local
        # value maps during residual evaluation and stored only after a
        # solve/ordered run finishes.
        #
        # Attribute lifecycle from here on:
        #   * compile products -- written by every _run_compile_pass:
        #     constant_default_values, default_provider_by_output,
        #     structural_blocks, primary_relations, relations,
        #     active_variable_names, block_core_names,
        #     derived_provider_by_output, derived_variable_names,
        #     blocked_relation_reasons, relations_by_name,
        #     relations_by_function, variable_scales/tolerances,
        #     _partition_diagnostics, plus the None-reset caches below;
        #   * seeding oracle -- _initial_guesses, _unevaluable_names (compile()),
        #   * completion plan -- _profile_specs, _constant_defaults_solver,
        #     _provider_plan, _completion_passes (every _run_compile_pass);
        #   * packed layout -- packed_specs, packed_dim, _packed_base_values,
        #     _uninitialized_free_variables (pack()).
        self._initial_guesses: dict[str, Any] = {}
        self._uninitialized_free_variables: list[str] = []
        # The canonical bipartite structural graph over the (immutable)
        # candidate relations; built once on first use.
        self._graph: nx.DiGraph | None = None
        # Caches derived from the compiled provider/active-relation plan;
        # reset to None by every _run_compile_pass.
        self._completion_plan_cache: list[tuple[Relation, bool]] | None = None
        self._sparsity_graph_cache: nx.DiGraph | None = None
        self._compiler_report_cache: dict[str, Any] | None = None
        self._completion_acyclic = False
        self._run_compile_pass()


    def _canonicalize_candidates(self, relations: Iterable[Relation], constraints: Any) -> None:
        """Canonicalize/validate the candidate relations and build the system
        constraint relations.

        ``canonicalize_relation`` rejects alias-degenerate relations (a declared
        output that resolves to one of its own inputs); registry relations are
        already validated at build time, but re-validating here keeps ad-hoc
        relations equally safe.
        """
        self.candidate_primary_relations = [canonicalize_relation(rel, self.variable_registry) for rel in relations]
        self.system_constraint_relations = [
            self._resolve_relation_names(
                constraint_from_expression(text, name=f"system_constraint_{index}", enforce=enforce, source_kind="system", source_name=self.name)
            )
            for index, (text, enforce) in enumerate(parse_constraint_specs(constraints))
        ]

    def _infer_profile_size(self) -> None:
        """Infer the single profile grid size shared by every 1-D variable;
        incompatible explicit sizes are an error."""
        profile_sizes: set[int] = set()
        for var in self.variables_by_name.values():
            if var.shape != 1:
                continue
            if var.size is not None:
                profile_sizes.add(int(var.size))
            elif isinstance(var.input_value, np.ndarray) and var.input_value.ndim == 1:
                profile_sizes.add(int(var.input_value.shape[0]))
        if len(profile_sizes) > 1:
            raise ValueError(f"Profile sizes are incompatible: {sorted(profile_sizes)}.")
        self.profile_size = next(iter(profile_sizes), self.variable_registry.profile_size_default)

    def _materialize_rho_grid(self) -> None:
        """Materialize the canonical fixed ``rho`` grid whenever profiles or
        rho-dependent relations are present."""
        if "rho" in self.variable_registry:
            uses_rho = any("rho" in rel.variables for rel in self.candidate_primary_relations)
            has_profile = any(var.shape == 1 for var in self.variables_by_name.values())
            if uses_rho or has_profile:
                rho_value = self.variable_registry.uniform_profile_grid(self.profile_size)
                if "rho" not in self.variables_by_name:
                    rho = Variable("rho", value=rho_value, size=self.profile_size, fixed=True)
                    self.variables_by_name["rho"] = rho
                else:
                    rho = self.variables_by_name["rho"]
                    if rho.input_value is None:
                        rho.size = self.profile_size
                        rho.set_input(rho_value)
                    rho.fixed = True

    def _broadcast_profile_values(self) -> None:
        """Broadcast scalar profile data onto the shared grid and validate
        explicitly supplied profile lengths."""
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

    def _split_supplied_profiles(self) -> None:
        """Split every supplied profile into a fixed shape plus a scalar
        average, kept linked by construction so ``mean(profile) == average``
        always holds (they can never diverge):

        * ``fixed`` profile -> shape AND level pinned.  The profile is
          authoritative, so its own volume-average defines the (also fixed)
          scalar average; a separately supplied scalar that disagrees is a
          contradiction, so the profile average wins and we warn.
        * supplied profile -> shape pinned, level free.  The average is a
          free solver DOF (referenced to the supplied scalar when given, else
          the profile's own average) and the profile is reconstructed as
          ``average * shape``, so it slides up and down with the level.
        """
        # One record per supplied profile: ``name -> (unit shape, fixed)``.
        # ``fixed`` profiles pin shape AND level; unfixed ones are shape-locked
        # (level controlled by the packed scalar average).
        self.supplied_profiles: dict[str, tuple[np.ndarray, bool]] = {}
        self.profile_average_by_name: dict[str, str] = {}
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
            self.supplied_profiles[name] = (shape.astype(float, copy=False), bool(var.fixed))
            supplied_average = avg_var.input_value
            profile_average = self._public_value(avg_name, avg)
            # The profile <-> average link is enforced by the
            # ``<x>_avg == volume_average(<x>)`` consistency relation, not by
            # pinning here: a fixed profile's average is seeded from the profile
            # (so it stays known/decidable) but left free, so a separately supplied
            # scalar that disagrees is surfaced and reconciled by the mode instead
            # of being silently overwritten.  A supplied average always wins the
            # seed; the residual then measures any disagreement with the profile.
            if supplied_average is None:
                avg_var.set_input(profile_average)

    # ── Public entry points and mode dispatch ────────────────────────────

    def verify(self, **options: Any) -> dict[str, Any]:
        return self.run("verify", **options)

    def reconcile(self, **options: Any) -> dict[str, Any]:
        return self.run("reconcile", **options)

    def optimize(self, **options: Any) -> dict[str, Any]:
        return self.run("optimize", **options)

    def ordered(self, **options: Any) -> dict[str, Any]:
        return self.run("ordered", **options)

    def relation_by_identifier(self, identifier: str) -> Relation | None:
        """Resolve a relation by user-facing name or decorated function name.

        Name is tried first to preserve existing semantics; the function-name
        index is the fallback.  Returns ``None`` when neither matches.
        """
        text = str(identifier)
        rel = self.relations_by_name.get(text)
        if rel is not None:
            return rel
        return self.relations_by_function.get(text)

    def compile(self) -> None:
        """Compile the active system, pruning relations needing unevaluable variables.

        Public entry to the compiled-execution interface; modes assume it has
        run.  ``run`` calls it before dispatch, so modes reached through ``run``
        need not call it; a caller invoking a mode function directly should call
        ``compile`` first.  It is not cheap (it re-runs the prune-to-fixpoint
        loop), so it is not called redundantly per mode.
        """
        self._compile_with_pruning()

    def run(self, mode: str = "verify", **options: Any) -> dict[str, Any]:
        """Prepare runtime initialization and dispatch to an isolated execution mode."""
        from .modes import get_mode

        self.compile()
        self.last_result = get_mode(mode)(self, **options)
        return self.last_result

    def _repr_html_(self) -> str:
        """Rich Jupyter table of this system's current variables."""
        from .reactor import variables_table

        return variables_table(self)

    # ── Compilation: active set, defaults activation, pruning ────────────

    def _run_compile_pass(self) -> None:
        """Run one full structural compile pass over the current candidates.

        Shared by the static compile in ``__init__`` and each round of
        :meth:`_compile_with_pruning`; it performs no pruning or
        initialization.  Stages, in order: filter usable candidates, activate
        registry defaults (bijection-closed forward decidability), select the
        active relation set (Dulmage-Mendelsohn pruning of undecidables),
        select block cores and derived providers, register profile
        generators, append guard relations, then refresh scales and freeze
        the completion plan.
        """
        # ── Clear caches derived from the previous provider/active-relation plan.
        self._completion_plan_cache = None
        self._sparsity_graph_cache = None
        self._compiler_report_cache = None

        supplied = {name for name, var in self.variables_by_name.items() if var.input_value is not None}
        inactive: dict[str, str] = {}
        usable = self._usable_candidates(inactive)
        pool, forward, seeded = self._activate_defaults(usable, supplied, inactive)
        active, decidable = self._select_active_relations(pool, forward, supplied, inactive)
        self._select_cores_and_providers(active, decidable, supplied, seeded)
        self._register_profile_generators()
        self._append_guard_relations()

        # ── Refresh variable scales and tolerances used by residuals.  Kept as a
        # method because reconcile also calls it post-solve to rescale around
        # stored values.
        self.refresh_scales()

        # ── Freeze the completion plan for this compiled structure: profile
        # reconstruction specs, held-constant defaults in solver form, and the
        # provider evaluation order.  complete() reads exactly these, so the
        # solve loop and the certification path share one completion.
        self._profile_specs = self._build_profile_specs()
        constant_defaults_solver: dict[str, Any] = {}
        for name, value in self.constant_default_values.items():
            try:
                constant_defaults_solver[name] = self._solver_value(name, value)
            except Exception:
                pass
        self._constant_defaults_solver = constant_defaults_solver
        self._provider_plan = self._completion_plan()
        # One completion pass is exact for an acyclic provider graph (the plan
        # is topologically ordered); only genuine provider cycles need iteration.
        self._completion_passes = 1 if self._completion_acyclic else 6


    def _usable_candidates(self, inactive: dict[str, str]) -> list[Relation]:
        """Return candidate relations usable this pass, ensuring their variables.

        A ``generated_profile`` relation whose outputs are all supplied and
        fixed is inactive: the supplied profile is authoritative.
        """
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
        return usable

    def _activate_defaults(self, usable: list[Relation], supplied: set[str], inactive: dict[str, str]) -> tuple[list[Relation], set[str], set[str]]:
        """Activate registry and relation defaults against forward decidability.

        Writes ``constant_default_values`` and ``default_provider_by_output``;
        marks never-activated defaults inactive.  Returns ``(pool, forward,
        seeded)``: the non-default + activated-default relation pool, the
        bijection-closed forward-decidable set, and the free-core default
        seeds.
        """
        non_default = [rel for rel in usable if not _is_default_relation(rel)]
        defaults = [rel for rel in usable if _is_default_relation(rel)]
        non_default_profile_outputs = {
            out for rel in non_default for out in rel.output_names
            if out in self.variable_registry and self.variable_registry.get(out).shape == 1
        }
        # A relation is an equation, not a one-way assignment: a two-variable
        # (bijective) relation determines either side from the other.  So forward
        # decidability is closed under inverting two-variable relations whose one
        # other side is known -- this lets ``kappa`` be derived from a supplied
        # ``kappa_95`` regardless of how ``Elongation 95%`` happens to be
        # written, keeping ``Default elongation`` a true last resort.  The
        # two-variable restriction is deliberate: it never reaches into
        # rank-deficient multi-variable cycles (the f_D/f_T/n_D/n_i fuel split,
        # advanced-fusion channels) that are structurally square but genuinely
        # rely on their default being pinned.
        def _forward_with_bijections(rels: list[Relation], extra_known: Iterable[str] = ()) -> set[str]:
            seed = set(supplied) | set(extra_known)
            decided = set(seed) | set(_forward_decision_rounds(rels, supplied, extra_known=seed)[0])
            changed_inv = True
            while changed_inv:
                changed_inv = False
                for rel in rels:
                    if len(rel.variables) != 2:
                        continue
                    unknown = [v for v in rel.variables if v not in decided]
                    if len(unknown) == 1:
                        decided.add(unknown[0])
                        changed_inv = True
            return decided

        base_forward = _forward_with_bijections(non_default)
        # Apply registry defaults to variables the user did not supply and that no
        # real relation already forward-decides.  A default is one of two kinds:
        #   * FREE core -- a default gated on a variable that is available
        #     (``default_requires``, e.g. the composition fractions gated on
        #     ``tau_p``).  The gate signals an active constraint (the steady-state
        #     balance) that will move the variable; the default is only the x0
        #     seed that breaks the n_X<->f_X cycle so the balance can activate.
        #   * DERIVED CONSTANT -- any other default.  Nothing can move it, so it
        #     is held at the default and never packed as a solver unknown (no
        #     extra free dimension, so unconstrained defaults cost nothing).
        # A supplied or forward-derivable value always wins over either.
        candidate_vars = {name for rel in usable for name in rel.variables}
        self.constant_default_values = {}
        seeded: set[str] = set()
        for name in sorted(candidate_vars):
            if name in supplied or name in base_forward or name not in self.variable_registry:
                continue
            spec = self.variable_registry.get(name)
            if spec.default is None or self.variables_by_name[name].fixed:
                continue
            gated_available = (
                spec.default_requires is not None
                and self.variable_registry.resolve(spec.default_requires) in (supplied | base_forward)
            )
            if gated_available:
                seeded.add(name)
            elif not isinstance(spec.default, str):
                self.constant_default_values[name] = float(spec.default)
        forward = _forward_with_bijections(non_default, extra_known=seeded | set(self.constant_default_values))
        # Profile defaults seed the forward closure, but only for a profile that
        # is neither supplied nor produced by a non-default relation.  Without
        # the ``supplied`` guard a reactor that loads a profile (e.g. a fixed CSV
        # T_i/T_e/n_e) would still activate the uniform-profile default, whose
        # enforced residual then fights the supplied profile and fails.
        active_defaults: list[Relation] = [
            rel for rel in sorted(defaults, key=lambda item: (len(item.variables), item.name))
            if any(
                out in self.variable_registry and self.variable_registry.get(out).shape == 1
                and out not in non_default_profile_outputs
                and out not in supplied
                for out in rel.output_names
            )
        ]
        known_defaults = seeded | set(self.constant_default_values)
        if active_defaults:
            forward = _forward_with_bijections(non_default + active_defaults, extra_known=known_defaults)
        changed = True
        while changed:
            changed = False
            for rel in sorted(defaults, key=lambda item: (len(item.variables), item.name)):
                if rel in active_defaults:
                    continue
                if any(out not in forward for out in rel.output_names) and all(inp in forward for inp in rel.input_names):
                    active_defaults.append(rel)
                    forward = _forward_with_bijections(non_default + active_defaults, extra_known=known_defaults)
                    changed = True
        # Only activated defaults are completion fallbacks.  A default whose
        # output a non-default relation can determine (forward or by a
        # two-variable inversion) is never registered as a provider, so it cannot
        # overwrite that derived value in completion (verify) or in reconcile.
        self.default_provider_by_output = {}
        for rel in sorted(active_defaults, key=lambda item: item.name):
            for out in rel.output_names:
                self.default_provider_by_output.setdefault(out, rel)
        pool = non_default + active_defaults
        for rel in defaults:
            if rel not in active_defaults and rel.name not in inactive:
                inactive.setdefault(rel.name, "inactive_default_not_needed")
        return pool, forward, seeded

    def _select_active_relations(self, pool: list[Relation], forward: set[str], supplied: set[str], inactive: dict[str, str]) -> tuple[list[Relation], set[str]]:
        """Partition unknowns by structural determinacy and select active relations.

        Deactivates relations touching undecidable (or previously unevaluable)
        variables, writes the active relation/variable sets, the structural
        blocks and the partition diagnostics.  Returns ``(active, decidable)``.
        """
        partition = self._structural_partition(pool, forward)
        block_decidable = set(partition["determined_variables"])
        decidable = supplied | forward | block_decidable
        undecidable = set(partition["underdetermined_variables"]) - decidable
        # Variables a previous prune round found unevaluable are treated as
        # undecidable so the relations that need them are deactivated.  Supplied
        # values are never unevaluable, so they are never pruned.
        unevaluable = self._unevaluable_names - supplied
        undecidable |= unevaluable
        self.structural_blocks = list(partition["blocks"])
        active: list[Relation] = []
        for rel in pool:
            undec = sorted(set(rel.variables) & undecidable)
            if undec:
                unev = sorted(set(rel.variables) & unevaluable)
                if unev:
                    inactive[rel.name] = "inactive_unevaluable: requires unevaluable " + ", ".join(unev)
                else:
                    inactive[rel.name] = "inactive_undecidable: cannot determine " + ", ".join(undec)
            else:
                active.append(rel)
        self.primary_relations = active
        self.relations = list(active)
        self.active_variable_names = {name for rel in active for name in rel.variables}
        for name in sorted(self.active_variable_names):
            self._ensure_variable_exists(name)
        self.blocked_relation_reasons = inactive
        # Structural-partition locals captured for the lazy ``compiler_report``
        # property -- the only place these diagnostics are exposed.
        self._partition_diagnostics = {
            "determined_missing_variables": tuple(sorted(partition["determined_variables"])),
            "undecidable_variables": tuple(sorted(undecidable)),
            "deficiencies": partition["deficiencies"],
        }

        return active, decidable

    def _select_cores_and_providers(self, active: list[Relation], decidable: set[str], supplied: set[str], seeded: set[str]) -> None:
        """Select packed block cores and the derived-variable providers."""
        produced = {out for rel in active for out in rel.output_names if not rel.implicit}
        self.block_core_names = {
            name for name in self.active_variable_names
            if name in decidable
            and name not in supplied
            and name not in produced
            and name not in self.constant_default_values
            and not self.variables_by_name[name].fixed
        } | {
            # Free-core defaults are packed unknowns seeded with their default,
            # never forward-derived, so an enforced relation (the balance) can
            # move them off the seed.  This wins over any relation that could
            # otherwise produce them (the redundant f_X = integral(n_X)/integral(n_i)),
            # which then acts as a closure residual instead of a producer.
            name for name in (seeded & self.active_variable_names)
            if not self.variables_by_name[name].fixed
            and self.variables_by_name[name].input_value is None
        }
        known_cores = self.block_core_names | set(self.constant_default_values)
        _, forward_decider = _forward_decision_rounds(active, supplied, extra_known=known_cores)
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
        # Constant defaults are held at their default value and never packed: they
        # are derived variables whose provider is the registry default itself.
        self.derived_variable_names |= {
            name for name in self.constant_default_values if name in self.active_variable_names
        }
        # Shape-locked supplied profiles are reconstructed from their scalar
        # average (``average * shape``): the profile is a derived variable (never
        # packed as a full-profile DOF), while its average is the packed level
        # control.  Registered after the derived set is built above so the reset
        # at the top of this phase does not drop them.
        for name, (_shape, fixed) in self.supplied_profiles.items():
            if fixed:
                continue
            avg_name = self.profile_average_by_name[name]
            self._ensure_variable_exists(avg_name)
            self.active_variable_names.add(name)
            self.active_variable_names.add(avg_name)
            self.derived_variable_names.add(name)
            self.derived_variable_names.discard(avg_name)
            self.block_core_names.discard(name)

    def _register_profile_generators(self) -> None:
        """Register explicit lower-dimensional profile generators as providers,
        activating their scalar-average controls."""
        for rel in list(self.relations):
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

    def _append_guard_relations(self) -> None:
        """Append active relation/variable/system guards whose variables are
        all active, and build the relation-name indexes."""
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
        all_relations = [*self.candidate_primary_relations, *self.system_constraint_relations, *self.relations]
        self.relations_by_name = {rel.name: rel for rel in all_relations}
        self.relations_by_function = {rel.function_name: rel for rel in all_relations}

    def _compile_with_pruning(self, max_rounds: int = 20) -> None:
        """Compile the active system, pruning relations that need unevaluable variables.

        Each round runs the full compile (active set, profile generators,
        guards, scales) and then the initialization oracle.  A variable that is
        active, non-fixed, unsupplied, not a forward-derived output and not a
        determined block core, and for which initialization can produce no
        value, cannot be evaluated; it is recorded and every relation that
        references it is deactivated on the next round.  This repeats to a
        fixpoint because removing a relation can orphan further variables.
        """
        self._unevaluable_names = set()
        self._initial_guesses = {}
        for _ in range(max_rounds):
            self._run_compile_pass()
            newly = self._detect_unevaluable_variables()
            if newly <= self._unevaluable_names:
                break
            self._unevaluable_names |= newly

    def _detect_unevaluable_variables(self) -> set[str]:
        """Return active free variables that no initialization path can value.

        Uses the same forward-propagation + anchored-block initialization the
        modes use as a read-only oracle (it does not write ``Variable.value``).
        A variable that ends up packed as a free solver unknown with no supplied
        value and no initial guess is unevaluable.
        """
        try:
            initial_values = self.initial_values_from_graph()
        except Exception:
            initial_values = {}
        self._initial_guesses = dict(initial_values)
        self.pack()
        return set(self._uninitialized_free_variables)

    # ── Structural graph views and compiler report ───────────────────────

    def _structural_graph(self) -> nx.DiGraph:
        """Return the one canonical annotated bipartite graph ``self._graph``.

        This is the single structural source for the whole system.  It has a
        node per variable and per candidate relation, with ``input -> relation``
        and ``relation -> output`` edges carrying each relation's declared
        direction.  Every other structural view is computed from it: the report
        incidence (:meth:`compiler_report`), the Dulmage-Mendelsohn partition
        (:meth:`_structural_partition`) and the Jacobian sparsity
        (:meth:`_sparsity_variable_names`).

        Nodes are annotated so consumers read state off the graph instead of
        re-deriving it:

        * variable nodes: ``kind='variable'``, ``shape``, ``supplied`` (an input
          value is present) and ``fixed``;
        * relation nodes: ``kind='relation'``, the ``relation`` object, its
          ordered ``variables`` tuple, ``enforce`` and ``is_default``.

        Returns:
            The cached :class:`networkx.DiGraph` held on ``self._graph``.
        """
        cached = self._graph
        if cached is not None:
            return cached
        graph = nx.DiGraph()
        for rel in self.candidate_primary_relations:
            # One relation node carrying everything a consumer needs about it.
            rnode = ("relation", rel.name)
            graph.add_node(
                rnode,
                kind="relation",
                relation=rel,
                variables=rel.variables,
                enforce=bool(rel.enforce),
                is_default=_is_default_relation(rel),
            )
            # Edges encode the declared direction: inputs feed the relation, the
            # relation feeds its outputs.
            for name in rel.input_names:
                self._add_variable_node(graph, name)
                graph.add_edge(("variable", name), rnode)
            for name in rel.outputs:
                self._add_variable_node(graph, name)
                graph.add_edge(rnode, ("variable", name))
        self._graph = graph
        return graph

    def _add_variable_node(self, graph: nx.DiGraph, name: str) -> None:
        """Add or annotate one variable node on the canonical graph.

        Args:
            graph: The canonical bipartite graph being built.
            name: Canonical variable name to add as a ``('variable', name)`` node.
        """
        var = self.variables_by_name.get(name)
        graph.add_node(
            ("variable", name),
            kind="variable",
            shape=self.variable_registry.get(name).shape if name in self.variable_registry else 0,
            supplied=bool(var is not None and var.input_value is not None),
            fixed=bool(var is not None and var.fixed),
        )

    @property
    def compiler_report(self) -> dict[str, Any]:
        """Diagnostic view of the compiled system, built on first access.

        Pure diagnostics: no execution path reads it.  It unifies what were two
        overlapping eager surfaces (the old ``compiler_report`` dict and the
        ``graph`` property) into one dict.  A run produces several result dicts
        (the main result, early returns, the trailing verify), so the dict is
        cached for the current compiled structure and the structural-graph walk
        runs once instead of per result; the cache is cleared by
        :meth:`_run_compile_pass` whenever the structure is recompiled.
        """
        cached = self._compiler_report_cache
        if cached is not None:
            return cached
        # Incidence views of the single structural graph, derived on the fly.
        relation_to_vars: dict[str, tuple[str, ...]] = {}
        var_to_rel_names: dict[str, list[str]] = {}
        for node, data in self._structural_graph().nodes(data=True):
            if data.get("kind") != "relation":
                continue
            relation_to_vars[node[1]] = data["variables"]
            for var in data["variables"]:
                var_to_rel_names.setdefault(var, []).append(node[1])
        var_to_relations = {name: tuple(rels) for name, rels in var_to_rel_names.items()}
        supplied = tuple(sorted(name for name, var in self.variables_by_name.items() if var.input_value is not None))
        report = {
            "activation_semantics": "decidability_closure",
            "variables": tuple(sorted(self.variables_by_name)),
            "supplied_variables": supplied,
            "active_variables": tuple(sorted(self.active_variable_names)),
            "derived_variables": tuple(sorted(self.derived_variable_names)),
            "relations": tuple(rel.name for rel in self.relations),
            "active_relations": tuple(rel.name for rel in self.primary_relations),
            "enforced_relations": tuple(rel.name for rel in self.relations if rel.enforce),
            "inactive_relations": dict(sorted(self.blocked_relation_reasons.items())),
            "default_provider_outputs": {name: rel.name for name, rel in sorted(self.default_provider_by_output.items())},
            "derived_provider_by_output": {name: rel.name for name, rel in self.derived_provider_by_output.items()},
            "relation_to_vars": relation_to_vars,
            "var_to_relations": var_to_relations,
            "profile_average_by_name": dict(sorted(self.profile_average_by_name.items())),
            "profile_source_by_name": {
                name: "fixed_supplied_profile" if self.supplied_profiles[name][1] else "supplied_profile"
                for name in sorted(self.supplied_profiles)
            },
            "unevaluable_variables": tuple(sorted(self._unevaluable_names)),
            "structural_determinacy": {
                **self._partition_diagnostics,
                "blocks": tuple(self.structural_blocks),
            },
        }
        self._compiler_report_cache = report
        return report

    # ── Profile/average split helpers ─────────────────────────────────────

    def _profile_average_name(self, name: str) -> str | None:
        """Return the scalar-average variable controlling a profile, or None.

        Thin delegate to :meth:`VariableRegistry.average_of`, which owns the
        profile -> average mapping (``average_variable`` metadata plus the
        ``<name>_avg`` alias convention).
        """
        return self.variable_registry.average_of(name)

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

    def _relation_is_residual_relation(self, rel: Relation) -> bool:
        """Return whether a relation belongs in the nonlinear residual core.

        Explicit-output relations whose outputs are uniquely owned non-fixed
        derived variables are evaluated structurally by :meth:`complete`
        on every candidate value map.  They are still verified during final
        certification, but they are not soft least-squares residuals.

        Relations remain residual relations when they are outputless, implicit,
        have fixed/non-derived outputs, have ambiguous producers, or are guard /
        constraint relations.
        """
        if rel.implicit or not rel.output_names:
            return True
        providers = self.derived_provider_by_output
        if not providers:
            return True
        # A relation may have multiple outputs. It is structural only if every
        # declared output is owned by this same provider.  Partial ownership must
        # stay in the residual core so final values cannot silently ignore one
        # side of the equation.
        return not all(providers.get(out) is rel for out in rel.output_names)

    # ── Structural determinacy (Dulmage-Mendelsohn) ──────────────────────

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
        graph = self._structural_graph()

        def rel_vars(rel: Relation) -> tuple[str, ...]:
            rnode = ("relation", rel.name)
            return graph.nodes[rnode]["variables"] if graph.has_node(rnode) else rel.variables

        unknowns = sorted({name for rel in relations for name in rel_vars(rel) if name not in known})
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

        # Variables are single-value graph nodes: one column each.  A profile's
        # grid dimension is internal to relation evaluation, never expanded into
        # the graph, so DM treats every variable as one structural unknown.
        col_span: dict[str, tuple[int, int]] = {name: (index, index + 1) for index, name in enumerate(unknowns)}
        n_cols = len(unknowns)

        # One row per scalar equation; inequalities determine nothing.  Relations
        # are adirectional, so outputs only count constraints here: one per
        # declared output, or one for an outputless equality.
        row_adj: list[list[int]] = []
        row_relation: list[str] = []
        for rel in relations:
            if not rel.outputs and rel.op != "==":
                continue
            cols = [c for name in rel_vars(rel) if name in col_span for c in range(*col_span[name])]
            if not cols:
                continue
            scalar_rows = sum(1 for name in rel.output_names if name in self.variable_registry) if rel.output_names else 1
            for _ in range(max(1, scalar_rows)):
                row_adj.append(cols)
                row_relation.append(rel.name)

        # Maximum bipartite matching between scalar-equation rows and unknown
        # columns, computed with networkx (Hopcroft-Karp) on a small bipartite
        # graph built from ``row_adj``.  The Dulmage-Mendelsohn coarse partition
        # (over/well/under-determined) and fine blocks are invariant to *which*
        # maximum matching is chosen, so this yields the same partition as the
        # former scipy.sparse matching while keeping the whole decomposition in
        # networkx.
        match_col = np.full(max(len(row_adj), 1), -1, dtype=int)
        match_row = np.full(n_cols, -1, dtype=int)
        if row_adj:
            bipartite = nx.Graph()
            row_nodes = [("row", r) for r in range(len(row_adj))]
            bipartite.add_nodes_from(row_nodes)
            bipartite.add_nodes_from(("col", c) for c in range(n_cols))
            for r, cols in enumerate(row_adj):
                for c in cols:
                    bipartite.add_edge(("row", r), ("col", c))
            # ``maximum_matching`` returns both directions; read the row side and
            # mirror it into the per-column ``match_row``.
            matching = nx.bipartite.maximum_matching(bipartite, top_nodes=row_nodes)
            for r in range(len(row_adj)):
                partner = matching.get(("row", r))
                if partner is not None:
                    c = int(partner[1])
                    match_col[r] = c
                    match_row[c] = r

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

        # A column variable that no relation in this pool produces and that
        # appears only in outputless (constraint) relations is a free parameter
        # the constraints cannot pin (e.g. an unsupplied ``tau_p`` referenced
        # only by the particle balances).  Force it underdetermined so those
        # constraints deactivate, instead of letting the matching invent a value
        # for it and activate a meaningless balance.
        produced_names = {out for rel in relations for out in rel.output_names}
        for name, (start, stop) in col_span.items():
            if name in produced_names:
                continue
            refs = [rel for rel in relations if name in rel_vars(rel)]
            if refs and all((not rel.outputs and rel.op == "==") for rel in refs):
                under_cols.update(range(start, stop))

        name_of_col = {c: name for name, (start, stop) in col_span.items() for c in range(start, stop)}
        under_names = {name for name, (start, stop) in col_span.items() if any(c in under_cols for c in range(start, stop))}
        result["determined_variables"] = set(unknowns) - under_names
        result["underdetermined_variables"] = under_names
        result["blocks"] = _structural_block_plan(row_adj, match_row, under_cols, name_of_col)

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

    # ── Seeding: initial values from the graph (the x0 oracle) ───────────

    def initial_values_from_graph(self) -> dict[str, Any]:
        """Fill solver start values by direct propagation from supplied values.

        Iteratively solves every relation that has exactly one missing variable
        (the 1x1 / acausal step), to a fixed point.  These are exact values, not
        movement references.  Variables that remain missing are the free unknowns
        of larger coupled blocks (block cores); they are packed directly and
        determined by the simultaneous reconcile against their block's supplied
        anchor, so no separate block solver is needed here.

        Returns:
            ``{name: value}`` for every variable the oracle seeded (supplied
            values are never included).
        """
        values = self.input_values()
        self._apply_profile_specs(values)
        original = set(values)
        seeded: set[str] = set()
        # Constant defaults are known values from the start (they are held, not
        # solved), so downstream propagation can use them.
        for name, value in self.constant_default_values.items():
            if values.get(name) is None:
                try:
                    values[name] = self._solver_value(name, value)
                    seeded.add(name)
                except Exception:
                    pass
        # Propagate everything derivable from the supplied values.
        self._propagate_known(values, seeded, original)
        # Seed registry defaults for variables that supplied-propagation left
        # missing, then re-propagate so downstream values (n_X = n_i * f_X, ...)
        # fill in.  Defaults are pure x0 seeds -- never enforced -- applied to a
        # fixpoint so variable-reference defaults (T_i = T_e) resolve once their
        # source has a value.
        for _ in range(50):
            if not self._seed_defaults(values, seeded, original):
                break
            self._propagate_known(values, seeded, original)
        return {name: values[name] for name in values if name in seeded}

    def _propagate_known(self, values: dict[str, Any], seeded: set[str], original: set[str]) -> None:
        """Fill values derivable from the currently known namespace.

        Stage 1 runs direct 1x1/acausal propagation to a fixed point; stage 2
        solves the determined blocks (2x2 ... N x N) for their cores, with a
        final merged-block sweep for variables left in no individual block.
        """
        # Stage 1: direct 1x1/acausal propagation to a fixed point.
        for _direct_pass in range(50):
            if not self._compute_direct_outputs(values, seeded, original):
                break
        # Stage 2: solve the determined blocks (2x2 ... N x N) for their cores.
        progress = True
        while progress:
            progress = False
            for block in self.structural_blocks:
                if self._compute_planned_block(block, values, seeded, original):
                    progress = True
                    for _direct_pass in range(50):
                        if not self._compute_direct_outputs(values, seeded, original):
                            break
        merged = tuple(
            name
            for block in self.structural_blocks
            for name in block
            if (name not in values or values[name] is None) and name not in original
        )
        if merged and self._compute_planned_block(merged, values, seeded, original):
            for _direct_pass in range(50):
                if not self._compute_direct_outputs(values, seeded, original):
                    break

    def _seed_defaults(self, values: dict[str, Any], seeded: set[str], original: set[str]) -> bool:
        """Seed still-missing active variables from their registry default.

        A default is either a number (a constant x0 seed) or the name of another
        variable (copy that variable's current value).  Seeds are pure initial
        points: a variable a relation determines is moved off its seed by the
        global solve, and a variable no enforced relation touches keeps it
        (zero-gradient).  A default is applied only when the variable is active,
        not supplied/fixed and still missing; variable-reference defaults whose
        source is not yet known are skipped (the caller iterates to a fixpoint).
        """
        progress = False
        for name in sorted(self.active_variable_names):
            if name in values and values[name] is not None:
                continue
            if name in original or name not in self.variable_registry:
                continue
            var = self.variables_by_name.get(name)
            if var is not None and (var.fixed or var.input_value is not None):
                continue
            spec = self.variable_registry.get(name)
            default = spec.default
            if default is None:
                continue
            if spec.default_requires is not None:
                required = self.variable_registry.resolve(spec.default_requires)
                if values.get(required) is None:
                    continue
            if isinstance(default, str):
                if default not in self.variable_registry:
                    continue
                source = self.variable_registry.get(default).canonical_name
                if source not in values or values[source] is None:
                    continue
                raw: Any = values[source]
            else:
                raw = float(default)
            try:
                value = self._solver_value(name, raw)
                if not self._candidate_value_is_valid(name, value):
                    continue
            except Exception:
                continue
            values[name] = value
            seeded.add(name)
            progress = True
        return progress


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
            if _is_default_relation(rel):
                by_name[rel.name] = rel
        return list(by_name.values())

    def _seed_accepts(self, name: str, original: set[str]) -> bool:
        """Return whether seeding may write a value for one variable.

        Seeding only fills genuinely missing degrees of freedom: it never
        overrides a user-supplied value (``original``) or a fixed variable, and
        it ignores names the registry does not know.

        Args:
            name: Candidate variable name to write.
            original: Names that already had a value before seeding began.

        Returns:
            ``True`` when ``name`` may be written by seeding.
        """
        if name not in self.variable_registry or name in original:
            return False
        var = self.variables_by_name.get(name)
        return not (var is not None and var.fixed)

    def _compute_direct_outputs(self, values: dict[str, Any], seeded: set[str], original: set[str]) -> bool:
        """Seed values by solving every relation that has exactly one unknown.

        Seeding is *adirectional*: a relation is an equation, so whenever all but
        one of its variables are known the remaining one is obtained by inverting
        the canonical relation through :meth:`Relation.solve`, regardless of
        whether that variable is a declared input or a declared output.  This is
        the single-unknown step the seeding fixed point is built on; the caller
        repeats it until nothing more can be solved, then hands the remaining
        coupled variables to the block solver.

        A relation whose inputs are all known but which still has several unknown
        outputs is additionally evaluated forward to fill those outputs together
        (each is an independent forward computation, not a directional
        assumption).  Implicit relations are skipped: a variable appearing on
        both sides cannot be trusted to a blind inverse.  Only missing, non-fixed,
        non-supplied variables are written, and every value must be finite and
        in-domain, so ``seeded`` stays a pure record of seeded names.

        Args:
            values: Working solver-unit namespace, mutated in place.
            seeded: Names seeded so far, mutated in place.
            original: Names supplied before seeding (never overwritten).

        Returns:
            ``True`` if any value was filled this pass.
        """
        progress = False
        for rel in self._initial_direct_relation_pool():
            # Primary path: a relation with exactly one unknown variable is
            # solved in whatever direction closes it (input or output).
            if not rel.implicit:
                unknown = [name for name in rel.variables if self._value_missing(values, name)]
                if len(unknown) == 1 and self._seed_accepts(unknown[0], original):
                    name = unknown[0]
                    known = {vname: values[vname] for vname in rel.variables if vname != name}
                    try:
                        value = self._solver_value(name, rel(**known))
                        if self._candidate_value_is_valid(name, value):
                            values[name] = value
                            seeded.add(name)
                            progress = True
                            continue
                    except Exception:
                        pass

            # Secondary path: every input is known, so any still-missing outputs
            # are each computable forward in one evaluation.
            if rel.output_names and all(not self._value_missing(values, inp) for inp in rel.input_names):
                try:
                    mapped = rel.output_map(rel.evaluate(self._relation_evaluation_values(rel, values)))
                except Exception:
                    mapped = {}
                for name in rel.output_names:
                    if name not in mapped or not self._value_missing(values, name) or not self._seed_accepts(name, original):
                        continue
                    try:
                        value = self._solver_value(name, mapped[name])
                        if not self._candidate_value_is_valid(name, value):
                            continue
                    except Exception:
                        continue
                    values[name] = value
                    seeded.add(name)
                    progress = True
        return progress

    def _compute_planned_block(
        self,
        block: tuple[str, ...],
        values: dict[str, Any],
        seeded: set[str],
        original: set[str],
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
        solved = self._solve_initial_block(extended, rels, values, residual_tol=1.0)
        if solved is None:
            return False
        for name, value in solved.items():
            values[name] = value
            seeded.add(name)
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
        """Solve one small scalar initial-computation block; returns ``{name: value}`` or None.

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
            return self.complete(ns)

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
            lb, ub = scipy_bounds(self.variable_registry.get(name).solver_domain, zero_tol=ZERO_TOL)
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
                for point in _scalar_start_grid(lb, ub):
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
            grid = _scalar_start_grid(lb, ub)
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
            scale, offset, lo, hi, transform = self._pack_scalar(var, init, lb, ub, scale_ref=init)
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
        return solved

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
        ordered_rels = sorted(rels, key=lambda rel: not _is_default_relation(rel))
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

    def _candidate_value_is_valid(self, name: str, value: Any) -> bool:
        """Whether a candidate value is finite and in-domain (see :meth:`Variable.candidate_valid`)."""
        try:
            return self._variable_for(name).candidate_valid(value)
        except Exception:
            return False

    # ── Packing and the per-solve runtime plan ────────────────────────────

    def _use_log_transform(self, var: Variable, init: float, lower: float) -> bool:
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
        self, var: Variable, init: float, lb: float, ub: float, *, scale_ref: Any, allow_log: bool = True
    ) -> tuple[float, float, float, float, str]:
        """Map one scalar to a solver coordinate ``(scale, offset, lower, upper, transform)``.

        Positive-bounded scalars pack logarithmically; others linearly with a
        tolerance/reference ``scale``.  ``allow_log=False`` forces linear packing.
        """
        scale = var.scale(scale_ref)
        if allow_log and self._use_log_transform(var, init, lb):
            lower = np.log(lb / init) if np.isfinite(lb) and lb > 0.0 else -np.inf
            upper = np.log(ub / init) if np.isfinite(ub) and ub > 0.0 else np.inf
            return scale, init, lower, upper, "log"
        lower = (lb - init) / scale if np.isfinite(lb) else -np.inf
        upper = (ub - init) / scale if np.isfinite(ub) else np.inf
        return scale, init, lower, upper, "linear"

    def pack(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pack active non-fixed variables and prepare the per-solve runtime plan.

        Builds the packed layout -- one record per free variable
        ``(name, start, stop, offsets, scales, shape, transform)`` stored on
        ``self.packed_specs`` -- plus the immutable base value map that
        :meth:`unpack` layers each solver vector onto.  A reconcile/optimize
        solve calls the residual function thousands of times, so neither may
        be re-derived per call; completion itself reads the compile-cached
        plan (see :meth:`complete`).

        Returns:
            ``(x0, lower, upper)``.  ``x0`` is all zeros: the per-element
            ``offsets``/``scales`` (and the log transform for positive-bounded
            scalars) absorb the actual start values.
        """
        lower: list[float] = []
        upper: list[float] = []
        specs: list[tuple[str, int, int, np.ndarray, np.ndarray, int, str | None]] = []
        self._uninitialized_free_variables = []
        for name in sorted(self.active_variable_names):
            var = self.variables_by_name[name]
            if name in self.derived_variable_names:
                continue
            if var.fixed:
                if var.input_value is None:
                    raise ValueError(f"Fixed variable {name!r} has no value.")
                continue
            spec = self.variable_registry.get(name)
            lb, ub = scipy_bounds(spec.solver_domain, zero_tol=ZERO_TOL)
            size = self._variable_dim(name)
            start = len(lower)
            offsets: list[float] = []
            scales: list[float] = []
            span_transform: str | None = None
            try:
                initial_elements = [
                    float(self._initial_value(var, index=i if var.shape == 1 else None))
                    for i in range(size)
                ]
            except Exception:
                self._uninitialized_free_variables.append(name)
                continue
            for i, init in enumerate(initial_elements):
                ref = var.movement_reference(init, index=i if var.shape == 1 else None)
                scale, offset, lo, hi, transform = self._pack_scalar(var, init, lb, ub, scale_ref=ref)
                lower.append(lo)
                upper.append(hi)
                offsets.append(offset)
                scales.append(scale)
                if transform == "log":
                    span_transform = "log"
            specs.append((name, start, len(lower), np.asarray(offsets, dtype=float), np.asarray(scales, dtype=float), var.shape, span_transform))
        self.packed_specs = specs
        self.packed_dim = len(lower)
        # Immutable input values are the base every solver vector is layered
        # onto; completion itself reads the compile-cached plan.
        self._packed_base_values = self.input_values()
        return np.zeros(self.packed_dim), np.asarray(lower), np.asarray(upper)

    def _required_uninitialized_free_variables(self) -> list[str]:
        """Return uninitialized free variables required by enforced relations."""
        uninitialized = set(self._uninitialized_free_variables)
        if not uninitialized:
            return []
        required: list[str] = []
        for name in sorted(uninitialized):
            for rel in self.relations:
                if rel.enforce and name in rel.variables:
                    required.append(name)
                    break
        return required

    @staticmethod
    def _value_missing(values: Mapping[str, Any], name: str) -> bool:
        """Return whether ``name`` is absent or ``None`` in ``values``."""
        return name not in values or values[name] is None

    def unpack(self, x: np.ndarray) -> dict[str, Any]:
        """Rebuild a completed solver namespace from one packed solver vector.

        Args:
            x: Packed free-variable vector matching the layout built by the
                last :meth:`pack` call.

        Returns:
            A solver-unit namespace with packed variables unpacked and all
            derived/profile/default variables completed.
        """
        # Start from the immutable inputs, then overwrite the packed free vars.
        values = dict(self._packed_base_values)
        arr = np.asarray(x, dtype=float)
        for name, start, stop, offsets, scales, shape, transform in self.packed_specs:
            local_x = arr[start:stop]
            # Positive-bounded scalars are packed logarithmically; the rest affine.
            if transform == "log":
                actual = offsets * np.exp(local_x)
            else:
                actual = offsets + scales * local_x
            values[name] = actual.copy() if shape == 1 else float(actual[0])
        return self.complete(values)

    # ── Completion: providers, profiles, value namespaces ─────────────────

    def _apply_completion_providers(self, out: dict[str, Any]) -> None:
        """Evaluate completion providers in dependency order, in place.

        The provider stage of :meth:`complete`.  Each provider relation of the
        compiled plan whose inputs are all present is evaluated and writes the
        registry-known outputs it owns: an explicit provider recomputes its
        output, a default (``only_missing``) fills only a still-missing one.
        One pass is exact for an acyclic plan; a cyclic plan iterates until a
        pass changes nothing (value equality test) or the pass cap is reached.
        """
        active_vars = self.active_variable_names
        explicit_outputs = self.derived_provider_by_output
        for _pass in range(self._completion_passes):
            changed = False
            for rel, only_missing in self._provider_plan:
                # A provider can only fire once all of its inputs are known.
                if any(self._value_missing(out, inp) for inp in rel.input_names):
                    continue
                try:
                    # ``out`` is already a solver-form namespace here (this runs
                    # only from the solve-time and certification completion paths,
                    # both of which build solver-form values), so the relation is
                    # evaluated directly without the per-relation namespace copy.
                    mapped = rel.output_map(rel.evaluate(out))
                except Exception:
                    continue
                for out_name, out_value in mapped.items():
                    if out_name not in self.variable_registry:
                        continue
                    # Only write outputs this system owns (active or explicit).
                    if out_name not in active_vars and out_name not in explicit_outputs:
                        continue
                    old_missing = self._value_missing(out, out_name)
                    # Defaults fill only a missing output; explicit providers recompute.
                    if only_missing and not old_missing:
                        continue
                    try:
                        value = self._solver_value(out_name, out_value)
                    except Exception:
                        continue
                    old_value = out.get(out_name)
                    out[out_name] = value
                    # Progress = a newly filled value, or (for a cycle) one that
                    # actually moved; this is what lets the loop stop early.
                    if old_missing:
                        changed = True
                    else:
                        try:
                            old_arr = np.asarray(old_value, dtype=float)
                            new_arr = np.asarray(value, dtype=float)
                            if old_arr.shape != new_arr.shape or not np.allclose(old_arr, new_arr, rtol=0.0, atol=1.0e-300):
                                changed = True
                        except Exception:
                            if old_value != value:
                                changed = True
            if not changed:
                break

    def _build_profile_specs(self) -> list[tuple[str, str | None, np.ndarray | None, Any]]:
        """Return profile reconstruction specs ``(name, avg_name, shape, fixed_value)``.

        A fixed supplied profile carries its stored solver-form array as
        ``fixed_value``; a shape-controlled profile carries its scalar-average
        name and unit shape for ``average * shape`` reconstruction.  Built once
        per compile pass and cached on ``self._profile_specs``.
        """
        specs: list[tuple[str, str | None, np.ndarray | None, Any]] = []
        for name, (shape, fixed) in self.supplied_profiles.items():
            fixed_value = None
            if fixed:
                var = self.variables_by_name.get(name)
                if var is not None and var.input_value is not None:
                    fixed_value = self._solver_value(name, var.input_value)
            specs.append((name, self.profile_average_by_name.get(name), shape, fixed_value))
        return specs

    def _apply_profile_specs(self, values: dict[str, Any]) -> None:
        """Reconstruct fixed/shape-controlled profiles in place.

        The profile stage of :meth:`complete`, reading the compile-cached
        ``self._profile_specs``.
        """
        for name, avg_name, shape, fixed_value in self._profile_specs:
            if fixed_value is not None:
                values[name] = fixed_value
                continue
            if avg_name is None or self._value_missing(values, avg_name):
                continue
            avg = float(np.asarray(values[avg_name], dtype=float).reshape(-1)[0])
            values[name] = self._solver_value(name, avg * shape)

    def _apply_constant_defaults(self, values: dict[str, Any]) -> None:
        """Fill still-missing held-constant defaults in place (solver form).

        The constant-default stage of :meth:`complete`, reading the
        compile-cached ``self._constant_defaults_solver``.
        """
        for name, value in self._constant_defaults_solver.items():
            if self._value_missing(values, name):
                values[name] = value

    def _value_map(self, *, use_input: bool, solver_form: bool) -> dict[str, Any]:
        """Build a value map from variable state; missing variables are omitted."""
        values: dict[str, Any] = {}
        for name, var in self.variables_by_name.items():
            value = var.input_value if use_input else var.value
            if value is None:
                continue
            values[name] = var.solver_value(value) if solver_form else value
        return values

    def solver_values(self) -> dict[str, Any]:
        """Current values in solver form, uncompleted (pass to :meth:`complete` to close)."""
        return self._value_map(use_input=False, solver_form=True)

    def input_values(self) -> dict[str, Any]:
        """Immutable supplied inputs in solver form -- the movement/reference base."""
        return self._value_map(use_input=True, solver_form=True)

    def public_values(self) -> dict[str, Any]:
        """Current values in public form."""
        return self._value_map(use_input=False, solver_form=False)

    def complete(self, values: dict[str, Any]) -> dict[str, Any]:
        """Complete a solver namespace in place; the single completion path.

        Completion is the only place where missing variables may be filled.
        Three stages, in order: shape-controlled profiles are reconstructed
        from their scalar averages, held-constant defaults fill still-missing
        variables, then provider relations run in dependency order (explicit
        providers recompute their outputs; defaults fill only missing ones).
        No variable-name heuristic is used.  All stage inputs are compiled
        products cached by :meth:`_run_compile_pass`.

        Args:
            values: Solver-form namespace; mutated in place.

        Returns:
            The same ``values`` mapping, for convenience.
        """
        self._apply_profile_specs(values)
        self._apply_constant_defaults(values)
        self._apply_completion_providers(values)
        return values

    def _completion_plan(self) -> list[tuple[Relation, bool]]:
        """Return derived/default providers in dependency order, computed once.

        Completion runs each provider relation and writes *every* output the
        relation returns (its declared outputs), not just the one variable it was
        selected to provide.  The execution order must therefore be a topological
        sort over relations where ``R`` follows ``S`` whenever any input of ``R``
        is any output of ``S`` -- the per-variable provider graph alone misses
        these side-output dependencies, which is why completion previously needed
        a redundant second pass to settle (e.g. ``Charged fusion power`` reading
        ``P_fus_DT_alpha`` produced as a secondary output of a later relation).

        With this fuller ordering one pass is exact for an acyclic dependency
        graph; ``_completion_acyclic`` records that so completion skips the second
        pass.  A genuine provider cycle (a quasineutrality ``n_e<->n_i`` pair)
        becomes a multi-node component and still iterates to a fixpoint.  Explicit
        providers recompute their output (``only_missing`` False); defaults only
        fill a still-missing output.
        """
        cached = self._completion_plan_cache
        if cached is not None:
            return cached
        # One provider per variable; explicit ownership wins over a default.
        # Unique provider relations owning at least one output after that
        # resolution.  A relation is a fill-only default when no output of it
        # is owned by an explicit derived provider.
        provider_of: dict[str, Relation] = dict(self.default_provider_by_output)
        provider_of.update(self.derived_provider_by_output)
        explicit_providers = {rel.name for rel in self.derived_provider_by_output.values()}
        rels: dict[str, Relation] = {}
        for rel in provider_of.values():
            rels.setdefault(rel.name, rel)
        only_missing_by_name = {name: name not in explicit_providers for name in rels}

        # Relation dependency DAG: producer -> consumer over all declared outputs.
        out_to_rels: dict[str, list[str]] = {}
        for rel in rels.values():
            for out in rel.output_names:
                out_to_rels.setdefault(out, []).append(rel.name)
        dag = nx.DiGraph()
        dag.add_nodes_from(rels)
        for rel in rels.values():
            # Constants that are themselves produced variables (e.g. the per-
            # channel ``P_fus_*`` powers read as constants by ``Charged fusion
            # power``) are real ordering dependencies: the relation reads their
            # current value, falling back to the registry default only when no
            # producer ran.  They must order like inputs.
            for inp in (*rel.input_names, *rel.constant_names):
                for producer in out_to_rels.get(inp, ()):
                    if producer != rel.name:
                        dag.add_edge(producer, rel.name)
        condensation = nx.condensation(dag)
        self._completion_acyclic = all(
            len(condensation.nodes[comp]["members"]) == 1 for comp in condensation
        )
        ordered: list[tuple[Relation, bool]] = []
        for comp in nx.lexicographical_topological_sort(condensation, key=lambda c: min(condensation.nodes[c]["members"])):
            for rel_name in sorted(condensation.nodes[comp]["members"]):
                ordered.append((rels[rel_name], only_missing_by_name[rel_name]))
        self._completion_plan_cache = ordered
        return ordered

    # ── Residual blocks: relations, domains, movement ─────────────────────

    def solver_residual_vector(self, values: Mapping[str, Any]) -> tuple[np.ndarray, list[str]]:
        """Return the enforced-relation residual vector for one solver iterate.

        The hot path: called once per least-squares residual evaluation, so it
        computes numbers only (no ``verify_status`` dictionaries -- final
        success is decided by :meth:`certify_relations` on the returned
        candidate).  Structural provider relations contribute no rows: their
        outputs were recomputed by completion.  A missing or broken enforced
        relation contributes one large finite residual so the solver steps away
        from it instead of aborting.

        Returns:
            ``(residuals, errors)`` -- the stacked enforced residual rows and
            the missing-variable error messages.
        """
        blocks: list[np.ndarray] = []
        errors: list[str] = []
        for rel in self.relations:
            if not self._relation_is_residual_relation(rel):
                continue
            missing = [name for name in rel.variables if name not in values or values[name] is None]
            if missing:
                if rel.enforce:
                    errors.append(f"Relation {rel.name!r} missing variables {missing}.")
                    blocks.append(np.asarray([1.0e12]))
                continue
            if rel.enforce:
                # ``values`` is already in canonical solver form on every path
                # that reaches a residual evaluation (base inputs, unpacked
                # spans and completion outputs are all coerced).
                blocks.append(self._residual_vector(rel, values, safe=True))
        residuals = np.concatenate([block.reshape(-1) for block in blocks if block.size]) if blocks else np.empty(0, dtype=float)
        if not np.all(np.isfinite(residuals)):
            residuals = np.nan_to_num(residuals, nan=1.0e12, posinf=1.0e12, neginf=-1.0e12)
        return residuals, errors

    def certify_relations(self, values: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], np.ndarray, list[str], list[str]]:
        """Evaluate every active relation for certification.

        The certification path: builds a full ``verify_status`` dictionary per
        relation (including structural providers, whose residuals stay out of
        the certificate residual vector) plus the stacked enforced residual
        rows.  Used by :func:`fusdb.modes.verify.verify_values` -- solver
        termination is never a success condition; this is.

        Returns:
            ``(status_by_relation, residuals, errors, warnings)``.
        """
        status: dict[str, dict[str, Any]] = {}
        blocks: list[np.ndarray] = []
        errors: list[str] = []
        warnings: list[str] = []
        for rel in self.relations:
            missing = [name for name in rel.variables if name not in values or values[name] is None]
            if not self._relation_is_residual_relation(rel):
                # Structural provider: its outputs were recomputed by
                # completion.  It contributes no residual row but is still
                # checked as an enforced relation on the completed value map.
                if missing:
                    message = f"Relation {rel.name!r} missing variables {missing}."
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
                try:
                    rel_status = self._verify_status(rel, values)
                    rel_status["source"] = "derived_provider"
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
            if missing:
                message = f"Relation {rel.name!r} missing variables {missing}."
                status[rel.name] = {"relation": rel.name, "verified": False, "missing": missing, "errors": [message], "warnings": [], "enforced": rel.enforce}
                continue
            # Status and residual share one evaluation.
            rel_status, residual = self._status_and_residual(rel, values)
            status[rel.name] = rel_status
            warnings.extend(rel_status.get("warnings", []))
            if rel.enforce:
                blocks.append(residual)
                if rel_status.get("errors"):
                    errors.extend(f"{rel.name}: {err}" for err in rel_status["errors"])
            elif not rel_status["verified"]:
                if _is_default_relation(rel):
                    warnings.append(f"{rel.name}: weak default not satisfied after reconciliation")
                else:
                    errors.append(f"{rel.name}: check-only applicability failed")
        residuals = np.concatenate([block.reshape(-1) for block in blocks if block.size]) if blocks else np.empty(0, dtype=float)
        return status, residuals, errors, warnings

    def _sparsity_dependency_graph(self) -> nx.DiGraph:
        """Directed completion-dependency graph used for Jacobian sparsity.

        A residual row reads its relation's variables off the *completed*
        namespace, so changing a packed variable can move that row only through
        completion.  This graph therefore has an edge ``input -> output`` for
        every output of every completion provider relation (plus the
        ``average -> profile`` edges completion uses), so a variable's ancestors
        are every packed variable that can reach it through completion.

        It is deliberately fuller than the completion plan's per-variable
        provider selection, which keeps a single selected provider per
        variable and so encodes only one ``input -> selected_output`` edge
        per relation.  That single-provider view
        misses dependencies that flow through a relation's *side* outputs (e.g.
        ARC_V0's ``n_i``/fusion-power coupling to ``f_He``), which previously had
        to be recovered by probing the residual once per packed variable.
        Capturing every output here makes the structural pattern conservative on
        its own, so no finite-difference probe is needed.
        """
        cached = self._sparsity_graph_cache
        if cached is not None:
            return cached
        graph = nx.DiGraph()
        for rel, _only_missing in self._completion_plan():
            inputs = [*rel.input_names, *(c for c in rel.constant_names if c in self.variable_registry)]
            for out in rel.output_names:
                for inp in inputs:
                    if inp != out:
                        graph.add_edge(inp, out)
        for profile, avg in self.profile_average_by_name.items():
            if avg != profile:
                graph.add_edge(avg, profile)
        self._sparsity_graph_cache = graph
        return graph

    def _sparsity_variable_names(self, name: str) -> set[str]:
        """Return variables that can affect one variable through completion.

        These are ``name`` plus its ancestors in
        :meth:`_sparsity_dependency_graph`.  Conservative over-inclusion is
        always safe for sparse differencing; a missed dependency is what would
        corrupt the Jacobian.
        """
        graph = self._sparsity_dependency_graph()
        if name not in graph:
            return {name}
        return {name} | nx.ancestors(graph, name)

    def build_jac_sparsity(self, reference: Mapping[str, Any] | None = None):
        """Build conservative residual-variable sparsity for SciPy coloring.

        Reads the packed layout stored by the last :meth:`pack` call.  The
        matrix rows must match the complete least-squares residual vector:
        enforced relation residuals, domain rows, then movement-penalty rows
        (present only when ``reference`` is given).
        """
        if not self.packed_specs:
            return None
        span_by_name = {name: (start, stop) for name, start, stop, *_rest in self.packed_specs}
        packed_names = set(span_by_name)
        values = self.unpack(np.zeros(self.packed_dim, dtype=float))

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
        for name in sorted(self.active_variable_names):
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
            # Profiles contribute one reduced domain row per bound (the extremal
            # violation); scalars one per bound.  Either way ``sides`` rows.
            if dim > 0:
                row_specs.append((sides, self._sparsity_variable_names(name)))

        if reference is not None:
            # The movement penalty emits one crossing residual per input variable
            # (see movement_residuals), so each contributes a single row.
            # A packed input's row depends only on its own span.
            for name in span_by_name:
                if name not in values or name not in reference or reference[name] is None:
                    continue
                row_specs.append((1, {name}))
            # A supplied variable derived from an explicit equation (e.g. a
            # profile fit) depends on the packed inputs that reach it.
            for name in sorted(self.derived_variable_names - packed_names):
                var = self.variables_by_name.get(name)
                if var is None or var.input_value is None or name not in values or values[name] is None:
                    continue
                row_specs.append((1, self._sparsity_variable_names(name)))

        total_rows = sum(rdim for rdim, _names in row_specs)
        if total_rows <= 0:
            return None
        matrix = lil_matrix((total_rows, self.packed_dim), dtype=bool)
        row = 0
        for rdim, names in row_specs:
            for var_name in names:
                if var_name in span_by_name:
                    start, stop = span_by_name[var_name]
                    matrix[row:row + rdim, start:stop] = True
            row += rdim
        return matrix.tocsr()

    def domain_residuals(self, values: Mapping[str, Any]) -> np.ndarray:
        """Return physical-domain violation residuals in tolerance units.

        Domains are hard feasibility constraints.  This residual is zero inside
        the physical domain and positive/negative outside it, normalized by the
        variable tolerance width.  It is used only by reconcile/optimize solver
        objectives; final success is still decided by ``_domain_errors`` and
        canonical relation verification.
        """
        rows: list[np.ndarray] = []
        for name in sorted(self.active_variable_names):
            if name not in values or values[name] is None or name not in self.variable_registry:
                continue
            rows.extend(self._variable_for(name).domain_violation_rows(values[name]))
        return np.concatenate(rows) if rows else np.empty(0, dtype=float)

    def _movement_inputs(self, values: Mapping[str, Any], reference: Mapping[str, Any]):
        """Yield ``(name, current, reference)`` for every input carrying movement.

        These are the packed variables (from the layout stored by :meth:`pack`)
        with a supplied reference, plus the supplied variables that are derived
        from an explicit relation (e.g. a profile reconstructed from an
        average).  Both the movement residual and its IRLS reweighting iterate
        this same set so their rows stay aligned.
        """
        packed = set()
        for name, *_rest in self.packed_specs:
            packed.add(name)
            if name in values and name in reference and reference[name] is not None:
                yield name, values[name], reference[name]
        for name in sorted(self.derived_variable_names - packed):
            # A shape-locked (supplied, unfixed) profile carries no per-point
            # movement of its own: the level is controlled by its (packed)
            # scalar average, which already contributes the movement penalty.
            profile = self.supplied_profiles.get(name)
            if profile is not None and not profile[1]:
                continue
            var = self.variables_by_name.get(name)
            if var is None or var.input_value is None or name not in values or values[name] is None:
                continue
            yield name, values[name], self._solver_value(name, var.input_value)

    def movement_residuals(self, values: Mapping[str, Any], reference: Mapping[str, Any], weights: Mapping[str, float] | None = None) -> np.ndarray:
        """Return the reweighted-L1 movement penalty, one residual per input.

        Each input contributes ``sqrt(weight * excess)``, so its squared cost is
        ``weight * excess`` -- a deadzone L1 penalty on the beyond-tolerance
        excess.  The per-input ``weight`` (default 1) is what the reconcile
        IRLS loop adjusts via :meth:`movement_weights`: down-weighting inputs
        already far past tolerance and up-weighting the marginal ones reweights
        the L1 so repeated solves drive the *number* of crossings down (the
        convex, iteratively-reweighted surrogate for the L0 "fewest inputs
        beyond tolerance" aim).  The weights are mode-owned state, passed in
        per call.

        Args:
            values: Current solved values.
            reference: Supplied input reference values (solver units).
            weights: Per-input IRLS weights; missing entries default to 1.

        Returns:
            One residual per movement-carrying input variable.
        """
        weights = weights or {}
        rows = [
            np.asarray([np.sqrt(weights.get(name, 1.0) * self.variables_by_name[name].movement_excess(current, ref))], dtype=float)
            for name, current, ref in self._movement_inputs(values, reference)
        ]
        return np.concatenate(rows) if rows else np.empty(0, dtype=float)

    def movement_weights(self, values: Mapping[str, Any], reference: Mapping[str, Any], *, eps: float) -> dict[str, float]:
        """Return movement L1 weights from the current solution (one IRLS step).

        ``weight = 1 / (excess + eps)`` per input: an input already well
        past tolerance gets a small weight (cheap to leave changed), while one
        only marginally out gets a large weight (strongly pushed back inside).
        Re-solving with these weights is the iteratively-reweighted-L1 update
        whose fixed point minimises the count of inputs beyond tolerance.  The
        caller owns the weights and passes them to :meth:`movement_residuals`.

        Args:
            values: Latest solved namespace.
            reference: Supplied input reference values (solver units).
            eps: Reweighting floor; smaller drives sparser (more aggressive)
                solutions at some cost to stability.
        """
        return {
            name: 1.0 / (self.variables_by_name[name].movement_excess(current, ref) + float(eps))
            for name, current, ref in self._movement_inputs(values, reference)
        }

    # ── Store and final-value checks ──────────────────────────────────────

    def _fixed_value_errors(self, values: Mapping[str, Any]) -> list[str]:
        """Return errors for fixed variables changed in a candidate value map."""
        errors: list[str] = []
        for name, var in self.variables_by_name.items():
            if not var.fixed or var.input_value is None or name not in values or values[name] is None:
                continue
            try:
                if var.moved_from_input(values[name]):
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

    def store(self, values: Mapping[str, Any]) -> None:
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

    # ── Per-variable delegates and small helpers ──────────────────────────

    def _variable_for(self, name: str) -> Variable:
        """Return the live :class:`Variable` for a registry-known name.

        Value-form conversion is owned by :class:`Variable`
        (:meth:`Variable.solver_value` / :meth:`Variable.public_value`).  The
        rare name the system does not hold a variable for (a seeding write for
        an inactive relation's output) gets an unregistered throwaway carrying
        the same spec, so conversion works without mutating the system.
        """
        var = self.variables_by_name.get(name)
        if var is not None:
            return var
        spec = self.variable_registry.get(name)
        return Variable(spec.canonical_name, size=self.profile_size if spec.shape == 1 else None)

    def _solver_value(self, name: str, value: Any) -> Any:
        """Convert a public value to canonical solver shape (see :meth:`Variable.solver_value`)."""
        return self._variable_for(name).solver_value(value)

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

    def _status_and_residual(self, rel: Relation, eval_values: Mapping[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
        """Return one relation's verify status and residual vector from one evaluation."""
        return rel.status_and_residual(eval_values, scales=self.variable_scales, rel_tols=self.variable_tolerances, abs_tols=self.variable_abs_tolerances)

    def _public_value(self, name: str, value: Any) -> Any:
        """Project solver values to public values (see :meth:`Variable.public_value`)."""
        return self._variable_for(name).public_value(value)

    def refresh_scales(self) -> None:
        """Refresh variable scales and tolerances used by residuals.

        Domains and solver domains are admissible-value constraints, not
        numerical scales.  The finite scale floor comes from abs_tol / rel_tol,
        while current/reference magnitudes provide relative scaling.

        Also called post-solve in reconcile to rescale around stored values, so
        this stays a method rather than being inlined into the compile pass.
        """
        self.variable_tolerances = {
            name: float(var.rel_tol or self.variable_registry.rel_tol_default)
            for name, var in self.variables_by_name.items()
        }
        self.variable_abs_tolerances = {
            name: float(var.abs_tol) for name, var in self.variables_by_name.items()
        }
        # Scale each variable from its current/input reference magnitude (the
        # first of value/input_value with a finite element), defaulting to 0.0.
        self.variable_scales = {}
        for name, var in self.variables_by_name.items():
            reference = 0.0
            for value in (var.value, var.input_value):
                if value is None:
                    continue
                arr = np.asarray(value, dtype=float).reshape(-1)
                finite = arr[np.isfinite(arr)]
                if finite.size:
                    reference = float(np.max(np.abs(finite)))
                    break
            self.variable_scales[name] = var.scale(reference)

    def _initial_value(self, var: Variable, index: int | None = None) -> float:
        """Return an initial value for one variable element.

        Initial values may come only from user input or relation-generated guesses.
        Solver domains are constraints, not value providers.
        """
        # Relation-generated values are x0 hints, not movement references.  They
        # may override supplied non-fixed values for initialization only; the
        # original public value remains in the movement reference map.
        if self._initial_guesses.get(var.name) is not None:
            solver_value = var.solver_value(self._initial_guesses[var.name])
            var.check_solver_domain(solver_value)
            arr = np.asarray(solver_value, dtype=float).reshape(-1)
            if arr.size:
                return float(arr[min(index or 0, arr.size - 1)])

        if var.input_value is not None:
            solver_value = var.solver_value(var.input_value)
            var.check_solver_domain(solver_value)
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
        if var.name in self.block_core_names:
            return float(var.tolerance_floor())

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
        """Scalar-element count for one variable (see :attr:`Variable.dim`)."""
        return self._variable_for(name).dim

    def _resolve_relation_names(self, rel: Relation) -> Relation:
        return canonicalize_relation_names(rel, self.variable_registry)

    def _ensure_variable_exists(self, raw_name: str) -> Variable:
        if str(raw_name) not in self.variable_registry:
            raise ValueError(f"Relation requires unknown variable {str(raw_name)!r}.")
        spec = self.variable_registry.get(raw_name)
        name = spec.canonical_name
        if name in self.variables_by_name:
            return self.variables_by_name[name]
        var = Variable(name, size=self.profile_size if spec.shape == 1 else None)
        self.variables_by_name[name] = var
        return var

