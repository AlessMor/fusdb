"""RelationSystem container, graph compiler, seeding oracle and mode dispatcher."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import networkx as nx
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


from .relation import COORDINATE_NAMES, Relation, canonicalize_relation, canonicalize_relation_names, constraint_from_expression, is_default_relation
from .registry import VARIABLES
from .utils import ZERO_TOL, parse_constraint_specs, signed_scalar_grid, value_in_domain, volume_average
from .variable import Variable


# ── Pure structural helpers ────────────────────────────────────────────────
# Module-level functions with explicit inputs: they read nothing off a
# RelationSystem, so the compile pass's dataflow stays visible at call sites.


def relation_bipartite_graph(relations: Iterable[Relation]) -> nx.DiGraph:
    """Return the directed bipartite ``input -> relation -> output`` graph.

    One node per variable (``("variable", name)``) and per relation
    (``("relation", name)``); edges carry each relation's declared direction.
    Relation nodes are annotated with the ``relation`` object, its ordered
    ``variables`` tuple, ``enforce`` and ``is_default``; variable nodes carry
    only ``kind`` -- callers layer their own annotations on top (the compiled
    system adds ``shape`` plus the per-pass verdicts, the plotting views add
    display labels).

    This is the single graph definition shared by the compiled system
    (:meth:`RelationSystem._structural_graph`) and the registry-wide plotting
    views (:mod:`fusdb.plotting.relation_graph`).
    """
    graph = nx.DiGraph()
    for rel in relations:
        rnode = ("relation", rel.name)
        graph.add_node(
            rnode,
            kind="relation",
            relation=rel,
            variables=rel.variables,
            enforce=bool(rel.enforce),
            is_default=is_default_relation(rel),
        )
        for name in rel.input_names:
            graph.add_node(("variable", name), kind="variable")
            graph.add_edge(("variable", name), rnode)
        for name in rel.outputs:
            graph.add_node(("variable", name), kind="variable")
            graph.add_edge(rnode, ("variable", name))
    return graph


def _free_parameter_names(relations: list[Relation]) -> set[str]:
    """Return the free parameters of a relation pool.

    A variable that no relation in the pool produces and that is referenced
    only by outputless equality (constraint) relations is a free parameter the
    constraints cannot pin (e.g. an unsupplied ``tau_p`` that only the particle
    balances reference).  It must never be solved as "the last unknown" of a
    constraint -- callers treat these names as undecidable so the referencing
    constraints deactivate instead of inventing a value.  This is the single
    definition of the rule; forward seeding and the structural partition both
    read it.
    """
    produced = {out for rel in relations for out in rel.output_names}
    refs: dict[str, list[Relation]] = {}
    for rel in relations:
        for name in rel.variables:
            refs.setdefault(name, []).append(rel)
    return {
        name
        for name, rels in refs.items()
        if name not in produced and all((not rel.outputs and rel.op == "==") for rel in rels)
    }


def _forward_decision_rounds(
    relations: list[Relation], supplied: Iterable[str], extra_known: Iterable[str] = (), *, acausal: bool = True
) -> tuple[dict[str, int], dict[str, Relation]]:
    """Return forward-closure rounds and per-variable forward providers.

    ``supplied`` names are decided at round 0; ``extra_known`` seeds additional
    variables as decided at round 0 -- used to treat block cores as available
    so the block-downstream variables get forward providers.

    ``acausal=False`` restricts the closure to true forward decisions (every
    input known); the single-remaining-unknown fallback below is skipped.  The
    default-activation gate uses this stricter closure: a variable that is
    only *acausally* decidable -- the last unknown of a multi-variable
    relation, e.g. ``n_la`` by inverting a confinement scaling where it enters
    with exponent ~0.1 -- is structurally determined but usually terribly
    conditioned, so a registry default relation should still provide it.

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
    non_default = [rel for rel in relations if not is_default_relation(rel)]
    defaults = [rel for rel in relations if is_default_relation(rel)]
    # The acausal fallback must not solve a free parameter as "the last
    # unknown" of a constraint; leaving it undecided lets the structural
    # partition mark it underdetermined and deactivate the constraints that
    # need it (see _free_parameter_names).
    pure_input = _free_parameter_names(relations)
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
        if not acausal:
            continue
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
    and their topological order is the solve order.
    """
    determined = [c for c in range(len(match_row)) if c not in under_cols and match_row[c] >= 0]
    if not determined:
        return []
    determined_set = set(determined)
    # Dependency digraph over determined columns: c2 -> c when c2 appears in
    # the relation matched to c (so c depends on c2).  Strongly connected
    # components are the simultaneous solve blocks; the condensation gives
    # their dependency order.  Names and columns are one-to-one, so each
    # component maps to a disjoint name group.
    digraph = nx.DiGraph()
    digraph.add_nodes_from(determined)
    for c in determined:
        for c2 in row_adj[int(match_row[c])]:
            if c2 != c and c2 in determined_set:
                digraph.add_edge(c2, c)
    condensation = nx.condensation(digraph)

    def component_names(comp: int) -> list[str]:
        return sorted(name_of_col[col] for col in condensation.nodes[comp]["members"])

    return [
        tuple(component_names(comp))
        for comp in nx.lexicographical_topological_sort(condensation, key=component_names)
    ]




class RelationSystem:
    """Variables and relations compiled into one numeric system.

    Execution modes (:mod:`fusdb.modes`) drive a compiled system through this
    public interface and own their own algorithm and result shape:

    - :meth:`compile` -- build/prune the active system (``run`` calls this).
    - :meth:`pack` / :meth:`unpack` -- free variables <-> solver vector.
      ``pack`` stores the packed layout (``packed_specs``); the residual
      helpers read it.
    - :meth:`values` / :meth:`complete` -- read the value namespace; close it.
    - :meth:`residual_layout` + :meth:`layout_relation_rows` /
      :meth:`layout_domain_rows` / :meth:`layout_movement_rows` -- the
      fixed-shape stage residual blocks (modes weight them; the single
      residual protocol: freeze a layout, evaluate rows against it);
      :meth:`certify_relations` -- the certification statuses.
    - :meth:`store` -- write solved values back into the variables.
    - :func:`initial_values_from_graph` (module function),
      :meth:`build_jac_sparsity`, :meth:`movement_weights`,
      :meth:`refresh_scales` -- solve setup/initialization helpers.

    Trust boundary: ``modes/`` also reads four *compiled-plan* members that are
    underscore-prefixed but are part of the interface the modes depend on, not
    private implementation -- ``_enforced_residual_relations``,
    ``_profile_specs``, ``_constant_defaults_solver`` and
    ``_apply_completion_providers``.  They are the frozen products of
    :meth:`compile` (a Modelica-style compiled artifact lives here as method
    state rather than a separate object -- see the deferred CompiledSystem
    note in the refactor backlog), so a compile-internals change must keep
    these four stable or update the modes with it.

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
        # Result of the most recent run(); read by table rendering for header
        # colouring. None until a mode has been dispatched.
        self.last_result: dict[str, Any] | None = None
        # First swallowed exception per provider relation of the most recent
        # completion cycle (strings only, so results stay picklable).  The
        # solver keeps running on a failed provider -- a missing value
        # penalizes its own residual rows -- but the original cause must
        # survive into diagnostics instead of surfacing as a bare "value
        # missing" downstream.
        self.completion_errors: dict[str, str] = {}
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
        # Movement plan, one record per movement-carrying input
        # ``(name, reference, width, is_scalar)``; written by pack().  The
        # reference value and tolerance width are fixed for a whole solve, so
        # they are resolved once there instead of per residual call.
        self._movement_plan: list[tuple[str, Any, float, bool, float | None]] = []
        # Registry defaults that seed packed variables.  Every applied default
        # lands here: they may move to satisfy enforced relations, and reconcile
        # treats the default value as the movement reference.
        self.seeded_default_values: dict[str, float] = {}

        # ── Per-scenario state: plain dicts keyed by canonical name.  The
        # passed Variable records are ingested once; the system holds no
        # Variable objects at runtime (specs own the numerics).
        records = list(variables)
        self.inputs: dict[str, Any] = {}      # immutable supplied values (canonical units)
        self.values: dict[str, Any] = {}      # current/solved values (canonical units)
        self.fixed: set[str] = set()
        self.rel_tols: dict[str, float] = {}  # resolved per-name tolerances
        self.abs_tols: dict[str, float] = {}
        self.known: set[str] = set()          # every name the system tracks
        # Record-local constraint guards (rare; registry guards live on specs).
        self._record_guards: dict[str, tuple[Relation, ...]] = {}
        explicit_sizes: set[int] = set()
        for rec in records:
            if rec.name in self.known:
                raise ValueError(f"Duplicate variable {rec.name!r}.")
            self.known.add(rec.name)
            self.rel_tols[rec.name] = float(rec.rel_tol or VARIABLES.rel_tol_default)
            self.abs_tols[rec.name] = float(rec.abs_tol or 0.0)
            if rec.fixed:
                self.fixed.add(rec.name)
            if rec.input_value is not None:
                self.inputs[rec.name] = rec.input_value
                self.values[rec.name] = rec.value
            if rec.relations:
                self._record_guards[rec.name] = rec.relations
            if rec.spec.shape == 1:
                if rec.size is not None:
                    explicit_sizes.add(int(rec.size))
                elif isinstance(rec.input_value, np.ndarray) and rec.input_value.ndim == 1:
                    explicit_sizes.add(int(rec.input_value.shape[0]))

        # Construction phases: each reads the state left by earlier phases
        # and writes its own attributes onto self.
        # The raw constraint spec is kept so a picklable rebuild recipe can be
        # derived from a live system (worker processes cannot receive the parsed
        # constraint relations, whose functions do not pickle).
        self.constraints_spec = constraints
        self._canonicalize_candidates(relations, constraints)
        if len(explicit_sizes) > 1:
            raise ValueError(f"Profile sizes are incompatible: {sorted(explicit_sizes)}.")
        self.profile_size = next(iter(explicit_sizes), VARIABLES.profile_size_default)
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
        #     variable_roles (the one solve role per variable, see
        #     _assign_roles), constant_default_values, structural_blocks,
        #     primary_relations, relations, relations_by_name,
        #     relations_by_function, variable_scales/tolerances,
        #     _partition_diagnostics, plus the None-reset caches below.
        #     Relation activation, inactivation reasons and provider
        #     selection are graph verdicts: annotations on self._graph
        #     (see _reset_graph_verdicts), read back through the
        #     default_provider_by_output / derived_provider_by_output /
        #     blocked_relation_reasons view properties;
        #   * seeding oracle -- initial_guesses, seed_provenance,
        #     _unevaluable_names (compile()),
        #   * completion plan -- _profile_specs, _constant_defaults_solver,
        #     _provider_plan, _completion_passes (every _run_compile_pass);
        #   * packed layout -- packed_specs, packed_dim, _packed_base_values,
        #     uninitialized_free_variables (pack()).
        self.variable_roles: dict[str, str] = {}
        # Solver mechanics behind the roles, not origin: which values are
        # solver unknowns, and which of those get no seed entitlement.
        self.packed_variables: set[str] = set()
        self.unseeded_variables: set[str] = set()
        # Values expanded from a scalar average with an assumed shape --
        # orthogonal to the role, and true for inputs and computed alike.
        self.avg_to_profile: set[str] = set()
        # Supplied-name/fixed-set fingerprint of the last full compile; while it
        # matches, compile() refreshes only the value-dependent products.
        self._compile_fingerprint: tuple[frozenset[str], frozenset[str]] | None = None
        # Seeding tape: the ordered steps the oracle's discovery run took and
        # the name set it seeded.  Recorded by the full compile, replayed on
        # fingerprint hits, invalidated whenever the structure recompiles.
        self._seed_tape: list | None = None
        self._seed_tape_names: set[str] = set()
        self.initial_guesses: dict[str, Any] = {}
        # Per seeded name, the oracle source that produced its x0 value
        # ("held_constant", "relation:<name>", "block", "registry_default").
        self.seed_provenance: dict[str, str] = {}
        self.uninitialized_free_variables: list[str] = []
        # Profile-shaped free cores with no supplied data and no average+shape
        # reconstruction (S9).  Their uniform (flat) shape is the default and
        # their level is a free scalar left at its seed -- correct and uniform,
        # but the level is *defaulted*, not measured.  Detected at pack and
        # surfaced (compiler report + verify warning); such a level is "assumed".
        self.underdetermined_profiles: list[str] = []
        # The canonical bipartite structural graph over the (immutable)
        # candidate relations; built once on first use.  Its *structure*
        # never changes; compile passes rewrite only verdict annotations.
        self._graph: nx.DiGraph | None = None
        # Lazy view of the provider edge annotations ``(defaults, explicit)``;
        # invalidated by _reset_graph_verdicts and _set_provider_edge.
        self._provider_view: tuple[dict[str, Relation], dict[str, Relation]] | None = None
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

    def _materialize_rho_grid(self) -> None:
        """Materialize the canonical fixed ``rho`` grid whenever profiles or
        rho-dependent relations are present."""
        if "rho" in self.variable_registry:
            # rho is a relation *constant* now (S3), not a solved variable, so it
            # no longer appears in ``rel.variables``; detect its users through
            # ``constant_names``.  It stays framework-materialized here so every
            # evaluation namespace still carries the identical grid.
            uses_rho = any("rho" in rel.constant_names for rel in self.candidate_primary_relations)
            has_profile = any(self.variable_registry.get(name).shape == 1 for name in self.known)
            if uses_rho or has_profile:
                self.track("rho")
                if self.inputs.get("rho") is None:
                    rho_value = self.variable_registry.uniform_profile_grid(self.profile_size)
                    self.inputs["rho"] = rho_value
                    self.values["rho"] = rho_value.copy()
                self.fixed.add("rho")

    def _broadcast_profile_values(self) -> None:
        """Broadcast scalar profile data onto the shared grid and validate
        explicitly supplied profile lengths."""
        for name in sorted(self.known):
            if self.variable_registry.get(name).shape != 1:
                continue
            for store in (self.inputs, self.values):
                value = store.get(name)
                if value is None:
                    continue
                arr = np.asarray(value, dtype=float)
                if arr.ndim == 0:
                    store[name] = np.full(self.profile_size, float(arr))
                elif arr.ndim == 1 and arr.shape[0] != self.profile_size:
                    raise ValueError(f"Profile {name!r} has length {arr.shape[0]}, expected {self.profile_size}.")

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
        for name in sorted(self.known):
            if name == "rho" or self.variable_registry.get(name).shape != 1 or self.inputs.get(name) is None:
                continue
            avg_name = self.variable_registry.average_of(name)
            if avg_name is None:
                continue
            self.track(avg_name)
            arr = np.asarray(self.inputs[name], dtype=float)
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
            self.supplied_profiles[name] = (shape.astype(float, copy=False), name in self.fixed)
            supplied_average = self.inputs.get(avg_name)
            profile_average = self.public_value(avg_name, avg)
            # The profile <-> average link is enforced by the
            # ``<x>_avg == volume_average(<x>)`` consistency relation, not by
            # pinning here: a fixed profile's average is seeded from the profile
            # (so it stays known/decidable) but left free, so a separately supplied
            # scalar that disagrees is surfaced and reconciled by the mode instead
            # of being silently overwritten.  A supplied average always wins the
            # seed; the residual then measures any disagreement with the profile.
            if supplied_average is None:
                self.inputs[avg_name] = profile_average
                self.values[avg_name] = profile_average

    # ── Public entry points and mode dispatch ────────────────────────────

    def verify(self, **options: Any) -> dict[str, Any]:
        return self.run("verify", **options)

    def reconcile(self, **options: Any) -> dict[str, Any]:
        return self.run("reconcile", **options)

    def optimize(self, **options: Any) -> dict[str, Any]:
        return self.run("optimize", **options)

    def popcon(self, *, x: Any, y: Any, **options: Any) -> dict[str, Any]:
        """Run a two-dimensional POPCON scan."""
        return self.run("popcon", x=x, y=y, **options)

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

    def compile(self, *, force: bool = False) -> None:
        """Compile the active system, pruning relations needing unevaluable variables.

        Public entry to the compiled-execution interface; modes assume it has
        run.  ``run`` calls it before dispatch, so modes reached through ``run``
        need not call it; a caller invoking a mode function directly should call
        ``compile`` first.

        The structural verdicts depend only on *which* variables are supplied
        and fixed, never on their numeric values, so when that fingerprint is
        unchanged since the last full compile the structural passes are
        skipped and only the value-dependent products are refreshed: residual
        scales, the fixed-profile reconstruction specs and the seeding
        oracle's x0 values (replayed from the recorded seeding tape when one
        exists, see :func:`_replay_seed_tape`).  (Keeping the previous seeds
        verbatim instead of re-deriving them was measured 2026-07 and
        REJECTED: reviving stale/injected
        guesses as x0 slowed the popcon point solves ~60% and shifted
        certified values ~1-4%.)  If the new values leave a packed variable
        the fresh oracle cannot seed -- something a full compile would have
        pruned on -- the full prune-to-fixpoint loop runs as the fallback, so
        this path can never produce a quietly different structure.
        ``force=True`` always re-runs the full loop.
        """
        fingerprint = (
            frozenset(name for name, value in self.inputs.items() if value is not None),
            frozenset(self.fixed),
        )
        if not force and fingerprint == self._compile_fingerprint:
            self.refresh_scales()
            self._profile_specs = self._build_profile_specs()
            self._refresh_seeds()
            # The full-compile fixpoint guarantees any uninitialized packed
            # variable is already recorded unevaluable; anything beyond that
            # set means the new values broke a seed the pack relies on.
            self.pack()
            if set(self.uninitialized_free_variables) <= self._unevaluable_names:
                return
        # Cleared first so an exception mid-compile can never leave a stale
        # fingerprint claiming the partial products are valid.
        self._compile_fingerprint = None
        self._compile_with_pruning()
        self._compile_fingerprint = fingerprint

    def run(self, mode: str = "verify", *, save: Any = None, **options: Any) -> dict[str, Any]:
        """Prepare runtime initialization and dispatch to an isolated execution mode.

        ``save`` optionally names an HDF5 path the result is archived to (see
        :func:`fusdb.io.save_result`; requires the optional ``h5py`` extra).
        """
        from .modes import get_mode

        self.compile()
        self.last_result = get_mode(mode)(self, **options)
        if save is not None:
            from .io import save_result

            save_result(self.last_result, save)
        return self.last_result

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
        # ── Reset the graph verdicts and the caches derived from the previous
        # provider/active-relation plan.  The graph's structure is immutable
        # (one node per candidate relation/variable); each compile pass
        # rewrites only the verdict annotations -- relation activation and
        # reason, provider edges, per-variable supplied/fixed flags.
        self._reset_graph_verdicts()
        self._completion_plan_cache = None
        self._sparsity_graph_cache = None
        self._compiler_report_cache = None

        supplied = set(self.inputs)
        usable = self._usable_candidates()
        pool, forward, seeded = self._activate_defaults(usable, supplied)
        active, decidable, active_vars = self._select_active_relations(pool, forward, supplied)
        derived, cores = self._select_cores_and_providers(active, active_vars, decidable, supplied, seeded)
        self._register_profile_generators(active_vars, derived)
        self._append_guard_relations(active_vars)
        self.variable_roles = self._assign_roles(active_vars, derived, cores)

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
                constant_defaults_solver[name] = self.solver_value(name, value)
            except Exception:
                pass
        self._constant_defaults_solver = constant_defaults_solver
        # Provider records with the per-call plumbing frozen: input names and
        # the writable outputs resolved to their specs (registry-known, and
        # either active or explicitly owned), so the completion loop does no
        # name resolution or ownership checks per call.
        self._provider_plan = [
            (
                rel,
                only_missing,
                rel.input_names,
                tuple(
                    (out_name, self.variable_registry.get(out_name))
                    for out_name in rel.output_names
                    if out_name in self.variable_registry
                    and (self.variable_roles.get(out_name, "inactive") != "inactive" or out_name in self.derived_provider_by_output)
                ),
            )
            for rel, only_missing in self._completion_plan()
        ]
        # One completion pass is exact for an acyclic provider graph (the plan
        # is topologically ordered); only genuine provider cycles need iteration.
        self._completion_passes = 1 if self._completion_acyclic else 6
        # The enforced residual relations, in relation order.  The residual
        # vector is built once per least-squares evaluation, so membership is
        # decided here rather than per call.
        self._enforced_residual_relations = [
            rel for rel in self.relations if rel.enforce and self._relation_is_residual_relation(rel)
        ]
        # Domain-violation plan: active registry variables with a physical
        # domain, in sorted-name order.  Scalar entries carry an index into
        # the packed scalar bound/tolerance arrays and profiles an index into
        # the packed profile arrays, so _domain_rows checks each kind as
        # one vectorized batch per call; values of an unexpected type fall
        # back to the per-variable spec check.
        plan: list[tuple[str, Any, float, float, int]] = []
        names: list[str] = []
        lows: list[float] = []
        highs: list[float] = []
        rel_tols: list[float] = []
        abs_tols: list[float] = []
        floors: list[float] = []
        profile_index: dict[str, int] = {}
        p_lows: list[float] = []
        p_highs: list[float] = []
        p_rel_tols: list[float] = []
        p_abs_tols: list[float] = []
        p_floors: list[float] = []
        for name in sorted(self.variable_roles):
            if self.variable_roles[name] == "inactive" or name not in self.variable_registry:
                continue
            spec = self.variable_registry.get(name)
            lower, upper, lower_inc, upper_inc = spec.domain
            if lower is None and upper is None:
                continue
            rel_tol, abs_tol = self.tols_of(name)
            if spec.shape == 1:
                plan.append((name, spec, rel_tol, abs_tol, -1))
                profile_index[name] = len(p_lows)
                p_lows.append(-np.inf if lower is None else float(lower) + (ZERO_TOL if not lower_inc else 0.0))
                p_highs.append(np.inf if upper is None else float(upper) - (ZERO_TOL if not upper_inc else 0.0))
                p_rel_tols.append(rel_tol)
                p_abs_tols.append(abs_tol)
                p_floors.append(spec.tolerance_floor(rel_tol, abs_tol))
                continue
            plan.append((name, spec, rel_tol, abs_tol, len(names)))
            names.append(name)
            lows.append(-np.inf if lower is None else float(lower) + (ZERO_TOL if not lower_inc else 0.0))
            highs.append(np.inf if upper is None else float(upper) - (ZERO_TOL if not upper_inc else 0.0))
            rel_tols.append(rel_tol)
            abs_tols.append(abs_tol)
            floors.append(spec.tolerance_floor(rel_tol, abs_tol))
        self._domain_plan = plan
        self._domain_scalar_bounds = (
            np.asarray(lows, dtype=float),
            np.asarray(highs, dtype=float),
            np.asarray(rel_tols, dtype=float),
            np.asarray(abs_tols, dtype=float),
            np.asarray(floors, dtype=float),
        )
        self._domain_profile_index = profile_index
        self._domain_profile_bounds = (
            np.asarray(p_lows, dtype=float),
            np.asarray(p_highs, dtype=float),
            np.asarray(p_rel_tols, dtype=float),
            np.asarray(p_abs_tols, dtype=float),
            np.asarray(p_floors, dtype=float),
        )
        # Derived/held names that may carry a movement row, sorted once here
        # instead of per residual call (see _build_movement_plan).
        self._movement_candidate_names = sorted(
            name for name, role in self.variable_roles.items()
            if role in ("computed", "assumed") and name not in self.packed_variables
        )
        # Registry-default seed plan: the single decision point for which
        # variables may be seeded from their registry default and from what
        # source.  The seeding oracle (_seed_defaults) only applies this plan;
        # eligibility is never re-derived at seed time.  ``source`` is a float
        # (constant seed) or a canonical variable name (copy its value once
        # available); ``requires`` is the resolved gate variable or None.
        seed_plan: list[tuple[str, float | str, str | None]] = []
        for name in sorted(self.variable_roles):
            if self.variable_roles[name] == "inactive" or name not in self.variable_registry:
                continue
            if name in self.fixed or self.inputs.get(name) is not None:
                continue
            spec = self.variable_registry.get(name)
            if spec.default is None:
                continue
            requires = None if spec.default_requires is None else self.variable_registry.resolve(spec.default_requires)
            if isinstance(spec.default, str):
                if spec.default not in self.variable_registry:
                    continue
                source: float | str = self.variable_registry.get(spec.default).canonical_name
            else:
                source = float(spec.default)
            seed_plan.append((name, source, requires))
        self._default_seed_plan = seed_plan


    def _usable_candidates(self) -> list[Relation]:
        """Return candidate relations usable this pass, ensuring their variables.

        A ``generated_profile`` relation whose outputs are all supplied is
        inactive: the supplied profile shape is authoritative.  Non-fixed
        supplied profiles are still reconciled through the profile-average
        reconstruction path; the generator is only for missing profiles.
        """
        usable: list[Relation] = []
        for rel in self.candidate_primary_relations:
            if rel.dependency == "generated_profile" and rel.output_names and all(
                self.inputs.get(out) is not None for out in rel.output_names
            ):
                self._mark_relation_inactive(rel, "inactive_profile_supplied")
                continue
            usable.append(rel)
            for rel_name in rel.variables:
                self.track(rel_name)
        return usable

    def _activate_defaults(self, usable: list[Relation], supplied: set[str]) -> tuple[list[Relation], set[str], set[str]]:
        """Activate registry and relation defaults against forward decidability.

        Writes ``constant_default_values`` and the ``"default"`` provider
        edges; marks never-activated defaults inactive.  Returns ``(pool,
        forward, seeded)``: the non-default + activated-default relation pool,
        the bijection-closed forward-decidable set, and the free-core default
        seeds.
        """
        non_default = [rel for rel in usable if not is_default_relation(rel)]
        defaults = [rel for rel in usable if is_default_relation(rel)]
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
        def _forward_with_bijections(rels: list[Relation], extra_known: Iterable[str] = (), *, acausal: bool = True) -> set[str]:
            seed = set(supplied) | set(extra_known)
            decided = set(seed) | set(_forward_decision_rounds(rels, supplied, extra_known=seed, acausal=acausal)[0])
            changed_inv = True
            while changed_inv:
                changed_inv = False
                for rel in rels:
                    # A relation over a coordinate grid (rho) is a profile<->average
                    # reduction, not a scalar bijection: many profiles share one
                    # average, so an average must NOT be treated as invertible to a
                    # profile here.  rho is a constant now (S3), so guard on it
                    # explicitly to keep these relations out of the bijection
                    # closure exactly as when rho counted toward their arity.
                    if len(rel.variables) != 2 or any(c in COORDINATE_NAMES for c in rel.constant_names):
                        continue
                    unknown = [v for v in rel.variables if v not in decided]
                    if len(unknown) == 1:
                        decided.add(unknown[0])
                        changed_inv = True
            return decided

        base_forward = _forward_with_bijections(non_default)
        # Apply registry defaults to variables the user did not supply and that no
        # real relation already forward-decides.  Every default is a PENALIZED
        # INPUT: a packed unknown seeded with the default value, which an enforced
        # relation may move off that seed against a movement penalty anchored on
        # it.  It is never a hard pin -- the penalty is free inside the variable's
        # tolerance band and L1 beyond it, and reconcile weights relations far
        # above movement, so a default yields when the data requires it and stays
        # put when nothing pushes it.  A supplied or forward-derivable value
        # always wins over a default.
        #
        # ``default_requires`` still gates one class, and only that class: the
        # ash and impurity fractions, which must be EXACTLY their default (zero)
        # unless the gate variable -- a particle confinement time, per species or
        # the shared ``tau_p`` propagated by a default relation -- says a balance
        # exists to derive them.  Without it there is no mechanism to create
        # helium or an impurity, so "a little bit of ash" is not a weaker
        # assumption than none, it is a wrong one.  A gated default whose gate is
        # unavailable is therefore held exactly; everything else is a penalized
        # input.
        candidate_vars = {name for rel in usable for name in rel.variables}
        self.constant_default_values = {}
        self.seeded_default_values = {}
        seeded: set[str] = set()
        for name in sorted(candidate_vars):
            if name in supplied or name in base_forward or name not in self.variable_registry:
                continue
            spec = self.variable_registry.get(name)
            if spec.default is None or name in self.fixed:
                continue
            gate_unavailable = (
                spec.default_requires is not None
                and self.variable_registry.resolve(spec.default_requires) not in (supplied | base_forward)
            )
            if gate_unavailable:
                if not isinstance(spec.default, str):
                    self.constant_default_values[name] = float(spec.default)
                continue
            seeded.add(name)
            if not isinstance(spec.default, str):
                self.seeded_default_values[name] = float(spec.default)
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
            forward = _forward_with_bijections(non_default + active_defaults, extra_known=known_defaults | forward)
        # The activation gate is two-sided.  A default activates when its
        # output is not *truly* derivable -- the strict closure: true forward
        # decisions plus two-variable inversions, no acausal fallback -- while
        # its inputs are decidable in the loose (acausal-inclusive) sense,
        # because the solve will produce them.  Gating the output on the loose
        # closure would let mere acausal decidability (the last unknown of a
        # multi-variable relation, an often ill-conditioned inversion) block a
        # registry default relation from providing the value.
        strict_forward = _forward_with_bijections(non_default + active_defaults, extra_known=known_defaults, acausal=False)
        changed = True
        while changed:
            changed = False
            for rel in sorted(defaults, key=lambda item: (len(item.variables), item.name)):
                if rel in active_defaults:
                    continue
                if any(out not in strict_forward for out in rel.output_names) and all(inp in forward for inp in rel.input_names):
                    active_defaults.append(rel)
                    # The closure is monotone in its seed and relation pool, so
                    # extending each current fixed point with the enlarged pool
                    # is exact -- no from-scratch recomputation per activation.
                    forward = _forward_with_bijections(non_default + active_defaults, extra_known=known_defaults | forward)
                    strict_forward = _forward_with_bijections(non_default + active_defaults, extra_known=known_defaults | strict_forward, acausal=False)
                    changed = True
        # Only activated defaults are completion fallbacks.  A default whose
        # output a non-default relation can determine (forward or by a
        # two-variable inversion) is never registered as a provider, so it cannot
        # overwrite that derived value in completion (verify) or in reconcile.
        # A profile is ``average x shape``.  The uniform fallback supplies the
        # shape only when nothing else does, so an active SHAPE generator for the
        # same profile retires it -- otherwise both stay enforced, the fallback
        # (being forward-decidable from the average alone) wins the provider slot,
        # and the shape generator becomes a residual that can never be satisfied.
        # The activation gate above already intends this, but it is evaluated on
        # the forward closure as it stands mid-fixpoint: a generator whose average
        # only becomes decidable later (via composition/quasineutrality) has not
        # claimed its output yet, so the fallback activates anyway.  The pool is
        # final here, so the check is exact.
        shaped_outputs = {
            out
            for rel in non_default
            if "profile_shape" in rel.tags
            for out in rel.output_names
        }
        if shaped_outputs:
            retired = [rel for rel in active_defaults if shaped_outputs.intersection(rel.output_names)]
            for rel in retired:
                active_defaults.remove(rel)
                self._mark_relation_inactive(
                    rel, "inactive_default_superseded_by_shape_generator", replace=True
                )

        claimed: set[str] = set()
        for rel in sorted(active_defaults, key=lambda item: item.name):
            for out in rel.output_names:
                if out not in claimed:
                    self._set_provider_edge(out, rel, "default")
                    claimed.add(out)
        pool = non_default + active_defaults
        for rel in defaults:
            if rel not in active_defaults:
                self._mark_relation_inactive(rel, "inactive_default_not_needed")

        # Record the structural decidability class of every candidate variable
        # -- the named verdict of the two-sided activation gate above.  The
        # remaining classes ("block", "underdetermined") are filled in by
        # _select_active_relations once the DM partition has run.
        graph = self._structural_graph()
        for node, data in graph.nodes(data=True):
            if data["kind"] != "variable":
                continue
            name = node[1]
            if name in supplied:
                data["decidability"] = "supplied"
            elif name in known_defaults:
                data["decidability"] = "default"
            elif name in strict_forward:
                data["decidability"] = "forward"
            elif name in forward:
                data["decidability"] = "acausal"
        return pool, forward, seeded

    def _select_active_relations(self, pool: list[Relation], forward: set[str], supplied: set[str]) -> tuple[list[Relation], set[str], set[str]]:
        """Partition unknowns by structural determinacy and select active relations.

        Deactivates relations touching undecidable (or previously unevaluable)
        variables (graph verdicts), writes the active relation set, the
        structural blocks and the partition diagnostics.  Returns ``(active,
        decidable, active_vars)``.
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
                    self._mark_relation_inactive(rel, "inactive_unevaluable: requires unevaluable " + ", ".join(unev), replace=True)
                else:
                    self._mark_relation_inactive(rel, "inactive_undecidable: cannot determine " + ", ".join(undec), replace=True)
            else:
                active.append(rel)
                self._mark_relation_active(rel)
        self.primary_relations = active
        self.relations = list(active)
        active_vars = {name for rel in active for name in rel.variables}
        for name in sorted(active_vars):
            self.track(name)
        # Complete the decidability classification with the partition verdicts:
        # determined-block members and (forced-)underdetermined variables.
        graph = self._structural_graph()
        for name in block_decidable:
            node = ("variable", name)
            if node in graph and graph.nodes[node].get("decidability") is None:
                graph.nodes[node]["decidability"] = "block"
        for name in undecidable:
            node = ("variable", name)
            if node in graph:
                graph.nodes[node]["decidability"] = "underdetermined"
        # Structural-partition locals captured for the lazy ``compiler_report``
        # property -- the only place these diagnostics are exposed.
        self._partition_diagnostics = {
            "determined_missing_variables": tuple(sorted(partition["determined_variables"])),
            "undecidable_variables": tuple(sorted(undecidable)),
            "deficiencies": partition["deficiencies"],
        }

        return active, decidable, active_vars

    def _select_cores_and_providers(
        self, active: list[Relation], active_vars: set[str], decidable: set[str], supplied: set[str], seeded: set[str]
    ) -> tuple[set[str], set[str]]:
        """Select packed block cores and the derived-variable providers.

        Mutates ``active_vars`` in place (shape-locked profiles activate their
        scalar-average control).  Returns ``(derived, cores)``: the variables
        recomputed by completion (including held constant defaults) and the
        packed unknowns with no seed entitlement.
        """
        produced = {out for rel in active for out in rel.output_names if not rel.implicit}
        cores = {
            name for name in active_vars
            if name in decidable
            and name not in supplied
            and name not in produced
            and name not in self.constant_default_values
            and name not in self.fixed
        } | {
            # Free-core defaults are packed unknowns seeded with their default,
            # never forward-derived, so an enforced relation (the balance) can
            # move them off the seed.  This wins over any relation that could
            # otherwise produce them (the redundant f_X = integral(n_X)/integral(n_i)),
            # which then acts as a closure residual instead of a producer.
            name for name in (seeded & active_vars)
            if name not in self.fixed and self.inputs.get(name) is None
        }
        known_cores = cores | set(self.constant_default_values)
        _, forward_decider = _forward_decision_rounds(active, supplied, extra_known=known_cores)
        derived: set[str] = set()
        for name in sorted(active_vars):
            if name in self.fixed or self.inputs.get(name) is not None or name in cores:
                continue
            selected = forward_decider.get(name)
            if selected is None or selected not in active:
                continue
            self._set_provider_edge(name, selected, "explicit")
            derived.add(name)
        # Constant defaults are held at their default value and never packed: they
        # are derived variables whose provider is the registry default itself.
        derived |= {name for name in self.constant_default_values if name in active_vars}
        # Shape-locked supplied profiles are reconstructed from their scalar
        # average (``average * shape``): the profile is a derived variable (never
        # packed as a full-profile DOF), while its average is the packed level
        # control.  Registered after the derived set is built above so the reset
        # at the top of this phase does not drop them.
        for name, (_shape, fixed) in self.supplied_profiles.items():
            if fixed:
                continue
            avg_name = self.profile_average_by_name[name]
            self.track(avg_name)
            active_vars.add(name)
            active_vars.add(avg_name)
            derived.add(name)
            derived.discard(avg_name)
            cores.discard(name)
        return derived, cores

    def _register_profile_generators(self, active_vars: set[str], derived: set[str]) -> None:
        """Register explicit lower-dimensional profile generators as providers,
        activating their scalar-average controls.  Mutates ``active_vars`` and
        ``derived`` in place."""
        for rel in list(self.relations):
            profile_outputs = [
                out for out in rel.output_names
                if out in self.variable_registry and self.variable_registry.get(out).shape == 1 and out != "rho"
            ]
            if not profile_outputs or rel.implicit:
                continue
            lower_dimensional = True
            for inp in rel.input_names:
                if inp not in self.variable_registry or self.variable_registry.get(inp).shape == 1:
                    lower_dimensional = False
                    break
            if not lower_dimensional:
                continue
            for out in rel.output_names:
                if out not in self.variable_registry or self.variable_registry.get(out).shape != 1:
                    continue
                if out in self.fixed:
                    continue
                avg_name = self.variable_registry.average_of(out)
                if avg_name is not None:
                    self.track(avg_name)
                    self.profile_average_by_name.setdefault(out, avg_name)
                    active_vars.add(avg_name)
                self._set_provider_edge(out, rel, "explicit")
                derived.add(out)

    def _append_guard_relations(self, active_vars: set[str]) -> None:
        """Append active relation/variable/system guards whose variables are
        all active, and build the relation-name indexes."""
        active_names = {rel.name for rel in self.relations}
        for rel in list(self.primary_relations):
            for guard in rel.constraint_relations:
                guard = self._resolve_relation_names(guard)
                if guard.name not in active_names and set(guard.variables) <= active_vars:
                    self.relations.append(guard)
                    active_names.add(guard.name)
        for name in sorted(active_vars):
            for guard in (*self.variable_registry.get(name).constraint_relations, *self._record_guards.get(name, ())):
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

        The seeding oracle (:func:`initial_values_from_graph`) runs once, after
        the first compile pass.  Its output is invariant across prune rounds: a
        relation the oracle successfully used ends with every one of its
        variables valued, so no later round can deactivate it (verified
        empirically across the reactor fixtures).  Each round then re-packs
        against the current roles -- the pack is the evaluability oracle, and
        it reports TWO ways a variable can fail:

        * no seed exists (``uninitialized_free_variables``) -- active,
          non-fixed, unsupplied, not a forward-derived output, not a determined
          block core, and nothing can even start it; or
        * it is a profile that IS seedable but has nothing determining it
          (``underdetermined_profiles``) -- no supplied data and no surviving
          producer, so its level sits wherever the seed put it.

        Both are recorded and every relation that references them is
        deactivated on the next round; this repeats to a fixpoint because
        removing a relation can orphan further variables.

        The second test used to be recorded and then ignored, which is how a
        stale decidability verdict reached the answer: a variable matched to a
        producer that is *later* deactivated keeps its "determined" verdict,
        because deactivation does not re-run the matching.  Only the pack can
        see the result, so the pack has to be the one to veto it.
        """
        self._unevaluable_names = set()
        self.initial_guesses = {}
        self.seed_provenance = {}
        # The tape encodes the current structure's seeding steps; a structural
        # recompile must re-record it rather than replay a stale one.
        self._seed_tape = None
        self._seed_tape_names = set()
        for round_no in range(max_rounds):
            self._run_compile_pass()
            if round_no == 0:
                self._refresh_seeds()
            self.pack()
            # Two evaluability failures, not one.  A variable with NO SEED
            # cannot be evaluated; so can a profile that is seedable but has
            # nothing determining it -- no supplied data, no producer left, so
            # its level sits wherever the seed put it.  Feeding only the first
            # in let the second survive to the answer: on Eos, `n_D` was matched
            # to "D density from ion density and D fraction", that relation was
            # then deactivated for needing an unevaluable `n_i`, and n_D's
            # verdict was never revisited -- so it stayed packed as a free
            # profile at its seed and its rho-average/peak consumers reported
            # statistics of a meaningless curve, while `success` stayed True.
            newly = set(self.uninitialized_free_variables) | set(self.underdetermined_profiles)
            if newly <= self._unevaluable_names:
                break
            self._unevaluable_names |= newly

    def _refresh_seeds(self) -> None:
        """Re-run the seeding oracle against the current input values.

        Shared by the first prune round and the fingerprint-hit path of
        :meth:`compile`: the oracle's x0 values depend on the numeric inputs,
        so they are recomputed even when the compiled structure is reused.
        When a seeding tape from a previous run with this structure exists,
        the steps are replayed directly (:func:`_replay_seed_tape`) instead of
        re-discovered; a replay that fails or seeds a different name set falls
        back to the full discovery run, which re-records the tape.  The
        compiler report embeds ``seed_provenance``, so its cache is
        invalidated here as well.
        """
        self.initial_guesses = {}
        self.seed_provenance = {}
        replayed = None
        if self._seed_tape is not None:
            try:
                replayed = _replay_seed_tape(self)
            except Exception:
                replayed = None
        if replayed is not None:
            seeds, provenance = replayed
        else:
            tape: list = []
            try:
                seeds, provenance = initial_values_from_graph(self, tape)
                self._seed_tape = tape
                self._seed_tape_names = set(provenance)
            except Exception:
                seeds, provenance = {}, {}
                self._seed_tape = None
                self._seed_tape_names = set()
        self.initial_guesses = dict(seeds)
        self.seed_provenance = dict(provenance)
        self._compiler_report_cache = None

    def _assign_roles(self, active_vars: set[str], derived: set[str], cores: set[str]) -> dict[str, str]:
        """Return the one role of every variable -- the compile verdict.

        ONE classification, answering both "how did this number arise?" and
        "what did the solver do with it?".  Two axes, plus the orthogonal
        :attr:`avg_to_profile` flag:

        =========  ============  =====================================
        origin     role          meaning
        =========  ============  =====================================
        --         ``inactive``  touched by no active relation
        input      ``fixed``     supplied and pinned; never moved
        input      ``movable``   supplied, may move within its tolerance
        computed   ``computed``  the equations determine it
        computed   ``assumed``   nothing determines it -- a registry
                                 constant, or a value left at its start
                                 because the system under-determines it
        =========  ============  =====================================

        So ``fixed``/``movable`` is real data, ``computed`` is physics, and
        ``assumed`` is the honest warning label.  ``avg_to_profile`` marks a
        value expanded from a scalar average with an assumed shape, and applies
        to inputs and computed values alike.

        The solver needs two finer distinctions that are *mechanics*, not
        origin, so they are booleans rather than roles:
        :attr:`packed_variables` (which values are solver unknowns) and
        :attr:`unseeded_variables` (packed unknowns with no seed entitlement,
        which start at the tolerance-floor magnitude rather than at a default).
        """
        self.packed_variables = set()
        self.unseeded_variables = set()
        roles: dict[str, str] = {}
        for name in self.known:
            if name not in active_vars:
                roles[name] = "inactive"
            elif name in self.fixed:
                roles[name] = "fixed"
            elif name in self.constant_default_values and name in derived:
                roles[name] = "assumed"
            elif name in cores:
                roles[name] = "computed"
                self.packed_variables.add(name)
                self.unseeded_variables.add(name)
            elif name in derived:
                roles[name] = "computed"
            else:
                # Packed: a supplied input the solver may move within tolerance,
                # or a free unknown started from a registry default.  The seed is
                # solver mechanics -- what the value IS depends on whether data
                # supplied it.
                self.packed_variables.add(name)
                roles[name] = "movable" if self.inputs.get(name) is not None else "computed"
        return roles

    @property
    def active_variable_names(self) -> set[str]:
        """Active variable names -- a derived view of :attr:`variable_roles`."""
        return {name for name, role in self.variable_roles.items() if role != "inactive"}

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

        On top of this immutable structure, every compile pass rewrites the
        per-pass *verdict* annotations (relation activation, inactivation
        reasons, provider edges) -- see the graph-verdict section below.

        Returns:
            The cached :class:`networkx.DiGraph` held on ``self._graph``.
        """
        cached = self._graph
        if cached is not None:
            return cached
        # The shared builder supplies the structure and the relation-node
        # annotations; the immutable per-variable ``shape`` is layered here.
        # The mutable per-pass annotations (supplied/fixed/decidability and
        # the relation/provider verdicts) are written only by
        # :meth:`_reset_graph_verdicts` and the compile phases.
        graph = relation_bipartite_graph(self.candidate_primary_relations)
        for node, data in graph.nodes(data=True):
            if data["kind"] == "variable":
                name = node[1]
                data["shape"] = self.variable_registry.get(name).shape if name in self.variable_registry else 0
        self._graph = graph
        return graph

    # ── Graph verdicts: activation, inactivation reasons, providers ───────
    #
    # The structural graph is the single durable store of the per-compile
    # verdicts.  Its structure (nodes/edges) is immutable -- built once from
    # the candidate relations -- and every compile pass rewrites only these
    # annotations:
    #
    # * relation nodes: ``active`` and ``inactive_reason``;
    # * relation -> variable edges: ``provider`` (``"explicit"`` recomputes
    #   its output in completion, ``"default"`` fills only a missing one);
    # * variable nodes: ``supplied``/``fixed``, refreshed because inputs can
    #   change between compiles of one system (popcon rewrites scan values).
    #
    # ``default_provider_by_output`` / ``derived_provider_by_output`` /
    # ``blocked_relation_reasons`` are read-only views of these annotations,
    # so the maps can never disagree with the graph.

    def _reset_graph_verdicts(self) -> None:
        """Reset all verdict annotations for a fresh compile pass."""
        graph = self._structural_graph()
        for node, data in graph.nodes(data=True):
            if data["kind"] == "relation":
                data["active"] = False
                data["inactive_reason"] = None
            else:
                name = node[1]
                data["supplied"] = self.inputs.get(name) is not None
                data["fixed"] = name in self.fixed
                data["decidability"] = None
        for _u, _v, edata in graph.edges(data=True):
            edata.pop("provider", None)
        self._provider_view = None

    def _mark_relation_active(self, rel: Relation) -> None:
        data = self._structural_graph().nodes[("relation", rel.name)]
        data["active"] = True
        data["inactive_reason"] = None

    def _mark_relation_inactive(self, rel: Relation, reason: str, *, replace: bool = False) -> None:
        """Record one relation's inactivation reason (first reason wins unless ``replace``)."""
        data = self._structural_graph().nodes[("relation", rel.name)]
        data["active"] = False
        if replace or not data.get("inactive_reason"):
            data["inactive_reason"] = reason

    def _set_provider_edge(self, name: str, rel: Relation, kind: str) -> None:
        """Select ``rel`` as the one ``kind`` provider of ``name``.

        Provider selection is single-writer per variable and kind: any
        competing ``kind`` edge into ``name`` is cleared (an explicit
        re-selection replaces the previous one; a default never displaces an
        explicit provider because the kinds are tracked separately).
        """
        graph = self._structural_graph()
        vnode = ("variable", name)
        for unode in graph.predecessors(vnode):
            edata = graph.edges[unode, vnode]
            if edata.get("provider") == kind:
                del edata["provider"]
        graph.edges[("relation", rel.name), vnode]["provider"] = kind
        self._provider_view = None

    def _provider_views(self) -> tuple[dict[str, Relation], dict[str, Relation]]:
        """Return the ``(default, explicit)`` provider maps from the edge verdicts."""
        cached = self._provider_view
        if cached is not None:
            return cached
        graph = self._structural_graph()
        defaults: dict[str, Relation] = {}
        explicit: dict[str, Relation] = {}
        for unode, vnode, kind in graph.edges(data="provider"):
            if kind is None:
                continue
            target = defaults if kind == "default" else explicit
            target[vnode[1]] = graph.nodes[unode]["relation"]
        view = (dict(sorted(defaults.items())), dict(sorted(explicit.items())))
        self._provider_view = view
        return view

    @property
    def default_provider_by_output(self) -> dict[str, Relation]:
        """Activated-default completion fallbacks -- a view of the provider edges."""
        return self._provider_views()[0]

    @property
    def derived_provider_by_output(self) -> dict[str, Relation]:
        """Explicit derived-variable providers -- a view of the provider edges."""
        return self._provider_views()[1]

    @property
    def blocked_relation_reasons(self) -> dict[str, str]:
        """Inactive relation -> reason -- a view of the relation-node verdicts."""
        graph = self._structural_graph()
        return {
            node[1]: data["inactive_reason"]
            for node, data in graph.nodes(data=True)
            if data.get("kind") == "relation" and data.get("inactive_reason")
        }

    def _classify_avg_to_profile(self) -> None:
        """Mark values expanded from a scalar average with an assumed shape.

        Orthogonal to the role: it is true of a computed profile built by a
        scalar->profile generator, and equally of an input profile supplied as a
        bare average.  Call after providers and packing are known.
        """
        reg = self.variable_registry
        self.avg_to_profile = set()
        for name, role in self.variable_roles.items():
            if role == "inactive" or name in COORDINATE_NAMES or name not in reg:
                continue
            if reg.get(name).shape != 1:
                continue
            producer = self.derived_provider_by_output.get(name) or self.default_provider_by_output.get(name)
            if producer is not None and all(
                reg.get(inp).shape == 0 for inp in producer.input_names if inp in reg
            ):
                self.avg_to_profile.add(name)

    def reported_roles(self) -> dict[str, str]:
        """Per-variable role -- the single origin classification.

        ``fixed``/``movable`` are real data, ``computed`` is physics, and
        ``assumed`` is the honest warning label (a registry constant, or a value
        the system under-determines and therefore left at its start).  See
        :meth:`_assign_roles`; :attr:`avg_to_profile` is the orthogonal flag.

        Deliberately non-propagating: each value reports its OWN origin, not the
        weakest of its ancestors -- propagating would collapse every quantity
        downstream of one assumed profile and carry no information.  A caller
        wanting effective upstream provenance can walk
        ``derived_provider_by_output`` over these roles on demand.
        """
        return {
            name: role
            for name, role in self.variable_roles.items()
            if role != "inactive" and name not in COORDINATE_NAMES
        }

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
        report = {
            "variable_roles": dict(sorted(self.variable_roles.items())),
            "supplied_variables": tuple(sorted(self.inputs)),
            "active_variables": tuple(sorted(name for name, role in self.variable_roles.items() if role != "inactive")),
            "derived_variables": tuple(sorted(name for name, role in self.variable_roles.items() if role in ("computed", "assumed") and name not in self.packed_variables)),
            "active_relations": tuple(rel.name for rel in self.primary_relations),
            "inactive_relations": dict(sorted(self.blocked_relation_reasons.items())),
            "default_provider_outputs": {name: rel.name for name, rel in sorted(self.default_provider_by_output.items())},
            "derived_provider_by_output": {name: rel.name for name, rel in self.derived_provider_by_output.items()},
            "unevaluable_variables": tuple(sorted(self._unevaluable_names)),
            # Profile free cores with no data anchor -- solved shape is an
            # arbitrary seed, not physics (S9).
            "underdetermined_profiles": tuple(sorted(self.underdetermined_profiles)),
            # Per-variable data-origin tag (D3): measured/defaulted/
            "avg_to_profile": tuple(sorted(self.avg_to_profile)),
            "structural_determinacy": {
                **self._partition_diagnostics,
                "blocks": tuple(self.structural_blocks),
            },
            # Structural decidability class per candidate variable -- the named
            # verdict of the default-activation gate and the DM partition.
            "decidability": dict(sorted(
                (node[1], data["decidability"])
                for node, data in self._structural_graph().nodes(data=True)
                if data.get("kind") == "variable" and data.get("decidability")
            )),
            # Per seeded variable, the oracle source of its x0 value.
            "seed_provenance": dict(sorted(self.seed_provenance.items())),
        }
        self._compiler_report_cache = report
        return report

    # ── Profile/average split helpers ─────────────────────────────────────

    def _profile_average(self, value: Any) -> float:
        """Return the approximate volume average of a profile-like value.

        Uses the shared ``volume_average`` helper over the canonical ``rho`` grid
        when it is available, otherwise falls back to the helper's arithmetic
        average behavior.  Scalars return themselves and empty profiles return
        zero.
        """
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return float(arr)
        if arr.size == 0:
            return 0.0
        rho_value = self.values.get("rho")
        if self.inputs.get("rho") is not None and rho_value is not None:
            return float(volume_average(arr, rho_value))
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
            self.track(name)
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
        col_of = {name: index for index, name in enumerate(unknowns)}
        name_of_col = {index: name for index, name in enumerate(unknowns)}
        n_cols = len(unknowns)

        # One row per scalar equation; inequalities determine nothing.  Relations
        # are adirectional, so outputs only count constraints here: one per
        # declared output, or one for an outputless equality.
        row_adj: list[list[int]] = []
        row_relation: list[str] = []
        for rel in relations:
            if not rel.outputs and rel.op != "==":
                continue
            cols = [col_of[name] for name in rel_vars(rel) if name in col_of]
            if not cols:
                continue
            scalar_rows = sum(1 for name in rel.output_names if name in self.variable_registry) if rel.output_names else 1
            for _ in range(max(1, scalar_rows)):
                row_adj.append(cols)
                row_relation.append(rel.name)

        # One scratch digraph carries the whole decomposition: ``c -> r``
        # incidence edges between column nodes ``("c", j)`` and scalar-equation
        # row nodes ``("r", i)``.  The maximum matching (Hopcroft-Karp) runs on
        # its undirected view; the matched ``r -> c`` edges are then added so
        # alternating reachability (columns via any row, rows only via their
        # matched column) is plain descendant reachability, and the
        # underdetermined subgraph's connected components are the deficiency
        # groups.  The Dulmage-Mendelsohn coarse partition and fine blocks are
        # invariant to *which* maximum matching is chosen.
        match_row = np.full(n_cols, -1, dtype=int)
        scratch = nx.DiGraph()
        scratch.add_nodes_from(("c", c) for c in range(n_cols))
        row_nodes = [("r", r) for r in range(len(row_adj))]
        scratch.add_nodes_from(row_nodes)
        for r, cols in enumerate(row_adj):
            for c in cols:
                scratch.add_edge(("c", c), ("r", r))
        if row_adj:
            # ``maximum_matching`` returns both directions; read the row side
            # and mirror it into the per-column ``match_row``.
            matching = nx.bipartite.maximum_matching(scratch.to_undirected(as_view=True), top_nodes=row_nodes)
            for r in range(len(row_adj)):
                partner = matching.get(("r", r))
                if partner is not None:
                    c = int(partner[1])
                    match_row[c] = r
                    scratch.add_edge(("r", r), ("c", c))
        reached: set[tuple[str, int]] = set()
        for c in range(n_cols):
            if match_row[c] < 0:
                reached.add(("c", c))
                reached |= nx.descendants(scratch, ("c", c))
        under_cols = {c for kind, c in reached if kind == "c"}
        under_rows = {r for kind, r in reached if kind == "r"}

        # Free parameters are forced underdetermined so the constraints that
        # reference them deactivate, instead of the matching inventing a value
        # for them and activating a meaningless balance (see
        # _free_parameter_names for the rule).
        for name in _free_parameter_names(relations):
            if name in col_of:
                under_cols.add(col_of[name])

        under_names = {name_of_col[c] for c in under_cols}
        result["determined_variables"] = set(unknowns) - under_names
        result["underdetermined_variables"] = under_names
        result["blocks"] = _structural_block_plan(row_adj, match_row, under_cols, name_of_col)

        # Group the underdetermined part into connected deficiencies on the
        # (column, row) incidence restricted to the underdetermined nodes;
        # each group needs (cols - rows) more supplied values among its
        # variables.
        under_nodes = {("c", c) for c in under_cols} | {("r", r) for r in under_rows}
        deficiencies: list[dict[str, Any]] = []
        for comp in nx.connected_components(scratch.subgraph(under_nodes).to_undirected(as_view=True)):
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

    def pack_scalar(
        self, name: str, init: float, lb: float, ub: float, *, scale_ref: Any
    ) -> tuple[float, float, float, float, str]:
        """Map one scalar to a solver coordinate ``(scale, offset, lower, upper, transform)``.

        Positive-bounded scalars pack logarithmically; others linearly with a
        tolerance/reference ``scale``.  The log-transform decision is purely
        structural/numerical -- scalar variable, positive solver lower bound,
        positive finite initial value; no variable-name or physics-category
        assumptions are used.
        """
        scale = self.spec_of(name).scale_of(*self.tols_of(name), scale_ref)
        if self.spec_of(name).shape == 0 and np.isfinite(lb) and lb > 0.0 and np.isfinite(init) and init > 0.0:
            lower = np.log(lb / init)
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
        self.uninitialized_free_variables = []
        self.underdetermined_profiles = []
        for name, role in sorted(self.variable_roles.items()):
            if role == "inactive" or name not in self.packed_variables:
                if role == "fixed" and self.inputs.get(name) is None:
                    raise ValueError(f"Fixed variable {name!r} has no value.")
                continue
            spec = self.variable_registry.get(name)
            lb, ub = spec.solver_bounds
            size = self.profile_size if spec.shape == 1 else 1
            # A profile packed as raw free elements with no data anchor is
            # under-determined pointwise: its shape is fixed by the seed, not by
            # physics.  Record it so the certificate never claims it as solved.
            if spec.shape == 1 and self.inputs.get(name) is None:
                self.underdetermined_profiles.append(name)
            start = len(lower)
            offsets: list[float] = []
            scales: list[float] = []
            span_transform: str | None = None
            try:
                initial_elements = [
                    float(self.initial_value(name, index=i if spec.shape == 1 else None))
                    for i in range(size)
                ]
            except Exception:
                self.uninitialized_free_variables.append(name)
                continue
            # Supplied-input elements (solver form) anchor movement scaling.
            ref_elements: np.ndarray | None = None
            if self.inputs.get(name) is not None:
                try:
                    ref_elements = np.asarray(spec.solver_value(self.inputs[name], self.profile_size), dtype=float).reshape(-1)
                except Exception:
                    ref_elements = None
            for i, init in enumerate(initial_elements):
                if ref_elements is not None and ref_elements.size:
                    ref: Any = float(ref_elements[min(i if spec.shape == 1 else 0, ref_elements.size - 1)])
                else:
                    ref = init
                scale, offset, lo, hi, transform = self.pack_scalar(name, init, lb, ub, scale_ref=ref)
                lower.append(lo)
                upper.append(hi)
                offsets.append(offset)
                scales.append(scale)
                if transform == "log":
                    span_transform = "log"
            specs.append((name, start, len(lower), np.asarray(offsets, dtype=float), np.asarray(scales, dtype=float), spec.shape, span_transform))
        self.packed_specs = specs
        self.packed_dim = len(lower)
        # A profile packed as raw free elements has no physics pinning its
        # level -- it sits where its start put it -- so its role is the
        # warning label, not "computed".  Only knowable here, once packing
        # has run.
        for name in self.underdetermined_profiles:
            if self.variable_roles.get(name) == "computed":
                self.variable_roles[name] = "assumed"
        self._classify_avg_to_profile()
        self._compiler_report_cache = None
        # Immutable input values are the base every solver vector is layered
        # onto; completion itself reads the compile-cached plan.
        self._packed_base_values = self.input_values()
        self._movement_plan = self._build_movement_plan()
        return np.zeros(self.packed_dim), np.asarray(lower), np.asarray(upper)

    def required_uninitialized_free_variables(self) -> list[str]:
        """Return uninitialized free variables required by enforced relations."""
        uninitialized = set(self.uninitialized_free_variables)
        if not uninitialized:
            return []
        required: list[str] = []
        for name in sorted(uninitialized):
            for rel in self.relations:
                if rel.enforce and name in rel.variables:
                    required.append(name)
                    break
        return required

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

    def _apply_completion_providers(self, out: dict[str, Any], plan: list | None = None) -> None:
        """Evaluate completion providers in dependency order, in place.

        The provider stage of :meth:`complete`.  Each provider relation of the
        compiled plan whose inputs are all present is evaluated and writes the
        outputs the plan resolved as writable: an explicit provider recomputes
        its output, a default (``only_missing``) fills only a still-missing
        one.  One pass is exact for an acyclic plan; a cyclic plan iterates
        until a pass changes nothing (value equality test) or the pass cap is
        reached.

        ``plan`` restricts the pass to a sublist of the compiled provider plan
        (same records, same order) -- the grouped-difference Jacobian re-runs
        only the providers downstream of a perturbed column group.
        """
        size = self.profile_size
        check_changes = self._completion_passes > 1
        for _pass in range(self._completion_passes):
            changed = False
            for rel, only_missing, input_names, outs in (self._provider_plan if plan is None else plan):
                # A provider can only fire once all of its inputs are known.
                if any(out.get(inp) is None for inp in input_names):
                    continue
                try:
                    # ``out`` is already a solver-form namespace here (this runs
                    # only from the solve-time and certification completion paths,
                    # both of which build solver-form values), so the relation is
                    # evaluated directly without the per-relation namespace copy.
                    mapped = rel.output_map(rel.evaluate(out))
                except Exception as exc:
                    # Skip the provider but keep the first real cause (S10a).
                    self.completion_errors.setdefault(rel.name, f"{type(exc).__name__}: {exc}")
                    continue
                for out_name, spec in outs:
                    old_missing = out.get(out_name) is None
                    # Defaults fill only a missing output; explicit providers recompute.
                    if only_missing and not old_missing:
                        continue
                    try:
                        value = spec.solver_value(mapped[out_name], size)
                    except Exception as exc:
                        self.completion_errors.setdefault(rel.name, f"{out_name}: {type(exc).__name__}: {exc}")
                        continue
                    old_value = out.get(out_name) if check_changes and not old_missing else None
                    out[out_name] = value
                    if not check_changes:
                        continue
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
            if check_changes and not changed:
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
            if fixed and self.inputs.get(name) is not None:
                fixed_value = self.solver_value(name, self.inputs[name])
            specs.append((name, self.profile_average_by_name.get(name), shape, fixed_value))
        return specs

    def apply_profile_specs(self, values: dict[str, Any]) -> None:
        """Reconstruct fixed/shape-controlled profiles in place.

        The profile stage of :meth:`complete`, reading the compile-cached
        ``self._profile_specs``.
        """
        for name, avg_name, shape, fixed_value in self._profile_specs:
            if fixed_value is not None:
                values[name] = fixed_value
                continue
            if avg_name is None or values.get(avg_name) is None:
                continue
            avg = float(np.asarray(values[avg_name], dtype=float).reshape(-1)[0])
            values[name] = self.solver_value(name, avg * shape)

    def _apply_constant_defaults(self, values: dict[str, Any]) -> None:
        """Fill still-missing held-constant defaults in place (solver form).

        The constant-default stage of :meth:`complete`, reading the
        compile-cached ``self._constant_defaults_solver``.
        """
        for name, value in self._constant_defaults_solver.items():
            if values.get(name) is None:
                values[name] = value

    def _value_map(self, *, use_input: bool, solver_form: bool) -> dict[str, Any]:
        """Build a value map from variable state; missing variables are omitted."""
        source = self.inputs if use_input else self.values
        if not solver_form:
            return {name: value for name, value in source.items() if value is not None}
        size = self.profile_size
        return {
            name: self.variable_registry.get(name).solver_value(value, size)
            for name, value in source.items()
            if value is not None
        }

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
        self.completion_errors = {}
        self.apply_profile_specs(values)
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

    def enforced_residual_block(self, rel: Relation, values: Mapping[str, Any]) -> tuple[np.ndarray, str | None]:
        """One enforced residual relation's rows for one solver namespace.

        Returns ``(rows, error)``; ``error`` names missing variables, in which
        case the rows are the single large placeholder residual.  Shared by
        :meth:`layout_relation_rows` and the grouped-difference Jacobian so
        both paths see byte-identical rows.
        """
        missing = [name for name in rel.variables if values.get(name) is None]
        if missing:
            return np.asarray([1.0e12]), f"Relation {rel.name!r} missing variables {missing}."
        # ``values`` is already in canonical solver form on every path that
        # reaches a residual evaluation (base inputs, unpacked spans and
        # completion outputs are all coerced).
        return self.relation_residual_vector(rel, values, safe=True), None

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
                    rel_status = self.relation_status_and_residual(rel, values)[0]
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
            rel_status, residual = self.relation_status_and_residual(rel, values)
            status[rel.name] = rel_status
            warnings.extend(rel_status.get("warnings", []))
            if rel.enforce:
                blocks.append(residual)
                if rel_status.get("errors"):
                    errors.extend(f"{rel.name}: {err}" for err in rel_status["errors"])
            elif not rel_status["verified"]:
                if is_default_relation(rel):
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

    def residual_layout(self, values: Mapping[str, Any], include_movement: bool = False) -> dict[str, Any]:
        """Freeze the residual-row layout on one probe namespace.

        The layout pins the exact rows of a stage objective -- enforced
        relation rows, domain rows, then movement rows -- so every residual
        and Jacobian evaluation of that stage produces the same vector size:
        a value that goes missing (or stops matching its slot) fills its own
        rows with the large penalty instead of changing the shape SciPy sees.
        :meth:`layout_relation_rows`, :meth:`layout_domain_rows`,
        :meth:`layout_movement_rows` and :meth:`build_jac_sparsity` all read
        exactly this one object, so their row orders align by construction.

        Returns:
            ``{"relation_dims", "domain_sel", "domain_sel_names",
            "domain_tail", "movement_names", "size"}``; ``movement_names`` is
            ``None`` when movement rows are excluded.
        """
        dims: list[int] = []
        for rel in self._enforced_residual_relations:
            if any(values.get(name) is None for name in rel.variables):
                dims.append(1)
                continue
            try:
                eval_values = self.relation_evaluation_values(rel, values)
                dims.append(max(1, int(self.relation_residual_vector(rel, eval_values, safe=True).size)))
            except Exception:
                dims.append(max(1, self.relation_row_dim(rel)))
        domain_sel, domain_sel_names, domain_tail = self._domain_layout(values)
        lo, hi, *_tols = self._domain_scalar_bounds
        domain_rows = (
            int(np.isfinite(lo[domain_sel]).sum()) + int(np.isfinite(hi[domain_sel]).sum())
            + sum(n_rows for _name, _j, n_rows in domain_tail)
        )
        movement_names = [name for name, _excess in self._movement_rows(values)] if include_movement else None
        return {
            "relation_dims": dims,
            "domain_sel": domain_sel,
            "domain_sel_names": domain_sel_names,
            "domain_tail": domain_tail,
            "movement_names": movement_names,
            "size": sum(dims) + domain_rows + len(movement_names or ()),
        }

    def layout_relation_rows(self, values: Mapping[str, Any], layout: Mapping[str, Any]) -> np.ndarray:
        """Enforced relation rows for one namespace at a frozen layout.

        A relation whose rows no longer match its layout span (evaluation
        failure, missing variables) fills that span with the large penalty.
        """
        blocks: list[np.ndarray] = []
        for rel, rdim in zip(self._enforced_residual_relations, layout["relation_dims"]):
            rows, _error = self.enforced_residual_block(rel, values)
            rows = np.asarray(rows, dtype=float).reshape(-1)
            blocks.append(rows if rows.size == rdim else np.full(rdim, 1.0e12, dtype=float))
        return np.concatenate(blocks) if blocks else np.empty(0, dtype=float)

    def layout_domain_rows(self, values: Mapping[str, Any], layout: Mapping[str, Any]) -> np.ndarray:
        """Domain rows for one namespace at a frozen layout."""
        return self._domain_rows(values, layout["domain_sel"], layout["domain_sel_names"], layout["domain_tail"])

    def layout_movement_rows(
        self,
        values: Mapping[str, Any],
        layout: Mapping[str, Any],
        weights: Mapping[str, float] | None = None,
        *,
        deadzone: bool = True,
    ) -> np.ndarray:
        """Movement rows for one namespace at a frozen layout.

        Each movement input contributes ``sqrt(weight * excess)``, so its
        squared cost is ``weight * excess`` -- a deadzone L1 penalty on the
        beyond-tolerance excess.  The per-input ``weight`` (default 1) is what
        the reconcile IRLS loop adjusts via :meth:`movement_weights`:
        down-weighting inputs already far past tolerance and up-weighting the
        marginal ones reweights the L1 so repeated solves drive the *number*
        of crossings down (the convex, iteratively-reweighted surrogate for
        the L0 "fewest inputs beyond tolerance" aim).  The weights are
        mode-owned state, passed in per call; references and tolerance widths
        come from the pack-time movement plan.  A movement input missing from
        ``values`` fills its row with the large penalty.
        """
        names = layout["movement_names"] or ()
        if not names:
            return np.empty(0, dtype=float)
        weights = weights or {}
        excess_by_name = dict(self._movement_rows(values, deadzone=deadzone))
        rows = np.empty(len(names), dtype=float)
        for i, name in enumerate(names):
            excess = excess_by_name.get(name)
            rows[i] = 1.0e12 if excess is None else np.sqrt(weights.get(name, 1.0) * excess)
        return rows

    def build_jac_sparsity(self, layout: Mapping[str, Any]):
        """Build conservative residual-variable sparsity for SciPy coloring.

        Reads the packed layout stored by the last :meth:`pack` call.  Rows
        follow ``layout`` (from :meth:`residual_layout`) exactly: enforced
        relation rows, the domain scalar batch (lower rows, then upper rows),
        the domain tail entries, then one row per movement input.
        """
        if not self.packed_specs:
            return None
        span_by_name = {name: (start, stop) for name, start, stop, *_rest in self.packed_specs}

        row_specs: list[tuple[int, set[str]]] = []
        for rel, rdim in zip(self._enforced_residual_relations, layout["relation_dims"]):
            names: set[str] = set()
            # Constants that are themselves produced variables are read off the
            # namespace exactly like inputs, so the rows depend on them too.
            for var_name in (*rel.variables, *(c for c in rel.constant_names if c in self.variable_registry)):
                names.update(self._sparsity_variable_names(var_name))
            row_specs.append((rdim, names))

        # Domain residual rows are part of the hard feasibility system.
        # Derived outputs may violate domains even though they are not packed
        # directly, so dependencies recurse through structural providers.
        sel = layout["domain_sel"]
        if sel.size:
            lo, hi, *_tols = self._domain_scalar_bounds
            sel_names = layout["domain_sel_names"]
            for bound in (lo[sel], hi[sel]):
                for j in np.nonzero(np.isfinite(bound))[0]:
                    row_specs.append((1, self._sparsity_variable_names(sel_names[int(j)])))
        for name, _j, n_rows in layout["domain_tail"]:
            if n_rows:
                row_specs.append((n_rows, self._sparsity_variable_names(name)))

        # One movement row per input.  A packed input's row depends on its own
        # span -- but completion may also overwrite a packed name as a provider
        # side output (ARC's n0 from the density-peaking relation), and a
        # supplied derived variable depends on the packed inputs that reach it,
        # so every movement row carries the full completion ancestry.
        # Conservative over-inclusion is always safe for differencing.
        for name in layout["movement_names"] or ():
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

    def jacobian_plan(self, layout: Mapping[str, Any]) -> dict[str, Any] | None:
        """Structural plan for reconcile's grouped-difference Jacobian.

        A finite-difference Jacobian normally pays one full residual call per
        column group, and each of those re-runs every completion provider even
        though a perturbed group only moves its own completion descendants.
        This plan precomputes, per group of structurally independent columns
        (the standard finite-difference coloring, from :meth:`build_jac_sparsity`):

        * ``cols`` -- the packed column indices perturbed together;
        * ``spans`` -- the packed-spec records of the perturbed spans;
        * ``deleted`` -- the non-packed variables whose values change (the
          spans' descendants in the completion dependency graph); they are
          dropped from the copied base namespace so fill-only defaults
          recompute them too;
        * ``providers`` -- the provider-plan sublist that recomputes them, in
          plan order;
        * ``relations`` -- indices into ``_enforced_residual_relations`` whose
          rows can move.

        Returns ``None`` when there is nothing to pack or no sparsity.
        """
        sparsity = self.build_jac_sparsity(layout)
        if sparsity is None or not self.packed_specs:
            return None
        csc = sparsity.tocsc()
        n = int(self.packed_dim)
        col_rows = [set(csc.indices[csc.indptr[j]:csc.indptr[j + 1]].tolist()) for j in range(n)]
        group_cols: list[list[int]] = []
        group_rows: list[set[int]] = []
        for j in range(n):
            for cols, rows in zip(group_cols, group_rows):
                if not (rows & col_rows[j]):
                    cols.append(j)
                    rows.update(col_rows[j])
                    break
            else:
                group_cols.append([j])
                group_rows.append(set(col_rows[j]))
        name_of_col: dict[int, str] = {}
        for record in self.packed_specs:
            for j in range(record[1], record[2]):
                name_of_col[j] = record[0]
        span_by_name = {record[0]: record for record in self.packed_specs}
        packed_names = set(span_by_name)
        graph = self._sparsity_dependency_graph()
        groups: list[dict[str, Any]] = []
        for cols in group_cols:
            span_names = {name_of_col[j] for j in cols}
            affected: set[str] = set()
            for name in span_names:
                affected.add(name)
                if name in graph:
                    affected |= nx.descendants(graph, name)
            groups.append(
                {
                    "cols": np.asarray(cols, dtype=int),
                    "spans": [span_by_name[name] for name in sorted(span_names)],
                    "deleted": sorted(affected - packed_names),
                    "providers": [
                        record
                        for record in self._provider_plan
                        if any(out_name in affected for out_name, _spec in record[3])
                    ],
                    "relations": [
                        index
                        for index, rel in enumerate(self._enforced_residual_relations)
                        if affected.intersection(rel.variables)
                        or any(c in affected for c in rel.constant_names if c in self.variable_registry)
                    ],
                }
            )
        return {"sparsity": csc, "groups": groups}

    def _domain_layout(self, values: Mapping[str, Any]) -> tuple[np.ndarray, list[str], list[tuple[str, int, int]]]:
        """Freeze the domain-row layout on one value map.

        Returns ``(sel, sel_names, tail)``: the scalar plan indices whose
        values are present floats (the scalar batch) with their names, and
        the ordered tail entries ``(name, j, n_rows)`` -- ``j`` indexes the
        profile batch, -1 marks the per-variable fallback for values of an
        unexpected type.  The layout fixes exactly which rows
        :meth:`_domain_rows` emits, so a stage's residual size cannot drift
        with the values.
        """
        sel: list[int] = []
        sel_names: list[str] = []
        tail: list[tuple[str, int, int]] = []
        size = self.profile_size
        lo, hi, *_tols = self._domain_profile_bounds
        for name, spec, rel_tol, abs_tol, k in self._domain_plan:
            value = values.get(name)
            if value is None:
                continue
            if k >= 0 and isinstance(value, float):
                sel.append(k)
                sel_names.append(name)
            elif (
                k < 0
                and isinstance(value, np.ndarray)
                and value.ndim == 1
                and value.shape[0] == size
                and value.dtype == np.float64
            ):
                j = self._domain_profile_index[name]
                tail.append((name, j, int(np.isfinite(lo[j])) + int(np.isfinite(hi[j]))))
            else:
                fallback = spec.domain_violation_rows(value, rel_tol, abs_tol)
                tail.append((name, -1, int(sum(rows.size for rows in fallback))))
        return np.asarray(sel, dtype=int), sel_names, tail

    def _domain_rows(self, values: Mapping[str, Any], sel: np.ndarray, sel_names: list[str], tail: list[tuple[str, int, int]]) -> np.ndarray:
        """Return the domain rows for one value map at a frozen layout.

        Emission order: the vectorized scalar batch (all lower-bound rows,
        then all upper-bound rows), then the tail entries -- profiles as one
        vectorized batch over the shared grid, unexpected types through the
        per-variable spec check.  A value that is missing (or no longer
        matches its layout slot) fills its rows with the large penalty, so
        the returned size always matches the layout.
        """
        rows: list[np.ndarray] = []
        if sel.size:
            lo, hi, rel_tols, abs_tols, floors = self._domain_scalar_bounds
            v = np.empty(sel.size, dtype=float)
            for i, name in enumerate(sel_names):
                value = values.get(name)
                v[i] = value if isinstance(value, float) else np.nan
            low_bound, high_bound = lo[sel], hi[sel]
            width = np.maximum(np.maximum(abs_tols[sel], rel_tols[sel] * np.maximum(np.abs(v), floors[sel])), 1.0e-300)
            low = np.maximum(low_bound - v, 0.0) / width
            high = np.maximum(v - high_bound, 0.0) / width
            bad = ~np.isfinite(v)
            if bad.any():
                low = np.where(bad, 1.0e12, low)
                high = np.where(bad, 1.0e12, high)
            rows.append(low[np.isfinite(low_bound)])
            rows.append(high[np.isfinite(high_bound)])
        # Profile batch: gather the well-shaped tail values, one matrix op.
        size = self.profile_size
        batch: list[tuple[int, np.ndarray, int]] = []
        for pos, (name, j, _n_rows) in enumerate(tail):
            if j < 0:
                continue
            value = values.get(name)
            if isinstance(value, np.ndarray) and value.ndim == 1 and value.shape[0] == size and value.dtype == np.float64:
                batch.append((pos, value, j))
        low_by_pos: dict[int, float] = {}
        high_by_pos: dict[int, float] = {}
        if batch:
            p_lo, p_hi, p_rel, p_abs, p_floor = self._domain_profile_bounds
            sel_j = np.asarray([j for _pos, _value, j in batch], dtype=int)
            matrix = np.stack([value for _pos, value, _j in batch])
            # Same arithmetic as VariableSpec.domain_violation_rows: tolerance
            # width from the per-point magnitude (floored), then the
            # worst-point violation per bound; a non-finite profile pins both.
            width = np.maximum(
                np.maximum(p_abs[sel_j][:, None], p_rel[sel_j][:, None] * np.maximum(np.abs(matrix), p_floor[sel_j][:, None])),
                1.0e-300,
            )
            low = np.max(np.maximum(p_lo[sel_j][:, None] - matrix, 0.0) / width, axis=1)
            high = np.max(np.maximum(matrix - p_hi[sel_j][:, None], 0.0) / width, axis=1)
            bad = ~np.all(np.isfinite(matrix), axis=1)
            if bad.any():
                low = np.where(bad, 1.0e12, low)
                high = np.where(bad, 1.0e12, high)
            for i, (pos, _value, _j) in enumerate(batch):
                low_by_pos[pos] = float(low[i])
                high_by_pos[pos] = float(high[i])
        p_lo, p_hi, *_ptols = self._domain_profile_bounds
        for pos, (name, j, n_rows) in enumerate(tail):
            if j >= 0 and pos in low_by_pos:
                entry = [row for row, bound in ((low_by_pos[pos], p_lo[j]), (high_by_pos[pos], p_hi[j])) if np.isfinite(bound)]
                rows.append(np.asarray(entry, dtype=float))
                continue
            if j < 0:
                value = values.get(name)
                if value is not None:
                    try:
                        fallback = self.spec_of(name).domain_violation_rows(value, *self.tols_of(name))
                        entry_rows = np.concatenate(fallback) if fallback else np.empty(0, dtype=float)
                    except Exception:
                        entry_rows = np.empty(0, dtype=float)
                    if entry_rows.size == n_rows:
                        rows.append(entry_rows)
                        continue
            # Missing value or a value that no longer matches its layout slot.
            rows.append(np.full(n_rows, 1.0e12, dtype=float))
        return np.concatenate(rows) if rows else np.empty(0, dtype=float)

    def _build_movement_plan(self) -> list[tuple[str, Any, float, bool, float | None]]:
        """Return movement records ``(name, reference, width, is_scalar, log_width)``.

        Movement inputs are the packed variables (from the layout stored by
        :meth:`pack`) with a supplied reference, plus the supplied variables
        that are derived from an explicit relation (e.g. a profile
        reconstructed from an average).  Movement anchors only the immutable
        supplied inputs, so each record's reference value and tolerance width
        are fixed for the whole solve and are resolved once here at pack time;
        the residual, the IRLS weights and the Jacobian-sparsity movement rows
        all iterate this one plan (:meth:`_movement_rows`) so their rows stay
        aligned.
        """
        plan: list[tuple[str, Any, float, bool, float | None]] = []
        packed: set[str] = set()
        for name, *_rest in self.packed_specs:
            packed.add(name)
            reference = self._packed_base_values.get(name)
            if reference is None and name in self.seeded_default_values:
                reference = self.solver_value(name, self.seeded_default_values[name])
            if reference is not None:
                plan.append(self._movement_record(name, reference))
        for name in self._movement_candidate_names:
            if name in packed:
                continue
            # A shape-locked (supplied, unfixed) profile carries no per-point
            # movement of its own: the level is controlled by its (packed)
            # scalar average, which already contributes the movement penalty.
            profile = self.supplied_profiles.get(name)
            if profile is not None and not profile[1]:
                continue
            ref_input = self.inputs.get(name)
            if ref_input is None:
                continue
            plan.append(self._movement_record(name, self.solver_value(name, ref_input)))
        return plan

    def _movement_record(self, name: str, reference: Any) -> tuple[str, Any, float, bool, float | None]:
        """Build one movement-plan record from a solver-form reference value.

        ``log_width`` is non-None for strictly-positive variables, whose
        movement is measured multiplicatively (see
        :attr:`VariableSpec.movement_is_multiplicative`).
        """
        spec = self.spec_of(name)
        rel_tol, abs_tol = self.tols_of(name)
        width = max(float(spec.tolerance_width(spec.scale_of(rel_tol, abs_tol, reference), rel_tol, abs_tol)), 1.0e-300)
        log_width = spec.movement_log_width(width, reference)
        if spec.shape == 0:
            return name, float(np.asarray(reference, dtype=float).reshape(-1)[0]), width, True, log_width
        return name, reference, width, False, log_width

    def _movement_rows(self, values: Mapping[str, Any], *, deadzone: bool = True):
        """Yield ``(name, excess)`` for every movement input present in ``values``.

        The scalar fast path computes the deadzone excess in plain float
        arithmetic (identical to :meth:`VariableSpec.movement_excess`); profile
        or unexpectedly-shaped values fall back to that spec method.

        ``deadzone=False`` (reconcile's ``exact`` option) drops the free
        tolerance band: movement is penalised from the first deviation, in
        units of the pack-time tolerance width.
        """
        for name, reference, width, is_scalar, log_width in self._movement_plan:
            current = values.get(name)
            if current is None:
                continue
            if not deadzone:
                cur = np.asarray(current, dtype=float)
                ref = np.asarray(reference, dtype=float)
                if log_width is not None and bool(np.all(cur > 0.0)):
                    yield name, float(np.max(np.abs(np.log(cur / ref)))) / log_width
                else:
                    yield name, float(np.max(np.abs(cur - ref))) / width
            elif is_scalar and isinstance(current, float):
                if log_width is not None and current > 0.0:
                    excess = abs(np.log(current / reference)) / log_width - 1.0
                else:
                    excess = abs(current - reference) / width - 1.0
                yield name, excess if excess > 0.0 else 0.0
            else:
                yield name, self.spec_of(name).movement_excess(current, reference, *self.tols_of(name))

    def movement_weights(self, values: Mapping[str, Any], *, eps: float, deadzone: bool = True) -> dict[str, float]:
        """Return movement L1 weights from the current solution (one IRLS step).

        ``weight = 1 / (excess + eps)`` per input: an input already well
        past tolerance gets a small weight (cheap to leave changed), while one
        only marginally out gets a large weight (strongly pushed back inside).
        Re-solving with these weights is the iteratively-reweighted-L1 update
        whose fixed point minimises the count of inputs beyond tolerance.  The
        caller owns the weights and passes them to :meth:`layout_movement_rows`.

        Args:
            values: Latest solved namespace.
            eps: Reweighting floor; smaller drives sparser (more aggressive)
                solutions at some cost to stability.
        """
        return {name: 1.0 / (excess + float(eps)) for name, excess in self._movement_rows(values, deadzone=deadzone)}

    # ── Store and final-value checks ──────────────────────────────────────

    def fixed_value_errors(self, values: Mapping[str, Any]) -> list[str]:
        """Return errors for fixed variables changed in a candidate value map."""
        errors: list[str] = []
        for name in sorted(self.fixed):
            ref = self.inputs.get(name)
            if ref is None or values.get(name) is None:
                continue
            try:
                old = np.asarray(self.solver_value(name, ref), dtype=float).reshape(-1)
                new = np.asarray(values[name], dtype=float).reshape(-1)
                atol = max(ZERO_TOL, 1e-10 * max(1.0, float(np.max(np.abs(old))) if old.size else 1.0))
                if old.shape != new.shape or not np.allclose(old, new, rtol=0.0, atol=atol):
                    errors.append(f"Fixed variable {name!r} changed during candidate solve.")
            except Exception as exc:
                errors.append(f"Could not validate fixed variable {name!r}: {exc}")
        return errors

    def domain_errors(self, values: Mapping[str, Any]) -> list[str]:
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
                # domain.  Do not call public_value here, because that may
                # project a violating value back onto a solver boundary and hide
                # an invalid reconciliation candidate.
                #
                # The boundary slack is a single GLOBAL constant, not the
                # variable's abs_tol.  It exists to absorb FLOATING-POINT NOISE
                # and nothing else: abs_tol is a physical tolerance, and using
                # it would have let P_brem_imp (abs_tol 1 MW) sit at -0.5 MW
                # unnoticed.  Anything that legitimately reaches a boundary is
                # fixed at the source instead -- P_brem_imp is now a sum over
                # impurity species rather than a difference of two totals, and
                # the peaking factors no longer claim a `>= 1` domain, because a
                # hollow profile is physical.
                if not value_in_domain(value, spec.domain, zero_tol=ZERO_TOL):
                    errors.append(f"Variable {name!r} violates domain {spec.domain!r}.")
            except Exception as exc:
                errors.append(f"Could not validate domain for variable {name!r}: {exc}")
        return errors

    def store(self, values: Mapping[str, Any]) -> None:
        """Overwrite current public values from a solver-domain value map.

        Inputs are not modified. Fixed variables keep their input-only state.
        """
        names = sorted((self.active_variable_names | set(values)) & self.known)
        for name in names:
            if name in self.fixed or values.get(name) is None:
                continue
            try:
                public = self.public_value(name, values[name])
                if not value_in_domain(public, self.spec_of(name).domain, zero_tol=0.0):
                    continue
            except Exception:
                continue
            self.values[name] = public
        # Keep profile-average controls consistent with the stored profiles.
        # A later system built from these stored values would otherwise invent
        # the missing average input itself, so re-running a mode on the solved
        # state would appear to create new values.
        for name in names:
            if self.spec_of(name).shape != 1 or name == "rho" or self.values.get(name) is None:
                continue
            avg_name = self.profile_average_by_name.get(name) or self.variable_registry.average_of(name)
            if avg_name is None or avg_name not in self.known:
                continue
            if avg_name in self.fixed or self.values.get(avg_name) is not None:
                continue
            try:
                average = self._profile_average(self.solver_value(name, self.values[name]))
                public = self.public_value(avg_name, average)
                if value_in_domain(public, self.spec_of(avg_name).domain, zero_tol=0.0):
                    self.values[avg_name] = public
            except Exception:
                continue

    # ── Per-variable delegates and small helpers ──────────────────────────

    def spec_of(self, name: str):
        """Registry spec for a canonical name -- the owner of its numerics."""
        return self.variable_registry.get(name)

    def tols_of(self, name: str) -> tuple[float, float]:
        """Resolved (rel_tol, abs_tol) for one name; spec defaults if untracked."""
        rel = self.rel_tols.get(name)
        if rel is not None:
            return rel, self.abs_tols[name]
        spec = self.spec_of(name)
        return float(spec.rel_tol or self.variable_registry.rel_tol_default), float(spec.abs_tol or 0.0)

    def solver_value(self, name: str, value: Any) -> Any:
        """Convert a public value to canonical solver shape (see :meth:`VariableSpec.solver_value`)."""
        return self.variable_registry.get(name).solver_value(value, self.profile_size)

    def relation_evaluation_values(self, rel: Relation, values: Mapping[str, Any]) -> dict[str, Any]:
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
        # verification; they are not algebraic projections.  ``_coerce_names``
        # is the precomputed input+constant tuple (disjoint), so the hot
        # per-evaluation path does no set construction.
        for name in rel._coerce_names:
            if name in out and out[name] is not None and name in self.variable_registry:
                out[name] = self.solver_value(name, out[name])
        return out

    def relation_residual_vector(self, rel: Relation, eval_values: Mapping[str, Any], *, safe: bool) -> np.ndarray:
        """Return one relation's scaled residual vector using system tolerances."""
        return rel.residual_vector(eval_values, scales=self.variable_scales, rel_tols=self.variable_tolerances, abs_tols=self.variable_abs_tolerances, safe=safe)

    def relation_status_and_residual(self, rel: Relation, eval_values: Mapping[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
        """Return one relation's verify status and residual vector from one evaluation."""
        return rel.status_and_residual(eval_values, scales=self.variable_scales, rel_tols=self.variable_tolerances, abs_tols=self.variable_abs_tolerances)

    def public_value(self, name: str, value: Any) -> Any:
        """Project solver values to public values (see :meth:`VariableSpec.public_value`)."""
        return self.variable_registry.get(name).public_value(value, self.profile_size)

    def refresh_scales(self) -> None:
        """Refresh variable scales and tolerances used by residuals.

        Domains and solver domains are admissible-value constraints, not
        numerical scales.  The finite scale floor comes from abs_tol / rel_tol,
        while current/reference magnitudes provide relative scaling.

        Also called post-solve in reconcile to rescale around stored values, so
        this stays a method rather than being inlined into the compile pass.
        """
        self.variable_tolerances = dict(self.rel_tols)
        self.variable_abs_tolerances = dict(self.abs_tols)
        # Scale each variable from its current/input reference magnitude (the
        # first of value/input with a finite element), defaulting to 0.0.
        self.variable_scales = {}
        for name in self.known:
            reference = 0.0
            for value in (self.values.get(name), self.inputs.get(name)):
                if value is None:
                    continue
                arr = np.asarray(value, dtype=float).reshape(-1)
                finite = arr[np.isfinite(arr)]
                if finite.size:
                    reference = float(np.max(np.abs(finite)))
                    break
            self.variable_scales[name] = self.spec_of(name).scale_of(self.rel_tols[name], self.abs_tols[name], reference)

    def initial_value(self, name: str, index: int | None = None) -> float:
        """Return an initial value for one variable element.

        Initial values may come only from user input or relation-generated guesses.
        Solver domains are constraints, not value providers.
        """
        spec = self.spec_of(name)
        size = self.profile_size
        # Relation-generated values are x0 hints, not movement references.  They
        # may override supplied non-fixed values for initialization only; the
        # original public value remains in the movement reference map.
        for candidate in (self.initial_guesses.get(name), self.inputs.get(name)):
            if candidate is None:
                continue
            solver_value = spec.solver_value(candidate, size)
            spec.check_solver_domain(solver_value, size)
            arr = np.asarray(solver_value, dtype=float).reshape(-1)
            if arr.size:
                return float(arr[min(index or 0, arr.size - 1)])

        # A block core (the free unknown of a determined block, for example V_p
        # inverted from the supplied P_fus) is not forward-reachable, so it has
        # no seed.  It is determined by the global solve against its block's
        # supplied anchor; the start here is only a numerical initial point, not
        # an invented physical value, so a determined block converges to the
        # same unique answer regardless.  The magnitude comes from the declared
        # registry ``nominal`` when present, else from the variable tolerance
        # scale, which the log transform then explores.
        if name in self.unseeded_variables:
            if spec.nominal is not None:
                arr = np.asarray(spec.solver_value(spec.nominal, size), dtype=float).reshape(-1)
                if arr.size:
                    value = float(arr[min(index or 0, arr.size - 1)])
                    lb, ub = spec.solver_bounds
                    if np.isfinite(lb):
                        value = max(value, float(lb))
                    if np.isfinite(ub):
                        value = min(value, float(ub))
                    return value
            value = float(spec.tolerance_floor(*self.tols_of(name)))
            lb, ub = spec.solver_bounds
            if np.isfinite(lb):
                value = max(value, float(lb))
            if np.isfinite(ub):
                value = min(value, float(ub))
            return value

        raise ValueError(
            f"No initial value for variable {name!r}: it was not supplied "
            "and was not generated by an active relation."
        )

    def relation_row_dim(self, rel: Relation) -> int:
        """Return the number of scalar comparison rows the relation produces.

        Output relations contribute one comparison per output dimension.
        Outputless residual relations contribute one row, vectorized over the
        profile grid when they touch profile variables.
        """
        if rel.output_names:
            return sum(self._variable_dim(name) for name in rel.output_names if name in self.variable_registry)
        return max([1, *(self._variable_dim(name) for name in rel.variables if name in self.variable_registry and self.variable_registry.get(name).shape == 1)])

    def _variable_dim(self, name: str) -> int:
        """Scalar-element count: 1 for scalars, the shared grid size for profiles."""
        return 1 if self.spec_of(name).shape != 1 else self.profile_size

    def _resolve_relation_names(self, rel: Relation) -> Relation:
        return canonicalize_relation_names(rel, self.variable_registry)

    def track(self, raw_name: str) -> str:
        """Track one registry-known variable name; returns the canonical name.

        Tracking a name resolves its tolerances once (spec defaults unless a
        record supplied overrides).  No object is created: specs own the
        numerics and the value dicts start empty.
        """
        if str(raw_name) not in self.variable_registry:
            raise ValueError(f"Relation requires unknown variable {str(raw_name)!r}.")
        spec = self.variable_registry.get(raw_name)
        name = spec.canonical_name
        if name not in self.known:
            self.known.add(name)
            self.rel_tols[name] = float(spec.rel_tol or self.variable_registry.rel_tol_default)
            self.abs_tols[name] = float(spec.abs_tol or 0.0)
        return name

# ── Batched completion: popcon grid namespaces ────────────────────────────
#
# The batched counterpart of :meth:`RelationSystem._apply_completion_providers`
# / :meth:`RelationSystem.complete`: one namespace holds every grid point at
# once and the compiled provider plan is replayed on it.  Kept here, next to
# the per-point completion loop it mirrors, so the two cannot drift apart
# unseen; they differ in exactly four deliberate ways:
#
#   * supplied/fixed inputs are never overwritten (the batch pins the
#     scenario; the per-point loop may overwrite a packed supplied name as a
#     provider side output),
#   * shape-controlled profiles are re-levelled from their scalar averages on
#     every pass (an average may itself be provider-derived from a scan axis),
#   * relations are evaluated batched-first with a per-relation trust verdict
#     and a point-by-point fallback (an implementation that broadcasts wrongly
#     poisons only its own points),
#   * the change test treats NaN as equal (a poisoned point must not keep the
#     pass loop spinning).
#
# Shape discipline: scalars are (N, 1), profiles are (N, P), the rho grid
# stays (P,).  Scalar x rho expressions inside relation code then broadcast
# to (N, P) exactly as scalar x rho broadcasts to (P,) in the per-point
# world, and profile reductions (trapezoid over the last axis) produce (N,)
# which the write-time coercion restores to (N, 1).


def coerce_batched(value: Any, shape: int, n: int, profile_size: int) -> np.ndarray | None:
    """Coerce one relation output to the batched layout, or ``None`` if it
    cannot be interpreted for the registry shape."""
    arr = np.asarray(value, dtype=float)
    if shape == 0:
        if arr.ndim == 0:
            return np.full((n, 1), float(arr))
        if arr.shape in ((n,), (n, 1)):
            return arr.reshape(n, 1)
        if arr.shape in ((1,), (1, 1)):
            return np.full((n, 1), float(arr.reshape(-1)[0]))
        return None
    if arr.ndim == 0:
        return np.full((n, profile_size), float(arr))
    if arr.shape == (profile_size,):
        return np.broadcast_to(arr, (n, profile_size)).copy()
    if arr.shape == (n, profile_size):
        return arr
    if arr.shape in ((n, 1), (n,)):
        return np.broadcast_to(arr.reshape(n, 1), (n, profile_size)).copy()
    return None


def slice_point(system: Any, ns: Mapping[str, np.ndarray], index: int, names: Any = None) -> dict[str, Any]:
    """Extract one grid point's per-point solver namespace from the batch.

    ``names`` restricts the slice to the variables actually needed (a single
    relation's inputs), avoiding a full ~200-entry namespace rebuild on every
    point of the point-by-point fallback -- the dominant cost of a large scan.
    """
    out: dict[str, Any] = {}
    items = ((name, ns.get(name)) for name in names) if names is not None else ns.items()
    for name, arr in items:
        if arr is None:
            continue
        if arr.ndim == 2 and arr.shape[1] == 1:
            out[name] = float(arr[index, 0])
        elif arr.ndim == 2:
            out[name] = np.ascontiguousarray(arr[index])
        elif arr.ndim == 1 and arr.shape[0] != 1:
            # Unbatched shared arrays (the rho grid) pass through whole.
            out[name] = arr
        else:
            out[name] = float(np.asarray(arr).reshape(-1)[0])
    return out


def _eval_pointwise(system: Any, rel: Any, ns: Mapping[str, np.ndarray], n: int, out_specs: Sequence[tuple[str, Any]]) -> dict[str, np.ndarray]:
    """Evaluate one relation point-by-point; failed points contribute NaN."""
    profile_size = system.profile_size
    needed = set(rel.variables) | set(getattr(rel, "constant_names", ()))
    collected: dict[str, list[Any]] = {name: [] for name, _spec in out_specs}
    with np.errstate(all="ignore"):
        for index in range(n):
            try:
                mapped = rel.output_map(rel.evaluate(slice_point(system, ns, index, needed)))
            except Exception:
                mapped = {}
            for name, _spec in out_specs:
                collected[name].append(mapped.get(name))
    out: dict[str, np.ndarray] = {}
    for name, spec in out_specs:
        if spec.shape == 0:
            out[name] = np.asarray(
                [np.nan if v is None else float(np.asarray(v, dtype=float).reshape(-1)[0]) for v in collected[name]],
                dtype=float,
            ).reshape(n, 1)
        else:
            rows = []
            for v in collected[name]:
                if v is None:
                    rows.append(np.full(profile_size, np.nan))
                else:
                    arr = np.asarray(v, dtype=float)
                    rows.append(np.broadcast_to(arr, (profile_size,)).astype(float) if arr.ndim == 0 else arr.reshape(-1)[:profile_size])
            out[name] = np.vstack(rows)
    return out


def _matches_point(batched: np.ndarray, reference: Any, index: int) -> bool:
    """Compare one batched row against a per-point recomputation."""
    if reference is None:
        return False
    ref = np.asarray(reference, dtype=float).reshape(-1)
    row = np.asarray(batched[index], dtype=float).reshape(-1)
    if ref.size == 1 and row.size >= 1:
        ref = np.broadcast_to(ref, row.shape)
    if ref.shape != row.shape:
        return False
    return bool(np.allclose(row, ref, rtol=1.0e-9, atol=0.0, equal_nan=True))


def relation_outputs(
    system: Any,
    rel: Any,
    ns: dict[str, np.ndarray],
    n: int,
    out_specs: Sequence[tuple[str, Any]],
    trust: dict[str, str],
) -> dict[str, np.ndarray]:
    """Batched outputs of one relation, coerced to the batched layout.

    Tries the vectorised call first (identical to the per-point completion
    call, just with array values).  The result is accepted only if every
    requested output coerces to the expected batched shape AND a two-point
    spot check (first and last grid point recomputed per-point) matches --
    this catches implementations whose internal reductions broadcast into
    outer products or that silently collapse the batch to one point's value.
    Anything else falls back to the point-by-point loop, where a failing
    point contributes NaN and poisons only itself.  The verdict is cached per
    relation for the scan (``trust``): broadcastability is a structural
    property of the implementation, so the spot check runs once, not once
    per solver iteration.
    """
    profile_size = system.profile_size
    verdict = trust.get(rel.name)
    if verdict == "pointwise":
        return _eval_pointwise(system, rel, ns, n, out_specs)
    with np.errstate(all="ignore"):
        try:
            mapped = rel.output_map(rel.evaluate(ns))
        except Exception:
            mapped = None
    if mapped is not None:
        coerced: dict[str, np.ndarray] = {}
        for name, spec in out_specs:
            arr = None if mapped.get(name) is None else coerce_batched(mapped[name], spec.shape, n, profile_size)
            if arr is None:
                coerced = {}
                break
            coerced[name] = arr
        if coerced:
            if verdict == "batched":
                return coerced
            # Spot-check on finite rows: a NaN row is a poisoned grid point
            # whose per-point recomputation may raise instead of matching, and
            # a broadcast defect shows on healthy rows just as well.
            finite = np.ones(n, dtype=bool)
            for name, _spec in out_specs:
                finite &= np.all(np.isfinite(coerced[name]), axis=-1)
            rows_to_check = np.flatnonzero(finite)
            if rows_to_check.size:
                checks = tuple(dict.fromkeys((int(rows_to_check[0]), int(rows_to_check[-1]))))
            else:
                checks = (0, n - 1) if n > 1 else (0,)
            reference: dict[int, dict[str, Any]] = {}
            with np.errstate(all="ignore"):
                for index in checks:
                    try:
                        reference[index] = rel.output_map(rel.evaluate(slice_point(system, ns, index)))
                    except Exception:
                        reference[index] = {}
            if all(
                _matches_point(coerced[name], reference[index].get(name), index)
                for name, _spec in out_specs
                for index in checks
            ):
                trust[rel.name] = "batched"
                return coerced
    trust[rel.name] = "pointwise"
    return _eval_pointwise(system, rel, ns, n, out_specs)


def apply_completion_providers_batched(
    system: Any, ns: dict[str, np.ndarray], n: int, trust: dict[str, str], dirty: set[str] | None = None
) -> set[str]:
    """Run the compiled completion plan on the batched namespace, in place.

    Mirrors :meth:`RelationSystem._apply_completion_providers` -- providers in
    dependency order, explicit providers recompute, defaults fill only
    missing outputs, cyclic plans iterate to a fixed point -- with the four
    deliberate differences listed in the section comment above.  Solver-domain
    projection is skipped here because certification re-checks domains per
    point.

    ``dirty`` names the variables whose values changed since the previous
    replay on this namespace (``None`` means treat everything as changed).
    A provider whose inputs -- declared and constant -- are all untouched and
    whose outputs are already present is skipped: relations are pure
    functions, so re-evaluating it could only rewrite the same values.  This
    is what makes the Gauss-Newton Jacobian affordable (perturbing one core
    re-evaluates only its downstream cone, not the whole plan).  Returns the
    set of variables this call actually changed, for chaining.
    """
    held = {name for name in system.inputs if system.inputs.get(name) is not None} | set(system.fixed)
    changed_names: set[str] = set()
    active: set[str] | None = None if dirty is None else set(dirty)
    for _pass in range(system._completion_passes):
        changed = False
        pass_changed: set[str] = set()
        # Re-level shape-controlled profiles from their (possibly provider-
        # updated) scalar averages, mirroring the profile stage of
        # RelationSystem.complete().  A supplied profile is level-free: its
        # shape is fixed but its level tracks its average, which may itself be
        # a scan axis or be derived from one (e.g. T_i_avg = T_e_avg), so it
        # must be rebuilt each pass rather than frozen at the compile midpoint.
        for name, avg_name, shape, fixed_value in system._profile_specs:
            if fixed_value is None and avg_name is not None and ns.get(avg_name) is not None:
                if active is not None and avg_name not in active and ns.get(name) is not None:
                    continue
                relevelled = ns[avg_name] * np.asarray(shape, dtype=float)
                if ns.get(name) is None or not np.array_equal(ns[name], relevelled):
                    ns[name] = relevelled
                    changed = True
                    pass_changed.add(name)
                    if active is not None:
                        active.add(name)
        for rel, only_missing, input_names, outs in system._provider_plan:
            if any(ns.get(name) is None for name in input_names):
                continue
            # Supplied inputs are the scenario and are held exactly: a
            # provider whose output is a held input never overwrites it and
            # becomes a pure consistency check at certification.
            writable = [
                (out_name, spec)
                for out_name, spec in outs
                if out_name not in held and not (only_missing and ns.get(out_name) is not None)
            ]
            if not writable:
                continue
            if (
                active is not None
                and all(ns.get(out_name) is not None for out_name, _spec in writable)
                and active.isdisjoint(input_names)
                and active.isdisjoint(getattr(rel, "constant_names", ()))
            ):
                continue
            outputs = relation_outputs(system, rel, ns, n, writable, trust)
            for out_name, _spec in writable:
                arr = outputs.get(out_name)
                if arr is None:
                    continue
                old = ns.get(out_name)
                ns[out_name] = arr
                if old is None or old.shape != arr.shape or not np.array_equal(old, arr, equal_nan=True):
                    changed = True
                    pass_changed.add(out_name)
                    if active is not None:
                        active.add(out_name)
        changed_names |= pass_changed
        if not changed:
            break
        # Later passes only need to revisit consumers of this pass's writes.
        active = set(pass_changed)
    return changed_names


# ── Seeding: initial solver values (the x0 oracle) and the small-block solver ─
#
# Module-level functions taking the compiled RelationSystem as their first
# argument (the same convention as fusdb.modes).  They read the compiled
# products (active relations, structural blocks, variable roles) and the
# per-variable helpers; nothing here writes Variable state -- the oracle fills
# a working solver-unit namespace only.  Two entry points are consumed
# elsewhere: initial_values_from_graph (compile-time pruning and reconcile x0)
# and solve_block (shared with ordered mode's simultaneous blocks, which opt
# in to profile-valued cores via allow_profile_core).


def initial_values_from_graph(system: "RelationSystem", tape: list | None = None) -> tuple[dict[str, Any], dict[str, str]]:
    """Fill solver start values by direct propagation from supplied values.

    Iteratively solves every relation that has exactly one missing variable
    (the 1x1 / acausal step), to a fixed point.  These are exact values, not
    movement references.  Variables that remain missing are the free unknowns
    of larger coupled blocks (block cores); they are packed directly and
    determined by the simultaneous reconcile against their block's supplied
    anchor, so no separate block solver is needed here.

    ``tape``, when given, records every successful seeding step in execution
    order -- ``("invert", rel, name)``, ``("forward", rel, names)``,
    ``("default", name)``, ``("block", names)`` -- so a later run with the
    same structure but new values can replay the steps directly
    (:func:`_replay_seed_tape`) instead of re-discovering them.

    Returns:
        ``(seeds, provenance)``: ``{name: value}`` for every variable the
        oracle seeded (supplied values are never included), and per seeded
        name the source that produced it -- ``"held_constant"``,
        ``"relation:<name>"``, ``"block"`` or ``"registry_default"``.
    """
    values = system.input_values()
    system.apply_profile_specs(values)
    original = set(values)
    seeded: dict[str, str] = {}
    # Constant defaults are known values from the start (they are held, not
    # solved), so downstream propagation can use them.
    for name, value in system.constant_default_values.items():
        if values.get(name) is None:
            try:
                values[name] = system.solver_value(name, value)
                seeded[name] = "held_constant"
            except Exception:
                pass
    # Propagate everything derivable from the supplied values.  The strict
    # passes seed a variable that has a declared producer only *forward*
    # through a producer: inverting an unrelated relation (a consistency
    # closure, or a producer solved backwards) at a provisional flat seed
    # plants the whole downstream chain in the wrong basin (measured: the
    # peaking closure seeded density_peaking = 1 from the flat profile, then
    # Angioni inverted to beta_T ~ 0.15 -- 10x -- and reconcile stayed there).
    _propagate_known(system, values, seeded, original, tape, strict=True)
    # Seed registry defaults for variables that supplied-propagation left
    # missing, then re-propagate so downstream values (n_X = n_i * f_X, ...)
    # fill in.  Defaults are pure x0 seeds -- never enforced -- applied to a
    # fixpoint so variable-reference defaults (T_i = T_e) resolve once their
    # source has a value.
    for _ in range(50):
        if not _seed_defaults(system, values, seeded, original, tape):
            break
        _propagate_known(system, values, seeded, original, tape, strict=True)
    # Fallback for cycles: a coupled loop (peaking -> profile -> pressure ->
    # beta -> peaking) has no all-forward entry, so allow ONE deferred
    # inversion (e.g. a flat profile from its average), then resume the
    # producer-first fixpoint so the cycle's own producers fill the rest
    # forward -- entering the cycle anywhere else (a closure pinning the
    # produced variable, a producer inverted backwards) seeds the wrong basin.
    # Each step seeds at least one variable, so this terminates.
    while _compute_direct_outputs(system, values, seeded, original, tape, strict=False, single=True):
        _propagate_known(system, values, seeded, original, tape, strict=True)
    return {name: values[name] for name in values if name in seeded}, seeded


def _replay_seed_tape(system: "RelationSystem") -> tuple[dict[str, Any], dict[str, str]] | None:
    """Replay the recorded seeding tape at the current input values.

    The tape (recorded by :func:`initial_values_from_graph`) freezes *which*
    relation seeds *which* variable by *which* method; only the numeric values
    change between runs with the same compile fingerprint.  Replay therefore
    skips the discovery passes -- crucially including every *failed* solve
    attempt discovery retries pass after pass -- and executes exactly the
    recorded steps with the same calls, so its seeds are bit-identical to a
    fresh discovery at the same values.  (A local root-finder warm-started
    from the previous point's seeds was measured 2026-07 and REJECTED: it
    saved ~7% of replay while making seeds history-dependent -- a nearby but
    different root shifts certified popcon values ~3%.)  Any step that fails
    -- or a replay that seeds a different name set than the recording --
    returns ``None`` so the caller falls back to the full oracle and
    re-records.
    """
    tape = system._seed_tape
    if tape is None:
        return None
    values = system.input_values()
    system.apply_profile_specs(values)
    original = set(values)
    seeded: dict[str, str] = {}
    for name, value in system.constant_default_values.items():
        if values.get(name) is None:
            try:
                values[name] = system.solver_value(name, value)
                seeded[name] = "held_constant"
            except Exception:
                pass
    default_plan = {name: (source, requires) for name, source, requires in system._default_seed_plan}
    size = system.profile_size
    for step in tape:
        kind = step[0]
        try:
            if kind == "default":
                name = step[1]
                if values.get(name) is not None:
                    continue
                source, requires = default_plan[name]
                if requires is not None and values.get(requires) is None:
                    return None
                raw = values.get(source) if isinstance(source, str) else source
                if raw is None:
                    return None
                value = system.solver_value(name, raw)
                if not system.spec_of(name).candidate_valid(value, size):
                    return None
                values[name] = value
                seeded[name] = "registry_default"
            elif kind == "forward":
                rel, names = step[1], step[2]
                mapped = rel.output_map(rel.evaluate(system.relation_evaluation_values(rel, values)))
                for name in names:
                    value = system.solver_value(name, mapped[name])
                    if not system.spec_of(name).candidate_valid(value, size):
                        return None
                    values[name] = value
                    seeded[name] = f"relation:{rel.name}"
            elif kind == "invert":
                rel, name = step[1], step[2]
                known = {vname: values[vname] for vname in rel.variables if vname != name}
                if any(value is None for value in known.values()):
                    return None
                raw = None
                # Fast canonical direction first, exactly like discovery.
                if name in rel.output_names and all(known.get(inp) is not None for inp in rel.input_names):
                    mapped = rel.output_map(rel.evaluate(system.relation_evaluation_values(rel, known)))
                    raw = mapped.get(name)
                if raw is None:
                    raw = rel(**known)
                value = system.solver_value(name, raw)
                if not system.spec_of(name).candidate_valid(value, size):
                    return None
                values[name] = value
                seeded[name] = f"relation:{rel.name}"
            else:  # "block"
                if not _compute_planned_block(system, step[1], values, seeded, original):
                    return None
        except Exception:
            return None
    if set(seeded) != system._seed_tape_names:
        return None
    return {name: values[name] for name in seeded}, seeded


def _propagate_known(system: "RelationSystem", values: dict[str, Any], seeded: dict[str, str], original: set[str], tape: list | None = None, strict: bool = False) -> None:
    """Fill values derivable from the currently known namespace.

    Stage 1 runs direct 1x1/acausal propagation to a fixed point; stage 2
    solves the determined blocks (2x2 ... N x N) for their cores, with a
    final merged-block sweep for variables left in no individual block.
    With ``strict`` a produced variable is only seeded forward through one of
    its declared producers (see :func:`initial_values_from_graph`).
    """
    # Stage 1: direct 1x1/acausal propagation to a fixed point.
    for _direct_pass in range(50):
        if not _compute_direct_outputs(system, values, seeded, original, tape, strict):
            break
    # Stage 2: solve the determined blocks (2x2 ... N x N) for their cores.
    progress = True
    while progress:
        progress = False
        for block in system.structural_blocks:
            if _compute_planned_block(system, block, values, seeded, original, tape):
                progress = True
                for _direct_pass in range(50):
                    if not _compute_direct_outputs(system, values, seeded, original, tape, strict):
                        break
    merged = tuple(
        name
        for block in system.structural_blocks
        for name in block
        if (name not in values or values[name] is None) and name not in original
    )
    if merged and _compute_planned_block(system, merged, values, seeded, original, tape):
        for _direct_pass in range(50):
            if not _compute_direct_outputs(system, values, seeded, original, tape, strict):
                break


def _seed_defaults(system: "RelationSystem", values: dict[str, Any], seeded: dict[str, str], original: set[str], tape: list | None = None) -> bool:
    """Seed still-missing variables from the compiled default plan.

    Seeds are pure initial points: a variable a relation determines is moved
    off its seed by the global solve, and a variable no enforced relation
    touches keeps it (zero-gradient).  Which variables are eligible and from
    what source is decided once at compile (``_default_seed_plan``); this
    checks only the runtime conditions -- still missing, gate variable
    available, copy-source available -- and skipped variable-reference seeds
    resolve on a later pass (the caller iterates to a fixpoint).
    """
    progress = False
    for name, source, requires in system._default_seed_plan:
        if values.get(name) is not None or name in original:
            continue
        if requires is not None and values.get(requires) is None:
            continue
        if isinstance(source, str):
            raw: Any = values.get(source)
            if raw is None:
                continue
        else:
            raw = source
        try:
            value = system.solver_value(name, raw)
            if not system.spec_of(name).candidate_valid(value, system.profile_size):
                continue
        except Exception:
            continue
        values[name] = value
        seeded[name] = "registry_default"
        if tape is not None:
            tape.append(("default", name))
        progress = True
    return progress


def _direct_relation_pool(system: "RelationSystem") -> list[Relation]:
    """Relations allowed for direct output initial computation.

    The global reconcile still uses ``system.relations``.  For initial guesses
    only, inactive weak/default providers may fill missing values when their
    inputs are already known.  This makes defaults useful as x0 generators
    without adding them as extra enforced residuals or movement references.
    """
    by_name = {rel.name: rel for rel in system.relations}
    for rel in system.candidate_primary_relations:
        if rel.name in by_name:
            continue
        if is_default_relation(rel):
            by_name[rel.name] = rel
    return list(by_name.values())


def _seed_accepts(system: "RelationSystem", name: str, original: set[str]) -> bool:
    """Return whether seeding may write a value for one variable.

    Seeding only fills genuinely missing degrees of freedom: it never
    overrides a user-supplied value (``original``) or a fixed variable, and
    it ignores names the registry does not know.

    Args:
        system: The compiled relation system.
        name: Candidate variable name to write.
        original: Names that already had a value before seeding began.

    Returns:
        ``True`` when ``name`` may be written by seeding.
    """
    if name not in system.variable_registry or name in original:
        return False
    return name not in system.fixed


def _compute_direct_outputs(system: "RelationSystem", values: dict[str, Any], seeded: dict[str, str], original: set[str], tape: list | None = None, strict: bool = False, single: bool = False) -> bool:
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
        system: The compiled relation system.
        values: Working solver-unit namespace, mutated in place.
        seeded: Names seeded so far, mutated in place.
        original: Names supplied before seeding (never overwritten).

    Returns:
        ``True`` if any value was filled this pass.
    """
    progress = False
    pool = _direct_relation_pool(system)
    # In strict mode a variable some relation declares as an output may only
    # be seeded through such a producer (forward, or the producer's own
    # 1-unknown solve); inversion through any other relation is deferred to
    # the caller's final non-strict pass.
    produced = {out for r in pool for out in r.output_names} if strict else frozenset()
    for rel in pool:
        # Primary path: a relation with exactly one unknown variable is
        # solved in whatever direction closes it (input or output).
        if not rel.implicit:
            unknown = [name for name in rel.variables if values.get(name) is None]
            if (
                len(unknown) == 1
                and _seed_accepts(system, unknown[0], original)
                and not (strict and unknown[0] in produced and unknown[0] not in rel.output_names)
            ):
                name = unknown[0]
                known = {vname: values[vname] for vname in rel.variables if vname != name}
                try:
                    value = system.solver_value(name, rel(**known))
                    if system.spec_of(name).candidate_valid(value, system.profile_size):
                        values[name] = value
                        seeded[name] = f"relation:{rel.name}"
                        if tape is not None:
                            tape.append(("invert", rel, name))
                        progress = True
                        if single:
                            return True
                        continue
                except Exception:
                    pass

        # Secondary path: every input is known, so any still-missing outputs
        # are each computable forward in one evaluation.
        if rel.output_names and all(not values.get(inp) is None for inp in rel.input_names):
            try:
                mapped = rel.output_map(rel.evaluate(system.relation_evaluation_values(rel, values)))
            except Exception:
                mapped = {}
            written: list[str] = []
            for name in rel.output_names:
                if name not in mapped or not values.get(name) is None or not _seed_accepts(system, name, original):
                    continue
                try:
                    value = system.solver_value(name, mapped[name])
                    if not system.spec_of(name).candidate_valid(value, system.profile_size):
                        continue
                except Exception:
                    continue
                values[name] = value
                seeded[name] = f"relation:{rel.name}"
                written.append(name)
                progress = True
            if written and tape is not None:
                tape.append(("forward", rel, tuple(written)))
            if written and single:
                return True
    return progress


def _compute_planned_block(
    system: "RelationSystem",
    block: tuple[str, ...],
    values: dict[str, Any],
    seeded: dict[str, str],
    original: set[str],
    tape: list | None = None,
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
        if name not in system.known or name in system.fixed:
            return False
    extended, rels = _block_closure(system, unknowns, values)
    if not rels:
        return False
    solved = solve_block(system, extended, rels, values, residual_tol=1.0)
    if solved is None:
        return False
    for name, value in solved.items():
        values[name] = value
        seeded[name] = "block"
    if tape is not None:
        tape.append(("block", block))
    return True


def _block_closure(system: "RelationSystem", unknowns: tuple[str, ...], values: Mapping[str, Any]) -> tuple[tuple[str, ...], list[Relation]]:
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
        for rel in system.relations:
            if rel.implicit or not rel.output_names:
                continue
            if not all(inp in available or inp in extended for inp in rel.input_names):
                continue
            for out in rel.output_names:
                if out in available or out in extended or out not in system.variable_registry:
                    continue
                if out in system.fixed:
                    continue
                extended.add(out)
                changed = True
    rels: list[Relation] = []
    for rel in system.relations:
        missing = {name for name in rel.variables if name not in available}
        if not missing or not missing <= extended:
            continue
        rels.append(rel)
    return tuple(sorted(extended)), rels


def solve_block(
    system: "RelationSystem",
    unknowns: tuple[str, ...],
    rels: list[Relation],
    values: Mapping[str, Any],
    *,
    residual_tol: float,
    allow_profile_core: bool = False,
) -> dict[str, Any] | None:
    """Solve one small initial-computation block; returns ``{name: value}`` or None.

    Unknowns that are declared outputs of a block relation are recomputed
    from that relation inside the residual, so the numerical search runs
    only over the remaining core unknowns.  Core starts come from
    supplied/current values when present, otherwise from the magnitudes
    of the known values around the block, refined by a coarse log-grid
    scan.  Solver domains constrain the search; they never provide a
    start on their own.

    The reconcile seeding path keeps the numerical core scalar so the search
    dimension never explodes pointwise (profiles are recomputed from their
    producers).  Ordered-mode blocks pass ``allow_profile_core=True``: a
    profile-valued core is then packed per element (log for positive-bounded
    elements, affine otherwise) and the coarse grid refinement is applied to
    the scalar core members only.
    """
    for name in unknowns:
        if name not in system.known or name in system.fixed:
            return None

    producers = _block_producers(system, unknowns, rels, values)
    core = [name for name in unknowns if name not in producers]
    if not core:
        # A fully produced cycle still needs one numerical degree of freedom.
        core = [unknowns[0]]
        producers.pop(unknowns[0], None)
    profile_core = {name for name in core if system.spec_of(name).shape == 1}
    if profile_core and not allow_profile_core:
        return None
    scalar_core = [name for name in core if name not in profile_core]
    core_dim = len(scalar_core) + len(profile_core) * system.profile_size
    enforced_rows = sum(max(1, system.relation_row_dim(rel)) for rel in rels if rel.enforce)
    if enforced_rows < core_dim:
        return None

    def namespace_from(core_values: Mapping[str, Any]) -> dict[str, Any]:
        ns = dict(values)
        ns.update(core_values)
        for produced, rel in producers.items():
            try:
                mapped = rel.output_map(rel.evaluate(system.relation_evaluation_values(rel, ns)))
                if mapped.get(produced) is not None:
                    ns[produced] = system.solver_value(produced, mapped[produced])
            except Exception:
                continue
        return system.complete(ns)

    def residual_from(core_values: Mapping[str, Any]) -> np.ndarray:
        ns = namespace_from(core_values)
        blocks: list[np.ndarray] = []
        for rel in rels:
            if not rel.enforce:
                continue
            if any(name not in ns or ns[name] is None for name in rel.variables):
                blocks.append(np.asarray([1.0e6], dtype=float))
                continue
            try:
                eval_values = system.relation_evaluation_values(rel, ns)
                blocks.append(system.relation_residual_vector(rel, eval_values, safe=True))
            except Exception:
                blocks.append(np.asarray([1.0e6], dtype=float))
        out = np.concatenate([block.reshape(-1) for block in blocks if block.size]) if blocks else np.empty(0, dtype=float)
        return np.nan_to_num(out, nan=1.0e6, posinf=1.0e6, neginf=-1.0e6)

    def score(core_values: Mapping[str, Any]) -> float:
        residual = residual_from(core_values)
        return float(np.max(np.abs(residual))) if residual.size else np.inf

    bounds_by_name: dict[str, tuple[float, float]] = {}
    starts: dict[str, Any] = {}
    for name in core:
        lb, ub = system.spec_of(name).solver_bounds
        bounds_by_name[name] = (lb, ub)
        size = system.profile_size if name in profile_core else 1
        elements = np.empty(size, dtype=float)
        for i in range(size):
            try:
                elements[i] = float(system.initial_value(name, index=i if name in profile_core else None))
            except Exception:
                start = _block_start_from_knowns(rels, values, lb, ub)
                if start is None:
                    return None
                elements[i] = start
        starts[name] = elements if name in profile_core else float(elements[0])

    def solve_from(current_starts: Mapping[str, Any]):
        """Solve from one set of starts, returning the solution plumbing."""
        offsets: list[float] = []
        scales: list[float] = []
        lower: list[float] = []
        upper: list[float] = []
        transforms: list[str] = []
        spans: list[tuple[str, int, int]] = []
        for name in core:
            lb, ub = bounds_by_name[name]
            elements = np.asarray(current_starts[name], dtype=float).reshape(-1)
            start_index = len(offsets)
            for element in elements:
                init = min(max(float(element), lb), ub) if np.isfinite(lb) or np.isfinite(ub) else float(element)
                scale, offset, lo, hi, transform = system.pack_scalar(name, init, lb, ub, scale_ref=init)
                offsets.append(offset)
                scales.append(scale)
                lower.append(lo)
                upper.append(hi)
                transforms.append(transform)
            spans.append((name, start_index, len(offsets)))

        def core_values_from(x: np.ndarray) -> dict[str, Any]:
            arr = np.asarray(x, dtype=float)
            out: dict[str, Any] = {}
            for name, start, stop in spans:
                elements = np.empty(stop - start, dtype=float)
                for j, idx in enumerate(range(start, stop)):
                    if transforms[idx] == "log":
                        elements[j] = offsets[idx] * np.exp(arr[idx])
                    else:
                        elements[j] = offsets[idx] + scales[idx] * arr[idx]
                out[name] = elements if name in profile_core else float(elements[0])
            return out

        def residual(x: np.ndarray) -> np.ndarray:
            return residual_from(core_values_from(x))

        x0 = np.zeros(core_dim, dtype=float)
        try:
            probe = residual(x0)
            if probe.size < core_dim:
                return None
            sol = least_squares(
                residual,
                x0,
                bounds=(np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)),
                method="trf",
                x_scale=np.ones_like(x0),
                max_nfev=200 if profile_core else 80,
                xtol=1e-10,
                ftol=1e-10,
                gtol=1e-10,
            )
            final_residual = residual(sol.x)
        except Exception:
            return None
        max_abs = float(np.max(np.abs(final_residual))) if final_residual.size else 0.0
        if not np.isfinite(max_abs) or max_abs > residual_tol:
            return None
        return sol, core_values_from

    # Most structural blocks are small and well scaled once variables are
    # packed.  Try the local solve first; a coordinate grid search is a fallback
    # for genuinely difficult nonlinear blocks.  Previously every cold block
    # ran the grid first, and each grid point replayed the entire completion
    # graph (about one hundred full graph evaluations even for the linear
    # R_min/R_max geometry block present in every reactor).
    solved_block = solve_from(starts)
    if solved_block is None and scalar_core:
        for _sweep in range(1 if len(scalar_core) == 1 else 2):
            for name in scalar_core:
                lb, ub = bounds_by_name[name]
                best, best_score = starts[name], score(starts)
                for point in signed_scalar_grid(lb, ub, decades=30, step=2):
                    point_score = score({**starts, name: point})
                    if point_score < best_score:
                        best, best_score = point, point_score
                starts[name] = best
        solved_block = solve_from(starts)
    if solved_block is None:
        return None
    sol, core_values_from = solved_block

    # Reject a genuinely flat core direction.  The local Jacobian handles the
    # common full-rank case without more graph replays; only a rank-deficient
    # result pays for the wider probes that distinguish a flat direction from a
    # nonlinear root with a locally zero derivative.
    jac = np.asarray(sol.jac, dtype=float)
    if jac.ndim != 2 or np.linalg.matrix_rank(jac) < core_dim:
        for name in scalar_core:
            lb, ub = bounds_by_name[name]
            grid = signed_scalar_grid(lb, ub, decades=30, step=2)
            if len(grid) < 3:
                continue
            probes = [score({**starts, name: point}) for point in (grid[0], grid[len(grid) // 2], grid[-1])]
            if max(probes) - min(probes) <= 1e-9 and min(probes) <= residual_tol:
                return None

    ns = namespace_from(core_values_from(sol.x))
    solved: dict[str, Any] = {}
    for name in unknowns:
        if name not in ns or ns[name] is None:
            return None
        value = ns[name]
        if not system.spec_of(name).candidate_valid(value, system.profile_size):
            return None
        solved[name] = system.solver_value(name, value)
    return solved


def _block_producers(system: "RelationSystem", unknowns: tuple[str, ...], rels: list[Relation], values: Mapping[str, Any]) -> dict[str, Relation]:
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
    ordered_rels = sorted(rels, key=lambda rel: not is_default_relation(rel))
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


def _block_start_from_knowns(rels: list[Relation], values: Mapping[str, Any], lb: float, ub: float) -> float | None:
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
