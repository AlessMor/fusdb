from pathlib import Path

path = Path('src/fusdb/relationsystem.py')
text = path.read_text()


def replace_between(src: str, start: str, end: str, replacement: str) -> str:
    a = src.index(start)
    b = src.index(end, a)
    return src[:a] + replacement + src[b:]

# CompilePlan becomes scenario-only: all declaration/model preparation is read
# from RelationSystem and no Variable/relation/constraint reconstruction occurs.
new_init = '''    def __init__(
        self,
        model: "RelationSystem",
        *,
        inputs: Mapping[str, Any] | None = None,
        fixed: Iterable[str] | None = None,
    ) -> None:
        self.model = model
        self.name = model.name
        self.variable_registry = model.variable_registry
        self.constraints_spec = model.constraints_spec
        self.candidate_primary_relations = list(model.candidate_primary_relations)
        self.system_constraint_relations = list(model.system_constraint_relations)
        self.rel_tols = dict(model.rel_tols)
        self.abs_tols = dict(model.abs_tols)
        self.known = set(model.known_names)
        self._record_guards = dict(model.record_guards)
        self.profile_size = model.profile_size
        self.profile_average_by_name = dict(model.profile_average_by_name)

        self.last_result: dict[str, Any] | None = None
        self.completion_errors: dict[str, str] = {}
        self._unevaluable_names: set[str] = set()
        self.packed_specs: list[tuple[str, int, int, np.ndarray, np.ndarray, int, str | None]] = []
        self.packed_dim = 0
        self._movement_plan: list[tuple[str, Any, float, bool, float | None]] = []
        self.seeded_default_values: dict[str, float] = {}

        self.inputs = model.scenario_inputs(inputs)
        self.values = {
            name: (value.copy() if isinstance(value, np.ndarray) else value)
            for name, value in self.inputs.items()
        }
        self.fixed = model.scenario_fixed(fixed)
        self._broadcast_profile_values()
        self._split_supplied_profiles()

        self.variable_roles: dict[str, str] = {}
        self.packed_variables: set[str] = set()
        self.unseeded_variables: set[str] = set()
        self.avg_to_profile: set[str] = set()
        self._seed_tape: list | None = None
        self._seed_tape_names: set[str] = set()
        self.initial_guesses: dict[str, Any] = {}
        self.seed_provenance: dict[str, str] = {}
        self.uninitialized_free_variables: list[str] = []
        self.underdetermined_profiles: list[str] = []
        self._graph: nx.DiGraph | None = None
        self._provider_view: tuple[dict[str, Relation], dict[str, Relation]] | None = None
        self._dependency_closure_cache: tuple[
            dict[str, frozenset[str]], dict[str, frozenset[str]]
        ] | None = None
        self._compiler_report_cache: dict[str, Any] | None = None
        self._completion_acyclic = False

        # A CompilePlan is compiled by definition. There is no mutable
        # "uncompiled plan" lifecycle and no public recompile operation.
        self._compile_with_pruning()

'''
text = replace_between(text, '    def __init__(\n        self,\n        variables: Iterable[Variable],', '\n\n    def _canonicalize_candidates', new_init)

# Model preparation is now owned by RelationSystem. Keep the old helper methods
# temporarily only where plan-local numerical profile normalisation still uses
# them; canonicalisation/materialisation are removed from the plan.
a = text.index('    def _canonicalize_candidates', text.index('class CompilePlan:'))
b = text.index('    def _broadcast_profile_values', a)
text = text[:a] + text[b:]

# CompilePlan is already compiled at construction. Remove the public compile
# method and its per-plan fingerprint lifecycle.
a = text.index('    def compile(self, *, force: bool = False)', text.index('class CompilePlan:'))
b = text.index('    def run(self, mode: str = "verify"', a)
text = text[:a] + text[b:]

# Remove stale assignments/references to _compile_fingerprint if present.
text = text.replace('        self._compile_fingerprint = None\n', '')
text = text.replace('        self._compile_fingerprint = fingerprint\n', '')

# Replace the thin RelationSystem with a prepared reusable model.
new_model = '''class RelationSystem:
    """Prepared reusable relation model.

    ``RelationSystem`` owns every invariant part of a prepared model: canonical
    declarations, tolerances, candidate relations, parsed constraints, profile
    grid metadata, the canonical bipartite topology, and reusable structural
    caches. :meth:`compile` overlays only numerical supplied values/fixedness and
    returns an independent :class:`CompilePlan`.
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
        self.variable_registry = VARIABLES
        self.constraints_spec = constraints
        self.variables = tuple(variables)

        records = list(self.variables)
        seen: set[str] = set()
        self.base_inputs: dict[str, Any] = {}
        self.base_fixed: frozenset[str]
        fixed_names: set[str] = set()
        self.rel_tols: dict[str, float] = {}
        self.abs_tols: dict[str, float] = {}
        self.record_guards: dict[str, tuple[Relation, ...]] = {}
        known: set[str] = set()
        explicit_sizes: set[int] = set()
        for rec in records:
            if rec.name in seen:
                raise ValueError(f"Duplicate variable {rec.name!r}.")
            seen.add(rec.name)
            known.add(rec.name)
            self.rel_tols[rec.name] = float(rec.rel_tol or VARIABLES.rel_tol_default)
            self.abs_tols[rec.name] = float(rec.abs_tol or 0.0)
            if rec.fixed:
                fixed_names.add(rec.name)
            if rec.input_value is not None:
                value = rec.value
                self.base_inputs[rec.name] = value.copy() if isinstance(value, np.ndarray) else value
            if rec.relations:
                self.record_guards[rec.name] = rec.relations
            if rec.spec.shape == 1:
                if rec.size is not None:
                    explicit_sizes.add(int(rec.size))
                elif isinstance(rec.input_value, np.ndarray) and rec.input_value.ndim == 1:
                    explicit_sizes.add(int(rec.input_value.shape[0]))

        if len(explicit_sizes) > 1:
            raise ValueError(f"Profile sizes are incompatible: {sorted(explicit_sizes)}.")
        self.profile_size = next(iter(explicit_sizes), VARIABLES.profile_size_default)

        self.candidate_primary_relations = tuple(
            canonicalize_relation(rel, self.variable_registry) for rel in relations
        )
        self.system_constraint_relations = tuple(
            canonicalize_relation_names(
                constraint_from_expression(
                    text,
                    name=f"system_constraint_{index}",
                    enforce=enforce,
                    source_kind="system",
                    source_name=self.name,
                ),
                self.variable_registry,
            )
            for index, (text, enforce) in enumerate(parse_constraint_specs(constraints))
        )

        for rel in (*self.candidate_primary_relations, *self.system_constraint_relations):
            for raw_name in (*rel.variables, *rel.constant_names):
                if raw_name not in self.variable_registry:
                    continue
                spec = self.variable_registry.get(raw_name)
                name = spec.canonical_name
                known.add(name)
                self.rel_tols.setdefault(name, float(spec.rel_tol or self.variable_registry.rel_tol_default))
                self.abs_tols.setdefault(name, float(spec.abs_tol or 0.0))

        # Profile-average companions are model metadata, not discoveries made
        # while compiling a scenario.
        self.profile_average_by_name: dict[str, str] = {}
        for name in tuple(sorted(known)):
            if name not in self.variable_registry or self.variable_registry.get(name).shape != 1:
                continue
            avg_name = self.variable_registry.average_of(name)
            if avg_name is None:
                continue
            known.add(avg_name)
            self.profile_average_by_name[name] = avg_name
            spec = self.variable_registry.get(avg_name)
            self.rel_tols.setdefault(avg_name, float(spec.rel_tol or self.variable_registry.rel_tol_default))
            self.abs_tols.setdefault(avg_name, float(spec.abs_tol or 0.0))

        uses_rho = any("rho" in rel.constant_names for rel in self.candidate_primary_relations)
        has_profile = any(
            name in self.variable_registry and self.variable_registry.get(name).shape == 1
            for name in known
        )
        if "rho" in self.variable_registry and (uses_rho or has_profile):
            known.add("rho")
            rho_spec = self.variable_registry.get("rho")
            self.rel_tols.setdefault("rho", float(rho_spec.rel_tol or self.variable_registry.rel_tol_default))
            self.abs_tols.setdefault("rho", float(rho_spec.abs_tol or 0.0))
            if self.base_inputs.get("rho") is None:
                self.base_inputs["rho"] = self.variable_registry.uniform_profile_grid(self.profile_size)
            fixed_names.add("rho")

        # Normalize declaration profile values once. Scenario overrides are
        # normalized against the same model grid by scenario_inputs().
        for name, value in tuple(self.base_inputs.items()):
            self.base_inputs[name] = self._normalize_input(name, value)

        self.base_fixed = frozenset(fixed_names)
        self.known_names = frozenset(known)
        self.relations_by_name = {
            rel.name: rel for rel in (*self.candidate_primary_relations, *self.system_constraint_relations)
        }
        self.relations_by_function = {
            rel.function_name: rel
            for rel in (*self.candidate_primary_relations, *self.system_constraint_relations)
            if rel.function_name
        }
        self.candidate_defaults_by_output: dict[str, tuple[Relation, ...]] = {}
        defaults: dict[str, list[Relation]] = {}
        for rel in self.candidate_primary_relations:
            if not is_default_relation(rel):
                continue
            for output in rel.output_names:
                defaults.setdefault(output, []).append(rel)
        self.candidate_defaults_by_output = {name: tuple(items) for name, items in defaults.items()}

        self._graph: nx.DiGraph | None = None
        # Structural caching is model-owned. The first implementation caches
        # immutable plan snapshots only after a successful compile; numeric
        # seeds/values never enter this cache.
        self._structure_cache: dict[tuple[frozenset[str], frozenset[str]], Any] = {}

    def _normalize_input(self, name: str, value: Any) -> Any:
        if value is None or name not in self.variable_registry:
            return value
        spec = self.variable_registry.get(name)
        if spec.shape != 1:
            return value
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return np.full(self.profile_size, float(arr))
        if arr.ndim != 1:
            raise ValueError(f"Profile {name!r} must be one-dimensional.")
        if arr.shape[0] != self.profile_size:
            raise ValueError(
                f"Profile {name!r} has length {arr.shape[0]}, expected {self.profile_size}."
            )
        return arr.copy()

    def scenario_inputs(self, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
        out = {
            name: (value.copy() if isinstance(value, np.ndarray) else value)
            for name, value in self.base_inputs.items()
        }
        if overrides is None:
            return out
        for raw_name, value in overrides.items():
            name = self.variable_registry.resolve(str(raw_name))
            if name not in self.known_names:
                raise ValueError(f"Unknown scenario variable {name!r} for this RelationSystem.")
            if value is None:
                out.pop(name, None)
            else:
                out[name] = self._normalize_input(name, value)
        # rho is model geometry and cannot disappear from a profile model.
        if "rho" in self.base_inputs:
            out["rho"] = self.base_inputs["rho"].copy()
        return out

    def scenario_fixed(self, fixed: Iterable[str] | None = None) -> set[str]:
        if fixed is None:
            return set(self.base_fixed)
        names = {self.variable_registry.resolve(str(name)) for name in fixed}
        unknown = names - set(self.known_names)
        if unknown:
            raise ValueError(f"Unknown fixed variable(s): {', '.join(sorted(unknown))}.")
        if "rho" in self.base_fixed:
            names.add("rho")
        return names

    @property
    def graph(self) -> nx.DiGraph:
        """The one durable relation/variable topology for this prepared model."""
        if self._graph is None:
            graph = relation_bipartite_graph(self.candidate_primary_relations)
            for node, data in graph.nodes(data=True):
                if data["kind"] == "variable":
                    name = node[1]
                    data["shape"] = (
                        self.variable_registry.get(name).shape
                        if name in self.variable_registry
                        else 0
                    )
            self._graph = graph
        return self._graph

    def compile(
        self,
        *,
        inputs: Mapping[str, Any] | None = None,
        fixed: Iterable[str] | None = None,
    ) -> CompilePlan:
        """Compile one independent executable scenario against this model."""
        return CompilePlan(self, inputs=inputs, fixed=fixed)
'''
text = replace_between(text, 'class RelationSystem:', '\n\n# ── Batched completion:', new_model)

# Update CompilePlan doc language now that there is no compile method.
text = text.replace('    - :meth:`compile` -- build/prune the active system (``run`` calls this).\n', '')
text = text.replace('        variables: Scenario variables.\n        relations: Post-filter relation candidates.\n        constraints: System-level constraints.\n        name: Diagnostic system name.\n', '        model: Prepared reusable relation model.\n        inputs: Optional complete scenario input overlay.\n        fixed: Optional replacement fixed-variable set.\n')

# POPCON may still mutate a plan internally today; keep an internal recompile
# hook rather than a public lifecycle API while its worker path is migrated.
insert = '''    def _recompile(self) -> "CompilePlan":
        """Internal compatibility hook for algorithms that intentionally mutate scenario state."""
        self._compile_with_pruning()
        return self

'''
marker = '    def run(self, mode: str = "verify"'
idx = text.index(marker, text.index('class CompilePlan:'))
text = text[:idx] + insert + text[idx:]

path.write_text(text)

# Migrate internal POPCON recompilation to the private hook. Public callers now
# obtain a new compiled plan from RelationSystem.compile().
pop = Path('src/fusdb/modes/popcon.py')
pop_text = pop.read_text().replace('        self.compile()\n', '        self._recompile()\n')
pop.write_text(pop_text)

# Strengthen architecture tests around model-owned preparation.
test = Path('tests/test_relation_system_compile_plan.py')
test.write_text('''from fusdb import CompilePlan, RelationSystem, Variable\nfrom fusdb.registry import RELATIONS\n\n\ndef _model():\n    return RelationSystem(\n        [Variable("R", 6.0), Variable("a", 2.0)],\n        [RELATIONS.get("Aspect ratio")],\n        name="compile_plan_contract",\n    )\n\n\ndef test_relation_system_owns_prepared_model_state():\n    model = _model()\n    assert model._graph is None\n    assert model.known_names >= {"R", "a", "A"}\n    assert model.base_inputs == {"R": 6.0, "a": 2.0}\n    assert model.candidate_primary_relations\n    assert not hasattr(model, "pack")\n    assert not hasattr(model, "run")\n\n    plan = model.compile()\n    assert isinstance(plan, CompilePlan)\n    assert plan.model is model\n    assert not hasattr(plan, "compile")\n    assert plan.candidate_primary_relations == list(model.candidate_primary_relations)\n    assert plan.profile_size == model.profile_size\n\n\ndef test_compile_plans_are_independent():\n    model = _model()\n    plan_a = model.compile(fixed={"R"})\n    plan_b = model.compile(inputs={"R": 9.0, "a": 3.0}, fixed={"R", "a"})\n\n    before = dict(plan_a.values)\n    plan_b.values["R"] = 10.0\n    assert plan_a.values == before\n    assert model.base_inputs["R"] == 6.0\n\n\ndef test_model_rejects_grid_changing_scenario_override():\n    model = RelationSystem(\n        [Variable("n_e", [1.0, 2.0, 3.0], size=3)],\n        [],\n    )\n    try:\n        model.compile(inputs={"n_e": [1.0, 2.0]})\n    except ValueError as exc:\n        assert "expected 3" in str(exc)\n    else:\n        raise AssertionError("grid-changing override should require a new RelationSystem")\n\n\ndef test_compile_plan_runs_without_model_rebuild():\n    model = _model()\n    relation_id = id(model.candidate_primary_relations[0])\n    plan = model.compile(fixed={"R", "a"})\n    result = plan.run("verify")\n    assert result["mode"] == "verify"\n    assert result["success"]\n    assert id(model.candidate_primary_relations[0]) == relation_id\n''')

print('applied RelationSystem model ownership refactor')
