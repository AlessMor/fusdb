from pathlib import Path

rs = Path('src/fusdb/relationsystem.py')
text = rs.read_text()
text = text.replace('_reset_graph_verdicts', '_reset_compile_verdicts')
text = text.replace('graph verdicts', 'compile verdicts')
text = text.replace('graph verdict', 'compile verdict')
text = text.replace('provider edge annotations', 'provider selections')
text = text.replace('view of the relation-node verdicts', 'view of the plan-local relation verdicts')
rs.write_text(text)

Path('docs/code_docs/relationsystem_class.md').write_text('''# RelationSystem and CompilePlan

`RelationSystem` is the reusable prepared relation model. It owns the parts of a
problem that do not change between numerical scenarios: canonical variable
metadata and tolerances, canonicalized candidate relations, parsed system
constraints, profile/grid metadata, the canonical `rho` grid, static relation
indexes, the one relation/variable bipartite graph, and the cache of reusable
structural compilation results.

`RelationSystem.compile(inputs=..., fixed=...)` overlays only scenario values
and fixedness and returns an independent `CompilePlan`. Changing the relation
selection, constraints, profile grid size, source-profile coordinate model,
variable tolerances/guards, or other prepared-model metadata requires building a
new `RelationSystem`.

## RelationSystem

Important model-owned state includes:

- `base_inputs` / `base_fixed`: declaration defaults used when a compile call
  does not override the scenario;
- `rel_tols` / `abs_tols` and `known_names`: resolved static variable metadata;
- `candidate_primary_relations` / `system_constraint_relations`: canonicalized
  once when the model is built;
- `profile_size`, the canonical `rho` input, and profile-average metadata;
- `graph`: the single durable relation topology. Compilation never annotates or
  copies this graph with scenario verdicts;
- the structural cache keyed by supplied/fixed names. A cache hit reuses only
  structural facts; values, seeds, packing, scales and execution products are
  refreshed for every plan.

A profile override must fit the model grid. A different profile size is a
different model and therefore requires a new `RelationSystem`.

## CompilePlan

A `CompilePlan` is one ephemeral executable scenario. It is compiled when
created; there is no public second `compile()` lifecycle. It owns:

- `inputs`, current/solved `values`, and `fixed`;
- supplied-profile shapes and scalar-average controls;
- active/inactive relation verdicts, decidability, providers, variable roles and
  structural blocks;
- initial guesses and seed provenance;
- the completion/provider schedule, residual/domain/movement plans, dependency
  closure and Jacobian sparsity products;
- `packed_specs` / `packed_dim` and the mutable solve state;
- `last_result` and completion diagnostics.

Execution modes (`verify`, `reconcile`, `optimize`, `ordered`, `popcon`) consume
a `CompilePlan`, not a `RelationSystem`. Multiple plans from one model may
coexist without mutating the model or each other.

The main numerical interface is `plan.run(mode, **options)`, together with
`pack()` / `unpack()`, value-map accessors, `complete()`, residual-layout and
certification helpers, Jacobian sparsity helpers, and `store()`.

## Compilation pipeline

Compilation has two separate stages:

1. structural compilation selects active relations, computes determinacy,
   providers, blocks, roles and packed-variable identities. Pruning may repeat
   this stage until it reaches a fixed point;
2. `_freeze_execution_plan()` builds value-dependent numerical products exactly
   once for the final structure: scales, profile specs, provider schedule,
   residual relations, domains, movement metadata and default-seed metadata.

A successful structural result can be cached on the model by the pair
`(supplied_names, fixed_names)`. On reuse, the new plan refreshes its numerical
seeds and validates packing. If the new numerical values make that candidate
unevaluable, compilation falls back to a full plan-local structural pass rather
than altering the shared cache.

POPCON workers reconstruct and retain one prepared `RelationSystem` per worker
recipe so chunks can reuse model preparation and structural cache entries.
POPCON itself deliberately retains its historical warm-start semantics by
recompiling its own ephemeral scan plan through a private hook after changing
scan-axis state; this is not a public plan lifecycle.

## Graph invariant

FusDB has one authoritative semantic relation topology per prepared model.
Structural analysis, dependency analysis, plotting and numerical sparsity are
projections or compiled products of `RelationSystem.graph`, never independently
maintained semantic graphs. Transient NetworkX/SciPy structures used inside an
algorithm remain implementation adapters and are discarded after use.
''')

resp = Path('docs/code_docs/responsibility_model.md')
r = resp.read_text()
r = r.replace(
'Values in `fusdb` pass through four layers — the registry, `Variable`,\n`RelationSystem` and `Reactor` —',
'Values in `fusdb` pass through five layers — the registry, `Variable`,\n`RelationSystem`, `CompilePlan` and `Reactor` —'
)
r = r.replace(
'| R4 | The scenario *declaration*: which names are supplied, with what canonical values, fixed flags, per-run tolerance overrides, profile size, record-local guards | per-run, **immutable once declared** (it defines the problem) | `Variable` (frozen), projected into `RelationSystem.inputs` / `fixed` / `rel_tols` |',
'| R4 | The prepared model: declaration defaults, resolved tolerances/guards, relation set, constraints, profile grid and canonical topology | reusable, immutable model definition | `RelationSystem` |'
)
r = r.replace(
'| R5 | Working/solved state evolving under a mode; must support snapshot/restore (popcon) and bulk overwrite | per-run, **mutable** | `RelationSystem.values` (written via `store()`) |',
'| R5 | One compiled scenario and its working/solved state | ephemeral, **mutable** | `CompilePlan` (`inputs`, `fixed`, `values`, providers, packing and execution products) |'
)
r = r.replace(
'| R6 | Provenance: how each value came to be (supplied / default-seeded / relation-derived / block-solved / moved-beyond-tolerance) | per-run, derived facts | `RelationSystem.variable_roles`, `seed_provenance`, graph decidability annotations, `inputs_beyond_tolerance`, `SolvedColumn` |',
'| R6 | Provenance: how each value came to be (supplied / default-seeded / relation-derived / block-solved / moved-beyond-tolerance) | per-run, derived facts | `CompilePlan.variable_roles`, `seed_provenance`, plan-local decidability/provider verdicts, `inputs_beyond_tolerance`, `SolvedColumn` |'
)
r = r.replace('Nothing writes a solved value back onto it. `RelationSystem.values` is the only\nmutable per-run state,', 'Nothing writes a solved value back onto it. `CompilePlan.values` is the mutable\nper-run state,')
r = r.replace('the compile fingerprint and seed tape can safely key off it.', 'the prepared `RelationSystem` can safely reuse structural results across plans with the same supplied/fixed signature.')
r = r.replace('writes system dicts,', 'writes plan value dicts,')
r = r.replace('provenance in the system', 'provenance in the plan')
r = r.replace('the actors live on the\nsystem;', 'the actors live on the\ncompiled plan;')
resp.write_text(r)

# Add an explicit no-structural-pass-on-cache-hit contract.
test = Path('tests/test_relation_system_compile_plan.py')
t = test.read_text()
if 'def test_cache_hit_skips_structural_compile_pass' not in t:
    t += '''\n\ndef test_cache_hit_skips_structural_compile_pass(monkeypatch):\n    model = _model()\n    model.compile(inputs={"R": 6.0, "a": 2.0}, fixed={"R"})\n    calls = 0\n    original = CompilePlan._compile_structure_pass\n\n    def counted(self):\n        nonlocal calls\n        calls += 1\n        return original(self)\n\n    monkeypatch.setattr(CompilePlan, "_compile_structure_pass", counted)\n    plan = model.compile(inputs={"R": 7.0, "a": 2.5}, fixed={"R"})\n    assert plan._structure_cache_hit\n    assert calls == 0\n'''
test.write_text(t)

print('applied final RelationSystem cleanup and documentation')
