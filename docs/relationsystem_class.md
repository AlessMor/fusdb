# RelationSystem

RelationSystem solves a set of `Relation` objects against `Variable` objects using a dict-based bipartite graph.

**Core inputs (dataclass fields)**
- `relations`: list of `Relation`
- `variables`: list of `Variable`
- `mode`: `"overwrite"` or `"check"`
- `verbose`: bool
- `n_max`: max block size (default 4)
- `max_passes`: max overwrite passes
- `default_rel_tol`: fallback tolerance for variables without overrides

**Example**
```python
from fusdb import RelationSystem, Variable, Relation

rel = Relation(
    name="Aspect ratio",
    output="A",
    func=lambda R, a: R / a,
    inputs=["R", "a"],
    tags=("geometry",),
)
system = RelationSystem(
    relations=[rel],
    variables=[
        Variable(name="R", values=[3.0], input_source="explicit"),
        Variable(name="a", values=[1.0], input_source="explicit"),
    ],
    mode="overwrite",
)
system.solve()
```

**Internal maps (examples)**
- `_vars = {"R": Variable(...), "a": Variable(...), "P_fus": None}`
- `_vars_to_rels = {"R": {relA, relB}, "a": {relC}, ...}`
- `_rels_to_vars = {relA: ("R", "a", "B0"), ...}`
- `_var_order = ["R", "a", "B0", ...]`
- `_var_constraints_map = {"R": ("R > 0",), ...}`
- `_rel_constraints_map = {relA: ("a > 0",), ...}`

**Methods (concise, all)**
- `__post_init__()` builds maps, constraints, logger, and pending relations.
- `variables_dict` property returns `{name: Variable}`.
- `_get_value(name)` returns the effective value considering pass/override logic.
- `_values_dict()` returns the current values dict.
- `_accept_candidate_values(...)` validates and commits candidate values.
- `_set_value(...)` writes a value and updates pending relations/metadata.
- `_expected(rel, values)` computes expected output (or None).
- `_residual(rel, values, scaled=False)` computes `actual - expected`.
- `_residual_derivative(rel, name, values, current=None)` finite-difference derivative.
- `_candidate_better(key, best_key, tol=...)` compares candidate scores.
- `_constraints_violated(values, rel=None, names=None)` checks constraints.
- `_solve_for_value(rel, name, values_map, prefer_eval_output=False)` solves a single var.
- `_apply_relation(rel, rel_values, missing_inputs, mode_overwrite=...)` forward/backward apply.
- `_infer_var_bounds(name)` infers simple numeric bounds from constraints.
- `_constraint_residuals(constraints, values, penalty)` returns constraint penalties.
- `_least_squares_block_compact(relations, unknowns, values_map)` nxn LSQ solver.
- `_solve_block(relations, unknowns, values_map)` picks 1×1 vs nxn path.
- `_build_unknown_map(rel_nodes, values)` groups relations by unknown set.
- `_solve_unknown_blocks(unknown_map, values, mode_overwrite=...)` solves blocks up to `n_max`.
- `_enforce_pending_relations(rel_index, mode_overwrite=...)` processes the pending queue.
- `_start_pass()` seeds overrides and resets pending for a new pass.
- `_select_culprit(rels, values, rel_nodes)` chooses an override candidate.
- `solve()` runs the main loop.
- `_violated_relations(values, rels=None)` returns violated relations.
- `_relation_status(rel, values)` returns (status, residual).
- `_culprit_for_relation(rel, values)` finds a likely culprit variable.
- `diagnose_relations(values_override=None, return_culprits=False)` diagnostics.
- `diagnose_variables()` diagnostics.
- `export_relation_graph(path="relation_graph.html")` writes an HTML graph.

**Solve flow (detailed)**
1. If `mode == "check"`, exit early (no mutation).
2. Seed the pending queue with all relations.
3. Loop passes:
4. Enforce pending relations via `_enforce_pending_relations` (forward eval; backward solve if exactly one input is missing).
5. Update violated relations based on current values.
6. Build unknown blocks and try to solve 1×1..`n_max` with `_solve_unknown_blocks` (1×1 uses `solve_for_value`; nxn uses compact LSQ with bounds and penalties).
7. If progress was made, continue; if not and violations exist in overwrite mode, pick a culprit and override, then start a new pass.
8. Exit when no progress remains (or after `max_passes`).

**Outputs and diagnostics**
- `variables_dict` → `{name: Variable}` (built from `_vars` + `_var_order`)
- `diagnose_relations()` → list of `(relation, status, residual)` (optional culprits)
- `diagnose_variables()` → list of `(variable, status, rank)`
