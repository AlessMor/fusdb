# RelationSystem

`RelationSystem` orchestrates solving, verification, diagnostics, and dense scan
evaluation for a set of `Relation` objects and `Variable` objects.

The public class set stays small: `Variable`, `Relation`, `Reactor`,
`RelationSystem`, and `Popcon`. `RelationSystem` keeps an indexed `SystemGraph`,
warning state, compiled constraint cache, and the latest solve result as durable state.

Profile integration is explicit in relation functions. `RelationSystem` does not
auto-integrate profile outputs for scalar variables.

**Core inputs**
- `relations`: list of `Relation`
- `variables`: list of `Variable`
- `mode`: `"overwrite"` or `"check"`
- `n_max`: max block size
- `max_passes`: max reconciliation attempts
- `default_rel_tol`: fallback relative tolerance
- `solving_order`: optional ordered relation names or relation-domain tags

**Canonical runtime state**
- `graph`: `SystemGraph` with indexed `Relation` and `Variable` object lists plus adjacency tables
- `ndim`: variable dimensionality lookup
- `variable_bounds`: simple scalar bounds inferred from `Variable.constraints`
- `compiled_constraints`: compiled constraint-expression cache
- `last_result`: latest `solve()` payload with `stop_reason`, `final_check`, `metrics`, and `violated_relations`

**Main methods**
- `variables_dict`: returns current `{name: Variable}`.
- `solve()`: fills missing values, reconciles violated relations, commits final state, and updates `last_result`.
- `evaluate(values, chunk_size=None)`: dense forward evaluation for scalar/grid inputs, used by POPCON.
- `diagnose(values_override=None)`: relation status, likely culprits, variable issues, and structural summary.

Plotting helpers live in `fusdb.plotting`; use
`fusdb.plotting.export_relation_graph(system, path)` for a lightweight HTML
graph of one relation system.

**Solve flow**
1. Seed explicit/default inputs.
2. Run closure passes to fill directly solvable missing values.
3. Solve structurally closed scalar blocks up to `n_max`.
4. Route overconstrained or multi-writer conflicts through reconciliation.
5. Run a final network verification and store the result in `last_result`.

**Profile-aware behavior**
Profiles are `Variable(ndim=1)` values. Scalar inputs to a profile variable are
broadcast to a flat profile. Explicit profile/average fallbacks such as `n_D`
and `n_D_avg` are handled by name, but scalar profile-dependent physics should
remain visible in relation functions through explicit integration.
