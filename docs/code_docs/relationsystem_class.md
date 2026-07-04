# RelationSystem

`RelationSystem` compiles a selected set of `Relation` objects and a set of
`Variable` objects into one numeric system. Execution modes (`fusdb.modes`)
drive a compiled system through its public interface and own their own
algorithm and result shape; the SciPy backend uses
`scipy.optimize.least_squares` for reconcile/optimize solves.

**Core inputs**
- `variables`: an iterable of `Variable` instances (the system's variables)
- `relations`: an iterable of `Relation` instances (selected relations to evaluate)
- `constraints`: optional extra system-level constraint expressions
- `name`: optional system name

**Canonical runtime state** (plain dicts — the system holds no `Variable`
objects; per-variable numerics live on the process-lifetime `VariableSpec`)
- `inputs` / `values`: supplied and current values, `{name: value}` in
  canonical units; `fixed`: the fixed-name set; `rel_tols` / `abs_tols`:
  resolved per-name tolerances; `known`: every tracked name
- `variable_roles`: the compile verdict — exactly one solve role per variable
  (`inactive` / `fixed` / `held` / `derived` / `core` / `packed`); packing,
  completion, movement and reporting all switch on it
- `relations`: full list of active `Relation` objects (including relation-local guards)
- `primary_relations`: the active relations selected by compilation
- `relations_by_name`: mapping `{name: Relation}`
- `profile_size`: inferred common profile length used for profile variables
- `packed_specs` / `packed_dim`: the packed free-variable layout written by
  `pack()` — one `(name, start, stop, offsets, scales, shape, transform)`
  record per packed variable

**Main methods**
- `run(mode='verify', **options)`: compile, then dispatch to one of the
  available modes (`verify`, `reconcile`, `optimize`, `ordered`) and return a
  result dict. `verify()`/`reconcile()`/`optimize()`/`ordered()` are shortcuts.
- `compile()`: build/prune the active system (`run` calls this first).
- `pack()` / `unpack(x)`: free variables <-> solver vector. `pack()` returns
  `(x0, lower, upper)` (with `x0` all zeros — the layout's offsets/scales
  absorb the start values) and stores the packed layout on `packed_specs`.
- `solver_values()` / `input_values()` / `public_values()`: named value-map
  accessors (current solver-form, immutable solver-form inputs, current
  public-form). `complete(values)` is the single completion path: it closes a
  namespace in place (profiles → constant defaults → providers) using the
  plan frozen at compile time.
- `solver_residual_vector(values)` / `domain_residuals(values)` /
  `movement_residuals(values, reference, weights)`: the residual blocks; modes
  weight and stack them. `certify_relations(values)` builds the full
  per-relation certification statuses. IRLS movement weights are mode-owned
  and produced by `movement_weights(values, reference, eps=...)`.
- `build_jac_sparsity(reference=None)`: conservative Jacobian sparsity for the
  current packed layout.
- `store(values)`: write solved values back into the variables.
- `initial_values_from_graph()`: the seeding oracle (direct propagation plus
  the small structural block solver) used to build solver start values.

Per-variable numerics (solver/public value conversion, shape coercion, domain
checks, scales and tolerances) are owned by the frozen `VariableSpec`
(computed once per process, with precomputed bounds/projection constants);
the system passes its `profile_size` and resolved tolerances as arguments.
`Variable` is a boundary input record only (see
[Variable Class](variable_class.md)).

The result dictionaries returned by `run()` include standard keys such as
`mode`, `success`, `errors`, `warnings`, `relation_status`, `residuals`,
`variables` (a mapping of `Variable` objects), `relations` (the active
`Relation` objects), `solver` (solver metadata), and `compiler_report`
(the structural diagnostics view).

**Profile-aware behavior**
Profile variables (shape 1) are handled explicitly: scalar inputs are broadcast
to profile arrays when required, and the system infers a common `profile_size`
from supplied profile values when needed. Supplied profiles are split into a
fixed shape plus a scalar average control kept linked by construction, and the
system inserts the canonical normalized `rho` grid when profiles are present.

Plotting helpers that visualize relation graphs or results live elsewhere in
the docs and tooling; the `RelationSystem` focuses on numeric evaluation and
diagnostics.
