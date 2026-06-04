# RelationSystem

`RelationSystem` orchestrates simultaneous numeric evaluation and solving of a
selected set of `Relation` objects against a set of `Variable` objects. The
current SciPy-based numeric backend uses `scipy.optimize.least_squares` to run
reconcile/optimize solves and provides verification and ordered-evaluation
modes.

**Core inputs**
- `variables`: an iterable of `Variable` instances (the system's variables)
- `relations`: an iterable of `Relation` instances (selected relations to evaluate)
- `constraints`: optional extra system-level constraint expressions
- `name`: optional system name
- `verbose`: enable additional runtime warnings and prints
- `variable_registry`: optional `VariableRegistry` used to resolve aliases

**Canonical runtime state**
- `variables`: list of `Variable` objects
- `variables_by_name`: mapping `{name: Variable}` used internally and returned in results
- `relations`: full list of active `Relation` objects (including relation-local guards)
- `primary_relations`: the relations originally selected for the system
- `relations_by_name`: mapping `{name: Relation}`
- `profile_size`: inferred common profile length used for profile variables
- `_last_vector_spans`, `_last_solver_values`: internal solver bookkeeping

**Main methods**
- `run(mode='verify', **options)`: dispatch to one of the available modes
	(`verify`, `reconcile`, `optimize`, `ordered`) and return a result dict.
- `verify_current()`: evaluate all active relations at the current variable state
	without changing variables; returns diagnostic result payload.
- `reconcile(**options)`: run the numeric reconcile solver (uses SciPy least_squares).
- `optimize(**options)`: run optimization-mode solve (objective-aware least-squares).
- `ordered(**options)`: run forward ordered evaluation where later relation outputs
	overwrite earlier values.
- `solve_mode(mode, **kwargs)`: lower-level entry that implements reconcile/optimize.
- `ordered_evaluate(order=None, passes=1)`: explicit ordered evaluation helper.
- `compatibility_report()`: lightweight mapping of active relations to backend labels.

The result dictionaries returned by `run()` and the mode helpers include
standard keys such as `mode`, `success`, `errors`, `warnings`, `relation_status`,
`variable_status`, `residuals`, `variables` (a mapping of `Variable` objects),
and `relations` (the selected `Relation` objects).

**Profile-aware behavior**
Profile variables (shape 1) are handled explicitly: scalar inputs are broadcast
to profile arrays when required, and the system infers a common `profile_size`
from supplied profile values when needed. The system also inserts defaulted
profile variables (for example a normalized `rho` grid) and registry-driven
uniform-profile fallback relations when appropriate.

Plotting helpers that visualize relation graphs or results live elsewhere in
the docs and tooling; the `RelationSystem` focuses on numeric evaluation and
diagnostics.
