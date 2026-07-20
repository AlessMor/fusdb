# Reactor

`Reactor` is a user-facing scenario object that loads a `reactor.yaml`
specification, holds variable values and relation-selection settings, and can
build a `RelationSystem` for numeric evaluation.

## Reactor Data Layout

Each reactor lives in one of these forms:

- `reactors/<reactor_id>/reactor.yaml`
- `reactors/<reactor_id>.yaml`

High-level structure:

- `metadata`: id/name/year/country/source information
- `tags`: labels used for relation filtering
- `solver_tags`: solve mode, verbosity, optional solving order
- `variables`: scalar/profile variable values and optional solver hints

The generated [reactor YAML reference](reactors/index.md) is built from the
current files under `reactors/`.

## Loading and Solving

```python
from fusdb import Reactor

reactor = Reactor.from_yaml("reactors/ARC_V0")
result = reactor.run()
```

**Core fields**
- `name`, `organization`, `country`, `year`, `doi`, `notes`
- `tags`: tuple of strings used to filter relations
- `mode`: one of `"verify"`, `"reconcile"`, `"optimize"`, or `"ordered"`
- `variables`: mapping of loaded `Variable` instances keyed by canonical name
- `relation_include`, `relation_exclude`, `relation_order`: relation selection controls
- `constraints`: optional system-level constraint expressions
- `grid_size`: optional shared profile size inferred from the reactor grid block
- `verbose`: runtime verbosity flag

**Example**
```python
from fusdb import Reactor, Variable

reactor = Reactor(
  name="Example Reactor",
  tags=("tokamak", "hmode"),
  variables={
    "R": Variable("R", value=3.2, unit="m"),
    "a": Variable("a", value=1.1, unit="m"),
  },
)
```

**Methods**
- `from_yaml(path)` loads `reactor.yaml`, parses variables (including numeric
  profile files), and applies registry defaults.
- `selected_relations(...)` returns the relation list after applying includes,
  excludes, tags, and ordering.
- `relation_system(...)` builds and returns a `RelationSystem` bound to the
  reactor's current variables and selected relations.
- `run(**kwargs)` convenience wrapper that builds a `RelationSystem` and runs
  the configured `mode`. `Variable` is immutable, so a solve never rewrites
  the reactor's declarations; the solved system is kept on `last_system`, and
  `reactor.<var>` (`get_variable`) returns a `SolvedVariable` read-through
  view -- `.declared` is the original, unchanged `Variable`, `.value` reads
  the latest solved value from `last_system` when one exists (falling back to
  the declaration otherwise). A later `run()` therefore starts from the same
  declarations again, not the previous solution; call
  `reactor.restart_from_solution()` first to opt into that explicitly.
- To display variables, first build `TableData` with
  `variable_table_data(*sources)`, then call `render_table(data, format="html"`
  or `format="text")`. This supports several reactors, already-run
  `RelationSystem`s, and/or `SolvedColumn`s, and retains input→output colouring,
  active-variable highlighting, and relation tooltips.

**Parallel solving**
- `solve_reactors(paths, mode="reconcile", workers=N)` solves many reactor YAMLs
  at once, one worker process each (reconcile is CPU-bound, so processes — not
  threads — give the speedup). Live reactors/systems can't cross a process
  boundary, so each worker loads from YAML and returns a picklable `SolvedColumn`
  (the same display column `variable_table_data` consumes). A reactor that fails to
  load or solve comes back as a failed-result column instead of aborting the
  batch. `paths` may be YAML paths/directories or `from_yaml`-loaded reactors
  (their `source_path` is used).

**Variable table example**
```python
from IPython.display import HTML, display
from fusdb import Reactor, render_table, variable_table_data

# Current (input) values, no solve:
reactors = [Reactor.from_yaml(path) for path in reactor_paths]
display(HTML(render_table(variable_table_data(*reactors))))

# Reconciled values, solved in parallel:
from fusdb import solve_reactors
columns = solve_reactors(reactor_paths, mode="reconcile")
display(HTML(render_table(variable_table_data(*columns))))
```

**Profiles and variables**
- Variables are declared with `value` (scalar or 1D profile), optional `unit`,
  `rel_tol`, `fixed`, `size`, and `constraints` entries. Profile files may be
  supplied via `file: path.csv` or `value: path.csv` in YAML; simple numeric
  strings are left to variable coercion.
- When selected relations require variables that were not provided, `relation_system`
  will create placeholder `Variable` objects using registry defaults (including
  profile `size` from `grid.size` when available).

**Reactor YAML (example)**
```yaml
metadata:
  id: REACTOR_ID
  name: Reactor Name

tags:
  - device_tag
  - regime_tag

solver_tags:
  mode: verify         # verify | reconcile | optimize | ordered
  verbosity: false

grid:
  size: 46

variables:
  R: 3.2
  a:
    value: 1.1
    unit: m
    rel_tol: 0.02
    fixed: false
  n_e:
    file: profiles/ne.csv
    delimiter: ","
    skiprows: 1
```
