# Reactor

`Reactor` is a user-facing scenario object that loads a `reactor.yaml`
specification, holds variable values and relation-selection settings, and can
build a `RelationSystem` for numeric evaluation.

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
  the configured `mode`.

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
