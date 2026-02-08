# Reactor

Reactor loads a `reactor.yaml` specification and orchestrates solving.

**Core fields**
- `path`, `id`, `name`, `organization`, `country`, `year`, `doi`, `notes`
- `tags`: list of strings used for relation filtering
- `solving_order`: optional ordered list of domains/relations to solve in sequence
- `solver_mode`: `"overwrite"` or `"check"`
- `verbose`: bool
- `relations`: list of `Relation`
- `default_relations`: list of `Relation` (defaults applied at load)
- `variables_dict`: dict of `Variable` objects keyed by name

**Example**
```python
from fusdb import Reactor, Variable

reactor = Reactor(
    id="EXAMPLE",
    name="Example Reactor",
    tags=["tokamak", "hmode"],
    variables_dict={
        "R": Variable(name="R", values=[3.2], unit="m", input_source="explicit"),
        "a": Variable(name="a", values=[1.1], unit="m", input_source="explicit"),
    },
)
```

**Methods**
- `__post_init__()` initializes the per-instance logger context.
- `from_yaml(path)` loads YAML, parses variables, applies defaults, filters relations.
- `_relation_filter_inputs()` returns variable names and method overrides for filtering.
- `_ordered_relations()` yields relations in solving order (domains or names).
- `solve(mode=None, verbose=None)` runs one or more `RelationSystem` passes and updates `variables_dict`.
- `diagnose()` returns `{"violated_relations": [...], "likely_culprits": {...}, "variable_issues": [...]}` using `"check"` mode.
- `__repr__()` returns a compact summary string.
- `plot_cross_sections()` plots the plasma cross-section using `R`, `a`, `kappa_95`, `delta_95`.

**Reactor YAML (example)**
```yaml
metadata:
  id: REACTOR_ID        # required
  name: Reactor Name    # optional, defaults to id
  organization: Org     # optional
  country: Country      # optional
  year: 2025            # optional
  doi: 10.0000/example  # optional
  notes: Optional notes # optional

tags: # used to filter relations.
  - device_tag
  - regime_tag

solver_tags:
  mode: overwrite   # overwrite | check
  verbosity: false  # optional
  solving_order: # optional, controls the order of `RelationSystem` runs and may warn on overlapping outputs. Each entry can be a domain tag or an exact `Relation.name`.
    - domain_a
    - domain_b
    - "Specific Relation Name"

variables:
  R: 3.2            # shorthand value (unit defaults to registry)
  a:
    value: 1.1      # explicit value
    unit: m         # optional unit (converted to registry default using pint)
    method: relation_name # optional, selects a specific relation when multiple are available.
    rel_tol: 0.02   # optional, tolerance override
    abs_tol: 0.0    # optional, tolerance override
    fixed: false    # optional, prevents solver changes (useful for certain values such as geometry)
```
