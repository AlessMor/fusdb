# Reactor

Reactor loads a `reactor.yaml` specification and orchestrates solving.

**Core fields**
- `path`, `id`, `name`, `organization`, `country`, `year`, `doi`, `notes`
- `tags`: list of strings used for relation filtering
- `solving_order`: optional ordered list of domains/relations passed to `RelationSystem`
- `solver_mode`: `"overwrite"` or `"check"`
- `verbose`: bool
- `relations`: optional caller-provided list of `Relation`
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
- `from_yaml(path)` loads YAML, parses variables, and applies defaults.
- `make_relationsystem(mode=None, verbose=None)` builds one runtime `RelationSystem` bound to reactor values.
- `solve(mode=None, verbose=None)` runs one `RelationSystem` solve and updates `variables_dict`.
- `diagnose()` runs one `"check"` mode `RelationSystem` and returns diagnostics payload.
- `popcon(...)` evaluates POPCON-style scans over one or more axes.
- `plot_popcon(...)` plots masked fills + contour overlays from POPCON results.
- `__repr__()` returns a compact summary string.
- `plot_cross_sections()` plots the plasma cross-section using `R`, `a`, `kappa_95`, `delta_95`.

**Profiles**
- Profile-capable variables accept a dict payload with `coord`, `x`, `y`, optional `meta`.
- `n_e`, `n_i`, `T_e`, and `T_i` are profile-valued; scalar inputs are treated as flat profiles.
- If `n_avg`/`T_avg` are provided without profiles, a flat profile on `r_minor` is created by defaults.
- If profiles are provided without averages, `n_avg`/`T_avg` are computed as simple means.

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
  rel_tol: 0.01     # optional default relative tolerance for all variables
  verbosity: false  # optional
  solving_order: # optional, controls relation order inside RelationSystem.
    - domain_a
    - domain_b
    - "Specific Relation Name"

variables:
  R: 3.2            # shorthand value (unit defaults to registry)
  a:
    value: 1.1      # explicit value
    unit: m         # optional unit (converted to registry default using pint)
    method: relation_name # optional, selects a specific relation when multiple are available.
    rel_tol: 0.02   # optional per-variable relative tolerance override
    fixed: false    # optional, prevents solver changes (useful for certain values such as geometry)
  n_e:
    value:
      coord: r_minor
      x: [0.0, 0.5, 1.0]
      y: [1.0e20, 0.9e20, 0.7e20]
```
