# Reactors

A `Reactor` loads a scenario from `reactor.yaml`, applies defaults, filters relations by tags, and solves for unknown variables.

## Reactor Data Layout

Each reactor lives in one of these forms:

- `reactors/<reactor_id>/reactor.yaml`
- `reactors/<reactor_id>.yaml`

High-level structure:

- `metadata`: id/name/year/country/source information
- `tags`: labels used for relation filtering
- `solver_tags`: solve mode, verbosity, optional solving order
- `variables`: scalar/profile variable values and optional solver hints

## Loading and Solving

```python
from fusdb import Reactor

reactor = Reactor.from_yaml("reactors/ARC_2015")
reactor.solve()
```

## Detailed Reactor Class Reference

See [Reactor Class](../code_docs/reactor_class.md) for the complete field and
method description, including YAML examples.
