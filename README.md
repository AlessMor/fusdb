# fusdb
Fusdb is a small toolkit for storing and validating fusion reactor scenarios. Each scenario is described by a `reactor.yaml` containing metadata plus grouped parameters (geometry, plasma, power, etc.) . Relations encoded in Python modules cross-check values, infer missing ones, and flag inconsistencies with configurable tolerances.

## What’s inside
- A `Reactor` dataclass that normalizes inputs, applies constraint systems, and exposes a convenient Python API.
- A loader that reads one or many `reactor.yaml` files and returns `Reactor` instances.
- Relation libraries for geometry, plasma parameters, power exhaust, and confinement scalings (bidirectional, priority-aware).
- An example reactor file documenting every supported field and option, plus test scenarios for regression coverage.

## Reactor YAML at a glance
- Top-level metadata (ids, names, organization, notes).
- `plasma_geometry`: radii, aspect ratio, shaping terms, optional extents.
- `plasma_parameters`: temperatures/densities, pressure/energy, beta, confinement (`confinement_time` with value + method).
- `power_and_efficiency`: fusion/power flows, wall loading, power exhaust metrics.

See `reactors/example_reactor.yaml` for an annotated template listing all supported fields and confinement scaling method names.

## Usage
- CLI: install editable (`pip install -e .`) then list or show reactors with the bundled `fusdb` commands.
- Python: `from fusdb.loader import load_all_reactors` to load scenarios, then access attributes on the returned `Reactor` objects.

## Interactive Relation Graph
The `relation_map.ipynb` notebook generates an interactive graph of the relations between variables, which is saved as `relation_graph.html`.
[View the interactive relation graph](https://AlessMor.github.io/fusdb/relation_graph.html) 

## Useful links:
- https://www.fusionenergybase.com/projects
- https://github.com/RemDelaporteMathurin/fusion-world
- cfspopcon
- PROCESS

## TODO:
- [ ] add relations for radiated power
- [ ] update species fractions and equilibrium solver