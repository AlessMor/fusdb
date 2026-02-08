# fusdb

A small toolkit for storing and validating fusion reactor scenarios.

Each scenario is described by a `reactor.yaml` file, containing variables that describe the steady-state condition of the plasma. Relations encoded in Python modules cross-check values, infer missing ones, and recognise inconsistencies in the input data.

## Project Structure

```
fusdb/
├── reactors/         # Reactor data files (YAML)
├── src/fusdb/        # Main source code
│   ├── registry/     # Allowed variables, constants, and default values
│   ├── relations/    # Physics and engineering relations
│   └── cli.py        # Command-line interface
├── tests/            # Tests
└── docs/             # Documentation and notebooks
```

## Reactors

Each reactor is defined by a `reactor.yaml` file. See `src/fusdb/registry/reactor_example.yaml` for an annotated template.

Each reactor data is taken from papers and represent a plasma scenario for a fusion reactor.

## Relation and RelationSystem

Relations are defined in Python modules within `src/fusdb/relations/`. They are used to:
- Cross-check values
- Infer missing values
- Recognise inconsistencies

## Usage

- **CLI**: Install with `pip install -e .`, then use `fusdb` to list or show reactors.
- **Python**: Use `from fusdb.loader import load_all_reactors` to load scenarios into `Reactor` objects.

## Interactive Relation Graph

The `docs/relation_map.ipynb` notebook generates an interactive graph of the relations between variables.

[View the interactive relation graph](https://AlessMor.github.io/fusdb/docs/relation_graph.html) 

## Useful links and references:

0D plasma codes:
- [cfspopcon](https://github.com/cfs-energy/cfspopcon)
- [PROCESS](https://github.com/ukaea/PROCESS)
Databeses for fusion reactors worldwide:
- [Fusion Energy Base](https://www.fusionenergybase.com/projects)
- [Fusion World on Github](https://github.com/RemDelaporteMathurin/fusion-world)


## TODO:

- [ ] check default/global solve modes for RelationSystem
- [ ] add relations for radiated power
- [ ] update species fractions and equilibrium solver
- [ ] add density and temperature profiles (complex, requires re-evaluation of relation class: is it possible to adapt current relations to work with both profiles and avgs?)
- [ ] add reactor optimization by: splitting yaml loading from solving, pick axes (default n_avg, T_avg), solve the implicit system at each point, then filter by constraints and rank by an objective.
- [ ] check relation enforcing a "solve_for" constraint (P_aux, P_loss)
