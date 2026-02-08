# fusdb

A small toolkit for storing and validating fusion reactor plasma scenarios.

Each scenario is described by a `reactor.yaml` file, containing variables that describe the steady-state condition of the plasma. Relations encoded in Python modules cross-check values, infer missing ones, and recognize inconsistencies in the input data.

## Disclaimers:
This is a personal project made by Alessandro Morandi, PhD student at PoliTO, with the aim of having a single place to store data on fusion reactors and useful formulas to use in other studies.  
Due to the personal nature of the project, use of AI LLM models was made to speed up the process (especially inside the classes and to set up docs, while physical formulas have been added manually).  
**The validation of reactors should not be considered a criticism to the published papers, and are often due to the simplifications of this model. Always double-check the results if they should be used for scientific analyses.**

## Project Structure

```
fusdb/
├── reactors/                 # Reactor data files (YAML)
├── src/fusdb/                # Main source code
│   ├── __init__.py           # Public API exports
│   ├── reactor_class.py      # Reactor loader/solver
│   ├── relation_class.py     # Relation definition/decorator
│   ├── relationsystem_class.py # Solver engine
│   ├── variable_class.py     # Variable container
│   ├── relation_util.py      # Relation discovery/filtering helpers
│   ├── relations/            # Relations grouped by domain
│   └── registry/             # Allowed variables, constants, defaults, tags
├── docs/                     # Sphinx docs (Markdown via MyST), notebooks, artifacts
├── lib/                      # JS/CSS assets used by HTML graph outputs
└── tests/                    # Tests
```

## Reactors

Each reactor lives in `reactors/<reactor_id>/reactor.yaml` or as a standalone `reactors/<reactor_id>.yaml`.
See `docs/reactor_class.md` for the YAML schema and examples.

## Relations and RelationSystem

Relations are defined in `src/fusdb/relations/` and grouped by domain (geometry, power_balance, confinement, etc.). The solver engine is `RelationSystem` in `src/fusdb/relationsystem_class.py`.

## Usage

```python
from fusdb import Reactor

reactor = Reactor.from_yaml("reactors/ARC_2015")
reactor.solve()

print(reactor.variables_dict["P_fus"].current_value)
```

## Documentation

Sphinx docs live in `docs/` and use Markdown via MyST.

```bash
pip install -e .[docs]
sphinx-build -b html docs docs/_build/html
```

## Useful Links and References

0D plasma codes:
- [cfspopcon](https://github.com/cfs-energy/cfspopcon)
- [PROCESS](https://github.com/ukaea/PROCESS)

Databases for fusion reactors worldwide:
- [Fusion Energy Base](https://www.fusionenergybase.com/projects)
- [Fusion World on Github](https://github.com/RemDelaporteMathurin/fusion-world)

## TODO
- [ ] check default/global solve modes for RelationSystem
- [ ] add relations for radiated power
- [ ] update species fractions and equilibrium solver
- [ ] add density and temperature profiles (complex, requires re-evaluation of relation class: is it possible to adapt current relations to work with both profiles and avgs?)
- [ ] add reactor optimization by: splitting yaml loading from solving, pick axes (default n_avg, T_avg), solve the implicit system at each point, then filter by constraints and rank by an objective.
- [ ] check relation enforcing a "solve_for" constraint (P_aux, P_loss)
