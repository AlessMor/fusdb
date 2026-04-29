# Relation Class

`fusdb.relation_class.Relation` represents one physics relation:
one output computed from explicit inputs.

Relations are usually declared with the `@relation` decorator in
`fusdb.relation_class`.

## Core Fields

Set at construction:

- `name`: human-readable relation name (defaults to function name when using decorator)
- `inputs`: canonical input variable names
- `outputs`: canonical output variable names
- `forward`: callable accepting named inputs
- `tags`: classification tags, for example `("geometry", "plasma")`
- `constraints`: validation constraints, for example `("R > 0", "a < R")`
- `initial_guesses`: optional variable-to-callable map for numeric solver initial values
- `solve_for`: optional target metadata for explicit inverse solvers
- `sympy_expression`: optional symbolic form used for automatic inversion

## Key Property and Methods

- `symbols` -> symbolic variables keyed by canonical name
- `evaluate(values)` -> computed output value
- `apply(values)` -> output assignments as a mapping
- `input_names(output=None)` -> ordered input names for a target
- `inverse_solver(unknown)` -> callable or `None`
- `solve_for_value(unknown, values)` -> solved value or `None`

## Creating Relations

### Direct Construction

```python
from fusdb.relation_class import Relation

rel = Relation(
    name="Aspect ratio",
    inputs=("R", "a"),
    outputs=("A",),
    forward=lambda R, a: R / a,
    tags=("geometry",),
    constraints=("R > 0", "a > 0", "a < R"),
)
```

### Decorator (Recommended)

```python
from fusdb.relation_class import relation

@relation(
    name="Aspect ratio",
    output="A",
    tags=("geometry",),
    constraints=("R > 0", "a > 0", "a < R"),
    initial_guesses={"a": lambda values: 0.3 * values.get("R", 1.0)},
    solve_for={"a": lambda values: values["R"] / values["A"]},
)
def aspect_ratio(R: float, a: float) -> float:
    return R / a
```

Decorator parameters:

- `name`: relation display name
- `output`: output variable name
- `tags`: classification tags
- `constraints`: validation constraints
- `initial_guesses`: initial guesses for numeric solvers
- `solve_for`: explicit inverse solver metadata

!!! note
    For symbolic inversion to work, keep relation functions SymPy-friendly:
    avoid NumPy-only expressions and use SymPy-compatible math/branching.

## Example With RelationSystem

```python
from fusdb.relationsystem_class import RelationSystem
from fusdb.variable_class import Variable
from fusdb.relation_class import relation

@relation(
    name="Aspect ratio",
    output="A",
    tags=("geometry",),
    constraints=("R > 0", "a > 0", "a < R"),
)
def aspect_ratio(R: float, a: float) -> float:
    return R / a

R = Variable.make(name="R", ndim=0, unit="m")
a = Variable.make(name="a", ndim=0, unit="m")
A = Variable.make(name="A", ndim=0, unit="1")

R.add_value(3.0, as_input=True)
a.add_value(1.0, as_input=True)

rel_system = RelationSystem(
    relations=[aspect_ratio],
    variables=[R, a, A],
    mode="overwrite",
)
rel_system.solve()
print(rel_system.variables_dict["A"].current_value)  # 3.0
```
