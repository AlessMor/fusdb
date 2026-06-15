# Relation Class

`fusdb.Relation` represents one physical relation: a callable that
computes outputs (or a residual) from explicit inputs.

Relations are usually declared with the `@relation` decorator exported by
the `fusdb` package.

## Core Fields

Set at construction:

- `name`: stable relation name (defaults to the function name when using the decorator)
- `input_names`: ordered canonical input variable names
- `outputs`: declared output variable names (empty means the function implements an outputless residual)
- `func`: Python callable implementing the forward evaluation
- `tags`: classification tags, for example `("geometry", "plasma")`
- `constraints`: relation-local validity constraints
- `enforce`: whether the relation is enforced (True) or warning-only (False)

## Key properties and methods

- `input_names` -> tuple of input names
- `evaluate(namespace)` -> call the underlying Python function with the provided namespace
- `output_map(result)` -> map a function result to declared outputs
- `comparisons(namespace)` -> list of comparison tuples `(lhs, op, rhs, output_name)` used by the solver
- `implicit` -> whether outputs also appear among inputs
- `from_function(...)` and the `relation` decorator -> helpers to construct and register `Relation` objects

## Creating Relations

### Direct construction

```python
from fusdb.relation import Relation

rel = Relation(
    name="Aspect ratio",
    func=lambda R, a: R / a,
    input_names=("R", "a"),
    outputs=("A",),
    tags=("geometry",),
    constraints=("R > 0", "a > 0", "a < R"),
)
```

### Decorator (recommended)

```python
from fusdb.relation import relation

@relation(
    name="Aspect ratio",
    outputs=("A",),
    tags=("geometry",),
    constraints=("R > 0", "a > 0", "a < R"),
)
def aspect_ratio(R: float, a: float) -> float:
    return R / a
```

Decorator parameters:

- `name`: relation name used for registration
- `outputs`: output variable name or tuple of names
- `tags`: classification tags
- `enforce`: make a relation warning-only by setting `enforce=False`
- `constraints`: relation-local validity constraints

!!! note
    For symbolic inversion to work, keep relation functions SymPy-friendly:
    avoid NumPy-only expressions and use SymPy-compatible math/branching.

## Example with RelationSystem

```python
from fusdb import RelationSystem
from fusdb.variable import Variable
from fusdb.relation import relation

@relation(name="Aspect ratio", outputs=("A",))
def aspect_ratio(R: float, a: float) -> float:
    return R / a

R = Variable("R", value=3.0, unit="m")
a = Variable("a", value=1.0, unit="m")
A = Variable("A", value=None, unit="1")

rel_system = RelationSystem(variables=[R, a, A], relations=[aspect_ratio])
result = rel_system.run(mode="verify")
print(result["variables"]["A"].value)
```
