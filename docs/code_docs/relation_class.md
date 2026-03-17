---

status: Online

---

# Relation Class

`fusdb.relation_class.Relation` represents one physics relation:
one output computed from explicit inputs.

Relations are usually declared with the `@relation` decorator in
`fusdb.relation_util`.

## Core Fields

Set at construction:

- `name`: human-readable relation name (defaults to function name when using decorator)
- `output`: canonical output variable name
- `func`: callable accepting ordered inputs
- `inputs`: ordered tuple/list of input variable names
- `tags`: classification tags, for example `("geometry", "plasma")`
- `rel_tol_default`, `abs_tol_default`: default tolerances for consistency checks
- `constraints`: hard constraints, for example `("R > 0", "a < R")`
- `soft_constraints`: warning-only constraints
- `initial_guesses`: optional variable-to-callable map for numeric solver initial values
- `inverse_functions`: optional variable-to-callable map for explicit inverse solvers
- `sympy_expr`: optional symbolic form used for automatic inversion

## Key Property and Methods

- `variables` -> tuple of `(input_1, ..., input_n, output)`
- `evaluate(values)` -> computed output value
- `constraint_violations(values)` -> violated hard constraints
- `soft_constraint_violations(values)` -> violated soft constraints
- `check_satisfied(values, rel_tol=None, abs_tol=None, check_constraints=True)` -> `(satisfied, status, residual)`
- `get_residual(values)` -> `actual - expected`
- `inverse_solver(unknown)` -> callable or `None`
- `solve_for_value(unknown, values)` -> solved value or `None`

## Profile Integration Convention

For scalar outputs that depend on profile inputs:

- integrate explicitly inside the relation function (for example using
  `fusdb.utils.integrate_profile_over_volume`);
- keep profile dependencies explicit in variable naming (for example
  `sigmav_DT_profile`);
- do not rely on `RelationSystem` to auto-integrate profile-valued inputs.

See [Profile Integration](profile_integration.md).

## Creating Relations

### Direct Construction

```python
from fusdb.relation_class import Relation

rel = Relation(
    name="Aspect ratio",
    output="A",
    func=lambda R, a: R / a,
    inputs=["R", "a"],
    tags=("geometry",),
    rel_tol_default=0.01,
    constraints=("R > 0", "a > 0", "a < R"),
)
```

### Decorator (Recommended)

```python
from fusdb.relation_util import relation

@relation(
    name="Aspect ratio",
    output="A",
    tags=("geometry",),
    rel_tol_default=0.01,
    constraints=("R > 0", "a > 0", "a < R"),
    soft_constraints=("A > 2",),
    initial_guesses={"a": lambda values: 0.3 * values.get("R", 1.0)},
    inverse_functions={"a": lambda values: values["R"] / values["A"]},
)
def aspect_ratio(R: float, a: float) -> float:
    return R / a
```

Decorator parameters:

- `name`: relation display name
- `output`: output variable name
- `tags`: classification tags
- `rel_tol_default`, `abs_tol_default`: default tolerances
- `constraints`: hard constraints
- `soft_constraints`: warning-only constraints
- `initial_guesses`: initial guesses for numeric solvers
- `inverse_functions`: explicit inverse solvers

!!! note
    For symbolic inversion to work, keep relation functions SymPy-friendly:
    avoid NumPy-only expressions and use SymPy-compatible math/branching.

## Example With RelationSystem

```python
from fusdb.relationsystem_class import RelationSystem
from fusdb.variable_util import make_variable
from fusdb.relation_util import relation

@relation(
    name="Aspect ratio",
    output="A",
    tags=("geometry",),
    constraints=("R > 0", "a > 0", "a < R"),
)
def aspect_ratio(R: float, a: float) -> float:
    return R / a

R = make_variable(name="R", ndim=0, unit="m")
a = make_variable(name="a", ndim=0, unit="m")
A = make_variable(name="A", ndim=0, unit="1")

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
