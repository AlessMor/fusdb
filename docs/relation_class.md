# Relation

`Relation` represents one physics relation (one output computed from inputs).

**Core fields**
- `name`, `output`, `func`, `inputs`
- `tags`: tuple of strings
- `rel_tol_default`, `abs_tol_default`
- `constraints`: tuple of string expressions (e.g. `"a > 0"`)
- `initial_guesses`: `{var: callable}` for numeric solvers
- `solve_for`: `{var: callable}` explicit inverses (optional)
- `sympy_expr`: symbolic expression used for auto-inversion (optional)

**Derived / properties**
- `variables` → `tuple(inputs + [output])`  
  Example: `("R", "a", "B0")`

**Methods**
- `evaluate(**kwargs)` → output value
- `constraint_violations(values)` → list of violated constraint expressions
- `check_satisfied(values, rel_tol=None, abs_tol=None, check_constraints=True)`
- `get_residual(values)` → `actual - expected`
- `inverse_solver(unknown)` → cached symbolic inverse (if available)
- `solve_for_value(unknown, values)` → explicit solver, else symbolic inverse

**Example**
```python
from fusdb import Relation

rel = Relation(
    name="relation name",
    output="output_symbol",  # as in registry/allowed_variables.yaml
    func=lambda input_symbol_1, input_symbol_2: any_function_of(input_symbol_1, input_symbol_2),
    inputs=["input_symbol_1", "input_symbol_2"],
    tags=("plasma", "tokamak"),
    constraints=("input_symbol_1 > 0",),
)
```

or as a decorator:

```python
@Relation(
    name="relation name",
    output="output_symbol",  # as in registry/allowed_variables.yaml
    tags=("plasma", "tokamak"),
    constraints=("input_symbol_1 > 0",),
    initial_guesses={"input_symbol_1": lambda values: 0.5},
    solve_for={"input_symbol_1": lambda values: values["output_symbol"] / values["input_symbol_2"]},
)
def function_name(input_symbol_1: float, input_symbol_2: float) -> float:
    """Short description of the relation."""
    output_symbol = any_function_of(input_symbol_1, input_symbol_2)
    return output_symbol
```

Notes
- Relations are plain deterministic Python functions with named scalar inputs.
- If you want symbolic inversion, keep the function SymPy-friendly (avoid NumPy-only ops).
- Use `sympy.Piecewise` instead of Python `if/else` for symbolic branches.
