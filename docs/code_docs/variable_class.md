---

status: Online

---

# Variable Class

The variable system provides typed containers for scalar and profile values used by
`fusdb.reactor_class.Reactor` and `fusdb.relationsystem_class.RelationSystem`.

Related modules:

- `fusdb.variable_class`
- `fusdb.variable_util`

Related pages:

- [Reactor Class](reactor_class.md)
- [Relation Class](relation_class.md)
- [RelationSystem](relationsystem_class.md)

## Class Hierarchy

- `Variable`: abstract base class.
- `Variable0D`: scalar values (`ndim=0`).
- `Variable1D`: profile arrays (`ndim=1`).
- `make_variable(...)`: factory returning `Variable0D` or `Variable1D`.

## Shared Fields

- `name`: variable symbol.
- `unit`: display/registry unit.
- `ndim`: dimensionality (`0` scalar, `1` profile).
- `rel_tol`, `abs_tol`: optional tolerance overrides.
- `method`: optional preferred relation name.
- `input_source`: provenance label for input values.
- `fixed`: if `True`, solver should not overwrite value.

Runtime value fields:

- `input_value`: first accepted input snapshot (set with `as_input=True`).
- `current_value`: latest accepted value.

### Notes on `pass_id` and `reason`

`add_value(...)` accepts solver-context fields `pass_id` and `reason`.
Current implementation does not persist full per-pass history.

## Variable0D (Scalar)

`Variable0D.add_value(...)` accepts finite scalar values.

Signature:

```python
add_value(
    value: float | None,
    *,
    pass_id: int | None = None,
    reason: str | None = None,
    as_input: bool = False
) -> bool
```

Behavior:

- stores finite numeric values as `float`;
- returns `True` only when stored value changes;
- returns `False` for `None` or unchanged value;
- raises `ValueError` for invalid inputs;
- sets `input_value` on first call with `as_input=True`.

## Variable1D (Profile)

Additional fields:

- `coord`: profile coordinate label (default `"a"`).
- `profile_size`: size used when broadcasting scalar to profile (default `51`).

`Variable1D.add_value(...)` accepts finite scalar or 1D array values.

Signature:

```python
add_value(
    value: float | np.ndarray | None,
    *,
    pass_id: int | None = None,
    reason: str | None = None,
    as_input: bool = False
) -> bool
```

Behavior:

- scalar input broadcasts to uniform profile (`profile_size`);
- 1D arrays are stored as float64 arrays;
- returns `True` only when stored array changes;
- raises `ValueError` for invalid arrays (NaN, non-1D, empty, non-finite);
- sets `input_value` on first call with `as_input=True`.

Useful properties:

- `current_value_mean`
- `input_value_mean`

!!! note
    Profiles are normalized on `[0, 1]`. Keep `coord` accurate so integrals
    use the intended geometry interpretation.

## Example

```python
import numpy as np
from fusdb.variable_util import make_variable

# Scalar variable
R = make_variable(name="R", ndim=0, unit="m", rel_tol=0.02)
R.add_value(3.2, as_input=True)
R.add_value(3.3)
print(R.input_value, R.current_value)  # 3.2, 3.3

# Profile variable
n_e = make_variable(name="n_e", ndim=1, unit="m^-3", coord="a", profile_size=5)
n_e.add_value(1.0e20, as_input=True)
n_e.add_value(np.array([1.1e20, 1.05e20, 1.0e20, 0.95e20, 0.9e20]))
print(n_e.input_value_mean)
print(n_e.current_value_mean)
```
