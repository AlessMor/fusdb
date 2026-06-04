# Variable Class

The variable system provides typed containers for scalar and profile values used by
`fusdb_pyomo.Reactor` and `fusdb_pyomo.RelationSystem`.

Related modules:

- `fusdb.variable_class`

Related pages:

- [Reactor Class](reactor_class.md)
- [Relation Class](relation_class.md)
- [RelationSystem](relationsystem_class.md)

## Class Structure

- `Variable`: container for scalar (shape==0) and profile (shape==1) values.

## Shared Fields

- `name`: canonical variable name
- `unit`: canonical registry unit
- `shape`: `0` for scalar, `1` for 1D profile
- `rel_tol`: optional relative tolerance override
- `constraints`: validation constraints (expressions)
- `fixed`: whether solve modes may change the value

Runtime value fields:

- `value`: the current canonical-unit value (scalar or 1D `numpy.ndarray`)
- `reference_value`: a copy of the initial `value` captured at construction
- `source`: provenance label such as `"given"`, `"missing"`, or `"computed"`

## API and behavior

Construct a variable with:

```python
from fusdb.variable import Variable

v = Variable(name="R", value=3.2, unit="m", rel_tol=0.02, fixed=False)
```

Key methods and helpers:

- `clone(**changes)` -> return a fresh `Variable` with selected overrides
- `set_value(value, *, source=None)` -> set a canonical-unit value and update `source`
- `as_dict()` -> serializable view useful for result payloads

Profiles (shape==1) accept scalar inputs (broadcast to the profile length) or
1D arrays; the constructor or `set_value` will validate shape and size and
convert numeric inputs to `numpy.ndarray` where appropriate.

Validation errors (NaN, wrong dimensionality, out-of-domain) raise `ValueError`.

## Example

```python
import numpy as np
from fusdb.variable import Variable

# Scalar variable
R = Variable("R", value=3.2, unit="m", rel_tol=0.02)
R.set_value(3.3, source="user")
print(R.reference_value, R.value)

# Profile variable
n_e = Variable("n_e", value=np.array([1.1e20, 1.05e20, 1.0e20]), unit="m^-3")
print(np.mean(n_e.value))
```
