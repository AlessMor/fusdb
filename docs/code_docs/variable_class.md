# Variable Class

The variable system provides typed containers for scalar and profile values used by
`fusdb.Reactor` and `fusdb.RelationSystem`.

Related modules:

- `fusdb.variable`

Related pages:

- [Reactor Class](reactor_class.md)
- [Relation Class](relation_class.md)
- [RelationSystem](relationsystem_class.md)

## Class Structure

- `Variable`: container for scalar (shape==0) and profile (shape==1) values.
  A `Variable` is its registry `VariableSpec` (the immutable definition: name,
  aliases, unit, shape, domain, tolerances, read through `.spec`) plus the
  per-scenario state.

## Shared Fields

- `name`: canonical variable name
- `unit`: canonical registry unit
- `shape`: `0` for scalar, `1` for 1D profile (from the registry spec)
- `rel_tol` / `abs_tol`: tolerance overrides (registry defaults when omitted)
- `constraints`: additional local constraints or applicability guards
- `fixed`: whether solve modes may change the value
- `size`: profile length for shape==1 variables

Runtime value fields:

- `value`: the current canonical-unit value (scalar or 1D `numpy.ndarray`)
- `input_value`: the immutable user-supplied value (the movement reference)

## API and behavior

Construct a variable with:

```python
from fusdb.variable import Variable

v = Variable("R", value=3.2, unit="m", rel_tol=0.02, fixed=False)
```

State methods:

- `clone(**changes)` -> return a fresh `Variable` with selected overrides
- `set_input(value)` -> set the user/input value (canonical units); also resets `value`
- `set_value(value)` -> set the current public value (canonical units)

`Variable` is a **boundary input record only**: it validates and
unit-converts one user/yaml input and is then ingested by `RelationSystem`
into plain value dicts.  All per-variable numerics (`solver_value`,
`public_value`, `coerce`, `check_solver_domain`, `candidate_valid`,
`tolerance_floor`/`tolerance_width`/`scale_of`, `movement_excess`,
`domain_violation_rows`) live on the frozen `VariableSpec` in the registry —
computed once per process with precomputed bounds/projection constants, taking
the profile size and resolved tolerances as arguments.

Profiles (shape==1) accept scalar inputs (broadcast to the profile length) or
1D arrays; the constructor and setters validate shape, size and physical
domain, converting numeric inputs to `numpy.ndarray` where appropriate.
Validation errors (NaN, wrong dimensionality, out-of-domain) raise `ValueError`.

## Example

```python
import numpy as np
from fusdb.variable import Variable

# Scalar variable
R = Variable("R", value=3.2, unit="m", rel_tol=0.02)
R.set_value(3.3)
print(R.input_value, R.value)

# Profile variable
n_e = Variable("n_e", value=np.full(46, 1.0e20), unit="m^-3")
print(np.mean(n_e.value), n_e.dim)
```
