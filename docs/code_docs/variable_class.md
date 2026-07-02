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

Per-variable numerics (owned by the variable, consumed by `RelationSystem`):

- `dim` -> number of scalar elements (1 for scalars, grid size for profiles)
- `coerce_shape(value)` -> value coerced to this variable's registry shape
- `solver_value(value)` / `public_value(value)` -> conversion between the
  public form and the numerically safe solver form (physical-domain boundary
  values are projected onto solver-domain bounds and back)
- `check_solver_domain(value)` -> raise if a value violates the solver domain
- `candidate_valid(value)` -> whether a prospective value is finite and in-domain
- `scale(*refs)` / `tolerance_floor()` / `tolerance_width(scale)` -> the
  residual/movement scaling quantities derived from `rel_tol`/`abs_tol`
- `movement_reference(fallback, index=None)` -> one supplied-input element for
  movement scaling
- `movement_excess(current, reference)` -> worst tolerance-band crossing of a
  solved value against the supplied input (the reconcile objective quantity)
- `domain_violation_rows(value)` -> tolerance-normalized physical-domain
  violation rows for the solver feasibility residual
- `moved_from_input(value)` -> whether a candidate value moved off the
  supplied input (used to reject solves that changed a fixed variable)

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
