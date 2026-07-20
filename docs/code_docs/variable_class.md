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

- `Variable`: an **immutable, frozen** declaration of one scalar (shape==0)
  or profile (shape==1) variable for one scenario.  A `Variable` is its
  registry `VariableSpec` (the immutable definition: name, aliases, unit,
  shape, domain, tolerances, read through `.spec`) plus that one scenario's
  declared value, `fixed` flag, and tolerance/size/guard overrides.
  Constructing a `Variable` *is* the ingestion event -- unit conversion,
  shape coercion, domain validation -- and the instance is frozen
  immediately afterward: there are no setters, and a changed declaration is
  always a new `Variable` (see `clone()` below), never a mutation of an
  existing one.

## Shared Fields

- `name`: canonical variable name
- `unit`: canonical registry unit
- `shape`: `0` for scalar, `1` for 1D profile (from the registry spec)
- `rel_tol` / `abs_tol`: tolerance overrides (registry defaults when omitted)
- `constraints`: additional local constraints or applicability guards
- `fixed`: whether solve modes may change the value
- `size`: profile length for shape==1 variables
- `value`: the declared canonical-unit value (scalar or 1D `numpy.ndarray`)
- `input_value`: alias of `value` at declaration time (they can never
  diverge, since nothing may write to either after construction)

## API and behavior

Construct a variable with:

```python
from fusdb.variable import Variable

v = Variable("R", value=3.2, unit="m", rel_tol=0.02, fixed=False)
```

- `clone(**changes)` -> return a fresh, independently re-ingested `Variable`
  with the given fields overridden (everything else carried over from
  `self`).  This is the *only* way to change a declaration:
  `v.clone(value=3.3)`, `v.clone(fixed=True)`.  Because `clone()` goes
  through the same construction/validation path as `Variable(...)`, the
  result is re-unit-converted and re-domain-checked, not a raw field copy.

`Variable` is a **boundary input record only**: it validates and
unit-converts one user/yaml input and is then ingested by `RelationSystem`
into plain value dicts (a solve reads/writes those dicts, never the
`Variable` objects that seeded them).  All per-variable numerics
(`solver_value`, `public_value`, `coerce`, `check_solver_domain`,
`candidate_valid`, `tolerance_floor`/`tolerance_width`/`scale_of`,
`movement_excess`, `domain_violation_rows`) live on the frozen
`VariableSpec` in the registry — computed once per process with
precomputed bounds/projection constants, taking the profile size and
resolved tolerances as arguments.

A `Reactor` never rewrites a declaration after a solve either: it exposes a
read-through view instead (`reactor.<name>` /
`reactor.get_variable(name)` -> `fusdb.reactor.SolvedVariable`, pairing the
frozen declaration with the latest value from `reactor.last_system`).  See
[Reactor Class](reactor_class.md).

Profiles (shape==1) accept scalar inputs (broadcast to the profile length) or
1D arrays; the constructor validates shape, size and physical domain,
converting numeric inputs to `numpy.ndarray` where appropriate.  Validation
errors (NaN, wrong dimensionality, out-of-domain) raise `ValueError` --
including from `clone()`, since it re-validates.

## Example

```python
import numpy as np
from fusdb.variable import Variable

# Scalar variable
R = Variable("R", value=3.2, unit="m", rel_tol=0.02)
R2 = R.clone(value=3.3)          # a new Variable; R itself is unchanged
print(R.value, R2.value)         # 3.2 3.3

# Profile variable
n_e = Variable("n_e", value=np.full(46, 1.0e20), unit="m^-3")
print(np.mean(n_e.value), n_e.value.shape)
```
