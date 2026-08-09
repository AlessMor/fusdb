# Profile coordinates and normalization

This document defines the profile/geometry contract used by fusdb.

## Core coordinate

`rho` is the common normalized computational coordinate on `[0, 1]`, from the
core center to the separatrix. It is deliberately not identified with minor
radius, enclosed-volume radius, toroidal-flux radius, poloidal-flux radius, or a
mirror axial coordinate. All core profiles are represented once on this common
grid.

Geometry relations may provide normalized coordinate mappings on that grid,
for example:

- `v_norm(rho) = V(<rho) / V_p`
- `rho_minor(rho) = r / a`
- `rho_tor(rho) = sqrt(Phi / Phi_edge)`
- `rho_pol(rho) = sqrt(psi / psi_edge)`
- `rho_radial(rho)` for a reduced mirror radial coordinate

A relation must request the physical coordinate it actually assumes. Bare
`rho` is only a sampling/index coordinate.

## Profiles

Every solved/generated profile is represented as

`profile(rho) = volume_average * shape(rho)`

with the shape normalized using the geometry-provided volume measure. In the
`v_norm` form this means

`integral(shape, d v_norm) = 1`.

If no shape information is available the default shape is uniform. Scalar
reductions such as volume average, center value, maximum and physical line
average remain separate registered variables and ordinary relations; they are
not metadata attached to the profile value.

A supplied absolute profile is retained as source data. Its samples determine
the input shape and initial volume average. During reconciliation the scalar
average is the amplitude degree of freedom rather than one independent solver
unknown per profile point.

## Coordinate conversion

External profiles may be supplied on another normalized coordinate/grid. The
source samples and source coordinate are retained, while the physical profile
used by relations is evaluated on the common `rho` grid through the current
geometry mapping. If geometry changes during a solve, that mapping and the
reinterpolation change with it.

Coordinate conversion is therefore treated like a geometry-dependent unit
conversion: it is deterministic, visible to the dependency graph, and is not
an independent set of solver degrees of freedom.

Interpolation is allowed only inside the supplied coordinate interval, apart
from a small numerical endpoint tolerance. Fusdb must not silently extrapolate
a core profile into an uncovered region.

A supplied physical mapping (`rho_minor`, `rho_tor`, `rho_radial`, `v_norm`, or
`w_V`) is authoritative by default. Device fallback providers are removed from
the compiled system in that case, so imported equilibrium data are not forced
back onto a reduced default mapping. A non-empty variable-local
`default_relation` explicitly opts back into simultaneous provider
certification/reconciliation.

## Volume average

The general definition is

`<f>_V = integral(f w_V d rho) / integral(w_V d rho)`

where `w_V` is proportional to `dV/drho`. If geometry instead supplies
`v_norm`, the equivalent direct-coordinate form is

`<f>_V = integral(f d v_norm) / integral(d v_norm)`.

For the self-similar reduced defaults, `v_norm = rho**2` and `w_V = rho`. This
choice preserves fusdb's historical discrete tokamak averaging exactly.

## Geometry defaults and overrides

Reactor/device tags select geometry defaults. Geometry remains ordinary
variables plus relations; there is no reactor-wide `Geometry` object or
mandatory single geometry model.

Current reduced coordinate defaults are:

- tokamak: `rho_minor = rho`, `v_norm = rho**2`, `w_V = rho`;
- stellarator: `rho_tor = rho`, `v_norm = rho**2`, `w_V = rho` when no
  equilibrium-derived mapping is supplied;
- mirror: `rho_radial = rho`, `v_norm = rho**2`, `w_V = rho` for the reduced
  radial model, with axial physics deliberately represented separately.

The stellarator and mirror defaults are low-dimensional fallbacks, not claims of
high-fidelity equilibrium geometry. Future VMEC/stellarator or mirror-equilibrium
adapters should reduce external equilibria to these same mapping variables and
replace the fallback providers without changing profile consumers.

A scenario may override a variable's `default_relation`. A list of relations
means those providers are active simultaneously and must reconcile. A
multi-output provider is atomic: selecting it makes all of its outputs part of
that physical model, and incompatible explicit provider selections must fail
at compile time rather than be silently resolved.

Mixed geometry conventions are allowed when they can reconcile. Provenance of
each geometry quantity should remain visible in compiler/verification output.

## Separatrix and edge

The common core coordinate stops at the separatrix: `rho = 1`. Edge/SOL
profiles are a separate future subsystem and are not represented by extending
core `rho` beyond one.

For mirrors, the same limitation is explicit: the common profile coordinate
represents a radial/nested-volume description only. Genuinely axial mirror
physics enters through mirror-specific reduced quantities/moments rather than
overloading `rho` with a second physical direction.

## Migration rule

Existing tokamak/Sauter results are the regression baseline. Relations that
still interpret bare `rho` as a physical coordinate must be migrated one at a
time to an explicit mapping such as `rho_minor`, `rho_tor`, or `rho_radial`.
Numerical changes are physics changes, not an acceptable side effect of this
architectural refactor, and must be isolated and quantified before becoming
defaults.
