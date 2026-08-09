# Profile coordinates and normalization

This document defines the profile/geometry contract used by fusdb.

## Core coordinate

`rho` is the common normalized computational coordinate on `[0, 1]`, from the
core center to the separatrix. It is deliberately not identified with minor
radius, enclosed-volume radius, toroidal-flux radius, or poloidal-flux radius.
All core profiles are represented once on this common grid.

Geometry relations may provide normalized coordinate mappings on that grid,
for example:

- `v_norm(rho) = V(<rho) / V_p`
- `rho_minor(rho) = r / a`
- `rho_tor(rho) = sqrt(Phi / Phi_edge)`
- `rho_pol(rho) = sqrt(psi / psi_edge)`

A relation must request the physical coordinate it actually assumes. Bare
`rho` is only a sampling/index coordinate.

## Profiles

Every solved/generated profile is represented as

`profile(rho) = volume_average * shape(rho)`

with the shape normalized so that

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
reinterpolation must change with it.

Coordinate conversion is therefore treated like a geometry-dependent unit
conversion: it is deterministic, visible to the dependency graph, and is not
an independent set of solver degrees of freedom.

Interpolation is allowed only inside the supplied coordinate interval, apart
from a small numerical endpoint tolerance. Fusdb must not silently extrapolate
a core profile into an uncovered region.

## Volume average

The general definition is

`<f>_V = integral(f, d v_norm) / integral(1, d v_norm)`.

Until geometry mappings are fully wired through the relation graph, the
existing self-similar `rho` weighting remains the default implementation for
backward compatibility. For the default self-similar tokamak convention this
corresponds in the continuum to `v_norm = rho**2`.

## Geometry defaults and overrides

Reactor/device tags select geometry defaults. A `tokamak` uses Sauter geometry
as the default convention; stellarator and mirror relations may define their
own reduced geometry defaults.

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
can represent a radial/nested-volume description, while genuinely axial mirror
physics must initially enter through mirror-specific reduced quantities rather
than overloading `rho` with a second physical direction.

## Migration rule

Existing tokamak/Sauter results are the regression baseline. Relations that
currently interpret bare `rho` as normalized minor radius must be migrated one
at a time to an explicit `rho_minor` mapping. Numerical changes are physics
changes, not an acceptable side effect of this architectural refactor, and must
be isolated and quantified before becoming defaults.
