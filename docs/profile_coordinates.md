# Profile coordinates and normalization

This document defines the profile/geometry contract used by fusdb.

## Core coordinate

`rho` is the common normalized computational coordinate on `[0, 1]`, from the
core center to the separatrix. It is deliberately not identified with minor
radius, enclosed-volume radius, toroidal-flux radius, poloidal-flux radius, or a
mirror axial coordinate. All core profiles are represented once on this common
grid.

Geometry relations may provide normalized mappings on that grid:

- `v_norm(rho) = V(<rho) / V_p`
- `rho_minor(rho) = r / a`
- `rho_tor(rho) = sqrt(Phi / Phi_edge)`
- `rho_pol(rho) = sqrt(psi / psi_edge)`
- `rho_radial(rho)` for the reduced mirror radial coordinate
- `w_V(rho)`, proportional to `dV / d rho`

A relation must request the physical coordinate it actually assumes. Bare
`rho` is a sampling/index coordinate. It may also be used deliberately as the
integration parameter when a separate physical measure such as `w_V` is
supplied.

## Profiles

Every solved/generated profile is represented as

`profile(rho) = volume_average * shape(rho)`

with the shape normalized using the selected volume measure. In the `v_norm`
form,

`integral(shape, d v_norm) = 1`.

If no shape information is available the default shape is uniform. Scalar
reductions such as volume average, center value, maximum, and physical line
average remain registered variables and ordinary relations; they are not
metadata attached to the profile value.

A supplied absolute profile is retained as immutable source data. Its source
coordinate is retained as provenance. The profile used by relations is
reinterpolated onto the common `rho` grid through the current geometry mapping.
If geometry changes during a solve, the mapping and reinterpolation change with
it. The scalar average, rather than every source sample, is the amplitude degree
of freedom when the profile level is free.

Interpolation is strict: the current target mapping must remain within the
source-coordinate interval apart from roundoff-sized endpoint tolerance. Fusdb
does not silently extrapolate profiles. If a source omits a small axis or
separatrix interval, extending it is an explicit input-model choice. The GIGA
reactor demonstrates this with separate `*_rho01.csv` files that document the
endpoint extension while retaining the untouched digitized source files.

## Volume average

The general definition is

`<f>_V = integral(f w_V d rho) / integral(w_V d rho)`

where `w_V` is proportional to `dV/d rho`. If geometry instead supplies
`v_norm`, the equivalent direct-coordinate form is

`<f>_V = integral(f d v_norm) / integral(d v_norm)`.

For the reduced self-similar defaults, `v_norm = rho**2` and `w_V = rho`. This
preserves fusdb's historical discrete averaging exactly. The weight needs no
absolute factor involving `R`, `a`, or `kappa` because such a constant cancels
in a normalized average.

`Sauter self-similar profile volume mapping` is an opt-in tokamak provider whose
mapping depends on geometry. It is intentionally not the default: imported
equilibrium mappings are preferable, while the reduced defaults remain the
regression baseline.

## Geometry defaults and overrides

Reactor/device tags select geometry defaults. Geometry remains ordinary
variables plus relations; there is no reactor-wide `Geometry` object or
mandatory single geometry model.

Current reduced coordinate defaults are:

- tokamak: `rho_minor = rho`, `rho_tor = rho`, `v_norm = rho**2`, `w_V = rho`;
- stellarator: `rho_tor = rho`, and where source data are explicitly published
  against normalized minor radius also `rho_minor = rho`; `v_norm = rho**2` and
  `w_V = rho` until an equilibrium-derived mapping is supplied;
- mirror: `rho_radial = rho`, `v_norm = rho**2`, `w_V = rho` for the reduced
  radial model, with axial physics represented separately.

The tokamak `rho_tor = rho` relation is a migration-compatible fallback for
FUSE/IMAS H-mode profiles, which are defined versus normalized toroidal-flux
radius. It preserves the historical fusdb sampling when no equilibrium-derived
`rho_tor` is available; it is not a claim that `sqrt(Phi/Phi_edge) = r/a` for a
real equilibrium.

The stellarator and mirror mappings are likewise low-dimensional fallbacks, not
high-fidelity equilibrium claims. Future equilibrium adapters should reduce
external equilibria to these same mapping variables and replace the fallback
providers without changing profile consumers.

A supplied physical mapping (`rho_minor`, `rho_tor`, `rho_pol`, `rho_radial`,
`v_norm`, or `w_V`) is authoritative by default. Device fallback providers are
removed in that case so imported equilibrium data are not forced back onto a
reduced identity/self-similar mapping. A non-empty variable-local
`default_relation` explicitly opts into simultaneous provider
certification/reconciliation.

A list of selected provider relations means those providers are active
simultaneously and must reconcile. Multi-output providers are atomic. Mixed
geometry conventions are allowed when they can reconcile, and provider
provenance remains visible through the compiler maps.

## Static and dynamic mappings

Geometry-independent tokamak fallback mappings are materialized once as fixed
profile data by `build_relation_system`. They are not nonlinear unknowns, do not
add relation residuals, and do not add solver-domain rows. Their registry domains
are checked when the fixed mappings are constructed.

Supplied or geometry-derived mappings remain ordinary active variables and keep
their dependency ancestry and domain enforcement. Dynamic volume measures are
promoted into profile-generator graph dependencies; reduced deterministic
fallback measures remain constants so the legacy solver topology and runtime are
preserved.

The reduced stellarator and mirror providers intentionally remain explicit
relations. This keeps their compatibility contract visible and allows a future
stronger provider to supersede them through the normal selection mechanism.

## Line averages and coordinate-space reductions

A physical line average must name the physical coordinate it assumes. The
tokamak electron-density line average therefore uses `rho_minor`. The current
reduced stellarator/mirror `n_la` relation intentionally retains the historical
straight-`rho` average until a device-specific diagnostic or equilibrium model
is supplied.

Not every mathematical reduction over a profile is a physical line integral.
For example, the steady-state species-balance relations reduce a profile
residual to the scalar degree of freedom carried by each species fraction. That
reduction deliberately uses computational `rho`; changing it to `rho_minor` or
`w_V` would change the balance model rather than merely clarify geometry.

## Separatrix and edge

The common core coordinate stops at the separatrix: `rho = 1`. Edge/SOL profiles
are a separate subsystem and are not represented by extending core `rho` beyond
one.

For mirrors, the common profile coordinate represents a radial/nested-volume
description only. Genuinely axial physics enters through mirror-specific reduced
quantities or moments rather than overloading `rho` with a second physical
direction.

## Migration rule

Existing reduced-device results are the numerical regression baseline.
Relations that use `rho` merely for sampling, profile maxima, computational
coordinate reductions, or as the integration parameter paired with an explicit
measure may keep it. Relations that interpret the coordinate itself as a
physical radius or flux label must request the corresponding explicit mapping.

A numerical change caused only by replacing an identity fallback with an
explicitly supplied/equilibrium-derived mapping is a geometry-physics change and
must be isolated and tested; it is not an acceptable accidental side effect of
the architecture migration.
