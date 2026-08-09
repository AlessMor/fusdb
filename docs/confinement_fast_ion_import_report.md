# Confinement, fast-ion, and wall-loading import report

Imported on 2026-08-08 into fusdb's relation/variable framework. Every formula
is exposed as an acausal `@relation`, uses registry variables and units, and is
source-labelled in its docstring. Existing defaults remain unchanged unless
the newly required inputs are supplied or a source-specific alternative is
explicitly included.

## Source snapshot

| Source | Revision inspected | Imported or reused |
|---|---:|---|
| PROCESS | `83d9f63f` | HELIOS pedestal temperature/density profiles; IPDG89/Ward fast-alpha beta; beam beta |
| cfspopcon | `57baea2` | Existing PRF tanh-pedestal profiles and Martin-Ryter threshold retained; no fast-ion pressure model exists upstream |
| FUSE.jl | `7e502fa` | Mode/profile semantics and thermal-plus-fast pressure aggregation |
| IMAS.jl (FUSE dependency) | `166bd10` | H-mode profile shape and corrected L-H threshold/accessibility model |
| bluemira | `dbbaa11` | PROCESS fast-alpha model names (`HENDER`, `WARD`); bluemira contains no independent formula |

## Framework integration

- PROCESS pedestal profiles are mode-tagged: temperature applies to H- and
  I-mode, while the density pedestal applies only to H-mode. If all pedestal
  controls are supplied, these relations supersede the generic parabolic shape
  for that output. FUSE/IMAS shapes remain explicit selectable alternatives.
- FUSE/IMAS H-mode accessibility is a checked-only certifier rather than an
  infinite solver residual. The corrected finite threshold includes the Martin
  density-rollover floor, isotope scaling, metallic-wall factor, and grad-B
  drift factor.
- PROCESS fast-alpha and beam-ion beta components remain separate from thermal
  beta. The Ward fast-alpha fit is the `beta_fast_alpha` default relation and
  the named IPDG89/Hender relation is a selectable alternative: include it and
  exclude Ward in a reactor configuration. There is no integer model switch.
  FUSE-style total pressure/beta diagnostics add the non-thermal components
  without changing legacy thermal-beta outputs.
- `S_wall` is a wall-geometry variable. Its default relation is `S_wall = A_p`;
  supplying a real wall area overrides that fallback, and neutron wall loading
  now divides by `S_wall`.

## Scope boundary

This imports the reduced models actually present in the surveyed system codes.
It does not invent a kinetic distribution-function solver: FUSE delegates
classical alpha slowing-down to its external `ALPHA` package, cfspopcon has no
fast-ion model, and bluemira delegates fast-alpha pressure to PROCESS.
