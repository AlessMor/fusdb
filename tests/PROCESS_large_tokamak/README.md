# PROCESS "Generic large tokamak" reproduction

Reproduces the plasma physics of the [PROCESS](https://github.com/ukaea/PROCESS)
large-tokamak examples in fusdb, and compares the two codes quantity by
quantity. **Every compared quantity agrees within 10%.**

```
pytest tests/PROCESS_large_tokamak          # 84 passed, 4 xfailed, ~50 s
```

## What is compared, and what is not

PROCESS's `introduction.ex.py` runs a constrained **optimisation**: minimise
major radius, 19 iteration variables, 26 constraints spanning plasma physics,
TF/CS coil engineering, radial build, burn time and net electric power. fusdb
models plasma physics only, so it cannot reproduce the optimisation — there are
no coil, radial-build, cost or plant-power models to constrain.

What it *can* reproduce, and what this suite does, is the plasma physics at a
design point. The comparison runs against PROCESS's **evaluation** mode
(`ioptimz = -2`), not its optimiser, and that choice is the whole point: against
an optimisation run fusdb must be handed PROCESS's converged radius, field,
current and profiles, so much of what comes out is downstream of what went in.
In evaluation mode the input file *is* the design vector and everything else is
forward-computed, so each difference is attributable to physics rather than to
bookkeeping.

Of PROCESS's five examples, two have physics fusdb can reproduce:

| example | reproducible? |
|---|---|
| `single_model_evaluation.ex.py` | **yes** — evaluation mode, tungsten sensitivity |
| `introduction.ex.py` | partly — its design point, not its optimisation |
| `scan.ex.py` | no — sweeps `b_tf_inboard_max`, a **TF-coil** limit fusdb has no model for |
| `optimum_solutions_comparison.ex.py` | no — varies required **net electric power**; the deliverable is the optimiser's solution vector |
| `vary_run_example.ex.py` | n/a — a solver-robustness utility, no physics reference |

## Layout

| path | role |
|---|---|
| `_process_mfile.py` | MFILE reader; handles `.DAT` and `.DAT.gz` transparently |
| `_process_fixture.py` | builds a fusdb `Reactor` from **any** PROCESS MFILE — the single PROCESS→fusdb mapping |
| `_process_compare.py` | the compared field list and its MFILE keys |
| `test_..._eval_point.py` | single design point, 24 quantities |
| `test_..._tungsten.py` | 5-point W sweep: trend, response magnitude, L-H agreement |
| `test_..._plasma_variants.py` | 9 runs, one input perturbed at a time — the **differential** test |
| `comparison_*.ipynb` | one notebook per test module |
| `reference/` | all 15 PROCESS runs (inputs, MFILEs, OUT.DATs) + the driver scripts |

`reference/` holds 48 files gzipped (42 MB → 9.2 MB). The driver `.ex.py`
scripts in `reference/scripts/` regenerate every run; they live in PROCESS's own
`examples/` directory upstream.

## Results

### Evaluation design point

`success=True`, `regime=h_mode`, 0 failed relations, no inputs beyond tolerance.

| quantity | error | | quantity | error |
|---|---|---|---|---|
| `V_p` | **+0.00%** | | `P_rad_core` | −0.32% |
| `qstar` | **+0.00%** | | `P_rad` | −0.45% |
| `n_e_avg` | −0.00% | | `P_aux` | −0.92% |
| `n_la` | −0.00% | | `W_th` | +1.45% |
| `beta` | +1.69% | | `f_BS` | +1.58% |
| `P_LH` | +2.58% | | `A_p` | +2.75% |
| `n_fuel_avg` | +3.05% | | `n_He4_avg` | +3.05% |
| `S_phi` | −3.18% | | `tau_E` | −3.49% |
| `Z_eff` | −4.94% | | `P_sync` | −5.65% |
| `P_loss` | +5.11% | | `P_fus` | +6.44% |
| `P_neutron` | +6.72% | | `Q_sci` | +8.22% |
| `P_sep` | +9.18% | | | |

24 quantities, all inside 10%. `qstar` and `V_p` are exact.

### Sweeps

Most sweep points are clean on all 24 fields. The recurring exceedance is
`P_sep`, and `Q_sci` inheriting it, at the high-radiation end — carried as
`xfail` with reasons in the test modules so an improvement surfaces as an
`xpass`. `P_sep = P_heating − P_rad` is a difference of two large numbers, so
few-percent residuals in each amplify once radiation approaches 70% of the
heating power.

**Both codes agree on which points can sustain H-mode.** At W = 1e-4 PROCESS's
own output reports `P_sep = 0.00` MW against `P_LH = 106.4` MW; fusdb *derives*
the regime and drops out of H-mode there, while PROCESS keeps using whichever
scaling its input file declares. Same physics, only one of them acts on it.

## Variable mapping

PROCESS and fusdb both carry these distinctions; the work was populating each
side correctly. All of it is commented at its point of use in
`_process_fixture.py`, and the conversions are **compared** (see `DERIVED` in
`_process_compare.py`) rather than asserted in prose.

**Four elongations, four variables.** PROCESS reports the separatrix elongation
(1.85000, its own `kappa` input), the areal one (1.71879 = `S_phi`/πa²), the
volume-equivalent `kappa_ipb` (1.68145 = `V_p`/2π²Ra²) and `kappa_95` (1.65179).
fusdb has a distinct variable for each, with its own relations, so each is
supplied its own value. PROCESS does not report an areal elongation directly, so
it is recovered from its cross-section — which is the definition fusdb's `kappa`
uses. Supplying `kappa_ipb` and letting `"IPB elongation from volume"` run
backwards recovers `V_p` **exactly**, since that is how PROCESS defines it.

**Two ion masses, one fusdb variable.** This one *is* a collapse: `afuel` was
aliased to **both** PROCESS's `m_fuel_amu` (2.5145) and `m_ions_total_amu`
(2.6981). PROCESS uses the fuel mass in its confinement scalings and the total
ion mass in its L-H thresholds, where the `2/A_i` isotope factor enters linearly
— a 7% difference. Split into `afuel_total` (see below).

**Composition denominators.** fusdb's `f_D`/`f_T`/`f_He4` are fractions of
**total ion** density; PROCESS quotes helium against **electron** density and its
50:50 split against **fuel** ions. Converted on the way in, and verified on the
way out: fusdb's volume-averaged fuel and helium densities both land +3.05%
against PROCESS's — the same factor on both channels, which is the `Z_eff`
difference propagating through quasineutrality rather than a conversion error.

**Stored energy.** PROCESS's `e_plasma_beta` includes fast-alpha beta (~14%
here); fusdb models no fast-alpha pressure, so `W_th` is compared against
`e_plasma_beta_thermal`. Confirmed by PROCESS's own
`W_thermal / P_loss` = 3.1340586588 against its reported `tau_E` = 3.1340586628.

**Converged ≠ input.** In the optimisation run the xenon fraction is iteration
variable `ixc = 135`, converging to 5.97e-4 against the input file's 3.8e-4.
`_process_fixture.py` reads composition from the MFILE, never the input file.
Related: the impurity array is **1-based** in `IN.DAT`, **0-based** in the MFILE.

**Grid resolution.** `P_aux` is a difference of large numbers and inherits the
profile quadrature error. fusdb's default 46-point grid leaves `Q` at +11.3%; the
fixture uses 201 points. `P_fus` is converged by 46 points, so its +6.4% is a
genuine model difference, not quadrature.

## Radiation decomposition (a source fix this drove)

The three radiation channels now each mean what their name says:

* **`P_brem`** — the **TOTAL** bremsstrahlung, hydrogenic and impurity alike
  (fusdb's `Z_eff`-weighted form, unchanged as the default).
* **`P_cool_imp`** *(new)* — the **total** impurity radiated power a cooling-rate
  `L_z` table returns: line, recombination continuum *and* the impurities' own
  bremsstrahlung. Mavrin, radas, Post-Jensen and PROCESS's tables now produce
  this, and the method-selection gate moved here with them.
* **`P_line`** — line (plus recombination) only, **derived**:
  `P_line = P_cool_imp − P_brem_imp`, where `P_brem_imp` is the impurity part of
  the bremsstrahlung, evaluated as the same local law at `Z_eff` minus at
  `Z_eff = 1` so the two are consistent term by term.

`P_rad = P_brem + P_line + P_sync` is then self-consistent, and it is
**numerically unchanged** — the restructure is a re-partition, not a correction to
the total. What it fixes is that assigning an `L_z` total straight to `P_line`
while `P_brem` was `Z_eff`-weighted counted the impurity bremsstrahlung twice
(34.4 MW here, 16% of `P_rad`), so every fixture had to override `P_brem` to the
hydrogenic form to compensate.

**cfspopcon does it differently, and that is now explicit.** It pairs its `L_z`
total with a hydrogenic-only `P_brem`, keeping the impurity bremsstrahlung inside
its impurity term. Self-consistent, but its `P_brem` is not the total and its
impurity term is not line-only — and its hydrogenic law is not the `Z_eff = 1`
limit of fusdb's, so the two are not interchangeable term by term.
`tests/cfspopcon_SPARC` selects that convention as a pair:
`Hydrogenic bremsstrahlung (cfspopcon)` plus
`Line radiation equals impurity cooling rate (cfspopcon convention)`.

The core/edge split is written on the **pre-split** channels (`P_cool_imp` and the
hydrogenic remainder) rather than on `P_line`/`P_brem`, because the two carry
different profile shapes: routing 34 MW of impurity bremsstrahlung through the
flatter total-bremsstrahlung shape moved `P_rad_core` +5.9% and pushed `P_sep`
past tolerance.

## Selector alignment

The input's `i_*` switches were resolved to concrete models so the comparison
measures physics rather than a selector mismatch:

| switch | value | model | fusdb action |
|---|---|---|---|
| `i_confinement_time` | 34 | IPB98(y,2) (confirmed by MFILE `tauelaw`) | already default |
| `i_density_limit` | 7 | Greenwald | already default |
| `i_beta_component` | 1 | limit on thermal beta | a constraint, not a producer |
| `i_bootstrap_current` | 4 | Sauter | **selected** |
| `i_l_h_threshold` | 19 | Martin 2008 **aspect-corrected** | **selected**, on the total ion mass — `P_LH` +23% → +2.6% |
| `i_plasma_current` | 4 | IPDG89 shaping at `kappa_95` | **selected** — `qstar` exact |
| `i_rad_loss` | 1 | core radiation only | **selected** (new relations, below) |
| `i_plasma_pedestal` | 1 | HELIOS pedestal | profile supplied directly |

`relation_include` **adds** rather than replaces, so each selection needs its
fusdb default explicitly excluded or the output is over-determined.

## Source changes this drove

`src/fusdb/relations/radiation/core_edge_process.py` — PROCESS's core/edge
radiation split and its power-balance levels, as gated `(PROCESS)` relations:
`Core radiation power`, `Edge radiation power`, `Plasma heating power`,
`Plasma loss power`, `Power crossing the separatrix`. New variables
`P_rad_core`, `P_rad_edge`, `radius_plasma_core_norm` (0.75),
`f_p_plasma_core_rad_reduction` (0.6). All gated: fusdb's defaults are unchanged
for every other reactor.

**Per-scaling H factors** (`_with_h_factor` in `src/fusdb/relation.py`). A
scaling declares `@relation(..., h_factor="H98_y2")` and the decorator injects
two optional constants — the scaling-specific H and the generic `H_factor` —
both defaulting to 1.0 and composing multiplicatively. The conditionality is
structural: only the active scaling is evaluated, so only its H is read.

This fixed a **live bug**: 50 of 65 `tau_E` producers took no H input at all
(PROCESS applies `hfact` in its orchestration layer, which the import correctly
skipped as non-physics, and nobody re-added it). `DEMO_2022` (`H98_y2: 0.98`)
and `STEP_2024` (`1.03`) were silently discarding their published confinement
enhancement.

**`Plasma shaping function for q_star (PROCESS IPDG89)`** — the same ITER Physics
Basis fit as fusdb's default, evaluated at `kappa_95` instead of the areal
`kappa`, which is what PROCESS's `i_plasma_current = 4` does. Gated. Selecting it
makes `qstar` exact.

**`afuel_total`** — new variable for the mass-averaged mass of *all* ions, split
out of `afuel` (which was aliased to both of PROCESS's masses), plus a gated
`L-H threshold Martin-2008 aspect nominal (total ion mass)`.

## Known model differences

These are real and are not expected to close:

* **`P_fus` +6.4%** — converged under grid refinement, and both codes use
  Bosch-Hale, so it is the profile-integration convention rather than the
  reactivity fit.
* **`Z_eff` −4.9%** — fusdb uses Mavrin mean charges; PROCESS uses its own
  `zav_of_te`, which the formula import deliberately skipped as composition.
* **`P_sync` −7.3%** — fusdb's own value on PROCESS's Albajar-Fidone relation.
* **`S_phi` −5.3%** — fusdb's Sauter cross-section against PROCESS's shape model.

## Reference provenance

`reference/introduction/` is a genuine run of the example's own input file
(`ifail = 1`, `sqsumsq = 2.35e-9`, PROCESS 3.4.2.dev101). It is **not** used by
the tests — see the scope note above — but is kept because
`reference/OPTIMUM.md` records how far the optimum moves between the example's
input and the `tests/` variant carried in the PROCESS repository.
