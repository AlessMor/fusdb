# cfspopcon formula import — final report

Module-by-module port of cfspopcon's `formulas/` into fusdb `@relation`s. Every
imported relation carries the notice *"Adapted from cfspopcon; see README.md
section 'Third-party Notices'."* and a `# CHECK` marker for review; every new
`variables.yaml` entry carries an inline `# CHECK`.

## Summary
- **Relations: 224 → 315** (+91 imported across ~20 modules).
- **93** `# CHECK` markers in relations, **109** new `# CHECK` variables, **27** files with the cfspopcon notice.
- Suite green throughout: **22 passed, 8 xfailed** (the 8 phase-1 SPARC xfails are unchanged).
- **Zero new runtime dependencies** (numpy + yaml only; no ADAS/radas/xarray/netCDF).

## Conventions applied
- **Units:** fusdb SI/registry units; cfspopcon units converted inside each function (e.g. `I_p` A↔MA, `lambda_q` m↔mm, `P` W↔MW, temps keV↔J via `KEV_TO_J`, keV↔eV where a coefficient is eV-based).
- **Variable mapping:** cfspopcon glossary name → fusdb canonical var; same quantity+unit under a new name → **alias**; same quantity/different unit → reuse + convert.
- **Iterative cfspopcon solvers** (two-point model) imported as their **closed-form component relations**; fusdb's reconcile solves the coupling (zero-residual). The iterative drivers themselves were skipped.
- **Enum dispatchers** (lambda_q, L-I, momentum loss, radiated-power method) split into **per-variant relations**, gated by `default_relation`.

## Imported by group
- **plasma_current/** — safety_factor (`f_shaping`, `qstar`), bootstrap (`f_BS`), resistive_heating (Spitzer/neoclassical resistivity, trapped enhancement, current relaxation, loop voltage, ohmic power, inductive current), flux_consumption (7 fluxes + internal inductance/inductivity) + Barr surface-inductance sub-model (external/vertical inductance, `invmu_0_dLedR`, vertical field — `fa..fh` ported to scalar numpy, Barr coeffs).
- **scrape_off_layer/** — lambda_q (4 scalings), heat_flux (B_pol/B_tor/pitch, q∥, q⊥), separatrix_density, **two-point model** (separatrix temp, upstream pressure, target temp/density/flux basics+factors+combined, target q∥, 6 momentum-loss fits, required power-loss), reattachment (ionization volume, neutral-flux factor, reattachment time).
- **separatrix_conditions/** — power crossing separatrix, full Martin+Ryter L-H, L-I (3 scalings), P_SOL/P_LH & P_SOL/P_LI ratios; **SepOS** (critical alpha_MHD, poloidal sound Larmor radius, L-H/MHD/density condition functions, ion/electron sustainment power).
- **radiated_power/ + impurities/** — synchrotron, core radiated fraction, min-radiation targets; **impurity radiation** (Mavrin coronal [default] / Mavrin noncoronal / Post-Jensen, per-species over Mavrin's 11), **edge Lengyel** (cooling integral N/Ne/Ar + edge concentration). Per-species coefficients in `registry/dataset/radiation/*.yaml`.
- **profiles/** — temperature peaking (peak temps), Angioni density peaking (effective collisionality, electron variant → `density_peaking`, ion variant → `ion_density_peaking`). fusdb keeps independent electron/ion peaking factors for both density and temperature; the ion factors default to the electron value (`Default ion {density,temperature} peaking from electron`) and can be supplied/scaled independently. The bootstrap uses `nu_n = (ion + electron)/2`, matching cfspopcon.
- **metrics/** — heat-exhaust proxies `PB_over_R`, `PBpRnSq`.
- **plasma_pressure/beta** — reviewed, fully covered by fusdb (0 imported).

## Multi-producer gates to review (`default_relation`)
| Variable | Default | Opt-in alternatives |
|---|---|---|
| `density_peaking` (electron) | Electron Angioni | — |
| `ion_density_peaking` | equals electron (default) | Ion Angioni |
| `P_LH` | fusdb simplified | Martin-Ryter (full) |
| `P_LI_thresh` | HubbardNF17 | AUG, HubbardNF12 |
| `internal_inductance` | cylindrical | non-cylindrical |
| `lambda_q` | Eich regression 15 | Brunner, Eich 9, Eich 14 |
| `SOL_momentum_loss_fraction` | KotovReiter | Sang, Jarvinen, Moulton, PerezH, PerezL |
| `min_P_radiation` | from fraction | from L-H factor |
| `P_line` | Mavrin coronal | Mavrin noncoronal, Post-Jensen |
| `L_int` | Ne | N, Ar |

## Correctness / unit decisions that most need review
- **Mavrin coronal:** cfspopcon's body swaps temp/density variable names; fusdb uses the physically-correct `Lz=fit(T_e[keV])`, `q=n_e²·Σ(c_s·Lz_s)`.
- **Mavrin noncoronal:** cfspopcon labels temp bins "eV" but feeds keV, and omits the `1e38` density factor the coronal path uses; replicated bin handling, used physically-consistent `n_e²[m⁻³]`. **Unverified vs cfspopcon.**
- **Profile volume integrals** (brems/synchrotron/impurity radiation): rho-uniform `V_p·trapezoid` (fusdb's existing convention; cfspopcon volume-weights `Σ f·2ρ·dρ·V`). Same limitation the SPARC phase-1 xfails already document.
- **`inductive_plasma_current` = I_p·(1−f_BS)** (cfspopcon faithful; with external CD use `f_NI = f_BS + f_CD`).
- **normalized beta** (pre-existing fusdb): uses `beta_T·100` vs cfspopcon's `beta_total` — left as-is, flagged.

## Skipped / deferred
- **Skipped (reformulations/clashes):** `solve_for_input_power`, `switch_*_confinement_*_below_threshold`, acausal inverses (`plasma_current_from_qstar`), clamps (`require_P_rad_less_than_P_in`, core-seeded `P_radiation` clamp), `calc_inductive_plasma_current`'s f_BS-only assumption note; fusion-power & composition modules (fusdb treats these in more detail, incl. `zeff_and_dilution` and Te-dependent charge state).
- **Deferred (intricate/unverifiable):** `calc_neutral_pressure_kallenbach` (mm/eV-flux/degrees + eV-based `kappa_ez` of unstated unit).
- **Not adopted:** radas / ADAS tables (one-off OpenADAS→netCDF generator; only unique outputs are mean-Z [not needed] and table-exact Lz [Mavrin is its reference]). Used Mavrin/Post-Jensen polynomial fits instead.

## New registry additions
- **Species** (radiating impurities): Li, Be, C, N, O, Ne, Ar, Kr, Xe, W.
- **Data files:** per-species `polynomialfit_{mavrin_coronal,mavrin_noncoronal,post_jensen}_*.yaml` resources under `registry/dataset/radiation/` (noncoronal & Post-Jensen coefficients extracted programmatically from cfspopcon source — no transcription error).
- **109 new variables** (each `# CHECK`), plus aliases mapping cfspopcon names to existing fusdb vars (e.g. `triangularity_psi95→delta_95`, `elongation_psi95→kappa_95`, `bootstrap_fraction→f_BS`, `q_star→qstar`, `P_rad_impurity→P_line`, `poloidal_circumference→L_p`).

## How to review
Grep `# CHECK` in `src/fusdb/relations/` (relation bodies) and `src/fusdb/registry/variables.yaml` (new vars). Numerically-verified relations: safety factor, bootstrap, lambda_q, heat flux, resistive heating, flux consumption (+Barr vs exact-slicing reference), SepOS, two-point model, Mavrin coronal (Argon), Post-Jensen (Carbon). Explicitly **unverified** (no cfspopcon install here): Mavrin noncoronal, edge Lengyel.
