# The example's actual optimum, and which constraints set it

Source: `large_tokamak_MFILE.DAT` / `large_tokamak_OUT.DAT`, copied from
`~/Scrivania/PROCESS/examples/output/introduction/`. The input there is
**byte-identical** to `examples/data/large_tokamak_IN.DAT`, `ifail=1`,
`sqsumsq = 2.35e-9`, PROCESS 3.4.2.dev101. This is the real thing — the earlier
provenance caveat is gone.

## It differs materially from the tests/integration run

The three extra f-value lines in the `tests/` input variant move the optimum a
long way. Anything targeted at the old numbers is aimed at the wrong point:

| quantity | tests/integration | **examples (correct)** | shift |
|---|---|---|---|
| `B0` [T] | 4.9710 | **5.0813** | +2.2% |
| `q95` | 3.4137 | **3.0001** | −12.1% (now at its lower bound) |
| `T_e_avg` [keV] | 12.473 | **16.277** | +30.5% |
| `n_e_avg` [m⁻³] | 7.9792e19 | **8.0732e19** | +1.2% |
| `T_e0` [keV] | 25.715 | **36.261** | +41.0% |
| `n_e0` [m⁻³] | 1.0495e20 | **9.4739e19** | −9.7% |
| `I_p` [MA] | 16.451 | **19.135** | +16.3% |
| `P_fus` [MW] | 1625.1 | **2064.1** | +27.0% |
| `Q` | 20.359 | **11.361** | −44.2% |
| `tau_E` [s] | 3.1341 | **2.8355** | −9.5% |
| `Z_eff` | 2.5625 | **4.1949** | +63.7% |
| `f_BS` | 0.42283 | **0.37545** | −11.2% |
| `c_Xe` | 3.8e-4 | **1.2388e-3** | +226% |
| `f_He4` | 0.077911 | **0.068028** | −12.7% |
| `V_p` [m³] | 1888.171 | 1888.171 | unchanged |

`V_p` is identical because `R`, `A`, `kappa`, `triang` are all inputs — the
geometry never moved. Everything else did.

Still confirmed on the new reference: `f_c_plasma_bootstrap ==
f_c_plasma_bootstrap_sauter` exactly (so `i_bootstrap_current=4` → Sauter is
the right selection), and `tauelaw = "IPB98(y,2)"`.

Note the MFILE key rename: `f_nd_alpha_electron` → `f_nd_alpha_thermal_electron`.

## The objective is degenerate

`rmajor` converged to 8.000000000090218 against `boundl(3) = 8.0`. **The
objective hit its box bound, not a physics constraint.** Minimising R in fusdb
over the same bound will also trivially return 8.0. The meaningful comparison is
therefore not the objective value but the rest of the operating point the
constraint set picks *at* R = 8.

## Which constraints actually bind

From the OUT.DAT inequality table (normalised residue; ~0 means active). This is
what decides which constraints are worth translating.

**Binding (|residue| < 1e-3):**

| constraint | residue | fusdb? |
|---|---|---|
| Electron density upper limit | −8.5e-6 | **yes** — `f_GW`/`n_GW` |
| Alpha/energy confinement ratio ≥ 5 | −3.3e-5 | **plasma**, needs `tau_alpha` |
| TF quench hotspot current density | −1.6e-5 | no (engineering) |
| Net electric power ≥ 400 MW | −5.6e-5 | no (plant) |
| TF quench dump voltage ≤ 10 kV | −5.3e-5 | no (engineering) |
| TF SC critical current density | −1.0e-4 | no (engineering) |
| TF coil case stress | −2.7e-4 | no (engineering) |
| CS Tresca yield | −1.1e-3 | no (engineering) |
| CS SC temperature margin ≥ 1.5 K | −1.0e-3 | no (engineering) |

**Near-binding:** `P_sep·B/(q95·A·R) ≤ 10` at −5.7e-3 (**fusdb has this** as
`P_sep_B_over_q95AR`); burn time ≥ 7200 s at −2.8e-2.

**Slack (do not bother translating):** P_aux ≤ 200 MW (−0.095), P_sep > P_LH
(−0.591), ⟨β⟩ limit (−0.222), TF peak field (−0.146), CS EOF (−0.298), CS BOP
(−0.092), TF temp margin (−0.118), VV stress (−0.607), n_e0 > n_ped (−0.301),
TF conduit stress (−0.133), neutron wall load (−0.354), fusion power ≤ 3000 MW
(−0.312).

## What this means for the fusdb optimize test

Of the nine binding constraints, **seven are engineering** — TF/CS current
density, quench, stress, and plant net electric power. Those are what actually
stop the design shrinking, and fusdb models none of them. The plasma-side
binding set is just two: the density limit and the alpha-confinement ratio,
plus `P_sep·B/(q95·A·R)` near-binding.

So a fusdb `optimize` over only the plasma constraints is **under-constrained
relative to PROCESS** and will not land on PROCESS's optimum — it has strictly
fewer things stopping it. The honest test is therefore not "does fusdb find the
same optimum" but one of:

1. **Fix R = 8 (its bound) and the engineering-set outcomes** (`B0`, `q95` at
   its bound 3.0), then optimize the remaining plasma freedoms against the
   plasma constraints, and compare the operating point. This isolates the
   plasma physics, which is the only part fusdb claims.
2. **Optimize freely and report the gap**, documenting that fusdb's optimum is
   more aggressive precisely because the engineering constraints are absent —
   quantifying how much of PROCESS's design point is set by engineering rather
   than plasma physics. This is arguably the more interesting result.

Recommend (1) as the regression test and (2) as the notebook's headline.
