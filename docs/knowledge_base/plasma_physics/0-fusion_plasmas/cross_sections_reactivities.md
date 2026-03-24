# Nuclear Fusion Reaction Rate, Reactivity and Cross-Section

!!! abstract "The Reaction Rate"
    $$
    R_r(T) = n_a * n_b * \langle \sigma v \rangle (T) * V
    $$

### Reaction Rate

Measuring the reaction rate of a fusion rection often reduces to finding a formula for the average reactivity of a reaction depending on temperature (or better, energy).

### Reactivity

The averaged reactivity is defined as the integral of the cross section of the reaction over the velocity distribution of the incident particles:

$$
\langle \sigma v \rangle = \int_0^\infty \sigma(v)vf(v)\,\mathrm{d}v.
$$

A Maxwellian distribution for the velocity distribution ($f_\mathrm{MB}(v) = 4\pi v^2 \left(\frac{\mu}{2\pi k_\mathrm{B}T}\right)^{3/2}\exp\left(-\frac{\mu v_i^2}{2k_\mathrm{B}T}\right)$) is often assumed for this purpose, resulting in the expression:

$$
\langle \sigma v \rangle = \frac{4}{\sqrt{2\pi\mu}}\frac{1}{(k_\mathrm{B}T)^{3/2}} \int_0^\infty \sigma(E)E\exp\left(-\frac{E}{k_\mathrm{B}T}\right)\,\mathrm{d}E,
$$

with $\mu = m_a m_X/(m_a+m_X)$ the reduced mass of the colliding particles.
<!-- NOTE: Here T is expressed in K -->

### Cross-Section

Considering a generic reaction a + X -> Y + b, the cross section $\sigma$($v$) = $\frac{number of reactions/nuclei X/ unit time}{number of incident particles/cm^2/unit time}$ [cm$^2$] 

Often the particles move with a distribution of relative velocities, $f(v)$
[@clayton_1983]
<!-- NOTE: Here velocity v and energy may be used interchangeably. This definition assumes particles X are static. -->
<!-- TODO: explain difference between lab and com reference frames -->

## Fusion plasma reactivity fits:

<div style="width: 100%; height: 760px; border: 1px solid #e1e4e5;">
  <iframe
    src="../../../code_docs/reactivity_plotter.html"
    style="width: 100%; height: 100%; border: 0;"
    loading="lazy"
  ></iframe>
</div>

The implemented `fusdb` relations can be browsed in the
[reactivity API reference](../../../code_docs/api/fusdb/relations/reactivities/reactivity_functions.md#fusdb.relations.reactivities.reactivity_functions)
and compared in the
[interactive reactivity plotter](../../../getting_started/reactivity_plotter.md).

### Bosch and Hale, 1992

The most renowened analytical fits for fusion relevant applications were defined by Bosch and Hale in 1992 [@bosch_hale_1992] (check also the corrigendum of 1993). Bosch & Hale provide a 9-parameters fit for $\sigma$(E) and Maxwellian $\langle \sigma \rangle$(T) for light ions fusion reactions, based on improved evaluations using R-matrix methods and newer measurements than their predecessors.

It is implemented in both cfspopcon and PROCESS codes.

In `fusdb`, Bosch-Hale reactivities are available for
[DT](../../../code_docs/api/fusdb/relations/reactivities/reactivity_functions.md#fusdb.relations.reactivities.reactivity_functions.sigmav_DT_BoschHale),
[DD](../../../code_docs/api/fusdb/relations/reactivities/reactivity_functions.md#fusdb.relations.reactivities.reactivity_functions.sigmav_DD_BoschHale),
and its two channels:
[DDn](../../../code_docs/api/fusdb/relations/reactivities/reactivity_functions.md#fusdb.relations.reactivities.reactivity_functions.sigmav_DDn_BoschHale),
[DDp](../../../code_docs/api/fusdb/relations/reactivities/reactivity_functions.md#fusdb.relations.reactivities.reactivity_functions.sigmav_DDp_BoschHale),
[DHe3](../../../code_docs/api/fusdb/relations/reactivities/reactivity_functions.md#fusdb.relations.reactivities.reactivity_functions.sigmav_DHe3_BoschHale).

### NRL plasma formualry, 1972 - ongoing

The NRL plasma formulary [@beresnyak_2023_nrl_plasma_formulary] fits and tabulated data comes from the 5-parameters fit of [@duane_1972] and reported also in [@miley_1974]. These fits are generally accepted in publications, but may give different results if compared to more modern data.

It also cites a more modern 3-parameters fit [@li_2008].

In `fusdb`, NRL reactivities are available for
[DT](../../../code_docs/api/fusdb/relations/reactivities/reactivity_functions.md#fusdb.relations.reactivities.reactivity_functions.sigmav_DT_NRL),
[DD](../../../code_docs/api/fusdb/relations/reactivities/reactivity_functions.md#fusdb.relations.reactivities.reactivity_functions.sigmav_DD_NRL),
[DHe3](../../../code_docs/api/fusdb/relations/reactivities/reactivity_functions.md#fusdb.relations.reactivities.reactivity_functions.sigmav_DHe3_NRL),
[TT](../../../code_docs/api/fusdb/relations/reactivities/reactivity_functions.md#fusdb.relations.reactivities.reactivity_functions.sigmav_TT_NRL),
[THe3](../../../code_docs/api/fusdb/relations/reactivities/reactivity_functions.md#fusdb.relations.reactivities.reactivity_functions.sigmav_THe3_NRL).


### ENDF/B, 1966 - ongoing

The Evaluated Nuclear Data File database includes experimental data for several nuclear reactions, including collisions of deuterons, tritons and helium-3 with other light nuclei. The latest release (as of 2026) is ENDF/B-VIII.1 [@nobre_endfb-viii1_2025].

In `fusdb`, ENDF/B reactivities are available for
[DT](../../../code_docs/api/fusdb/relations/reactivities/reactivity_functions.md#fusdb.relations.reactivities.reactivity_functions.sigmav_DT_ENDFB_VIII1),
[DD](../../../code_docs/api/fusdb/relations/reactivities/reactivity_functions.md#fusdb.relations.reactivities.reactivity_functions.sigmav_DD_ENDFB_VIII1) and its channels:
[DDn](../../../code_docs/api/fusdb/relations/reactivities/reactivity_functions.md#fusdb.relations.reactivities.reactivity_functions.sigmav_DDn_ENDFB_VIII1),
[DDp](../../../code_docs/api/fusdb/relations/reactivities/reactivity_functions.md#fusdb.relations.reactivities.reactivity_functions.sigmav_DDp_ENDFB_VIII1),
[DHe3](../../../code_docs/api/fusdb/relations/reactivities/reactivity_functions.md#fusdb.relations.reactivities.reactivity_functions.sigmav_DHe3_ENDFB_VIII1),
[TT](../../../code_docs/api/fusdb/relations/reactivities/reactivity_functions.md#fusdb.relations.reactivities.reactivity_functions.sigmav_TT_ENDFB_VIII1),
[He3He3](../../../code_docs/api/fusdb/relations/reactivities/reactivity_functions.md#fusdb.relations.reactivities.reactivity_functions.sigmav_He3He3_ENDFB_VIII1),
[THe3 total](../../../code_docs/api/fusdb/relations/reactivities/reactivity_functions.md#fusdb.relations.reactivities.reactivity_functions.sigmav_THe3_ENDFB_VIII1),
and branch-resolved
[THe3_D](../../../code_docs/api/fusdb/relations/reactivities/reactivity_functions.md#fusdb.relations.reactivities.reactivity_functions.sigmav_THe3_D_ENDFB_VIII1)
and
[THe3_np](../../../code_docs/api/fusdb/relations/reactivities/reactivity_functions.md#fusdb.relations.reactivities.reactivity_functions.sigmav_THe3_np_ENDFB_VIII1).

### Other fits:
[@caughlan_fowler_1988], [@hively_1977]

In `fusdb`, this currently includes
[DT Hively](../../../code_docs/api/fusdb/relations/reactivities/reactivity_functions.md#fusdb.relations.reactivities.reactivity_functions.sigmav_DT_Hively),
[DD Hively](../../../code_docs/api/fusdb/relations/reactivities/reactivity_functions.md#fusdb.relations.reactivities.reactivity_functions.sigmav_DD_Hively) and its two channels
[DDn Hively](../../../code_docs/api/fusdb/relations/reactivities/reactivity_functions.md#fusdb.relations.reactivities.reactivity_functions.sigmav_DDn_Hively),
[DDp Hively](../../../code_docs/api/fusdb/relations/reactivities/reactivity_functions.md#fusdb.relations.reactivities.reactivity_functions.sigmav_DDp_Hively),
[TT CF88](../../../code_docs/api/fusdb/relations/reactivities/reactivity_functions.md#fusdb.relations.reactivities.reactivity_functions.sigmav_TT_CF88),
[He3He3 CF88](../../../code_docs/api/fusdb/relations/reactivities/reactivity_functions.md#fusdb.relations.reactivities.reactivity_functions.sigmav_He3He3_CF88),
and
[THe3 CF88](../../../code_docs/api/fusdb/relations/reactivities/reactivity_functions.md#fusdb.relations.reactivities.reactivity_functions.sigmav_THe3_CF88) and its channels:
[THe3_D CF88](../../../code_docs/api/fusdb/relations/reactivities/reactivity_functions.md#fusdb.relations.reactivities.reactivity_functions.sigmav_THe3_D_CF88).
[THe3_np CF88](../../../code_docs/api/fusdb/relations/reactivities/reactivity_functions.md#fusdb.relations.reactivities.reactivity_functions.sigmav_THe3_np_CF88).

## References and Links:

### See also:

- [SciPython: Nuclear fusion cross-sections](https://scipython.com/blog/nuclear-fusion-cross-sections/)

- A comprehensive discussion on the reaction rate is done on [@freidberg_2007]

- ENDF website: [https://www.nndc.bnl.gov/endf/](https://www.nndc.bnl.gov/endf/)

### Bibliography:
\bibliography
