---
status: Draft
bibliography: ../../../bibliography/bibliography.bib
---

# Confinement Modes (L, I, H)

## Abstract

!!! abstract "Definition / Summary"
    A **confinement mode** is the plasma's *transport state* — specifically whether an edge
    transport barrier exists. L-mode has none; H-mode has a pedestal formed by turbulence
    suppression at the edge; I-mode has a temperature pedestal without the strong particle
    barrier of H-mode.

    The mode is **not** a heating method. "Ohmic" describes how a plasma is heated, and an
    ohmic discharge may be L-mode or H-mode. Nor is the mode determined by the power
    crossing the separatrix alone: the transition is a reorganisation of the coupled edge
    state, and $P_\mathrm{sep}/P_\mathrm{LH}$ is best read as an *empirical accessibility
    indicator*, not as a rule that assigns the mode.

## The physical picture

Increasing $P_\mathrm{sep}$ raises the edge temperature and the pressure gradients, which
changes the radial electric field $E_r$, the $E\times B$ shear, and hence the turbulence.
The transition occurs when this coupled edge state reorganises into a different transport
regime. Three features matter for reduced modelling:

- **L$\rightarrow$H** is a relatively abrupt bifurcation, associated with strong turbulence
  suppression and formation of an edge transport barrier.
- **L$\rightarrow$I** is typically more gradual, and produces a temperature pedestal
  *without* the strong particle barrier characteristic of H-mode.
- **I$\rightarrow$H** transitions are common, and **H$\rightarrow$I** can also occur,
  depending on the plasma trajectory and control.

The thresholds depend on density, magnetic field, isotope, geometry, drift direction,
heating partition, rotation, collisionality — and on plasma *history*. See
[@doyle_chapter_2007] and [@yoshida_transport_2025] for reviews.

!!! warning "Hysteresis cannot come from a static ratio"
    A static relation such as $P_\mathrm{sep}/P_\mathrm{LH} = 1$ cannot reproduce
    hysteresis. The same instantaneous set of global variables can correspond to different
    modes depending on whether the plasma arrived from L-mode or H-mode. Margin factors
    (1.1, 1.2, 1.3 …) used by system codes are **engineering accessibility margins**, not a
    physical memory model.

    What *is* physical is that the forward and back transitions occur at **different
    powers**: an edge barrier, once formed, is sustained below the power needed to create
    it, so $P_\mathrm{HL} < P_\mathrm{LH}$. DIII-D back-transition measurements give
    $P_\mathrm{HL}/P_\mathrm{LH} \approx 0.35\text{–}0.70$, and ITER studies suggest the
    ratio may need to be as low as $\sim 0.5$. Between the two powers **both** L- and
    H-mode are self-consistent, and which one is occupied is set by history. A reduced code
    with no time axis can still represent this by treating the *declared* mode as the
    stand-in for history and letting it select among the self-consistent branches.

### The historical OH / L / H shorthand

Early confinement literature places "Ohmic", "L-mode" and "H-mode" side by side as if they
were three regimes. This comes from how L-mode was identified: auxiliary heating raised the
temperature but *degraded* the energy confinement time relative to the preceding ohmic
phase, and that degraded auxiliary-heated state was named the L-mode. Databases and plots
labelled OH / L / H are useful experimental shorthand, but they conflate two different
classifications. A cleaner ontology separates three axes:

| Axis | Values |
|---|---|
| Heating method | Ohmic, NBI, ECRH, ICRH, LH |
| Confinement mode | L, H, I, … |
| Ohmic transport regime | LOC, SOC |

"Ohmic L-mode" and "Ohmic H-mode" are both physically meaningful — the ITER Physics Basis
lists ohmic heating among the methods by which H-mode has been produced, and its H-mode
database contains ohmically heated H-mode discharges.

## Discriminants: three levels

For a reduced (systems) code it is useful to separate three levels of "discriminant":

| Level | Quantity | Meaning |
|---|---|---|
| **1 — accessibility** | $P_\mathrm{sep}/P_\mathrm{threshold}$ | is the mode *plausible* here? |
| **2 — edge-state criterion** | $F(n_\mathrm{sep}, T_\mathrm{sep}, B, R, \alpha_t, \rho_s, \dots)$ | does the edge state sit on the H side? |
| **3 — actual mode state** | pedestal + transport + turbulence + history | genuine prediction |

Most system codes operate at **Level 1**. cfspopcon's SepOS machinery is an example of
**Level 2**. A genuinely self-consistent prediction of the transition requires **Level 3**,
including temporal evolution and hysteresis.

### Level 1 — power-threshold discriminants

$$D_\mathrm{LH} \equiv \frac{P_\mathrm{sep}}{P_\mathrm{LH}}, \qquad
  D_\mathrm{LH} < 1 \Rightarrow \text{H-mode not accessible}, \qquad
  D_\mathrm{LH} > 1 \Rightarrow \text{H-mode potentially accessible.}$$

The Martin 2008 scaling, used by both PROCESS and cfspopcon, is approximately

$$P_\mathrm{LH} = 0.0488\, \bar{n}_e^{0.717}\, B_T^{0.803}\, S_p^{0.941}\,
                 \left(\tfrac{2}{M_i}\right),$$

with the appropriate unit convention. Two refinements matter in practice:

- **Low-density rollover.** Below a minimum density
  $n_{e,\mathrm{min}}(I_p, B, a, R/a)$ the required threshold *rises* again instead of
  continuing to fall with density (Ryter 2014). cfspopcon's default Martin branch includes
  it.
- **Isotope mass convention.** The $2/M_i$ factor may be evaluated on the *fuel* mass or on
  the *total* ion mass including impurities and helium ash; the two differ by ~9% at reactor
  conditions and the factor enters linearly.

The same idea extends to I-mode with $D_\mathrm{LI} \equiv P_\mathrm{sep}/P_\mathrm{LI}$.
Hubbard 2017 gives approximately $P_\mathrm{LI} \simeq 0.162\,\bar{n}_{e,20}\,B_T^{0.26}\,S_p$,
and Hubbard 2012 the earlier form $P_\mathrm{LI} \simeq 2.11\,I_p^{0.94}\,\bar{n}_{e,20}^{0.65}$.
Hubbard's regression is quoted as $P(\mathrm{L\text{-}I})/(n_e S)$ with $n_e$ the
**line-averaged** electron density, so an implementation that substitutes the volume average
is not evaluating the published fit.

!!! note "What a threshold crossing does and does not tell you"
    $P_\mathrm{sep} > P_\mathrm{LI}$ means the point has *access* to I-mode under the
    experimental circumstances represented by that scaling. It does not uniquely imply the
    plasma **is** in I-mode. Satisfying both $P_\mathrm{LI}$ and $P_\mathrm{LH}$ does not
    tell a 0-D code whether the trajectory produces I$\rightarrow$H, or remains in I-mode.

### Level 2 — the SepOS edge-state criterion

The Separatrix Operational Space (SepOS) criterion decides the transition from an **edge
state** rather than from total power. It evaluates
$n_{e,\mathrm{sep}}$, $T_{e,\mathrm{sep}}$, $B_T$, $R$, $M_i$, the poloidal sound Larmor
radius $\rho_{s,\mathrm{pol}}$, the turbulence parameter $\alpha_t$ and the critical
$\alpha_\mathrm{MHD}$, and forms a dimensionless ratio of a **flow-shear stabilisation**
term against several **turbulence destabilisation** terms:

$$D_\mathrm{SepOS} = \frac{\text{flow-shear stabilisation}}
                          {\text{electron} + \text{ion} + \text{kinetic destabilisation}},
  \qquad D_\mathrm{SepOS} > 1 \Rightarrow \text{H-mode}.$$

Conceptually this is much closer to the physics: the input power acts on
$n_\mathrm{sep}$ and $T_\mathrm{sep}$, and the resulting *edge state* determines which side
of the transition condition the plasma occupies. For a system code this is the important
distinction — a **power-threshold model** versus an **edge-state bifurcation proxy**.

### The pedestal is itself a mode discriminator

Changing only $\tau_E$ between modes is not internally consistent, because the mode also
sets the boundary condition for core transport:

- **H-mode** → a finite pedestal whose height and width can come from an EPED-like model;
- **L-mode** → no H-mode pedestal; the core profile effectively continues to the separatrix;
- **I-mode** and other improved regimes require their own edge treatment.

## How system codes actually handle confinement modes

Most reactor system codes do **not** predict the L$\leftrightarrow$I$\leftrightarrow$H
bifurcation from first-principles edge physics. They generally:

1. choose or assume a confinement branch;
2. solve the steady-state power balance with the corresponding confinement scaling / pedestal model;
3. calculate one or more regime-accessibility discriminants;
4. reject or flag the solution if the assumed mode is inconsistent with those discriminants.

The discriminant checks whether a mode is *plausible*; it does not cause the mode to emerge
dynamically.

| Code | How the mode is represented | Main discriminant | Does it switch? |
|---|---|---|---|
| **PROCESS** | user selects an L-, I- or H-mode $\tau_E$ scaling (`i_confinement_time`) | $P_\mathrm{sep}/P_\mathrm{LH}$; L-I threshold models also available | **No.** Scaling and transition constraint are selected separately |
| **HELIOS** | essentially an H-mode model with pedestal | $P_\mathrm{sep} > P_\mathrm{LH}$; good H-mode taken around $P_\mathrm{sep} \gtrsim 1.3\,P_\mathrm{LH}$ | Not a true state transition; confinement degrades near threshold |
| **cfspopcon** | confinement and transition models are separate calculations | $P_\mathrm{sep}/P_\mathrm{LH}$, $P_\mathrm{sep}/P_\mathrm{LI}$, or the more physical SepOS criterion | Used as operating-space diagnostics/masks rather than a state machine |
| **PLASMOD / Fable approach** | core and pedestal treated separately | local/global indicators such as $P_\mathrm{sep}/P_\mathrm{LH}$ | Conceptually intended to select the *pedestal treatment* by mode |
| **FUSE** | 1.5-D transport coupled to an EPED-type pedestal | operating limits handled separately | Published workflow is H-mode-oriented rather than an L/I/H bifurcation model |

PROCESS is particularly illustrative. `i_confinement_time` explicitly selects the
confinement law — L-mode, H-mode and Hubbard I-mode scalings are all available. Separately
it computes L-H and L-I threshold powers, and a constraint can require
$P_\mathrm{sep} \ge f_H P_\mathrm{LH}$ (or the opposite, to enforce an L-mode point). The
threshold therefore does not select IPB98(y,2) automatically: the user selects the
confinement model and the threshold checks consistency with that choice.

!!! note "Implementation detail worth knowing"
    FUSE's *current source* goes further than its published workflow. Its pedestal actor
    holds an explicit L/H hysteresis band — enter H-mode above $1.2\times$, drop out below
    $0.8\times$, and *hold the previous state in between* — and the mode drives the
    pedestal, the density and $Z_\mathrm{eff}$, not only $\tau_E$. That is a Level 1
    discriminant combined with a history term.

### The same problem outside fusion

A system of equations whose *structure* depends on its own solution is a well-studied
problem class, and the failure modes have names:

- **Power-flow PV/PQ bus switching.** A generator bus is PV until its reactive limit is hit,
  then becomes PQ. Repeated switching without convergence is called *bus type identification
  divergence*. The heuristic remedy — delay switching until the base solve has converged
  "sufficiently" — is criticised as subjective; modern treatments use a **complementarity /
  smoothing** formulation instead, which converges faster than logical switching.
- **Flash and phase-stability calculations.** The number of phases is unknown and
  solution-dependent. The key lesson: *a simulation based on a wrong prediction of the number
  of phases may converge, but its results have no physical meaning.* The remedy is a separate
  **stability test** (Michelsen's tangent-plane distance), integrated into the flash so the
  phase count is never fixed a priori.
- **Hybrid DAE simulation (Modelica).** Mode switching on state events is handled by **event
  iteration**; the pathology is **chattering**, and dedicated algorithms exist to detect and
  damp it.
- **Multiphase flow regime maps.** Flow regime selects different closure relations. Codes
  either interpolate across **transition regions** — acknowledged as artificially smooth, and
  unable to capture the relaxation timescale — or deliberately choose closure relations that
  are **continuous across transitions**.

The transferable conclusion is that *convergence is not validity*, and that repeated
switching between modes is a solver pathology rather than a physical bistability.

## A reduced definition

Putting these together, $\mathrm{mode} = f(P_\mathrm{sep}/P_\mathrm{LH})$ is not a
physically complete definition. A better reduced form is

$$\text{confinement mode} = F(\text{edge state},\; \text{transition accessibility},\;
                              \text{previous mode}),$$

with $P_\mathrm{sep}/P_\mathrm{LH}$ and $P_\mathrm{sep}/P_\mathrm{LI}$ as useful
*observables and constraints* rather than the mode variable itself. A SepOS-like continuous
discriminant plus explicit transition/history logic is substantially more physical while
remaining compatible with a reduced systems-code model.

## Status in `fusdb`

- `confinement_mode` is a tag group holding **transport states only** — `l_mode`, `h_mode`,
  `i_mode`. Ohmic is deliberately absent (it is a heating method).
- Level 1 discriminants exist: `P_LH` (default: PROCESS's Martin-2008 aspect-corrected form
  on the total ion mass), `P_LI_thresh` (Hubbard NF17 default), and the ratios
  `ratio_of_P_SOL_to_P_LH`, `ratio_of_P_SOL_to_P_LI`.
  See `fusdb.relations.confinement.confinement_modes_threshold`.
- A Level 2 discriminant is **ported but not yet reachable**:
  `fusdb.relations.edge.separatrix_operational_space` provides `SepOS_LH_transition`,
  `SepOS_MHD_limit` and `SepOS_density_limit` (Eich 2021 / Manz 2023). It requires a
  separatrix state that no reactor currently declares.
- **Hysteresis is represented.** `P_HL` (default $0.7\,P_\mathrm{LH}$, overridable via
  `f_HL_hysteresis`) is the back-transition power, and the h-mode certifier reads it while
  the l-mode certifier reads $P_\mathrm{LH}$. The two therefore *overlap*: in the band both
  modes are admissible, and the declared `confinement_mode` tag — the stand-in for history —
  selects the branch.
- A mode is **admissible** when its own solve's relations hold *and* all of its certifiers
  hold. Exactly one admissible mode is the answer; more than one is a hysteresis band; zero
  signals an over-constrained operating point rather than a confinement state.
- Today the mode changes $\tau_E$ and essentially nothing else — there is no pedestal model,
  so the L/H boundary-condition distinction described above is not yet represented.
- **I$\leftrightarrow$H transitions are not yet allowed.** I-mode has no upper certifier (no
  I-H threshold exists in any surveyed code), so opening that edge lets a pathological
  high-power branch certify as I-mode. It is blocked on physics, not on plumbing.
- The ohmic *transport* regimes LOC/SOC are not modelled.

## References and Links:

Key scalings referenced above, not yet in the bibliography:

- Martin *et al.*, *J. Phys.: Conf. Ser.* **123** 012033 (2008) — L-H power threshold.
- Ryter *et al.*, *Nucl. Fusion* **54** 083003 (2014) — low-density branch.
- Hubbard *et al.*, *Nucl. Fusion* **52** 114009 (2012); **57** 126039 (2017) — I-mode thresholds.
- Eich & Manz, *Nucl. Fusion* **61** 086017 (2021) — separatrix operational space.
- Rice *et al.*, *Nucl. Fusion* **60** 105001 (2020) — LOC/SOC phenomenology.

### See also:

- [Impurity transport](Impurity_transport.md)
- `../../7-boundary_and_sheath_physics/` — separatrix and SOL conditions that feed Level 2 discriminants

### Bibliography:

\bibliography
