# Economic Viability of Fusion Power Plants

Positive net electric power is necessary for a fusion power plant, but it is not sufficient for economic viability. A commercial plant must also recover the costs of construction, financing, replacement, operation, maintenance, and consumed components over its lifetime.

This page embeds the **Fusion Power Plant Economics Calculator** developed by **Andrew W. Lo** as an interactive implementation of the framework published by **Whyte, Lo, Bielajew, Hancock, Moeykens & Shaw** in *Criteria for the economic viability of fusion power plants*, *Journal of Fusion Energy* 45, 49 (2026).

!!! info "Credit and source"
    The calculator below is an external project developed by **Andrew W. Lo** and is not part of `fusdb`. Its source is available at [andrewwlo/fusioneconomics](https://github.com/andrewwlo/fusioneconomics). The underlying peer-reviewed model is D. G. Whyte, A. Lo, R. Bielajew, M. Hancock, R. Moeykens & G. Shaw, *Criteria for the economic viability of fusion power plants*, *Journal of Fusion Energy* 45, 49 (2026), [doi:10.1007/s10894-026-00577-9](https://doi.org/10.1007/s10894-026-00577-9).

## Interactive calculator

<div style="width:100%; min-height:1000px; margin:1rem 0;">
  <iframe
    src="https://andrewwlo.github.io/fusioneconomics/"
    title="Fusion Power Plant Economics Calculator — Andrew W. Lo"
    loading="lazy"
    style="width:100%; height:1000px; border:1px solid #d0d7de; border-radius:8px;"
    allowfullscreen>
  </iframe>
</div>

If the embedded page is blocked by the browser or by a future change to the external site's embedding policy, open the [calculator directly](https://andrewwlo.github.io/fusioneconomics/).

## Purpose of the model

The model asks a deliberately high-level question:

> What combinations of fusion power density, component lifetime, replacement time, plant cost, financing, energy price, and conversion efficiency are necessary for a fusion power plant to have a positive economic return?

Its organizing quantity is an **economic gain**,

\[
Q_\mathrm{econ}=\frac{C_\mathrm{gain}}{C_\mathrm{cost}},
\]

where the numerator is the economic rate associated with the energy sold and the denominator is the combined modeled economic cost rate. A necessary condition for positive economic gain is

\[
Q_\mathrm{econ}\ge 1.
\]

The authors explicitly describe this as **necessary but insufficient** for real-world commercial viability: the framework intentionally excludes some project-specific costs and risks.

## The control-surface abstraction

A central assumption is that any fusion power plant can be enclosed by a conceptual energy-capture surface \(S\). All useful fusion energy must pass through this surface before it can become a commercial product.

The framework therefore normalizes many engineering and cost quantities to \(S\), using variables such as:

- fusion power density \(P_f/S\);
- component energy-fluence limit \(X_S\);
- replacement cost per unit surface area;
- total plant cost per unit surface area.

This normalization is what makes the framework largely independent of absolute plant power and applicable, at least at this level, to different confinement concepts and fuel cycles.

## Temporal-equilibrium assumption

The paper assumes that economic gain and cost rates can be represented in a repeating **temporal equilibrium** over a plant lifetime much longer than an individual operating/replacement cycle.

Each cycle contains two periods:

1. an operating period \(\tau_\mathrm{op}\), during which fusion power is produced;
2. a replacement/refurbishment period \(\tau_\mathrm{rep}\), during which no fusion power is produced.

Thus

\[
\tau_\mathrm{cycle}=\tau_\mathrm{op}+\tau_\mathrm{rep}.
\]

The operating time is linked to the energy-fluence limit of the replaceable surface:

\[
\tau_\mathrm{op}=\frac{X_S}{P_f/S}.
\]

The resulting utilization factor is

\[
U=\left[1+\frac{(P_f/S)\tau_\mathrm{rep}}{X_S}\right]^{-1}.
\]

This expression captures one of the framework's central trade-offs: increasing fusion power density raises revenue while operating, but can shorten component life and increase the fraction of time spent replacing exposed components.

## Economic gain rate

The economic value of the energy product is written in terms of the net price of energy, conversion efficiency, fusion power density, and utilization. In the notation of the paper,

\[
C_\mathrm{gain}
=8.76\times10^{-3}\,
POE_\mathrm{net}\,
\eta_E\,
(P_f/S)\,U,
\]

with the units chosen so the rate is expressed per unit capture-surface area per year.

This means economic gain is improved by:

- higher energy price;
- higher conversion efficiency;
- higher useful fusion power density;
- higher utilization.

However, these quantities are not independent because power density also affects component replacement frequency through the fluence limit.

## Cost terms

The framework separates several cost contributions.

### Consumable fusion targets

For concepts requiring discrete targets, the model includes a target cost rate proportional to the energy produced. The framework does not prescribe the target technology; the user must supply a cost per target or per fusion yield appropriate to the concept.

For concepts without discrete consumable targets, this contribution can be set to zero.

### Replacement of the energy-capture surface

The replaceable surface and associated components incur a replacement cost rate. The parameter is normalized per unit area of \(S\) and is intended to include the full replacement process: fabrication, qualification, installation, removal, and disposal as appropriate.

The paper stresses that \(S\) is a normalization device, not literally just a thin surface. The replaced object may include a three-dimensional blanket or other hardware behind the plasma-facing surface.

### Fixed plant, financing, and O&M cost

Construction and fixed plant costs are annualized using a standard amortization relation with a **real interest rate**. This is a simplifying financial model: it captures the cost of capital transparently but does not attempt to represent the full complexity of real project finance, depreciation, changing interest rates, tax structure, or other contractual details.

Fixed O&M is also represented at this aggregated level.

## Main assumptions and limitations

The most important assumptions of the Whyte–Lo framework are:

1. **All fusion energy passes through a common control surface.** This permits concept-independent normalization but compresses many geometry and technology details into surface-normalized parameters.

2. **Economic rates are in temporal equilibrium.** The model treats plant operation and component replacement as a repeating cycle with constant long-term average rates.

3. **The plant lifetime is much longer than an operating/replacement cycle.** This allows replacement costs to be treated as a recurring average rate.

4. **Fusion power is constant during the operating portion of a cycle and zero during replacement.** Detailed ramp-up, ramp-down, partial-power operation, storage, or dispatch are not modeled.

5. **Component degradation is represented by a single energy-fluence limit \(X_S\).** The model does not separately resolve neutron dpa, helium production, thermal fatigue, erosion, corrosion, coolant chemistry, or other failure mechanisms.

6. **Replacement duration includes all time in which the plant cannot produce the commercial product.** Decommissioning and recommissioning associated with replacement must therefore be included in \(\tau_\mathrm{rep}\).

7. **Plant and replacement costs are aggregated and normalized to area.** Detailed cost accounts are not individually modeled.

8. **Financing uses a constant real interest rate and amortization model.** This is useful for transparent sensitivity analysis but is not a project-finance model.

9. **The raw D-T fuel cost is treated as negligible in the base framework.** Discrete engineered fusion targets, when relevant, are treated separately as consumables.

10. **\(Q_\mathrm{econ}\ge1\) is only a necessary condition.** Regulatory risk, construction overruns, market structure, taxes, insurance, supply-chain constraints, technological learning, and many other real commercial factors can still determine whether a project is investable.

## Representative baseline assumptions

The paper explores wide ranges rather than claiming one definitive fusion cost case. For orientation, its example/base-case choices include approximately:

- \(\tau_\mathrm{rep}=0.1\) year, motivated by the timescale of a typical fission refueling outage;
- energy-conversion efficiency \(\eta_E=0.4\);
- real interest rate \(i=2\%\);
- net electricity price around \(110\ \$/\mathrm{MWh}\), derived from an approximate \(160\ \$/\mathrm{MWh}\) retail electricity price with generation taken as roughly two-thirds of that value;
- \(X_S\approx3.125\ \mathrm{MW\,y/m^2}\) as an illustrative D-T fast-neutron fluence limit;
- plant areal cost \(\sim10\ \mathrm{M\$/m^2}\);
- replaceable-surface cost \(\sim0.3\ \mathrm{M\$/m^2}\), about \(3\%\) of the assumed plant areal cost;
- zero target cost in the baseline, with target cost subsequently varied for concepts where it matters.

These are **illustrative assumptions**, not predictions. The paper deliberately varies them widely because the required values for commercial fusion plants remain highly uncertain.

## What the model teaches

A major result of the framework is that fusion economics cannot be reduced to "make the highest possible power density" or "make the cheapest possible machine." Power density, component life, replacement time, replacement cost, capital cost, and financing interact nonlinearly.

In particular:

- very high power density can become counterproductive if it causes frequent long outages;
- long component life is valuable, but fast and inexpensive replacement can partly compensate for limited lifetime;
- financing cost matters strongly for capital-intensive fusion plants;
- sufficiently low power density does not automatically guarantee economic viability because fixed plant costs must still be recovered from the energy sold;
- replaceable components must behave economically like consumables: their cost and replacement time have to remain small enough relative to the value of the energy produced.

The paper's sensitivity studies therefore emphasize **joint design requirements** rather than one optimal scalar target.

## Relation to `fusdb`

This model is a good example of why economic analysis fits naturally into `fusdb`'s relation-centric approach. Its quantities can be separated into reusable relations:

- fusion power density;
- energy conversion efficiency;
- component fluence and lifetime;
- utilization;
- replacement rate and cost;
- financing/amortization;
- electricity price and economic gain.

A future `fusdb` economics model could therefore connect these economic relations directly to reactor physics and engineering relations without treating economics as an isolated post-processing step. Alternative cost or lifetime assumptions could then be exchanged in the same way as alternative confinement or radiation relations.

## References and attribution

- D. G. Whyte, A. Lo, R. Bielajew, M. Hancock, R. Moeykens & G. Shaw, **Criteria for the economic viability of fusion power plants**, *Journal of Fusion Energy* **45**, 49 (2026), [doi:10.1007/s10894-026-00577-9](https://doi.org/10.1007/s10894-026-00577-9).
- **Andrew W. Lo**, [Fusion Power Plant Economics Calculator](https://andrewwlo.github.io/fusioneconomics/), with [source code on GitHub](https://github.com/andrewwlo/fusioneconomics).

The paper credits **D. G. Whyte** with deriving the principal economic framework and graphical evaluations; **A. Lo** with the viable-region analysis, additional financial considerations, and development of the calculator; **M. Hancock** with costing-parameter definitions; **G. Shaw** and **R. Bielajew** with graphical-evaluation code and fusion technical definitions; and **R. Moeykens** with the framework visualization.
