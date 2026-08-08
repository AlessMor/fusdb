# Fusion Power Economics

Fusion power economics connects reactor physics and plant engineering to the quantities that determine whether a plant can produce electricity competitively. The key distinction is between **scientific performance**, **engineering performance**, and **economic performance**.

A plasma can have a high scientific gain, \(Q_\mathrm{sci}=P_\mathrm{fus}/P_\mathrm{heat}\), without producing net electricity. The fusion power must first be converted into useful thermal or electrical power, while auxiliary heating, magnets, cryogenics, pumps, tritium systems, controls, and other plant loads consume part of the generated power. Even a plant with positive net electric output may still be uneconomic if its capital cost, replacement rate, financing cost, or downtime are too high.

For this reason it is useful to separate three levels:

1. **Plasma power balance** — fusion power, external heating, radiation and transport losses.
2. **Plant power balance** — gross electric generation minus recirculating and house loads.
3. **Techno-economic balance** — capital cost, component lifetime, maintenance, availability, financing, and electricity production over the plant life.

The two interactive tools embedded in this section illustrate complementary parts of this chain:

- [Fusion Energy Base — Fusion Power Plant Simulator](power_balance.md) focuses primarily on the **power-flow consequences** of scientific gain, heating-system efficiency, conversion efficiency, pulse rate, blanket multiplication, and house load.
- [Andrew Lo — Fusion Power Plant Economics Calculator](economic_viability.md) focuses on the **economic viability** of a fusion power plant, following the framework described by Whyte *et al.* for relating plant performance, capital cost, component lifetime, replacement, and financing to the value of the electricity produced.

!!! note "Scope"
    These tools are educational and exploratory models, not substitutes for a full systems-code cost model or a project-specific financial analysis. Their value is that their assumptions can be varied interactively, making the sensitivity of fusion economics to engineering choices easier to see.

## Common economic quantities

### Net electric power

A simple plant-level balance is

\[
P_\mathrm{net}=P_\mathrm{gross}-P_\mathrm{recirc}-P_\mathrm{house},
\]

where \(P_\mathrm{gross}\) is gross generated electricity, \(P_\mathrm{recirc}\) is electricity returned to plant systems such as heating and current drive, and \(P_\mathrm{house}\) represents the remaining internal loads.

### Capacity factor and availability

The energy sold over a year depends not only on rated net power but on the fraction of time the plant can operate:

\[
E_\mathrm{annual}=P_\mathrm{net}\,8760\,f_\mathrm{avail}.
\]

For fusion plants, availability can be strongly affected by scheduled replacement of neutron-facing components, unplanned outages, and maintenance duration.

### Levelized cost of electricity

A generic levelized-cost expression is

\[
\mathrm{LCOE}=\frac{\text{annualized capital cost}+\text{annual operating and replacement cost}}{\text{annual net electricity sold}}.
\]

Different models differ mainly in how they calculate and annualize these numerator terms, how they represent replacement and downtime, and which plant costs are included.

## Why fusion economics is tightly coupled to physics

Fusion economics is unusually sensitive to physical and engineering assumptions because several quantities couple strongly:

- higher fusion power may increase electrical output, but also neutron loading and replacement requirements;
- higher scientific gain reduces required plasma heating, but does not remove other recirculating loads;
- higher conversion efficiency directly raises net electric output;
- higher magnetic field can reduce device size in some concepts while increasing magnet requirements and stresses;
- longer component lifetime improves availability and reduces replacement cost;
- shorter replacement duration can matter almost as much as longer lifetime because both affect electricity sold;
- lower overnight capital cost has a large effect when financing and construction time are important.

This makes economics a natural application for a relation-centric framework such as `fusdb`: plant economics is not one isolated formula, but a network linking plasma quantities, component loads, lifetimes, efficiencies, cost scalings, and financial assumptions.

## Credits

The interactive tools embedded in the following pages are external projects and remain the work of their respective authors:

- **Fusion Energy Base**, *Fusion Power Plant Simulator*: <https://www.fusionenergybase.com/fusion-power-plant-simulator>
- **Andrew W. Lo**, *Fusion Power Plant Economics Calculator*: <https://andrewwlo.github.io/fusioneconomics/>; source code: <https://github.com/andrewwlo/fusioneconomics>

The second calculator states that it is based on the economic framework of Whyte *et al.* (2025), *Fundamental Criteria for Economic Viability of Fusion Power Plants*. Readers should consult the original work and the calculator's own documentation for authoritative definitions and assumptions.
