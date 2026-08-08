# Fusion Power-Plant Power Balance

The first economic hurdle for a fusion power plant is not cost but **net electric power**. A plasma may achieve a substantial scientific fusion gain while the overall plant still consumes more electricity than it exports.

This page embeds the **Fusion Power Plant Simulator** developed by [Fusion Energy Base](https://www.fusionenergybase.com/). Fusion Energy Base was founded by **Sam Wurzel** in 2019, and the accompanying explanatory article *Scientific Gain and Fusion Power Plants* is also authored by Sam Wurzel (2026).

!!! info "Credit and source"
    The simulator below is an external tool created and hosted by **Fusion Energy Base**. `fusdb` does not reproduce or modify its model. See the [original simulator](https://www.fusionenergybase.com/fusion-power-plant-simulator), the accompanying [Scientific Gain and Fusion Power Plants](https://www.fusionenergybase.com/articles/fusion-physics-part-1-scientific-gain-and-fusion-power-plants) article, and [About Fusion Energy Base](https://www.fusionenergybase.com/about).

## Interactive simulator

<div style="width:100%; min-height:900px; margin:1rem 0;">
  <iframe
    src="https://www.fusionenergybase.com/fusion-power-plant-simulator"
    title="Fusion Energy Base — Fusion Power Plant Simulator"
    loading="lazy"
    style="width:100%; height:900px; border:1px solid #d0d7de; border-radius:8px;"
    allowfullscreen>
  </iframe>
</div>

If the embedded page is blocked by the browser or by a future change to the external site's embedding policy, open the [Fusion Energy Base simulator directly](https://www.fusionenergybase.com/fusion-power-plant-simulator).

## What the model represents

The simulator is a **power-flow model**, not a full reactor costing model. Its controls expose the quantities that connect plasma gain to grid electricity:

- external heating power or, for pulsed operation, heating energy per pulse and repetition rate;
- scientific gain \(Q_\mathrm{sci}\);
- conversion efficiency from heat or fusion-product energy to electricity;
- wall-plug efficiency of the plasma-heating system;
- house load;
- optional separate conversion efficiencies for neutron, charged-particle, and heating-energy streams;
- blanket energy multiplication;
- steady-state or pulsed operation.

The central definition used in the associated Fusion Energy Base article is

\[
Q_\mathrm{sci}=\frac{P_\mathrm{fus}}{P_\mathrm{heat}},
\]

where the heating power is measured as power crossing into the vacuum vessel. This is deliberately a **scientific/plasma gain**, not an engineering gain. The simulator then follows the energy leaving the fusion system, its conversion to electricity, and the electrical power that must be recirculated to generate the required plasma heating.

For a single effective conversion efficiency \(\eta_e\) and heating-system efficiency \(\eta_h\), the underlying logic can be understood schematically as

\[
P_\mathrm{fus}=Q_\mathrm{sci}P_\mathrm{heat},
\]

\[
P_\mathrm{gross}\sim \eta_e\left(P_\mathrm{fus}+P_\mathrm{heat}\right),
\]

\[
P_\mathrm{recirc}=\frac{P_\mathrm{heat}}{\eta_h},
\]

and

\[
P_\mathrm{net}=P_\mathrm{gross}-P_\mathrm{recirc}-P_\mathrm{house}.
\]

The actual simulator can split the conversion streams and include blanket multiplication, so the detailed power accounting can be more specific than these schematic equations.

## Main assumptions

The model is intentionally simple and concept-agnostic. The most important assumptions are:

1. **The plasma is treated as a black-box power amplifier.** The simulator does not calculate confinement, fusion reactivity, plasma stability, radiation, current drive, or reactor geometry. These are compressed into the supplied heating power and \(Q_\mathrm{sci}\).

2. **Steady-state power accounting is used for the steady-state mode.** Energy entering the plasma ultimately exits together with the fusion power. This is why the useful thermal/electrical stream is not simply \(P_\mathrm{fus}\): externally supplied heating power also leaves the plasma system.

3. **Conversion efficiencies are user parameters.** The model does not determine thermodynamic-cycle efficiency from coolant temperature, blanket design, or direct-conversion hardware.

4. **Heating-system efficiency is a wall-plug efficiency.** Producing \(P_\mathrm{heat}\) in the plasma requires more electrical power when the heating system is inefficient. The associated Fusion Energy Base article explicitly introduces this correction after first considering the optimistic \(100\%\)-efficient case.

5. **Other internal plant loads are represented by a single house-load term.** Cryogenics, pumps, magnets, tritium processing, controls, vacuum systems, and other auxiliaries are therefore not independently modeled.

6. **The advanced split-conversion option separates energy streams phenomenologically.** Neutron energy, charged-particle energy, and recovered heating energy can be assigned different efficiencies without modeling the detailed hardware that performs those conversions.

7. **Blanket multiplication is represented by a scalar factor.** Detailed neutronics, tritium breeding, shielding, and material activation are outside the model.

8. **Pulsed operation is reduced to pulse energy and repetition rate.** The simulator does not attempt to resolve detailed start-up, dwell, thermal-storage, fatigue, or grid-buffering dynamics.

These assumptions make the tool useful for answering questions such as **"What scientific gain is required before this combination of conversion and heating efficiencies can export electricity?"** They are not sufficient to answer whether the resulting plant is economically viable.

## Interpreting scientific and engineering gain

A common source of confusion is treating \(Q_\mathrm{sci}>1\) as equivalent to a power plant producing net electricity. It is not.

At fixed \(Q_\mathrm{sci}\), poor heating efficiency increases recirculating electricity. Poor conversion efficiency reduces gross electric generation. House loads further decrease the net export. Consequently, the required \(Q_\mathrm{sci}\) for net electricity is a **systems-level quantity** and can vary substantially with engineering assumptions.

Conversely, very efficient energy conversion and heating can reduce the scientific-gain threshold for net electric production. This is the main qualitative lesson of the model: plasma performance and plant engineering cannot be judged independently.

## What the model does not include

The simulator does not directly account for:

- capital cost or financing;
- component lifetime and neutron damage;
- scheduled and unscheduled availability;
- replacement cost and replacement duration;
- operations and maintenance costs;
- electricity market price;
- construction duration;
- detailed balance-of-plant or thermodynamic-cycle constraints.

Those questions are addressed more directly by the [economic-viability model](economic_viability.md) in the next page.
