# How fusdb fits among fusion system codes

`fusdb` is **not intended to be another monolithic fusion power-plant systems code**.
Its main abstraction is different: it treats published equations, scalings, constraints,
and data-backed models as reusable **relations**, then composes selected relations into a
`RelationSystem` that can be evaluated, verified, reconciled, scanned, or optimized.

A useful shorthand is:

> **PROCESS models a fusion power plant; cfspopcon models plasma operating space; FUSE and bluemira model integrated fusion designs; fusdb models the relationships themselves.**

This distinction is the main reason for `fusdb` to exist.

## Positioning

Traditional fusion systems codes are normally organized around a particular reactor model,
workflow, or set of subsystem modules. That is appropriate when the goal is to produce a
self-consistent plant design. `fusdb` instead starts one level lower: from the equations and
data used by those models.

Its intended role is therefore an **executable fusion-relations database and consistency
framework**. A reactor systems model, a POPCON calculation, a benchmark, or an inverse
problem can then be assembled from the same underlying relations.

Conceptually:

```text
                         Applications
        POPCON · optimization · reactor studies · UQ
                              │
                       RelationSystem
          verify · reconcile · ordered · scan · optimize
                              │
                         Relations
       equations · constraints · validity · provenance
                              │
                         Variables / data
              identity · units · domains · values
```

The goal is not to compete with mature plant codes on engineering breadth or with
multi-physics frameworks on model fidelity. The goal is to make the **physics and
engineering relationships themselves easy to inspect, reuse, exchange, replace, and solve
in different contexts**.

## The key distinction: the relation network is primary

Consider a relation such as

\[
P_\mathrm{fus} = n_D n_T \langle \sigma v \rangle E_\mathrm{fus} V.
\]

In a conventional calculation chain, this equation normally has a prescribed direction:
inputs are supplied and fusion power is returned. In `fusdb`, the Python function still has
a forward implementation, but once it is part of a `RelationSystem` the *system-level
problem* is not tied to that direction. Depending on what is supplied and what is free, the
same network can be used to:

- calculate missing quantities;
- verify an externally supplied operating point;
- reconcile mutually inconsistent values while respecting fixed quantities;
- infer quantities that are not naturally "outputs" of a calculation chain;
- scan operating space, including POPCON-like studies;
- optimize a figure of merit subject to the same physical relations and constraints.

This is why `verify`, `reconcile`, `ordered`, `popcon`, and `optimize` are best understood as
different ways of interrogating the same relation network, rather than as separate physical
models.

## Comparison with other tools

The comparison below is about **software role and abstraction**, not about which code is
"better". The projects solve different problems and are often complementary.

| Project | Primary role | Typical abstraction | What `fusdb` does differently | Code | Representative publication |
|---|---|---|---|---|---|
| **PROCESS** | Whole-plant systems analysis and constrained optimization | Reactor model with iteration variables, equality/inequality constraints, engineering and economic modules | `fusdb` makes individual relations independently reusable and lets the selected relation network define the problem, rather than starting from one fixed plant model | [ukaea/PROCESS](https://github.com/ukaea/PROCESS) | Kovari et al., *Fusion Engineering and Design* 89 (2014), [doi:10.1016/j.fusengdes.2014.09.018](https://doi.org/10.1016/j.fusengdes.2014.09.018); engineering: Kovari et al. (2016), [doi:10.1016/j.fusengdes.2016.01.007](https://doi.org/10.1016/j.fusengdes.2016.01.007) |
| **cfspopcon** | Fast tokamak POPCON and operating-space studies | 0-D plasma physics functions assembled into a directed operating-point calculation | POPCON is one possible `RelationSystem` application in `fusdb`, not the organizing abstraction; the same relations can also be used for verification, reconciliation, or inverse problems | [cfs-energy/cfspopcon](https://github.com/cfs-energy/cfspopcon) | Body, Hasse & Creely, *The SPARC Primary Reference Discharge defined by cfsPOPCON* (2023), [arXiv:2311.05016](https://arxiv.org/abs/2311.05016) |
| **OpenPOPCON** | Open POPCON analysis | POPCON-oriented plasma operating-space workflow | `fusdb` is broader than POPCON and separates reusable relations from any particular scan workflow | [hansec/OpenPOPCON](https://github.com/hansec/OpenPOPCON) | See repository documentation |
| **VSC** | Multi-configuration 0-D fusion power-balance and POPCON studies | Common solver interface with configuration-specific physics models | VSC standardizes the *configuration solver interface*; `fusdb` aims to standardize the underlying reusable relation network | [VSC web application](https://hub.veloalpha.cn/vsc/) | Wang et al., *VSC: A Zero-Dimensional Fusion Design Platform for Multiple Magnetic Configurations* (2026), [arXiv:2607.11208](https://arxiv.org/abs/2607.11208) |
| **FUSE** | Integrated, multi-fidelity fusion pilot-plant design | Shared IMAS-based data structure plus physics/engineering actors and workflows | FUSE focuses on integrated model coupling and increasing fidelity; `fusdb` focuses on lightweight, explicit relations that can remain useful outside a large integrated workflow | [ProjectTorreyPines/FUSE.jl](https://github.com/ProjectTorreyPines/FUSE.jl) | Meneghini et al., *FUSE (Fusion Synthesis Engine): A Next Generation Framework for Integrated Design of Fusion Pilot Plants* (2024), [arXiv:2409.05894](https://arxiv.org/abs/2409.05894) |
| **bluemira / BLUEPRINT** | Automated and increasingly multi-fidelity tokamak reactor design | Parametrized reactor-design activities and subsystem workflows | bluemira automates the route from systems inputs toward integrated reactor geometry/design; `fusdb` stays intentionally closer to the equation/data layer | [Fusion-Power-Plant-Framework/bluemira](https://github.com/Fusion-Power-Plant-Framework/bluemira) | Coleman & McIntosh, *BLUEPRINT: A novel approach to fusion reactor design* (2019), [doi:10.1016/j.fusengdes.2018.12.036](https://doi.org/10.1016/j.fusengdes.2018.12.036) |
| **D0FUS** | Lightweight 0-D fusion modelling | Compact zero-dimensional model implementation | `fusdb` is intended as a reusable relation/data layer from which different 0-D models can be assembled | [IRFM/D0FUS](https://github.com/IRFM/D0FUS) | See repository documentation |

Adjacent tools can also be complementary rather than direct competitors. For example,
[fusion economics](https://andrewwlo.github.io/fusioneconomics/) focuses on plant costing,
while `fusdb` can provide traceable physics or engineering quantities used as inputs to such
analyses.

## Where fusdb should *not* compete

`fusdb` should not measure success by reproducing every subsystem contained in PROCESS or
by matching the multi-fidelity integration scope of FUSE or bluemira. Those projects have a
different objective and substantially broader integrated-design responsibilities.

Instead, `fusdb` should remain useful **before, inside, and beside** those workflows:

- as a catalog of published equations and datasets;
- as a compact executable reference implementation of those relations;
- as a way to compare alternative closures or scalings without rewriting an entire reactor model;
- as a consistency checker for published or externally generated reactor scenarios;
- as an inverse/reconciliation layer when only a partial set of quantities is known;
- as the backend for lightweight POPCON, parameter scans, and optimization studies.

## Why model interchangeability matters

System-code predictions can differ because of the physical models chosen rather than because
of the numerical solver. A useful example is the PROCESS/TPC benchmarking study by Nakamura
et al.: most of the compared plasma outputs agreed well, while radiation losses differed
because the two codes treated impurity radiation differently. See Nakamura et al., *Fusion
Engineering and Design* 87 (2012), [doi:10.1016/j.fusengdes.2012.02.034](https://doi.org/10.1016/j.fusengdes.2012.02.034).

This motivates making alternative relations explicit and swappable. A confinement scaling,
radiation model, bootstrap-current model, or engineering limit should be identifiable as a
model choice, rather than being hidden inside a larger calculation sequence.

The longer-term objective is therefore not just a large collection of functions, but an
**executable knowledge base** in which relations can carry enough context to answer:

- What variables does this model connect?
- Which publication or dataset does it come from?
- Under which assumptions and validity range is it applicable?
- Which alternative models describe the same quantity?
- What changes in a reactor scenario if one relation is replaced by another?

## Relation-centric, not module-centric

Many systems-code architectures naturally group calculations into modules such as plasma,
magnets, blanket, power balance, and costing. `fusdb` deliberately keeps a finer-grained
abstraction: the relation itself.

That does **not** mean subsystems are unimportant. Relations can still be grouped by tags,
selected as coherent model sets, or executed in an ordered workflow when that is physically
necessary. The distinction is that a relation remains independently discoverable and usable;
it does not have to exist only as an implementation detail of one subsystem class or one
reactor workflow.

## A concise description

For papers, presentations, and project descriptions, the intended positioning can be summarized as:

> **fusdb is a lightweight executable database of fusion relations and data, with a relation-system layer for composing them into self-consistent models that can be evaluated, verified, reconciled, scanned, or optimized.**

Or, more compactly:

> **A fusion systems code is one application that can be built from `fusdb`; it is not the definition of `fusdb`.**

## Related literature

The following publications provide useful context for the design philosophy of fusion systems
codes and integrated reactor-design frameworks:

- M. Kovari et al., *"PROCESS": A systems code for fusion power plants—Part 1: Physics*, Fusion Engineering and Design 89 (2014), [doi:10.1016/j.fusengdes.2014.09.018](https://doi.org/10.1016/j.fusengdes.2014.09.018).
- M. Kovari et al., *"PROCESS": A systems code for fusion power plants—Part 2: Engineering*, Fusion Engineering and Design 104 (2016), [doi:10.1016/j.fusengdes.2016.01.007](https://doi.org/10.1016/j.fusengdes.2016.01.007).
- C. Reux et al., *DEMO reactor design using the new modular system code SYCOMORE*, Nuclear Fusion 55 (2015), [doi:10.1088/0029-5515/55/7/073011](https://doi.org/10.1088/0029-5515/55/7/073011).
- J. Johner, *HELIOS: A Zero-Dimensional Tool for Next Step and Reactor Studies*, Fusion Science and Technology 59 (2011), [doi:10.13182/FST11-A11650](https://doi.org/10.13182/FST11-A11650).
- M. Coleman & S. McIntosh, *BLUEPRINT: A novel approach to fusion reactor design*, Fusion Engineering and Design 139 (2019), [doi:10.1016/j.fusengdes.2018.12.036](https://doi.org/10.1016/j.fusengdes.2018.12.036).
- O. Meneghini et al., *FUSE (Fusion Synthesis Engine): A Next Generation Framework for Integrated Design of Fusion Pilot Plants* (2024), [arXiv:2409.05894](https://arxiv.org/abs/2409.05894).
- Z. Wang et al., *VSC: A Zero-Dimensional Fusion Design Platform for Multiple Magnetic Configurations* (2026), [arXiv:2607.11208](https://arxiv.org/abs/2607.11208).
- M. Nakamura et al., *Efforts towards improvement of systems codes for the Broader Approach DEMO design*, Fusion Engineering and Design 87 (2012), [doi:10.1016/j.fusengdes.2012.02.034](https://doi.org/10.1016/j.fusengdes.2012.02.034).
