# Relation Interactions

This page describes how coupled relations interact during a solve.

## Interaction Pattern

At runtime, `RelationSystem` builds a bipartite graph:

- variable nodes (`R`, `a`, `P_fus`, `n_e`, ...)
- relation nodes (each equation with one output)

Edges connect relation inputs/output to variable names.

## Solve Mechanics

In overwrite mode, the solver iterates by:

1. forward evaluation when all inputs are known;
2. backward single-unknown solve when one input is missing;
3. block solve for coupled unknown sets (`1x1` to `n_max x n_max`);
4. culprit-based override if unresolved violations remain.

In check mode, no variable mutation is performed and only diagnostics are returned.

## Typical Cross-Domain Couplings

- Geometry -> volume/shape -> profile integrals -> fusion and radiation power.
- Composition -> density partition and species fractions -> reactivity and pressure.
- Confinement (`tau_E`) <-> power balance (`P_loss`) through implicit loops.
- Operational limits consume solved state and report feasibility boundaries.

## Constraint Roles

- Hard constraints: enforce mathematical/physical admissibility and can block candidate values.
- Soft constraints: recorded as warnings for design-space awareness.

Both relation-level and variable-level constraints influence candidate ranking.

## Profile-Aware Flows

Profiles are explicit variables and are not automatically integrated for scalar outputs.
If a scalar quantity depends on a profile, the relation function must integrate it directly.
This keeps coupling visible in the relation graph and avoids hidden aggregation behavior.
