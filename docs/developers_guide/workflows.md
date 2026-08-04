# Workflow

Practical workflows for extending and validating relation-driven models.

## Add a New Relation

1. Place the module in the correct physics domain under `src/fusdb/...`.
2. Declare relations with the `@relation(outputs=..., tags=...)` decorator.
<!-- #TODO: more complete steps... outputs is necessary, but inputs are read from function args -->
3. Add hard/soft constraints as needed (use the `constraints` argument).
4. Add tests covering forward evaluation and at least one verification/solve path.

## Add a New Variable

1. Add variable metadata in `src/fusdb/registry/variables.yaml`.
2. Define default behavior and tolerances where required.
3. Update registry defaults if needed.
4. Ensure variable unit and dimensionality match all relations that consume it.
<!-- #TODO: add testing for variable consistency -->

## Diagnose Inconsistencies in a Reactor

1. Run `Reactor.run()` or `RelationSystem.run(mode="verify")` to evaluate
  relations without committing solver changes.
2. Inspect `relation_status`, `failed_relations` and `max_residual` in the result dict; a solve mode also reports `inputs_beyond_tolerance` and `likely_culprits`.
3. Confirm variable tolerances are realistic for the physics regime.
4. If needed, constrain relation selection via `relation_include` /
  `relation_exclude` on the `Reactor`. Note that `relation_include` **adds** to
  the tag-selected set rather than replacing it, so a swapped-in relation runs alongside the default producer unless that one is excluded by name.
  `relation_order` affects `ordered` mode only, not selection.

## Keep Knowledge and Code Aligned

- When formulas or assumptions change, update both:
  - relation code / tests
  - Knowledge Base pages explaining coupling assumptions
- Prefer explicit naming for profile vs volume-integrated quantities.
