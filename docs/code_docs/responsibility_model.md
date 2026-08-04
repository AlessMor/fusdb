# Responsibility Model

Values in `fusdb` pass through four layers — the registry, `Variable`,
`RelationSystem` and `Reactor` — and each layer owns a different, non-overlapping
part of the job. This page is the map: which layer owns what, and why the
declaration a user supplies is immutable.

Related modules:

- `fusdb.registry.variable_registry`
- `fusdb.variable`
- `fusdb.relationsystem`
- `fusdb.reactor`

Related pages:

- [Variable Class](variable_class.md)
- [RelationSystem](relationsystem_class.md)
- [Reactor Class](reactor_class.md)

## The Nine Responsibilities

| # | Responsibility | Nature | Owner |
|---|---|---|---|
| R1 | Identity, canonical naming, alias resolution | shared, immutable | `VariableRegistry.resolve` |
| R2 | Quantity metadata: unit, shape, physical/solver domains, default tolerances, registry defaults/`nominal`, registry guards | shared, immutable | `VariableSpec` (frozen, slotted, process-wide) |
| R3 | Ingestion: unit→canonical conversion, shape coercion, domain/NaN validation | a **boundary event**, not state — happens once per supplied value | `Variable.__post_init__` |
| R4 | The scenario *declaration*: which names are supplied, with what canonical values, fixed flags, per-run tolerance overrides, profile size, record-local guards | per-run, **immutable once declared** (it defines the problem) | `Variable` (frozen), projected into `RelationSystem.inputs` / `fixed` / `rel_tols` |
| R5 | Working/solved state evolving under a mode; must support snapshot/restore (popcon) and bulk overwrite | per-run, **mutable** | `RelationSystem.values` (written via `store()`) |
| R6 | Provenance: how each value came to be (supplied / default-seeded / relation-derived / block-solved / moved-beyond-tolerance) | per-run, derived facts | `RelationSystem.variable_roles`, `seed_provenance`, graph decidability annotations, `inputs_beyond_tolerance`, `SolvedColumn` |
| R7 | Serialization for parallel runs: plain-data projection across process boundaries | projection requirement on R4/R5's owner | `_system_spec` tuples, YAML paths, `SolvedColumn` |
| R8 | Re-run semantics: what the next run starts from | per-run **policy** | `Reactor.restart_from_solution()` — explicit, never implicit |
| R9 | User ergonomics: literal construction, `reactor.T_e.value` reads | boundary API | `Variable(...)`, `Reactor.__getattr__` → `SolvedVariable` |

Two consequences fall straight out of this table.

**R2 and R4 are different kinds of thing** — shared metadata versus a per-run
declaration. Any design that merges them ("`Variable` becomes the registry
quantity") has to reinvent a per-run record anyway.

**R4 and R5 are also different things** — the declaration versus the evolving
answer. Keeping them apart is what the next section is about.

## Why the Declaration Is Frozen

`Variable` is `@dataclass(frozen=True)`. Constructing one *is* the ingestion
event (R3); after that it is inert. There are no setters, and a changed
declaration is always a new object via `clone()`, which re-runs the full
construction path rather than copying fields.

Nothing writes a solved value back onto it. `RelationSystem.values` is the only
mutable per-run state, and `Reactor` answers enriched reads by pairing the frozen
record with that state in a `SolvedVariable` view — `declared` is what the user
supplied, `value` reads through to `last_system` and is resolved fresh on every
access.

This buys four things:

- **Provenance without machinery.** The question users actually ask — "what did
  I supply, versus what did the solve produce?" — is answered by never
  destroying the declaration. The question the system can answer ("who derived
  this?") already lives in the compile products under R6.
- **Parallelism.** A frozen record of plain data *is* its own picklable
  projection, and can be shared across many worker tasks with different
  objectives without defensive cloning.
- **Structure reuse.** Because a declaration cannot drift between runs, the
  compile fingerprint and seed tape can safely key off it.
- **One enforced home for unit conversion**, now guaranteed by immutability
  rather than by convention.

!!! warning "Re-running does not resume automatically"
    Solved values do **not** become the next run's inputs on their own. Running
    `reconcile` twice starts from the same declaration both times. To continue
    from where a solve ended, call `Reactor.restart_from_solution()` — R8 is a
    policy you invoke, not one that happens to you.

## Rejected Alternatives

Five other shapes were considered before this one. Condensed, with the deciding
fact for each:

- **Mutable `Variable` with post-solve writeback** (the original design) —
  destroyed the user's declaration on every solve and kept two mutable copies of
  the same state in hand-rolled sync.
- **No `Variable` at all**, ingestion as a free function over dict literals —
  loses the *enforced* single conversion/validation point (it becomes a
  convention callers can bypass) and the per-variable handles `reactor.T_e`,
  for no runtime gain: `Variable` is on no hot path.
- **Event-sourced cell** recording its own modification history — false in
  practice, because every actor (modes, popcon snapshot/restore, `store`)
  writes system dicts, so the cells would either miss the history or force the
  whole pipeline through an object API. A popcon scan would also generate
  thousands of meaningless events.
- **Values in `Variable`, provenance in the system** — honest about who the
  historian is, but keeps the dual mutable store and the sync seam. This is the
  fallback if immutability ever proves untenable.
- **Value-less `Variable` as a pure registry handle** — duplicates
  `VariableSpec`'s role exactly, while the per-run declaration still needs a
  record; it decomposes into "spec + record + dicts" minus the
  `Variable("R", 3.0)` constructor.

The through-line: provenance belongs with the actors, and the actors live on the
system; the declaration belongs to the user and should outlive the solve.

## Design Record

*The model above was adopted 2026-07-17 and implemented the same day.* The full
decision record — the six candidate branches, the criteria matrix that ranked
them across all five execution modes, and the falsification evidence that would
have overturned the choice — is preserved in git history at
`docs/design/variable_design.md`.

Regression coverage:
`tests/test_reactor_table.py::test_reconcile_moves_value_without_touching_the_declaration`
proves the declared/solved split with a value that actually moves during a
solve.
