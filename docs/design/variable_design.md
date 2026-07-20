# Design note: what should `Variable` be?

*2026-07-17 — design exploration; recommendation (option F) implemented the
same day. Sections 1-6 below are the original exploration and describe the
implementation as it was **before** the change (thin mutable `Variable`,
the `set_input`/`set_value` writeback); they are kept as the historical
record of the reasoning, not updated in place. See [Variable Class](../code_docs/variable_class.md)
and [Reactor Class](../code_docs/reactor_class.md) for the current API.*

**What actually landed, vs. the sketch in §5:** `Variable` is now
`@dataclass(frozen=True)` with a `clone(**changes)` method
(`dataclasses.replace` under the hood, so a clone is a fresh, independently
re-validated ingestion, not a raw field copy); `set_input`/`set_value` are
deleted. `Reactor.get_variable`/`__getattr__` return a new
`fusdb.reactor.SolvedVariable` view (`declared`: the frozen `Variable`;
`value`: a property reading through to `last_system.values` when active,
else falling back to `declared.value`) instead of the raw `Variable`. The
`_run_once` writeback loop is deleted outright; `_absorb_regime_candidate`
collapsed to adopting the candidate's declarations wholesale (they are
provably identical to the parent's, since nothing mutates them anymore).
`Reactor.restart_from_solution()` is the sketched explicit opt-in for R8 (continue
from the previous solve), replacing the old implicit overwrite-on-solve.
Regression coverage: `tests/test_reactor_table.py::test_reconcile_moves_value_without_touching_the_declaration`
proves the declared/solved split with a value that actually moves during a
solve (the no-op `verify` case in the older test couldn't distinguish the
two designs, since verify never moves anything).

## 1. Orientation: what the code actually does today

`Variable` (`src/fusdb/variable.py:15-139`) is a dataclass holding a registry
`spec` reference plus five pieces of per-scenario state: `input_value` (the
declared value), `value` (the current value), `fixed`, resolved
`rel_tol`/`abs_tol`, and parsed record-local guard relations. All of its
behaviour runs once, in `__post_init__` (`variable.py:57-94`): alias
resolution, unit conversion to canonical SI (`variable.py:65`), domain
validation, profile-shape coercion. After construction it is inert data with
two mutators, `set_input` (overwrites *both* `input_value` and `value`,
`variable.py:105-115`) and `set_value`.

Its complete lifecycle touches nine places:

**Constructed (3):** YAML parsing (`reactor.py:281`), regime cloning
(`reactor.py:618`), popcon worker rebuild (`modes/popcon.py:546`).

**Consumed (2):** `RelationSystem.__init__` copies each record's state into
plain dicts — `inputs`, `values`, `fixed`, `rel_tols`, `abs_tols` — and
discards the objects (`relationsystem.py:262-292`; the comment at 263-264 is
explicit: *"the system holds no Variable objects at runtime (specs own the
numerics)"*). The display path reads `v.input_value`/`v.value`/tolerances in
the Reactor branch of `_table_column` (`plotting/tables.py:130-140`).

**Mutated (3):** the post-solve writeback `var.set_input(value)` for every
solved value (`reactor.py:459-462`), regime absorption
`dst.set_input(src.value)` (`reactor.py:677`), and user code
(`examples/tau_E_solver.ipynb` calls `variable.set_input(...)` and sets
`variable.fixed` interactively; `tests/test_popcon_mode.py:155` likewise).

Everything else in the solve pipeline bypasses `Variable` entirely:

- All five modes operate on the system dicts. `verify` completes a local copy
  and never stores (`modes/verify.py:13-67`). `reconcile` and `optimize` call
  `system.store(completed_values)` (`modes/reconcile.py:79,413`,
  `modes/optimize.py:97`), which writes public values into `system.values`.
  `ordered` writes `system.values` live per step (`modes/ordered.py:71-78`).
  `popcon` mutates `system.inputs`/`values` directly per grid point
  (`modes/popcon.py:480-484`, `_pin_axes_and_compile` at 590) and restores
  snapshot dicts in `finally` (`modes/popcon.py:675-676/764-768` pointwise,
  `855-856/942-947` batched).
- Parallelism never pickles a `Variable`: popcon workers receive plain tuples
  `(name, value, fixed, rel_tol, abs_tol)` (`_system_spec`,
  `modes/popcon.py:514-538`) and rebuild Variables locally
  (`_rebuild_system`, 540-551); `solve_reactors` ships YAML *paths* and gets
  back frozen `SolvedColumn` snapshots (`reactor.py:854-879`).
- The 2026-07 compile-fingerprint and seed-tape reuse machinery keys entirely
  off the system dicts (`relationsystem.py:499-539`) — same-structure reuse
  across many runs never consults a `Variable`.

One symptom that the design question is unresolved *in the code itself*: two
comments assert opposite ownership. `variable_registry.py:335`: *"The registry
only stores metadata. Values belong to `Variable` objects."*
`relationsystem.py:263-264`: *"the system holds no Variable objects at
runtime."* Both are half-true: values belong to Variables **at the boundary**
and to the system dicts **during a run**, and the writeback at
`reactor.py:459-462` smears the two together by overwriting declared inputs
with solved values.

A performance note that frames everything below: `Variable` is on **no hot
path**. The residual loop, completion, packing, popcon batching — all read
specs and dicts. Every branch evaluated here is solver-performance-neutral;
this is a question of boundary correctness, ergonomics, and state ownership,
not speed.

## 2. First principles: responsibilities something must own

| # | Responsibility | Nature | Current owner |
|---|---|---|---|
| R1 | Identity, canonical naming, alias resolution | shared, immutable | `VariableRegistry.resolve` (`variable_registry.py:441-450`) |
| R2 | Quantity metadata: unit, shape, physical/solver domains, default tolerances, registry defaults/`nominal`, registry guards | shared, immutable | `VariableSpec` (frozen, slotted, precomputed projections; `variable_registry.py:56-120`) |
| R3 | Ingestion: unit→canonical conversion, shape coercion, domain/NaN validation | a **boundary event**, not state — happens once per supplied value | `Variable.__post_init__` / `set_input` (`variable.py:57-94,105`) |
| R4 | The scenario *declaration*: which names are supplied, with what canonical values, fixed flags, per-run tolerance overrides, profile size, record-local guards | per-run; logically **immutable once declared** (it defines the problem) | split: `Variable` fields *and* `system.inputs`/`fixed`/`rel_tols` (duplicated) |
| R5 | Working/solved state evolving under a mode; must support snapshot/restore (popcon) and bulk overwrite (`store`) | per-run, **mutable** | `system.values` (+ solver-local namespaces) |
| R6 | Provenance: how each value came to be (supplied / default-seeded / relation-derived / block-solved / moved-beyond-tolerance) | per-run, derived facts | distributed on the system: `seed_provenance`, `variable_roles`, graph decidability annotations, `inputs_beyond_tolerance` (`modes/reconcile.py:528-554`), `SolvedColumn(inputs, values)` |
| R7 | Serialization for parallel runs: plain-data projection across process boundaries | projection requirement on R4/R5's owner | `_system_spec` tuples, YAML paths, `SolvedColumn` |
| R8 | Re-run semantics: what the next run starts from | per-run **policy** | implemented implicitly as the `set_input` overwrite (`reactor.py:462`) |
| R9 | User ergonomics: literal construction, `reactor.T_e.value` reads, interactive mutation | boundary API | `Variable(...)`, `Reactor.__getattr__` (`reactor.py:383-388`) |

Two observations fall straight out of this table. First, R2 and R4 are
*different kinds of thing* (shared metadata vs per-run declaration), so any
design that merges them — "Variable becomes the registry quantity" — must
reinvent a per-run record anyway. Second, R4 and R5 are *also* different
things (the declaration vs the evolving answer), and the current design's one
genuine defect is that `set_input` conflates them: after a solve, the
declaration is destroyed and replaced by the answer, so the original inputs
survive only inside `last_system.inputs`.

## 3. Candidate branches

**A — Status quo.** Thin mutable `Variable`; dual store (Variable fields in
Reactor, dicts in the system); post-solve writeback overwrites `input_value`.

**B — Remove `Variable` entirely.** Ingestion becomes a function
(`parse_value(name, value, unit, ...) -> (canonical_name, canonical_value)`);
Reactor holds plain records/dicts; users pass `{"R": 3.0, "a": {"value": 1.0,
"fixed": True}}` literals.

**C — Event-sourced cell.** `Variable` records its own modification history:
every write appends `(actor, old, new)`; provenance queries go to the cell.

**D — Variable holds values; `RelationSystem` owns all provenance.** The
current split, formalized: keep dual storage but declare the system the sole
historian (roles, seed provenance, movement) and strip any ambition of
history from the cell.

**E — Pure reference.** `Variable` carries no values; it is a handle to a
registry quantity, and values live in per-run state keyed by canonical name —
i.e. `Variable` collapses toward `VariableSpec` + the system dicts.

**F — Frozen declaration record + system-owned state (added branch).**
`Variable` remains exactly the ingestion record it already is (R3 + the R4
declaration), but becomes **immutable after construction**: `set_input`
writeback is deleted; the system dicts are the *only* mutable per-run state
(R5); Reactor answers enriched reads (`reactor.P_fus`) by reading through
`last_system.values`, with declared inputs permanently preserved on the
records. Interactive "change an input and re-run" builds a new record (or a
`replace()`-style copy) instead of mutating in place. Provenance stays on the
system (as in D).

## 4. Criteria matrix

Verdicts: **good** / ok / *poor*, with the deciding fact.

| Criterion | A: status quo | B: no Variable | C: history cell | D: values in Var, provenance in system | E: pure reference | F: frozen record + system state |
|---|---|---|---|---|---|---|
| `verify` fit | ok (writeback is a silent no-op — asymmetry vs reconcile) | good | *poor* (nothing writes cells; history stays empty) | ok (same asymmetry) | good | **good** (no writeback to be asymmetric about) |
| `reconcile` fit | ok (works, but declared inputs destroyed by `set_input`) | good | *poor* (`store` bulk-writes dicts; cells double-write or lie) | ok | good | **good** (declaration preserved; solved state on system) |
| `optimize` fit | ok (same as reconcile) | good | *poor* | ok | good | **good** |
| `ordered` fit | ok | good | *poor* (per-step writes go to `system.values`, `ordered.py:71-78`; cells see nothing) | ok | good | **good** |
| `popcon` fit | ok (mode bypasses Variables; snapshot/restore on dicts) | good | **worst case**: 1 020 points × mutate/restore would either spam cell history or bypass it, making history wrong | ok | good | **good** (dict snapshot/restore is already the native pattern) |
| Parallel, *different* systems (`solve_reactors`) | good (workers rebuild from YAML) | good | *poor* (history must be stripped or custom-pickled) | good | good | **good** |
| Parallel, same mode, different aims (optimize) | *poor* — `solve_reactors` is single-`options` (`reactor.py:917`); rolling your own means cloning mutable Variables safely | ok | *poor* | *poor* (same as A) | ok | **good**: frozen records are trivially shareable across worker tasks; only the per-task option differs |
| Same structure reused across many runs (fingerprint/seed tape) | ok (reuse keys off system dicts; Variables irrelevant but writeback mutates the base between runs) | good | *poor* (reuse machinery would invalidate/append history it never reads) | ok | good | **good** (declaration immutable ⇒ base state can never drift between runs) |
| Unit conversion home | good: one place, `__post_init__` | ok: a free function — same one place, but nothing *forces* callers through it | good | good | *poor*: conversion needs a home; either the registry (wrong: registry is shared and unit-of-supply is per-value) or a helper that reinvents branch B | **good**: unchanged, and immutability guarantees it can only ever happen once |
| Overlap/duplication with `VariableSpec` | some: `rel_tol`/`abs_tol`/`unit` shadow spec fields as resolved copies | none | some + history machinery | some | *maximal by construction*: a value-less Variable **is** a spec handle; the branch dissolves into "spec + dicts" = B with extra indirection | minimal: record holds only per-run declaration; spec read through `.spec` (already true, `variable.py:19-24`) |
| Provenance / history | *poor*: overwrite destroys the declared input, the one provenance fact users ask about ("what did I supply vs what did the solve produce?") | ok (system provenance only) | good **in theory**; in practice false, because the actors (modes) write dicts, not cells | good (honest about who knows what) | ok | **good**: declaration preserved forever on the record; change-provenance stays on the system where the actors actually live (`seed_provenance`, `variable_roles`, `inputs_beyond_tolerance`) |
| Pickling / process-pool cost | good (never pickled; plain-tuple projection exists, `popcon.py:514-538`) | good | *poor* (history bloats or needs stripping — reinvents `_picklable_result`, deleted 2026-07 for exactly this reason) | good | good | **good**: a frozen record of plain data is *itself* the picklable projection; `_system_spec` could become `tuple(variables)` |
| Batched popcon scans | good (dict-native) | good | *poor* | good | good | **good** |
| User ergonomics (R9) | good: `Variable("R", 3.0)`, `reactor.T_e.value`, notebook mutation | *poor→ok*: dict literals are fine for construction, but `reactor.T_e.value` and per-variable handles disappear or get reinvented | ok | good | *poor*: `Variable("R", 3.0)` can't exist (no value slot) | good, one change: in-place `set_input` mutation becomes construct-anew (`tau_E_solver.ipynb` pattern needs a one-line migration) |

## 5. Recommendation

**Branch F: `Variable` is the frozen per-run declaration record; the
`RelationSystem` dicts are the only mutable state; provenance lives on the
system; Reactor reads solved values through `last_system`.**

This is a *surgical evolution*, not a redesign, and that scope is the point:
the runtime has already converged on the right shape. Specs own shared
metadata (R1-R2), the system dicts own working state and provenance (R5-R6),
Variables already do ingestion once and are already discarded by the solver
(R3; `relationsystem.py:263-264`). The only thing actually wrong is a ~60-line
seam: the post-solve writeback (`reactor.py:459-462`), regime absorption
(`reactor.py:677`), and the clone reconstruction they force
(`reactor.py:618-639`) — the machinery that exists solely to keep a second
mutable copy of state synchronized, and that destroys the user's declaration
(R4/R8 conflation) as it does so. F deletes that seam instead of the class.

Why F beats the alternatives on the criteria that matter most here:

- **The three parallel/reuse criteria** are where fusdb is actively investing
  (process-pool popcon, compile fingerprint, seed tape), and they all reward
  immutable declarations + dict state. A frozen record is its own picklable
  projection and can be shared across N optimize tasks with different
  objectives without defensive cloning — the exact scenario the current API
  cannot express.
- **Provenance** is answered honestly: the question users actually ask
  ("declared vs solved?") is answered by *never destroying the declaration*;
  the question the system can answer ("who derived this?") already lives in
  compile products and results. No cell history needed.
- **Unit conversion** keeps its single enforced home, now with a guarantee
  (immutability) instead of a convention.
- It preserves `Variable("R", 3.0)` and `reactor.T_e.value` — the ergonomics
  that make branch B and E losers despite their conceptual cleanliness.

What F changes, concretely (for a future implementation, not now): make the
dataclass frozen-after-init; delete `set_input`/`set_value`; replace the
`_run_once` writeback with `last_system`-backed reads in
`Reactor.__getattr__`/`get_variable` (returning a lightweight view that pairs
the frozen record with the solved value); make regime absorption keep the
winning *system* instead of copying values into records; make "re-run from
solved state" an explicit operation (`reactor.restart_from_solution()`)
instead of a silent side effect — R8 becomes a policy the user invokes, not
one that happens to them.

**Main risk.** The overwrite-on-solve semantic is load-bearing in three known
places: chained runs that rely on solved values becoming the next run's
inputs, `_absorb_regime_candidate`'s value copy-back, and interactive
notebook mutation (`tau_E_solver.ipynb`). If user workflows depend on
*implicit* restart-from-solution (run reconcile twice, expecting the second to
start where the first ended, without calling anything), F breaks them into an
explicit call.

**Falsification evidence.** (1) A survey of real notebooks/scripts showing
implicit chained-solve reliance is widespread rather than rare — then R8's
default should stay "overwrite", and F should keep the writeback but target
`value` only, never `input_value` (a weaker but still worthwhile fix). (2) The
regression harness (`scripts/reconcile_all.py`) or the regime-switching tests
failing under a prototype in ways that trace to absorption semantics rather
than bugs — that would show the dual store is compensating for something the
system state model cannot express, and branch D (formalized dual store) is the
honest fallback.

## 6. Discarded options

- **A — status quo.** Pros: zero migration; familiar. Cons: destroys
  declarations on solve; two mutable copies with hand-rolled sync; verify vs
  reconcile enrichment asymmetry is accidental; blocks clean multi-objective
  parallelism. Discarded because its one advantage (inertia) is priced in ~60
  lines of seam code and a stale pair of contradictory ownership comments.
- **B — no Variable.** Pros: fewest concepts; dict literals are pleasant.
  Cons: loses the enforced single ingestion point (conversion/validation
  becomes a convention), loses per-variable handles (`reactor.T_e`),
  breaks a user-facing constructor for zero runtime gain (Variable is not on
  any hot path). Discarded: pays real ergonomic cost to remove an object that
  costs nothing.
- **C — event-sourced cell.** Pros: attractive audit story on paper. Cons:
  false in practice — every actor (modes, popcon snapshot/restore, `store`)
  writes system dicts, so cells would either miss the history or force the
  whole pipeline through an object API; pickling bloat re-creates the
  `_picklable_result` stripping problem deleted in 2026-07; popcon would
  generate thousands of meaningless events per scan. Discarded: provenance
  belongs with the actors, and the actors live on the system.
- **D — values in Variable, provenance in system.** Pros: honest about the
  historian; minimal change. Cons: keeps the dual mutable store and the sync
  seam; keeps the declaration-destroying writeback unless separately fixed.
  Discarded in favour of F, which is D plus the one fix that matters
  (immutability of the declaration); D remains the fallback if F's risk
  materializes.
- **E — pure reference.** Pros: conceptual minimalism. Cons: a value-less
  Variable duplicates `VariableSpec`'s role exactly (the registry is already
  the shared quantity reference), while the per-run declaration still needs a
  record — so E decomposes into "spec + record + dicts", which is F wearing a
  costume, minus the `Variable("R", 3.0)` constructor. Discarded as B/F hybrid
  with the worst ergonomics of both.

## Appendix A: F's Variable vs the registry, and the B/E/F worked example

**Variable vs `VariableSpec` under F** — schema vs row. The spec answers
"what *is* `P_aux`" (unit, shape, domains, default tolerances, registry
defaults/guards; one per canonical name, process-wide, frozen). The F-Variable
answers "what does *this scenario* say about `P_aux`" (declared value in
canonical units, `fixed`, per-run tolerance overrides, profile size, local
guards; zero-or-one per scenario, frozen after construction). Discriminator:
true for every reactor → spec; a fact about this scenario → Variable; changes
during a solve → neither (that is `system.values`). F's unique addition:
the Variable is a **receipt** — constructing it *is* the ingestion event
(unit conversion `variable.py:65`, domain/shape validation), and immutability
guarantees the event happened exactly once. Neither the spec (no value slot)
nor a dict (anyone can write it) can play that role.

**Why B and E fail, mechanically.** B does not remove the design, it un-names
it: the five per-declaration facts (canonical value, fixed, tol override,
size, guards) scatter into parallel dicts zipped by hand at every boundary —
popcon's `_system_spec` positional 5-tuple (`modes/popcon.py:514-538`) is a
preview — and `inputs["P_aux"] = 25.0` (raw MW) becomes representable and
silent. E is unstable: strip the value and the handle either keeps per-run
facts (then it is not a pure reference — they would leak across runs) or
keeps only name+spec (then it *is* `VARIABLES.get(name)` with an extra hop),
while the evicted state lands in B's dicts. Every configuration of E resolves
to F or B wearing a costume.

**One case, three ways** — declare `P_aux` = 25 MW (canonical is W) with a 2%
tolerance override on SPARC; reconcile (it moves); ask declared-vs-solved;
ship to popcon workers:

```python
# F: frozen declaration record
reactor.declare(Variable("P_aux", 25.0, unit="MW", rel_tol=0.02))  # ingested+frozen: 2.5e7 W
reactor.reconcile()
reactor.P_aux.declared   # 2.5e7  (immutable)
reactor.P_aux.value      # 2.71e7 (read-through -> last_system.values)
tasks = [(reactor.variables, {"objective": o}) for o in objectives]  # frozen => share, no clone
```
*Pros:* conversion guaranteed once; declared/solved never conflated; records
shareable across parallel tasks; `reactor.P_aux` handles kept. *Cons:*
in-place `set_input` mutation replaced by construct-anew (one-line notebook
migration); one more concept than B.

```python
# B: dicts + a parse function
reactor.inputs["P_aux"] = parse_value("P_aux", 25.0, unit="MW")  # must remember parse_value
reactor.tols["P_aux"] = 0.02                                     # second structure
reactor.fixed.discard("P_aux")                                   # third structure
```
*Pros:* fewest classes; dict literals pleasant for simple cases. *Cons:*
`inputs["P_aux"] = 25.0` (unconverted) runs silently — canonical units become
a discipline; declaration smeared over three hand-zipped structures; no
per-variable handles.

```python
# E: value-less handle + keyed state
p_aux = Variable("P_aux")                    # duplicates VARIABLES.get("P_aux")
reactor.state[p_aux] = convert(25.0, "MW")   # conversion by convention (B's flaw)
reactor.state.tols[p_aux] = 0.02             # per-run fact -> side map (B's dicts)
reactor.state[p_aux]                          # declared or solved? A's conflation returns
```
*Pros:* clean-looking reference/state split. *Cons:* redundant with the spec;
per-run facts homeless on a shared handle; worst construction ergonomics;
single state slot re-creates the declared-vs-solved conflation.

All three must place the same five facts somewhere: F gives them one named,
frozen home whose constructor doubles as validation; B scatters them and
downgrades the units invariant to a convention; E splits them between a
redundant handle and B's dicts.
