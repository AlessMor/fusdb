# TODO

Open items only. Landed work and its measurements are in `.claude/scratchpad.md`; the
physics background, the three-level discriminant framing and how PROCESS / cfspopcon /
HELIOS / PLASMOD / FUSE each handle confinement modes are in
`docs/knowledge_base/plasma_physics/5-transport_and_confinement/Confinement_modes.md`.

## Confinement modes

Ordered by how much each would change an answer.

- **The mode changes `tau_E` and essentially nothing else.** 58 of 64 mode-tagged relations
  output only `tau_E`; the rest are certifiers. The mode also sets the CORE BOUNDARY
  CONDITION -- H-mode a finite pedestal, L-mode the core profile continuing to the
  separatrix, I-mode a temperature pedestal without the particle barrier -- and none of that
  is represented. Biggest structural gap.
  INVESTIGATE FIRST: profiles are `avg x shape` with shape generators opt-in via the
  `profile_shape` tag. If a mode-tagged PEDESTAL generator can use that existing gate, this
  is a tag change rather than a new subsystem. Establish that before designing anything.

- **The L-H threshold has no accessibility limits and no geometry/wall corrections.**
  IMAS.jl's `scaling_L_to_H_power` adds physics no other surveyed code has:
    * INACCESSIBLE (no X-point, or triangularity < 0)  -- derivable now via `delta`
    * x2 unfavourable grad-B drift                     -- needs X-point z + a B0 sign convention
    * x0.8 metallic wall                               -- needs a wall-material input
    * n_e clamped up to the Martin rollover minimum    -- derivable now
  The grad-B factor of 2 dwarfs every correction argued about so far. `delta_95`'s domain was
  widened to [-1, 1] for the negative-triangularity sweep in `examples/01`, so fusdb will
  currently compute a finite P_LH for a plasma that cannot reach H-mode at all.
  ENCODING: an `h_mode`-tagged `enforce=False` certifier, never `Inf` -- an Inf residual would
  blow up the least-squares solve. The two missing inputs are scope needing approval.

- **I<->H is blocked on an I-mode ceiling, not on plumbing.** The complete graph was built and
  REVERTED 2026-08-07: with i_mode reachable from h_mode, the tungsten point `w_1.0e-04`
  (which has no consistent mode) certified as i_mode on an absurd solve -- P_sep 7179.6 MW
  against PROCESS's 0.0 MW, P_aux +9424%. Cause: i_mode's ONLY certifier is
  `P_sep >= P_LI_thresh`, which an inflated P_sep satisfies trivially.
  UNBLOCK WITH: an I-H threshold (an I-mode upper certifier). No surveyed code has one --
  cfspopcon has three L->I entry scalings and no exit, PROCESS does not switch at all, and
  FUSE/IMAS do not model I-mode. Then swap the chains in `_candidate_regimes` for
  `[declared, *others]` and delete the test that locks the restriction.

- **Nothing physically separates H from I.** When the declared mode is inadmissible and both
  upper branches are admissible, the pick is by candidate order; that now sets
  `regime_ambiguous` and says so, but it is still not a discriminator. H and I differ by
  topology, drift direction and edge state, none of which is in the decision. A real one
  drops in as ONE extra certifier relation -- no driver change.

- **SepOS is reachable but degenerate.** Level 2 now evaluates (770/770 SPARC cells) after
  `Z_bar` was given a producer, but it says H-mode on **100%** of them against 77% for
  `P_sep/P_LH`, never L. `alpha_t` spans [0.0077, 0.0886], entirely below the O(0.1-1) band
  where Eich places the boundary, driven by T_sep ~ 200-430 eV (cfspopcon's own, matched to
  0.10%). Safe to add as a certifier -- conditions AND together, so a discriminant satisfied
  everywhere contributes nothing -- but it buys nothing until the alpha_t calibration is
  understood. Decide whether to add it as a reported diagnostic first.

- **`confinement_mode` is a tag, not a variable.** The declared mode is therefore not
  verifiable as data. If a numeric verdict is needed, expose it as reported diagnostic
  data, never as an integer model-selection variable: the tag remains the ASSUMPTION
  (it selects relations at compile time), and the diagnostic is the VERDICT.

- **`verify` does not switch and does not report admissibility.** It includes the certifiers
  and reports their status, but has no notion of the admissible set.

- **popcon's fallback fills unclaimed cells with `l_mode` silently.** The scalar path now
  flags `regime_over_constrained` when nothing is admissible; the per-cell path still assigns
  the fallback with no equivalent marker. Same defect, one layer down, across 1200 cells.

- **Guard classification differs between the two paths.** popcon uses `residual <= 1e-9`;
  reconcile hits the outputless-relation tolerance (`rel_tol_default` 1e-3, `abs_tol` 0).
  Six orders apart, both negligible in practice, but it should be one number.

- **Near-boundary popcon cells are unasserted.** 62 of 770 SPARC cells (8.1%) sit within
  +/-20% of the L-H threshold, 30 within +/-10%, 15 within +/-5% -- real coverage that no
  test checks. A rejected hysteresis-band experiment passed the ENTIRE suite while changing
  nothing any test asserted. Assert per-cell regime assignment there before any further L-H
  change.

- **PerezH/PerezL cannot be selected by regime.** They carry `h_mode`/`l_mode` tags and are
  the one genuinely regime-specific pair, but `default_relation` filters AFTER tag matching,
  so tags only choose WITHIN a whitelist -- and `SOL_momentum_loss_fraction` whitelists the
  untagged KotovReiter, which matches every reactor. The three ways to open the gate and why
  each is blocked are enumerated next to the whitelist in `variables.yaml`. Needs a ruling on
  the cfspopcon-SPARC parity cost.

- **LOC/SOC is unmodelled.** Removing ohmic from the confinement axis was right, but the
  ohmic TRANSPORT regimes are still absent. cfspopcon's rule is a continuous min over tau_E
  (`where(tau_E/tau_LOC > 1, tau_LOC, tau_E)`), NOT a tag axis.
  `cfspopcon_loc_confinement_time` is ported and unwired.

## Physics gaps

- **fusdb has no non-thermal ion populations.** Zero hits for `beta_fast`, `fast_alpha`,
  `slowing_down`, `thermalis*`. fusdb's beta is thermal-only; PROCESS's is
  `beta_thermal + beta_fast_alpha + beta_beam`. At the PROCESS eval point `beta_fast_alpha` is
  **13.7% of total beta** (4.435e-3 of 3.230e-2), and `beta_norm_total` 2.847 vs
  `beta_norm_thermal` 2.456.

  Note this is NOT a missing element. A beam ion is D or T at Z=1 ("Only deuterium and tritium
  in the beams"); a fast alpha is the same He as the ash. What separates them is the
  DISTRIBUTION FUNCTION -- injected/born at high energy, slowing down, never Maxwellian at
  T_i. So it is a new *axis* in the composition model (a second population per isotope
  carrying its own energy, excluded from thermal pressure but included in quasineutrality),
  not a new row in `species.yaml`. fusdb's composition is one thermal population per isotope
  and has nowhere to put it.

  LEAD, NOT PROVEN: the reactors declare published design betas, which usually quote TOTAL
  beta. DEMO declared beta_N 0.025 -> solved 0.0226 (-9.6%); STEP declared 0.0393 -> 0.0374
  (-4.8%). Both LOW, sign consistent with the missing fast-alpha pressure and magnitudes in
  the 13.7% ballpark; `beta_N` is beyond tolerance on 4 reactors (DEMO 11.8, ARC_V0 25.9,
  STEP 9.4, ARC_V3A). FALSIFICATION TEST: establish which beta convention each reactor's
  declared value uses (thermal vs total) BEFORE building a fast-alpha pressure model. The
  PROCESS fixture is already honest about this -- it maps fusdb `beta` to PROCESS's THERMAL
  beta and matches +0.48%.

  Fast alphas bite on every burning plasma; beam ions only on NBI-heated non-ignited points
  (PROCESS default `f_nd_beam_electron = 0.005`). If this is ever built, fast alphas first.
  cfspopcon models neither -- it has zero occurrences of `beam`/`fast ion`/`slowing`.

## Infrastructure

- **Repointing any `default_relation` silently breaks fixtures that excluded the old default
  BY NAME.** The registry logs `relations.exclude names X, which was not active anyway`, then
  the new default runs alongside the fixture's include, the output is over-determined, and the
  failure surfaces far away as NaN fields / `assert False`. This cost 4 popcon tests and then
  3 PROCESS tests on the 2026-08-07 `P_LH` change, neither pointing at `P_LH`. Worth either a
  registry check (warn when an exclude names a relation that a `default_relation` no longer
  lists) or a test that asserts each fixture's excludes are live.

- rewrite tests

- Selectable reconcile movement penalty. The "culprit" verdict (`inputs_beyond_tolerance`) is an optimiser *outcome*, not a ranking: it falls out of eligibility (`fixed` vs supplied) x deadzone (tolerances) x distance metric x aggregation norm x relation-weight continuation. (`likely_culprits` is a different, purely post-hoc thing: a count of how many failed relations each variable appears in, with no feedback into the solve.) Two knobs worth exposing, both defaulting to today's behaviour:

  - `movement_objective` (default `count`) — *what belief about the data you are encoding*.
    - `count` = L0 via IRLS `1/(excess+eps)`: minimises HOW MANY inputs are contradicted. Short, nameable verdict; right prior for published design points. Also a good model-bug detector — the tau_p 999.9-tolerance outlier that led to the Mavrin `n_i_avg` fix would have been smeared across a dozen inputs under L2. Cost: it will abandon one input entirely to save the rest.
    - `sum` = plain L1 (`irls_iterations=0`, **already works**): still sparse, never actively abandons an input, cheaper (skips up to 3 warm-started re-solves), less decisive.
    - `least_squares` = L2: spreads the correction in proportion to tolerance. Right when inputs are measurements with error bars; never names a culprit. Only this one needs code — row `= excess` instead of `sqrt(excess)`; the other two just need naming/discoverability.

  - `movement_metric` (default `absolute`) — matters mainly under `count`, which is what creates the incentive to abandon an input.
    - `absolute` = `|x-x0|/width`. Required wherever 0 is a legitimate value (`f_He4`, every `c_*`).
    - `log` = decades, for strictly-positive variables (domain open at 0 — 102 of them, incl. `tau_p`, `V_p`). Fixes a real defect: the absolute metric caps a collapse at `x0/width = 1/rel_tol = 1000`, so driving `tau_p` 13 decades to zero costs *less* than doubling it, with no gradient left to pull it back. Any metric finite at 0 has this defect (asinh included), which is why it must key off the domain.
    - PROTOTYPED + measured (2026-07-27): deadzone preserved, collapse 9330 vs 693 for doubling; but 1 reactor better / 2 worse on the beyond-tolerance count and **6x nfev on ARC_V0**. Recommend off by default — the defect is currently latent, since the tau_p collapse that motivated it was really the Mavrin conflict.

  Also cheap and unexposed: a "movement-anchored but not blameable" tier, weaker than `fixed: true` (which today also removes the variable as a solver unknown entirely).

  UNMEASURED: how the objective interacts with popcon / parametric-sweep smoothness. L0's discrete "which input gets blamed" decision is a plausible source of the sweep scatter seen in examples/01.
