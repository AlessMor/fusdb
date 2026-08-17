# TODO

Open items only — delete an entry when it is done. Landed work and its measurements go to
memory (`~/.claude/projects/<slug>/memory/`), never here; the
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

- **The ST / stellarator scaling subdivision is unverified** (`relations/confinement/scalings/
  __init__.py`). It decides which confinement scalings each device class can see, and two
  reactors currently have NO confinement closure at all (Polomac, HAMMIR — see Infrastructure),
  so a wrong subdivision is not cosmetic. Establish which scalings legitimately apply to ST and
  stellarator before adding any device-class scaling.

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

- **Only 1 of 11 reactors declares any heating channel.** `Total auxiliary power` now reads its
  four channels as optional contributors and activates for whichever a scenario declares, but
  only Polomac uses it (`P_ECRH` -> `P_aux = 7.5 kW`). Everywhere else `P_aux` is either a lump
  input (ARC_V3A 21.5, DEMO 110.6, INFINITY_TWO 21, SPARC 40.5, STELLARIS 49 MW) or falls out of
  the power balance (ARC_V0 1722, GIGA 624, STEP 129 MW). Declaring the channels each machine
  actually has — from the same papers the other inputs come from — turns those lumps into a
  checkable decomposition; where `P_aux` is ALSO declared the sum becomes a consistency check,
  which is how ARC_V3A's `P_RF,max` was caught being recorded as an operating value.

- **`species.yaml` carries no per-species mass in kg** (marked `TODO(high)` in the file), and no
  per-species atomic data (`TODO(low)`; radas-style, as cfspopcon has). Masses are currently
  derived where needed rather than stored, which is why the isotopic-vs-mass-number convention
  had to be settled by hand.

- **Developer-guide gaps flagged in `docs/developers_guide/`**: no linting setup
  (`TODO(med)`, PROCESS uses ruff), no testing/debugging guide, no pre-commit guide. Cheap and
  independent of everything else here.

- **Decouple the solver parameterisation from the declared domain.** `pack_scalar` gates its
  log transform on the SOLVER lower bound, so a physics statement silently chooses solver
  coordinates. Blocks two things: setting `Z_eff` domain to the physically correct `[1, inf)`
  (STELLARIS moves off its pinned design point), and deleting the ~31 `solver_domain:
  [1e-12, inf)` declarations that exist only to trip that branch. Widening the gate to
  "non-negative physical domain" was measured and rejected (GIGA 0 -> 2 beyond tolerance).
  See memory `zero-tol-absolute-magnitude`.

- **`success` is not a physical verdict.** It reports equation balance only, so a reactor
  certifies at a meaningless point: ARC_V0 `success=True` at Q = 0.30 with 288 tolerances of
  `tau_E` movement, SPARC at `P_fus = 6.8e-06 MW` under an experimental setting. Fold
  `inputs_beyond_tolerance` into the verdict, or add a second field. Cheapest open item, and it
  would have surfaced ARC_V0 long ago.

- **The `f_GW >= 1%` non-triviality certifier only catches DENSITY collapse.** SPARC collapsed
  at `f_GW = 0.175` (density kept, temperature and fuel lost) and ARC_V0 at `f_GW` fine but
  `tau_E` down 9.4x. A confinement-side discriminant would have to compare `tau_E` against its
  scaling, not against `H98` — ARC_V0 holds `H98 = 1.80` while `tau_E` collapses.

- **Polomac and HAMMIR have no confinement closure.** No scaling matches their tags, so
  `W_th = P_loss * tau_E` is one equation in two unknowns and `tau_E` runs free (~1e12 s for
  Polomac). Pinned in `test_polomac_uses_default_producers_and_is_physical`; HAMMIR is still
  latent because it declares no auxiliary channel. Give each a `tau_E` determination, then
  tighten that test back to `success is True`.

- **Gi bootstrap scaling returns ~1e4 for a flat temperature profile.**
  `calc_bootstrap_fraction` computes `temp_delta**-0.416` with `temp_delta` floored at 1e-12,
  so `temperature_peaking = 1.0` gives `f_BS = 11379` into a variable whose domain is [0, 1]
  (1.05 -> 0.445 for comparison). A flat profile is legitimate for any reactor declaring
  neither species' temperature peaking. The scaling should refuse rather than return 1e4.

- **`INVERSE_BOUND_FLOOR = 1e-12` is the same absolute-epsilon hazard, untouched.** It brackets
  the per-relation inverse root-find above the physical range of any small-scale target. No
  reachable defect today; `abs_tol` is NOT a safe substitute (`T_e`'s is 0.05 keV). Fix in log
  coordinates when the parameterisation item above is settled.

- **`structural_blocks` is computed pre-prune.** The DM partition runs on the candidate pool
  and its output is saved without recomputation, while consumers (popcon's certification cone,
  reconcile's x0 block solver) read it as describing the ACTIVE system. Mitigated — relations
  that cannot survive any matching are now dropped first — but the underlying prune <-> partition
  ordering is unresolved, and reconcile's x0 seeding is load-bearing.

- **Decide ARC_V0 (ARC 2015) pinning.** An all-29-variables-pinned copy is measured:
  `P_aux` 1722 -> 94.2 MW, `Q` 0.30 -> 5.57, 13 named contradictions including four CSV-profile
  vs declared-average mismatches. Not applied. Trade-off: it costs the culprit ranking, so the
  minimum pin set may be preferable (`tau_E` plus geometry is the first thing to measure).

- rewrite tests

- Selectable reconcile movement penalty — **LANDED 2026-08-12**. The "culprit" verdict (`inputs_beyond_tolerance`) is an optimiser *outcome*, not a ranking: it falls out of eligibility (`fixed` vs supplied) x deadzone (tolerances) x distance metric x aggregation norm x relation-weight continuation. (`likely_culprits` is a different, purely post-hoc thing: a count of how many failed relations each variable appears in, with no feedback into the solve.) Both knobs now exist on `reconcile.run`, defaulting to the behaviour that shipped before them; an unrecognised value is rejected with `termination="invalid options"` rather than falling through to the default. Both are reported back on `result["solver"]`.

  - `movement_objective` (default `count`) — *what belief about the data you are encoding*.
    - `count` = L0 via IRLS `1/(excess+eps)`: minimises HOW MANY inputs are contradicted. Short, nameable verdict; right prior for published design points. Also a good model-bug detector — the tau_p 999.9-tolerance outlier that led to the Mavrin `n_i_avg` fix would have been smeared across a dozen inputs under L2. Cost: it will abandon one input entirely to save the rest.
    - `sum` = plain L1: still sparse, never actively abandons an input, cheaper (skips up to 3 warm-started re-solves), less decisive. Same solve as `irls_iterations=0`, which still works and is still honoured under `count`.
    - `least_squares` = L2: spreads the correction in proportion to tolerance. Right when inputs are measurements with error bars; never names a culprit. Row is `weight * excess` instead of `sqrt(weight * excess)`, and is never reweighted.
    - MEASURED on STELLARIS (2026-08-12, one process each, `success=True` and 0 failed relations throughout): `count` blames **2** inputs (tau_E 97.4, T_e_avg 47.7); `sum` blames **9** — the same two plus V_p 25.4, then six at 1.0–1.09 tolerances (beta_N, P_fus, n0, P_aux, tau_p, n_e_avg), marginal crossings that are artifacts of the norm rather than findings; `least_squares` blames **9** with the movement spread much wider (T_e_avg 70.7 down to Z_eff 1.32) and no culprit distinguishable. Same physics, three different verdicts — which is the whole point of naming the choice.

  - `movement_metric` (default `auto`) — matters mainly under `count`, which is what creates the incentive to abandon an input.
    - `auto` = **what the code has actually done since 2026-07-30** (`8720735`): each variable's declared domain picks its own metric, log for the 117 whose lower bound excludes zero, absolute for the other 452. Kept as the default because it is byte-identical to the shipped behaviour, *not* because it is the right answer — see the defect below.
    - `absolute` = `|x-x0|/width` for every input. Required wherever 0 is a legitimate value (`f_He4`, every `c_*`), and the only setting under which `deviation_tol` is one unit across the whole report.
    - `log` = decades wherever a log distance is defined, falling back to absolute per variable where it is not (an input supplied as exactly 0). Fixes a real defect: the absolute metric caps a collapse at `x0/width`, so driving `tau_p` 13 decades to zero costs *no more* than doubling it, with no gradient left to pull it back. Measured on tau_E at ref 1.0, tolerance width 5e-3 (abs_tol-dominated there, so the cap is 200 rather than the `1/rel_tol` = 1000 you get where rel_tol sets the width): doubling costs 199 absolute vs 138 log, but collapsing to 1e-12 costs **199 absolute vs 5539 log** — i.e. exactly the same as doubling, under absolute. Any metric finite at 0 has this defect (asinh included), which is why it keys off the domain.
    - Earlier prototype note (2026-07-27) recommended log off by default; it was superseded by `8720735` landing it always-on three days later, and the recommendation was never applied. Recorded here so the drift is not re-derived: the 2026-07-27 measurements (1 reactor better / 2 worse, 6x nfev on ARC_V0) were taken against a tree where absolute was universal, which is no longer any reachable configuration.

  OPEN, and the reason `auto` is only a compatibility default: under `auto` the metric is chosen by a YAML punctuation detail rather than a modelling decision. `tau_E` (domain `(0, inf)`) is log-normalised while `P_fus` (domain `[0, inf)`) is not, though both are physically positive — so the two sit side by side in one `inputs_beyond_tolerance` list carrying different units and are implicitly ranked against each other. Deciding the default properly needs a full-reactor before/after on `absolute` vs `auto`; STELLARIS alone moves from tau_E 97.4 / T_e_avg 47.7 (`auto`) to 82.7 / 43.2 (`absolute`), same two culprits, and to a *different* set of three under `log` (tau_E 136.9, P_aux 49.6, T_e_avg 47.9).

  Also cheap and unexposed: a "movement-anchored but not blameable" tier, weaker than `fixed: true` (which today also removes the variable as a solver unknown entirely).

  UNMEASURED: how the objective interacts with popcon / parametric-sweep smoothness. L0's discrete "which input gets blamed" decision is a plausible source of the sweep scatter seen in examples/01 — `movement_objective="least_squares"` is now the cheap way to test that.
