# VariableRegistry

Stores the allowed variable schema.

## Expected behavior:

- Loads allowed_variables.yaml.
- Stores canonical variable names.
- Stores aliases for each canonical variable.
- Checks duplicate canonical names.
- Checks duplicate aliases.
- Checks alias/canonical-name collisions.
- Resolves aliases to canonical names.
- Stores default computation unit.
- Stores shape: 0 for scalar, 1 for profile.
- Stores default variable-level constraint strings.
- Does not store reactor-specific values.

## Example:

R:
  aliases:
    - major_radius
  unit: m
  shape: 0
  constraints:
    enforce:
      - R > 0

## Internally:

major_radius -> R
R -> R

The registry is the only source of truth for aliases.

# Variable

Stores one scenario-specific variable instance.

## Expected behavior:

- Stores canonical name only.
- Does not store aliases.
- Resolves input name through VariableRegistry at construction.
- Stores value, if provided.
- Converts value to registry computation unit before runtime.
- Stores unit as the registry computation unit after conversion.
- Stores rel_tol.
- Stores fixed True/False.
- Stores size for profile variables.
- Stores shape copied from VariableRegistry.
- Accepts scalar input for profile variables and expands to constant profile if size/grid is known.
- Does not own default constraints.

## Recommended fields:

Variable:
    name: str                  # canonical after construction
    value: float | ndarray | None
    unit: str                  # computation unit after conversion
    rel_tol: float | None
    fixed: bool
    shape: int                 # 0 or 1, from registry
    size: int | None

Variable-level constraints come from VariableRegistry and are converted into Relation objects when building the active model.

# Relation

The only equation/constraint object.

A Relation represents:

lhs_function(inputs) op rhs

where rhs may be:

- numeric constant
- variable name
- tuple of variable names

## Expected behavior:

- Stores relation name.
- Stores input names from function signature.
- Stores constants from default-valued function parameters.
- Stores rhs.
- Stores op: ==, !=, <, <=, >, >=.
- Stores enforce True/False.
- Stores profile_mode: pointwise or global.
- Stores tags.
- Can be used independently of Pyomo.
- Can evaluate lhs from known inputs.
- Can check lhs op rhs when rhs values are also supplied.
- Does not import or expose Pyomo.

## Recommended fields:

Relation:
    name: str
    input_names: tuple[str, ...]
    constant_names: tuple[str, ...]
    func: Callable
    rhs: str | tuple[str, ...] | int | float
    op: str = "=="
    enforce: bool = True
    profile_mode: str = "pointwise"  # or "global"
    tags: tuple[str, ...] = ()
Output-defining relation
@relation(rhs="A")
def aspect_ratio(R, a):
    return R / a

Means:

R / a == A
Constraint-like relation
@relation(op="<=", rhs=3)
def beta_limit(beta_N):
    return beta_N

Means:

beta_N <= 3
Global profile constraint
@relation(op="==", rhs=0, profile_mode="global")
def he4_particle_balance(rho, n_D, n_T, n_He4, sigmav_DT, tau_p_He4):
    local = n_D * n_T * sigmav_DT - n_He4 / tau_p_He4
    return trapz(local * 2.0 * rho, rho)

Means:

∫ dn_He4/dt dV == 0

No dummy output variable is required.

## @relation decorator

The only decorator.

### Expected behavior:

- Creates a Relation object.
- Requires rhs.
- Defaults op to "==".
- Infers inputs from function signature.
- Treats parameters with defaults as constants.
- Does not use return annotations.
- Resolves variable names through VariableRegistry.
- Allows rhs to be a variable alias, canonical variable name, tuple of names, or number.
- Allows tags.
- Allows enforce True/False.
- Allows profile_mode.
- Accepts simple relation-local constraints as strings, which are converted into additional Relation objects.

### Example:

@relation(rhs="P_fus_DT", constraints=["P_fus_DT >= 0"])
def fusion_power_dt(P_fus_DT_alpha, P_fus_DT_n):
    return P_fus_DT_alpha + P_fus_DT_n

### Produces:

1. fusion_power_dt:
       P_fus_DT_alpha + P_fus_DT_n == P_fus_DT

2. fusion_power_dt_constraint_0:
       P_fus_DT >= 0

Both are Relation objects.

# UI/YAML constraints

Since Constraint is removed, simple constraints are represented as strings and parsed into outputless Relation objects.

Example variable registry:

R:
  aliases:
    - major_radius
  unit: m
  shape: 0
  constraints:
    enforce:
      - R > 0
    warn:
      - R < 20

Parsed as:

Relation:
    lhs = R
    op = >
    rhs = 0
    enforce = True

Relation:
    lhs = R
    op = <
    rhs = 20
    enforce = False

Global reactor constraints:

constraints:
  enforce:
    - f_D + f_T + f_He3 + f_He4 == 1
  warn:
    - beta_N < 4

Also become Relation objects.

Recommended parser support:

Allowed:
    names
    numbers
    + - * / **
    parentheses
    comparisons

Rejected:
    function calls
    attribute access
    indexing
    strings
    imports
    boolean logic

Profile/integral constraints should be Python-decorated @relation(...), not YAML strings.

# RelationRegistry

Stores available Relation objects.

## Expected behavior:

- Stores registered Relation objects.
- Checks duplicate relation names.
- Emits warning when multiple relations share the same rhs/output variable set.
- Loads relation defaults from relation_defaults.yaml.
- Selects relations by name.
- Selects all relations when no include list is provided.
- Resolves duplicate-output alternatives using defaults or reactor relation-name tags.

## Example defaults:

tau_E: tau_E_ipb98y2
P_fus: total_fusion_power

If several active relations define tau_E, the registry chooses:

1. relation explicitly named in reactor tags, if present
2. relation_defaults.yaml default
3. otherwise error

# TagRegistry

Stores allowed tags and tag hierarchy.

## Expected behavior:

- Loads allowed_tags.yaml.
- Forgives upper/lower case.
- Does exact normalized matching otherwise.
- Rejects unknown tags unless they match a registered relation name.
- Supports tag groups.
- Supports child implying parent.
- Interprets tags in same group as OR.
- Interprets tags across different groups as AND.

## Example YAML:

device:
  - tokamak:
      - spherical_tokamak
      - compact_tokamak
      - conventional_tokamak
  - stellarator

confinement_mode:
  - h_mode
  - l_mode

Relation:

@relation(rhs="tau_E", tags=("tokamak", "h_mode", "l_mode"))
def tau_E_ipb98y2(...):
    ...

Means:

device: tokamak
AND
confinement_mode: h_mode OR l_mode

A reactor tagged spherical_tokamak also matches tokamak.

# Reactor

Stores one scenario.

## Expected behavior:

- Reads reactor.yaml.
- Stores metadata.
- Stores reactor tags.
- Stores raw selected relation include/exclude/order settings.
- Stores scenario variables as Variable objects.
- Converts input values to computation units.
- Expands scalar values to constant profiles when a profile variable is expected and grid size is known.
- Stores reactor-level constraint strings.
- Does not compile Pyomo directly except through to_relation_system() or run().

Example relation section:

relations:
  include: null
  exclude: []
  order: []

## Expected relation selection:

If include is provided:
    use exactly those relation names
    apply exclude
    skip tag filtering

If include is omitted/null:
    start from all registry relations
    keep generic untagged relations
    keep tagged relations matching reactor tags
    apply exclude
    resolve duplicate rhs/output alternatives

order does not select relations. It only changes ordered execution preference.

# RelationSystem

Receives concrete Variable objects and active Relation objects.

## Expected behavior:

- Stores active variables.
- Stores active relations.
- Builds variable lookup by canonical name.
- Checks duplicate active variable names.
- Checks duplicate active relation names.
- Checks that every relation input and rhs variable exists in active variables.
- Collects variable-registry constraints and reactor constraints as additional Relation objects before compile.
- Does not resolve aliases; objects should already be canonicalized.
- Does not know about YAML.
- Does not expose Pyomo to user objects.

The smallest interface between RelationSystem and Relation is:

relation.input_names
relation.rhs
relation.op
relation.enforce
relation.profile_mode
relation.evaluate(namespace)

RelationSystem compiles:

lhs_function(inputs) op rhs

into Pyomo.

Pointwise mode

For profile variables:

@relation(rhs="Rr_DT", profile_mode="pointwise")
def reaction_rate_DT(n_D, n_T, sigmav_DT):
    return n_D * n_T * sigmav_DT

Compiles to:

n_D[i] * n_T[i] * sigmav_DT[i] == Rr_DT[i]
for every i
Global mode
@relation(op="==", rhs=0, profile_mode="global")
def he4_particle_balance(...):
    return trapz(...)

Compiles to one global Pyomo constraint:

trapz(...) == 0
RelationSystem modes
### verify

Check whether provided values satisfy the active relations.

#### Expected behavior:

- Variables with values are fixed.
- fixed flag is irrelevant for variables with values; they are fixed anyway.
- Variables without values are free unknowns.
- Enforced relations are compiled.
- Warning relations are evaluated after solve/check.
- If infeasible, diagnostic soft-relation mode may be used.

Interpretation:

"Do these inputs form a feasible model?"
### reconcile
Adjust non-fixed values minimally to satisfy the model.

#### Expected behavior:

- fixed=True variables are fixed.
- fixed=False variables with values are initialized but free.
- Variables with values are penalized for deviation using rel_tol.
- Variables without values are free.
- Objective minimizes normalized deviation from supplied values.

Interpretation:

"Find the closest feasible scenario to my supplied data."
### optimize

Optimize an objective while respecting fixed values and ranges.

#### Expected behavior:

- fixed=True variables are fixed.
- Variables with value and no active range/bounds are fixed by default.
- Variables with value and active range/bounds are initialized at value but free within range.
- Variables without value are free, bounded if constraints/ranges exist.
- Objective is user-provided.

Important consequence:

A value alone means fixed in optimize.
A value plus a range means initial guess inside allowed range.

Example:

P_fus:
  value: 500
  unit: MW

fixed in optimize unless a range exists.

P_fus:
  value: 500
  unit: MW
  constraints:
    enforce:
      - P_fus >= 300e6
      - P_fus <= 800e6

free in optimize within bounds.

### ordered

Run relations in a requested order without building a full global solve,
except for grouped blocks.

#### Expected behavior:

- Accepts relation names.
- Accepts tag groups.
- Relations may be repeated.
- A single relation entry is executed directly if possible.
- A tag group is executed as a verify block.
- The output of earlier steps becomes input to later steps.
- If a relation cannot be directly evaluated because multiple unknowns remain, it fails unless it is inside a verify block.

#### Example:

system.ordered([
    "aspect_ratio",
    {"tags": ["fusion"]},
    "total_fusion_power",
])

Expected behavior:

1. Run aspect_ratio directly.
2. Build a temporary verify block with all active fusion-tagged relations.
3. Run total_fusion_power directly.

For grouped tag blocks:

- collect matching relations
- compile temporary RelationSystem in verify mode
- fix currently known values
- solve missing block variables
- merge results back into ordered state

This gives you a hybrid between sequential evaluation and local solve blocks.

# Final conceptual model

The refactor reduces the system to:

VariableRegistry:
    allowed variable schema

Variable:
    scenario value

Relation:
    every equation, relation, and constraint

RelationRegistry:
    available model equations

Reactor:
    scenario + relation selection

RelationSystem:
    variables + relations -> Pyomo model