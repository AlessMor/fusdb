# Species Registry

`SPECIES` is the registry of ion species: the nuclear metadata (charge, mass)
that composition, quasineutrality and the fusion rates need, plus the record of
which atomic-data datasets describe each element. It is loaded once at import
from `species.yaml` and is immutable thereafter.

Related modules:

- `fusdb.registry.species_registry`

Related pages:

- [Variable Class](variable_class.md)
- [Responsibility Model](responsibility_model.md)

## Scope: what is a row

**Every row in `species.yaml` is a fully ionised species — a bare nucleus.**

`atomic_number` is the nuclear charge $Z$, i.e. the charge when completely
stripped; `atomic_mass` (mass number) and `isotopic_mass_u` are nuclear masses.
Fuel and ash (H/D/T, He3/He4) are assumed fully stripped in the core, which is
what makes quasineutrality and the fusion rates exact.

Impurities are the subtle case, and they are the reason no charge state appears
in the file. An impurity's actual charge is $\bar{Z}(T_e)$ — a **continuous**
function from the Mavrin/radas fits — not an integer. A YAML row can only hold
an integer, so the row is the nucleus and the ionisation state is a *relation*.

This is why neutral atoms, partially-ionised charge states and molecules are
deliberately absent -- see [Charge States, Neutrals and Molecules](#charge-states-neutrals-and-molecules).

## File Structure

The registry needs two keyings, and they are expressed **structurally** rather
than through a pointer field:

- the **element** is the top-level key. Atomic data (radas/Mavrin $L_z(T_e)$ and
  $\bar{Z}(T_e)$) is set by electronic structure, so it is stated once there and
  every isotope of that element shares it.
- the **isotope** is a key under `isotopes:`. It states only what differs from
  its element and inherits everything else; anything it does state overrides.

```yaml
He:
  full_name: helium
  atomic_number: 2
  atomic_data: [polynomialfit_mavrin_coronal, coolingcurve_radas_coronal, ...]
  isotopes:
    He3: {symbol: [He3, 3He], full_name: helium-3, atomic_mass: 3, isotopic_mass_u: 3.016}
    He4: {symbol: [He4, 4He], full_name: helium-4, atomic_mass: 4, isotopic_mass_u: 4.003}
```

`He3` and `He4` inherit `atomic_number: 2` and the whole `atomic_data` list;
they override the masses. There is no `atomic_symbol` field — the parent key
*is* the element symbol, exposed as `SpeciesSpec.element`.

Impurities have no `isotopes:` block, because nothing in fusdb resolves them
isotopically. A flat element row is the degenerate case of the nested form, so
those rows are unchanged from the pre-nesting file.

!!! note "Nesting is a file-layout choice, not a runtime cost"
    `SpeciesRegistry.from_yaml` **flattens at import** into one dict of
    fully-resolved specs. `SPECIES["He4"]` is a single `dict` access with every
    inherited field already merged — there is no parent lookup at runtime.

Both element and isotope rows are species: `SPECIES["He"]` is helium,
`SPECIES["He4"]` is helium-4. `SpeciesSpec.is_element` distinguishes them, and
`SPECIES.elements()` yields the element rows in registry order.

## Symbols and Aliases

`symbol` may be a single spelling or a list of accepted ones; every entry
resolves to the same species. It works on element rows as well as isotope rows.

```yaml
T:
  symbol: [T, H3, 3H, H_3, 3_H]   # SPECIES["3_H"] is tritium
H:
  symbol: [H, p, protium]          # SPECIES["p"] is hydrogen
```

The row's own key must come first; the loader rejects a list that starts with
anything else, and rejects an alias claimed by two rows.

`symbol` and `isotopes` are **row identity, not physical properties**, so they
are the two fields that do not inherit. Were `symbol` inherited, an isotope
that did not declare one would answer to its element's spellings — `T` would be
reachable as `H`, and two species would share a lookup key.

!!! warning "`H2` is molecular hydrogen, not deuterium"
    The AMJUEL datasets and `plotting/atomic_physics.py` use `H2`/`H2+` for the
    hydrogen molecule and its ion. Deuterium is therefore spelled `D`/`2H`/`2_H`
    and must never claim `H2` or `H_2`.

`p` resolves to `H` because a fully ionised hydrogen atom *is* a proton; that is
what lets every participant in `reactions.yaml` be a registry species, with no
"bare nucleon" escape hatch. `n` has its own row ($Z = 0$, no atomic data).

## Atomic Data

`atomic_data` lists the per-species datasets available in
`registry/dataset/radiation/`, by their `{datatype}_{source}` stem. The dataset
id is that stem plus `_{symbol}`.

This is the single source for **which species each radiation / mean-charge
method supports**. `SPECIES.with_atomic_data(stem)` returns the element symbols
carrying that dataset, in registry order:

```python
SPECIES.with_atomic_data("coolingcurve_radas_coronal")
# ('He', 'Li', 'Be', 'C', 'N', 'O', 'Ne', 'Ar', 'Xe', 'W')
```

Isotopes are excluded — they inherit their element's atomic data, so listing
them would double-count. Method-specific exceptions that used to live in prose
comments are now rows: krypton has no `coolingcurve_radas_coronal` entry
(its coronal $L_z$ is all-zero in the source `radas_dir`), and lithium has no
`coolingcurve_PROCESS_coronal` entry.

`tests/test_species_registry.py::test_atomic_data_matches_the_datasets_on_disk`
asserts the declaration equals the files actually present, so it cannot drift.

!!! danger "Never `zip()` a derived species tuple against positional arguments"
    The `c_X` keyword arguments in the radiation and mean-charge relations are
    **deliberately hand-written** — the relation graph is built from those
    parameter names, so they cannot be generated. Pair them with an explicit
    dict literal, never `dict(zip(DERIVED_TUPLE, (c_He, c_Li, ...)))`: once the
    tuple comes from YAML, reordering the file silently remaps every
    concentration onto the wrong element.

## Charge States, Neutrals and Molecules

fusdb touches four kinds of "species". Only the first is a registry row, and the
reason is empirical: **nothing in fusdb tracks a per-charge-state density.**
There is no `n_{C+}` and no `c_Ar1` — the variables are per-element
concentrations. Charge states instead arrive as *reaction labels*: the AMJUEL
datasets are self-describing (`reaction: 2.6B1 e+C+ → e+C+++e`,
`category: ionization`, `species: C_plus`) and surface in fusdb as `*_rate`
entries in `variables.yaml`, which is the right namespace for them.

| class | lives in | why |
| --- | --- | --- |
| fully ionised | `species.yaml` rows | has a nucleus: $Z$, $A$, mass |
| partially ionised | $\bar{Z}(T_e)$ via `atomic_data` | the charge is a continuous function, not an integer |
| neutral atoms | AMJUEL dataset metadata + `*_rate` variables | a reaction label, not a tracked density |
| molecules | same | same |

A row per (element, charge) would also be combinatorial: tungsten alone has 74
charge states, and only the handful AMJUEL happens to tabulate would appear —
that is dataset availability, not physics.

**The one condition that would overturn this:** if fusdb ever solves for a
charge-state density such as $n_{C^+}$, charge states become plasma
constituents and do need rows. Until then they are relations and labels.

## Extending the Registry

**A new isotope of an existing element** — add it under that element's
`isotopes:` with only the fields that differ (usually `full_name`,
`atomic_mass`, `isotopic_mass_u`) and a `symbol` list if it has alternative
spellings.

**A new radiating impurity** — add the element row *and* its `atomic_data`
entries, and drop the dataset files into `registry/dataset/radiation/`. The
registry-vs-disk test fails if you do one without the other. The relation's
`c_X` keyword argument and its dict-literal entry are added by hand.

**A non-radiating species** (needed for a reaction, a variable, or an AMJUEL
dataset) — add the element row with no `atomic_data`. `B` and `Fe` are there on
these grounds, not because they radiate.

**A charge state, neutral or molecule** — do not add it here; see
[Charge States, Neutrals and Molecules](#charge-states-neutrals-and-molecules).

## Class Structure

- `SpeciesSpec`: an **immutable, frozen** record for one species — one element,
  or one isotope of an element. Isotope specs are *resolved*: every field the
  isotope did not state has already been inherited from its element, so a spec
  is complete on its own and never consults a parent.
- `SpeciesRegistry`: the flattened, alias-indexed collection of those specs.
  `SPECIES` is the process-wide instance, built from `species.yaml` at import.

## Shared Fields

- `key`: canonical species name, and the row's key in `species.yaml`
- `element`: the parent element's key (its own key, for element rows)
- `aliases`: every accepted spelling, canonical one first
- `full_name`: human-readable name
- `atomic_number`: nuclear charge $Z$ (the charge when fully stripped)
- `atomic_mass`: mass number $A$, an integer
- `isotopic_mass_u`: nuclear mass in unified atomic mass units
- `atomic_data`: `{datatype}_{source}` stems of the available radiation datasets

Two derived properties: `symbol` (the canonical spelling, i.e. `key`) and
`is_element` (whether this is the element row rather than one of its isotopes).

## API and behavior

```python
from fusdb.registry import SPECIES

SPECIES["He4"].atomic_number        # 2   -- inherited from the He element row
SPECIES["3_H"] is SPECIES["T"]      # True -- alias lookup
"p" in SPECIES                      # True -- alias-aware membership
```

- `SPECIES[key]` / `SPECIES.get(key)` -> the `SpeciesSpec` for a canonical name
  **or any of its aliases**; `KeyError` if unknown.
- `key in SPECIES` -> alias-aware membership.
- iteration -> every row, elements and isotopes alike, in file order.
- `SPECIES.elements()` -> the element rows only, in file order.
- `SPECIES.with_atomic_data(stem)` -> element symbols carrying that dataset,
  in file order (see [Atomic Data](#atomic-data)).

The registry is built once at import and is read-only: `SpeciesSpec` is frozen,
and the underlying maps are `MappingProxyType`. `SpeciesRegistry.from_yaml`
raises on a `symbol` list that does not start with the row's own key, on an
alias claimed by two rows, and on a species declared twice.

## Migration Record

The pre-nesting file was flat and keyed by species, with an `atomic_symbol`
field pointing at the element. Moving to nesting changed **no value**:
`tests/test_species_registry.py::GOLDEN` pins every key that existed before to
its exact prior `full_name` / `element` / `atomic_number` / `atomic_mass` /
`isotopic_mass_u`, and all five derived dataset tuples reproduce the previously
hardcoded ones in order. `scripts/reconcile_all.py` was unchanged on
`dim`/`nfev`/`res_sz`/`#act`/`>tol` across every solving reactor.

Three hardcoded species tuples (`_MAVRIN_CHARGE_SPECIES`, `_RADAS_LZ_SPECIES`,
`_PROCESS_LZ_SPECIES`) were replaced: one now derives from `atomic_data`, and
two were deleted outright once their last readers went away.
