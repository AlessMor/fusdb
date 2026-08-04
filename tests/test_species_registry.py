"""Species registry: migration pin + structural checks.

``GOLDEN`` was dumped from the flat ``allowed_species.yaml`` that preceded the
nested ``species.yaml``.  It is the proof that moving to nesting + inheritance
changed no value: every key that existed before must still resolve to exactly
the same fields.  New keys (the element rows that carry the atomic data, e.g.
``He``) are allowed and checked separately.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fusdb.registry import SPECIES

# key: (full_name, element, atomic_number, atomic_mass, isotopic_mass_u)
GOLDEN = {
    # `H` is the ELEMENT row and carries no mass: protium was split out into
    # its own isotope row below, so that `c_H` (per element) and `n_p`/`f_p`
    # (protium only) can never resolve to the same species.  The mass fields
    # the flat file put on `H` are pinned on `p` instead -- nothing was lost.
    "H": ("hydrogen", "H", 1, None, None),
    "p": ("protium", "H", 1, 1, 1.008),
    "D": ("deuterium", "H", 1, 2, 2.014),
    "T": ("tritium", "H", 1, 3, 3.016),
    "He3": ("helium-3", "He", 2, 3, 3.016),
    "He4": ("helium-4", "He", 2, 4, 4.003),
    "Li": ("lithium", "Li", 3, 7, 6.941),
    "Be": ("beryllium", "Be", 4, 9, 9.012),
    "C": ("carbon", "C", 6, 12, 12.011),
    "N": ("nitrogen", "N", 7, 14, 14.007),
    "O": ("oxygen", "O", 8, 16, 15.999),
    "Ne": ("neon", "Ne", 10, 20, 20.180),
    "Ar": ("argon", "Ar", 18, 40, 39.948),
    "Kr": ("krypton", "Kr", 36, 84, 83.798),
    "Xe": ("xenon", "Xe", 54, 131, 131.293),
    "W": ("tungsten", "W", 74, 184, 183.84),
}


@pytest.mark.parametrize("key", sorted(GOLDEN))
def test_pre_nesting_rows_resolve_unchanged(key: str) -> None:
    """Every species that existed before nesting resolves to the same values."""
    spec = SPECIES[key]
    assert (
        spec.full_name,
        spec.element,
        spec.atomic_number,
        spec.atomic_mass,
        spec.isotopic_mass_u,
    ) == GOLDEN[key]


def test_isotopes_inherit_and_override() -> None:
    """Isotopes take the element's properties unless they state their own."""
    assert SPECIES["T"].atomic_number == SPECIES["H"].atomic_number  # inherited
    assert SPECIES["T"].atomic_mass != SPECIES["H"].atomic_mass  # overridden
    # Atomic data is a property of the electronic structure, so both helium
    # isotopes inherit the same set from the element.
    assert SPECIES["He3"].atomic_data == SPECIES["He4"].atomic_data == SPECIES["He"].atomic_data


def test_symbol_aliases_resolve_to_the_same_spec() -> None:
    """Alternative spellings declared in ``symbol`` reach the canonical row."""
    for alias in ("T", "H3", "3H", "H_3", "3_H"):
        assert alias in SPECIES
        assert SPECIES[alias] is SPECIES["T"]
    assert SPECIES["T"].symbol == "T"
    assert SPECIES["4He"] is SPECIES["He4"]


def test_element_rows_may_declare_symbols_too() -> None:
    """``symbol`` is not isotope-only: element rows carry alias lists as well."""
    assert SPECIES["neutron"] is SPECIES["n"]


def test_protium_resolves_to_the_isotope_not_the_element() -> None:
    """``p`` is the light isotope, NOT hydrogen-the-element.

    Composition is keyed two ways and they must not collapse onto one row:
    ``c_H`` is per ELEMENT (p + D + T over n_e) while ``n_p``/``f_p`` are
    protium alone.  While the aliases sat on the element row, ``SPECIES["p"]``
    answered with all of hydrogen, so a species loop reading ``p`` would have
    silently picked up the deuterium and tritium inventory too.
    """
    assert SPECIES["p"] is SPECIES["protium"]
    assert SPECIES["p"] is not SPECIES["H"]
    assert SPECIES["p"].element == "H"


def test_atomic_data_matches_the_datasets_on_disk() -> None:
    """``atomic_data`` declares exactly the per-species datasets that exist.

    This is what lets the radiation/mean-charge modules derive their supported
    species from the registry instead of hardcoding them: the declaration is
    only trustworthy if it cannot drift from ``dataset/radiation/``.
    """
    from fusdb.registry import dataset as dataset_pkg

    root = Path(dataset_pkg.__file__).parent / "radiation"
    on_disk: dict[str, set[str]] = {}
    for path in root.glob("*.yaml"):
        stem, _, symbol = path.stem.rpartition("_")
        on_disk.setdefault(stem, set()).add(symbol)

    declared: dict[str, set[str]] = {}
    for spec in SPECIES.elements():
        for stem in spec.atomic_data:
            declared.setdefault(stem, set()).add(spec.key)

    assert declared == on_disk


@pytest.mark.parametrize(
    ("stem", "expected"),
    [
        # Pinned from the tuples these replaced: order is load-bearing only in
        # that the relations iterate it, but a silent change here would silently
        # change which impurities radiate.
        ("polynomialfit_mavrin_coronal", ("He", "Li", "Be", "C", "N", "O", "Ne", "Ar", "Kr", "Xe", "W")),
        # H leads this one deliberately: it is the only family that tabulates
        # hydrogen, which is what lets a species loop cover the fuel too.
        ("coolingcurve_PROCESS_coronal", ("H", "He", "Be", "C", "N", "O", "Ne", "Ar", "Kr", "Xe", "W")),
        ("coolingcurve_radas_coronal", ("He", "Li", "Be", "C", "N", "O", "Ne", "Ar", "Xe", "W")),
    ],
)
def test_with_atomic_data_reproduces_the_hardcoded_tuples(stem: str, expected: tuple[str, ...]) -> None:
    assert SPECIES.with_atomic_data(stem) == expected


def test_with_atomic_data_excludes_isotopes() -> None:
    """He3/He4 inherit He's atomic data but must not be listed alongside it."""
    assert "He4" not in SPECIES.with_atomic_data("coolingcurve_radas_coronal")


def test_identity_fields_are_not_inherited() -> None:
    """``symbol`` is row identity, not a physical property: it never inherits.

    Were it inherited, ``T`` would answer to ``H`` and two species would share
    a lookup key.
    """
    assert SPECIES["H"] is not SPECIES["T"]
    assert "H" not in SPECIES["T"].aliases


def test_every_element_with_isotope_fractions_has_a_concentration_bridge() -> None:
    """Each element carrying isotope-resolved ``f_*`` needs one ``c_*`` bridge.

    fusdb keeps composition two ways on purpose: ``f_X`` is isotope-keyed and
    denominated in ``n_i`` (what reactivity and the ash balance need), ``c_X``
    is element-keyed and denominated in ``n_e`` (what the cooling curves, Mavrin
    Zbar and Z_eff need, all set by electronic structure).  The two are linked
    by ``c_X = sum(f_isotopes) * n_i/n_e``.  A new isotope fraction added
    without extending its element's bridge would silently drop that isotope out
    of every atomic-data path, so assert the coverage instead of trusting it.
    """
    from fusdb.registry import RELATIONS, VARIABLES

    fractions_by_element: dict[str, set[str]] = {}
    for species in SPECIES:
        # Aliases count: a fraction may be spelled with any of its row's
        # accepted symbols (``f_p`` reaches the ``p`` row via its alias list).
        for spelling in {species.key, *species.aliases}:
            name = f"f_{spelling}"
            if name in VARIABLES:
                fractions_by_element.setdefault(species.element, set()).add(name)

    assert fractions_by_element, "no f_<isotope> variables found -- test is vacuous"

    def consumed(relation) -> set[str]:
        # Optional isotopes are kwargs-with-defaults, so they land in
        # ``constant_names`` rather than ``argument_names``.  They still enter
        # the sum, and they MUST stay optional: requiring ``f_He3`` positionally
        # made the helium bridge unevaluable on any D-T machine, which pruned it
        # and silently dropped helium out of the radiation entirely.
        return set(relation.argument_names) | set(relation.constant_names)

    for element, fractions in sorted(fractions_by_element.items()):
        bridges = [r for r in RELATIONS if r.output_names == (f"c_{element}",)
                   and fractions & consumed(r)]
        assert bridges, f"no c_{element} bridge for fractions {sorted(fractions)}"
        covered: set[str] = set()
        for bridge in bridges:
            covered |= consumed(bridge)
        missing = fractions - covered
        assert not missing, f"c_{element} bridge ignores {sorted(missing)}"
