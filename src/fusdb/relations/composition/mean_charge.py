"""Mean ionisation states Zbar_X(T_e).

The single home of the coronal-equilibrium mean charge.  Each element has one
strong ``radas coronal`` producer (the default Zbar_X source) and a weak Mavrin
fallback; a reactor that knows its own charge states supplies ``Zbar_X`` and
overrides both.  The second charge moment ``Zsq_X`` = <q^2> follows the same
pattern, from the radas charge-state distribution.

Hydrogen is absent on purpose: it is fully stripped at any temperature fusdb
models, so it enters the charge moments at Zbar = 1, <q^2> = 1 exactly.

Zbar is LINEAR in the charge (sum_q q n_q / sum_q n_q); <q^2> is its second
moment.  Z_eff = sum_X c_X <q^2>_X is built from these.
"""

from typing import Any

from fusdb.relation import relation

from ..radiation.impurity_radiation import (
    mavrin_mean_charge,
    radas_mean_charge,
    radas_mean_square_charge,
    _RADAS_MEANCHARGE_SPECIES,
    _RADAS_MEANSQ_SPECIES,
)


@relation(
    name="Mean ionisation state of helium (Mavrin coronal)",
    tags=("default", "plasma", "composition"),
    outputs="Zbar_He",
)
def mean_ionisation_state_helium(T_e_avg: Any) -> Any:
    """Zbar_He(T_e_avg) from the Mavrin 2018 coronal fit."""
    return mavrin_mean_charge("He", T_e_avg)


@relation(
    name="Mean ionisation state of lithium (Mavrin coronal)",
    tags=("default", "plasma", "composition"),
    outputs="Zbar_Li",
)
def mean_ionisation_state_lithium(T_e_avg: Any) -> Any:
    """Zbar_Li(T_e_avg) from the Mavrin 2018 coronal fit."""
    return mavrin_mean_charge("Li", T_e_avg)


@relation(
    name="Mean ionisation state of beryllium (Mavrin coronal)",
    tags=("default", "plasma", "composition"),
    outputs="Zbar_Be",
)
def mean_ionisation_state_beryllium(T_e_avg: Any) -> Any:
    """Zbar_Be(T_e_avg) from the Mavrin 2018 coronal fit."""
    return mavrin_mean_charge("Be", T_e_avg)


@relation(
    name="Mean ionisation state of carbon (Mavrin coronal)",
    tags=("default", "plasma", "composition"),
    outputs="Zbar_C",
)
def mean_ionisation_state_carbon(T_e_avg: Any) -> Any:
    """Zbar_C(T_e_avg) from the Mavrin 2018 coronal fit."""
    return mavrin_mean_charge("C", T_e_avg)


@relation(
    name="Mean ionisation state of nitrogen (Mavrin coronal)",
    tags=("default", "plasma", "composition"),
    outputs="Zbar_N",
)
def mean_ionisation_state_nitrogen(T_e_avg: Any) -> Any:
    """Zbar_N(T_e_avg) from the Mavrin 2018 coronal fit."""
    return mavrin_mean_charge("N", T_e_avg)


@relation(
    name="Mean ionisation state of oxygen (Mavrin coronal)",
    tags=("default", "plasma", "composition"),
    outputs="Zbar_O",
)
def mean_ionisation_state_oxygen(T_e_avg: Any) -> Any:
    """Zbar_O(T_e_avg) from the Mavrin 2018 coronal fit."""
    return mavrin_mean_charge("O", T_e_avg)


@relation(
    name="Mean ionisation state of neon (Mavrin coronal)",
    tags=("default", "plasma", "composition"),
    outputs="Zbar_Ne",
)
def mean_ionisation_state_neon(T_e_avg: Any) -> Any:
    """Zbar_Ne(T_e_avg) from the Mavrin 2018 coronal fit."""
    return mavrin_mean_charge("Ne", T_e_avg)


@relation(
    name="Mean ionisation state of argon (Mavrin coronal)",
    tags=("default", "plasma", "composition"),
    outputs="Zbar_Ar",
)
def mean_ionisation_state_argon(T_e_avg: Any) -> Any:
    """Zbar_Ar(T_e_avg) from the Mavrin 2018 coronal fit."""
    return mavrin_mean_charge("Ar", T_e_avg)


@relation(
    name="Mean ionisation state of krypton (Mavrin coronal)",
    tags=("default", "plasma", "composition"),
    outputs="Zbar_Kr",
)
def mean_ionisation_state_krypton(T_e_avg: Any) -> Any:
    """Zbar_Kr(T_e_avg) from the Mavrin 2018 coronal fit."""
    return mavrin_mean_charge("Kr", T_e_avg)


@relation(
    name="Mean ionisation state of xenon (Mavrin coronal)",
    tags=("default", "plasma", "composition"),
    outputs="Zbar_Xe",
)
def mean_ionisation_state_xenon(T_e_avg: Any) -> Any:
    """Zbar_Xe(T_e_avg) from the Mavrin 2018 coronal fit."""
    return mavrin_mean_charge("Xe", T_e_avg)


@relation(
    name="Mean ionisation state of tungsten (Mavrin coronal)",
    tags=("default", "plasma", "composition"),
    outputs="Zbar_W",
)
def mean_ionisation_state_tungsten(T_e_avg: Any) -> Any:
    """Zbar_W(T_e_avg) from the Mavrin 2018 coronal fit."""
    return mavrin_mean_charge("W", T_e_avg)


# ── radas coronal mean charge (the DEFAULT Zbar; Mavrin above is the fallback) ──
# One relation per element that has a `meancharge_radas_coronal` dataset.  These
# are NON-default (strong providers) while the Mavrin producers above are now
# `default` (weak fallbacks), so radas wins the Zbar_X provider slot wherever its
# table exists; Mavrin is used only if a radas dataset is absent, or when a
# reactor selects a Mavrin relation by name.  A reactor that supplies Zbar_X
# still overrides both (supplied > derived).
# radas reads (T_e_avg, n_e_avg) -- the table is 2-D in (T_e, n_e) -- vs Mavrin's
# T_e-only fit; it is materially better for partially-ionised heavies (W ~+25%
# at 0.4-1.2 keV) and at edge temperatures where the Mavrin fit floors.


def _make_radas_meancharge_relation(symbol: str) -> Any:
    def _relation_func(T_e_avg: Any, n_e_avg: Any) -> Any:
        return radas_mean_charge(symbol, T_e_avg, n_e_avg)

    _relation_func.__name__ = f"radas_mean_charge_{symbol}"
    _relation_func.__doc__ = (
        f"Zbar_{symbol}(T_e_avg, n_e_avg) from the radas (OpenADAS) coronal "
        "mean-charge table -- the full 2-D dependence cfspopcon uses (log-spaced "
        "interpolation in both coordinates, edge-clamped).  The n_e dependence is "
        "weak over physical core densities (<=2.5% for Xe, ~0 for W)."
    )
    return _relation_func


for _sym in _RADAS_MEANCHARGE_SPECIES:
    relation(
        name=f"Mean ionisation state of {_sym} (radas coronal)",
        tags=("plasma", "composition"),
        outputs=f"Zbar_{_sym}",
    )(_make_radas_meancharge_relation(_sym))


# ── radas coronal mean-square charge <q^2>_X (the second charge moment) ────────
# Z_eff = sum_X c_X <q^2>_X reads these.  <q^2> = sum_q q^2 y_q from the radas
# coronal charge-state distribution -- the TRUE variance, not Zbar^2.  radas is
# the sole producer (no Mavrin <q^2>), so no whitelist is needed.


def _make_radas_meansq_relation(symbol: str) -> Any:
    def _relation_func(T_e_avg: Any, n_e_avg: Any) -> Any:
        return radas_mean_square_charge(symbol, T_e_avg, n_e_avg)

    _relation_func.__name__ = f"radas_mean_square_charge_{symbol}"
    _relation_func.__doc__ = (
        f"Zsq_{symbol}(T_e_avg, n_e_avg) = <q^2> from the radas coronal charge-"
        "state distribution (2-D log-spaced interpolation, edge-clamped)."
    )
    return _relation_func


for _sym in _RADAS_MEANSQ_SPECIES:
    relation(
        name=f"Mean-square charge of {_sym} (radas coronal)",
        tags=("plasma", "composition"),
        outputs=f"Zsq_{_sym}",
    )(_make_radas_meansq_relation(_sym))
