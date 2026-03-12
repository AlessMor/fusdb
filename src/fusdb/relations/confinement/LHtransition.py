"""L-H transition relations."""

from __future__ import annotations

from fusdb.relation_util import relation

# TODO: add more L-H transition relations

@relation(
    name="L-H transition threshold power",
    output="P_LH",
    tags=("confinement", "hmode"),
    constraints=("A_p > 0", "B0 > 0", "n_avg >= 0"),
    # Provide explicit inverses to allow backward solves without expensive sympy.
    inverse_functions={
        "P_LH": lambda values: values["P_LH"],
        "n_avg": lambda values: 1e20 * (values["P_LH"] / (1e6 * 0.0488 * (values["B0"] ** 0.803) * (values["A_p"] ** 0.941)))** (1.0 / 0.717),
        "B0": lambda values: (values["P_LH"] / (1e6 * 0.0488 * ((values["n_avg"] / 1e20) ** 0.717) * (values["A_p"] ** 0.941)))** (1.0 / 0.803),
        "A_p": lambda values: (values["P_LH"] / (1e6 * 0.0488 * ((values["n_avg"] / 1e20) ** 0.717) * (values["B0"] ** 0.803)))** (1.0 / 0.941),
    },
)
def lh_transition_power(n_avg: float, B0: float, A_p: float) -> float:
    """Return the L-H transition threshold power using a Martin-2008 style scaling.

    Args:
        n_avg: Line-averaged density [1/m^3].
        B0: Toroidal magnetic field [T].
        A_p: Plasma surface area [m^2].

    Returns:
        L-H transition threshold power [W].
    """
    n20 = n_avg / 1e20
    # P_LH [MW] = 0.0488 * n20^0.717 * B0^0.803 * A_p^0.941
    return 1e6 * 0.0488 * (n20 ** 0.717) * (B0 ** 0.803) * (A_p ** 0.941)
