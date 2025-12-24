"""Device-agnostic plasma relations (pressure, energy, beta)."""

import math

from fusiondb.relations_util import require_nonzero
from fusiondb.relations_values import PRIORITY_RELATION, Relation

KEV_TO_J = 1.602176634e-16  # conversion factor for temperatures stored in keV
MU0 = 4 * math.pi * 1e-7

# 1.1 Thermal pressure: p_th = (ne Te + ni Ti) * keV_to_J
THERMAL_PRESSURE_REL = Relation(
    "Thermal pressure",
    ("p_th", "n_e", "T_e", "n_i", "T_i"),
    lambda v: v["p_th"] - (v["n_e"] * v["T_e"] + v["n_i"] * v["T_i"]) * KEV_TO_J,
    priority=PRIORITY_RELATION,
)

# 1.2 Thermal stored energy: W_th = 1.5 * p_th * V_p
THERMAL_ENERGY_REL = Relation(
    "Thermal stored energy",
    ("W_th", "p_th", "V_p"),
    lambda v: v["W_th"] - 1.5 * v["p_th"] * v["V_p"],
    priority=PRIORITY_RELATION,
)

# 1.3 Energy confinement: tau_E = W_th / P_loss
CONFINEMENT_REL = Relation(
    "Energy confinement time",
    ("tau_E", "W_th", "P_loss"),
    lambda v: require_nonzero(v["P_loss"], "P_loss", "confinement relation") or v["tau_E"] - v["W_th"] / v["P_loss"],
    priority=PRIORITY_RELATION,
)

# 1.4 Beta definitions
TOROIDAL_BETA_REL = Relation(
    "Toroidal beta",
    ("beta_T", "p_th", "B0"),
    lambda v: require_nonzero(v["B0"], "B0", "toroidal beta") or v["beta_T"] - (2 * MU0 * v["p_th"]) / (v["B0"] ** 2),
    priority=PRIORITY_RELATION,
)

POLOIDAL_BETA_REL = Relation(
    "Poloidal beta",
    ("beta_p", "p_th", "B_p"),
    lambda v: require_nonzero(v["B_p"], "B_p", "poloidal beta") or v["beta_p"] - (2 * MU0 * v["p_th"]) / (v["B_p"] ** 2),
    priority=PRIORITY_RELATION,
)

BETA_SUM_REL = Relation(
    "Beta decomposition",
    ("beta", "beta_T", "beta_p"),
    lambda v: (
        require_nonzero(v["beta"], "beta", "beta decomposition")
        or require_nonzero(v["beta_T"], "beta_T", "beta decomposition")
        or require_nonzero(v["beta_p"], "beta_p", "beta decomposition")
        or (1 / v["beta"] - 1 / v["beta_T"] - 1 / v["beta_p"])
    ),
    priority=PRIORITY_RELATION,
)

PLASMA_RELATIONS: tuple[Relation, ...] = (
    THERMAL_PRESSURE_REL,
    THERMAL_ENERGY_REL,
    CONFINEMENT_REL,
    TOROIDAL_BETA_REL,
    POLOIDAL_BETA_REL,
    BETA_SUM_REL,
)

__all__ = [
    "KEV_TO_J",
    "MU0",
    "PLASMA_RELATIONS",
    "THERMAL_PRESSURE_REL",
    "THERMAL_ENERGY_REL",
    "CONFINEMENT_REL",
    "TOROIDAL_BETA_REL",
    "POLOIDAL_BETA_REL",
    "BETA_SUM_REL",
]
