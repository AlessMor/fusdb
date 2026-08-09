"""Reduced non-thermal ion pressure and beta relations.

PROCESS supplies the standalone fast-alpha and neutral-beam beta fits. FUSE
stores thermal and fast-ion pressures separately and forms total pressure by
adding the fast components; that aggregation is represented explicitly here.
cfspopcon has no non-thermal-ion pressure model, while bluemira names the
underlying PROCESS fast-alpha fits through ``AlphaPressureModel``.

Adapted from PROCESS, bluemira, and FUSE; see README.md section
"Third-party Notices".
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fusdb.relation import relation
from fusdb.registry import KEV_TO_J, MU0
from fusdb.utils import volume_average


@relation(
    name="Density-weighted electron temperature",
    tags=("plasma", "profile", "process"),
    outputs="temp_plasma_electron_density_weighted",
)
def density_weighted_electron_temperature(n_e: Any, T_e: Any, rho: Any) -> Any:
    """Return ``<n_e T_e>/<n_e>`` on the fusdb volume-average convention."""
    density = np.asarray(n_e, dtype=float)
    numerator = volume_average(density * np.asarray(T_e, dtype=float), rho)
    denominator = volume_average(density, rho)
    return numerator / np.maximum(denominator, 1e-300)


@relation(
    name="Density-weighted ion temperature",
    tags=("plasma", "profile", "process"),
    outputs="temp_plasma_ion_density_weighted",
)
def density_weighted_ion_temperature(n_i: Any, T_i: Any, rho: Any) -> Any:
    """Return ``<n_i T_i>/<n_i>`` for the PROCESS fast-alpha fit."""
    density = np.asarray(n_i, dtype=float)
    numerator = volume_average(density * np.asarray(T_i, dtype=float), rho)
    denominator = volume_average(density, rho)
    return numerator / np.maximum(denominator, 1e-300)


def _fast_alpha_beta_process(
    B_total: Any,
    n_e_avg: Any,
    n_fuel_avg: Any,
    n_i_avg: Any,
    temp_plasma_electron_density_weighted: Any,
    temp_plasma_ion_density_weighted: Any,
    P_fus_DT_alpha: Any,
    P_alpha_beam: Any,
    f_D: Any,
    fraction: Any,
) -> Any:
    """Apply a named PROCESS fast-alpha fraction to the thermal beta."""
    field = np.asarray(B_total, dtype=float)
    ne = np.asarray(n_e_avg, dtype=float)
    ni = np.asarray(n_i_avg, dtype=float)
    te = np.asarray(temp_plasma_electron_density_weighted, dtype=float)
    ti = np.asarray(temp_plasma_ion_density_weighted, dtype=float)

    beta_thermal = 2.0 * MU0 * KEV_TO_J * (ne * te + ni * ti) / field**2
    fraction = np.maximum(np.asarray(fraction, dtype=float), 0.0)

    plasma_alpha = np.asarray(P_fus_DT_alpha, dtype=float)
    beam_alpha = np.asarray(P_alpha_beam, dtype=float)
    alpha_power_factor = np.where(
        plasma_alpha > 0.0,
        (plasma_alpha + beam_alpha) / np.maximum(plasma_alpha, 1e-300),
        0.0,
    )
    produces_dt_alphas = np.asarray(f_D, dtype=float) < 1.0
    return np.where(produces_dt_alphas, beta_thermal * fraction * alpha_power_factor, 0.0)


@relation(
    name="Fast-alpha beta Ward (PROCESS)",
    tags=("plasma", "fusion_power", "process"),
    outputs="beta_fast_alpha",
)
def fast_alpha_beta_ward_process(
    B_total: Any,
    n_e_avg: Any,
    n_fuel_avg: Any,
    n_i_avg: Any,
    temp_plasma_electron_density_weighted: Any,
    temp_plasma_ion_density_weighted: Any,
    P_fus_DT_alpha: Any,
    P_alpha_beam: Any,
    f_D: Any,
) -> Any:
    """PROCESS's modified Ward fast-alpha beta fit (framework default).

    This is a named relation, not a PROCESS integer switch. To reproduce the
    IPDG89/Hender fit, include :func:`fast_alpha_beta_ipdg89_process` and
    exclude this default relation in the reactor configuration. bluemira's
    ``AlphaPressureModel.WARD`` confirms the source-model naming.

    Adapted from PROCESS; model name cross-checked against bluemira. See
    README.md section "Third-party Notices".
    """
    dilution_sq = (np.asarray(n_fuel_avg, dtype=float) / np.maximum(np.asarray(n_e_avg, dtype=float), 1e-300)) ** 2
    temperature_sum = (
        np.asarray(temp_plasma_electron_density_weighted, dtype=float)
        + np.asarray(temp_plasma_ion_density_weighted, dtype=float)
    )
    fraction = np.minimum(
        0.30,
        0.26 * dilution_sq * np.sqrt(np.maximum(temperature_sum / 20.0 - 0.65, 0.0)),
    )
    return _fast_alpha_beta_process(
        B_total, n_e_avg, n_fuel_avg, n_i_avg,
        temp_plasma_electron_density_weighted, temp_plasma_ion_density_weighted,
        P_fus_DT_alpha, P_alpha_beam, f_D, fraction,
    )


@relation(
    name="Fast-alpha beta IPDG89 (PROCESS)",
    tags=("plasma", "fusion_power", "process"),
    outputs="beta_fast_alpha",
)
def fast_alpha_beta_ipdg89_process(
    B_total: Any,
    n_e_avg: Any,
    n_fuel_avg: Any,
    n_i_avg: Any,
    temp_plasma_electron_density_weighted: Any,
    temp_plasma_ion_density_weighted: Any,
    P_fus_DT_alpha: Any,
    P_alpha_beam: Any,
    f_D: Any,
) -> Any:
    """PROCESS IPDG89/Hender fast-alpha beta fit.

    Select this named alternative when reproducing an IPDG89/Hender PROCESS
    case; the default relation is the separately named Ward fit.

    Adapted from PROCESS; model name cross-checked against bluemira. See
    README.md section "Third-party Notices".
    """
    dilution_sq = (np.asarray(n_fuel_avg, dtype=float) / np.maximum(np.asarray(n_e_avg, dtype=float), 1e-300)) ** 2
    temperature_sum = (
        np.asarray(temp_plasma_electron_density_weighted, dtype=float)
        + np.asarray(temp_plasma_ion_density_weighted, dtype=float)
    )
    fraction = np.minimum(0.30, 0.29 * dilution_sq * (temperature_sum / 20.0 - 0.37))
    return _fast_alpha_beta_process(
        B_total, n_e_avg, n_fuel_avg, n_i_avg,
        temp_plasma_electron_density_weighted, temp_plasma_ion_density_weighted,
        P_fus_DT_alpha, P_alpha_beam, f_D, fraction,
    )


@relation(
    name="Neutral-beam beta (PROCESS)",
    tags=("plasma", "current_drive", "process"),
    outputs="beta_beam",
)
def neutral_beam_beta_process(
    n_beam_hot: Any,
    e_beam_deposited: Any,
    B_total: Any,
    beta_beam_multiplier: Any,
) -> Any:
    """PROCESS neutral-beam beta from hot-ion density and deposited energy."""
    return (
        np.asarray(beta_beam_multiplier, dtype=float)
        * 4.03e-22
        * (2.0 / 3.0)
        * np.asarray(n_beam_hot, dtype=float)
        * np.asarray(e_beam_deposited, dtype=float)
        / np.asarray(B_total, dtype=float) ** 2
    )


@relation(
    name="Fast-alpha pressure from beta",
    tags=("plasma", "fusion_power"),
    outputs="p_fast_alpha",
)
def fast_alpha_pressure_from_beta(beta_fast_alpha: Any, B_total: Any) -> Any:
    """Convert the PROCESS beta component to its equivalent average pressure."""
    return np.asarray(beta_fast_alpha, dtype=float) * np.asarray(B_total, dtype=float) ** 2 / (2.0 * MU0)


@relation(
    name="Neutral-beam fast-ion pressure from beta",
    tags=("plasma", "current_drive"),
    outputs="p_fast_beam",
)
def beam_fast_ion_pressure_from_beta(beta_beam: Any, B_total: Any) -> Any:
    """Convert neutral-beam beta to its equivalent average fast-ion pressure."""
    return np.asarray(beta_beam, dtype=float) * np.asarray(B_total, dtype=float) ** 2 / (2.0 * MU0)


@relation(
    name="Total pressure including fast ions (FUSE)",
    tags=("plasma",),
    outputs="p_with_fast_ions",
)
def total_pressure_including_fast_ions_fuse(
    p_th: Any,
    p_fast_alpha: Any,
    p_fast_beam: Any,
) -> Any:
    """FUSE-style total pressure: thermal plus all ion ``pressure_fast`` components.

    Adapted from FUSE; see README.md section "Third-party Notices".
    """
    return (
        np.asarray(p_th, dtype=float)
        + np.asarray(p_fast_alpha, dtype=float)
        + np.asarray(p_fast_beam, dtype=float)
    )


@relation(
    name="Total beta including fast ions",
    tags=("plasma",),
    outputs="beta_with_fast_ions",
)
def total_beta_including_fast_ions(beta: Any, beta_fast_alpha: Any, beta_beam: Any) -> Any:
    """Add the PROCESS non-thermal beta components to fusdb's thermal beta."""
    return (
        np.asarray(beta, dtype=float)
        + np.asarray(beta_fast_alpha, dtype=float)
        + np.asarray(beta_beam, dtype=float)
    )


@relation(
    name="Normalized total beta including fast ions",
    tags=("plasma", "tokamak"),
    outputs="beta_N_with_fast_ions",
)
def normalized_total_beta_including_fast_ions(
    beta_with_fast_ions: Any,
    a: Any,
    B0: Any,
    I_p: Any,
) -> Any:
    """Troyon normalization of thermal plus non-thermal beta."""
    return (
        np.asarray(beta_with_fast_ions, dtype=float)
        * np.asarray(a, dtype=float)
        * np.asarray(B0, dtype=float)
        / (np.asarray(I_p, dtype=float) / 1.0e6)
    )
