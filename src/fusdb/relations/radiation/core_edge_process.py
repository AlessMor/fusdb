"""PROCESS core/edge radiation split and its transport-loss power.

Parallel forms of relations fusdb already has, wired the way PROCESS wires them,
so the two codes can be compared on a matched definition. All are GATED -- the
fusdb defaults are untouched for every other reactor.

PROCESS splits radiation at ``radius_plasma_core_norm`` and subtracts only the
CORE part from the power fed to the confinement scaling
(``i_rad_loss = 1``, ``CORE_ONLY``). fusdb's own ``P_loss`` subtracts no
radiation at all, so at the large-tokamak design point the two differ by ~60%
in ``P_loss`` and ~42% in ``tau_E``.

Adapted from PROCESS; see README.md section "Third-party Notices".
Ported from ``process/models/physics/radiation_power.py`` (the core/edge split)
and ``process/models/physics/confinement_time.py`` lines 154-181 (the loss
power assembly).
"""

from typing import Any

import numpy as np

from fusdb.relation import relation
from fusdb.utils import volume_average

from .impurity_radiation import _MAVRIN_T_MAX, _MAVRIN_T_MIN, _binned_log10_Lz, _load_radiation_dataset


def _impurity_rad_density(n_e, T_e, concentrations):
    """Impurity radiated power density [W/m^3], Mavrin 2018 coronal.

    Same integrand as ``Impurity line radiation (Mavrin coronal)``; factored out
    here because PROCESS needs it resolved on the radial grid rather than already
    volume-integrated, and fusdb's own relation returns only the scalar.
    """
    Te = np.clip(np.asarray(T_e, dtype=float), _MAVRIN_T_MIN, _MAVRIN_T_MAX)
    c_times_Lz = np.zeros_like(Te)
    for symbol, concentration in concentrations.items():
        if float(concentration) == 0.0:
            continue
        entry = _load_radiation_dataset("polynomialfit", "mavrin_coronal", symbol)
        Lz = 10.0 ** _binned_log10_Lz(Te, entry["temperature_bin_borders"], entry["radc"])
        c_times_Lz = c_times_Lz + concentration * Lz
    return np.nan_to_num(np.asarray(n_e, dtype=float) ** 2 * c_times_Lz, nan=0.0)


def _hydrogenic_brem_density(n_e, T_e):
    """Hydrogenic bremsstrahlung power density [W/m^3], used only as a shape."""
    n_e20 = np.asarray(n_e, dtype=float) / 1e20
    return 5.35e-3 * (n_e20**2) * (np.asarray(T_e, dtype=float) ** 0.5) * 1e6


@relation(
    name="Core radiation power (PROCESS)",
    tags=("power_balance", "process"),
    outputs="P_rad_core",
)
def core_radiation_power_process(
    n_e: Any, T_e: Any, rho_minor: Any, w_V: Any, P_cool_imp: Any, P_brem: Any, P_sync: Any,
    radius_plasma_core_norm: Any = 0.75,
    f_p_plasma_core_rad_reduction: Any = 0.6,
    c_He: Any = 0.0, c_Li: Any = 0.0, c_Be: Any = 0.0, c_C: Any = 0.0, c_N: Any = 0.0,
    c_O: Any = 0.0, c_Ne: Any = 0.0, c_Ar: Any = 0.0, c_Kr: Any = 0.0, c_Xe: Any = 0.0,
    c_W: Any = 0.0, rho: Any = None,
) -> Any:
    """Radiated power from inside ``radius_plasma_core_norm`` [W].

    PROCESS defines ``radius_plasma_core_norm`` on normalized physical minor
    radius, so the core mask uses ``rho_minor`` explicitly. The fractional power
    integral is instead evaluated on fusdb's common computational ``rho`` grid
    with the geometry-provided volume measure ``w_V``. This keeps the source
    model's radial definition separate from the device's volume integration.

    PROCESS: ``pden_plasma_core_rad_mw = pden_impurity_core_rad_total_mw +
    pden_plasma_sync_mw`` -- synchrotron is assumed to come entirely from the
    core, so it is added whole and is NOT scaled by the core reduction factor.

    Written in terms of the pre-split radiation channels on purpose:
    ``P_cool_imp`` carries the Mavrin L_z profile shape and ``P_brem`` carries
    the hydrogenic n_e^2 sqrt(T_e) shape. ``P_brem`` is used directly because
    fusdb's default ``P_brem`` is the hydrogenic (fuel-only) partner of the
    whole-L_z ``P_cool_imp``.
    """
    # CHECK
    rho_grid = np.asarray(rho if rho is not None else rho_minor, dtype=float)
    minor = np.asarray(rho_minor, dtype=float)
    weight = np.asarray(w_V, dtype=float)
    concentrations = {"He": c_He, "Li": c_Li, "Be": c_Be, "C": c_C, "N": c_N, "O": c_O,
                      "Ne": c_Ne, "Ar": c_Ar, "Kr": c_Kr, "Xe": c_Xe, "W": c_W}
    inside_core = (minor <= radius_plasma_core_norm).astype(float)

    def core_fraction(density):
        total = volume_average(density, rho_grid, weight=weight)
        if not np.all(np.isfinite(total)) or np.all(total == 0.0):
            return 0.0
        return volume_average(density * inside_core, rho_grid, weight=weight) / total

    f_cool = core_fraction(_impurity_rad_density(n_e, T_e, concentrations))
    f_hyd = core_fraction(_hydrogenic_brem_density(n_e, T_e))
    impurity_core = P_cool_imp * f_cool + P_brem * f_hyd
    return f_p_plasma_core_rad_reduction * impurity_core + P_sync


@relation(
    name="Edge radiation power (PROCESS)",
    tags=("power_balance", "process"),
    outputs="P_rad_edge",
)
def edge_radiation_power_process(P_rad: Any, P_rad_core: Any) -> Any:
    """Radiation from outside the core region [W] (PROCESS ``p_plasma_outer_rad_mw``)."""
    return P_rad - P_rad_core


@relation(
    name="Plasma heating power (PROCESS)",
    tags=("power_balance", "process"),
    outputs="P_heating",
)
def plasma_heating_power_process(
    P_charged: Any, P_aux: Any, P_ohmic: Any,
    f_p_alpha_plasma_deposited: Any = 0.95,
) -> Any:
    """Total power heating the plasma [W], before any radiation subtraction.

    PROCESS keeps three distinct levels that fusdb's default collapses into two:
    heating power, transport loss power (heating - P_rad_core), and separatrix
    power (heating - P_rad_total).

    The alpha deposition fraction is applied to the whole charged fusion power
    here rather than to the alpha channel alone. Non-alpha charged power is
    ~0.5% of P_charged at reactor conditions, so the difference is ~0.03% and
    avoids depending on the P_alpha_total aggregator being active.
    """
    return f_p_alpha_plasma_deposited * P_charged + P_ohmic + P_aux


@relation(
    name="Plasma loss power (PROCESS)",
    tags=("power_balance", "process"),
    outputs="P_loss",
)
def plasma_loss_power_process(P_heating: Any, P_rad_core: Any) -> Any:
    """Transport loss power fed to the confinement scaling [W]."""
    return P_heating - P_rad_core


@relation(
    name="Power crossing the separatrix (PROCESS)",
    tags=("power_balance", "process"),
    outputs="P_sep",
)
def separatrix_power_process(P_heating: Any, P_rad: Any) -> Any:
    """Power crossing the separatrix [W], PROCESS convention."""
    return P_heating - P_rad
