"""PROCESS core/edge radiation split and its transport-loss power.

Parallel forms of relations fusdb already has, wired the way PROCESS wires them,
so the two codes can be compared on a matched definition. All are GATED -- the
fusdb defaults are untouched for every other reactor.

PROCESS splits radiation at ``radius_plasma_core_norm`` and subtracts only the
CORE part from the power fed to the confinement scaling
(``i_rad_loss = 1``, ``CORE_ONLY``).  fusdb's own ``P_loss`` subtracts no
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
    here because PROCESS needs it resolved on the rho grid rather than already
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
    """Hydrogenic bremsstrahlung power density [W/m^3], used only as a SHAPE."""
    n_e20 = np.asarray(n_e, dtype=float) / 1e20
    return 5.35e-3 * (n_e20**2) * (np.asarray(T_e, dtype=float) ** 0.5) * 1e6


@relation(
    name="Core radiation power (PROCESS)",
    tags=("power_balance", "process"),
    outputs="P_rad_core",
)
def core_radiation_power_process(
    n_e: Any, T_e: Any, rho: Any, P_cool_imp: Any, P_brem: Any, P_sync: Any,
    radius_plasma_core_norm: Any = 0.75,
    f_p_plasma_core_rad_reduction: Any = 0.6,
    c_He: Any = 0.0, c_Li: Any = 0.0, c_Be: Any = 0.0, c_C: Any = 0.0, c_N: Any = 0.0,
    c_O: Any = 0.0, c_Ne: Any = 0.0, c_Ar: Any = 0.0, c_Kr: Any = 0.0, c_Xe: Any = 0.0,
    c_W: Any = 0.0,
) -> Any:
    """Radiated power from inside ``radius_plasma_core_norm`` [W].

    PROCESS: ``pden_plasma_core_rad_mw = pden_impurity_core_rad_total_mw +
    pden_plasma_sync_mw`` -- synchrotron is assumed to come entirely from the
    core, so it is added whole and is NOT scaled by the core reduction factor.

    Written in terms of the PRE-SPLIT radiation channels on purpose:

    * ``P_cool_imp`` -- the impurity cooling-rate total, which carries the Mavrin
      L_z profile shape; and
    * ``P_brem`` -- the hydrogenic bremsstrahlung, which carries the flatter
      n_e^2 sqrt(T_e) shape.

    Those are the two quantities PROCESS itself splits at the core radius (it
    folds hydrogenic radiation into its impurity array, which is why its
    ``p_plasma_rad_mw`` needs no separate bremsstrahlung term).  Going through the
    ``P_line``/``P_brem`` pair instead reassigns ~34 MW of impurity
    bremsstrahlung from the Mavrin-shaped term to the differently-shaped total,
    which moved ``P_rad_core`` +5.9% and ``P_sep`` past tolerance.

    ``P_brem`` is used DIRECTLY here.  It was ``P_brem - P_brem_imp`` while
    ``P_brem`` meant the TOTAL bremsstrahlung; now that the default ``P_brem`` is
    the hydrogenic (fuel-only) form -- the partner of a whole-L_z ``P_cool_imp``
    -- subtracting the impurity part again would drop it twice.  That mistake is
    worth 34.4 MW here, i.e. ``P_rad_core`` -18.7%.
    """
    # CHECK
    concentrations = {"He": c_He, "Li": c_Li, "Be": c_Be, "C": c_C, "N": c_N, "O": c_O,
                      "Ne": c_Ne, "Ar": c_Ar, "Kr": c_Kr, "Xe": c_Xe, "W": c_W}
    inside_core = (np.asarray(rho, dtype=float) <= radius_plasma_core_norm).astype(float)

    def core_fraction(density):
        total = volume_average(density, rho)
        if not np.all(np.isfinite(total)) or np.all(total == 0.0):
            return 0.0
        return volume_average(density * inside_core, rho) / total

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

    PROCESS keeps THREE distinct levels that fusdb's default collapses into two:

    * heating power        = f_alpha_dep*P_alpha + P_non_alpha + P_ohmic + P_inj
    * transport loss power = heating - P_rad_core   (fed to the tau_E scaling)
    * separatrix power     = heating - P_rad_total

    fusdb's default ``Plasma loss power`` is ``P_charged + P_aux``, i.e. it is
    really the HEATING power wearing the ``P_loss`` name.  This is emitted as a
    SEPARATE variable rather than as fusdb's ``P_in``: ``P_in`` has no default
    producer (it is inferred from the steady-state ``P_in = P_loss`` balance),
    so providing it here would silently become the sole provider on every
    reactor -- which broke four popcon/SPARC tests when first tried.

    # CHECK: the alpha deposition fraction is applied to the whole charged
    # fusion power rather than to the alpha channel alone.  Non-alpha charged
    # power is ~0.5% of P_charged at reactor conditions, so the error is ~0.03%
    # -- far below the comparison tolerance -- and it avoids depending on the
    # P_alpha_total aggregator being an active provider.
    """
    return f_p_alpha_plasma_deposited * P_charged + P_ohmic + P_aux


@relation(
    name="Plasma loss power (PROCESS)",
    tags=("power_balance", "process"),
    outputs="P_loss",
)
def plasma_loss_power_process(P_heating: Any, P_rad_core: Any) -> Any:
    """Transport loss power fed to the confinement scaling [W].

    PROCESS (``confinement_time.py`` 154-181, ``i_rad_loss = CORE_ONLY``)::

        P_loss = f_alpha_dep * P_alpha + P_non_alpha_charged + P_ohmic
                 + P_inj  (non-ignited)
                 - P_rad_core

    Only the CORE radiation is subtracted, and only from the confinement-scaling
    power -- ``P_sep`` keeps subtracting the total radiation from ``P_in``.
    """
    return P_heating - P_rad_core


@relation(
    name="Power crossing the separatrix (PROCESS)",
    tags=("power_balance", "process"),
    outputs="P_sep",
)
def separatrix_power_process(P_heating: Any, P_rad: Any) -> Any:
    """Power crossing the separatrix [W], PROCESS convention.

    The TOTAL radiation comes off the HEATING power -- not off the transport
    loss power, which has already had the core radiation removed.  Taking it off
    the loss power subtracts the core radiation twice.
    """
    return P_heating - P_rad
