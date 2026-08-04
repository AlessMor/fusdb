"""Energy confinement balance relations."""

from typing import Any

import numpy as np

from fusdb.relation import relation
from fusdb.registry import KEV_TO_J


@relation(
    name="Energy confinement balance",
    tags=("confinement", "power_balance", "plasma"),
)
def energy_confinement_balance(W_th: float, P_loss: float, tau_E: float) -> Any:
    """Return the normalized residual for ``W_th = P_loss * tau_E``.

    This is intentionally an outputless residual relation rather than
    ``tau_E = W_th / P_loss``.  That lets it be active at the same time as a
    confinement-scaling relation that also determines ``tau_E``.  In reconcile
    mode the two relations then jointly enforce consistency instead of one
    relation replacing or shadowing the other.

    Args:
        W_th: Thermal stored energy.
        P_loss: Plasma loss power.
        tau_E: Energy confinement time.

    Returns:
        Dimensionless residual, equal to zero when the balance is satisfied.
    """
    # Reject non-physical power and confinement values before forming the balance.
    # Per-point (scalar) evaluation raises as before; a batched array poisons
    # only its own bad rows with NaN, so one infeasible grid point cannot force
    # the whole popcon batch onto the point-by-point fallback.
    p_arr = np.asarray(P_loss, dtype=float)
    t_arr = np.asarray(tau_E, dtype=float)
    bad = ~np.isfinite(p_arr) | (p_arr <= 0.0) | ~np.isfinite(t_arr) | (t_arr <= 0.0)
    if np.any(bad):
        if p_arr.ndim == 0 and t_arr.ndim == 0:
            raise ValueError("P_loss and tau_E must be finite and positive")
    lhs = np.asarray(W_th, dtype=float)
    rhs = np.where(bad, np.nan, p_arr * t_arr)
    scale = np.maximum(np.maximum(np.abs(lhs), np.abs(rhs)), 1.0)
    return (lhs - rhs) / scale


@relation(
    name='Plasma stored energy (cfspopcon)',
    tags=('plasma', 'confinement'),
    outputs='W_th',
)
def plasma_stored_energy_cfspopcon(
    n_e_avg: float, T_e_avg: float, n_i_avg: float, T_i_avg: float, V_p: float
) -> Any:
    """Plasma thermal stored energy on cfspopcon's product-of-averages convention.

    Adapted from cfspopcon; see README.md section "Third-party Notices".

    cfspopcon's ``calc_plasma_stored_energy`` builds the stored energy that
    feeds its tau_E/P_in confinement solve from the *volume averages* alone:
    ``W = 3/2 (<n_e><T_e> + <n_i><T_i>) V_p`` -- it carries no
    profile-correlation term.  fusdb's default producers integrate the actual
    profiles (``3/2 <p_th>_V V_p`` with ``p_th = <n(rho) T(rho)>``), which runs
    higher when peaked density and temperature correlate on axis; the IPB98-type
    power balance amplifies that difference as ``P_in ~ W^3.2``, so matching
    cfspopcon's auxiliary power (and with it ``Q_cfspopcon``) requires this
    convention.  fusdb's ``n_i_avg`` counts every ion species, i.e. cfspopcon's
    ``average_ion_density + summed_impurity_density``.

    Args:
        n_e_avg: Volume-averaged electron density [m^-3]
        T_e_avg: Volume-averaged electron temperature [keV]
        n_i_avg: Volume-averaged ion density, all species [m^-3]
        T_i_avg: Volume-averaged ion temperature [keV]
        V_p: Plasma volume [m^3]

    Returns:
        W_th: Thermal stored energy [J]
    """
    return 1.5 * KEV_TO_J * (n_e_avg * T_e_avg + n_i_avg * T_i_avg) * V_p


@relation(
    name='Thermal stored energy',
    tags=('plasma',),
    outputs='W_th',
)
def thermal_stored_energy(p_th: float, V_p: float) -> float:
    """Return thermal stored energy from pressure and plasma volume.

    Args:
        p_th: Volume-averaged thermal pressure.
        V_p: Plasma volume.

    Returns:
        Thermal stored energy.
    """
    return 1.5 * p_th * V_p
