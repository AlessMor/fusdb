"""Energy confinement balance relations."""

from typing import Any

import numpy as np

from fusdb.numerics import volume_average
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
    name="Confinement time from scaling",
    tags=("confinement", "plasma"),
    outputs="tau_E",
)
def confinement_time_from_scaling(tau_E_scaling: Any, H_factor: Any) -> Any:
    """Achieved confinement time as the raw scaling fit times the H factor.

    Confinement scalings are published as a *raw* fit; what a device achieves is
    that fit times an enhancement factor H.  Keeping the two on separate
    variables -- ``tau_E_scaling`` for the fit, ``tau_E`` for the achievement --
    is what makes H a derived quantity: with ``tau_E`` also constrained by the
    ``W_th = P_loss * tau_E`` balance above, this relation reports the H the
    design point actually implies instead of consuming a declared one.

    ``H_factor`` defaults to 1.0 and carries every published scaling-specific
    name (``H98_y2``, ``H89_P``, ``H_<scaling>``, ``hfact``,
    ``confinement_time_scalar``) as an alias, so a reactor still declares the H
    its reference quotes.  A declared H is then an ordinary reconcile input:
    it seeds the solve and is reported as beyond-tolerance if the point cannot
    support it.

    Args:
        tau_E_scaling: Raw confinement-time scaling prediction.
        H_factor: Confinement enhancement factor.

    Returns:
        tau_E: Achieved energy confinement time.
    """
    return np.asarray(H_factor) * np.asarray(tau_E_scaling)


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


@relation(name="Thermal stored energy (VSC profile model)", tags=("plasma", "confinement"), outputs="W_th")
def thermal_stored_energy_vsc_profile(n_i_peak: Any, T_i_peak: Any, n0: Any, T0: Any, alphan: Any, alphat: Any, V_p: Any) -> Any:
    """Alternative W_th producer for VSC Eq. (13); FusDB's existing producer remains default.

    Adapted from Wang et al. (2026), arXiv:2607.11208 ("VSC" reduced multi-configuration model).
    """
    f_nT = 1.0 / (1.0 + np.asarray(alphan) + np.asarray(alphat))
    return 1.5 * KEV_TO_J * np.asarray(V_p) * (np.asarray(n_i_peak) * np.asarray(T_i_peak) + np.asarray(n0) * np.asarray(T0)) * f_nT


@relation(name="Electron thermal stored energy", tags=("plasma", "confinement"), outputs="W_e")
def electron_thermal_stored_energy(n_e: Any, T_e: Any, V_p: Any, rho: Any, w_V: Any = None) -> Any:
    """Electron thermal stored energy from the volume-averaged n_e T_e product.

    Adapted from Wang et al. (2026), arXiv:2607.11208 ("VSC" reduced multi-configuration model).
    """
    return 1.5 * KEV_TO_J * np.asarray(V_p) * volume_average(np.asarray(n_e) * np.asarray(T_e), rho, weight=w_V)


@relation(name="Ion thermal stored energy", tags=("plasma", "confinement"), outputs="W_i")
def ion_thermal_stored_energy(n_i: Any, T_i: Any, V_p: Any, rho: Any, w_V: Any = None) -> Any:
    """Ion thermal stored energy from the volume-averaged n_i T_i product.

    Adapted from Wang et al. (2026), arXiv:2607.11208 ("VSC" reduced multi-configuration model).
    """
    return 1.5 * KEV_TO_J * np.asarray(V_p) * volume_average(np.asarray(n_i) * np.asarray(T_i), rho, weight=w_V)


@relation(name="Dipole density-confinement product", tags=("dipole", "confinement"), outputs="n0_tau_E")
def dipole_density_confinement_product(n0: Any, tau_E: Any) -> Any:
    """Levitated-dipole central density-confinement product n0 tau_E.

    Adapted from Wang et al. (2026), arXiv:2607.11208 ("VSC" reduced multi-configuration model).
    """
    return np.asarray(n0) * np.asarray(tau_E)
