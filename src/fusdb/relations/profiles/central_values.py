"""On-axis (central) plasma values.

Cherry-picked standalone physics formulas from PROCESS
``process/models/physics/plasma_profiles.py`` (the ``PlasmaProfile``
orchestration class). Only the self-contained, parameterisation-agnostic
algebraic relations are ported; PROCESS's profile-shape generators, pedestal
HELIOS ``ncore``/``tcore`` core-value solvers, alpha-index peak-value formulas
(``n0 = <n>(1+alphan)`` etc.), density-weighted temperatures and stellarator
gradient lengths are NOT ported -- fusdb parameterises profiles by peaking
factors (see ``profiles/peaking.py``) rather than by profile indices, and
already produces ``n0``/``T0``/``n_i_peak``/``T_i_peak``/``n_la``.
"""

from scipy.special import beta as _beta_fn

from fusdb.relation import relation
from fusdb.registry import KEV_TO_J


# NOTE: PROCESS's ``alphap = alphan + alphat`` definition is intentionally NOT
# ported as a relation. Coupling the three defaulted profile-index variables
# destabilises the pinned-reconcile (popcon) solve, and its only consumer
# (the gated Connor-Hastie shaping function) is off by default. ``alphap``
# stays a pure input, like ``alphan``/``alphat``.


@relation(
    name="Central plasma pressure from ideal gas law",
    tags=("plasma", "profile", "process"),
    outputs="pres_plasma_on_axis",
)
def central_plasma_pressure(n0: float, T0: float, n_i_peak: float, T_i_peak: float) -> float:
    """Central (on-axis) thermal plasma pressure from the ideal gas law
    p0 = (n_e0 T_e0 + n_i0 T_i0) k.

    Adapted from PROCESS; see README.md section "Third-party Notices".

    fusdb supplies the on-axis electron/ion densities and temperatures from its
    peaking-factor profiles; the ideal-gas central pressure is independent of
    how those peak values were obtained. Temperatures are in keV, converted to
    joules via ``KEV_TO_J`` (PROCESS ``KILOELECTRON_VOLT``).
    """
    # CHECK
    return (n0 * T0 + n_i_peak * T_i_peak) * KEV_TO_J


@relation(
    name="Central plasma current density",
    tags=("plasma", "profile", "tokamak", "process"),
    outputs="j_plasma_on_axis",
)
def central_current_density(I_p: float, alphaj: float, S_phi: float) -> float:
    """Central (on-axis) plasma current density, assuming a parabolic current
    profile j(rho) = j0 (1 - rho^2)^alphaj integrated over the poloidal
    cross-section.

    Adapted from PROCESS; see README.md section "Third-party Notices".

    ``S_phi`` is the plasma poloidal cross-sectional area (PROCESS
    ``a_plasma_poloidal``).
    """
    # CHECK
    return I_p * 2.0 / (_beta_fn(0.5, alphaj + 1.0) * S_phi)
