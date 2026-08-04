"""Separatrix shape relations."""

from typing import Any

from fusdb.relation import relation


def calc_separatrix_elongation_from_areal_elongation(areal_elongation, elongation_ratio_sep_to_areal):
    """cfspopcon: separatrix_elongation = areal_elongation * elongation_ratio_sep_to_areal."""
    return areal_elongation * elongation_ratio_sep_to_areal


def calc_separatrix_triangularity_from_triangularity95(triangularity_psi95, triangularity_ratio_sep_to_psi95):
    """cfspopcon: separatrix_triangularity = triangularity_psi95 * triangularity_ratio_sep_to_psi95."""
    return triangularity_psi95 * triangularity_ratio_sep_to_psi95


def calc_vertical_minor_radius_from_elongation_and_minor_radius(minor_radius, separatrix_elongation):
    """cfspopcon: vertical_minor_radius = minor_radius * separatrix_elongation."""
    return minor_radius * separatrix_elongation


@relation(
    name="Elongation at psi95 from areal elongation (cfspopcon)",
    tags=("geometry", "tokamak"),
    outputs="kappa_95",
)
def calc_elongation_at_psi95_from_areal_elongation(
    areal_elongation: Any, elongation_ratio_areal_to_psi95: Any
) -> Any:
    """95%-flux-surface elongation from the areal elongation and cfspopcon's ratio.

    Adapted from cfspopcon; see README.md section "Third-party Notices".
    ``kappa_95 = areal_elongation / elongation_ratio_areal_to_psi95``.  fusdb's
    default "Elongation 95%" uses the fixed IPDG89 ratio (kappa/1.12); cfspopcon
    carries the ratio as an input (1.025 for SPARC), giving a larger kappa_95
    that flows through the identical q_cyl -> internal_inductivity ->
    internal_inductance chain to reproduce the reference's internal inductance
    and CS-flux consumption (and hence the flat-top duration).  Gated (fusdb's
    IPDG89 form stays the default).
    """
    # CHECK
    return areal_elongation / elongation_ratio_areal_to_psi95


@relation(
    name="Separatrix elongation from geometric elongation",
    tags=("geometry",),
    outputs="kappa_separatrix",
)
def calc_separatrix_elongation_from_geometric(
    kappa: Any, elongation_ratio_sep_to_geom: Any
) -> Any:
    """``kappa_separatrix = kappa * elongation_ratio_sep_to_geom``.

    ``kappa`` is the GEOMETRIC elongation of the smooth shape parameterisation;
    ``kappa_separatrix`` is the elongation of the REAL last closed flux surface.
    They differ whenever the plasma is diverted, because the X-point extension
    raises Z_max without adding cross-sectional area -- so the two are tied by a
    ratio rather than equated.  The ratio defaults to 1.0 (no X-point
    extension), which is a penalized input: a device that declares both
    elongations DETERMINES the ratio instead of fighting it, which a bare
    identity ``kappa_separatrix = kappa`` could not do.

    Its purpose is to stop ``kappa_separatrix`` being a packed, un-anchored,
    single-relation variable -- exactly the state that let it settle at
    0.24 / 0.11 on reactors that never declared it, tuning ``P_sync`` to absorb
    an unrelated residual.
    """
    return kappa * elongation_ratio_sep_to_geom
