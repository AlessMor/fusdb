"""Reduced levitated-dipole geometry relations.

Adapted from Wang et al. (2026), arXiv:2607.11208 ("VSC" reduced multi-configuration model).
VSC section 3.4.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fusdb.numerics import volume_average
from fusdb.relation import relation
from fusdb.registry import MU0


@relation(name="Point-dipole shell coordinate", tags=("dipole", "geometry", "default"), outputs="L_shell", dependency="generated_profile")
def point_dipole_shell_coordinate(L_in: Any, L_out: Any, *, rho: Any) -> Any:
    x = np.asarray(rho, dtype=float)
    lo = np.asarray(L_in, dtype=float)
    hi = np.asarray(L_out, dtype=float)
    if lo.ndim > 0 and lo.shape[-1] != 1:
        lo, hi = lo[..., None], hi[..., None]
    return lo + x * (hi - lo)


@relation(name="Point-dipole plasma volume", tags=("dipole", "geometry"), outputs="V_p")
def point_dipole_plasma_volume(L_in: Any, L_out: Any) -> Any:
    """Shell volume from VSC Eq. (79)."""
    coefficient = 64.0 * np.pi / 105.0
    return coefficient * (np.asarray(L_out) ** 3 - np.asarray(L_in) ** 3)


@relation(name="Point-dipole normalized enclosed volume", tags=("dipole", "geometry", "default"), outputs="v_norm", dependency="generated_profile")
def point_dipole_normalized_enclosed_volume(L_in: Any, L_out: Any, *, rho: Any) -> Any:
    x = np.asarray(rho, dtype=float)
    lo, hi = np.asarray(L_in, dtype=float), np.asarray(L_out, dtype=float)
    if lo.ndim > 0 and lo.shape[-1] != 1:
        lo, hi = lo[..., None], hi[..., None]
    shell = lo + x * (hi - lo)
    return (shell**3 - lo**3) / (hi**3 - lo**3)


@relation(name="Point-dipole volume integration weight", tags=("dipole", "geometry", "default"), outputs="w_V", dependency="generated_profile")
def point_dipole_volume_integration_weight(L_in: Any, L_out: Any, *, rho: Any) -> Any:
    """Weight proportional to dV/drho for V(<L>) proportional to L^3."""
    x = np.asarray(rho, dtype=float)
    lo, hi = np.asarray(L_in, dtype=float), np.asarray(L_out, dtype=float)
    if lo.ndim > 0 and lo.shape[-1] != 1:
        lo, hi = lo[..., None], hi[..., None]
    shell = lo + x * (hi - lo)
    return shell**2


@relation(name="Finite dipole ring current", tags=("dipole", "geometry"), outputs="I_ring")
def finite_dipole_ring_current(B_ring: Any, r_ring: Any) -> Any:
    """VSC Eq. (81): mu0 I_ring = 4 r_ring B_ring."""
    return 4.0 * np.asarray(r_ring) * np.asarray(B_ring) / MU0


@relation(name="Point-dipole equatorial field profile", tags=("dipole", "geometry"), outputs="B")
def point_dipole_equatorial_field(B_ring: Any, r_ring: Any, L_shell: Any) -> Any:
    """Point-dipole B proportional to L^-3, normalized by the supplied reference field."""
    b, r = np.asarray(B_ring, dtype=float), np.asarray(r_ring, dtype=float)
    if b.ndim and b.shape[-1] != np.asarray(L_shell).shape[-1]:
        b, r = b[..., None], r[..., None]
    return b * (r / np.asarray(L_shell)) ** 3


@relation(name="Point-dipole U ratio", tags=("dipole", "geometry"), outputs="U_ratio")
def point_dipole_u_ratio(L_in: Any, L_out: Any) -> Any:
    """Point-dipole U proportional to L^4 (VSC discussion following Eq. 82)."""
    return (np.asarray(L_out) / np.asarray(L_in)) ** 4


@relation(name="Point-dipole normalized U coordinate", tags=("dipole", "geometry", "default"), outputs="rho_U", dependency="generated_profile")
def point_dipole_normalized_u_coordinate(L_in: Any, L_out: Any, *, rho: Any) -> Any:
    """VSC Eq. (85); U proportional to L^4 makes its unknown normalization cancel."""
    x = np.asarray(rho, dtype=float)
    lo, hi = np.asarray(L_in, dtype=float), np.asarray(L_out, dtype=float)
    if lo.ndim > 0 and lo.shape[-1] != 1:
        lo, hi = lo[..., None], hi[..., None]
    shell = lo + x * (hi - lo)
    return np.log(shell / lo) / np.log(hi / lo)


@relation(name="Point-dipole specific-volume profile", tags=("dipole", "geometry"), outputs="U")
def point_dipole_specific_volume(B_ring: Any, r_ring: Any, L_shell: Any) -> Any:
    """Dimensional representative U retaining the exact VSC point-dipole L^4 scaling.

    Profile shapes and rho_U depend only on U ratios, so the arbitrary overall
    point-dipole normalization does not affect the 0-D power account.
    """
    b, r = np.asarray(B_ring, dtype=float), np.asarray(r_ring, dtype=float)
    if b.ndim and b.shape[-1] != np.asarray(L_shell).shape[-1]:
        b, r = b[..., None], r[..., None]
    return np.asarray(L_shell) ** 4 / (b * r**3)


@relation(name="Dipole spherical wall proxy", tags=("dipole", "geometry"), outputs="S_wall")
def dipole_spherical_wall_proxy(R_wall_proxy: Any) -> Any:
    return 4.0 * np.pi * np.asarray(R_wall_proxy) ** 2


@relation(name="Dipole B2.5 field moment", tags=("dipole", "geometry"), outputs="G_B25")
def dipole_b25_field_moment(B: Any, B_ring: Any, rho: Any, w_V: Any = None) -> Any:
    from fusdb.numerics import volume_average
    return volume_average(np.abs(np.asarray(B) / np.asarray(B_ring)) ** 2.5, rho, weight=w_V)
