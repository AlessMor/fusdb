"""Profile peaking and parabolic profile relations."""

from functools import lru_cache
from typing import Any

import numpy as np

from fusdb.relation import relation
from fusdb.utils import volume_average


# Small pedestal so a generated profile never reaches exactly zero at rho=1.
# A real plasma has finite separatrix values; the exact zero of (1-rho^2)^alpha
# makes temperature-driven relations (fusion reactivity, line/synchrotron
# radiation) singular at the edge and stiffens the confinement solve.
_EDGE_PEDESTAL = 0.02


# From plasma_profiles/density_peaking.py
# TODO(low): from cfspopcon add
    # density peaking for tokamaks Equation 3 from p1334 of Angioni et al.
    # different forms are used

# From plasma_profiles/temperature_peaking.py
# TODO(low): add from cfspopcon


@lru_cache(maxsize=16)
def _peaking_table(rho_key: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Return a monotone ``(peakings, alphas)`` table for one rho grid.

    ``peak/volume_average`` of ``(1-rho^2)^alpha`` is strictly increasing in
    alpha, so one precomputed table per grid inverts the peaking->alpha map in
    O(log n) by interpolation instead of an 80-step bisection on every profile
    build.
    """
    rho = np.asarray(rho_key, dtype=float)
    base = np.maximum(1.0 - rho**2, 0.0)
    alphas = np.linspace(0.0, 50.0, 2001)
    peaks = np.empty_like(alphas)
    for i, alpha in enumerate(alphas):
        shape = base**alpha
        shape = shape * (1.0 - _EDGE_PEDESTAL) + _EDGE_PEDESTAL
        mean = float(volume_average(shape, rho))
        peaks[i] = shape[0] / max(mean, 1e-300)
    return peaks, alphas


def _alpha_for_peaking(peaking: float, rho: np.ndarray) -> float:
    """Return alpha for shape=(1-rho^2)^alpha with requested peak/average."""
    target = max(float(peaking), 1.0)
    if target <= 1.0 + 1e-12:
        return 0.0
    peaks, alphas = _peaking_table(tuple(float(v) for v in np.asarray(rho, dtype=float)))
    if target >= peaks[-1]:
        return float(alphas[-1])
    return float(np.interp(target, peaks, alphas))


def _parabolic_profile(average: Any, peaking: Any, rho: Any) -> np.ndarray:
    """Return an ``average * shape`` profile whose volume-average equals ``average``.

    ``average`` and ``peaking`` may be scalars (per-point) or batched ``(N, 1)``
    columns (the popcon grid), in which case the result is ``(N, P)`` with each
    row's shape exponent solved from that row's peaking.
    """
    rho_arr = np.asarray(rho, dtype=float)
    if rho_arr.ndim != 1:
        raise ValueError("rho must be a one-dimensional profile grid")
    peak_arr = np.asarray(peaking, dtype=float).reshape(-1)
    if peak_arr.size == 1:
        alpha: Any = _alpha_for_peaking(float(peak_arr[0]), rho_arr)
    else:
        peaks, alphas = _peaking_table(tuple(float(v) for v in rho_arr))
        target = np.maximum(peak_arr, 1.0)
        alpha = np.where(target >= peaks[-1], alphas[-1], np.interp(target, peaks, alphas))
        alpha = np.where(target <= 1.0 + 1e-12, 0.0, alpha)[:, None]
    shape = np.maximum(1.0 - rho_arr**2, 0.0) ** alpha
    shape = shape * (1.0 - _EDGE_PEDESTAL) + _EDGE_PEDESTAL
    mean = volume_average(shape, rho_arr)
    unit = shape / np.maximum(np.asarray(mean, dtype=float), 1e-300)
    avg_arr = np.asarray(average, dtype=float)
    if avg_arr.ndim == 0:
        return float(avg_arr) * unit
    return avg_arr.reshape(avg_arr.shape[0], 1) * unit


def _profile_max(profile: Any, label: str) -> Any:
    """Return the finite maximum of a radial profile.

    A single profile ``(P,)`` yields the scalar peak (and raises on a
    non-finite/empty profile).  A batched ``(N, P)`` stack -- the popcon grid --
    yields the per-row peak ``(N,)``, reducing over the profile grid only so the
    relation stays batched instead of collapsing to one point's value; rows that
    are non-finite become ``NaN`` and poison only their own grid point.
    """
    arr = np.asarray(profile, dtype=float)
    if arr.ndim == 0 or arr.shape[-1] == 0:
        raise ValueError(f"{label} profile must be finite and non-empty")
    if arr.ndim == 1:
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{label} profile must be finite and non-empty")
        return float(np.max(arr))
    finite_rows = np.all(np.isfinite(arr), axis=-1)
    return np.where(finite_rows, np.max(arr, axis=-1), np.nan)


def _peaking_residual(peak: Any, average: Any, peaking: Any) -> Any:
    """Return the normalized residual of ``peak == average * peaking``.

    Normalized by the operand magnitude so it stays O(1) regardless of the
    quantity's physical scale.  A bare ``peak - average * peaking`` difference
    carries no scale, so certification derives the tolerance from the residual
    itself and can never accept a non-zero value -- which for densities
    (~1e19) fails every point on the ~1e-6 discretization error of the
    generated profile.  Mirrors the convention of the energy-confinement
    balance residual.
    """
    lhs = np.asarray(peak, dtype=float)
    rhs = np.asarray(average, dtype=float) * np.asarray(peaking, dtype=float)
    scale = np.maximum(np.maximum(np.abs(lhs), np.abs(rhs)), 1.0)
    return (lhs - rhs) / scale


@relation(
    name="Parabolic ion temperature profile",
    tags=("plasma", "profile", "profile_shape"),
    outputs="T_i",
    dependency="generated_profile",
)
def parabolic_ion_temperature_profile(T_i_avg: float, ion_temperature_peaking: float, rho: Any) -> Any:
    """Generate an ion-temperature profile from average and peaking factor."""
    return _parabolic_profile(T_i_avg, ion_temperature_peaking, rho)


@relation(
    name="Parabolic electron temperature profile",
    tags=("plasma", "profile", "profile_shape"),
    outputs="T_e",
    dependency="generated_profile",
)
def parabolic_electron_temperature_profile(T_e_avg: float, temperature_peaking: float, rho: Any) -> Any:
    """Generate an electron-temperature profile from average and peaking factor."""
    return _parabolic_profile(T_e_avg, temperature_peaking, rho)


@relation(
    name="Parabolic ion density profile",
    tags=("plasma", "profile", "profile_shape"),
    outputs="n_fuel",
    dependency="generated_profile",
)
def parabolic_ion_density_profile(n_fuel_avg: float, ion_density_peaking: float, rho: Any) -> Any:
    """Generate an ion-density profile from average and peaking factor."""
    return _parabolic_profile(n_fuel_avg, ion_density_peaking, rho)


@relation(
    name="Parabolic electron density profile",
    tags=("plasma", "profile", "profile_shape"),
    outputs="n_e",
    dependency="generated_profile",
)
def parabolic_electron_density_profile(n_e_avg: float, density_peaking: float, rho: Any) -> Any:
    """Generate an electron-density profile from average and peaking factor."""
    return _parabolic_profile(n_e_avg, density_peaking, rho)


@relation(
    name="Peak magnetic-field profile value",
    tags=("plasma", "profile", "tokamak", "stellarator", "mirror"),
    outputs="B_profile_max",
)
def peak_magnetic_field_profile_value(B: Any, rho: Any) -> float:
    """Return the maximum magnetic-field value from the magnetic-field profile."""
    return _profile_max(B, "magnetic-field")


@relation(
    name="Peak electron temperature from profile",
    tags=("plasma", "profile", "tokamak", "stellarator"),
    outputs="T0",
)
def peak_electron_temperature_from_profile(T_e: Any, rho: Any) -> float:
    """Return the maximum electron temperature from the electron profile."""
    return _profile_max(T_e, "electron temperature")


@relation(
    name="Peak ion temperature from profile",
    tags=("plasma", "profile", "tokamak", "stellarator"),
    outputs="T_i_peak",
)
def peak_ion_temperature_from_profile(T_i: Any, rho: Any) -> float:
    """Return the maximum ion temperature from the ion profile."""
    return _profile_max(T_i, "ion temperature")


@relation(
    name="Peak electron density from profile",
    tags=("plasma", "profile", "tokamak", "stellarator", "mirror"),
    outputs="n0",
)
def peak_electron_density_from_profile(n_e: Any, rho: Any) -> float:
    """Return the maximum electron density from the electron-density profile."""
    return _profile_max(n_e, "electron density")


@relation(
    name="Peak ion density from profile",
    tags=("plasma", "profile", "tokamak", "stellarator", "mirror"),
    outputs="n_fuel_peak",
)
def peak_ion_density_from_profile(n_fuel: Any, rho: Any) -> float:
    """Return the maximum fuel-ion density from the fuel-ion-density profile."""
    return _profile_max(n_fuel, "fuel ion density")


@relation(
    name="Peak total ion density from profile",
    tags=("plasma", "profile", "tokamak", "stellarator", "mirror"),
    outputs="n_i_peak",
)
def peak_total_ion_density_from_profile(n_i: Any, rho: Any) -> float:
    """Return the maximum TOTAL ion density (fuel + impurity) from the profile.

    Feeds the peak plasma pressure ``n0 T0 + n_i_peak T_i_peak``, which counts
    all ions -- consistent with the volume-averaged thermal pressure using the
    total ion density n_i = n_fuel + n_imp.
    """
    return _profile_max(n_i, "total ion density")


@relation(
    name="Peak deuterium density from profile",
    tags=("plasma", "profile", "tokamak", "stellarator", "mirror"),
    outputs="n_D_peak",
)
def peak_deuterium_density_from_profile(n_D: Any, rho: Any) -> float:
    """Return the maximum deuterium density from the deuterium-density profile."""
    return _profile_max(n_D, "deuterium density")


@relation(
    name="Peak tritium density from profile",
    tags=("plasma", "profile", "tokamak", "stellarator", "mirror"),
    outputs="n_T_peak",
)
def peak_tritium_density_from_profile(n_T: Any, rho: Any) -> float:
    """Return the maximum tritium density from the tritium-density profile."""
    return _profile_max(n_T, "tritium density")


@relation(
    name="Peak helium-3 density from profile",
    tags=("plasma", "profile", "tokamak", "stellarator", "mirror"),
    outputs="n_He3_peak",
)
def peak_helium3_density_from_profile(n_He3: Any, rho: Any) -> float:
    """Return the maximum helium-3 density from the helium-3-density profile."""
    return _profile_max(n_He3, "helium-3 density")


@relation(
    name="Peak helium-4 density from profile",
    tags=("plasma", "profile", "tokamak", "stellarator", "mirror"),
    outputs="n_He4_peak",
)
def peak_helium4_density_from_profile(n_He4: Any, rho: Any) -> float:
    """Return the maximum helium-4 density from the helium-4-density profile."""
    return _profile_max(n_He4, "helium-4 density")


@relation(
    name="Peak DT reactivity from profile",
    tags=("plasma", "profile", "tokamak", "stellarator", "mirror"),
    outputs="sigmav_DT_peak",
)
def peak_dt_reactivity_from_profile(sigmav_DT: Any, rho: Any) -> float:
    """Return the maximum D-T reactivity from the reactivity profile."""
    return _profile_max(sigmav_DT, "D-T reactivity")


@relation(
    name="Peak DDn reactivity from profile",
    tags=("plasma", "profile", "tokamak", "stellarator", "mirror"),
    outputs="sigmav_DDn_peak",
)
def peak_ddn_reactivity_from_profile(sigmav_DDn: Any, rho: Any) -> float:
    """Return the maximum D-D (He3+n) reactivity from the reactivity profile."""
    return _profile_max(sigmav_DDn, "D-D (He3+n) reactivity")


@relation(
    name="Peak DD reactivity from profile",
    tags=("plasma", "profile", "tokamak", "stellarator", "mirror"),
    outputs="sigmav_DD_peak",
)
def peak_dd_reactivity_from_profile(sigmav_DD: Any, rho: Any) -> float:
    """Return the maximum total D-D reactivity from the reactivity profile."""
    return _profile_max(sigmav_DD, "D-D reactivity")


@relation(
    name="Peak DDp reactivity from profile",
    tags=("plasma", "profile", "tokamak", "stellarator", "mirror"),
    outputs="sigmav_DDp_peak",
)
def peak_ddp_reactivity_from_profile(sigmav_DDp: Any, rho: Any) -> float:
    """Return the maximum D-D (T+p) reactivity from the reactivity profile."""
    return _profile_max(sigmav_DDp, "D-D (T+p) reactivity")


@relation(
    name="Peak DHe3 reactivity from profile",
    tags=("plasma", "profile", "tokamak", "stellarator", "mirror"),
    outputs="sigmav_DHe3_peak",
)
def peak_dhe3_reactivity_from_profile(sigmav_DHe3: Any, rho: Any) -> float:
    """Return the maximum D-He3 reactivity from the reactivity profile."""
    return _profile_max(sigmav_DHe3, "D-He3 reactivity")


@relation(
    name="Peak TT reactivity from profile",
    tags=("plasma", "profile", "tokamak", "stellarator", "mirror"),
    outputs="sigmav_TT_peak",
)
def peak_tt_reactivity_from_profile(sigmav_TT: Any, rho: Any) -> float:
    """Return the maximum T-T reactivity from the reactivity profile."""
    return _profile_max(sigmav_TT, "T-T reactivity")


@relation(
    name="Peak He3He3 reactivity from profile",
    tags=("plasma", "profile", "tokamak", "stellarator", "mirror"),
    outputs="sigmav_He3He3_peak",
)
def peak_he3he3_reactivity_from_profile(sigmav_He3He3: Any, rho: Any) -> float:
    """Return the maximum He3-He3 reactivity from the reactivity profile."""
    return _profile_max(sigmav_He3He3, "He3-He3 reactivity")


@relation(
    name="Peak THe3 reactivity from profile",
    tags=("plasma", "profile", "tokamak", "stellarator", "mirror"),
    outputs="sigmav_THe3_peak",
)
def peak_the3_reactivity_from_profile(sigmav_THe3: Any, rho: Any) -> float:
    """Return the maximum total T-He3 reactivity from the reactivity profile."""
    return _profile_max(sigmav_THe3, "T-He3 reactivity")


@relation(
    name="Peak THe3_D reactivity from profile",
    tags=("plasma", "profile", "tokamak", "stellarator", "mirror"),
    outputs="sigmav_THe3_D_peak",
)
def peak_the3_d_reactivity_from_profile(sigmav_THe3_D: Any, rho: Any) -> float:
    """Return the maximum T-He3 alpha+D branch reactivity from the reactivity profile."""
    return _profile_max(sigmav_THe3_D, "T-He3 alpha+D branch reactivity")


@relation(
    name="Peak THe3_np reactivity from profile",
    tags=("plasma", "profile", "tokamak", "stellarator", "mirror"),
    outputs="sigmav_THe3_np_peak",
)
def peak_the3_np_reactivity_from_profile(sigmav_THe3_np: Any, rho: Any) -> float:
    """Return the maximum T-He3 alpha+n+p branch reactivity from the reactivity profile."""
    return _profile_max(sigmav_THe3_np, "T-He3 alpha+n+p branch reactivity")


@relation(
    name="Electron temperature peaking from peak and average",
    tags=("plasma", "profile", "tokamak", "stellarator", "mirror"),
    outputs="temperature_peaking",
    # Warned, not enforced: a HOLLOW profile (peaking below 1) is physical --
    # it occurs at the very centre of the core -- so it must not fail the
    # reconcile.  But it is unusual enough to surface, and a reactor that
    # wants it rejected can re-declare the same constraint enforced.
    constraints=(("temperature_peaking >= 1.0", False),),
)
def electron_temperature_peaking_from_peak_and_average(T0: float, T_e_avg: float) -> float:
    """Return electron-temperature peaking from peak and volume-average values.

    The ``>= 1`` bound lives on ``temperature_peaking`` (domain ``[1, inf)``), not
    here: clamping inside the relation makes it non-invertible, so solving it for
    ``T0`` at peaking 1 admits every ``T0 <= T_e_avg``.  See NOTES.md.
    """
    return T0 / T_e_avg


@relation(
    name="Ion temperature peaking from peak and average",
    tags=("plasma", "profile", "tokamak", "stellarator", "mirror"),
    outputs="ion_temperature_peaking",
    # Warned, not enforced: a HOLLOW profile (peaking below 1) is physical --
    # it occurs at the very centre of the core -- so it must not fail the
    # reconcile.  But it is unusual enough to surface, and a reactor that
    # wants it rejected can re-declare the same constraint enforced.
    constraints=(("ion_temperature_peaking >= 1.0", False),),
)
def ion_temperature_peaking_from_peak_and_average(T_i_peak: float, T_i_avg: float) -> float:
    """Return ion-temperature peaking from peak and volume-average values.

    Bound carried by ``ion_temperature_peaking``'s domain; see the electron
    variant above.
    """
    return T_i_peak / T_i_avg


@relation(
    name="Electron density peaking from peak and average",
    tags=("plasma", "profile", "tokamak", "stellarator", "mirror"),
    outputs="density_peaking",
    # Warned, not enforced: a HOLLOW profile (peaking below 1) is physical --
    # it occurs at the very centre of the core -- so it must not fail the
    # reconcile.  But it is unusual enough to surface, and a reactor that
    # wants it rejected can re-declare the same constraint enforced.
    constraints=(("density_peaking >= 1.0", False),),
)
def electron_density_peaking_from_peak_and_average(n0: float, n_e_avg: float) -> float:
    """Return electron-density peaking from peak and volume-average values.

    Bound carried by ``density_peaking``'s domain; see the temperature variant
    above.
    """
    return n0 / n_e_avg


@relation(
    name="Ion density peaking from peak and average",
    tags=("plasma", "profile", "tokamak", "stellarator", "mirror"),
    outputs="ion_density_peaking",
    # Warned, not enforced: a HOLLOW profile (peaking below 1) is physical --
    # it occurs at the very centre of the core -- so it must not fail the
    # reconcile.  But it is unusual enough to surface, and a reactor that
    # wants it rejected can re-declare the same constraint enforced.
    constraints=(("ion_density_peaking >= 1.0", False),),
)
def ion_density_peaking_from_peak_and_average(n_fuel_peak: float, n_fuel_avg: float) -> float:
    """Return ion-density peaking from peak and volume-average values.

    Bound carried by ``ion_density_peaking``'s domain; see the temperature
    variant above.
    """
    return n_fuel_peak / n_fuel_avg


@relation(
    name="Electron temperature peaking consistency",
    tags=("plasma", "profile", "tokamak", "stellarator"),
)
def electron_temperature_peaking_consistency(T0: float, T_e_avg: float, temperature_peaking: float) -> float:
    """Constrain electron peaking as ``T0 / T_e_avg`` independent of direction.

    Adapted from cfspopcon; see README.md section "Third-party Notices".

    This relation is outputless by design: with any two of ``T0``, ``T_e_avg``
    and ``temperature_peaking`` known, the third can be reconstructed by the
    acausal seeding/solve path.

    Args:
        T_e_avg: :term:`glossary link<average_electron_temp>`
        T0: :term:`glossary link<average_electron_temp>`
        temperature_peaking: :term:`glossary link<temperature_peaking>`

    Returns:
        Zero when ``T0 == T_e_avg * temperature_peaking``.
    """
    return _peaking_residual(T0, T_e_avg, temperature_peaking)


@relation(
    name="Ion temperature peaking consistency",
    tags=("plasma", "profile", "tokamak", "stellarator"),
)
def ion_temperature_peaking_consistency(T_i_peak: float, T_i_avg: float, ion_temperature_peaking: float) -> float:
    """Constrain ion peaking as ``T_i_peak / T_i_avg`` independent of direction.

    ``ion_temperature_peaking`` defaults to ``temperature_peaking`` (the
    electron value), which recovers cfspopcon's single shared peaking unless the
    ion profile or a supplied ion peaking value says otherwise.

    Args:
        T_i_avg: :term:`glossary link<average_ion_temp>`
        T_i_peak: :term:`glossary link<average_ion_temp>`
        ion_temperature_peaking: :term:`glossary link<ion_temperature_peaking>`

    Returns:
        Zero when ``T_i_peak == T_i_avg * ion_temperature_peaking``.
    """
    return _peaking_residual(T_i_peak, T_i_avg, ion_temperature_peaking)


@relation(
    name="Electron density peaking consistency",
    tags=("plasma", "profile", "tokamak", "stellarator", "mirror"),
)
def electron_density_peaking_consistency(n0: float, n_e_avg: float, density_peaking: float) -> float:
    """Constrain electron-density peaking as ``n0 / n_e_avg`` independent of direction."""
    return _peaking_residual(n0, n_e_avg, density_peaking)


@relation(
    name="Ion density peaking consistency",
    tags=("plasma", "profile", "tokamak", "stellarator", "mirror"),
)
def ion_density_peaking_consistency(n_fuel_peak: float, n_fuel_avg: float, ion_density_peaking: float) -> float:
    """Constrain ion-density peaking as ``n_fuel_peak / n_fuel_avg`` independent of direction."""
    return _peaking_residual(n_fuel_peak, n_fuel_avg, ion_density_peaking)
