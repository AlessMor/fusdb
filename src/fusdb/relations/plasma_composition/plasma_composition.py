"""Density-based plasma composition relations."""

from __future__ import annotations

import math

import numpy as np

from fusdb.registry import load_allowed_species, load_allowed_variables
from fusdb.relation_util import relation
from fusdb.utils import integrate_profile

allowed_species = load_allowed_species()
allowed_variables, _, _ = load_allowed_variables()


@relation(
    name="Ion density from tracked species densities",
    output="n_i",
    tags=("plasma",),
)
def ion_density_from_tracked_species_densities(
    n_D: float,
    n_T: float,
    n_He3: float,
    n_He4: float,
    n_imp: float,
) -> float:
    """Return the total tracked ion density profile.

    Args:
        n_D: Deuterium density.
        n_T: Tritium density.
        n_He3: Helium-3 density.
        n_He4: Helium-4 density.
        n_imp: Generic impurity density.

    Returns:
        Total tracked ion density.
    """
    # Sum fuel and impurity densities into the total ion inventory.
    return n_D + n_T + n_He3 + n_He4 + n_imp


@relation(
    name="Electron density from tracked species densities",
    output="n_e",
    tags=("plasma",),
)
def electron_density_from_tracked_species_densities(
    n_D: float,
    n_T: float,
    n_He3: float,
    n_He4: float,
    n_imp: float,
) -> float:
    """Return the electron density profile from tracked ion densities.

    Args:
        n_D: Deuterium density.
        n_T: Tritium density.
        n_He3: Helium-3 density.
        n_He4: Helium-4 density.
        n_imp: Generic impurity density.

    Returns:
        Electron density derived from charge neutrality.
    """
    # Load impurity charge state from the species registry.
    z_imp = float(allowed_species["Imp"]["atomic_number"])

    # TODO(med): make function more flexible to handle more complex impurity compositions with multiple charge states.
    # Sum charge-weighted ion densities to get electron density.
    return n_D + n_T + 2.0 * n_He3 + 2.0 * n_He4 + z_imp * n_imp


@relation(
    name="Integrated D fraction from density profiles",
    output="f_D",
    tags=("plasma",),
    rel_tol_default=1e-12,
    abs_tol_default=1e-12,
)
def integrated_deuterium_fraction_from_density_profiles(
    n_D: float,
    n_i: float,
) -> float:
    """Return the integrated deuterium fraction from density profiles.

    Args:
        n_D: Deuterium density profile.
        n_i: Total ion density profile.

    Returns:
        Integrated deuterium fraction.
    """
    # Integrate the deuterium inventory.
    numerator_total = integrate_profile(n_D, error_label="density")
    # Integrate the total ion inventory.
    denominator_total = integrate_profile(n_i, error_label="density")
    # Ensure a positive denominator when numerical data are provided.
    if getattr(denominator_total, "free_symbols", None) is None and denominator_total <= 0.0:
        raise ValueError("Tracked ion inventory must be positive")
    # Normalize the deuterium inventory to the total inventory.
    return numerator_total / denominator_total


@relation(
    name="Integrated T fraction from density profiles",
    output="f_T",
    tags=("plasma",),
    rel_tol_default=1e-12,
    abs_tol_default=1e-12,
)
def integrated_tritium_fraction_from_density_profiles(
    n_T: float,
    n_i: float,
) -> float:
    """Return the integrated tritium fraction from density profiles.

    Args:
        n_T: Tritium density profile.
        n_i: Total ion density profile.

    Returns:
        Integrated tritium fraction.
    """
    # Integrate the tritium inventory.
    numerator_total = integrate_profile(n_T, error_label="density")
    # Integrate the total ion inventory.
    denominator_total = integrate_profile(n_i, error_label="density")
    # Ensure a positive denominator when numerical data are provided.
    if getattr(denominator_total, "free_symbols", None) is None and denominator_total <= 0.0:
        raise ValueError("Tracked ion inventory must be positive")
    # Normalize the tritium inventory to the total inventory.
    return numerator_total / denominator_total


@relation(
    name="Integrated He3 fraction from density profiles",
    output="f_He3",
    tags=("plasma",),
    rel_tol_default=1e-12,
    abs_tol_default=1e-12,
)
def integrated_helium3_fraction_from_density_profiles(
    n_He3: float,
    n_i: float,
) -> float:
    """Return the integrated helium-3 fraction from density profiles.

    Args:
        n_He3: Helium-3 density profile.
        n_i: Total ion density profile.

    Returns:
        Integrated helium-3 fraction.
    """
    # Integrate the helium-3 inventory.
    numerator_total = integrate_profile(n_He3, error_label="density")
    # Integrate the total ion inventory.
    denominator_total = integrate_profile(n_i, error_label="density")
    # Ensure a positive denominator when numerical data are provided.
    if getattr(denominator_total, "free_symbols", None) is None and denominator_total <= 0.0:
        raise ValueError("Tracked ion inventory must be positive")
    # Normalize the helium-3 inventory to the total inventory.
    return numerator_total / denominator_total


@relation(
    name="Integrated He4 fraction from density profiles",
    output="f_He4",
    tags=("plasma",),
    rel_tol_default=1e-12,
    abs_tol_default=1e-12,
)
def integrated_helium4_fraction_from_density_profiles(
    n_He4: float,
    n_i: float,
) -> float:
    """Return the integrated helium-4 fraction from density profiles.

    Args:
        n_He4: Helium-4 density profile.
        n_i: Total ion density profile.

    Returns:
        Integrated helium-4 fraction.
    """
    # Integrate the helium-4 inventory.
    numerator_total = integrate_profile(n_He4, error_label="density")
    # Integrate the total ion inventory.
    denominator_total = integrate_profile(n_i, error_label="density")
    # Ensure a positive denominator when numerical data are provided.
    if getattr(denominator_total, "free_symbols", None) is None and denominator_total <= 0.0:
        raise ValueError("Tracked ion inventory must be positive")
    # Normalize the helium-4 inventory to the total inventory.
    return numerator_total / denominator_total


@relation(
    name="Integrated Imp fraction from density profiles",
    output="f_Imp",
    tags=("plasma",),
    rel_tol_default=1e-12,
    abs_tol_default=1e-12,
)
def integrated_impurity_fraction_from_density_profiles(
    n_imp: float,
    n_i: float,
) -> float:
    """Return the integrated generic impurity fraction from density profiles.

    Args:
        n_imp: Generic impurity density profile.
        n_i: Total ion density profile.

    Returns:
        Integrated impurity fraction.
    """
    # Integrate impurity inventory.
    numerator_total = integrate_profile(n_imp, error_label="density")
    # Integrate total ion inventory.
    denominator_total = integrate_profile(n_i, error_label="density")
    # Ensure a positive denominator when numerical data are provided.
    if getattr(denominator_total, "free_symbols", None) is None and denominator_total <= 0.0:
        raise ValueError("Tracked ion inventory must be positive")
    # Normalize the impurity inventory to the total inventory.
    return numerator_total / denominator_total


@relation(
    name="Integrated tracked fractions",
    outputs=tuple(
        f"f_{species}"
        for species in allowed_species
        if (
            ("n_imp" if species == "Imp" else f"n_{species}") in allowed_variables
            and f"f_{species}" in allowed_variables
        )
    ),
    tags=("plasma",),
    rel_tol_default=1e-12,
    abs_tol_default=1e-12,
)
def integrated_tracked_fractions(
    n_D: float,
    n_T: float,
    n_He3: float,
    n_He4: float,
    n_imp: float,
    n_i: float,
) -> dict[str, float]:
    """Return the full integrated tracked-fraction bundle from density profiles.

    Args:
        n_D: Deuterium density profile.
        n_T: Tritium density profile.
        n_He3: Helium-3 density profile.
        n_He4: Helium-4 density profile.
        n_imp: Generic impurity density profile.
        n_i: Total ion density profile.

    Returns:
        Mapping of fraction names (``f_*``) to their integrated values.
    """
    # Collect the density profiles by species.
    densities = {
        "D": n_D,
        "T": n_T,
        "He3": n_He3,
        "He4": n_He4,
        "Imp": n_imp,
    }
    # Determine which species have both density and fraction variables.
    tracked_species = tuple(
        species
        for species in allowed_species
        if (
            ("n_imp" if species == "Imp" else f"n_{species}") in allowed_variables
            and f"f_{species}" in allowed_variables
        )
    )
    # Integrate the total ion inventory.
    denominator = integrate_profile(n_i, error_label="density")
    # Ensure a positive denominator when numerical data are provided.
    if getattr(denominator, "free_symbols", None) is None and denominator <= 0.0:
        raise ValueError("Tracked ion inventory must be positive")
    # Normalize each species inventory to the total ion inventory.
    return {
        f"f_{species}": integrate_profile(densities[species], error_label="density") / denominator
        for species in tracked_species
    }


@relation(
    name="Average fuel mass number",
    output="afuel",
    tags=("plasma",),
)
def average_fuel_mass_number(
    f_D: float,
    f_T: float,
    f_He3: float,
) -> float:
    """Return the integrated average fuel mass number.

    Args:
        f_D: Deuterium fraction.
        f_T: Tritium fraction.
        f_He3: Helium-3 fraction.

    Returns:
        Fuel-only average ion mass number.
    """
    # Sum the fuel fractions to get the fuel-only inventory.
    fuel_total = f_D + f_T + f_He3
    # Ensure a positive total when numerical data are provided.
    if getattr(fuel_total, "free_symbols", None) is None and fuel_total <= 0.0:
        raise ValueError("Fuel ion inventory must be positive")
    # Load fuel mass numbers from the species registry.
    # Form the weighted fuel mass numerator.
    numerator = (
        f_D * float(allowed_species["D"]["atomic_mass"])
        + f_T * float(allowed_species["T"]["atomic_mass"])
        + f_He3 * float(allowed_species["He3"]["atomic_mass"])
    )
    # Normalize by the fuel inventory.
    return numerator / fuel_total


def plasma_balance_ode(
    n_D: float,
    n_T: float,
    n_He3: float,
    n_He4: float,
    sigmav_DT: float,
    sigmav_DDn: float,
    sigmav_DDp: float,
    sigmav_DHe3: float,
    sigmav_TT: float,
    sigmav_He3He3: float,
    sigmav_THe3_D: float,
    sigmav_THe3_np: float,
    tau_p_D: float | None,
    tau_p_T: float | None,
    tau_p_He3: float | None,
    tau_p_He4: float | None,
    *,
    injection_fractions: np.ndarray | tuple[float, float, float, float] | None = None,
) -> tuple[float, float, float, float]:
    """Return the tracked-species ODE with implicit fueling to hold total density fixed.

    Args:
        n_D: Deuterium density.
        n_T: Tritium density.
        n_He3: Helium-3 density.
        n_He4: Helium-4 density.
        sigmav_DT: DT reactivity.
        sigmav_DDn: DDn reactivity.
        sigmav_DDp: DDp reactivity.
        sigmav_DHe3: DHe3 reactivity.
        sigmav_TT: TT reactivity.
        sigmav_He3He3: He3He3 reactivity.
        sigmav_THe3_D: THe3-to-D branch reactivity.
        sigmav_THe3_np: THe3-to-np branch reactivity.
        tau_p_D: Deuterium particle confinement time.
        tau_p_T: Tritium particle confinement time.
        tau_p_He3: Helium-3 particle confinement time.
        tau_p_He4: Helium-4 particle confinement time.
        injection_fractions: Optional fuel split ``(f_D, f_T, f_He3, f_He4)``. When
            ``None``, use the local state fractions.

    Returns:
        Time derivatives ``(dn_D/dt, dn_T/dt, dn_He3/dt, dn_He4/dt)`` including
        the implicit fueling source.
    """
    # Combine fusion production, fusion burn, and particle losses for D.
    dn_D_dt = (
        - n_D * n_T * sigmav_DT
        - n_D**2 * (sigmav_DDn + sigmav_DDp)
        - n_D * n_He3 * sigmav_DHe3
        + n_T * n_He3 * sigmav_THe3_D
        - (0.0 if tau_p_D is None else 1.0 / float(tau_p_D)) * n_D
    )

    # Combine fusion production, fusion burn, and particle losses for T.
    dn_T_dt = (
        + 0.5 * n_D**2 * sigmav_DDp
        - n_D * n_T * sigmav_DT
        - n_T**2 * sigmav_TT
        - n_T * n_He3 * (sigmav_THe3_D + sigmav_THe3_np)
        - (0.0 if tau_p_T is None else 1.0 / float(tau_p_T)) * n_T
    )

    # Combine fusion production, fusion burn, and particle losses for He3.
    dn_He3_dt = (
        + 0.5 * n_D**2 * sigmav_DDn
        - n_D * n_He3 * sigmav_DHe3
        - n_He3**2 * sigmav_He3He3
        - n_T * n_He3 * (sigmav_THe3_D + sigmav_THe3_np)
        - (0.0 if tau_p_He3 is None else 1.0 / float(tau_p_He3)) * n_He3
    )

    # Combine fusion production and particle losses for He4 ash.
    dn_He4_dt = (
        + n_D * n_T * sigmav_DT
        + n_D * n_He3 * sigmav_DHe3
        + 0.5 * n_T**2 * sigmav_TT
        + 0.5 * n_He3**2 * sigmav_He3He3
        + n_T * n_He3 * (sigmav_THe3_D + sigmav_THe3_np)
        - (0.0 if tau_p_He4 is None else 1.0 / float(tau_p_He4)) * n_He4
    )

    # Determine the fueling split that keeps the total density fixed.
    total_density = n_D + n_T + n_He3 + n_He4
    if total_density <= 0.0:
        # Skip the implicit source if the total density is zero.
        return dn_D_dt, dn_T_dt, dn_He3_dt, dn_He4_dt

    if injection_fractions is None:
        # Default to feeding the same split as the local state.
        feed = np.asarray(
            [n_D, n_T, n_He3, n_He4],
            dtype=float,
        ) / total_density
    else:
        # Validate and normalize the provided injection split.
        feed = np.asarray(injection_fractions, dtype=float)
        if feed.shape != (4,) or not np.isfinite(feed).all() or np.any(feed < 0.0):
            raise ValueError("injection_fractions must be a length-4 non-negative vector")
        if float(np.sum(feed)) <= 0.0:
            raise ValueError("injection_fractions must sum to a positive value")
        feed = feed / float(np.sum(feed))

    # Compute the total balance and distribute the source to hold total density.
    net_balance = dn_D_dt + dn_T_dt + dn_He3_dt + dn_He4_dt
    source = -net_balance * feed

    return (
        dn_D_dt + float(source[0]),
        dn_T_dt + float(source[1]),
        dn_He3_dt + float(source[2]),
        dn_He4_dt + float(source[3]),
    )


@relation(
    name="Steady-state plasma composition",
    inputs=(
        "n_D",
        "n_T",
        "n_He3",
        "n_He4",
        "sigmav_DT",
        "sigmav_DDn",
        "sigmav_DDp",
        "sigmav_DHe3",
        "sigmav_TT",
        "sigmav_He3He3",
        "sigmav_THe3_D",
        "sigmav_THe3_np",
        "tau_p_D",
        "tau_p_T",
        "tau_p_He3",
        "tau_p_He4",
    ),
    outputs=("n_D", "n_T", "n_He3", "n_He4"),
    tags=("plasma",),
    rel_tol_default=1e-10,
    abs_tol_default=1e-12,
)
def steady_state_plasma_composition(
    n_D: np.ndarray,
    n_T: np.ndarray,
    n_He3: np.ndarray,
    n_He4: np.ndarray,
    sigmav_DT: np.ndarray,
    sigmav_DDn: np.ndarray,
    sigmav_DDp: np.ndarray,
    sigmav_DHe3: np.ndarray,
    sigmav_TT: np.ndarray,
    sigmav_He3He3: np.ndarray,
    sigmav_THe3_D: np.ndarray,
    sigmav_THe3_np: np.ndarray,
    tau_p_D: float | None,
    tau_p_T: float | None,
    tau_p_He3: float | None,
    tau_p_He4: float | None,
    *,
    tol: float = 1e-10,
    max_iter: int = 500,
    method: str = "hybr",
) -> dict[str, np.ndarray]:
    """Solve the pointwise steady-state D/T/He3/He4 composition.

    Uses a root-finding approach to solve the local composition ODE at each
    point. The implicit fueling source keeps the total density fixed, and its
    species split mirrors the seeded input densities at each point.

    #NOTE: impurities are not included since they do not participate in fusion reactions.
    Since the function considers densities and not fractions, impurity density is not needed.
    
    Args:
        n_D: Seeded deuterium density profile.
        n_T: Seeded tritium density profile.
        n_He3: Seeded helium-3 density profile.
        n_He4: Seeded helium-4 density profile.
        sigmav_DT: DT reactivity profile.
        sigmav_DDn: DDn reactivity profile.
        sigmav_DDp: DDp reactivity profile.
        sigmav_DHe3: DHe3 reactivity profile.
        sigmav_TT: TT reactivity profile.
        sigmav_He3He3: He3He3 reactivity profile.
        sigmav_THe3_D: THe3 to D branch reactivity profile.
        sigmav_THe3_np: THe3 to np branch reactivity profile.
        tau_p_D: Deuterium particle confinement time.
        tau_p_T: Tritium particle confinement time.
        tau_p_He3: Helium-3 particle confinement time.
        tau_p_He4: Helium-4 particle confinement time.
        tol: Root-solver tolerance.
        max_iter: Maximum root-solver evaluations per radial point.
        method: SciPy root-solver method.

    Returns:
        Solved tracked density profiles.
    """
    from scipy.optimize import root

    # Validate the profile inputs before the pointwise solve.
    profiles = (
        n_D,
        n_T,
        n_He3,
        n_He4,
        sigmav_DT,
        sigmav_DDn,
        sigmav_DDp,
        sigmav_DHe3,
        sigmav_TT,
        sigmav_He3He3,
        sigmav_THe3_D,
        sigmav_THe3_np,
    )
    # Ensure every input profile is a 1D numpy array.
    if not all(isinstance(v, np.ndarray) and v.ndim == 1 for v in profiles):
        raise TypeError(
            "Density and reactivity inputs must already be 1D numpy arrays before relation evaluation."
        )
    # Ensure every input profile has the same length.
    if len({v.size for v in profiles}) != 1:
        raise ValueError("Density and reactivity profiles must all have the same length")

    # Validate that confinement times are positive when supplied.
    for name, tau in (("tau_p_D", tau_p_D), ("tau_p_T", tau_p_T), ("tau_p_He3", tau_p_He3), ("tau_p_He4", tau_p_He4)):
        if tau is not None and ((tau := float(tau)) <= 0.0 or not math.isfinite(tau)):
            raise ValueError(f"{name} must be positive or None")

    n_points = n_D.size

    # Allocate the solved density profiles.
    out_D = np.zeros_like(n_D, dtype=float)
    out_T = np.zeros_like(n_T, dtype=float)
    out_He3 = np.zeros_like(n_He3, dtype=float)
    out_He4 = np.zeros_like(n_He4, dtype=float)

    # Solve the steady-state composition independently at each radial point.
    for i in range(n_points):
        # Pack the seed densities for this point and validate them.
        input_densities = np.asarray([n_D[i], n_T[i], n_He3[i], n_He4[i]], dtype=float)
        if not np.isfinite(input_densities).all() or np.any(input_densities < 0.0):
            raise ValueError("Seeded densities must be finite and non-negative")
        total_density = float(np.sum(input_densities))
        # Skip points with zero total density.
        if total_density <= 0.0:
            continue
        # Normalize the seed densities to get the implicit injection split.
        initial_density_fractions = input_densities / total_density

        def residual(fractions: np.ndarray) -> np.ndarray:
            """Compute the local steady-state residual with an implicit source.

            Args:
                fractions: Candidate density fractions at this radial point.

            Returns:
                Residual vector for D, T, He3, and the total-density constraint.
            """
            # Work in fractions to keep the solver scale near unity.
            fractions = np.asarray(fractions, dtype=float)
            # Convert fractions back to physical densities.
            state = total_density * fractions

            # Evaluate the balance with the seeded fuel split.
            dn_D_dt, dn_T_dt, dn_He3_dt, dn_He4_dt = plasma_balance_ode(
                n_D=float(state[0]),
                n_T=float(state[1]),
                n_He3=float(state[2]),
                n_He4=float(state[3]),
                sigmav_DT=float(sigmav_DT[i]),
                sigmav_DDn=float(sigmav_DDn[i]),
                sigmav_DDp=float(sigmav_DDp[i]),
                sigmav_DHe3=float(sigmav_DHe3[i]),
                sigmav_TT=float(sigmav_TT[i]),
                sigmav_He3He3=float(sigmav_He3He3[i]),
                sigmav_THe3_D=float(sigmav_THe3_D[i]),
                sigmav_THe3_np=float(sigmav_THe3_np[i]),
                tau_p_D=tau_p_D,
                tau_p_T=tau_p_T,
                tau_p_He3=tau_p_He3,
                tau_p_He4=tau_p_He4,
                injection_fractions=initial_density_fractions,
            )

            # Enforce D, T, and He3 balances together with the total-density target.
            return np.asarray(
                [
                    dn_D_dt / total_density,
                    dn_T_dt / total_density,
                    dn_He3_dt / total_density,
                    float(np.sum(fractions)) - 1.0,
                ],
                dtype=float,
            )

        #NOTE: the solver uses fractions instead of densities to keep the solution scale near unity, which generally improves convergence behavior. 
        # The densities are retrieved after the solve.
        result = root(
            residual,
            initial_density_fractions,
            method=method,
            tol=tol,
            options={"maxfev": max_iter, "diag": [1.0, 1.0, 1.0, 1.0]},
        )
        solved_fraction = np.asarray(result.x, dtype=float)
        
        # Reject invalid converged states before the final residual check.
        if ((not np.isfinite(solved_fraction).all()) or (np.any(solved_fraction < 0.0))):
            raise RuntimeError("Steady-state composition solve produced an invalid state")
        
        # Check that the nonlinear solve satisfied every balance tightly.
        residual_vector = residual(solved_fraction)
        residual_norm = float(np.linalg.norm(residual_vector))
        if (
            not math.isfinite(residual_norm)
            or residual_norm > 1e-10
            or abs(float(residual_vector[3])) > 1e-12
        ):
            raise RuntimeError(
                "Steady-state composition solve did not converge tightly enough "
                f"(status={result.success}, message={result.message!r}, "
                f"D={residual_vector[0]:.3e}, T={residual_vector[1]:.3e}, "
                f"He3={residual_vector[2]:.3e}, total={residual_vector[3]:.3e})"
            )
        
        # Scale the solved fractions back to densities and store the result.
        solved_state = total_density * solved_fraction
        out_D[i], out_T[i], out_He3[i], out_He4[i] = solved_state

    return {"n_D": out_D, "n_T": out_T, "n_He3": out_He3, "n_He4": out_He4}
