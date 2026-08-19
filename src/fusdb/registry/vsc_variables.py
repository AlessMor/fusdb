"""Additional variable metadata for VSC-style multi-configuration models.

This is a small registry overlay, analogous to ``coordinate_variables.py``.
It adds only configuration-extension quantities and makes the existing FusDB
stored-energy and transport-loss producers explicit defaults; VSC variants are
therefore opt-in alternatives rather than competing global equations.
"""

from __future__ import annotations

from dataclasses import replace

from .variable_registry import VariableRegistry, VariableSpec

_POS = (0.0, None, False, True)
_NONNEG = (0.0, None, True, True)
_UNIT = (0.0, 1.0, True, True)
_ANY = (None, None, True, True)


def _spec(name: str, unit: str = "dimensionless", *, shape: int = 0, domain=_ANY,
          aliases: tuple[str, ...] = (), description: str = "", default=None,
          default_relation: tuple[str, ...] = ()) -> VariableSpec:
    return VariableSpec(
        name=name, aliases=aliases, unit=unit, shape=shape,
        domain=domain, solver_domain=domain, description=description,
        rel_tol=1.0e-3, abs_tol=1.0e-9 if shape else 0.0,
        default=default, default_relation=default_relation,
    )


def with_vsc_variables(base: VariableRegistry) -> VariableRegistry:
    """Add VSC quantities while preserving the existing FusDB abstractions."""
    specs: list[VariableSpec] = []
    for spec in base:
        # The user's requested semantics: one W_th and one P_loss.  The original
        # relations remain defaults; VSC/profile and mirror-loss forms are named
        # alternatives selected explicitly per reactor.
        if spec.name == "W_th" and not spec.default_relation:
            spec = replace(spec, default_relation=("Thermal stored energy",))
        elif spec.name == "P_loss" and not spec.default_relation:
            spec = replace(spec, default_relation=("Plasma loss power",))
        specs.append(spec)

    additions = [
        # Common coordinates, moments, and two-temperature accounting.
        _spec("rho_vol", shape=1, domain=_UNIT, description="Normalized volume radius sqrt(V(<rho)/V_p).",
              default_relation=("Normalized volume radius (VSC)",)),
        _spec("rho_U", shape=1, domain=_UNIT, description="Normalized logarithmic dipole flux-tube-volume coordinate.",
              default_relation=("Point-dipole normalized U coordinate",)),
        _spec("G_B25", domain=_NONNEG, description="Normalized volume moment <|B/B_ref|^2.5>."),
        _spec("M_B25", "T^2.5*m^3", domain=_NONNEG, description="Dimensional integral of B^2.5 over plasma volume."),
        _spec("tau_C", "s", domain=_POS, description="Prescribed cyclotron/synchrotron loss time."),
        _spec("E_fast_crit", "keV", domain=_NONNEG, description="Stix critical fast-particle energy."),
        _spec("E_fast_product", "keV", domain=_NONNEG, description="Representative birth energy of the fast charged fusion product."),
        _spec("f_fast_ion", domain=_UNIT, description="Fraction of fast-product energy deposited to ions."),
        _spec("f_charged_dep", domain=_UNIT, description="Fraction of charged fusion-product power deposited in the plasma.", default=1.0),
        _spec("f_aux_e", domain=_UNIT, description="Fraction of auxiliary heating deposited in the electron channel.", default=1.0),
        _spec("P_charged_dep", "W", domain=_NONNEG, description="Locally deposited charged fusion-product power."),
        _spec("tau_ei", "s", domain=_POS, description="Electron-ion thermal equilibration time."),
        _spec("P_ei", "W", domain=_ANY, description="Signed ion-to-electron collisional exchange power."),
        _spec("W_e", "J", domain=_NONNEG, description="Electron thermal stored energy."),
        _spec("W_i", "J", domain=_NONNEG, description="Ion thermal stored energy."),
        _spec("P_aux_required_raw", "W", domain=_ANY, description="Signed external heating required by the VSC power account."),

        # Mirror.
        _spec("a_c", "m", domain=_POS, description="Mirror central-cell plasma radius."),
        _spec("L_c", "m", domain=_POS, description="Mirror central-cell length."),
        _spec("L_th", "m", domain=_NONNEG, description="Length of one mirror throat transition."),
        _spec("B_vac", "T", domain=_POS, description="Mirror central-cell vacuum magnetic field."),
        _spec("R_m", domain=_POS, aliases=("mirror_ratio",), description="Vacuum mirror ratio."),
        _spec("B_c", "T", domain=_POS, description="Diamagnetically corrected central-cell field."),
        _spec("R_mc", domain=_POS, description="Diamagnetically corrected mirror ratio."),
        _spec("phi_i", "keV", domain=_NONNEG, description="Mirror ion ambipolar barrier energy per unit charge."),
        _spec("phi_e", "keV", domain=_NONNEG, description="Mirror electron ambipolar barrier energy per unit charge."),
        _spec("tau_ii", "s", domain=_POS, description="Ion-ion collision time."),
        _spec("lambda_ii", "m", domain=_NONNEG, description="Ion-ion collisional mean free path."),
        _spec("rho_i", "m", domain=_POS, description="Ion gyroradius."),
        _spec("v_th_i", "m/s", domain=_POS, description="Ion thermal speed."),
        _spec("tau_Past", "s", domain=_POS, description="Pastukhov mirror confinement time."),
        _spec("tau_gd", "s", domain=_POS, description="Gas-dynamic mirror confinement time."),
        _spec("tau_rho", "s", domain=_POS, description="Radial mirror confinement time."),
        _spec("tau_m", "s", domain=_POS, description="Assembled mirror particle-loss time."),
        _spec("A_th", "m^2", domain=_POS, description="Mirror throat area."),
        _spec("q_throat", "W/m^2", domain=_NONNEG, description="Mirror end-loss power flux at the throat."),
        _spec("q_collector", "W/m^2", domain=_NONNEG, description="Collector-diluted mirror end-loss power flux."),
        _spec("collector_area_ratio", domain=_POS, description="Collector area divided by throat area.", default=1.0),
        _spec("mirror_regime_ratio", domain=_NONNEG, description="Mirror collisionality ratio lambda_ii/(R_mc L_c)."),

        # FRC.
        _spec("r_s", "m", domain=_POS, description="FRC separatrix radius."),
        _spec("r_w", "m", domain=_POS, description="FRC wall/chamber radius."),
        _spec("L_s", "m", domain=_POS, description="FRC separatrix axial length."),
        _spec("x_s", domain=(0.0, 1.0, False, True), description="FRC separatrix-to-wall radius ratio r_s/r_w."),
        _spec("E_frc", domain=_POS, description="FRC elongation L_s/(2 r_s)."),
        _spec("K_frc", domain=_POS, description="Rigid-rotor equilibrium parameter K."),
        _spec("B_e", "T", domain=_POS, description="FRC external/equilibrium magnetic-field scale."),
        _spec("B_signed", "T", shape=1, domain=_ANY, description="Signed FRC reversed magnetic field profile."),
        _spec("zeta_ne_ni", domain=_POS, description="FRC electron-to-ion density ratio zeta.", default=1.0),
        _spec("p_peak_frc", "Pa", domain=_POS, description="FRC field-null peak pressure."),
        _spec("G1_frc", domain=_NONNEG, description="FRC normalized first density moment."),
        _spec("G2_frc", domain=_NONNEG, description="FRC normalized squared-density moment."),
        _spec("G_B_frc", domain=_NONNEG, description="FRC normalized mean absolute magnetic-field moment."),
        _spec("p_shape_frc", domain=_POS, description="FRC superellipse exponent.", default=2.0),
        _spec("m_shape_frc", domain=_POS, description="FRC Ma-Xie axial shape exponent.", default=2.0),
        _spec("phi_p", "Wb", domain=_NONNEG, description="FRC trapped poloidal flux."),
        _spec("eta_plasma", "ohm*m", domain=_POS, description="Plasma resistivity used by reduced FRC diffusion estimates."),
        _spec("tau_eta", "s", domain=_POS, description="FRC resistive flux-diffusion time."),
        _spec("tau_E_over_tau_eta", domain=_NONNEG, description="Energy-confinement to resistive-diffusion time ratio."),
        _spec("rho_ie", "m", domain=_POS, description="FRC effective ion gyroradius entering the kinetic s parameter."),
        _spec("s_bar", domain=_NONNEG, description="FRC kinetic parameter r_s/rho_ie."),
        _spec("s_over_E", domain=_NONNEG, description="FRC s/E tilt-stability proxy."),
        _spec("D_classical", "m^2/s", domain=_NONNEG, description="FRC classical diffusion coefficient."),
        _spec("D_Bohm", "m^2/s", domain=_NONNEG, description="FRC Bohm diffusion coefficient."),
        _spec("tau_classical", "s", domain=_POS, description="FRC classical confinement-time bracket."),
        _spec("tau_Bohm", "s", domain=_POS, description="FRC Bohm confinement-time bracket."),

        # Dipole.
        _spec("L_shell", "m", shape=1, domain=_POS, description="Dipole equatorial shell radius coordinate.",
              default_relation=("Point-dipole shell coordinate",)),
        _spec("U", "m/T", shape=1, domain=_POS, description="Dipole flux-tube specific volume |dV/dpsi|."),
        _spec("U_ratio", domain=_POS, description="Outer-to-inner dipole flux-tube-volume ratio."),
        _spec("L_in", "m", domain=_POS, description="Inner dipole plasma shell radius."),
        _spec("L_out", "m", domain=_POS, description="Outer dipole plasma shell radius."),
        _spec("r_ring", "m", domain=_POS, description="Finite dipole current-ring radius."),
        _spec("B_ring", "T", domain=_POS, description="Reference field used to calibrate a finite current ring."),
        _spec("I_ring", "A", domain=_NONNEG, description="Finite dipole current-ring current."),
        _spec("beta_in", domain=_NONNEG, description="Dipole beta on the inner shell."),
        _spec("beta_out", domain=_NONNEG, description="Dipole beta on the outer shell."),
        _spec("n0_tau_E", "s/m^3", domain=_NONNEG, description="Dipole central-density confinement product."),
        _spec("R_wall_proxy", "m", domain=_POS, description="Spherical-wall proxy radius used by the reduced dipole model."),

        # Stellarator geometry quantities used by VSC closures.
        _spec("N_fp", domain=_POS, aliases=("number_of_field_periods",), description="Number of stellarator field periods."),
        _spec("a_vol", "m", domain=_POS, description="Volume-equivalent stellarator minor radius."),
        _spec("iota_2_3", domain=_POS, description="Rotational transform at normalized radius 2/3."),
        _spec("eta_bar", "1/m", domain=_ANY, description="Near-axis magnetic-field-strength parameter."),
        _spec("n_la_geom", "1/m^3", domain=_NONNEG, description="Geometry-derived stellarator line-averaged density."),

        # Proton-boron reaction quantities. Reactivity is supplied until a cited
        # provider is added; the VSC paper does not reproduce its fit coefficients.
        _spec("n_B11", "1/m^3", shape=1, domain=_NONNEG, description="Boron-11 ion density profile."),
        _spec("f_B11", domain=_UNIT, description="Boron-11 fraction of the modeled fuel-ion population."),
        _spec("sigmav_pB11", "m^3/s", shape=1, domain=_NONNEG, description="Maxwellian p-B11 fusion reactivity."),
        _spec("Rr_pB11", "1/s", domain=_NONNEG, description="Volume-integrated p-B11 reaction rate."),
        _spec("P_fus_pB11", "W", domain=_NONNEG, description="p-B11 fusion power."),
    ]

    present = {spec.name for spec in specs}
    specs.extend(spec for spec in additions if spec.name not in present)
    augmented = VariableRegistry(specs, rel_tol_default=base.rel_tol_default,
                                 profile_size_default=base.profile_size_default)
    base._specs = augmented._specs
    base._alias_to_name = augmented._alias_to_name
    return base
