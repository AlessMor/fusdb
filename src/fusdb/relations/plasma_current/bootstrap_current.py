# TODO(high): add from cfspopcon
    # bootstrap fraction (Gi et al. 2014) or other methods from PROCESS
    # Inductive plasma current = I_p * (1 - bootstrap fraction)
#
# cfspopcon ports (UNDECORATED scaffolds), source:
# cfspopcon/formulas/plasma_current/bootstrap_fraction.py
# Review formula + variable name/unit mapping (e.g. plasma_current->I_p,
# inverse_aspect_ratio->eps, beta_poloidal->beta_p, z_effective->Z_eff,
# *_density_peaking->density_peaking, temperature_peaking) then add @relation to activate.


# TODO(cfspopcon): activate as a fusdb relation (bootstrap_fraction output).
def calc_bootstrap_fraction(
    ion_density_peaking,
    electron_density_peaking,
    temperature_peaking,
    z_effective,
    q_star,
    inverse_aspect_ratio,
    beta_poloidal,
):
    """cfspopcon: bootstrap current fraction (Gi 2014 scaling, assumes q0=1)."""
    nu_n = (ion_density_peaking + electron_density_peaking) / 2

    bootstrap_fraction = 0.474 * (
        (temperature_peaking - 1.0 + nu_n - 1.0) ** 0.974
        * (temperature_peaking - 1.0) ** -0.416
        * z_effective**0.178
        * q_star**-0.133
        * inverse_aspect_ratio**0.4
        * beta_poloidal
    )

    return bootstrap_fraction


# TODO(cfspopcon): activate as a fusdb relation (inductive_plasma_current output).
def calc_inductive_plasma_current(plasma_current, bootstrap_fraction):
    """cfspopcon: inductive_plasma_current = plasma_current * (1 - bootstrap_fraction)."""
    return plasma_current * (1.0 - bootstrap_fraction)
