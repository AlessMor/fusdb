"""Ohmic confinement scaling relations."""

from fusdb.relation import relation


@relation(
    name="neo_alcator_confinement_time",
    tags=("confinement", "ohmic_mode", "regime_default"),
    outputs="tau_E",
)
def neo_alcator_confinement_time(
    n_avg: float, rminor: float, rmajor: float, qstar: float
) -> float:
    """Calculate the Nec-Alcator(NA) OH scaling confinement time
    Adapted from PROCESS; see README.md section "Third-party Notices".

    Parameters
    ----------
    dene20 :
        Volume averaged electron density in units of 10**20 m**-3
    rminor :
        Plasma minor radius [m]
    rmajor :
        Plasma major radius [m]
    qstar :
        Equivalent cylindrical edge safety factor

    Returns
    -------
    :
        float: Neo-Alcator confinement time [s]


    References
    ----------
        - N. A. Uckan, International Atomic Energy Agency, Vienna (Austria) and
          ITER Physics Group, "ITER physics design guidelines: 1989", no. No. 10.
          Feb. 1990.
    """
    # Convert canonical FusDB inputs to PROCESS scaling units.
    dene20 = n_avg / 1.0e20
    return 0.07e0 * dene20 * rminor * rmajor * rmajor * qstar


@relation(
    name="cfspopcon_loc_confinement_time",
    tags=("confinement", "ohmic_mode", "loc"),
    outputs="tau_E",
)
def cfspopcon_loc_confinement_time(
    H98_y2: float,
    n_avg: float,
    qstar: float,
    kappa: float,
    eps: float,
    R: float,
) -> float:
    """Calculate the LOC confinement time scaling.
    Adapted from cfspopcon; see README.md section "Third-party Notices".

    Reference
    ---------
        - Linear Ohmic Confinement, from page 2 of J.E. Rice et al 2020
          Nucl. Fusion 60 105001, 'Understanding LOC/SOC phenomenology in
          tokamaks'

    Notes
    -----
        - Regime: LOC
    """
    return (
        H98_y2
        * 0.0070
        * (n_avg / 1.0e19) ** 1.0
        * qstar**1.0
        * kappa**0.5
        * eps**1.0
        * R**3.0
    )
