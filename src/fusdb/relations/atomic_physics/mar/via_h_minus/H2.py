"""AMJUEL H.4 molecular-assisted recombination chain fits via H-."""

from typing import Any

from fusdb.relation import relation
from fusdb.registry.dataset.evaluation import evaluate_amjuel_h4_rate

@relation(
    name='AMJUEL H.4 2.2.17r H2 MAR via H- rate',
    tags=("atomic_physics",),
    outputs='H2_mar_via_h_minus_rate',
    constraints=(
        'n_e_edge >= 1e+14',
        'n_e_edge <= 1e+22',
        'T_edge > 0.0',
    ),
)
def amjuel_h4_2_2_17r_h2_mar_via_h_rate(n_e_edge: Any, T_edge: Any) -> Any:
    """Return H2_mar_via_h_minus_rate from AMJUEL H.4 subsection 4.54.

    Reaction: 2.2.17r e + H2 (+p) → H + H + H (MAR via H-, cold H2)
    Inputs are n_e_edge [m^-3] and T_edge [keV]; AMJUEL H.4 uses density in 1e8 cm^-3
    and temperature in eV internally. The returned rate coefficient is converted from cm^3/s to m^3/s.

    Source: AMJUEL H.4 coefficient fit.
    AMJUEL comments:
    - H2 multi-step model, intermediate H- condensed MAR rate coefﬁcient cm3=s. Data: Sawada/Fujimoto/Greenland
    - [7] H2(v = 0) transported, H- in QSS with H2, EH2 = 0:1 eV
    - H2(v  1) is also in QSS with H2(v = 0) . Vibrational distribution P (v) as fct. of Te only
    - (assuming ne = np, so density cancels here).
    - The MAR rate coefﬁcient is a fct. of ne and T (Te = Tp), and must be multiplied with density
    - ne to turn it into a collision rate 1=s, and then with nH2(v=0) to turn it into a volumetric reaction
    - rate (cm-3s-1).
    """
    return evaluate_amjuel_h4_rate("polynomialfit_AMJUEL-H4-2.2.17r_H2-mar-via-h-minus", n_e_edge, T_edge)


@relation(
    name='AMJUEL H.4 2.2.17d H2 MAD via H- rate',
    tags=("atomic_physics",),
    outputs='H2_mad_via_h_minus_rate',
    constraints=(
        'n_e_edge >= 1e+14',
        'n_e_edge <= 1e+22',
        'T_edge > 0.0',
    ),
)
def amjuel_h4_2_2_17d_h2_mad_via_h_rate(n_e_edge: Any, T_edge: Any) -> Any:
    """Return H2_mad_via_h_minus_rate from AMJUEL H.4 subsection 4.55.

    Reaction: 2.2.17d e + H2 (+p) → p + H + H (MAD via H-, cold H2)
    Inputs are n_e_edge [m^-3] and T_edge [keV]; AMJUEL H.4 uses density in 1e8 cm^-3
    and temperature in eV internally. The returned rate coefficient is converted from cm^3/s to m^3/s.

    Source: AMJUEL H.4 coefficient fit.
    AMJUEL comments:
    - H2 multi-step model, H - condensed MAD rate coefﬁcient cm3=s,
    - same conditions as for effective MAR rate coefﬁcient: ne = np; Te = Tp to remove np; Tp
    - dependence in second step, EH2 = 0:1 eV .
    """
    return evaluate_amjuel_h4_rate("polynomialfit_AMJUEL-H4-2.2.17d_H2-mad-via-h-minus", n_e_edge, T_edge)