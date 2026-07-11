"""AMJUEL H.4 molecular-assisted recombination chain fits via H2+."""

from pathlib import Path
from typing import Any

from fusdb.relation import relation
from fusdb.relations.atomic_physics._amjuel import evaluate_amjuel_h4_rate

_DATA_DIR = Path(__file__).resolve().parent

@relation(
    name='AMJUEL H.4 3.2.3r H2 MAR via H2+ rate',
    tags=("atomic_physics",),
    outputs='H2_mar_via_h2_plus_rate',
    constraints=(
        'n_e_edge >= 1e+14',
        'n_e_edge <= 1e+22',
        'T_edge > 0.0',
    ),
)
def amjuel_h4_3_2_3r_h2_mar_via_h2_rate(n_e_edge: Any, T_edge: Any) -> Any:
    """Return H2_mar_via_h2_plus_rate from AMJUEL H.4 subsection 4.49.

    Reaction: 3.2.3r p + H2 (+e) → H + H + H (MAR via H2+)
    Inputs are n_e_edge [m^-3] and T_edge [keV]; AMJUEL H.4 uses density in 1e8 cm^-3
    and temperature in eV internally. The returned rate coefficient is converted from cm^3/s to m^3/s.
    Fit error (AMJUEL): max 12.2399 %, mean 5.5197 %.

    Source: AMJUEL H.4 coefficient fit.
    AMJUEL comments:
    - 2 , cold H2)
    - H2 multi-step model, MAR rate coefﬁcient cm3=s. Data: Sawada/Fujimoto [ 7] H2(v = 0)
    - transported, H +
    - 2 in QSS with H2, EH2 = 0:1 eV
    - H2(v  1) is also in QSS with H2(v = 0) . Vibrational distribution P (v) as fct. of Te (= Tp)
    - only (assuming ne = np, so density cancels here).
    - The MAR rate coefﬁcient is a fct. of ne and T (Te = Tp), and must be multiplied with density
    - np to turn it into a collision rate 1=s, and then with nH2(v=0) to turn it into a volumetric reaction
    - rate (cm-3s-1). This is consistent with underlying P (v) only for ne = np.
    """
    return evaluate_amjuel_h4_rate(_DATA_DIR / "amjuel_h4_3_2_3r.yaml", n_e_edge, T_edge)


@relation(
    name='AMJUEL H.4 3.2.3d H2 MAD via H2+ rate',
    tags=("atomic_physics",),
    outputs='H2_mad_via_h2_plus_rate',
    constraints=(
        'n_e_edge >= 1e+14',
        'n_e_edge <= 1e+22',
        'T_edge > 0.0',
    ),
)
def amjuel_h4_3_2_3d_h2_mad_via_h2_rate(n_e_edge: Any, T_edge: Any) -> Any:
    """Return H2_mad_via_h2_plus_rate from AMJUEL H.4 subsection 4.50.

    Reaction: 3.2.3d p + H2 (+e) → p + H + H (+e) (MAD via H2+)
    Inputs are n_e_edge [m^-3] and T_edge [keV]; AMJUEL H.4 uses density in 1e8 cm^-3
    and temperature in eV internally. The returned rate coefficient is converted from cm^3/s to m^3/s.
    Fit error (AMJUEL): max 11.3558 %, mean 5.3396 %.

    Source: AMJUEL H.4 coefficient fit.
    AMJUEL comments:
    - 2 , cold
    - H2)
    - H2 multi-step model, MAD rate coefﬁcient cm3=s,
    - same conditions as for effective MAR rate coefﬁcient: ne = np to remove ne and np dependence
    - in P (v), Te = Tp to remove Tp dependence in P (v), EH2 = 0:1 eV .
    """
    return evaluate_amjuel_h4_rate(_DATA_DIR / "amjuel_h4_3_2_3d.yaml", n_e_edge, T_edge)


@relation(
    name='AMJUEL H.4 3.2.3i H2 MAI via H2+ rate',
    tags=("atomic_physics",),
    outputs='H2_mai_via_h2_plus_rate',
    constraints=(
        'n_e_edge >= 1e+14',
        'n_e_edge <= 1e+22',
        'T_edge > 0.0',
    ),
)
def amjuel_h4_3_2_3i_h2_mai_via_h2_rate(n_e_edge: Any, T_edge: Any) -> Any:
    """Return H2_mai_via_h2_plus_rate from AMJUEL H.4 subsection 4.51.

    Reaction: 3.2.3i p + H2 (+e) → p + p + H + e (+e) (MAI via H2+)
    Inputs are n_e_edge [m^-3] and T_edge [keV]; AMJUEL H.4 uses density in 1e8 cm^-3
    and temperature in eV internally. The returned rate coefficient is converted from cm^3/s to m^3/s.
    Fit error (AMJUEL): max 3.7214 %, mean 1.4562 %.

    Source: AMJUEL H.4 coefficient fit.
    AMJUEL comments:
    - 2 , cold
    - H2)
    - H2 multi-step model, MAI rate coefﬁcient cm3=s, Data: Sawada/Fujimoto ,[ 7]
    - same conditions as for effective MAR rate coefﬁcient: ne = np to remove ne and np dependence
    - in P (v), Te = Tp to remove Tp dependence in P (v), EH2 = 0:1 eV .
    """
    return evaluate_amjuel_h4_rate(_DATA_DIR / "amjuel_h4_3_2_3i.yaml", n_e_edge, T_edge)