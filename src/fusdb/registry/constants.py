"""Shared physical constants and reaction energies."""

MEV_TO_J = 1.602176634e-13  # 1 MeV in joules.
KEV_TO_J = 1.602176634e-16  # 1 keV in joules.
MU0 = 4.0e-7 * 3.141592653589793  # Vacuum permeability (H/m).
DT_REACTION_ENERGY_J = 17.6 * MEV_TO_J
DT_ALPHA_ENERGY_J = 3.5 * MEV_TO_J
DT_N_ENERGY_J = DT_REACTION_ENERGY_J - DT_ALPHA_ENERGY_J
DD_T_ENERGY_J = 1.01 * MEV_TO_J
DD_HE3_ENERGY_J = 0.82 * MEV_TO_J
DD_P_ENERGY_J = 3.02 * MEV_TO_J
DD_N_ENERGY_J = 2.45 * MEV_TO_J
DHE3_ALPHA_ENERGY_J = 3.6 * MEV_TO_J
DHE3_P_ENERGY_J = 14.7 * MEV_TO_J
TT_REACTION_ENERGY_J = 11.3 * MEV_TO_J
# NOTE: could be improved by doing as in PROCESS: (mass difference)*c^2