"""Shared physical constants for FusDB."""

from __future__ import annotations

# Define base conversion factors once so derived reaction energies stay consistent.
ATOMIC_MASS_UNIT_KG = 1.66053906660e-27
ELECTRON_CHARGE_C = 1.602176634e-19
KEV_TO_J = 1.0e3 * ELECTRON_CHARGE_C
MEV_TO_J = 1.0e6 * ELECTRON_CHARGE_C

# Keep package-wide physical constants in one import location.
MU0 = 1.25663706212e-6

# Preserve the package's existing reference energies exactly.
DT_REACTION_ENERGY_J = 2.8198311e-12
DT_ALPHA_ENERGY_J = 5.6076218e-13
DT_N_ENERGY_J = 2.2590689e-12
DD_T_ENERGY_J = 1.6182e-13
DD_HE3_ENERGY_J = 1.3137848e-13
DD_P_ENERGY_J = 4.8385734e-13
DD_N_ENERGY_J = 3.9253334e-13
DHE3_ALPHA_ENERGY_J = 5.7678388e-13
DHE3_P_ENERGY_J = 2.3551998e-12
HE3HE3_ALPHA_ENERGY_J = 3.67 * MEV_TO_J
HE3HE3_P_ENERGY_J = 9.19 * MEV_TO_J
TT_ALPHA_ENERGY_J = 3.76 * MEV_TO_J
TT_N_ENERGY_J = 7.54 * MEV_TO_J
THE3_D_ALPHA_ENERGY_J = 4.8 * MEV_TO_J
THE3_D_D_ENERGY_J = 9.5 * MEV_TO_J
THE3_NP_ALPHA_ENERGY_J = (12.1 / 6.0) * MEV_TO_J
THE3_NP_P_ENERGY_J = (12.1 * 5.0 / 12.0) * MEV_TO_J
THE3_NP_N_ENERGY_J = (12.1 * 5.0 / 12.0) * MEV_TO_J
# TODO: add references for energies
