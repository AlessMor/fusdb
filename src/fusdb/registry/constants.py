"""Shared physical constants for FusDB."""

from __future__ import annotations

# Define base conversion factors once so derived reaction energies stay consistent.
ATOMIC_MASS_UNIT_KG = 1.66053906660e-27
ELECTRON_CHARGE_C = 1.602176634e-19
KEV_TO_J = 1.0e3 * ELECTRON_CHARGE_C
MEV_TO_J = 1.0e6 * ELECTRON_CHARGE_C

# Keep package-wide physical constants in one import location.
MU0 = 1.25663706212e-6
EPSILON0 = 8.8541878128e-12  # vacuum permittivity [F/m] (CODATA)
ELECTRON_MASS_KG = 9.1093837015e-31  # electron mass [kg] (CODATA)
PROTON_MASS_KG = 1.67262192369e-27  # proton mass [kg] (CODATA)

# The per-branch reaction energies (DT_ALPHA_ENERGY_J and friends) now live in
# reaction_registry.py, derived from reactions.yaml alongside the stoichiometry
# they belong to.  They are still re-exported from `fusdb.registry`.
