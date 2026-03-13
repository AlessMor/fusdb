from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import sympy as sp
from numpy import float64
from numpy.typing import NDArray

from fusdb.registry import KEV_TO_J


_ATOMIC_MASS_UNIT_KG = 1.66053906660e-27
_TABLES_DIR = Path(__file__).with_name("tables")
_ENERGY_GRID_KEV = np.logspace(0.0, 5.0, 1000, dtype=float)
_REACTANT_MASSES_U: dict[str, float] = {
    "D": 2.014,
    "T": 3.016,
    "He3": 3.016,
}
_TABLE_METADATA: dict[str, dict[str, str]] = {
    "DT_ENDFB_VIII0": {
        "filename": "DT_xsection_ENDFB-VIII0.txt",
        "projectile": "D",
        "target": "T",
        "symbolic_name": "sigmav_DT_ENDFB_VIII0",
    },
    "DDn_ENDFB_VIII0": {
        "filename": "DDn_xsection_ENDFB-VIII0.txt",
        "projectile": "D",
        "target": "D",
        "symbolic_name": "sigmav_DDn_ENDFB_VIII0",
    },
    "DDp_ENDFB_VIII0": {
        "filename": "DDp_xsection_ENDFB-VIII0.txt",
        "projectile": "D",
        "target": "D",
        "symbolic_name": "sigmav_DDp_ENDFB_VIII0",
    },
    "DHe3_ENDFB_VIII0": {
        "filename": "DHe3_xsection_ENDFB-VIII0.txt",
        "projectile": "D",
        "target": "He3",
        "symbolic_name": "sigmav_DHe3_ENDFB_VIII0",
    },
    "TT_ENDFB_VIII0": {
        "filename": "TT_xsection_ENDFB-VIII0.txt",
        "projectile": "T",
        "target": "T",
        "symbolic_name": "sigmav_TT_ENDFB_VIII0",
    },
    "He3He3_ENDFB_VIII0": {
        "filename": "He3He3_xsection_ENDFB-VIII0.txt",
        "projectile": "He3",
        "target": "He3",
        "symbolic_name": "sigmav_He3He3_ENDFB_VIII0",
    },
    "THe3n_ENDFB_VIII0": {
        "filename": "THe3n_xsection_ENDFB-VIII0.txt",
        "projectile": "T",
        "target": "He3",
        "symbolic_name": "sigmav_THe3_np_ENDFB_VIII0",
    },
    "THe3D_ENDFB_VIII0": {
        "filename": "THe3D_xsection_ENDFB-VIII0.txt",
        "projectile": "T",
        "target": "He3",
        "symbolic_name": "sigmav_THe3_D_ENDFB_VIII0",
    },
}


@dataclass(frozen=True)
class CrossSectionTable:
    """Interpolated cross section data on a shared center-of-mass energy grid."""

    reaction_id: str
    symbolic_name: str
    reduced_mass_kg: float
    energy_grid_keV: NDArray[np.float64]
    cross_section_grid_m2: NDArray[np.float64]

    def symbolic(self, value: sp.Expr) -> sp.Expr:
        """Return a symbolic placeholder for the tabulated reactivity."""
        return sp.Function(self.symbolic_name)(value)


def _read_table_rows(path: Path) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Read numeric two-column rows from a cross-section table."""
    energy_mev: list[float] = []
    cross_section_barn: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped == "//":
                continue
            parts = stripped.split()
            if len(parts) != 2:
                continue
            energy_mev.append(float(parts[0]))
            cross_section_barn.append(float(parts[1]))
    return (
        np.asarray(energy_mev, dtype=float),
        np.asarray(cross_section_barn, dtype=float),
    )


def _center_of_mass_energy_keV(
    incident_energy_mev: NDArray[np.float64],
    *,
    projectile: str,
    target: str,
) -> NDArray[np.float64]:
    """Convert fixed-target incident energy from MeV to center-of-mass keV."""
    m_projectile = _REACTANT_MASSES_U[projectile]
    m_target = _REACTANT_MASSES_U[target]
    return incident_energy_mev * 1.0e3 * m_target / (m_projectile + m_target)


@lru_cache(maxsize=None)
def load_cross_section_table(reaction_id: str) -> CrossSectionTable:
    """Load, interpolate, and cache one cross-section table on the shared grid."""
    metadata = _TABLE_METADATA[reaction_id]
    energy_mev, cross_section_barn = _read_table_rows(_TABLES_DIR / metadata["filename"])
    energy_cm_keV = _center_of_mass_energy_keV(
        energy_mev,
        projectile=metadata["projectile"],
        target=metadata["target"],
    )
    cross_section_grid_m2 = np.interp(
        _ENERGY_GRID_KEV,
        energy_cm_keV,
        cross_section_barn * 1.0e-28,
        left=0.0,
        right=0.0,
    )
    m_projectile = _REACTANT_MASSES_U[metadata["projectile"]]
    m_target = _REACTANT_MASSES_U[metadata["target"]]
    reduced_mass_kg = m_projectile * m_target / (m_projectile + m_target) * _ATOMIC_MASS_UNIT_KG
    return CrossSectionTable(
        reaction_id=reaction_id,
        symbolic_name=metadata["symbolic_name"],
        reduced_mass_kg=reduced_mass_kg,
        energy_grid_keV=_ENERGY_GRID_KEV.copy(),
        cross_section_grid_m2=cross_section_grid_m2,
    )


def tabulated_reactivity(
    reaction_id: str,
    ion_temp_profile: float64 | NDArray[np.float64] | sp.Expr,
) -> float64 | NDArray[np.float64] | sp.Expr:
    """Return a Maxwellian-averaged reactivity from an interpolated cross-section table."""
    table = load_cross_section_table(reaction_id)
    if isinstance(ion_temp_profile, sp.Expr):
        return table.symbolic(ion_temp_profile)

    temperatures = np.asarray(ion_temp_profile, dtype=float)
    is_scalar = temperatures.ndim == 0
    flat_temperatures = temperatures.reshape(-1)
    sigmav = np.zeros_like(flat_temperatures, dtype=float)

    positive = flat_temperatures > 0.0
    if np.any(positive):
        kT = flat_temperatures[positive] * KEV_TO_J
        energy_joule = table.energy_grid_keV * KEV_TO_J
        prefactor = np.sqrt(8.0 / (np.pi * table.reduced_mass_kg)) / (kT ** 1.5)
        integrand = (
            table.cross_section_grid_m2[None, :]
            * energy_joule[None, :]
            * np.exp(-energy_joule[None, :] / kT[:, None])
        )
        sigmav[positive] = prefactor * np.trapezoid(integrand, energy_joule, axis=1)

    reshaped = sigmav.reshape(temperatures.shape)
    if is_scalar:
        return float64(reshaped.item())
    return reshaped.astype(np.float64, copy=False)
