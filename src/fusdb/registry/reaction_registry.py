"""Fusion reaction metadata loaded from ``reactions.yaml``."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import yaml

from .constants import MEV_TO_J


def _energy_j(entry: Mapping[str, float], where: str) -> float:
    """Return an energy in joules from a ``{J: ...}`` or ``{MeV: ...}`` entry."""
    if "J" in entry:
        return float(entry["J"])
    if "MeV" in entry:
        return float(entry["MeV"]) * MEV_TO_J
    raise ValueError(f"Energy entry for {where} must give either 'J' or 'MeV'.")


@dataclass(frozen=True, slots=True)
class ReactionSpec:
    """One fusion reaction channel, including its branching."""

    key: str
    reactants: tuple[str, ...]
    products: tuple[str, ...]
    energies_j: Mapping[str, float]
    total_energy_j: float | None = None

    @property
    def like_reactants(self) -> bool:
        return len(set(self.reactants)) == 1

    def stoichiometry(self, species: str) -> int:
        return self.products.count(species) - self.reactants.count(species)


def _load_reactions(path: str | Path) -> Mapping[str, ReactionSpec]:
    """Load and validate reaction metadata as an immutable mapping."""
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    specs: dict[str, ReactionSpec] = {}
    for key, entry in raw.items():
        entry = entry or {}
        reactants = tuple(str(item) for item in entry.get("reactants", ()))
        products = tuple(str(item) for item in entry.get("products", ()))
        if len(reactants) != 2:
            raise ValueError(f"Reaction {key!r} must have exactly two reactants.")
        if not products:
            raise ValueError(f"Reaction {key!r} must have at least one product.")
        energies = {
            str(species): _energy_j(values or {}, f"{key}/{species}")
            for species, values in (entry.get("energies", {}) or {}).items()
        }
        unknown = set(energies) - set(products)
        if unknown:
            raise ValueError(f"Reaction {key!r} gives energies for non-products {sorted(unknown)}.")
        total = entry.get("total_energy")
        specs[str(key)] = ReactionSpec(
            key=str(key),
            reactants=reactants,
            products=products,
            energies_j=MappingProxyType(energies),
            total_energy_j=_energy_j(total, f"{key}/total") if total else None,
        )
    return MappingProxyType(specs)


REACTIONS = _load_reactions(Path(__file__).with_name("reactions.yaml"))

# Existing public energy constants, derived from the metadata above.
DT_ALPHA_ENERGY_J = REACTIONS["DT"].energies_j["He4"]
DT_N_ENERGY_J = REACTIONS["DT"].energies_j["n"]
DT_REACTION_ENERGY_J = REACTIONS["DT"].total_energy_j
DD_HE3_ENERGY_J = REACTIONS["DDn"].energies_j["He3"]
DD_N_ENERGY_J = REACTIONS["DDn"].energies_j["n"]
DD_T_ENERGY_J = REACTIONS["DDp"].energies_j["T"]
DD_P_ENERGY_J = REACTIONS["DDp"].energies_j["p"]
DHE3_ALPHA_ENERGY_J = REACTIONS["DHe3"].energies_j["He4"]
DHE3_P_ENERGY_J = REACTIONS["DHe3"].energies_j["p"]
TT_ALPHA_ENERGY_J = REACTIONS["TT"].energies_j["He4"]
TT_N_ENERGY_J = REACTIONS["TT"].energies_j["n"]
HE3HE3_ALPHA_ENERGY_J = REACTIONS["He3He3"].energies_j["He4"]
HE3HE3_P_ENERGY_J = REACTIONS["He3He3"].energies_j["p"]
THE3_D_ALPHA_ENERGY_J = REACTIONS["THe3_D"].energies_j["He4"]
THE3_D_D_ENERGY_J = REACTIONS["THe3_D"].energies_j["D"]
THE3_NP_ALPHA_ENERGY_J = REACTIONS["THe3_np"].energies_j["He4"]
THE3_NP_P_ENERGY_J = REACTIONS["THe3_np"].energies_j["p"]
THE3_NP_N_ENERGY_J = REACTIONS["THe3_np"].energies_j["n"]
