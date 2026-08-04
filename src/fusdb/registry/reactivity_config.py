"""Reactivity table configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True, slots=True)
class ReactivityTableConfig:
    """Configuration for reactivity-table lookup and interpolation."""

    energy_grid_start_log10_kev: float
    energy_grid_stop_log10_kev: float
    energy_grid_num_points: int
    allowed_interpolation_kinds: tuple[str, ...]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ReactivityTableConfig":
        document_path = Path(path)
        with document_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, dict):
            raise TypeError("reactivity_tables.yaml must contain a mapping.")

        settings = raw.get("settings", {}) or {}
        if not isinstance(settings, dict):
            raise TypeError("reactivity_tables.yaml settings must be a mapping.")
        energy_grid = settings.get("energy_grid", {}) or {}
        if not isinstance(energy_grid, dict):
            raise TypeError("reactivity_tables.yaml settings.energy_grid must be a mapping.")

        return cls(
            energy_grid_start_log10_kev=float(energy_grid.get("start_log10_kev", 0.0)),
            energy_grid_stop_log10_kev=float(energy_grid.get("stop_log10_kev", 5.0)),
            energy_grid_num_points=int(energy_grid.get("num_points", 1000)),
            allowed_interpolation_kinds=tuple(
                str(item) for item in (settings.get("allowed_interpolation_kinds", ()) or ())
            ),
        )


_DEFAULT_PATH = Path(__file__).with_name("reactivity_tables.yaml")
REACTIVITY_TABLES = ReactivityTableConfig.from_yaml(_DEFAULT_PATH)
