"""Field-reversed-configuration kinetic tilt-stability proxies.

Adapted from Wang et al. (2026), arXiv:2607.11208 ("VSC" reduced multi-configuration model).
VSC section 3.3.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fusdb.relation import relation


@relation(name="FRC kinetic parameter", tags=("frc", "stability"), outputs="s_bar")
def frc_kinetic_parameter(r_s: Any, rho_ie: Any) -> Any:
    """VSC Eq. (75)."""
    return np.asarray(r_s) / np.asarray(rho_ie)


@relation(name="FRC kinetic tilt parameter", tags=("frc", "stability"), outputs="s_over_E")
def frc_kinetic_tilt_parameter(s_bar: Any, E_frc: Any) -> Any:
    return np.asarray(s_bar) / np.asarray(E_frc)
