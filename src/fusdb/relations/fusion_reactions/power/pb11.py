"""p-B11 fusion power.

Adapted from Wang et al. (2026), arXiv:2607.11208 ("VSC" reduced multi-configuration model).
The VSC paper identifies Nevins-Swain and Sikora-Weller as selectable reactivity
models but does not publish their complete coefficients, so ``sigmav_pB11``
stays an ordinary supplied/profile variable and only the source-independent
accounting lives here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fusdb.relation import relation
from fusdb.registry import MEV_TO_J


_PB11_ENERGY_MEV = 8.68


@relation(name="p-B11 fusion power", tags=("fusion_power",), outputs="P_fus_pB11")
def fusion_power_pb11(Rr_pB11: Any) -> Any:
    """p+B11 -> 3 alpha total released energy, 8.68 MeV per reaction."""
    return np.asarray(Rr_pB11) * _PB11_ENERGY_MEV * MEV_TO_J
