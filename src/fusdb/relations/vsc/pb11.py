"""p-B11 reaction accounting used by the VSC advanced-fuel branch.

The VSC paper identifies Nevins-Swain and Sikora-Weller as selectable reactivity
models but does not publish their complete coefficients/data.  FusDB therefore
keeps ``sigmav_pB11`` as an ordinary supplied/profile variable here and adds the
source-independent reaction-rate and power relations.  A cited reactivity
provider can be added separately without changing this accounting.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fusdb.numerics import volume_average
from fusdb.relation import relation
from fusdb.registry import MEV_TO_J


_PB11_ENERGY_MEV = 8.68


@relation(name="p-B11 reaction rate", tags=("fusion_power",), outputs="Rr_pB11")
def reaction_rate_pb11(
    n_p: Any,
    n_B11: Any,
    sigmav_pB11: Any,
    V_p: Any,
    rho: Any,
    w_V: Any = None,
) -> Any:
    """Volume-integrated p + B11 reaction rate."""
    local = np.asarray(n_p) * np.asarray(n_B11) * np.asarray(sigmav_pB11)
    return np.asarray(V_p) * volume_average(local, rho, weight=w_V)


@relation(name="p-B11 fusion power", tags=("fusion_power",), outputs="P_fus_pB11")
def fusion_power_pb11(Rr_pB11: Any) -> Any:
    """p+B11 -> 3 alpha total released energy, 8.68 MeV per reaction."""
    return np.asarray(Rr_pB11) * _PB11_ENERGY_MEV * MEV_TO_J
