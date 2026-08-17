from __future__ import annotations

from pathlib import Path

import numpy as np

from fusdb import Reactor


ARC_YAML = Path(__file__).parents[1] / "reactors" / "ARC_V0" / "reactor.yaml"


def test_arc_reconcile_jacobian_sparsity_matches_live_residual_shape() -> None:
    reactor = Reactor.from_yaml(ARC_YAML)
    # Mirror Reactor._run_with_regime_verification's solve candidate: the
    # reconciliation itself uses the declared confinement tag without guards.
    candidate = reactor._clone_for_regime("i_mode", include_guards=False)
    system = candidate.relation_system().compile()
    x0, _lower, _upper = system.pack()
    values = system.unpack(x0)
    layout = system.residual_layout(values, include_movement=True)
    live_residual = np.concatenate(
        (
            system.layout_relation_rows(values, layout),
            system.layout_domain_rows(values, layout),
            system.layout_movement_rows(values, layout),
        )
    )

    sparsity = system.build_jac_sparsity(layout)

    assert sparsity is not None
    assert live_residual.size == layout["size"]
    assert sparsity.shape == (live_residual.size, x0.size)
