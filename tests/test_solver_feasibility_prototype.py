"""Independent prototype of a feasibility-only reconcile kernel.

This intentionally does not call fusdb.modes.reconcile.run.  It uses only the
compiled RelationSystem public numerical protocol and scipy least_squares, so
we can validate the proposed solver semantics before changing production code.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from fusdb import Reactor
from fusdb.modes.verify import verify_values


ROOT = Path(__file__).parents[1]
ARC_YAML = ROOT / "reactors" / "ARC_V0" / "reactor.yaml"
DEMO_YAML = ROOT / "reactors" / "DEMO_2022" / "reactor.yaml"


def _system(path: Path, regime: str):
    reactor = Reactor.from_yaml(path)
    candidate = reactor._clone_for_regime(regime, include_guards=False)
    system = candidate.relation_system()
    system.compile()
    return system


def _feasibility_only(system, *, max_nfev: int = 600):
    """Solve only enforced physics/domain residuals; no movement objective."""
    x0, lower, upper = system.pack()
    if x0.size == 0:
        certificate = verify_values(system, system.solver_values(), complete=True)
        return certificate, {"nfev": 0, "residual_calls": 0}

    layout = system.residual_layout(system.unpack(x0), include_movement=False)
    calls = 0

    def residual(x: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        values = system.unpack(x)
        relation = system.layout_relation_rows(values, layout)
        domain = system.layout_domain_rows(values, layout)
        if domain.size:
            return np.concatenate((relation, domain))
        return relation

    kwargs = {
        "bounds": (lower, upper),
        "method": "trf",
        "max_nfev": max_nfev,
        "ftol": 1.0e-12,
        "gtol": 1.0e-12,
        "xtol": 1.0e-12,
    }
    sparsity = system.build_jac_sparsity(layout)
    if sparsity is not None and sparsity.shape == (int(layout["size"]), int(x0.size)):
        kwargs["jac_sparsity"] = sparsity

    solved = least_squares(residual, x0, **kwargs)
    certificate = verify_values(system, system.unpack(solved.x), complete=False)
    return certificate, {"nfev": int(solved.nfev), "residual_calls": calls}


def test_feasibility_only_solves_arc_without_movement_penalties() -> None:
    system = _system(ARC_YAML, "i_mode")
    certificate, stats = _feasibility_only(system)
    assert certificate["verified"], (
        certificate.get("errors"),
        certificate.get("failed_relations"),
        stats,
    )


def test_feasibility_only_solves_demo_without_movement_penalties() -> None:
    system = _system(DEMO_YAML, "h_mode")
    certificate, stats = _feasibility_only(system)
    assert certificate["verified"], (
        certificate.get("errors"),
        certificate.get("failed_relations"),
        stats,
    )


def test_solver_success_is_not_acceptance_criterion() -> None:
    # Adversarial numerical contract: least_squares can terminate successfully
    # at a nonzero compromise for inconsistent equations.  fusdb must therefore
    # continue to use verify/certification as the authority after any refactor.
    def residual(x: np.ndarray) -> np.ndarray:
        return np.array((x[0] - 1.0, x[0] - 2.0))

    result = least_squares(residual, np.array((1.5,)))
    assert result.success
    assert np.max(np.abs(residual(result.x))) > 0.1
