"""Reconcile mode."""

from __future__ import annotations

from typing import Any


def run(
    system: Any,
    *,
    max_nfev: int | None = None,
    movement_weight: float = 0.1,
    relation_weight: float = 10.0,
    workers: Any = None,
    verbose: int = 0,
) -> dict[str, Any]:
    """Move non-fixed variables to satisfy active relations.

    Args:
        system: RelationSystem instance.
        max_nfev: Optional maximum SciPy function evaluations.
        movement_weight: Weight for movement-from-reference residuals.
        relation_weight: Weight for relation residuals during the solve.
        workers: Optional map-like callable for finite-difference parallelism.
        verbose: SciPy verbosity level.

    Returns:
        Result dictionary.
    """
    return system.solve_mode(
        "reconcile",
        max_nfev=max_nfev,
        movement_weight=movement_weight,
        relation_weight=relation_weight,
        workers=workers,
        verbose=verbose,
    )
