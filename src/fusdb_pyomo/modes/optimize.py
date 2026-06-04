"""Optimize mode."""

from __future__ import annotations

from typing import Any, Callable


def run(
    system: Any,
    *,
    objective: str | Callable[[dict[str, Any]], Any] | None = None,
    sense: str = "minimize",
    max_nfev: int | None = None,
    movement_weight: float = 0.1,
    relation_weight: float = 10.0,
    objective_weight: float = 1.0,
    workers: Any = None,
    verbose: int = 0,
) -> dict[str, Any]:
    """Solve active relations while nudging a scalar objective.

    Args:
        system: RelationSystem instance.
        objective: Variable name or callable accepting the current value mapping.
        sense: ``minimize`` or ``maximize``.
        max_nfev: Optional maximum SciPy function evaluations.
        movement_weight: Weight for movement-from-reference residuals.
        relation_weight: Weight for relation residuals during the solve.
        objective_weight: Weight for the objective residual.
        workers: Optional map-like callable for finite-difference parallelism.
        verbose: SciPy verbosity level.

    Returns:
        Result dictionary.
    """
    return system.solve_mode(
        "optimize",
        objective=objective,
        sense=sense,
        max_nfev=max_nfev,
        movement_weight=movement_weight,
        relation_weight=relation_weight,
        objective_weight=objective_weight,
        workers=workers,
        verbose=verbose,
    )
