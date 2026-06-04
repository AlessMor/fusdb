"""Ordered evaluation mode."""

from __future__ import annotations

from typing import Any, Iterable


def run(system: Any, order: Iterable[Any] | None = None, *, passes: int = 1) -> dict[str, Any]:
    """Evaluate relations in a user-given order.

    Args:
        system: RelationSystem instance.
        order: Optional explicit relation order.
        passes: Number of passes through the relation sequence.

    Returns:
        Result dictionary.
    """
    return system.ordered_evaluate(order=order, passes=passes)
