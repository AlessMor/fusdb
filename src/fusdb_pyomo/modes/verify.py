"""Verify mode."""

from __future__ import annotations

from typing import Any


def run(system: Any) -> dict[str, Any]:
    """Evaluate active relations without changing variables.

    Args:
        system: RelationSystem instance.

    Returns:
        Result dictionary with relation and variable statuses.
    """
    return system.verify_current()
