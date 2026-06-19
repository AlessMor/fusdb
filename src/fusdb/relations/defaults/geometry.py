"""Default geometry relation helpers."""

from fusdb import relation


@relation(
    name="Default squareness",
    tags=("default", "geometry", "tokamak", "stellarator", "mirror"),
    outputs="squareness",
)
def default_squareness() -> float:
    """Fallback plasma squareness for shape geometry.

    Used only when no value is supplied.  Zero squareness is the standard
    D-shaped cross-section assumption.
    """
    return 0.0
