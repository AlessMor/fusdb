"""Confinement and scaling relation helpers."""

# Import scalings to register decorated relations when confinement is imported first.
from . import scalings  # noqa: F401
from fusdb.reactor_util import relations_for


def relations(
    tags: str | tuple[str, ...] = ("confinement",),
    *,
    require_all: bool = True,
    exclude: tuple[str, ...] | None = None,
):
    """Return confinement relations matching the requested tags."""
    return relations_for(tags, require_all=require_all, exclude=exclude)


def relations_by_name():
    """Map confinement relation outputs to their Relation objects."""
    return {rel.variables[0]: rel for rel in relations_for(("confinement",), require_all=False)}
