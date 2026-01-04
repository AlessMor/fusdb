"""Confinement and scaling relation helpers."""

# Import scalings to register decorated relations when confinement is imported first.
from . import scalings  # noqa: F401
from fusdb import reactors_class as _rc

__all__ = [
    "relations",
    "relations_by_name",
]


def relations(
    tags: str | tuple[str, ...] = ("confinement",),
    *,
    require_all: bool = True,
    exclude: tuple[str, ...] | None = None,
):
    return _rc.confinement_relations(tags, require_all=require_all, exclude=exclude)


def relations_by_name():
    return _rc.confinement_relations_by_name()
