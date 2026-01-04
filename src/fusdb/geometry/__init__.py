from fusdb.relations_util import REL_TOL_DEFAULT
from fusdb import reactors_class as _rc

# Import plasma_geometry to register decorated relations when geometry is imported first.
from . import plasma_geometry  # noqa: F401
from .plasma_geometry import plasma_surface_area, plasma_volume

__all__ = [
    "REL_TOL_DEFAULT",
    "relations",
    "plasma_volume",
    "plasma_surface_area",
]


def relations(
    tags: str | tuple[str, ...] = ("geometry",),
    *,
    require_all: bool = True,
    exclude: tuple[str, ...] | None = None,
):
    return _rc.relations_for(tags, require_all=require_all, exclude=exclude)
