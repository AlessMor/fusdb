from fusdb.relation_util import REL_TOL_DEFAULT
from fusdb.reactor_util import relations_for

# Import plasma_geometry to register decorated relations when geometry is imported first.
from . import plasma_geometry  # noqa: F401
from .plasma_geometry import plasma_surface_area, plasma_volume

# TODO: check if this function can be deleted in favor of simpler import
def relations(
    tags: str | tuple[str, ...] = ("geometry",),
    *,
    require_all: bool = True,
    exclude: tuple[str, ...] | None = None,
):
    """Return geometry relations matching the requested tags."""
    return relations_for(tags, require_all=require_all, exclude=exclude)
