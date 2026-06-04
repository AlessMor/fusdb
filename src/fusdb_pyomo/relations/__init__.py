"""Relation modules grouped by domain."""

from importlib import import_module
from pkgutil import walk_packages


def import_relations() -> None:
    """Import all relation modules so decorators register their relations."""
    for module in walk_packages(__path__, prefix=f"{__name__}."):
        import_module(module.name)
