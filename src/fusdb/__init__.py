from __future__ import annotations

import sys

from .relation_class import Relation
from .relation_util import relation
from .relationsystem_class import RelationSystem
from .reactor_class import Reactor
from .reactor_util import reactor_from_yaml
from .variable_class import Variable, Variable0D, Variable1D
from .variable_util import make_variable
from .utils import integrate_profile_over_volume, compare_plasma_volume_with_integrated_dv
from .registry import (
    KEV_TO_J,
    MU0,
    ALLOWED_CONFINEMENT_MODES,
    ALLOWED_REACTOR_FAMILIES,
    ALLOWED_REACTOR_CONFIGURATIONS,
    ALLOWED_SOLVING_ORDER,
    ALLOWED_VARIABLES,
    OPTIONAL_METADATA_FIELDS,
    REQUIRED_FIELDS,
    RESERVED_KEYS,
)


def _export_relations_at_package_level() -> tuple[str, ...]:
    """Expose relation objects and public functions as ``fusdb.<name>``."""
    from . import relations

    relations.import_relations()

    exported: list[str] = []
    package_globals = globals()

    for module_name, module in tuple(sys.modules.items()):
        if module is None or not module_name.startswith("fusdb.relations."):
            continue

        for attr_name, attr_value in vars(module).items():
            if attr_name.startswith("_"):
                continue
            is_relation = isinstance(attr_value, Relation)
            is_own_callable = (
                callable(attr_value)
                and not is_relation
                and getattr(attr_value, "__module__", None) == module_name
            )
            if not is_relation and not is_own_callable:
                continue
            if attr_name in package_globals and package_globals[attr_name] is not attr_value:
                continue
            package_globals[attr_name] = attr_value
            exported.append(attr_name)

    return tuple(sorted(set(exported)))


_EXPORTED_RELATIONS = _export_relations_at_package_level()


__all__ = [
    "Relation",
    "relation",
    "RelationSystem",
    "Reactor",
    "reactor_from_yaml",
    "Variable",
    "Variable0D",
    "Variable1D",
    "make_variable",
    "integrate_profile_over_volume",
    "compare_plasma_volume_with_integrated_dv",
    "KEV_TO_J",
    "MU0",
    "ALLOWED_CONFINEMENT_MODES",
    "ALLOWED_REACTOR_FAMILIES",
    "ALLOWED_REACTOR_CONFIGURATIONS",
    "ALLOWED_SOLVING_ORDER",
    "ALLOWED_VARIABLES",
    "OPTIONAL_METADATA_FIELDS",
    "REQUIRED_FIELDS",
    "RESERVED_KEYS",
    *_EXPORTED_RELATIONS,
]
