from __future__ import annotations

import sys

from .relation_class import Relation
from .relation_class import relation
from .relationsystem_class import RelationSystem
from .reactor_class import Reactor
from .popcon_class import Popcon
from .variable_class import Variable
from .utils import integrate_profile_over_volume
from .registry import (
    KEV_TO_J,
    MU0,
    RESERVED_KEYS,
    load_allowed_reactions,
    load_allowed_species,
    load_allowed_tags,
    load_allowed_variables,
)


def _export_relations_at_package_level() -> tuple[str, ...]:
    """Expose registered relation objects as ``fusdb.<name>``.

    Returns:
        Sorted relation names exported at package level.
    """
    from . import relations

    # Ensure all relation modules are imported before scanning modules.
    relations.import_relations()

    exported: list[str] = []
    package_globals = globals()

    # Walk imported relation modules and expose relation objects directly.
    for module_name, module in tuple(sys.modules.items()):
        if module is None or not module_name.startswith("fusdb.relations."):
            continue

        for attr_name, attr_value in vars(module).items():
            if attr_name.startswith("_"):
                continue
            if not isinstance(attr_value, Relation):
                continue
            if attr_name in package_globals and package_globals[attr_name] is not attr_value:
                continue
            package_globals[attr_name] = attr_value
            exported.append(attr_name)

    return tuple(sorted(set(exported)))


_EXPORTED_RELATIONS = _export_relations_at_package_level()
