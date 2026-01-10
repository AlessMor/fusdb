from .relation_class import PRIORITY_EXPLICIT, PRIORITY_RELATION, PRIORITY_STRICT, Relation, RelationSystem
from .reactors_class import Reactor
from .confinement.plasma_stored_energy import KEV_TO_J
from .plasma_pressure.beta import MU0
from .loader import (
    find_reactor_dirs,
    load_all_reactors,
    load_reactor_yaml,
    reactor_table,
)
from .reactor_util import (
    ALLOWED_CONFINEMENT_MODES,
    ALLOWED_REACTOR_FAMILIES,
    ALLOWED_REACTOR_CONFIGURATIONS,
    ALLOWED_RELATION_DOMAINS,
    ALLOWED_VARIABLES,
    ARTIFACT_FIELDS,
    DEFAULT_GEOMETRY_VALUES,
    OPTIONAL_METADATA_FIELDS,
    REQUIRED_FIELDS,
    RESERVED_KEYS,
)

PLASMA_RELATIONS = tuple(
    rel for _tags, rel in Reactor.get_relations_with_tags(("plasma",), require_all=True)
)

__all__ = [
    "Reactor",
    "load_reactor_yaml",
    "find_reactor_dirs",
    "load_all_reactors",
    "reactor_table",
    "Relation",
    "RelationSystem",
    "PRIORITY_EXPLICIT",
    "PRIORITY_RELATION",
    "PRIORITY_STRICT",
    "PLASMA_RELATIONS",
    "KEV_TO_J",
    "MU0",
    "ALLOWED_CONFINEMENT_MODES",
    "ALLOWED_REACTOR_FAMILIES",
    "ALLOWED_REACTOR_CONFIGURATIONS",
    "ALLOWED_RELATION_DOMAINS",
    "ALLOWED_VARIABLES",
    "ARTIFACT_FIELDS",
    "DEFAULT_GEOMETRY_VALUES",
    "OPTIONAL_METADATA_FIELDS",
    "REQUIRED_FIELDS",
    "RESERVED_KEYS",
]
