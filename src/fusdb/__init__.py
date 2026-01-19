from .relation_class import Relation, RelationSystem
from .reactor_class import Reactor
from .registry.constants import KEV_TO_J, MU0
from .loader import (
    find_reactor_dirs,
    load_all_reactors,
    load_reactor_yaml,
)
from .reactor_util import (
    ALLOWED_CONFINEMENT_MODES,
    ALLOWED_REACTOR_FAMILIES,
    ALLOWED_REACTOR_CONFIGURATIONS,
    ALLOWED_RELATION_DOMAINS,
    ALLOWED_VARIABLES,
    DEFAULT_GEOMETRY_VALUES,
    OPTIONAL_METADATA_FIELDS,
    REQUIRED_FIELDS,
    RESERVED_KEYS,
)

PLASMA_RELATIONS = tuple(
    rel for _tags, rel in Reactor.get_relations_with_tags(("plasma",), require_all=True)
)
