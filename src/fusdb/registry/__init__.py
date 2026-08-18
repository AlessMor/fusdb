"""FusDB registries and constants."""

from __future__ import annotations

import numpy as np

from .constants import *
from .reactivity_config import *
from .reaction_registry import *
from .reaction_registry import REACTIONS, ReactionSpec
from .species_registry import SPECIES, SpeciesRegistry, SpeciesSpec
from .tag_registry import TAGS, TagRegistry
from .variable_registry import VARIABLES as _BASE_VARIABLES, VariableRegistry, VariableSpec
from .coordinate_variables import with_coordinate_variables

VARIABLES = with_coordinate_variables(_BASE_VARIABLES)

from .. import relation as _relation_module
_relation_module._VARIABLE_REGISTRY = VARIABLES

from .relation_registry import RELATIONS, RelationRegistry, get_relations
from .dataset import DATASETS, DatasetDocument, LoadedTable, load_dataset, resolve_dataset
from .unitregistry import convert_value
