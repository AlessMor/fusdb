"""FusDB registries and constants."""

from __future__ import annotations

import numpy as np

from .constants import *
from .reactivity_config import *
from .reaction_registry import *
from .reaction_registry import REACTIONS, ReactionRegistry, ReactionSpec
from .species_registry import SPECIES, SpeciesRegistry, SpeciesSpec
from .tag_registry import TAGS, TagRegistry
from .variable_registry import VARIABLES as _BASE_VARIABLES, VariableRegistry, VariableSpec
from .coordinate_variables import with_coordinate_variables

# Apply the staged coordinate contract before relation discovery. Importers of
# ``fusdb.registry.VARIABLES`` therefore see the same augmented registry used to
# canonicalize every discovered relation.
VARIABLES = with_coordinate_variables(_BASE_VARIABLES)

# ``relation.py`` resolves registry metadata lazily to avoid an import cycle.
# Point that cache at the active staged registry now, so standalone relation
# calls and RelationSystem compilation use one identical alias/domain contract.
from .. import relation as _relation_module
_relation_module._VARIABLE_REGISTRY = VARIABLES

from .relation_registry import RELATIONS, RelationRegistry
from .dataset import DATASETS, DatasetDocument, DatasetRegistry, load_dataset
from .unitregistry import convert_value
