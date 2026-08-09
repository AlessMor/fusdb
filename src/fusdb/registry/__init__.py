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

# Apply the staged coordinate contract before relation discovery.  Importers of
# ``fusdb.registry.VARIABLES`` therefore see the same augmented registry used to
# canonicalize every discovered relation.
VARIABLES = with_coordinate_variables(_BASE_VARIABLES)

from .relation_registry import RELATIONS, RelationRegistry
from .dataset import DATASETS, DatasetDocument, DatasetRegistry, load_dataset
from .unitregistry import convert_value
