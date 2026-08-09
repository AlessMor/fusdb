"""FusDB registries and constants."""

from __future__ import annotations

import numpy as np

from .constants import *
from .reactivity_config import *
from .reaction_registry import *
from .reaction_registry import REACTIONS, ReactionRegistry, ReactionSpec
from .species_registry import SPECIES, SpeciesRegistry, SpeciesSpec
from .tag_registry import TAGS, TagRegistry
from . import variable_registry as _variable_registry
from .variable_registry import VariableRegistry, VariableSpec
from .coordinate_variables import with_coordinate_variables

# Apply the staged profile-coordinate overlay before relation discovery.  The
# relation registry imports VARIABLES from variable_registry, so update the
# module singleton as well as this package-level binding.
VARIABLES = with_coordinate_variables(_variable_registry.VARIABLES)
_variable_registry.VARIABLES = VARIABLES

from .relation_registry import RELATIONS, RelationRegistry
from .dataset import DATASETS, DatasetDocument, DatasetRegistry, load_dataset


from .unitregistry import convert_value
