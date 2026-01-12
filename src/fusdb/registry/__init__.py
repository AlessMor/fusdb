"""Static registry data and paths."""

from pathlib import Path

REGISTRY_PATH = Path(__file__).resolve().parent
TAGS_PATH = REGISTRY_PATH / "allowed_tags.yaml"
VARIABLES_PATH = REGISTRY_PATH / "allowed_variables.yaml"
DEFAULTS_PATH = REGISTRY_PATH / "reactor_defaults.yaml"
