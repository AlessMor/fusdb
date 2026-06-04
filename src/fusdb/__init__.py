"""Public FusDB package."""

from __future__ import annotations

import importlib
import sys

_IMPL_NAME = "fusdb_pyomo"
_impl = importlib.import_module(_IMPL_NAME)

from fusdb_pyomo import *  # noqa: F401,F403

__path__ = list(getattr(_impl, "__path__", []))

for suffix in ("reactor", "relation", "relationsystem", "variable", "utils", "registry", "modes", "relations"):
    sys.modules[f"{__name__}.{suffix}"] = importlib.import_module(f"{_IMPL_NAME}.{suffix}")