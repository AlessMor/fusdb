"""Utility helpers for reactor loading."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import logging

import yaml

from .registry import parse_variables, validate_solver_tags
from .utils import normalize_country, normalize_solver_mode, normalize_tags_to_tuple

if TYPE_CHECKING:
    from .reactor_class import Reactor
    from .relation_class import Relation

logger = logging.getLogger(__name__)


def reactor_from_yaml(path_like: str | Path, *, cls: type["Reactor"]) -> "Reactor":
    """Build a reactor object from a YAML path-like input.

    Args:
        path_like: Path to reactor yaml, directory, or reactor name-like reference.
        cls: Reactor class (or subclass) to instantiate.

    Returns:
        Reactor instance with parsed variables and filtered relations.
    """
    path = Path(path_like).expanduser()
    if not path.is_file():
        if path.is_dir() and (path / "reactor.yaml").is_file():
            path = path / "reactor.yaml"
        else:
            start = Path.cwd()
            root = start
            for parent in (start, *start.parents):
                if (parent / "reactors").is_dir() and (parent / "src" / "fusdb").is_dir():
                    root = parent
                    break
            candidate = root / path
            if candidate.is_file():
                path = candidate
            elif candidate.is_dir() and (candidate / "reactor.yaml").is_file():
                path = candidate / "reactor.yaml"
            elif (root / "reactors" / path).is_dir():
                path = root / "reactors" / path / "reactor.yaml"
            elif (root / "reactors" / f"{path}.yaml").is_file():
                path = root / "reactors" / f"{path}.yaml"

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    metadata = raw.get("metadata", {}) or {}
    reactor_id = metadata.get("id") or path.parent.name
    reactor_name = metadata.get("name") or reactor_id

    tags = normalize_tags_to_tuple(raw.get("tags", []) or ())
    solver_tags = raw.get("solver_tags", {}) or {}
    solver_mode = normalize_solver_mode(solver_tags.get("mode", "overwrite"))
    solver_tags_for_validation = dict(solver_tags)
    if "mode" in solver_tags_for_validation:
        solver_tags_for_validation["mode"] = solver_mode
    validate_solver_tags(solver_tags_for_validation, log=logger)
    verbose = bool(solver_tags.get("verbosity", False))
    solving_order = list(solver_tags.get("solving_order", []) or ())

    variables_dict = parse_variables(raw.get("variables", {}))

    default_relations: list["Relation"] = []
    try:
        from .registry.reactor_defaults import apply_reactor_defaults

        default_relations = apply_reactor_defaults(variables_dict)
    except Exception:
        pass

    reactor = cls(
        path=path,
        id=reactor_id,
        name=reactor_name,
        organization=metadata.get("organization"),
        country=normalize_country(metadata.get("country")),
        year=metadata.get("year"),
        doi=metadata.get("doi"),
        notes=metadata.get("notes"),
        tags=tags,
        solving_order=solving_order,
        solver_mode=solver_mode,
        verbose=verbose,
        variables_dict=variables_dict,
        default_relations=default_relations,
    )
    reactor.relations = list(reactor._ordered_relations())
    return reactor
