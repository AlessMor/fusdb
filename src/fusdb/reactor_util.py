from __future__ import annotations

from functools import lru_cache
from typing import Callable, Iterable, Mapping, Sequence
import warnings

import yaml

from fusdb.relation_class import Relation
from fusdb.registry import DEFAULTS_PATH, TAGS_PATH, VARIABLES_PATH



@lru_cache(maxsize=1)
def load_allowed_tags() -> dict[str, object]:
    """Load allowed tag lists and metadata from the YAML registry."""
    data = yaml.safe_load(TAGS_PATH.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError("allowed_tags.yaml must contain a mapping")
    tags: dict[str, object] = {}
    for key, value in data.items():
        if value is None:
            tags[key] = ()
        elif key == "relation_domains" and isinstance(value, dict):
            domains: dict[str, int] = {}
            for domain, order in value.items():
                try:
                    domains[str(domain)] = int(order)
                except (TypeError, ValueError):
                    raise ValueError(
                        f"allowed_tags.yaml entry {key!r} must map domain names to integers"
                    ) from None
            tags[key] = domains
        elif isinstance(value, list):
            tags[key] = tuple(str(item) for item in value)
        else:
            raise ValueError(f"allowed_tags.yaml entry {key!r} must be a list")
    return tags

@lru_cache(maxsize=1)
def load_default_layers() -> list[dict[str, object]]:
    """Load default parameter layers from the YAML registry."""
    data = yaml.safe_load(DEFAULTS_PATH.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError("reactor_defaults.yaml must contain a mapping")
    raw_layers = data.get("layers", [])
    if raw_layers is None:
        raw_layers = []
    if not isinstance(raw_layers, list):
        raise ValueError("reactor_defaults.yaml 'layers' must be a list")
    layers: list[dict[str, object]] = []
    for layer in raw_layers:
        if not isinstance(layer, dict):
            raise ValueError("Each layer in reactor_defaults.yaml must be a mapping")
        raw_tags = layer.get("tags", [])
        if raw_tags is None:
            raw_tags = []
        if isinstance(raw_tags, str):
            tags = [raw_tags]
        elif isinstance(raw_tags, list):
            tags = raw_tags
        else:
            raise ValueError("Default layer tags must be a string or list")
        raw_defaults = layer.get("defaults", {})
        if raw_defaults is None:
            raw_defaults = {}
        if not isinstance(raw_defaults, dict):
            raise ValueError("Default layer defaults must be a mapping")
        priority_raw = layer.get("priority", 0)
        try:
            priority = int(priority_raw)
        except (TypeError, ValueError):
            raise ValueError("Default layer priority must be an integer") from None
        name = str(layer.get("name", ""))
        layers.append(
            {
                "name": name,
                "tags": tuple(str(tag) for tag in tags),
                "defaults": dict(raw_defaults),
                "priority": priority,
            }
        )
    return layers
_ALLOWED_TAGS = load_allowed_tags()


def _relation_domain_order_from_tags(tags: Mapping[str, object]) -> dict[str, int]:
    raw = tags.get("relation_domains", ())
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return {str(name): int(order) for name, order in raw.items()}
    if isinstance(raw, (list, tuple)):
        return {str(name): index + 1 for index, name in enumerate(raw)}
    raise ValueError("allowed_tags.yaml relation_domains must be a list or mapping")
_RELATION_DOMAIN_ORDER = _relation_domain_order_from_tags(_ALLOWED_TAGS)



ALLOWED_REACTOR_FAMILIES: tuple[str, ...] = tuple(_ALLOWED_TAGS.get("reactor_families", ()))
ALLOWED_REACTOR_CONFIGURATIONS: tuple[str, ...] = tuple(_ALLOWED_TAGS.get("reactor_configurations", ()))
ALLOWED_CONFINEMENT_MODES: tuple[str, ...] = tuple(_ALLOWED_TAGS.get("confinement_modes", ()))
ALLOWED_RELATION_DOMAINS: tuple[str, ...] = tuple(_RELATION_DOMAIN_ORDER.keys())
_REQUIRED_METADATA_FIELDS = _ALLOWED_TAGS.get("required_metadata_fields")
if _REQUIRED_METADATA_FIELDS is None:
    raise ValueError("allowed_tags.yaml must define required_metadata_fields")
REQUIRED_FIELDS: list[str] = list(_REQUIRED_METADATA_FIELDS)
_OPTIONAL_METADATA_FIELDS = _ALLOWED_TAGS.get("optional_metadata_fields")
if _OPTIONAL_METADATA_FIELDS is None:
    raise ValueError("allowed_tags.yaml must define optional_metadata_fields")
OPTIONAL_METADATA_FIELDS: list[str] = list(_OPTIONAL_METADATA_FIELDS)
RESERVED_KEYS: set[str] = set(REQUIRED_FIELDS + OPTIONAL_METADATA_FIELDS)

RELATION_MODULES: tuple[str, ...] = ("fusdb.relations",)


@lru_cache(maxsize=1)
def load_allowed_variables() -> dict[str, dict[str, object]]:
    """Load allowed variable metadata from the YAML registry."""
    data = yaml.safe_load(VARIABLES_PATH.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError("allowed_variables.yaml must contain a mapping")
    return data
ALLOWED_VARIABLES = load_allowed_variables()


WarnFunc = Callable[[str, type[Warning] | None], None]


def normalize_allowed(
    value: str | None,
    allowed: Iterable[str],
    *,
    field_name: str,
) -> str | None:
    """Normalize a value against an allowed list using case-insensitive matching."""
    if value is None:
        return None
    mapping = {entry.lower(): entry for entry in allowed}
    key = value.lower()
    if key not in mapping:
        allowed_list = ", ".join(allowed)
        raise ValueError(f"{field_name} must be one of {allowed_list} (case-insensitive); got {value!r}")
    return mapping[key]


def normalize_country(
    value: str | None,
    *,
    warn: WarnFunc | None = None,
) -> str | None:
    """Normalize a country name/alpha-2/alpha-3 value to ISO 3166-1 alpha-3."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"country must be a string; got {type(value).__name__}")
    text = value.strip()
    if not text:
        return None
    alias_map = {
        "EU": "EUU",
        "EUU": "EUU",
        "UK": "GBR",
    }
    text_upper = text.upper()
    if text_upper in alias_map:
        normalized = alias_map[text_upper]
        if warn is not None and text != normalized:
            warn(
                f"Normalized country {value!r} to ISO 3166-1 alpha-3 code {normalized}.",
                UserWarning,
            )
        return normalized
    try:
        import pycountry
    except ImportError as exc:
        raise ValueError("country normalization requires pycountry; install it to validate ISO 3166-1 codes") from exc

    try:
        record = pycountry.countries.lookup(text)
    except LookupError as exc:
        raise ValueError(
            f"country must be ISO 3166-1 alpha-3, alpha-2, or a recognized country name; got {value!r}"
        ) from exc

    normalized = getattr(record, "alpha_3", None)
    if not normalized:
        raise ValueError(
            f"country must resolve to an ISO 3166-1 alpha-3 code; got {value!r}"
        )

    if warn is not None and text != normalized:
        warn(f"Normalized country {value!r} to ISO 3166-1 alpha-3 code {normalized}.", UserWarning)
    return normalized


def configuration_tags(reactor_configuration: str | None) -> tuple[str, ...]:
    """Derive configuration tags from a free-form reactor configuration string."""
    if not reactor_configuration:
        return ()
    concept = reactor_configuration.lower()
    tags: set[str] = set()
    normalized = concept.replace(" ", "_")
    if normalized:
        tags.add(normalized)
    if "tokamak" in concept:
        tags.add("tokamak")
    if "spherical" in concept and "tokamak" in concept:
        tags.add("spherical_tokamak")
    if "compact" in concept and "tokamak" in concept:
        tags.add("compact_tokamak")
    if "stellarator" in concept:
        tags.add("stellarator")
    if "frc" in concept:
        tags.add("frc")
    if "mirror" in concept:
        tags.add("mirror")
    return tuple(sorted(tags))


def relation_domain_order() -> dict[str, int]:
    """Return the configured solve order for relation domains."""
    return dict(_RELATION_DOMAIN_ORDER)


def relation_domain_stages() -> tuple[tuple[str, ...], ...]:
    """Return ordered domain stages grouped by solve order."""
    stages: dict[int, list[str]] = {}
    for name, order in _RELATION_DOMAIN_ORDER.items():
        stages.setdefault(int(order), []).append(name)
    return tuple(tuple(stages[order]) for order in sorted(stages))


_CONFIG_EXCLUDE_TAGS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("spherical tokamak", ("stellarator", "frc", "mirror")),
    ("stellarator", ("tokamak", "spherical_tokamak", "frc", "mirror")),
    ("tokamak", ("stellarator", "spherical_tokamak", "frc", "mirror")),
)


def config_exclude_tags(reactor_configuration: str | None) -> tuple[str, ...]:
    """Return relation tags that should be excluded for the configuration."""
    if not reactor_configuration:
        return ()
    concept = reactor_configuration.lower()
    for needle, excludes in _CONFIG_EXCLUDE_TAGS:
        if needle in concept:
            return excludes
    return ()


@lru_cache(maxsize=1)
def variable_aliases() -> dict[str, str]:
    """Build a mapping of alias names to canonical variable names."""
    aliases: dict[str, str] = {}
    for name, meta in ALLOWED_VARIABLES.items():
        if not isinstance(meta, dict):
            continue
        raw = meta.get("aliases")
        if raw is None:
            continue
        if isinstance(raw, str):
            items = [raw]
        elif isinstance(raw, (list, tuple)):
            items = list(raw)
        else:
            raise ValueError(f"aliases for {name!r} must be a string or list")
        for alias in items:
            alias_name = str(alias)
            if alias_name == name:
                raise ValueError(f"aliases for {name!r} include the canonical name")
            if alias_name in ALLOWED_VARIABLES:
                raise ValueError(f"alias {alias_name!r} duplicates a canonical variable")
            existing = aliases.get(alias_name)
            if existing is not None and existing != name:
                raise ValueError(f"alias {alias_name!r} already assigned to {existing!r}")
            aliases[alias_name] = name
    return aliases


def default_parameters_for_tags(tags: Iterable[str]) -> dict[str, object]:
    """Return merged defaults for layers that match the provided tags."""
    tag_set = set(tags)
    layers = load_default_layers()
    matched = [
        layer for layer in layers if set(layer["tags"]).issubset(tag_set)
    ]
    matched.sort(key=lambda layer: (layer["priority"], layer["name"]))
    defaults: dict[str, object] = {}
    for layer in matched:
        defaults.update(layer["defaults"])
    return defaults


DEFAULT_GEOMETRY_VALUES: dict[str, float] = {
    name: value
    for name, value in default_parameters_for_tags(()).items()
    if name in {"kappa", "delta", "squareness"}
}


def relations_for(
    groups: str | tuple[str, ...],
    *,
    require_all: bool = True,
    exclude: tuple[str, ...] | None = None,
) -> tuple["Relation", ...]:
    """Return relations that match the requested tag groups."""
    from fusdb.reactor_class import Reactor

    tagged = Reactor.get_relations_with_tags(groups, require_all=require_all, exclude=exclude)
    return tuple(rel for _tags, rel in tagged)


def relations_with_tags(
    groups: str | tuple[str, ...],
    *,
    require_all: bool = True,
    exclude: tuple[str, ...] | None = None,
) -> tuple[tuple[tuple[str, ...], Relation], ...]:
    """Return relations and their tag tuples that match the requested groups."""
    from fusdb.reactor_class import Reactor

    return Reactor.get_relations_with_tags(groups, require_all=require_all, exclude=exclude)


def select_relations(
    relations: Sequence[tuple[tuple[str, ...], Relation]],
    *,
    parameter_methods: Mapping[str, str] | None = None,
    config_tags: Iterable[str] = (),
    warn: WarnFunc | None = None,
) -> tuple[Relation, ...]:
    """Select method-specific relations, falling back to defaults when needed."""
    if not relations:
        return ()
    if not parameter_methods:
        return tuple(rel for _tags, rel in relations)

    warn_func = warn or warnings.warn
    # Index relations by output variable for quick method matching.
    by_output: dict[str, list[tuple[tuple[str, ...], Relation]]] = {}
    for tags, rel in relations:
        if not rel.variables:
            continue
        output = rel.variables[0]
        by_output.setdefault(output, []).append((tags, rel))

    selected: dict[str, Relation] = {}
    for output, method in parameter_methods.items():
        if not method:
            continue
        candidates = by_output.get(output)
        if not candidates:
            continue
        # Normalize the method and relation names so matching ignores separators.
        method_key = "".join(ch for ch in method.lower() if ch.isalnum())
        if not method_key:
            continue
        exact: list[Relation] = []
        partial: list[Relation] = []
        for _tags, rel in candidates:
            name_key = "".join(ch for ch in rel.name.lower() if ch.isalnum())
            keys = {name_key}
            if rel.variables:
                output_key = "".join(ch for ch in rel.variables[0].lower() if ch.isalnum())
                if output_key and name_key.startswith(output_key):
                    stripped = name_key[len(output_key):]
                    if stripped:
                        keys.add(stripped)
            if any(method_key == key for key in keys):
                exact.append(rel)
                continue
            if any(method_key in key for key in keys):
                partial.append(rel)
        matches = sorted(exact or partial, key=lambda r: r.name)
        if not matches:
            warn_func(
                f"Method {method!r} for {output} is not implemented; "
                f"using default relations for reactor tags {sorted(config_tags)}.",
                UserWarning,
            )
            continue
        if len(matches) > 1:
            match_names = ", ".join(rel.name for rel in sorted(matches, key=lambda r: r.name))
            warn_func(
                f"Method {method!r} for {output} is ambiguous ({match_names}); using {matches[0].name}.",
                UserWarning,
            )
        selected[output] = matches[0]

    adjusted: list[Relation] = []
    for _tags, rel in relations:
        output = rel.variables[0] if rel.variables else None
        chosen = selected.get(output)
        if chosen is None or rel is chosen:
            adjusted.append(rel)
            continue
        # When a method override exists, prevent other relations from solving the same output.
        if output is None:
            adjusted.append(rel)
            continue
        if rel.solve_for is None:
            if not rel.variables or rel.variables[0] != output:
                adjusted.append(rel)
                continue
            new_targets = tuple(var for var in rel.variables if var != output)
        else:
            if output not in rel.solve_for:
                adjusted.append(rel)
                continue
            new_targets = tuple(target for target in rel.solve_for if target != output)
        if not new_targets:
            new_targets = ("__relation_skip__",)
        adjusted.append(
            Relation(
                rel.name,
                rel.variables,
                rel.expr,
                priority=rel.priority,
                rel_tol=rel.rel_tol,
                solve_for=new_targets,
                initial_guesses=rel.initial_guesses,
                max_solve_iterations=rel.max_solve_iterations,
                constraints=rel.constraints,
            )
        )
    return tuple(adjusted)
