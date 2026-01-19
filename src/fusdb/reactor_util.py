from __future__ import annotations

from functools import lru_cache
from typing import Callable, Iterable, Mapping, Sequence
import warnings

import yaml

from fusdb.registry.reactor_defaults import default_parameters_for_tags
from fusdb.relation_class import Relation
from fusdb.registry import SPECIES_PATH, TAGS_PATH, VARIABLES_PATH


def _load_yaml(path: "Path") -> object:
    data = yaml.safe_load(path.read_text())
    return {} if data is None else data


def _load_mapping(path: "Path", *, label: str) -> dict[str, object]:
    data = _load_yaml(path)
    if not isinstance(data, dict):
        raise ValueError(f"{label} must contain a mapping")
    return data


@lru_cache(maxsize=1)
def load_allowed_tags() -> dict[str, object]:
    """Load allowed tag lists and metadata from the YAML registry."""
    data = _load_mapping(TAGS_PATH, label="allowed_tags.yaml")
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
def load_allowed_species() -> dict[str, dict[str, object]]:
    """Load allowed ion species metadata from the YAML registry."""
    data = _load_yaml(SPECIES_PATH)
    if isinstance(data, list):
        return {str(item): {} for item in data}
    if isinstance(data, dict):
        if "species" in data and isinstance(data.get("species"), list):
            return {str(item): {} for item in data.get("species", [])}
        species: dict[str, dict[str, object]] = {}
        for key, meta in data.items():
            if meta is None:
                meta = {}
            if not isinstance(meta, dict):
                raise ValueError("allowed_species.yaml entries must be mappings")
            species[str(key)] = dict(meta)
        return species
    raise ValueError("allowed_species.yaml must contain a mapping of species to metadata")

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
RELATION_DOMAIN_ORDER: dict[str, int] = dict(_RELATION_DOMAIN_ORDER)

_RELATION_DOMAIN_STAGES: tuple[tuple[str, ...], ...] = ()
if _RELATION_DOMAIN_ORDER:
    stages: dict[int, list[str]] = {}
    for name, order in _RELATION_DOMAIN_ORDER.items():
        stages.setdefault(int(order), []).append(name)
    _RELATION_DOMAIN_STAGES = tuple(tuple(stages[order]) for order in sorted(stages))
RELATION_DOMAIN_STAGES: tuple[tuple[str, ...], ...] = _RELATION_DOMAIN_STAGES



ALLOWED_REACTOR_FAMILIES: tuple[str, ...] = tuple(_ALLOWED_TAGS.get("reactor_families", ()))
ALLOWED_REACTOR_CONFIGURATIONS: tuple[str, ...] = tuple(_ALLOWED_TAGS.get("reactor_configurations", ()))
ALLOWED_CONFINEMENT_MODES: tuple[str, ...] = tuple(_ALLOWED_TAGS.get("confinement_modes", ()))
ALLOWED_RELATION_DOMAINS: tuple[str, ...] = tuple(_RELATION_DOMAIN_ORDER.keys())
ALLOWED_SPECIES_META: dict[str, dict[str, object]] = load_allowed_species()
ALLOWED_SPECIES: tuple[str, ...] = tuple(ALLOWED_SPECIES_META.keys())
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
    return _load_mapping(VARIABLES_PATH, label="allowed_variables.yaml")
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


def normalize_key(value: str) -> str:
    """Normalize a free-form key by stripping non-alphanumerics and lowercasing."""
    return "".join(ch for ch in value.lower() if ch.isalnum())


def parse_solve_strategy(strategy_raw: str | Sequence[str] | None) -> tuple[str, list[str] | None]:
    """Parse solve_strategy into a normalized mode and optional user steps."""
    if strategy_raw is None:
        return "default", None
    if isinstance(strategy_raw, str):
        strategy = strategy_raw.strip().lower()
        return (strategy or "default"), None
    if isinstance(strategy_raw, (list, tuple)):
        return "user", [str(step) for step in strategy_raw]
    raise ValueError("solve_strategy must be a string, list, or omitted")


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
        if "spherical" in concept:
            tags.add("spherical_tokamak")
        if "compact" in concept:
            tags.add("compact_tokamak")
    for needle in ("stellarator", "frc", "mirror"):
        if needle in concept:
            tags.add(needle)
    return tuple(sorted(tags))


def relation_domain_order() -> dict[str, int]:
    """Return the configured solve order for relation domains."""
    return dict(RELATION_DOMAIN_ORDER)


def relation_domain_stages() -> tuple[tuple[str, ...], ...]:
    """Return ordered domain stages grouped by solve order."""
    return RELATION_DOMAIN_STAGES


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
    selected_tags: dict[str, tuple[str, ...]] = {}
    for output, method in parameter_methods.items():
        if not method:
            continue
        candidates = by_output.get(output)
        if not candidates:
            continue
        # Normalize the method and relation names so matching ignores separators.
        method_key = normalize_key(method)
        if not method_key:
            continue
        exact: list[Relation] = []
        partial: list[Relation] = []
        for _tags, rel in candidates:
            name_key = normalize_key(rel.name)
            keys = {name_key}
            if rel.variables:
                output_key = normalize_key(rel.variables[0])
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
        chosen = matches[0]
        selected[output] = chosen
        for tags, rel in candidates:
            if rel is chosen:
                selected_tags[output] = tags
                break

    adjusted: list[Relation] = []
    for _tags, rel in relations:
        if not rel.variables:
            adjusted.append(rel)
            continue
        output = rel.variables[0]
        chosen = selected.get(output)
        if chosen is None or rel is chosen:
            adjusted.append(rel)
            continue
        chosen_tags = selected_tags.get(output)
        if not chosen_tags:
            continue
        # Drop other relations only when they overlap the chosen relation's tags.
        if set(_tags).intersection(chosen_tags):
            continue
        adjusted.append(rel)
    return tuple(adjusted)
