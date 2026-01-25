"""
Reactor utilities: tag loading, relation selection, normalization.

This module provides utilities for working with reactor configurations
and relation selection. It handles:
- Loading and parsing configuration from YAML files
- Normalizing metadata values (country codes, configuration names)
- Selecting appropriate relations based on reactor configuration
- Managing relation domain stages (order of relation application)

The configuration files define:
- allowed_tags.yaml: Valid tags for reactor configurations, families, domains
- allowed_species.yaml: Valid plasma species (D, T, He, etc.)
- allowed_variables.yaml: Valid parameter names with metadata

Example:
    >>> from fusdb.reactor_util import configuration_tags, relations_with_tags
    >>> tags = configuration_tags("tokamak")
    >>> tags
    ('tokamak',)
    >>> rels = relations_with_tags("geometry", exclude=("stellarator",))
"""
from __future__ import annotations
from functools import lru_cache
from typing import Callable, Iterable, Mapping, Sequence
import warnings
import yaml
from fusdb.relation_class import Relation
from fusdb.registry import SPECIES_PATH, TAGS_PATH, VARIABLES_PATH


def _load_yaml(path) -> object:
    """Load and parse a YAML file."""
    return yaml.safe_load(path.read_text()) or {}


def _load_mapping(path, *, label: str) -> dict:
    """
    Load a YAML file that must contain a mapping.
    
    Args:
        path: Path to the YAML file
        label: Label for error messages
        
    Returns:
        The parsed dictionary
        
    Raises:
        ValueError: If the file doesn't contain a mapping
    """
    d = _load_yaml(path)
    if not isinstance(d, dict):
        raise ValueError(f"{label} must contain a mapping")
    return d


@lru_cache(maxsize=1)
def load_allowed_tags() -> dict[str, object]:
    """
    Load the allowed_tags.yaml configuration file.
    
    This file defines valid values for various categorization tags:
    - reactor_families: Valid reactor family names
    - reactor_configurations: Valid configuration types
    - confinement_modes: Valid plasma confinement modes
    - relation_domains: Domains for organizing relations (with order)
    - required_metadata_fields: Fields that must be present in reactor YAML
    - optional_metadata_fields: Fields that may be present
    
    Returns:
        Dictionary mapping tag category names to their allowed values
    """
    data = _load_mapping(TAGS_PATH, label="allowed_tags.yaml")
    tags = {}
    
    for k, v in data.items():
        if v is None:
            tags[k] = ()
        elif k == "relation_domains" and isinstance(v, dict):
            # Domain ordering is specified as name -> order mapping
            tags[k] = {str(d): int(o) for d, o in v.items()}
        elif isinstance(v, list):
            tags[k] = tuple(str(i) for i in v)
        else:
            raise ValueError(f"allowed_tags.yaml entry {k!r} must be a list")
    
    return tags


@lru_cache(maxsize=1)
def load_allowed_species() -> dict[str, dict[str, object]]:
    """
    Load the allowed_species.yaml configuration file.
    
    This file defines valid plasma species (e.g., D, T, He) and their
    metadata (atomic mass, charge, etc.).
    
    Returns:
        Dictionary mapping species names to their metadata dictionaries
    """
    d = _load_yaml(SPECIES_PATH)
    
    # Handle simple list format
    if isinstance(d, list):
        return {str(i): {} for i in d}
    
    if isinstance(d, dict):
        # Handle nested format with "species" key
        if "species" in d and isinstance(d.get("species"), list):
            return {str(i): {} for i in d.get("species", [])}
        
        # Handle direct mapping format
        sp = {}
        for k, m in d.items():
            if m is None:
                m = {}
            if not isinstance(m, dict):
                raise ValueError("allowed_species.yaml entries must be mappings")
            sp[str(k)] = dict(m)
        return sp
    
    raise ValueError("allowed_species.yaml must contain a mapping")


# Load tags at module import time
_TAGS = load_allowed_tags()


def _domain_order(tags) -> dict[str, int]:
    """
    Extract domain ordering from the loaded tags.
    
    Relation domains are solved in a specific order (e.g., geometry
    before power balance). This function extracts that ordering.
    
    Args:
        tags: The loaded tags dictionary
        
    Returns:
        Dictionary mapping domain names to their order (lower = earlier)
    """
    raw = tags.get("relation_domains", ())
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return {str(n): int(o) for n, o in raw.items()}
    if isinstance(raw, (list, tuple)):
        return {str(n): i + 1 for i, n in enumerate(raw)}
    raise ValueError("allowed_tags.yaml relation_domains must be a list or mapping")


# Compute domain ordering at module import time
_ORDER = _domain_order(_TAGS)
RELATION_DOMAIN_ORDER = dict(_ORDER)

# Group domains into stages (domains with same order are solved together)
_STAGES = ()
if _ORDER:
    st = {}
    for n, o in _ORDER.items():
        st.setdefault(int(o), []).append(n)
    _STAGES = tuple(tuple(st[o]) for o in sorted(st))
RELATION_DOMAIN_STAGES = _STAGES

# Export allowed values as module-level constants
ALLOWED_REACTOR_FAMILIES = tuple(_TAGS.get("reactor_families", ()))
ALLOWED_REACTOR_CONFIGURATIONS = tuple(_TAGS.get("reactor_configurations", ()))
ALLOWED_CONFINEMENT_MODES = tuple(_TAGS.get("confinement_modes", ()))
ALLOWED_RELATION_DOMAINS = tuple(_ORDER.keys())
ALLOWED_SPECIES_META = load_allowed_species()
ALLOWED_SPECIES = tuple(ALLOWED_SPECIES_META.keys())

# Required and optional fields for reactor YAML files
_REQ = _TAGS.get("required_metadata_fields")
if _REQ is None:
    raise ValueError("allowed_tags.yaml must define required_metadata_fields")
REQUIRED_FIELDS = list(_REQ)

_OPT = _TAGS.get("optional_metadata_fields")
if _OPT is None:
    raise ValueError("allowed_tags.yaml must define optional_metadata_fields")
OPTIONAL_METADATA_FIELDS = list(_OPT)

RESERVED_KEYS = set(REQUIRED_FIELDS + OPTIONAL_METADATA_FIELDS)
RELATION_MODULES = ("fusdb.relations",)


@lru_cache(maxsize=1)
def load_allowed_variables() -> dict[str, dict[str, object]]:
    """
    Load the allowed_variables.yaml configuration file.
    
    This file defines valid parameter names with their metadata:
    - units: Physical units
    - description: Human-readable description
    - constraints: Validity constraints (e.g., "P >= 0")
    - aliases: Alternative names
    
    Returns:
        Dictionary mapping variable names to their metadata
    """
    return _load_mapping(VARIABLES_PATH, label="allowed_variables.yaml")


ALLOWED_VARIABLES = load_allowed_variables()

# Type alias for warning callback functions
WarnFunc = Callable[[str, type[Warning] | None], None]


def normalize_allowed(value: str | None, allowed: Iterable[str], *, field_name: str) -> str | None:
    """
    Normalize a value to its canonical form from an allowed list.
    
    Performs case-insensitive matching and returns the canonical
    (properly cased) version.
    
    Args:
        value: The value to normalize
        allowed: Iterable of allowed values
        field_name: Name of the field (for error messages)
        
    Returns:
        The canonical form of the value, or None if input was None
        
    Raises:
        ValueError: If value is not in the allowed list
        
    Example:
        >>> normalize_allowed("TOKAMAK", ["tokamak", "stellarator"], field_name="config")
        'tokamak'
    """
    if value is None:
        return None
    m = {e.lower(): e for e in allowed}
    k = value.lower()
    if k not in m:
        raise ValueError(f"{field_name} must be one of {', '.join(allowed)}; got {value!r}")
    return m[k]


def normalize_key(value: str) -> str:
    """
    Normalize a string for case-insensitive, symbol-insensitive matching.
    
    Converts to lowercase and removes all non-alphanumeric characters.
    
    Args:
        value: String to normalize
        
    Returns:
        Normalized string containing only lowercase alphanumeric characters
        
    Example:
        >>> normalize_key("P_fus")
        'pfus'
        >>> normalize_key("Ohm's Law")
        'ohmslaw'
    """
    return "".join(ch for ch in value.lower() if ch.isalnum())


def parse_solve_strategy(s) -> tuple[str, list[str] | None]:
    """
    Parse the solve_strategy field from reactor YAML.
    
    The solve strategy determines how relations are applied:
    - "default" or "staged": Apply relations in domain stages
    - "global": Apply all relations simultaneously
    - list: User-defined sequence of steps
    
    Args:
        s: The solve_strategy value (string, list, or None)
        
    Returns:
        Tuple of (strategy_name, steps) where steps is None for
        non-user strategies
        
    Raises:
        ValueError: If the value is not valid
    """
    if s is None:
        return "default", None
    if isinstance(s, str):
        return (s.strip().lower() or "default"), None
    if isinstance(s, (list, tuple)):
        return "user", [str(x) for x in s]
    raise ValueError("solve_strategy must be a string, list, or omitted")


def normalize_country(value: str | None, *, warn: WarnFunc | None = None) -> str | None:
    """
    Normalize a country value to ISO 3166-1 alpha-3 code.
    
    Accepts various formats:
    - ISO 3166-1 alpha-2 codes (e.g., "US", "GB")
    - ISO 3166-1 alpha-3 codes (e.g., "USA", "GBR")
    - Country names (e.g., "United States", "Germany")
    
    Args:
        value: The country value to normalize
        warn: Optional warning function for normalization notices
        
    Returns:
        ISO 3166-1 alpha-3 code, or None if input was None
        
    Raises:
        ValueError: If the value cannot be resolved to a country
        
    Example:
        >>> normalize_country("US")
        'USA'
        >>> normalize_country("Germany")
        'DEU'
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"country must be a string; got {type(value).__name__}")
    
    t = value.strip()
    if not t:
        return None
    
    # Handle special aliases
    alias = {"EU": "EUU", "EUU": "EUU", "UK": "GBR"}
    up = t.upper()
    if up in alias:
        n = alias[up]
        if warn and t != n:
            warn(f"Normalized country {value!r} to ISO 3166-1 alpha-3 code {n}.", UserWarning)
        return n
    
    # Use pycountry for lookup
    try:
        import pycountry
    except ImportError as e:
        raise ValueError("country normalization requires pycountry") from e
    
    try:
        rec = pycountry.countries.lookup(t)
    except LookupError as e:
        raise ValueError(f"country must be ISO 3166-1 alpha-3/alpha-2 or name; got {value!r}") from e
    
    n = getattr(rec, "alpha_3", None)
    if not n:
        raise ValueError(f"country must resolve to ISO 3166-1 alpha-3; got {value!r}")
    
    if warn and t != n:
        warn(f"Normalized country {value!r} to ISO 3166-1 alpha-3 code {n}.", UserWarning)
    return n


def configuration_tags(cfg: str | None) -> tuple[str, ...]:
    """
    Generate tags from a reactor configuration string.
    
    Parses the configuration name and extracts relevant tags for
    relation selection.
    
    Args:
        cfg: Reactor configuration (e.g., "tokamak", "spherical tokamak")
        
    Returns:
        Tuple of tags derived from the configuration
        
    Example:
        >>> configuration_tags("spherical tokamak")
        ('spherical_tokamak', 'tokamak')
    """
    if not cfg:
        return ()
    
    c = cfg.lower()
    tags = set()
    
    # Add normalized version
    n = c.replace(" ", "_")
    if n:
        tags.add(n)
    
    # Add component tags for tokamaks
    if "tokamak" in c:
        tags.add("tokamak")
        if "spherical" in c:
            tags.add("spherical_tokamak")
        if "compact" in c:
            tags.add("compact_tokamak")
    
    # Add tags for other configurations
    for x in ("stellarator", "frc", "mirror"):
        if x in c:
            tags.add(x)
    
    return tuple(sorted(tags))


def relation_domain_order() -> dict[str, int]:
    """Get the domain ordering dictionary."""
    return dict(RELATION_DOMAIN_ORDER)


def relation_domain_stages() -> tuple[tuple[str, ...], ...]:
    """Get the domain stages (groups of domains solved together)."""
    return RELATION_DOMAIN_STAGES


# Exclusion rules: (config_substring, excluded_tags)
_EXCL = (
    ("spherical tokamak", ("stellarator", "frc", "mirror")),
    ("stellarator", ("tokamak", "spherical_tokamak", "frc", "mirror")),
    ("tokamak", ("stellarator", "spherical_tokamak", "frc", "mirror")),
)


def config_exclude_tags(cfg: str | None) -> tuple[str, ...]:
    """
    Get tags to exclude based on reactor configuration.
    
    Different reactor types have incompatible physics models.
    This function returns tags that should be excluded when
    selecting relations for a given configuration.
    
    Args:
        cfg: Reactor configuration string
        
    Returns:
        Tuple of tags to exclude
        
    Example:
        >>> config_exclude_tags("tokamak")
        ('stellarator', 'spherical_tokamak', 'frc', 'mirror')
    """
    if not cfg:
        return ()
    c = cfg.lower()
    for needle, excl in _EXCL:
        if needle in c:
            return excl
    return ()


@lru_cache(maxsize=1)
def variable_aliases() -> dict[str, str]:
    """
    Load variable aliases from allowed_variables.yaml.
    
    Aliases allow alternative names for parameters (e.g., "major_radius"
    as an alias for "R").
    
    Returns:
        Dictionary mapping alias names to canonical names
        
    Raises:
        ValueError: If alias configuration is invalid
    """
    als = {}
    for n, m in ALLOWED_VARIABLES.items():
        if not isinstance(m, dict):
            continue
        raw = m.get("aliases")
        if raw is None:
            continue
        
        # Handle both single alias and list of aliases
        items = [raw] if isinstance(raw, str) else list(raw) if isinstance(raw, (list, tuple)) else None
        if items is None:
            raise ValueError(f"aliases for {n!r} must be a string or list")
        
        for a in items:
            an = str(a)
            if an == n:
                raise ValueError(f"aliases for {n!r} include the canonical name")
            if an in ALLOWED_VARIABLES:
                raise ValueError(f"alias {an!r} duplicates a canonical variable")
            ex = als.get(an)
            if ex is not None and ex != n:
                raise ValueError(f"alias {an!r} already assigned to {ex!r}")
            als[an] = n
    
    return als


# Default geometry values for reactors without explicit values
DEFAULT_GEOMETRY_VALUES = {"squareness": 0.0}


def relations_for(
    groups,
    *,
    require_all: bool = True,
    exclude=None
) -> tuple[Relation, ...]:
    """
    Get relations matching the given groups/tags.
    
    Convenience function that returns just the Relation objects
    (without tags).
    
    Args:
        groups: Tag(s) to match
        require_all: If True, must have all tags; if False, any tag matches
        exclude: Tags that disqualify a relation
        
    Returns:
        Tuple of matching Relation objects
    """
    from fusdb.reactor_class import Reactor
    return tuple(r for _, r in Reactor.get_relations_with_tags(groups, require_all=require_all, exclude=exclude))


def relations_with_tags(
    groups,
    *,
    require_all: bool = True,
    exclude=None
):
    """
    Get relations matching the given groups/tags, with their tags.
    
    Args:
        groups: Tag(s) to match
        require_all: If True, must have all tags; if False, any tag matches
        exclude: Tags that disqualify a relation
        
    Returns:
        Tuple of (tags, Relation) pairs for matching relations
    """
    from fusdb.reactor_class import Reactor
    return Reactor.get_relations_with_tags(groups, require_all=require_all, exclude=exclude)


def select_relations(
    rels: Sequence[tuple[tuple[str, ...], Relation]],
    *,
    parameter_methods: Mapping[str, str] | None = None,
    config_tags: Iterable[str] = (),
    warn: WarnFunc | None = None
) -> tuple[Relation, ...]:
    """
    Select which relations to use based on configuration and method preferences.
    
    When multiple relations can compute the same output parameter, this
    function selects which one to use based on the parameter_methods
    preferences specified in the reactor YAML.
    
    Args:
        rels: Sequence of (tags, Relation) pairs to select from
        parameter_methods: Mapping from parameter names to preferred method names
        config_tags: Tags from the reactor configuration
        warn: Optional warning function
        
    Returns:
        Tuple of selected Relation objects
        
    Example:
        >>> # If parameter_methods = {"W_E": "ITER_1999"}
        >>> # and there are relations "Stored energy ITER 1999" and "Stored energy IPB98"
        >>> # this will prefer the ITER 1999 relation for computing W_E
    """
    if not rels:
        return ()
    if not parameter_methods:
        return tuple(r for _, r in rels)
    
    wf = warn or warnings.warn
    
    # Group relations by their primary output
    by_out = {}
    for tags, rel in rels:
        if rel.variables:
            by_out.setdefault(rel.variables[0], []).append((tags, rel))
    
    # Select preferred relations based on parameter_methods
    sel = {}  # output -> selected relation
    sel_tags = {}  # output -> selected tags
    
    for out, meth in parameter_methods.items():
        if not meth:
            continue
        cands = by_out.get(out)
        if not cands:
            continue
        
        mk = normalize_key(meth)
        if not mk:
            continue
        
        # Find matching relations
        ex = []  # Exact matches
        pa = []  # Partial matches
        
        for _, rel in cands:
            nk = normalize_key(rel.name)
            keys = {nk}
            
            # Also try matching against name suffix (e.g., "ITER_1999" from "W_E_ITER_1999")
            if rel.variables:
                ok = normalize_key(rel.variables[0])
                if ok and nk.startswith(ok):
                    st = nk[len(ok):]
                    if st:
                        keys.add(st)
            
            if any(mk == k for k in keys):
                ex.append(rel)
                continue
            if any(mk in k for k in keys):
                pa.append(rel)
        
        matches = sorted(ex or pa, key=lambda r: r.name)
        
        if not matches:
            wf(f"Method {meth!r} for {out} not implemented; using default.", UserWarning)
            continue
        if len(matches) > 1:
            wf(f"Method {meth!r} for {out} is ambiguous; using {matches[0].name}.", UserWarning)
        
        sel[out] = matches[0]
        
        # Record the tags of the selected relation
        for tags, rel in cands:
            if rel is matches[0]:
                sel_tags[out] = tags
                break
    
    # Build adjusted relation list
    adj = []
    for tags, rel in rels:
        if not rel.variables:
            adj.append(rel)
            continue
        
        out = rel.variables[0]
        chosen = sel.get(out)
        
        # Include if no specific selection made, or if this is the chosen one
        if chosen is None or rel is chosen:
            adj.append(rel)
            continue
        
        # Exclude if has conflicting tags with chosen
        ct = sel_tags.get(out)
        if ct and set(tags) & set(ct):
            continue
        
        adj.append(rel)
    
    return tuple(adj)
