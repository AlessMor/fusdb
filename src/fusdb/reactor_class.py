from dataclasses import dataclass, field
import importlib
import inspect
import math
import sympy as sp
from sympy.core.relational import Relational
from pathlib import Path
from typing import Any, ClassVar

import warnings

from fusdb.relation_class import Relation, RelationSystem
from fusdb.relation_util import REL_TOL_DEFAULT, as_float, first_numeric, symbol
from fusdb.reactor_util import (
    ALLOWED_CONFINEMENT_MODES,
    ALLOWED_REACTOR_FAMILIES,
    ALLOWED_REACTOR_CONFIGURATIONS,
    ALLOWED_RELATION_DOMAINS,
    RELATION_MODULES,
    REQUIRED_FIELDS,
    OPTIONAL_METADATA_FIELDS,
    config_exclude_tags,
    configuration_tags,
    normalize_allowed,
    normalize_key,
    parse_solve_strategy,
    relation_domain_stages,
    relations_with_tags,
    select_relations,
)


Scalar = float | sp.Expr


@dataclass
class Reactor:
    """Reactor metadata and parameter container that resolves relations on load."""
    ALLOWED_CONFINEMENT_MODES: ClassVar[tuple[str, ...]] = ALLOWED_CONFINEMENT_MODES  # Allowed confinement modes.
    ALLOWED_REACTOR_FAMILIES: ClassVar[tuple[str, ...]] = ALLOWED_REACTOR_FAMILIES  # Allowed reactor families.
    ALLOWED_REACTOR_CONFIGURATIONS: ClassVar[tuple[str, ...]] = ALLOWED_REACTOR_CONFIGURATIONS  # Allowed types.
    _RELATION_MODULES: ClassVar[tuple[str, ...]] = RELATION_MODULES  # Module paths to import relations from.
    _RELATIONS: ClassVar[list[tuple[tuple[str, ...], Relation]]] = []  # Registered relation catalog.
    _RELATIONS_IMPORTED: ClassVar[bool] = False  # Guard for one-time relation imports.

    # required identity
    id: str = field(metadata={"section": "metadata_required"})  # Unique reactor identifier.
    name: str = field(metadata={"section": "metadata_required"})  # Human-readable reactor name.
    reactor_configuration: str = field(metadata={"section": "metadata_required"})  # Reactor type/configuration.
    organization: str = field(metadata={"section": "metadata_required"})  # Owning organization/lab.

    # optional metadata
    reactor_family: str | None = field(default=None, metadata={"section": "metadata_optional"})  # Program/family label.
    country: str | None = field(default=None, metadata={"section": "metadata_optional"})  # ISO alpha-3 country code.
    site: str | None = field(default=None, metadata={"section": "metadata_optional"})  # Site name or location.
    design_year: int | None = field(default=None, metadata={"section": "metadata_optional"})  # Design year.
    doi: str | list[str] | None = field(default=None, metadata={"section": "metadata_optional"})  # DOI(s).
    notes: str | None = field(default=None, metadata={"section": "metadata_optional"})  # Free-form notes.
    confinement_mode: str | None = field(default=None, metadata={"section": "metadata_optional"})  # H/L/I-mode, etc.
    solve_strategy: str | list[str] | None = field(default=None, metadata={"section": "metadata_optional"})  # Solve strategy: default, global, or list of domains/relation names.
    allow_relation_overrides: bool | None = field(
        default=True, metadata={"section": "metadata_optional"}
    )  # Allow explicit inputs to be overridden by relations.
    parameters: dict[str, Scalar | None] = field(default_factory=dict)  # Parameter values (explicit or solved).
    parameter_tolerances: dict[str, float] = field(default_factory=dict)  # Tolerances for explicit values.
    parameter_methods: dict[str, str] = field(default_factory=dict)  # Method overrides by output name.
    parameter_defaults: dict[str, Scalar] = field(default_factory=dict)  # Default seeds by tags.
    fallback_relations: tuple[Relation, ...] = field(default_factory=tuple)  # Per-reactor fallback relations.
    explicit_parameters: set[str] = field(default_factory=set)  # Explicit (or promoted) parameter names from reactor inputs.
    input_parameters: dict[str, Scalar | None] = field(default_factory=dict)  # Copy of raw reactor.yaml inputs.

    # internal
    root_dir: Path | None = field(default=None, metadata={"section": "internal"})  # Source directory.
    relations_used: list[tuple[tuple[str, ...], Relation]] = field(
        default_factory=list, metadata={"section": "internal"}, repr=False  # Relations applied.
    )

    def __post_init__(self) -> None:
        """Normalize metadata and solve configured relations in stages."""
        if self.parameter_methods is None:
            self.parameter_methods = {}
        if self.parameter_defaults is None:
            self.parameter_defaults = {}
        if self.fallback_relations is None:
            self.fallback_relations = ()
        if self.explicit_parameters is None:
            self.explicit_parameters = set()
        if self.allow_relation_overrides is None:
            self.allow_relation_overrides = True
        if not self.input_parameters:
            self.input_parameters = {
                name: self.parameters.get(name)
                for name in self.explicit_parameters
                if name in self.parameters
            }
        self.relations_used = []
        self.reactor_family = normalize_allowed(
            self.reactor_family, self.ALLOWED_REACTOR_FAMILIES, field_name="reactor_family"
        )
        
        # SET REACTOR CONFIGURATION (tokamak, stellarator, etc)
        self.reactor_configuration = normalize_allowed(
            self.reactor_configuration,
            self.ALLOWED_REACTOR_CONFIGURATIONS,
            field_name="reactor_configuration",
        )
        # SET CONFINEMENT MODE (L-mode, H-mode, etc)
        if self.confinement_mode and self.ALLOWED_CONFINEMENT_MODES:
            self.confinement_mode = normalize_allowed(
                self.confinement_mode,
                self.ALLOWED_CONFINEMENT_MODES,
                field_name="confinement_mode",
            )
        rel_tol = REL_TOL_DEFAULT
        config_tags = configuration_tags(self.reactor_configuration)
        exclude_tags = set(config_exclude_tags(self.reactor_configuration))
        if self.confinement_mode:
            for mode in self.ALLOWED_CONFINEMENT_MODES:
                if mode != self.confinement_mode:
                    exclude_tags.add(mode)
        exclude_tags = tuple(sorted(exclude_tags))

        strategy, user_steps = parse_solve_strategy(self.solve_strategy)

        if strategy in ("default", "staged"):
            for groups in relation_domain_stages():
                tagged = list(relations_with_tags(groups, require_all=False, exclude=exclude_tags))
                self._apply_relations(tuple(tagged), rel_tol, config_tags=config_tags)
        elif strategy == "global":
            tagged = list(
                relations_with_tags(ALLOWED_RELATION_DOMAINS, require_all=False, exclude=exclude_tags)
            )
            self._apply_relations(tuple(tagged), rel_tol, config_tags=config_tags)
        elif strategy == "user":
            if not user_steps:
                raise ValueError(
                    "solve_strategy set to 'user' but no steps were provided; use a list of domains or relation names"
                )

            domain_lookup = {domain.lower(): domain for domain in ALLOWED_RELATION_DOMAINS}
            self._ensure_relation_modules_loaded()
            name_map: dict[str, list[tuple[tuple[str, ...], Relation]]] = {}
            for tags, rel in self._RELATIONS:
                name_map.setdefault(normalize_key(rel.name), []).append((tags, rel))

            for step in user_steps:
                step_text = step.strip()
                if not step_text:
                    raise ValueError("solve_strategy steps must be non-empty strings")
                domain = domain_lookup.get(step_text.lower())
                if domain:
                    tagged = list(relations_with_tags(domain, require_all=False, exclude=exclude_tags))
                    self._apply_relations(tuple(tagged), rel_tol, config_tags=config_tags)
                    continue
                step_key = normalize_key(step_text)
                matches = name_map.get(step_key)
                if not matches:
                    raise ValueError(f"solve_strategy step {step_text!r} did not match a domain or relation name")
                for tags, rel in matches:
                    if exclude_tags and exclude_tags.intersection(tags):
                        warnings.warn(
                            f"Relation {rel.name!r} selected by name but excluded for configuration "
                            f"{self.reactor_configuration!r}.",
                            UserWarning,
                        )
                self._apply_relations(tuple(matches), rel_tol, config_tags=config_tags)
        else:
            raise ValueError(
                "solve_strategy must be 'default', 'global', or a list of domains/relation names"
            )

    def __getattr__(self, name: str) -> Any:
        """Expose parameters as attributes when present in the parameter map.
        So that reactor.major_radius works as a shortcut for reactor.parameters["major_radius"]."""
        params = self.__dict__.get("parameters", {})
        if name in params:
            return params[name]
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        """Route unknown attributes into the parameter map."""
        if name.startswith("_") or name in type(self).__dataclass_fields__:
            object.__setattr__(self, name, value)
            return
        params = self.__dict__.setdefault("parameters", {})
        params[name] = value

    def __dir__(self) -> list[str]:
        """Include parameter names in attribute completion."""
        base = set(super().__dir__())
        params = self.__dict__.get("parameters", {})
        return sorted(base | set(params.keys()))

    def _apply_relations(
        self,
        tagged_relations: tuple[tuple[tuple[str, ...], Relation], ...],
        rel_tol: float,
        *,
        config_tags: tuple[str, ...],
    ) -> None:
        relations = select_relations(
            tagged_relations,
            parameter_methods=self.parameter_methods,
            config_tags=config_tags,
            warn=warnings.warn,
        )
        if relations:
            selected = set(relations)
            self.relations_used.extend((tags, rel) for (tags, rel) in tagged_relations if rel in selected)
        fallback_relations: list[Relation] = []
        if self.fallback_relations:
            base_outputs = {rel.variables[0] for rel in relations if rel.variables}
            explicit_params = self.explicit_parameters or set()
            for rel in self.fallback_relations:
                if not rel.variables:
                    continue
                output = rel.variables[0]
                if output in base_outputs or output in explicit_params:
                    continue
                value = self.parameters.get(output)
                if value is None or isinstance(value, Relational):
                    fallback_relations.append(rel)
                    continue
                if isinstance(value, sp.Expr) and value.free_symbols:
                    fallback_relations.append(rel)

        if fallback_relations:
            self.relations_used.extend((("fallback",), rel) for rel in fallback_relations)
        combined_relations = tuple(relations) + tuple(fallback_relations)
        if not combined_relations:
            return
        self._solve_relations(combined_relations, rel_tol)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "Reactor":
        """Load a reactor YAML file and return a Reactor instance."""
        from fusdb.loader import load_reactor_yaml

        return load_reactor_yaml(path)

    def __repr__(self) -> str:
        """Return a column-style representation of metadata and parameters."""
        def text(value: object) -> str:
            return "" if value is None else str(value)

        fields = REQUIRED_FIELDS + OPTIONAL_METADATA_FIELDS
        names = sorted(self.parameters, key=lambda val: (val.lower(), val))
        lines = [f"{field}: {text(getattr(self, field, None))}" for field in fields]
        lines += [f"{name}: {text(self.parameters.get(name))}" for name in names]
        return "\n".join(lines)

    def plot_cross_section(
        self,
        ax: Any | None = None,
        *,
        n: int = 256,
        label: str | None = None,
        **plot_kwargs: Any,
    ):
        """Plot a reactor cross-section using a configuration-specific model."""
        config = (self.reactor_configuration or "").lower()
        if "tokamak" not in config:
            raise NotImplementedError(
                f"Cross-section plotting is not implemented for {self.reactor_configuration!r}"
            )

        major_radius = first_numeric(self.parameters, "R")
        minor_radius = first_numeric(self.parameters, "a")
        if major_radius is None or minor_radius is None:
            raise ValueError("Sauter cross-section requires numeric R and a")
        kappa = first_numeric(self.parameters, "kappa", "kappa_95", default=1.0)
        delta = first_numeric(self.parameters, "delta", "delta_95", default=0.0)
        squareness = first_numeric(self.parameters, "squareness", default=0.0)

        from fusdb.relations.geometry.plasma_geometry import sauter_cross_section

        r_vals, z_vals = sauter_cross_section(
            major_radius,
            minor_radius,
            kappa=kappa,
            delta=delta,
            squareness=squareness,
            n=n,
        )

        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        if label is None:
            label = self.id
        ax.plot(r_vals, z_vals, label=label, **plot_kwargs)
        return ax

    def _solve_relations(
        self,
        relations: tuple[Relation, ...],
        rel_tol: float,
    ) -> dict[str, Scalar | None]:
        """Apply a set of relations and update parameter values in place."""
        if not relations:
            return {}

        system = RelationSystem(
            relations,
            rel_tol=rel_tol,
            warn=warnings.warn,
        )
        seen_vars: set[str] = set()
        explicit_params = self.explicit_parameters or set(self.parameters.keys())
        values: dict[str, float] = {}

        for rel in relations:
            for var in rel.variables:
                if var in seen_vars:
                    continue
                seen_vars.add(var)
                if var in self.parameters and var in explicit_params:
                    numeric = as_float(self.parameters[var])
                    if numeric is not None:
                        values[var] = numeric

        for name, value in self.parameter_defaults.items():
            if name in explicit_params or name not in seen_vars or name in values:
                continue
            numeric = as_float(value)
            if numeric is not None:
                values[name] = numeric

        for name in seen_vars:
            if name in explicit_params or name in values:
                continue
            numeric = as_float(self.parameters.get(name))
            if numeric is not None:
                values[name] = numeric

        explicit = {name for name in explicit_params if name in values}
        mode = "override_input" if self.allow_relation_overrides else "locked_input"
        values = system.solve(values, mode=mode, tol=rel_tol, explicit=explicit)
        for var in seen_vars:
            if var in values:
                self.parameters[var] = values[var]

        return values

    @classmethod
    def _ensure_relation_modules_loaded(cls) -> None:
        """Import relation modules once to register decorated relations."""
        if cls._RELATIONS_IMPORTED:
            return
        for module_name in cls._RELATION_MODULES:
            module = importlib.import_module(module_name)
            import_relations = getattr(module, "import_relations", None)
            if callable(import_relations):
                import_relations()
        cls._RELATIONS_IMPORTED = True

    @classmethod
    def relation(
        cls,
        groups: str | tuple[str, ...],
        *,
        name: str,
        output: str | None = None,
        variables: tuple[str, ...] | None = None,
        solve_for: tuple[str, ...] | None = None,
        rel_tol: float = REL_TOL_DEFAULT,
        initial_guesses: dict[str, Any] | None = None,
        constraints: tuple[str | Relational, ...] | None = None,
    ):
        """Decorate a function as a symbolic relation registered on the class."""
        group_tuple = (groups,) if isinstance(groups, str) else tuple(groups)

        def decorator(func):
            """Wrap the function in a Relation and attach it to the class registry."""
            arg_names = tuple(variables) if variables is not None else tuple(inspect.signature(func).parameters.keys())
            output_name = output or func.__name__
            all_vars = (output_name, *arg_names)
            solve_targets = solve_for or all_vars

            arg_syms = [symbol(arg) for arg in arg_names]
            output_sym = symbol(output_name)
            expr = output_sym - func(*arg_syms)
            if constraints is None:
                merged_constraints = []
            elif isinstance(constraints, (str, Relational)):
                merged_constraints = [constraints]
            else:
                merged_constraints = list(constraints)
            relation = Relation(
                name,
                all_vars,
                expr,
                rel_tol=rel_tol,
                solve_for=solve_targets,
                initial_guesses=initial_guesses,
                constraints=tuple(merged_constraints),
            )
            cls._RELATIONS.append((group_tuple, relation))
            setattr(func, "relation", relation)
            return func

        return decorator

    @classmethod
    def get_relations_with_tags(
        cls,
        groups: str | tuple[str, ...],
        *,
        require_all: bool = True,
        exclude: tuple[str, ...] | None = None,
    ) -> tuple[tuple[tuple[str, ...], Relation], ...]:
        """Return relations with their tag tuples for filtering or diagnostics."""
        cls._ensure_relation_modules_loaded()
        requested = (groups,) if isinstance(groups, str) else tuple(groups)
        exclude_set = set(exclude or ())

        matches: list[tuple[tuple[str, ...], Relation]] = []
        seen: set[int] = set()
        for tags, rel in cls._RELATIONS:
            if exclude_set and exclude_set.intersection(tags):
                continue
            if require_all:
                if not all(tag in tags for tag in requested):
                    continue
            else:
                if not any(tag in tags for tag in requested):
                    continue
            if id(rel) in seen:
                continue
            seen.add(id(rel))
            matches.append((tags, rel))
        return tuple(matches)
