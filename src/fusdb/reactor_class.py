"""
Reactor dataclass with relation-based parameter resolution.

This module provides the Reactor class, which represents a fusion reactor
design with its parameters and metadata. The key feature is automatic
parameter resolution through a system of physical relations.

When a Reactor is instantiated:
1. Metadata is validated and normalized
2. Relations are selected based on reactor configuration
3. Parameters are resolved by solving the relation system
4. Conflicts and constraint violations are reported

The Reactor class also provides:
- Attribute-style access to parameters (reactor.R instead of reactor.parameters["R"])
- Cross-section plotting for tokamak configurations
- A decorator for registering new relations

Example:
    >>> reactor = Reactor.from_yaml("path/to/reactor.yaml")
    >>> print(reactor.R)  # Major radius
    9.0
    >>> print(reactor.P_fus)  # Fusion power
    2000000000.0
"""
from dataclasses import dataclass, field
import importlib
import inspect
import warnings
from pathlib import Path
from typing import Any, ClassVar
import sympy as sp
from sympy.core.relational import Relational
from fusdb.relation_class import Relation, RelationSystem
from fusdb.relation_util import REL_TOL_DEFAULT, as_float, first_numeric, symbol
from fusdb.reactor_util import (
    ALLOWED_CONFINEMENT_MODES, ALLOWED_REACTOR_FAMILIES, ALLOWED_REACTOR_CONFIGURATIONS,
    ALLOWED_RELATION_DOMAINS, RELATION_MODULES, REQUIRED_FIELDS, OPTIONAL_METADATA_FIELDS,
    config_exclude_tags, configuration_tags, normalize_allowed, normalize_key,
    parse_solve_strategy, relation_domain_stages, relations_with_tags, select_relations,
)

# Type alias for scalar values (can be numeric or symbolic)
Scalar = float | sp.Expr


@dataclass
class Reactor:
    """
    Reactor metadata and parameter container that resolves relations on load.
    
    The Reactor class is the main data structure for representing fusion reactor
    designs. It stores both metadata (name, organization, configuration) and
    physical parameters (R, a, P_fus, etc.). When instantiated, it automatically
    resolves unknown parameters using the registered physical relations.
    
    Attributes:
        id: Unique identifier for the reactor
        name: Human-readable name
        reactor_configuration: Type of reactor (e.g., "tokamak", "stellarator")
        organization: Organization responsible for the design
        reactor_family: Optional reactor family (e.g., "DEMO", "ITER-like")
        country: Optional ISO 3166-1 alpha-3 country code
        site: Optional site name
        design_year: Optional year of the design
        doi: Optional DOI reference(s) for the design
        notes: Optional notes about the design
        confinement_mode: Optional plasma confinement mode (e.g., "H-mode", "L-mode")
        solve_strategy: Optional solving strategy ('default', 'staged', 'global', or list)
        allow_relation_overrides: If True, relations can override explicit inputs
        parameters: Dictionary of physical parameters
        parameter_tolerances: Per-parameter tolerance overrides
        parameter_methods: Preferred methods for computing specific parameters
        parameter_defaults: Default values for parameters
        fallback_relations: Relations to use if main relations don't apply
        explicit_parameters: Set of parameter names that were explicitly provided
        input_parameters: Original input parameter values (before solving)
        root_dir: Directory containing the reactor YAML file
        relations_used: List of relations that were applied
        _warnings_issued: Set of warning messages (for deduplication)
        parameter_records: Provenance records for each parameter
        
    Class Attributes:
        ALLOWED_CONFINEMENT_MODES: Valid confinement mode values
        ALLOWED_REACTOR_FAMILIES: Valid reactor family values
        ALLOWED_REACTOR_CONFIGURATIONS: Valid configuration values
        _RELATION_MODULES: Modules containing relation definitions
        _RELATIONS: List of (tags, Relation) tuples for registered relations
        _RELATIONS_IMPORTED: Flag indicating if relations have been loaded
        
    Example:
        >>> reactor = Reactor(
        ...     id="test_reactor",
        ...     name="Test Reactor",
        ...     reactor_configuration="tokamak",
        ...     organization="Test Org",
        ...     parameters={"R": 9.0, "a": 3.0}
        ... )
        >>> print(reactor.R)
        9.0
    """
    # Class-level constants
    ALLOWED_CONFINEMENT_MODES: ClassVar[tuple[str, ...]] = ALLOWED_CONFINEMENT_MODES
    ALLOWED_REACTOR_FAMILIES: ClassVar[tuple[str, ...]] = ALLOWED_REACTOR_FAMILIES
    ALLOWED_REACTOR_CONFIGURATIONS: ClassVar[tuple[str, ...]] = ALLOWED_REACTOR_CONFIGURATIONS
    _RELATION_MODULES: ClassVar[tuple[str, ...]] = RELATION_MODULES
    _RELATIONS: ClassVar[list[tuple[tuple[str, ...], Relation]]] = []
    _RELATIONS_IMPORTED: ClassVar[bool] = False

    # Required metadata fields
    id: str = field(metadata={"section": "metadata_required"})
    name: str = field(metadata={"section": "metadata_required"})
    reactor_configuration: str = field(metadata={"section": "metadata_required"})
    organization: str = field(metadata={"section": "metadata_required"})
    
    # Optional metadata fields
    reactor_family: str | None = field(default=None, metadata={"section": "metadata_optional"})
    country: str | None = field(default=None, metadata={"section": "metadata_optional"})
    site: str | None = field(default=None, metadata={"section": "metadata_optional"})
    design_year: int | None = field(default=None, metadata={"section": "metadata_optional"})
    doi: str | list[str] | None = field(default=None, metadata={"section": "metadata_optional"})
    notes: str | None = field(default=None, metadata={"section": "metadata_optional"})
    confinement_mode: str | None = field(default=None, metadata={"section": "metadata_optional"})
    solve_strategy: str | list[str] | None = field(default=None, metadata={"section": "metadata_optional"})
    allow_relation_overrides: bool | None = field(default=True, metadata={"section": "metadata_optional"})
    
    # Parameter storage
    parameters: dict[str, Scalar | None] = field(default_factory=dict)
    parameter_tolerances: dict[str, float] = field(default_factory=dict)
    parameter_methods: dict[str, str] = field(default_factory=dict)
    parameter_defaults: dict[str, Scalar] = field(default_factory=dict)
    fallback_relations: tuple[Relation, ...] = field(default_factory=tuple)
    explicit_parameters: set[str] = field(default_factory=set)
    input_parameters: dict[str, Scalar | None] = field(default_factory=dict)
    
    # Internal fields
    root_dir: Path | None = field(default=None, metadata={"section": "internal"})
    relations_used: list[tuple[tuple[str, ...], Relation]] = field(default_factory=list, metadata={"section": "internal"}, repr=False)
    _warnings_issued: set[str] = field(default_factory=set, metadata={"section": "internal"}, repr=False)
    parameter_records: dict[str, dict] = field(default_factory=dict, metadata={"section": "internal"}, repr=False)

    def __post_init__(self):
        """
        Initialize the reactor after dataclass construction.
        
        This method:
        1. Sets default values for optional fields
        2. Normalizes metadata values to canonical forms
        3. Determines the solving strategy
        4. Applies relations in stages to resolve unknown parameters
        5. Records which relations were used
        """
        # Initialize optional fields to empty containers if None
        for attr in ('parameter_methods', 'parameter_defaults'):
            if getattr(self, attr) is None:
                setattr(self, attr, {})
        if self.fallback_relations is None:
            self.fallback_relations = ()
        if self.explicit_parameters is None:
            self.explicit_parameters = set()
        if self.allow_relation_overrides is None:
            self.allow_relation_overrides = True
        
        # Store original input parameters
        if not self.input_parameters:
            self.input_parameters = {
                n: self.parameters.get(n)
                for n in self.explicit_parameters
                if n in self.parameters
            }
        self.relations_used = []
        
        # Normalize metadata values
        self.reactor_family = normalize_allowed(
            self.reactor_family, self.ALLOWED_REACTOR_FAMILIES, field_name="reactor_family"
        )
        self.reactor_configuration = normalize_allowed(
            self.reactor_configuration, self.ALLOWED_REACTOR_CONFIGURATIONS, field_name="reactor_configuration"
        )
        if self.confinement_mode and self.ALLOWED_CONFINEMENT_MODES:
            self.confinement_mode = normalize_allowed(
                self.confinement_mode, self.ALLOWED_CONFINEMENT_MODES, field_name="confinement_mode"
            )
        
        # Set up solving configuration
        rel_tol = REL_TOL_DEFAULT
        config_tags = configuration_tags(self.reactor_configuration)
        
        # Determine which relations to exclude based on configuration
        exclude = set(config_exclude_tags(self.reactor_configuration))
        if self.confinement_mode:
            for m in self.ALLOWED_CONFINEMENT_MODES:
                if m != self.confinement_mode:
                    exclude.add(m)
        exclude = tuple(sorted(exclude))
        
        # Parse solving strategy
        strategy, steps = parse_solve_strategy(self.solve_strategy)

        # Apply relations according to strategy
        if strategy in ("default", "staged"):
            # Staged solving: apply relations domain by domain
            for grps in relation_domain_stages():
                self._apply(
                    tuple(relations_with_tags(grps, require_all=False, exclude=exclude)),
                    rel_tol, config_tags
                )
        elif strategy == "global":
            # Global solving: apply all relations at once
            self._apply(
                tuple(relations_with_tags(ALLOWED_RELATION_DOMAINS, require_all=False, exclude=exclude)),
                rel_tol, config_tags
            )
        elif strategy == "user":
            # User-defined steps
            if not steps:
                raise ValueError("solve_strategy set to 'user' but no steps provided")
            
            dom_lk = {d.lower(): d for d in ALLOWED_RELATION_DOMAINS}
            self._ensure_loaded()
            
            # Build name lookup for relations
            nm = {}
            for tags, rel in self._RELATIONS:
                nm.setdefault(normalize_key(rel.name), []).append((tags, rel))
            
            for step in steps:
                s = step.strip()
                if not s:
                    raise ValueError("solve_strategy steps must be non-empty")
                
                # Check if step is a domain name
                if d := dom_lk.get(s.lower()):
                    self._apply(
                        tuple(relations_with_tags(d, require_all=False, exclude=exclude)),
                        rel_tol, config_tags
                    )
                    continue
                
                # Otherwise, look up by relation name
                k = normalize_key(s)
                if not (m := nm.get(k)):
                    raise ValueError(f"solve_strategy step {s!r} did not match a domain or relation name")
                
                for tags, rel in m:
                    if exclude and set(exclude) & set(tags):
                        warnings.warn(
                            f"Relation {rel.name!r} selected but excluded for {self.reactor_configuration!r}.",
                            UserWarning
                        )
                self._apply(tuple(m), rel_tol, config_tags)
        else:
            raise ValueError("solve_strategy must be 'default', 'global', or a list")
        
        # Final pass with all relations (for staged/default)
        if strategy in ("default", "staged"):
            self._apply(
                tuple(relations_with_tags(ALLOWED_RELATION_DOMAINS, require_all=False, exclude=exclude)),
                rel_tol, config_tags
            )

    def __getattr__(self, name: str) -> Any:
        """
        Allow attribute-style access to parameters.
        
        Args:
            name: Attribute name to access
            
        Returns:
            The parameter value if found
            
        Raises:
            AttributeError: If not found in parameters
        """
        params = self.__dict__.get("parameters", {})
        if name in params:
            return params[name]
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def __setattr__(self, name: str, value: Any):
        """
        Allow attribute-style setting of parameters.
        
        Internal/dataclass fields are set normally; other names go to parameters.
        
        Args:
            name: Attribute name to set
            value: Value to set
        """
        if name.startswith("_") or name in type(self).__dataclass_fields__:
            object.__setattr__(self, name, value)
            return
        self.__dict__.setdefault("parameters", {})[name] = value

    def __dir__(self) -> list[str]:
        """Include parameter names in dir() output."""
        return sorted(set(super().__dir__()) | set(self.__dict__.get("parameters", {}).keys()))

    def _apply(
        self,
        tagged: tuple[tuple[tuple[str, ...], Relation], ...],
        rel_tol: float,
        config_tags: tuple[str, ...]
    ):
        """
        Apply a set of tagged relations to resolve parameters.
        
        Selects applicable relations based on configuration and parameter_methods,
        adds fallback relations if needed, then solves the combined system.
        
        Args:
            tagged: Tuple of (tags, Relation) pairs
            rel_tol: Relative tolerance for solving
            config_tags: Tags derived from reactor configuration
        """
        # Select relations based on configuration and method preferences
        rels = select_relations(
            tagged,
            parameter_methods=self.parameter_methods,
            config_tags=config_tags,
            warn=warnings.warn
        )
        
        # Record which relations are being used
        if rels:
            self.relations_used.extend((t, r) for t, r in tagged if r in set(rels))
        
        # Add fallback relations for parameters not covered or not yet solved
        fb = []
        if self.fallback_relations:
            base = {v for r in rels for v in r.variables if r.variables}
            exp = self.explicit_parameters or set()
            for r in self.fallback_relations:
                if not r.variables:
                    continue
                # Add fallback if:
                # 1. Any variable is missing from base rels and not explicit, OR
                # 2. The output variable (first) has no value yet and can be propagated
                missing = [v for v in r.variables if v not in base and v not in exp]
                output_var = r.variables[0]
                output_unresolved = self.parameters.get(output_var) is None and output_var not in exp
                
                if missing or output_unresolved:
                    # Check if at least one non-output variable is in base relations (will be solved)
                    # or already has a value
                    other_vars = r.variables[1:] if len(r.variables) > 1 else r.variables
                    has_source = any(
                        self.parameters.get(v) is not None or v in base
                        for v in other_vars
                    )
                    if has_source:
                        fb.append(r)
        
        if fb:
            self.relations_used.extend((("fallback",), r) for r in fb)
        
        # Solve the combined system
        comb = tuple(rels) + tuple(fb)
        if comb:
            self._solve(comb, rel_tol)
        
        # Update fractions for ash buildup when tau_p is available
        self._update_composition()

    def _update_composition(self):
        """Update ion fractions using steady-state composition solver.
        
        When particle confinement time (tau_p) is available, computes
        consistent fractions accounting for ash buildup from fusion reactions.
        """
        from fusdb.relations.plasma_composition import solve_steady_state_composition
        
        tau_p = self.parameters.get("tau_p")
        if tau_p is None or tau_p <= 0:
            return
        
        n_i = self.parameters.get("n_i") or self.parameters.get("n_avg")
        T_avg = self.parameters.get("T_avg") or self.parameters.get("T_i") or self.parameters.get("T_e")
        
        if n_i is None or T_avg is None:
            return
        
        fractions = {
            "f_D": self.parameters.get("f_D", 0.5),
            "f_T": self.parameters.get("f_T", 0.5),
            "f_He3": self.parameters.get("f_He3", 0.0),
            "f_He4": self.parameters.get("f_He4", 0.0),
        }
        
        ss = solve_steady_state_composition(n_i, T_avg, fractions, tau_p)
        
        explicit = self.explicit_parameters or set()
        for key in ("f_D", "f_T", "f_He3", "f_He4"):
            if key not in explicit:
                self.parameters[key] = ss[key]

    @classmethod
    def from_yaml(cls, path: Path | str) -> "Reactor":
        """
        Load a Reactor from a YAML file using the YAML loader

        Args:
            path: Path to the reactor YAML file
            
        Returns:
            A new Reactor object with formatted parameters
        """
        from fusdb.loader import load_reactor_yaml
        return load_reactor_yaml(path)

    def __repr__(self) -> str:
        """Return a detailed string representation of the reactor."""
        def txt(v):
            return "" if v is None else str(v)
        
        fs = REQUIRED_FIELDS + OPTIONAL_METADATA_FIELDS
        ns = sorted(self.parameters, key=lambda v: (v.lower(), v))
        return "\n".join(
            [f"{f}: {txt(getattr(self, f, None))}" for f in fs] +
            [f"{n}: {txt(self.parameters.get(n))}" for n in ns]
        )

    def plot_cross_section(
        self,
        ax: Any | None = None,
        *,
        n: int = 256,
        label: str | None = None,
        **kw
    ):
        """
        Plot the plasma cross-section for tokamak configurations.
        
        Uses the Sauter parametrization to compute the plasma boundary
        shape based on major radius R, minor radius a, elongation kappa,
        triangularity delta, and squareness.
        
        Args:
            ax: Matplotlib axes to plot on (creates new if None)
            n: Number of points for the boundary curve
            label: Label for the plot legend
            **kw: Additional kwargs passed to ax.plot()
            
        Returns:
            The matplotlib axes object
            
        Raises:
            NotImplementedError: If reactor is not a tokamak
            ValueError: If R or a are not available
        """
        cfg = (self.reactor_configuration or "").lower()
        if "tokamak" not in cfg:
            raise NotImplementedError(f"Cross-section not implemented for {self.reactor_configuration!r}")
        
        R = first_numeric(self.parameters, "R")
        a = first_numeric(self.parameters, "a")
        if R is None or a is None:
            raise ValueError("Sauter cross-section requires numeric R and a")
        
        kappa = first_numeric(self.parameters, "kappa", "kappa_95", default=1.0)
        delta = first_numeric(self.parameters, "delta", "delta_95", default=0.0)
        sq = first_numeric(self.parameters, "squareness", default=0.0)
        
        from fusdb.relations.geometry.plasma_geometry import sauter_cross_section
        rv, zv = sauter_cross_section(R, a, kappa=kappa, delta=delta, squareness=sq, n=n)
        
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(rv, zv, label=label or self.id, **kw)
        return ax

    def _solve(self, rels: tuple[Relation, ...], rel_tol: float) -> dict[str, Scalar | None]:
        """
        Solve a set of relations to compute unknown parameters.
        
        Creates a RelationSystem with the given relations and solves it
        using known parameter values. Updates self.parameters and
        self.parameter_records with the results.
        
        Args:
            rels: Tuple of Relation objects to solve
            rel_tol: Relative tolerance for the solver
            
        Returns:
            Dictionary of computed values
        """
        if not rels:
            return {}
        
        self._ensure_loaded()
        
        # Create the solver system
        sys = RelationSystem(
            rels,
            rel_tol=rel_tol,
            warn=warnings.warn,
            warnings_issued=self._warnings_issued,
            all_relations=[r for _, r in self._RELATIONS]
        )
        
        # Build initial values from explicit parameters
        seen = set()
        exp = self.explicit_parameters or set(self.parameters.keys())
        vals = {}
        
        for rel in rels:
            for v in rel.variables:
                if v in seen:
                    continue
                seen.add(v)
                if v in self.parameters and v in exp:
                    n = as_float(self.parameters[v])
                    if n is not None:
                        vals[v] = n
        
        # Add defaults for non-explicit parameters
        for n, val in self.parameter_defaults.items():
            if n in exp or n not in seen or n in vals:
                continue
            num = as_float(val)
            if num is not None:
                vals[n] = num
        
        # Add non-explicit current values
        for n in seen:
            if n in exp or n in vals:
                continue
            num = as_float(self.parameters.get(n))
            if num is not None:
                vals[n] = num
        
        # Solve the system
        mode = "override_input" if self.allow_relation_overrides else "locked_input"
        vals = sys.solve(vals, mode=mode, tol=rel_tol, explicit={n for n in exp if n in vals})
        
        # Update parameters
        for v in seen:
            if v in vals:
                self.parameters[v] = vals[v]
        
        # Merge parameter records (with priority for more severe statuses)
        pri = {"constraint": 3, "conflict": 2, "inconsistent": 1, "consistent": 0}
        for v, rec in sys.get_parameter_records().items():
            if v not in self.parameter_records:
                self.parameter_records[v] = rec
            else:
                ex = self.parameter_records[v]
                ex["final_value"] = rec["final_value"]
                if pri.get(rec["status"], 0) > pri.get(ex["status"], 0):
                    ex["status"] = rec["status"]
                if rec.get("constraint_limit"):
                    ex["constraint_limit"] = rec["constraint_limit"]
        
        return vals

    @classmethod
    def _ensure_loaded(cls):
        """
        Ensure all relation modules have been imported.
        
        Called before using relations to ensure all registered
        relation decorators have been executed.
        """
        if cls._RELATIONS_IMPORTED:
            return
        for m in cls._RELATION_MODULES:
            mod = importlib.import_module(m)
            if callable(f := getattr(mod, "import_relations", None)):
                f()
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
        constraints: tuple[str | Relational, ...] | None = None
    ):
        """
        Decorator to register a function as a relation.
        
        The decorated function should compute the output variable from
        the input variables. The relation is automatically converted to
        implicit form (output - function(...) = 0).
        
        Args:
            groups: Domain tag(s) for the relation (e.g., "geometry", "power_balance")
            name: Human-readable name for the relation
            output: Output variable name (defaults to function name)
            variables: Input variable names (defaults to function parameter names)
            solve_for: Variables that can be solved for
            rel_tol: Relative tolerance for the relation
            initial_guesses: Initial guesses for numeric solving
            constraints: Constraints on the variables
            
        Returns:
            Decorator function
            
        Example:
            >>> @Reactor.relation("power_balance", name="Ohm's Law")
            ... def P(V, I):
            ...     return V * I
        """
        grp = (groups,) if isinstance(groups, str) else tuple(groups)
        
        def dec(func):
            # Get variable names from function signature if not provided
            args = tuple(variables) if variables else tuple(inspect.signature(func).parameters.keys())
            out = output or func.__name__
            allv = (out, *args)
            tgts = solve_for or (out, *args)
            
            # Create symbolic expression
            asym = [symbol(a) for a in args]
            expr = symbol(out) - func(*asym)
            
            # Process constraints
            cons = []
            if constraints is None:
                pass
            elif isinstance(constraints, (str, Relational)):
                cons = [constraints]
            else:
                cons = list(constraints)
            
            # Create and register the relation
            rel = Relation(
                name, allv, expr,
                rel_tol=rel_tol,
                solve_for=tgts,
                initial_guesses=initial_guesses,
                constraints=tuple(cons)
            )
            cls._RELATIONS.append((grp, rel))
            
            # Attach relation to function for introspection
            func.relation = rel
            return func
        
        return dec

    @classmethod
    def get_relations_with_tags(
        cls,
        groups: str | tuple[str, ...],
        *,
        require_all: bool = True,
        exclude: tuple[str, ...] | None = None
    ) -> tuple[tuple[tuple[str, ...], Relation], ...]:
        """
        Get registered relations matching the given tags.
        
        Args:
            groups: Tag(s) to match
            require_all: If True, relation must have all tags; if False, any tag matches
            exclude: Tags that disqualify a relation
            
        Returns:
            Tuple of (tags, Relation) pairs for matching relations
        """
        cls._ensure_loaded()
        req = (groups,) if isinstance(groups, str) else tuple(groups)
        excl = set(exclude or ())
        matches = []
        seen = set()
        
        for tags, rel in cls._RELATIONS:
            # Skip if excluded
            if excl and excl & set(tags):
                continue
            # Check tag matching
            if require_all and not all(t in tags for t in req):
                continue
            if not require_all and not any(t in tags for t in req):
                continue
            # Skip duplicates
            if id(rel) in seen:
                continue
            seen.add(id(rel))
            matches.append((tags, rel))
        
        return tuple(matches)
