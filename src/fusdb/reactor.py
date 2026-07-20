"""User-facing reactor scenario object."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .modes import MODE_NAMES
from .registry import RELATIONS, TAGS, VARIABLES
from .relationsystem import RelationSystem
from .plotting.tables import SolvedColumn, _table_column
from .variable import Variable


@dataclass(frozen=True)
class SolvedVariable:
    """Read-through view pairing a frozen declaration with its solved value.

    Returned by :meth:`Reactor.get_variable` and attribute access
    (``reactor.<name>``).  ``declared`` is the :class:`Variable` exactly as
    supplied -- a solve never mutates it.  ``value`` is the latest solved
    value from the reactor's :attr:`Reactor.last_system` when one is active
    for this name, falling back to the declared value otherwise; it is
    resolved fresh on every access, so it always reflects the most recent
    run.  Every other attribute (``unit``, ``fixed``, ``rel_tol``, ``spec``,
    ...) delegates to ``declared``.  :meth:`clone` builds a new declaration,
    exactly like :meth:`Variable.clone`.
    """

    declared: Variable
    _system: RelationSystem | None

    @property
    def value(self) -> Any:
        if self._system is not None:
            solved = self._system.values.get(self.declared.name)
            if solved is not None:
                return solved
        return self.declared.value

    def clone(self, **changes: Any) -> Variable:
        return self.declared.clone(**changes)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.declared, name)

# Threshold-driven confinement-mode selection is a discrete outer problem:
# each candidate mode gets its own relation graph and therefore its own
# confinement-time scaling.  Everything regime-specific is discovered from the
# registries rather than hardcoded here:
#   * the regimes themselves are the ``confinement_mode`` tag group, listed in
#     allowed_tags.yaml in threshold-escalation order;
#   * a regime's sustainment guards are the relations tagged
#     ``("regime_guard", <regime>)``;
#   * a regime's fallback confinement-time scaling is the tau_E producer
#     tagged ``("regime_default", <regime>)``.
_REGIME_SOLVE_MODES = {"reconcile", "optimize", "popcon"}


def _regime_order() -> tuple[str, ...]:
    """Confinement regimes in threshold-escalation order (allowed_tags.yaml)."""
    return tuple(str(tag) for tag in (TAGS.raw.get("confinement_mode") or ()))


def _regime_guard_names(regime: str | None) -> tuple[str, ...]:
    """Sustainment guard relation names of one regime, in registration order."""
    if not regime:
        return ()
    return tuple(rel.name for rel in RELATIONS if "regime_guard" in rel.tags and regime in rel.tags)


def _all_regime_guard_names() -> tuple[str, ...]:
    """All sustainment guard relation names, in registration order."""
    return tuple(rel.name for rel in RELATIONS if "regime_guard" in rel.tags)


def _regime_tau_default_name(regime: str) -> str | None:
    """Name of the regime's fallback tau_E scaling, or None when undeclared."""
    for rel in RELATIONS:
        if "regime_default" in rel.tags and regime in rel.tags and "tau_E" in rel.output_names:
            return rel.name
    return None


def _confinement_regime(tags: Iterable[str]) -> str | None:
    """Return the first declared confinement-mode tag, if any."""
    order = _regime_order()
    return next((tag for tag in tags if tag in order), None)


def _unique_extend(base: Iterable[str], extra: Iterable[str]) -> tuple[str, ...]:
    """Append strings while preserving first occurrence order."""
    out: list[str] = []
    for item in (*tuple(base), *tuple(extra)):
        if item not in out:
            out.append(item)
    return tuple(out)


def _with_sustainment_guards(includes: tuple[str, ...], regime: str | None) -> tuple[str, ...]:
    """Append checked-only sustainment guards for ``regime``."""
    return _unique_extend(includes, _regime_guard_names(regime))


def _with_confinement_regime(tags: tuple[str, ...], regime: str) -> tuple[str, ...]:
    """Replace any confinement-mode tags with exactly ``regime``."""
    order = _regime_order()
    out: list[str] = []
    inserted = False
    for tag in tags:
        if tag in order:
            if not inserted:
                out.append(regime)
                inserted = True
            continue
        out.append(tag)
    if not inserted:
        out.append(regime)
    return tuple(out)


def _regime_warning(old: str, new: str, mode: str) -> str:
    """User-facing warning for automatic regime correction."""
    return (
        f"Declared {old} operating condition is inconsistent with confinement-mode thresholds; "
        f"switched to {new} for {mode}."
    )


def _bistable_warning(fallback: str, mode: str) -> str:
    """User-facing warning when no candidate regime is self-consistent."""
    return (
        "No confinement regime is self-consistent (each candidate's own solve "
        "violates its sustainment guard -- the L-H bistable/dithering band); "
        f"settled on {fallback} as the accessible regime for {mode}."
    )


def _candidate_regimes(declared: str | None) -> tuple[str, ...]:
    """Candidate regimes in preference order: declared first, then escalation.

    Candidates cover *both* directions from the declared tag -- an ohmic or
    L-mode machine whose ``P_sep`` crosses a threshold escalates, an H/I-mode
    machine below its threshold de-escalates.  The declared tag picks the
    machine's upper branch: H-mode and I-mode are alternative upper regimes
    selected by machine operation (topology, drift direction), not by heating
    power alone, so an ``h_mode`` machine never auto-selects ``i_mode`` and
    vice versa.  A machine declared ``l_mode``/``ohmic_mode`` can escalate
    into either branch; self-consistency of each candidate's own solve (its
    guards evaluated on its own values) disambiguates, ties broken by this
    order.  Regimes with no registered guards are dropped: self-consistency
    would be undefined for them.
    """
    if declared not in _regime_order():
        return ()
    chains = {
        "h_mode": ("h_mode", "l_mode", "ohmic_mode"),
        "i_mode": ("i_mode", "l_mode", "ohmic_mode"),
        "l_mode": ("l_mode", "h_mode", "i_mode", "ohmic_mode"),
        "ohmic_mode": ("ohmic_mode", "l_mode", "h_mode", "i_mode"),
    }
    chain = chains.get(declared, (declared,))
    return tuple(regime for regime in chain if _regime_guard_names(regime))


def _regime_verified_by_guards(statuses: Mapping[str, Mapping[str, Any]], regime: str) -> bool:
    """Whether every guard for ``regime`` is present and verified."""
    guards = _regime_guard_names(regime)
    return bool(guards) and all(bool((statuses.get(guard) or {}).get("verified", False)) for guard in guards)


def _regime_guards_indeterminate(statuses: Mapping[str, Mapping[str, Any]], regime: str) -> bool:
    """Whether ``regime``'s guards could not be evaluated (missing variables).

    A guard is indeterminate when its inputs are absent from the solved values
    -- e.g. a popcon scan restores itself to pure inputs, so a guard over a
    *derived* quantity like ``P_sep`` cannot be evaluated afterwards.  That is
    not a regime inconsistency (the per-point certification is the real
    arbiter), so switching/failure must not be driven by it.  Only guards that
    genuinely evaluated to a violated residual drive a switch.
    """
    guards = _regime_guard_names(regime)
    if not guards:
        return False
    for guard in guards:
        status = statuses.get(guard) or {}
        if status.get("verified", False):
            return False  # at least one guard did evaluate and hold
        errors = status.get("errors") or []
        if not any("missing variables" in str(error) for error in errors):
            return False  # a guard evaluated to a genuine violation
    return True


def _regime_guard_input_names(chain: Iterable[str]) -> set[str]:
    """Canonical variables every sustainment guard in ``chain`` reads.

    These (``P_sep``, ``P_LH``, ``P_OL_thresh``, ``P_LI_thresh``) must be produced
    by each per-regime scan so the guards can classify every grid point.
    """
    names: set[str] = set()
    for regime in chain:
        for guard in _regime_guard_names(regime):
            for inp in RELATIONS.get(guard).input_names:
                try:
                    names.add(VARIABLES.resolve(inp))
                except Exception:
                    names.add(inp)
    return names


def _regime_guard_holds(regime: str, fields: Mapping[str, Any], shape: tuple[int, ...]) -> np.ndarray:
    """Boolean grid: every sustainment guard of ``regime`` holds on ``fields``.

    Each guard returns a normalized residual that is 0 exactly when the regime is
    sustainable (e.g. h_mode's ``max(P_LH - P_sep, 0)/scale`` is 0 iff
    ``P_sep >= P_LH``), so the regime holds where every guard residual is ~0. A
    guard whose inputs are missing marks the point as not in this regime.
    """
    mask = np.ones(shape, dtype=bool)
    for guard in _regime_guard_names(regime):
        rel = RELATIONS.get(guard)
        args: dict[str, Any] = {}
        for inp in rel.input_names:
            value = fields.get(inp)
            if value is None:
                return np.zeros(shape, dtype=bool)
            args[inp] = value
        residual = np.asarray(rel.func(**args), dtype=float)
        mask &= np.isfinite(residual) & (residual <= 1.0e-9)
    return mask


def _copy_value(value: Any) -> Any:
    """Copy scalar/profile values used to clone reactor variables."""
    if isinstance(value, np.ndarray):
        return value.copy()
    return value


def _resolve_reactor_yaml(path_like: str | Path) -> Path:
    """Resolve a reactor path to a ``reactor.yaml`` file.

    Args:
        path_like: File or directory path.

    Returns:
        Concrete YAML path.
    """
    path = Path(path_like)
    if path.is_dir():
        path = path / "reactor.yaml"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _load_variable_file(path_like: Path, *, base_dir: Path, delimiter: str | None = None, usecols: Any = None, skiprows: int = 0) -> np.ndarray:
    """Load one numeric variable/profile file.

    Args:
        path_like: Absolute or reactor-relative file path.
        base_dir: Reactor directory.
        delimiter: Optional delimiter.
        usecols: Optional columns forwarded to ``numpy.loadtxt``.
        skiprows: Number of header rows.

    Returns:
        One-dimensional numeric array.
    """
    path = path_like.expanduser()
    if not path.is_absolute():
        path = base_dir / path
    if not path.is_file():
        raise FileNotFoundError(f"Variable data file {str(path_like)!r} does not exist.")
    delimiters = (delimiter,) if delimiter is not None else (None, ",", ";")
    last_error: Exception | None = None
    for candidate in delimiters:
        try:
            data = np.loadtxt(path, delimiter=candidate, usecols=usecols, skiprows=int(skiprows))
            break
        except Exception as exc:
            last_error = exc
    else:
        raise ValueError(f"Could not load numeric variable data from {path.name!r}: {last_error}")
    array = np.asarray(data, dtype=float)
    if array.ndim == 0:
        return np.asarray([float(array)], dtype=float)
    if array.ndim == 1:
        return array.astype(float)
    if array.ndim == 2 and usecols is None:
        return array[:, -1].astype(float)
    if array.ndim == 2 and array.shape[1] == 1:
        return array[:, 0].astype(float)
    raise ValueError(f"Variable data file {path.name!r} produced a non-1D value array.")


def _parse_variables(raw: Mapping[str, Any], *, grid_size: int | None, base_dir: Path) -> dict[str, Variable]:
    """Parse reactor YAML variables.

    Args:
        raw: Raw ``variables`` mapping.
        grid_size: Optional default profile size.
        base_dir: Base path for relative profile files.

    Returns:
        Variables keyed by canonical name.
    """
    variables: dict[str, Variable] = {}
    for raw_name, entry in raw.items():
        if entry is None:
            entry = {}
        if not isinstance(entry, Mapping):
            entry = {"value": entry}
        spec = VARIABLES.get(str(raw_name))
        value = entry.get("value")
        size = entry.get("size", grid_size if spec.shape == 1 else None)
        file_value = entry.get("file")
        if file_value is None and isinstance(value, str) and spec.shape == 1:
            try:
                float(value.strip())
            except ValueError:
                candidate = Path(value).expanduser()
                if not candidate.is_absolute():
                    candidate = base_dir / candidate
                if candidate.is_file():
                    file_value = value
        if file_value is not None:
            value = _load_variable_file(Path(str(file_value)), base_dir=base_dir, delimiter=entry.get("delimiter"), usecols=entry.get("usecols"), skiprows=entry.get("skiprows", 0))
            if np.asarray(value).ndim == 1:
                size = int(np.asarray(value).shape[0])
        var = Variable(str(raw_name), value=value, unit=entry.get("unit"), rel_tol=entry.get("rel_tol"), fixed=bool(entry.get("fixed", False)), size=size, constraints=entry.get("constraints"))
        variables[var.name] = var
    return variables


@dataclass
class Reactor:
    """A reactor scenario with variables and relation-selection settings."""

    name: str
    organization: str | None = None
    country: str | None = None
    year: int | None = None
    doi: str | None = None
    notes: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)
    variables: dict[str, Variable] = field(default_factory=dict)
    relation_include: tuple[str, ...] = field(default_factory=tuple)
    relation_exclude: tuple[str, ...] = field(default_factory=tuple)
    relation_order: tuple[Any, ...] = field(default_factory=tuple)
    constraints: Any = None
    grid_size: int | None = None

    def __post_init__(self) -> None:
        """Normalize simple user-facing fields."""
        # System produced by the most recent run(); displayed in place of the
        # loaded inputs so a solved reactor shows its reconciled values. None
        # until a mode has run. Set here so __getattr__ never intercepts it.
        self.last_system: RelationSystem | None = None
        # YAML this reactor was loaded from, if any; lets solve_reactors ship it
        # to a worker process. Set by from_yaml. Here so __getattr__ skips it.
        self.source_path: Path | None = None
        self.tags = tuple(str(tag).strip().lower() for tag in self.tags)
        self.relation_include = tuple(str(name) for name in (self.relation_include or ()))
        self.relation_exclude = tuple(str(name) for name in (self.relation_exclude or ()))
        self.relation_order = tuple(self.relation_order or ())
        self.variables = {var.name: var for var in self.variables.values()}
        if self.grid_size is not None:
            self.grid_size = int(self.grid_size)
            if self.grid_size <= 0:
                raise ValueError("grid.size must be positive.")

    @classmethod
    def from_yaml(cls, path_like: str | Path) -> "Reactor":
        """Load a reactor scenario from YAML.

        Args:
            path_like: Reactor directory or YAML file.

        Returns:
            Reactor instance.
        """
        path = _resolve_reactor_yaml(path_like)
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, Mapping):
            raise TypeError("reactor.yaml must contain a mapping.")
        metadata = raw.get("metadata", {}) or {}
        grid = raw.get("grid", {}) or {}
        grid_size = grid.get("size") if isinstance(grid, Mapping) else None
        relation_spec = raw.get("relations", {}) or {}
        reactor = cls(
            name=str(metadata.get("name") or metadata.get("id") or path.parent.name),
            organization=metadata.get("organization"),
            country=metadata.get("country"),
            year=metadata.get("year"),
            doi=metadata.get("doi"),
            notes=metadata.get("notes"),
            tags=tuple(raw.get("tags", ()) or ()),
            variables=_parse_variables(raw.get("variables", {}) or {}, grid_size=grid_size, base_dir=path.parent),
            relation_include=tuple(relation_spec.get("include", ()) or ()) if isinstance(relation_spec, Mapping) else (),
            relation_exclude=tuple(relation_spec.get("exclude", ()) or ()) if isinstance(relation_spec, Mapping) else (),
            relation_order=tuple(relation_spec.get("order", ()) or ()) if isinstance(relation_spec, Mapping) else (),
            constraints=raw.get("constraints"),
            grid_size=grid_size,
        )
        reactor.source_path = path
        return reactor

    def add_variable(self, var: Variable) -> None:
        """Add or replace one variable.

        Args:
            var: Variable to add.
        """
        if isinstance(var, SolvedVariable):
            raise TypeError(
                "add_variable() expects a Variable declaration, not a SolvedVariable "
                "read-through view. Use var.clone(...) to build a new declaration "
                "(optionally seeded from var.value), or var.declared to reuse the "
                "original declaration unchanged."
            )
        self.variables[var.name] = var

    def get_variable(self, name: str) -> SolvedVariable | None:
        """Return one loaded variable, as a read-through solved/declared view.

        Args:
            name: Canonical name or alias.

        Returns:
            :class:`SolvedVariable` pairing the frozen declaration with its
            latest solved value (from :attr:`last_system`, when active for
            this name), or ``None`` if no such variable was declared.
        """
        try:
            canonical = VARIABLES.resolve(name)
        except Exception:
            canonical = str(name)
        var = self.variables.get(canonical)
        if var is None:
            return None
        return SolvedVariable(var, self.last_system)

    def __getattr__(self, name: str) -> "SolvedVariable":
        """Expose loaded variables through attribute access."""
        var = self.get_variable(name)
        if var is not None:
            return var
        raise AttributeError(name)

    def relations(self) -> tuple[Any, ...]:
        """Return post-filter relation objects.

        Returns:
            Tuple of relations selected for this reactor.
        """
        return RELATIONS.get_filtered_relations(names=self.relation_include, tags=TAGS.expand(self.tags), exclude=self.relation_exclude, order=None)

    def relation_system(self) -> RelationSystem:
        """Build a RelationSystem for this reactor.

        Returns:
            RelationSystem instance.
        """
        # No clone: the system ingests the records into its own value dicts,
        # which is the isolation copy (values are replaced, never mutated).
        return RelationSystem(
            self.variables.values(),
            self.relations(),
            constraints=self.constraints,
            name=self.name,
        )

    def run(self, mode: str = "verify", **options: Any) -> dict[str, Any]:
        """Build a RelationSystem, run one mode, and keep the solved system.

        Declared variables (:attr:`variables`) are never changed by a run --
        a solve produces a new :class:`RelationSystem`, kept on
        :attr:`last_system`, and ``reactor.<var>`` (via :meth:`get_variable`)
        reads through to its solved values without mutating the declaration.
        A later ``reactor.run(...)`` therefore starts from the same
        declarations again, not from the previous solution; call
        :meth:`restart_from_solution` first to opt into that explicitly.

        Args:
            mode: Execution mode (``verify``, ``reconcile``, ``optimize``,
                ``popcon``, or ``ordered``); defaults to ``verify``.
            **options: Mode-specific options.

        Returns:
            Result dictionary.
        """
        chosen = str(mode or "verify")
        if chosen not in MODE_NAMES:
            raise ValueError(f"Unsupported reactor mode {chosen!r}.")
        # ``save`` archives the final (possibly regime-composed) result, so it
        # is handled here rather than forwarded to per-attempt system runs.
        save = options.pop("save", None)
        if chosen == "verify":
            result = self._run_guarded_once(chosen, **options)
        elif chosen == "popcon":
            result = self._run_popcon_auto_regime(**options)
        elif chosen in _REGIME_SOLVE_MODES:
            result = self._run_with_regime_verification(chosen, **options)
        else:
            result = self._run_once(chosen, **options)
        if save is not None:
            from .io import save_result

            save_result(result, save)
        return result

    def _run_once(self, mode: str, **options: Any) -> dict[str, Any]:
        """Run one mode with the currently configured relation set."""
        chosen = mode
        system = self.relation_system()
        self.last_system = system
        if chosen == "ordered":
            result = system.ordered(order=self.relation_order or None, **options)
        else:
            result = system.run(chosen, **options)
        return result

    def restart_from_solution(self) -> None:
        """Replace each declared variable's value with its latest solved value.

        The explicit form of what earlier fusdb versions did implicitly after
        every solve: nothing is mutated in place -- each affected declaration
        is replaced by a fresh one (:meth:`Variable.clone`) built from
        :attr:`last_system`'s solved value, so the *next* ``run()`` starts
        from where this one ended.  Fixed variables are unaffected (their
        solved value already equals their declared one).  A no-op if this
        reactor has not been run yet.
        """
        if self.last_system is None:
            return
        for name, value in self.last_system.values.items():
            var = self.variables.get(name)
            if var is not None and value is not None:
                self.variables[name] = var.clone(value=value)

    def _run_guarded_once(self, mode: str, **options: Any) -> dict[str, Any]:
        """Run one mode with the current regime sustainment guard included."""
        base_include = self.relation_include
        self.relation_include = _with_sustainment_guards(base_include, _confinement_regime(self.tags))
        try:
            return self._run_once(mode, **options)
        finally:
            self.relation_include = base_include

    def _run_with_regime_verification(self, mode: str, **options: Any) -> dict[str, Any]:
        """Run one regime at a time, switching only when verify proves it necessary.

        Acceptance is **self-consistency**: a regime is kept when its own
        solve satisfies its own sustainment guards.  The walk starts at the
        declared regime -- preferring it is the steady-state stand-in for L-H
        hysteresis: the declared tag states which branch of a bistable band
        the machine sits on -- and moves to the guard-suggested candidate,
        falling back to preference order when the cross-evaluation heuristic
        is silent.  Each candidate is solved at most once.  When no candidate
        is self-consistent (the L-H bistable/dithering band, where the H solve
        falls below the very threshold the L solve exceeds), the result
        settles on ``l_mode`` as the accessible regime with a warning -- the
        same rule the popcon composite applies per grid point.
        """
        declared = _confinement_regime(self.tags)
        candidates = _candidate_regimes(declared)
        if not candidates:
            return self._run_once(mode, **options)

        current = declared
        path = [declared]
        attempts: dict[str, tuple[dict[str, Any], "Reactor"]] = {}
        for _attempt in range(len(candidates)):
            clone = self._clone_for_regime(current, include_guards=False)
            # The solve/scan itself uses only the current tag and its
            # confinement-time relation.  A separate verify pass below includes
            # every guard exactly to decide whether the discrete tag must change.
            result = clone._run_once(mode, **options)
            verify_result = clone._verify_all_regime_guards()
            statuses = verify_result.get("relation_status") or {}
            attempts[current] = (result, clone)
            # A scan (popcon) restores itself to pure inputs, so guards over
            # derived quantities cannot be evaluated afterwards.  When the
            # current regime's guards are merely indeterminate (not genuinely
            # violated), keep the declared regime -- per-point certification is
            # the real arbiter -- rather than churn or fail on missing values.
            if _regime_guards_indeterminate(statuses, current):
                self._absorb_regime_candidate(clone)
                self._annotate_regime_result(result, declared, current, path, mode)
                return result
            if _regime_verified_by_guards(statuses, current):
                self._absorb_regime_candidate(clone)
                self._annotate_regime_result(result, declared, current, path, mode)
                return result
            suggested = clone._suggest_regime_from_verify(verify_result, declared, current)
            next_regime = (
                suggested
                if suggested is not None and suggested not in attempts
                else next((regime for regime in candidates if regime not in attempts), None)
            )
            if next_regime is None:
                break
            current = next_regime
            path.append(current)

        # No candidate is self-consistent: the bistable/dithering band.  Settle
        # on the accessible regime (l_mode when tried, else the last attempt),
        # exactly as the popcon composite fills such points, and say so.
        fallback = "l_mode" if "l_mode" in attempts else current
        result, clone = attempts[fallback]
        if path[-1] != fallback:
            path.append(fallback)
        self._absorb_regime_candidate(clone)
        self._annotate_regime_result(result, declared, fallback, path, mode)
        result["regime_bistable"] = True
        result.setdefault("warnings", []).insert(0, _bistable_warning(fallback, mode))
        return result

    def _annotate_regime_result(self, result: dict[str, Any], declared: str | None, selected: str | None, path: list[str], mode: str) -> None:
        """Attach regime metadata and switch warning to a mode result."""
        if selected is None:
            return
        result["regime"] = selected
        result["declared_regime"] = declared
        result["regime_path"] = list(path)
        if declared and selected != declared:
            result["warnings"] = [
                _regime_warning(declared, selected, mode),
                *(result.get("warnings") or []),
            ]

    def _verify_all_regime_guards(self) -> dict[str, Any]:
        """Verify the solved values against every sustainment guard.

        Every guard is an outputless, checked-only relation over registry
        variables, and the solved system already holds the compiled,
        completed post-solve namespace -- so the guards are evaluated
        directly on it instead of building and compiling a second
        RelationSystem per solve.  A guard whose variables the solve could
        not value reports unverified, exactly as it would after pruning.
        """
        system = self.last_system if self.last_system is not None else self.relation_system()
        excluded = set()
        for item in self.relation_exclude:
            try:
                excluded.add(RELATIONS.get(item).name)
            except KeyError:
                excluded.add(str(item))
        values = system.complete(system.solver_values())
        status: dict[str, Any] = {}
        for name in _all_regime_guard_names():
            if name in excluded:
                continue
            rel = RELATIONS.get(name)
            missing = [v for v in rel.variables if values.get(v) is None]
            if missing:
                status[name] = {
                    "relation": name,
                    "verified": False,
                    "enforced": rel.enforce,
                    "errors": [f"Relation {name!r} missing variables {missing}."],
                    "warnings": [],
                }
                continue
            try:
                status[name] = system.relation_status_and_residual(rel, system.relation_evaluation_values(rel, values))[0]
            except Exception as exc:
                status[name] = {"relation": name, "verified": False, "enforced": rel.enforce, "errors": [str(exc)], "warnings": []}
        return {"relation_status": status}

    def _suggest_regime_from_verify(self, verify_result: Mapping[str, Any], declared: str | None, current: str | None) -> str | None:
        """Return the verified regime to switch to, or ``declared``/``None``."""
        statuses = verify_result.get("relation_status") or {}
        if not isinstance(statuses, Mapping):
            return None
        candidates = _candidate_regimes(declared)
        if current in candidates and _regime_verified_by_guards(statuses, current):
            return current
        for regime in candidates:
            if regime != current and _regime_verified_by_guards(statuses, regime):
                return regime
        return None

    def _clone_for_regime(self, regime: str, *, include_guards: bool = True) -> "Reactor":
        """Return an isolated reactor candidate for one confinement regime."""
        # ``clone(value=...)`` carries every other declaration field through
        # unchanged and re-ingests once, so this is identical to hand-listing
        # all eight constructor arguments (and no longer silently drops a
        # field if ``Variable`` grows one).  Seeding from ``input_value``
        # keeps a scalar-supplied profile scalar, preserving grid inference.
        variables = {
            name: var.clone(value=_copy_value(var.input_value))
            for name, var in self.variables.items()
        }
        clone = Reactor(
            name=self.name,
            organization=self.organization,
            country=self.country,
            year=self.year,
            doi=self.doi,
            notes=self.notes,
            tags=_with_confinement_regime(self.tags, regime),
            variables=variables,
            relation_include=self._candidate_relation_include(regime, include_guards=include_guards),
            relation_exclude=self.relation_exclude,
            relation_order=self.relation_order,
            constraints=self.constraints,
            grid_size=self.grid_size,
        )
        clone.source_path = self.source_path
        return clone

    def _candidate_relation_include(self, regime: str, *, include_guards: bool = True) -> tuple[str, ...]:
        """Add optional regime guards and a mode-appropriate tau_E default relation."""
        extras = list(_regime_guard_names(regime)) if include_guards else []
        if not self._has_explicit_tau_e_scaling():
            scaling = _regime_tau_default_name(regime)
            if scaling and scaling not in self.relation_exclude:
                extras.append(scaling)
        return _unique_extend(self.relation_include, extras)

    def _has_explicit_tau_e_scaling(self) -> bool:
        """Whether relation_include already names a tau_E producer."""
        for identifier in self.relation_include:
            try:
                rel = RELATIONS.get(identifier)
            except KeyError:
                continue
            if "tau_E" in rel.output_names:
                return True
        return False

    def _absorb_regime_candidate(self, candidate: "Reactor") -> None:
        """Adopt the winning candidate's tags, declarations and solved system.

        ``candidate.variables`` are frozen declarations cloned verbatim from
        ``self.variables`` by :meth:`_clone_for_regime` and never mutated by a
        solve, so they are adopted directly rather than merged field-by-field;
        the candidate's solved state comes along on ``last_system``.
        """
        self.tags = candidate.tags
        self.last_system = candidate.last_system
        self.variables = dict(candidate.variables)

    def verify(self) -> dict[str, Any]:
        """Verify this reactor.

        Returns:
            Verification result.
        """
        return self.run("verify")

    def reconcile(self, **options: Any) -> dict[str, Any]:
        """Reconcile this reactor.

        Args:
            **options: Solver options.

        Returns:
            Reconciliation result.
        """
        return self.run("reconcile", **options)

    def optimize(self, **options: Any) -> dict[str, Any]:
        """Optimize this reactor.

        Args:
            **options: Optimization options.

        Returns:
            Optimization result.
        """
        return self.run("optimize", **options)

    def ordered(self, **options: Any) -> dict[str, Any]:
        """Execute this reactor's ordered recipe.

        Args:
            **options: Ordered-mode options.

        Returns:
            Ordered result.
        """
        return self.run("ordered", **options)

    def popcon(self, *, x: Any, y: Any, **options: Any) -> dict[str, Any]:
        """Run a batched 2-D popcon scan over two axis variables.

        The whole grid is evaluated as one batched computation with this
        reactor's inputs held exactly as given and the axis values pinned to
        the grid coordinates; every point is then individually certified.
        See :mod:`fusdb.modes.popcon` for the options and result payload.

        Args:
            x: X-axis spec (variable name plus grid values/range).
            y: Y-axis spec.
            **options: Popcon-mode options (``outputs``, ``verbose``).

        Returns:
            Popcon result dictionary.
        """
        return self.run("popcon", x=x, y=y, **options)

    def _run_popcon_auto_regime(self, **options: Any) -> dict[str, Any]:
        """Popcon scan with automatic per-point confinement regime.

        Instead of running the whole grid in one fixed regime, this runs the
        batched scan once per candidate regime (:func:`_candidate_regimes` of
        the declared tag -- the declared branch's escalation chain, e.g.
        ``h_mode``/``l_mode``/``ohmic_mode``, with both upper branches as
        candidates for an ``l_mode``/``ohmic_mode`` machine) and assigns each
        grid point the regime it actually
        sits in: the strongest regime whose sustainment guard holds on that
        regime's own solve (``P_sep >= P_LH`` for H-mode, ``P_sep <= P_OL_thresh``
        for ohmic), with L-mode as the accessible fallback for the intermediate /
        L-H-bistable band. The composited result carries a ``regime_index`` grid
        (index into ``regime_names``, ``-1`` where no regime certified).

        A reactor with no declared confinement regime falls back to a single
        plain scan.
        """
        declared = _confinement_regime(self.tags)
        chain = _candidate_regimes(declared)
        if len(chain) <= 1:
            return self._run_once("popcon", **options)
        fallback = "l_mode" if "l_mode" in chain else chain[-1]

        # Every per-regime scan must also compute the guard-threshold quantities
        # so each point can be classified; add them when outputs are restricted.
        requested = options.get("outputs")
        scan_options = dict(options)
        if requested is not None:
            requested = [VARIABLES.resolve(str(name)) for name in requested]
            scan_options["outputs"] = tuple(dict.fromkeys((*requested, *_regime_guard_input_names(chain))))

        per_regime: dict[str, dict[str, Any]] = {}
        clones: dict[str, "Reactor"] = {}
        warnings: list[str] = []
        for regime in chain:
            clones[regime] = self._clone_for_regime(regime, include_guards=False)
        # The per-regime scans are independent, so run them on a process pool.
        # Live systems cannot cross a process boundary; workers rebuild from
        # the same picklable recipe the pointwise scan uses.  ``workers=0``/``1``
        # or a pointwise (``solver="reconcile"``) scan stays serial -- the
        # pointwise solver parallelises internally, and nesting pools would
        # oversubscribe.  Any pool/pickling failure falls back to in-process.
        requested_workers = options.get("workers")
        parallel = (
            str(options.get("solver", "batched")) != "reconcile"
            and (requested_workers is None or int(requested_workers) > 1)
        )
        if parallel:
            try:
                from concurrent.futures import ProcessPoolExecutor

                from .modes.popcon import _system_spec

                tasks = [
                    (_system_spec(clones[regime].relation_system()), scan_options)
                    for regime in chain
                ]
                with ProcessPoolExecutor(max_workers=len(chain)) as executor:
                    for regime, scan in zip(chain, executor.map(_popcon_regime_scan_worker, tasks)):
                        per_regime[regime] = scan
            except Exception:
                per_regime = {}
        for regime in chain:
            if regime not in per_regime:
                per_regime[regime] = clones[regime]._run_once("popcon", **scan_options)
            for warning in per_regime[regime].get("warnings", ()):  # e.g. underivable output
                if warning not in warnings:
                    warnings.append(warning)
        if clones[chain[0]].last_system is None:
            # The parallel scans never ran in this process; a popcon restores
            # its system to the declared state anyway, so an equivalent
            # freshly-built system serves the read-through role.
            clones[chain[0]].last_system = clones[chain[0]].relation_system()
        self.last_system = clones[chain[0]].last_system

        base = per_regime[chain[0]]["popcon"]
        shape = base["success"].shape
        field_names = list(requested) if requested is not None else list(base["fields"])
        fields = {name: np.full(shape, np.nan) for name in field_names}
        success = np.zeros(shape, dtype=bool)
        regime_index = np.full(shape, -1, dtype=int)

        def _fill(regime: str, take: np.ndarray) -> None:
            regime_index[take] = chain.index(regime)
            success[take] = True
            regime_fields = per_regime[regime]["popcon"]["fields"]
            for name in field_names:
                grid = regime_fields.get(name)
                if grid is not None:
                    fields[name][take] = grid[take]

        # Guarded regimes first (self-sustaining bands, disjoint in P_sep), then
        # L-mode fills the remaining certified points as the accessible fallback.
        for regime in (r for r in chain if r != fallback):
            payload = per_regime[regime]["popcon"]
            holds = _regime_guard_holds(regime, payload["fields"], shape)
            _fill(regime, payload["success"] & holds & (regime_index < 0))
        fb = per_regime[fallback]["popcon"]
        _fill(fallback, fb["success"] & (regime_index < 0))

        # The L-H boundary is decided on the top regime's own P_sep vs P_LH, so
        # expose that regime's P_sep/P_LH as the single-solve L-H accessibility
        # reference (P_sep is regime-dependent, so the per-cell composite ratio is
        # discontinuous and does not track the boundary through the bistable band).
        top_fields = per_regime[chain[0]]["popcon"]["fields"]
        p_sep = top_fields.get(VARIABLES.resolve("P_sep"))
        p_lh = top_fields.get(VARIABLES.resolve("P_LH"))
        lh_ratio_reference = None
        if p_sep is not None and p_lh is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                lh_ratio_reference = np.where(np.abs(p_lh) > 0, p_sep / p_lh, np.nan)

        n = int(np.prod(shape))
        n_ok = int(success.sum())
        assigned = ", ".join(f"{r}={int((regime_index == chain.index(r)).sum())}" for r in chain)
        return {
            "success": n_ok > 0,
            "termination": f"popcon auto-regime: {n_ok}/{n} points solved ({assigned})",
            "n_points": n,
            "n_failed": n - n_ok,
            "warnings": warnings,
            "errors": [],
            "popcon": {
                "x": base["x"],
                "y": base["y"],
                "fields": fields,
                "success": success,
                "failures": [],
                "regime_index": regime_index,
                "regime_names": tuple(chain),
                "lh_ratio_reference": lh_ratio_reference,
            },
        }


def _popcon_regime_scan_worker(task: tuple[Mapping[str, Any], Mapping[str, Any]]) -> dict[str, Any]:
    """Worker: rebuild one regime clone's system and run its batched popcon scan.

    Top-level so it pickles for the process pool.  Mode results are plain data
    and pickle back as-is.
    """
    spec, options = task
    from .modes.popcon import _rebuild_system

    return _rebuild_system(spec).run("popcon", **options)


def _solve_reactor_path(task: tuple[str, str, Mapping[str, Any]]) -> SolvedColumn:
    """Worker: load one reactor YAML, solve it, return a display-ready column.

    Runs in a separate process for :func:`solve_reactors`. Mode results are
    plain data (no live relation objects), so the column pickles as-is. A
    load/solve failure is returned as a failed-result column rather than
    raised, so one bad reactor does not abort the whole batch.
    """
    path, mode, options = task
    try:
        reactor = Reactor.from_yaml(path)
        reactor.run(mode, **options)
        return _table_column(reactor.last_system)
    except Exception as exc:  # report the failure as a column; keep the batch alive
        location = Path(path)
        return SolvedColumn(
            location.parent.name or location.stem,
            {},
            {},
            {},
            {},
            frozenset(),
            {},
            {"success": False, "termination": f"crashed: {exc!r}", "errors": [repr(exc)]},
        )


def solve_reactors(
    paths: Iterable[str | Path | Reactor],
    *,
    mode: str = "reconcile",
    workers: int | None = None,
    **options: Any,
) -> list[SolvedColumn]:
    """Solve many reactor YAMLs in parallel, one worker process each.

    Live :class:`Reactor`/:class:`RelationSystem` objects cannot cross a process
    boundary, so each worker loads its reactor from YAML and returns a picklable
    :class:`SolvedColumn`. Render them with :func:`variable_table_data` and
    :func:`render_table`.

    Args:
        paths: Reactor YAML files or directories, or reactors loaded via
            :meth:`Reactor.from_yaml` (their ``source_path`` is used).
        mode: Mode to run on each reactor.
        workers: Maximum worker processes; ``None`` lets the pool decide.
        **options: Mode options forwarded to every run.

    Returns:
        One :class:`SolvedColumn` per input, in order. Duplicate reactor names
        are disambiguated with their file path.
    """
    from concurrent.futures import ProcessPoolExecutor

    resolved: list[str] = []
    for item in paths:
        if isinstance(item, Reactor):
            if item.source_path is None:
                raise ValueError(f"Reactor {item.name!r} was not loaded from YAML; pass its path.")
            resolved.append(str(item.source_path))
        else:
            resolved.append(str(item))

    tasks = [(path, mode, options) for path in resolved]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        columns = list(executor.map(_solve_reactor_path, tasks))

    name_counts: dict[str, int] = {}
    for column in columns:
        name_counts[column.name] = name_counts.get(column.name, 0) + 1
    labelled: list[SolvedColumn] = []
    for path, column in zip(resolved, columns):
        if name_counts[column.name] > 1:
            location = Path(path)
            column = column._replace(name=f"{column.name} ({location.parent.name}/{location.name})")
        labelled.append(column)
    return labelled
