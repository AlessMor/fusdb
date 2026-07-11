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
from .plotting.tables import SolvedColumn, _table_column, _variables_text_table, variables_table
from .variable import Variable

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


def _switch_candidate_regimes(declared: str | None) -> tuple[str, ...]:
    """Regime order used only after verify proves the current tag is inconsistent."""
    order = _regime_order()
    if declared not in order:
        return ()
    if declared == order[0]:
        return order
    return tuple(regime for regime in order if regime != order[0])


def _regime_verified_by_guards(statuses: Mapping[str, Mapping[str, Any]], regime: str) -> bool:
    """Whether every guard for ``regime`` is present and verified."""
    guards = _regime_guard_names(regime)
    return bool(guards) and all(bool((statuses.get(guard) or {}).get("verified", False)) for guard in guards)


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
        self.variables[var.name] = var

    def get_variable(self, name: str) -> Variable | None:
        """Return one loaded variable by canonical name or alias.

        Args:
            name: Canonical name or alias.

        Returns:
            Variable or None.
        """
        try:
            canonical = VARIABLES.resolve(name)
        except Exception:
            canonical = str(name)
        return self.variables.get(canonical)

    def __getattr__(self, name: str) -> Variable:
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
        """Build a RelationSystem, run one mode, and absorb the solved values.

        The solve runs on a clone, then each solved value replaces this reactor's
        corresponding input (both ``value`` and ``input_value`` are overwritten),
        so ``reactor.<var>`` reflects the latest solve and a re-run starts from
        it. The full solved system is kept on :attr:`last_system`, which still
        carries that run's original inputs for the input->output display.

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
        if chosen == "verify":
            return self._run_guarded_once(chosen, **options)
        if chosen in _REGIME_SOLVE_MODES:
            return self._run_with_regime_verification(chosen, **options)
        return self._run_once(chosen, **options)

    def _run_once(self, mode: str, **options: Any) -> dict[str, Any]:
        """Run one mode with the currently configured relation set."""
        chosen = mode
        system = self.relation_system()
        self.last_system = system
        if chosen == "ordered":
            result = system.ordered(order=self.relation_order or None, **options)
        else:
            result = system.run(chosen, **options)
        for name, value in system.values.items():
            var = self.variables.get(name)
            if var is not None and value is not None:
                var.set_input(value)
        return result

    def _run_guarded_once(self, mode: str, **options: Any) -> dict[str, Any]:
        """Run one mode with the current regime sustainment guard included."""
        base_include = self.relation_include
        self.relation_include = _with_sustainment_guards(base_include, _confinement_regime(self.tags))
        try:
            return self._run_once(mode, **options)
        finally:
            self.relation_include = base_include

    def _run_with_regime_verification(self, mode: str, **options: Any) -> dict[str, Any]:
        """Run one regime at a time, switching only when verify proves it necessary."""
        declared = _confinement_regime(self.tags)
        if declared not in _regime_order():
            return self._run_once(mode, **options)

        current = declared
        path = [declared]
        tried: set[str] = set()
        last_result: dict[str, Any] | None = None
        max_attempts = len(_switch_candidate_regimes(declared)) + 1
        for _attempt in range(max_attempts):
            clone = self._clone_for_regime(current, include_guards=False)
            # The solve/scan itself uses only the current tag and its
            # confinement-time relation.  A separate verify pass below includes
            # every guard exactly to decide whether the discrete tag must change.
            result = clone._run_once(mode, **options)
            last_result = result
            verify_result = clone._verify_all_regime_guards()
            suggested = clone._suggest_regime_from_verify(verify_result, declared, current)
            if suggested is not None and suggested != current:
                if suggested not in tried:
                    tried.add(current)
                    current = suggested
                    path.append(current)
                    continue
                result["unresolved_regime"] = True
                clone._apply_regime_verify_failure(result, verify_result, current)
            elif suggested is None and not _regime_verified_by_guards(verify_result.get("relation_status") or {}, current):
                clone._apply_regime_verify_failure(result, verify_result, current)

            self._absorb_regime_candidate(clone)
            self._annotate_regime_result(result, declared, current, path, mode)
            return result

        # Defensive fallback: a pathological guard cycle should not leave the
        # reactor un-updated or hide the last concrete mode result.
        if last_result is None:
            return self._run_once(mode, **options)
        last_result["unresolved_regime"] = True
        self._annotate_regime_result(last_result, declared, current, path, mode)
        return last_result

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
        candidates = _switch_candidate_regimes(declared)
        if current in candidates and _regime_verified_by_guards(statuses, current):
            return current
        for regime in candidates:
            if regime != current and _regime_verified_by_guards(statuses, regime):
                return regime
        return None

    def _apply_regime_verify_failure(self, result: dict[str, Any], verify_result: Mapping[str, Any], regime: str) -> None:
        """Mark a mode result failed because its final values do not verify for ``regime``."""
        result["success"] = False
        if "verified" in result:
            result["verified"] = False
        errors = result.setdefault("errors", [])
        for error in self._regime_guard_errors(verify_result, regime):
            if error not in errors:
                errors.append(error)

    def _regime_guard_errors(self, verify_result: Mapping[str, Any], regime: str) -> list[str]:
        """Return only the failed guard messages for one regime."""
        statuses = verify_result.get("relation_status") or {}
        if not isinstance(statuses, Mapping):
            return [f"Declared {regime} operating condition could not be verified."]
        errors: list[str] = []
        for guard in _regime_guard_names(regime):
            status = statuses.get(guard)
            if not isinstance(status, Mapping):
                errors.append(f"{guard}: relation was not checked")
            elif not bool(status.get("verified", False)):
                detail = status.get("errors") or [f"{guard}: relation did not verify"]
                errors.extend(str(item) for item in detail)
        return errors

    def _clone_for_regime(self, regime: str, *, include_guards: bool = True) -> "Reactor":
        """Return an isolated reactor candidate for one confinement regime."""
        variables = {
            name: Variable(
                var.name,
                value=_copy_value(var.input_value),
                unit=var.unit,
                rel_tol=var.rel_tol,
                abs_tol=var.abs_tol,
                fixed=var.fixed,
                size=var.size,
                constraints=var.constraints,
            )
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
        """Copy the winning candidate's public solve state into this reactor."""
        self.tags = candidate.tags
        self.last_system = candidate.last_system
        for name, src in candidate.variables.items():
            dst = self.variables.get(name)
            if dst is None:
                self.variables[name] = src
            elif src.value is not None:
                dst.set_input(src.value)

    def _display_source(self) -> Any:
        """Return the most recent solved system, or this reactor's inputs."""
        return self.last_system if self.last_system is not None else self

    def _repr_html_(self) -> str:
        """Rich Jupyter table of current variables (solved values after a run)."""
        return variables_table(self._display_source())

    def print_variables_table(self, *names: str) -> None:
        """Print this reactor's current variables as a plain-text table.

        Shows reconciled values after a run, otherwise the loaded inputs.

        Args:
            *names: Optional variable names to show. Defaults to all variables.
        """
        print(_variables_text_table(self._display_source(), names or None))

    def print_html_variables_table(self, *names: str) -> None:
        """Display this reactor's current variables as an HTML table (Jupyter).

        Shows reconciled values after a run (with input->output colouring,
        active-variable highlighting, and relation tooltips), otherwise the
        loaded inputs.

        Args:
            *names: Optional variable names to show. Defaults to all variables.
        """
        from IPython.display import HTML, display

        display(HTML(variables_table(self._display_source(), variable_names=names or None)))

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


def _picklable_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only the picklable, display/print-relevant parts of a run result.

    A full result dict may reference relation *functions* (which do not pickle by
    reference), so worker processes return this reduced copy.
    """
    status = result.get("relation_status", {}) or {}
    clean_status = {
        name: {"enforced": bool(state.get("enforced", True)), "verified": bool(state.get("verified", False))}
        for name, state in status.items()
        if isinstance(state, Mapping)
    }
    return {
        "success": bool(result.get("success", False)),
        "termination": result.get("termination"),
        "errors": [str(error) for error in (result.get("errors") or ())],
        "warnings": [str(warning) for warning in (result.get("warnings") or ())],
        "relation_status": clean_status,
    }


def _solve_reactor_path(task: tuple[str, str, Mapping[str, Any]]) -> SolvedColumn:
    """Worker: load one reactor YAML, solve it, return a display-ready column.

    Runs in a separate process for :func:`solve_reactors`. A load/solve failure
    is returned as a failed-result column rather than raised, so one bad reactor
    does not abort the whole batch.
    """
    path, mode, options = task
    try:
        reactor = Reactor.from_yaml(path)
        reactor.run(mode, **options)
        column = _table_column(reactor.last_system)
        return column._replace(result=_picklable_result(column.result))
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
    :class:`SolvedColumn`. Pass the columns straight to :func:`variables_table`.

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
