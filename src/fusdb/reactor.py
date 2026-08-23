"""User-facing reactor scenario object."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .modes._batch import map_chunks, parallel_chunk_size
from .modes import MODE_NAMES
from .profiles.system import build_relation_system
from .registry import RELATIONS, TAGS, VARIABLES
from .relationsystem import CompilePlan, RelationSystem
from .plotting.tables import SolvedColumn, _table_column
from .variable import Variable

logger = logging.getLogger("fusdb")


@dataclass(frozen=True)
class SolvedVariable:
    """Read-through view pairing a frozen declaration with its solved value.

    Returned by :meth:`Reactor.get_variable` and attribute access
    (``reactor.<name>``).  ``declared`` is the :class:`Variable` exactly as
    supplied -- a solve never mutates it.  ``value`` is the latest solved
    value from the reactor's :attr:`Reactor.last_plan` when one is active
    for this name, falling back to the declared value otherwise; it is
    resolved fresh on every access, so it always reflects the most recent
    run.  Every other attribute (``unit``, ``fixed``, ``rel_tol``, ``spec``,
    ...) delegates to ``declared``.  :meth:`clone` builds a new declaration,
    exactly like :meth:`Variable.clone`.
    """

    declared: Variable
    _plan: CompilePlan | None

    @property
    def value(self) -> Any:
        if self._plan is not None:
            solved = self._plan.values.get(self.declared.name)
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
#     ``("confinement_mode_threshold", <regime>)``;
#   * a regime's fallback confinement-time scaling is the tau_E_scaling producer
#     tagged ``("confinement_mode_default", <regime>)``.
_REGIME_SOLVE_MODES = {"reconcile", "optimize", "popcon"}


def _regime_order() -> tuple[str, ...]:
    """Confinement regimes in threshold-escalation order (allowed_tags.yaml)."""
    return tuple(str(tag) for tag in (TAGS.raw.get("confinement_mode") or ()))


def _regime_guard_names(regime: str | None) -> tuple[str, ...]:
    """Sustainment guard relation names of one regime, in registration order."""
    if not regime:
        return ()
    return tuple(rel.name for rel in RELATIONS if "confinement_mode_threshold" in rel.tags and regime in rel.tags)


def _all_regime_guard_names() -> tuple[str, ...]:
    """All sustainment guard relation names, in registration order."""
    return tuple(rel.name for rel in RELATIONS if "confinement_mode_threshold" in rel.tags)


def _regime_tau_default_name(regime: str) -> str | None:
    """Name of the regime's fallback tau_E scaling, or None when undeclared."""
    for rel in RELATIONS:
        if "confinement_mode_default" in rel.tags and regime in rel.tags and "tau_E_scaling" in rel.output_names:
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


def _candidate_regimes(declared: str | None) -> tuple[str, ...]:
    """Every confinement mode that can be tested, declared one first.

    All modes are candidates, so the transition graph is complete: L<->H, L<->I
    AND **I<->H**.  I->H is a common experimental trajectory and H->I occurs
    too, so a graph of ``H <-> L <-> I`` with no ``H <-> I`` edge is not a
    general transition model.

    Order matters only as a LAST-RESORT tie-break.  The verdict is the
    admissible COUNT (see :meth:`Reactor._run_with_regime_verification`), and
    the declared mode wins whenever it is itself admissible -- so an ``h_mode``
    machine that is still sustaining H-mode is never reclassified to I-mode
    just because its power also clears the L-I threshold.

    CAVEAT, and it is a real physics gap: when the declared mode is NOT
    admissible and several others are, H and I are separated only by this
    ordering.  Nothing in the decision reads topology, drift direction or the
    edge state, which is what actually distinguishes the two upper branches.
    That case is reported as ambiguous rather than presented as a verdict; a
    genuine discriminator (an edge-state criterion such as SepOS, or an I-mode
    accessibility condition) belongs as an extra certifier relation, which
    needs no change here.

    Modes with no registered certifier are dropped: admissibility would be
    undefined for them.
    """
    order = _regime_order()
    if declared not in order:
        return ()
    # MEASURED 2026-08-07: the complete graph (every mode a candidate, so
    # I<->H allowed) was implemented and REVERTED.  It regressed the tungsten
    # point `w_1.0e-04` badly -- that point has no consistent mode, and with
    # i_mode reachable from h_mode it was classified i_mode on an absurd solve:
    # P_sep 7179.6 MW against PROCESS's 0.0 MW, P_aux +9424%.  The cause is that
    # I-MODE HAS NO UPPER CERTIFIER: its only condition is
    # `P_sep >= P_LI_thresh`, which an inflated P_sep satisfies trivially, so a
    # pathological branch reads as admissible.
    # I<->H is therefore blocked on physics, not on this table: it needs either
    # an I-H threshold (an I-mode ceiling) or a real H/I discriminator -- H and
    # I differ by topology, drift direction and edge state, none of which is in
    # the decision.  Both belong as extra certifier relations; when one exists,
    # replace these chains with `[declared, *others]` and the graph completes.
    chains = {
        "h_mode": ("h_mode", "l_mode"),
        "i_mode": ("i_mode", "l_mode"),
        "l_mode": ("l_mode", "h_mode", "i_mode"),
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

    These (``P_sep``, ``P_LH``, ``P_LI_thresh``) must be produced
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
    ``P_sep >= P_HL``), so the regime holds where every guard residual is ~0. A
    guard whose inputs are missing marks the point as not in this regime.
    """
    mask = np.ones(shape, dtype=bool)

    def registry_default(name: str, seen: frozenset[str] = frozenset()) -> Any:
        """Resolve a guard-only field from its registry default."""
        if name in seen or name not in VARIABLES:
            return None
        spec = VARIABLES.get(name)
        if spec.default is None:
            return None
        if isinstance(spec.default, str):
            source = VARIABLES.resolve(spec.default)
            source_value = fields.get(source)
            if source_value is not None and np.isfinite(np.asarray(source_value, dtype=float)).any():
                return source_value
            return registry_default(source, seen | {name})
        return float(spec.default)

    for guard in _regime_guard_names(regime):
        rel = RELATIONS.get(guard)
        args: dict[str, Any] = {}
        for inp in rel.input_names:
            value = fields.get(inp)
            if value is None or not np.isfinite(np.asarray(value, dtype=float)).any():
                value = registry_default(inp)
            if value is None:
                return np.zeros(shape, dtype=bool)
            if np.asarray(value).ndim == 0:
                value = np.full(shape, value)
            args[inp] = value
        residual = np.asarray(rel.func(**args), dtype=float)
        mask &= np.isfinite(residual) & (residual <= 1.0e-9)
    return mask


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


def _load_variable_file(
    path_like: Path,
    *,
    base_dir: Path,
    delimiter: str | None = None,
    usecols: Any = None,
    skiprows: int = 0,
    keep_columns: bool = False,
) -> np.ndarray:
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
    if array.ndim == 2 and keep_columns:
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
        coordinate = entry.get("coordinate")
        coordinate_values = entry.get("coordinate_values")
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
            loaded = _load_variable_file(
                Path(str(file_value)),
                base_dir=base_dir,
                delimiter=entry.get("delimiter"),
                usecols=entry.get("usecols"),
                skiprows=entry.get("skiprows", 0),
                keep_columns=coordinate is not None and coordinate_values is None,
            )
            if coordinate is not None and coordinate_values is None and loaded.ndim == 2:
                if loaded.shape[1] < 2:
                    raise ValueError(
                        f"Variable {raw_name!r} source profile file needs at least two columns "
                        "when coordinate is declared without coordinate_values."
                    )
                coordinate_values = loaded[:, 0]
                value = loaded[:, -1]
            else:
                value = loaded
            if np.asarray(value).ndim == 1:
                size = int(np.asarray(value).shape[0])
        var = Variable(
            str(raw_name),
            value=value,
            unit=entry.get("unit"),
            rel_tol=entry.get("rel_tol"),
            abs_tol=entry.get("abs_tol"),
            fixed=bool(entry.get("fixed", False)),
            size=size,
            constraints=entry.get("constraints"),
            default_relation=entry.get("default_relation"),
            coordinate=coordinate,
            coordinate_values=coordinate_values,
        )
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
        self.last_plan: CompilePlan | None = None
        # YAML this reactor was loaded from, if any; lets solve_reactors ship it
        # to a worker process. Set by from_yaml. Here so __getattr__ skips it.
        self.source_path: Path | None = None
        self.tags = tuple(str(tag).strip().lower() for tag in self.tags)
        # An unregistered tag is kept (it is legitimate descriptive metadata --
        # a machine name, a programme) but it selects nothing, and a reactor
        # whose ONLY tag is unregistered silently gets no device-scoped
        # relations at all.  That failure is invisible otherwise, so say it out
        # loud once, at declaration time.
        unknown = tuple(tag for tag in self.tags if tag not in TAGS.allowed)
        if unknown:
            known_groups = ", ".join(sorted(TAGS.raw)) or "none"
            logger.warning(
                "Reactor %r declares tag(s) %s that are not in allowed_tags.yaml; "
                "they are kept as descriptive metadata but match no relation. "
                "%sAdd them to a group in allowed_tags.yaml (groups: %s) if they "
                "should select relations.",
                self.name,
                ", ".join(repr(tag) for tag in unknown),
                "This reactor has NO recognised tag, so every relation scoped to a "
                "tag group (e.g. device: tokamak/stellarator/mirror) is excluded. "
                if len(unknown) == len(self.tags) else "",
                known_groups,
            )
        self.relation_include = tuple(str(name) for name in (self.relation_include or ()))
        self.relation_exclude = tuple(str(name) for name in (self.relation_exclude or ()))
        self.relation_order = tuple(self.relation_order or ())
        self.variables = {var.name: var for var in self.variables.values()}
        if self.grid_size is not None:
            self.grid_size = int(self.grid_size)
            if self.grid_size <= 0:
                raise ValueError("grid.size must be positive.")

    @classmethod
    def from_name(cls, name: str) -> "Reactor":
        """Load a packaged or checkout reactor scenario by folder name.

        Resolution order: the optional ``fusdb-reactors`` data distribution
        (``pip install ./reactors`` from a checkout), then a ``reactors/``
        directory under the current working directory.  The base ``fusdb``
        install deliberately ships no scenarios.
        """
        candidates: list[Path] = []
        try:
            import fusdb_reactors  # optional data-only distribution (S14)

            candidates.append(Path(fusdb_reactors.reactors_dir()))
        except ImportError:
            pass
        candidates.append(Path.cwd() / "reactors")
        for base in candidates:
            target = base / name
            if target.exists():
                return cls.from_yaml(target)
        raise FileNotFoundError(
            f"Reactor scenario {name!r} not found. Install the scenario data "
            "(pip install ./reactors from a fusdb checkout) or run from a "
            "directory containing reactors/."
        )

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

    def clone(self) -> "Reactor":
        """Return an isolated copy of this reactor's declarations."""
        variables = {
            name: var.clone(value=var.input_value.copy() if isinstance(var.input_value, np.ndarray) else var.input_value)
            for name, var in self.variables.items()
        }
        clone = Reactor(
            name=self.name,
            organization=self.organization,
            country=self.country,
            year=self.year,
            doi=self.doi,
            notes=self.notes,
            tags=self.tags,
            variables=variables,
            relation_include=self.relation_include,
            relation_exclude=self.relation_exclude,
            relation_order=self.relation_order,
            constraints=self.constraints,
            grid_size=self.grid_size,
        )
        clone.source_path = self.source_path
        return clone

    def get_variable(self, name: str) -> SolvedVariable | None:
        """Return one loaded variable, as a read-through solved/declared view.

        Args:
            name: Canonical name or alias.

        Returns:
            :class:`SolvedVariable` pairing the frozen declaration with its
            latest solved value (from :attr:`last_plan`, when active for
            this name), or ``None`` if no such variable was declared.
        """
        try:
            canonical = VARIABLES.resolve(name)
        except Exception:
            canonical = str(name)
        var = self.variables.get(canonical)
        if var is None:
            return None
        return SolvedVariable(var, self.last_plan)

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
        names = _unique_extend(
            self._regime_filtered_include(_confinement_regime(self.tags)),
            self._supplied_shape_includes(),
        )
        defaults = {
            name: variable.default_relation
            for name, variable in self.variables.items()
            if variable.default_relation is not None
        }
        return RELATIONS.get_filtered_relations(
            names=names,
            tags=TAGS.expand(self.tags),
            exclude=self.relation_exclude,
            order=None,
            default_relations=defaults,
        )

    def _supplied_shape_includes(self) -> tuple[str, ...]:
        """Shape generators opted in by supplied profile information.

        A profile is ``average x shape``.  The average comes from the data; the
        shape is a modelling choice, so the SHAPE generators are opt-in (tag
        ``profile_shape``) and the default is a uniform profile at the average
        value.  But a reactor that supplies a peaking factor -- or a peak value
        which, with its average, determines one -- HAS given the shape, exactly
        as a reactor supplying the profile itself has.  Those generators are
        therefore selected automatically, so supplied data is never discarded in
        favour of a flat default.  ``relation_exclude`` still overrides.
        """
        supplied = {
            VARIABLES.resolve(name)
            for name in self.variables
            if name in VARIABLES
        }
        if not supplied:
            return ()

        def shape_is_supplied(peaking: str, seen: frozenset[str] = frozenset()) -> bool:
            """Whether ``peaking`` is pinned by supplied data.

            Directly supplied, or produced by a relation whose own inputs are
            each supplied or themselves shape-supplied.  The recursion matters:
            the ion peaking is inherited from the electron one by a default
            relation, so supplying ``n0`` shapes the ion density too -- and a
            peaking that is neither supplied nor determined is a shape the
            *solver* would invent, which is exactly what stays opt-in.
            """
            if peaking in supplied:
                return True
            if peaking in seen:
                return False
            seen = seen | {peaking}
            spec = VARIABLES.get(peaking) if peaking in VARIABLES else None
            for producer_name in (getattr(spec, "default_relation", None) or ()):
                try:
                    producer = RELATIONS.get(producer_name)
                except KeyError:
                    continue
                inputs = tuple(producer.input_names)
                if inputs and all(
                    inp in supplied or ("peaking" in inp and shape_is_supplied(inp, seen))
                    for inp in inputs
                ):
                    return True
            return False

        mode = _confinement_regime(self.tags)
        mode_axis = set(_regime_order())
        active_tags = TAGS.expand(self.tags)
        candidates: list[Any] = []

        def input_is_declared_or_defaulted(
            name: str,
            seen: frozenset[str] = frozenset(),
        ) -> bool:
            """Whether a shape input is supplied or deterministically available.

            Besides literal registry defaults, an input may be produced by its
            active ``default_relation``. This matters for geometry mappings such
            as ``rho_minor``: a tokamak supplies that mapping through the tagged
            identity provider even though it is not a user-declared scenario
            variable. Treating only scalar ``default`` values as available made
            H/I pedestal-profile selection regress when the coordinate migration
            replaced bare ``rho`` with explicit ``rho_minor``.
            """
            if name == "rho" or name in supplied:
                return True
            if name in seen or name not in VARIABLES:
                return False
            spec = VARIABLES.get(name)
            if spec.default is not None:
                requires = (
                    None
                    if spec.default_requires is None
                    else VARIABLES.resolve(spec.default_requires)
                )
                if requires is None or requires in supplied:
                    return True

            next_seen = seen | {name}
            for provider_name in spec.default_relation:
                try:
                    provider = RELATIONS.get(provider_name)
                except KeyError:
                    continue
                if not TAGS.relation_matches(provider.tags, active_tags):
                    continue
                if all(
                    input_is_declared_or_defaulted(inp, next_seen)
                    for inp in provider.input_names
                ):
                    return True
            return False

        for rel in RELATIONS:
            relation = RELATIONS.get(rel) if isinstance(rel, str) else rel
            if "profile_shape" not in relation.tags:
                continue
            relation_modes = mode_axis & set(relation.tags)
            if relation_modes and (mode is None or mode not in relation_modes):
                continue
            peakings = [name for name in relation.input_names if "peaking" in name]
            if peakings and all(shape_is_supplied(name) for name in peakings):
                candidates.append(relation)
                continue
            if "confinement_mode_profile_default" in relation.tags:
                physical_inputs = [name for name in relation.input_names if name != "rho"]
                if physical_inputs and all(input_is_declared_or_defaulted(name) for name in physical_inputs):
                    candidates.append(relation)

        # A source-backed mode profile supersedes the generic parabolic shape
        # for the same output only when all of its pedestal controls are
        # available. Otherwise the established parabolic/uniform fallbacks stay
        # untouched. Explicit ``relations.include`` remains the way to select a
        # non-default source such as FUSE/IMAS.
        mode_outputs = {
            output
            for relation in candidates
            if "confinement_mode_profile_default" in relation.tags
            for output in relation.output_names
        }
        selected = [
            relation.name
            for relation in candidates
            if "confinement_mode_profile_default" in relation.tags
            or not (set(relation.output_names) & mode_outputs)
        ]
        return tuple(selected)

    def relation_system(self) -> RelationSystem:
        """Build a RelationSystem for this reactor.

        Returns:
            RelationSystem instance.
        """
        # No clone: the system ingests the records into its own value dicts,
        # which is the isolation copy (values are replaced, never mutated).
        return build_relation_system(
            self.variables.values(),
            self.relations(),
            constraints=self.constraints,
            name=self.name,
            profile_size=self.grid_size,
        )

    def run(self, mode: str = "verify", **options: Any) -> dict[str, Any]:
        """Build a RelationSystem, run one mode, and keep the solved plan.

        Declared variables (:attr:`variables`) are never changed by a run --
        a solve produces a new :class:`CompilePlan`, kept on
        :attr:`last_plan`, and ``reactor.<var>`` (via :meth:`get_variable`)
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
        plan = self.relation_system().compile()
        self.last_plan = plan
        if chosen == "ordered":
            options = {"order": self.relation_order or None, **options}
        return plan.run(chosen, **options)

    def restart_from_solution(self) -> None:
        """Replace each declared variable's value with its latest solved value.

        The explicit form of what earlier fusdb versions did implicitly after
        every solve: nothing is mutated in place -- each affected declaration
        is replaced by a fresh one (:meth:`Variable.clone`) built from
        :attr:`last_plan`'s solved value, so the *next* ``run()`` starts
        from where this one ended.  Fixed variables are unaffected (their
        solved value already equals their declared one).  A no-op if this
        reactor has not been run yet.
        """
        if self.last_plan is None:
            return
        for name, value in self.last_plan.values.items():
            var = self.variables.get(name)
            if var is not None and value is not None:
                changes: dict[str, Any] = {"value": value}
                if var.has_source_grid:
                    arr = np.asarray(value)
                    changes.update(
                        coordinate=None,
                        coordinate_values=None,
                        size=int(arr.shape[-1]) if arr.ndim else None,
                    )
                self.variables[name] = var.clone(**changes)

    def _run_guarded_once(self, mode: str, **options: Any) -> dict[str, Any]:
        """Run one mode with the current regime sustainment guard included."""
        base_include = self.relation_include
        self.relation_include = _unique_extend(base_include, _regime_guard_names(_confinement_regime(self.tags)))
        try:
            return self._run_once(mode, **options)
        finally:
            self.relation_include = base_include

    def _run_with_regime_verification(self, mode: str, **options: Any) -> dict[str, Any]:
        """Solve each candidate confinement mode and keep the admissible one.

        A mode is ADMISSIBLE when both halves agree on that mode's OWN solve:

          * its mode-dependent RELATIONS agree -- ``failed_relations`` is empty;
          * its CERTIFIERS agree -- every relation tagged
            ``("confinement_mode_threshold", <mode>)`` verifies.

        Certifiers are independent, declaratively-discovered conditions that AND
        together, so a new discriminant (an edge-state criterion, an
        accessibility limit) is added by writing one tagged relation and needs
        no change here.  A discriminant that happens to hold everywhere simply
        contributes nothing rather than overriding a sharper one.

        Both halves are required because either alone is misleading: a solve can
        converge to a point that is not accessible in that mode, and a mode can
        satisfy its thresholds on a solve whose power balance never closed.
        This mirrors what the popcon composite already does per grid point
        (``success & guards_hold``); the scalar path used to consult only the
        certifiers.

        The admissible COUNT is then the verdict:

          * exactly one   -- that mode;
          * more than one -- a genuine hysteresis band, both branches stable.
            The declared tag is the steady-state stand-in for "which branch the
            machine came from" and breaks the tie;
          * zero          -- NOT a confinement state.  With ``tau_E,L < tau_E,H``
            a free operating point gives ``P_sep,L < P_sep,H``, so the
            accessibility conditions may overlap but cannot both fail; zero
            admissible means that ordering is inverted, which happens when the
            operating point is pinned hard enough that the other mode cannot
            relax into it.  Reported as an over-constrained point, with each
            mode's reason, rather than silently returning one branch's numbers.

        The declared mode is solved FIRST and short-circuits when admissible,
        because it wins any tie anyway -- so the verdict never depends on the
        modes that were skipped.  ``regime_admissible`` is then not exhaustive:
        it says the declared mode is admissible, not that nothing else is.
        Pass ``regime_scan=True`` to solve every candidate and get the full
        accessible set, which is what detects a genuinely multistable point.
        MEASURED 2026-08-07: exhaustive costs ~2.9x on the reactor sweep
        (118s -> 345s) because an h_mode machine then also solves the I-mode
        scaling, which converges poorly on a machine not designed for it --
        ARC_V3A 8.6 -> 49.9s, STEP 19 -> 83s.  Hence opt-in.
        """
        scan_all = bool(options.pop("regime_scan", False))
        declared = _confinement_regime(self.tags)
        candidates = _candidate_regimes(declared)
        if not candidates:
            return self._run_once(mode, **options)

        attempts: dict[str, tuple[dict[str, Any], "Reactor"]] = {}
        reasons: dict[str, str] = {}
        admissible: list[str] = []
        path: list[str] = []

        for regime in candidates:
            clone = self._clone_for_regime(regime, include_guards=False)
            # The solve itself uses only this mode's tag and its confinement-time
            # relation.  A separate verify pass evaluates every certifier on the
            # solved values -- a certifier must never shape the solution it judges.
            result = clone._run_once(mode, **options)
            statuses = (clone._verify_all_regime_guards() or {}).get("relation_status") or {}
            attempts[regime] = (result, clone)
            path.append(regime)

            # A scan restores itself to pure inputs, so certifiers over derived
            # quantities cannot be evaluated afterwards.  That is not a verdict:
            # keep the declared mode and let per-point certification arbitrate.
            if _regime_guards_indeterminate(statuses, regime):
                self._absorb_regime_candidate(clone)
                self._annotate_regime_result(result, declared, regime, path, mode)
                return result

            failed = list(result.get("failed_relations") or ())
            certified = _regime_verified_by_guards(statuses, regime)
            if not failed and certified:
                admissible.append(regime)
                if regime == declared and not scan_all:
                    break  # declared wins any tie; the skipped modes cannot change it
                continue
            if failed:
                reasons[regime] = f"mode relations failed: {', '.join(sorted(failed))}"
            else:
                unmet = [
                    name for name in _regime_guard_names(regime)
                    if not bool((statuses.get(name) or {}).get("verified", False))
                ]
                reasons[regime] = f"certifiers not met: {', '.join(unmet)}"

        if admissible:
            selected = declared if declared in admissible else admissible[0]
            result, clone = attempts[selected]
            self._absorb_regime_candidate(clone)
            self._annotate_regime_result(result, declared, selected, path, mode)
            result["regime_admissible"] = list(admissible)
            result["regime_inadmissible_reasons"] = dict(reasons)
            if len(admissible) > 1:
                # Several modes are each self-consistent here.  Which one the
                # machine actually occupies is set by how it got here, so the
                # declared mode -- the stated branch -- decides when it is one
                # of them.  That is the defensible case.
                result["regime_bistable"] = True
                if selected == declared:
                    detail = (
                        f"kept the declared {selected}, which is one of them -- "
                        "the declaration is the branch selector."
                    )
                else:
                    # The declared mode is NOT admissible and several others are.
                    # Nothing physical separates them here: H and I differ by
                    # topology, drift direction and edge state, none of which is
                    # in the decision.  Say so rather than present order as physics.
                    result["regime_ambiguous"] = True
                    detail = (
                        f"the declared {declared} is not among them, so {selected} was "
                        "taken by candidate order, NOT by a physical discriminator -- "
                        "treat this as ambiguous."
                    )
                result.setdefault("warnings", []).insert(
                    0,
                    f"{len(admissible)} confinement modes are simultaneously admissible "
                    f"({', '.join(admissible)}): {detail}",
                )
            return result

        # Zero admissible.  Report the declared mode's solve -- it is the stated
        # assumption -- and say plainly that nothing was consistent, with why.
        selected = declared if declared in attempts else path[-1]
        result, clone = attempts[selected]
        self._absorb_regime_candidate(clone)
        self._annotate_regime_result(result, declared, selected, path, mode)
        result["regime_admissible"] = []
        result["regime_inadmissible_reasons"] = dict(reasons)
        result["regime_over_constrained"] = True
        detail = "; ".join(f"{regime}: {why}" for regime, why in reasons.items())
        result.setdefault("warnings", []).insert(
            0,
            "No confinement mode is admissible -- this indicates an over-constrained "
            f"operating point rather than a confinement state ({detail}). Reporting the "
            f"declared {selected} solve; its values are not a certified operating point.",
        )
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
                (
                    f"Declared {declared} operating condition is inconsistent with confinement-mode thresholds; "
                    f"switched to {selected} for {mode}."
                ),
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
        system = self.last_plan if self.last_plan is not None else self.relation_system().compile()
        excluded = set()
        for item in self.relation_exclude:
            try:
                excluded.add(RELATIONS.get(item).name)
            except KeyError:
                excluded.add(str(item))
        values = system.complete(system.solver_values())

        def guard_value(name: str, seen: frozenset[str] = frozenset()) -> Any:
            """Return a solved value, or the registry default for guard-only input.

            Checked-only guards are intentionally absent from the solved relation
            system.  Consequently, a variable used *only* by a guard is not part
            of the compiler's default-seeding plan.  Resolve that variable here
            using the same registry declaration so adding a guard does not force
            every existing reactor file to repeat ordinary defaults.
            """
            value = values.get(name)
            if value is not None:
                return value
            if name in seen or name not in VARIABLES:
                return None
            spec = VARIABLES.get(name)
            if spec.default is None:
                return None
            if isinstance(spec.default, str):
                source = VARIABLES.resolve(spec.default)
                return guard_value(source, seen | {name})
            return float(spec.default)

        status: dict[str, Any] = {}
        for name in _all_regime_guard_names():
            if name in excluded:
                continue
            rel = RELATIONS.get(name)
            guard_values = dict(values)
            for variable_name in rel.variables:
                if guard_values.get(variable_name) is None:
                    guard_values[variable_name] = guard_value(variable_name)
            missing = [v for v in rel.variables if guard_values.get(v) is None]
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
                status[name] = system.relation_status_and_residual(
                    rel,
                    system.relation_evaluation_values(rel, guard_values),
                )[0]
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
        clone = self.clone()
        clone.tags = _with_confinement_regime(self.tags, regime)
        clone.relation_include = self._candidate_relation_include(regime, include_guards=include_guards)
        return clone

    def _candidate_relation_include(self, regime: str, *, include_guards: bool = True) -> tuple[str, ...]:
        """Add optional regime guards and a mode-appropriate tau_E default relation."""
        extras = list(_regime_guard_names(regime)) if include_guards else []
        if not self._has_explicit_tau_e_scaling(regime):
            scaling = _regime_tau_default_name(regime)
            if scaling and scaling not in self.relation_exclude:
                extras.append(scaling)
        return _unique_extend(self._regime_filtered_include(regime), extras)

    def _regime_filtered_include(self, regime: str | None) -> tuple[str, ...]:
        """Includes that apply in ``regime``.

        An included relation carrying a ``confinement_mode`` tag applies only in
        that regime; one carrying none applies always.  A scaling already knows
        which regime it belongs to, so a reactor can name one per regime --
        ``include: [cfspopcon_itpa20_il_confinement_time, goldston_confinement_time]``
        reads as "ITPA20-IL in H-mode, Goldston in L-mode" -- and the regime
        driver picks the right one when it switches, instead of carrying an
        H-mode scaling into an L-mode solve.
        """
        if not regime:
            return self.relation_include
        order = set(_regime_order())
        kept: list[str] = []
        for identifier in self.relation_include:
            try:
                rel = RELATIONS.get(identifier)
            except KeyError:
                kept.append(identifier)      # unknown name: leave it to the registry
                continue
            modes = set(rel.tags) & order
            if modes and regime not in modes:
                continue
            kept.append(identifier)
        return tuple(kept)

    def _has_explicit_tau_e_scaling(self, regime: str | None = None) -> bool:
        """Whether relation_include names a tau_E producer *for this regime*.

        Per-regime so that an H-mode-only override still falls back to the
        regime default when the machine drops to L-mode, rather than silently
        reusing the H-mode scaling there.
        """
        for identifier in self._regime_filtered_include(regime):
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
        the candidate's solved state comes along on ``last_plan``.
        """
        self.tags = candidate.tags
        self.last_plan = candidate.last_plan
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

        Grid points are evaluated in vectorized chunks with this reactor's
        inputs held exactly as given and the axis values pinned to the grid
        coordinates; every point is then individually certified.
        See :mod:`fusdb.modes.popcon` for the options and result payload.

        Args:
            x: X-axis spec (variable name plus grid values/range).
            y: Y-axis spec.
            **options: Popcon-mode options (``outputs``, ``verbose``).

        Returns:
            Popcon result dictionary.
        """
        return self.run("popcon", x=x, y=y, **options)

    def run_many(
        self,
        cases: Iterable[Mapping[str, Any]],
        *,
        mode: str = "reconcile",
        workers: int | None = None,
        chunk_size: int | None = None,
        **options: Any,
    ) -> list[SolvedColumn]:
        """Run any mode over independent parameter cases."""
        return run_many(
            self,
            cases,
            mode=mode,
            workers=workers,
            chunk_size=chunk_size,
            **options,
        )

    def _run_popcon_auto_regime(self, **options: Any) -> dict[str, Any]:
        """Popcon scan with automatic per-point confinement regime.

        Instead of running the whole grid in one fixed regime, this runs the
        batched scan once per candidate regime (:func:`_candidate_regimes` of
        the declared tag -- the declared branch's escalation chain, e.g.
        ``h_mode``/``l_mode``, with both upper branches as candidates for an
        ``l_mode`` machine) and assigns each grid point the regime it actually
        sits in: the strongest regime whose sustainment guard holds on that
        regime's own solve (``P_sep >= P_HL`` for H-mode, ``P_sep >=
        P_LI_thresh`` for I-mode), with L-mode as the accessible fallback for
        cells no regime certified. The composited result carries a
        ``regime_index`` grid (index into ``regime_names``, ``-1`` where no
        regime certified).

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
        for regime in chain:
            # POPCON itself owns chunking and parallelism.  Keeping regime
            # composition serial avoids a second nested process-pool policy.
            per_regime[regime] = clones[regime]._run_once("popcon", **scan_options)
            for warning in per_regime[regime].get("warnings", ()):  # e.g. underivable output
                if warning not in warnings:
                    warnings.append(warning)
        self.last_plan = clones[chain[0]].last_plan

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
            # Each per-regime scan carries its own certificate; the composite
            # scope is full-system by default, or cone when the caller opted
            # into certification_targets (S6-full).
            "certificate": {
                "scope": "cone" if options.get("certification_targets") is not None else "full",
                "per_regime": True,
            },
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
        result = reactor.run(mode, **options)
        # Carry the value ``run`` returned rather than whatever the system
        # happened to record: ``popcon`` returns its scan payload without
        # writing ``last_result``, so snapshotting the system alone drops it.
        return _table_column(reactor.last_plan)._replace(result=result)
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


def _solve_reactor_chunk(
    tasks: tuple[tuple[str, str, Mapping[str, Any]], ...],
) -> list[SolvedColumn]:
    """Solve one ordered reactor chunk through the common scheduler."""
    return [_solve_reactor_path(task) for task in tasks]


def _run_case(
    reactor: Reactor,
    overrides: Mapping[str, Any],
    mode: str,
    options: Mapping[str, Any],
) -> SolvedColumn:
    """Pin one case's overrides, solve, and snapshot.

    Called inside a :func:`run_many` chunk.  Overrides are pinned as
    fixed inputs before the run, so the case's values are held exactly and the
    rest of the system reconciles/optimizes around them.  A failure is returned
    as a failed column, keeping the multi-case run alive.
    """
    try:
        for name, value in overrides.items():
            reactor.add_variable(Variable(name, value=value, fixed=True))
        result = reactor.run(mode, **options)
        # See _solve_reactor_path: take the result ``run`` returned so every
        # mode's payload survives the process boundary, popcon included.
        column = _table_column(reactor.last_plan)._replace(result=result)
        label = ", ".join(f"{name}={value:g}" if isinstance(value, (int, float)) else f"{name}={value}" for name, value in overrides.items())
        return column._replace(name=label or column.name)
    except Exception as exc:
        label = ", ".join(f"{name}={value}" for name, value in overrides.items())
        return SolvedColumn(label or "case", dict(overrides), {}, {}, {}, frozenset(), {}, {"success": False, "termination": f"crashed: {exc!r}", "errors": [repr(exc)]})


def _run_case_chunk(
    path: str,
    mode: str,
    options: Mapping[str, Any],
    cases: tuple[Mapping[str, Any], ...],
) -> list[SolvedColumn]:
    """Load once, then process independent scalar solves in one ordered chunk."""
    base = Reactor.from_yaml(path)
    return [_run_case(base.clone(), case, mode, options) for case in cases]


def run_many(
    reactor: "str | Path | Reactor",
    cases: Iterable[Mapping[str, Any]],
    *,
    mode: str = "reconcile",
    workers: int | None = None,
    chunk_size: int | None = None,
    **options: Any,
) -> list[SolvedColumn]:
    """Run any mode over parameter cases through the shared chunk scheduler.

    Each case is a mapping of ``{variable_name: value}`` overrides pinned as
    fixed inputs for that run; every other quantity is solved by ``mode``
    around them.  Chunks are scheduled by the same harness POPCON uses; scalar
    modes run independent solves within each chunk, while POPCON vectorizes the
    points inside its chunks.

    Args:
        reactor: Reactor YAML path or a :class:`Reactor` loaded from YAML
            (its ``source_path`` is reused; workers reload from it).
        cases: One ``{name: value}`` override mapping per case.
        mode: Mode run on each case (``reconcile``, ``optimize``, ...).
        workers: Maximum worker processes; ``None`` lets the pool decide.
        chunk_size: Cases handled by one worker task.
        **options: Mode options forwarded to every case.

    Returns:
        One :class:`SolvedColumn` per case, in order; render with
        :func:`fusdb.plotting.variable_table_data` or read ``.values`` /
        ``.result`` directly.
    """
    if isinstance(reactor, Reactor):
        if reactor.source_path is None:
            raise ValueError(f"Reactor {reactor.name!r} was not loaded from YAML; pass its path.")
        path = str(reactor.source_path)
    else:
        path = str(reactor)
    case_list = [dict(case) for case in cases]
    size = chunk_size or parallel_chunk_size(len(case_list), workers)
    worker = partial(_run_case_chunk, path, mode, dict(options))
    return map_chunks(case_list, worker, workers=workers, chunk_size=size)


def solve_reactors(
    paths: Iterable[str | Path | Reactor],
    *,
    mode: str = "reconcile",
    workers: int | None = None,
    chunk_size: int | None = None,
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
    resolved: list[str] = []
    for item in paths:
        if isinstance(item, Reactor):
            if item.source_path is None:
                raise ValueError(f"Reactor {item.name!r} was not loaded from YAML; pass its path.")
            resolved.append(str(item.source_path))
        else:
            resolved.append(str(item))

    tasks = [(path, mode, options) for path in resolved]
    size = chunk_size or parallel_chunk_size(len(tasks), workers)
    columns = map_chunks(
        tasks,
        _solve_reactor_chunk,
        workers=workers,
        chunk_size=size,
    )

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
