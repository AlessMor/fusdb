"""Variable registry loaded from ``variables.yaml``."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import yaml

from ..utils import (
    ZERO_TOL,
    coerce_to_shape,
    domain_bounds_for_solver,
    parse_constraint_specs,
    parse_domain,
    scipy_bounds,
    unique_preserve_order,
    validate_solver_domain,
    value_in_domain,
)


# Parsed registry-constraint guards, one tuple per spec name (process cache).
_SPEC_GUARDS: dict[str, tuple] = {}


class _UniqueKeyLoader(yaml.SafeLoader):
    """YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(loader: _UniqueKeyLoader, node: yaml.nodes.MappingNode, deep: bool = False) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            mark = key_node.start_mark
            raise ValueError(
                f"Duplicate YAML key/variable {key!r} in {mark.name} "
                f"at line {mark.line + 1}, column {mark.column + 1}."
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True, slots=True)
class VariableSpec:
    """Registry metadata and per-variable numerics for one canonical variable.

    The spec is the centralized owner of everything derivable from the
    registry definition alone: value-form conversion (public <-> solver),
    shape coercion, domain checks and tolerance/scale arithmetic.  Specs are
    built once per process; per-scenario data (profile size, resolved
    tolerances) is passed as arguments, so no per-run objects are needed.
    """

    name: str
    aliases: tuple[str, ...] = ()
    unit: str = "dimensionless"
    shape: int = 0
    domain: tuple[float | None, float | None, bool, bool] = (None, None, True, True)
    solver_domain: tuple[float | None, float | None, bool, bool] = (None, None, True, True)
    constraints: tuple[tuple[str, bool], ...] = ()
    description: str = ""
    rel_tol: float = 0.001
    abs_tol: float = 0.0
    average_variable: str | None = None
    default_relation: tuple[str, ...] = ()
    default: float | str | None = None
    default_requires: str | None = None
    # Declared order-of-magnitude start (public units) for solver unknowns
    # with no seed entitlement (block cores).  Purely a numerical initial
    # point -- a determined block converges to the same answer regardless --
    # never a value provider; ``None`` falls back to the tolerance floor.
    nominal: float | None = None
    # Numeric constants precomputed once at construction (pure functions of
    # ``domain``/``solver_domain``); the hot conversion paths read these
    # instead of re-deriving bounds per call.
    solver_bounds: tuple[float, float] = field(init=False, repr=False, compare=False, default=(-np.inf, np.inf))
    # Per side: ``(physical boundary, projection target)`` or None when values
    # on that boundary need no projection (the common case -> fast path).
    solver_proj_low: tuple[float, float] | None = field(init=False, repr=False, compare=False, default=None)
    solver_proj_high: tuple[float, float] | None = field(init=False, repr=False, compare=False, default=None)
    # Per side: ``(solver boundary, physical boundary, atol)`` or None.
    public_proj_low: tuple[float, float, float] | None = field(init=False, repr=False, compare=False, default=None)
    public_proj_high: tuple[float, float, float] | None = field(init=False, repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        s_lo, s_hi = scipy_bounds(self.solver_domain, zero_tol=ZERO_TOL)
        object.__setattr__(self, "solver_bounds", (s_lo, s_hi))
        d_lo, d_hi, d_lo_inc, d_hi_inc = self.domain
        # Solver projection: physical-boundary values map onto the solver bound
        # (or just inside an exclusive bound).
        if d_lo is not None:
            lo = float(d_lo)
            target = float(s_lo) if np.isfinite(s_lo) and s_lo > lo else (lo + ZERO_TOL if not d_lo_inc else None)
            if target is not None:
                object.__setattr__(self, "solver_proj_low", (lo, target))
        if d_hi is not None:
            hi = float(d_hi)
            target = float(s_hi) if np.isfinite(s_hi) and s_hi < hi else (hi - ZERO_TOL if not d_hi_inc else None)
            if target is not None:
                object.__setattr__(self, "solver_proj_high", (hi, target))
        # Public projection: solver-bound values map back onto an inclusive
        # physical bound they stand in for.
        p_lo, p_hi = domain_bounds_for_solver(self.solver_domain, zero_tol=ZERO_TOL)
        if p_lo is not None and d_lo is not None and d_lo_inc and not np.isclose(float(p_lo), float(d_lo), rtol=0.0, atol=ZERO_TOL):
            object.__setattr__(self, "public_proj_low", (float(p_lo), float(d_lo), max(ZERO_TOL, abs(p_lo) * 1e-10)))
        if p_hi is not None and d_hi is not None and d_hi_inc and not np.isclose(float(p_hi), float(d_hi), rtol=0.0, atol=ZERO_TOL):
            object.__setattr__(self, "public_proj_high", (float(p_hi), float(d_hi), max(ZERO_TOL, abs(p_hi) * 1e-10)))

    @property
    def constraint_relations(self) -> tuple:
        """Registry-constraint guard relations, parsed once per process.

        Lazy because relation construction imports would otherwise cycle at
        module load; cached on the class-level dict keyed by spec name.
        """
        cached = _SPEC_GUARDS.get(self.name)
        if cached is None:
            from ..relation import constraint_from_expression

            cached = tuple(
                constraint_from_expression(
                    text,
                    name=f"{self.name}_registry_constraint_{index}",
                    enforce=enforce,
                    source_kind="variable",
                    source_name=self.name,
                )
                for index, (text, enforce) in enumerate(self.constraints)
            )
            _SPEC_GUARDS[self.name] = cached
        return cached

    @property
    def canonical_name(self) -> str:
        """The canonical variable name.

        Same value as ``name``; the explicit accessor used wherever a canonical
        name string is produced from a spec (resolution boundaries, error and
        warning messages).
        """
        return self.name

    # ── Value forms and domain checks ──────────────────────────────────────

    def coerce(self, value: Any, size: int) -> Any:
        """Return ``value`` with this variable's registry shape.

        Scalar variables must remain scalar.  A profile-shaped value for a
        scalar variable is a relation/planning error and is rejected instead of
        being displayed or stored as a fake scalar result.  Profile variables
        may receive a scalar, which is broadcast to the ``size`` grid.
        """
        coerced, _size = coerce_to_shape(
            self.name, value, is_profile=self.shape == 1, size=size if self.shape == 1 else None,
            squeeze_scalar=True, reject_nan=True,
        )
        return coerced

    def solver_value(self, value: Any, size: int) -> Any:
        """Convert a public value to canonical solver shape.

        Values lying exactly on a physical-domain boundary are projected onto
        the corresponding solver-domain boundary; this is the inverse of
        :meth:`public_value`, so a profile edge stored publicly as ``T = 0`` is
        evaluated at the numerically safe solver bound (for example 1e-12)
        instead of hitting singular physics formulas.  Only values within
        ``ZERO_TOL`` of the boundary are projected: interior values and real
        domain violations are never clipped.
        """
        if self.solver_proj_low is None and self.solver_proj_high is None:
            # Fast path: no projection applies; only shape coercion matters.
            # NaN must still be rejected (``value == value`` is False for NaN),
            # matching the general path's reject_nan coercion.
            if self.shape == 0 and isinstance(value, float) and value == value:
                return value
            if self.shape == 1 and isinstance(value, np.ndarray) and value.ndim == 1 and value.shape[0] == size and value.dtype == np.float64 and not np.isnan(value).any():
                return value
            return self.coerce(value, size)
        out = np.asarray(self.coerce(value, size), dtype=float).astype(float, copy=True)
        if self.solver_proj_low is not None:
            lo, target = self.solver_proj_low
            out = np.where((out >= lo - ZERO_TOL) & (out <= lo + ZERO_TOL), target, out)
        if self.solver_proj_high is not None:
            hi, target = self.solver_proj_high
            out = np.where((out >= hi - ZERO_TOL) & (out <= hi + ZERO_TOL), target, out)
        return float(out) if out.ndim == 0 else out

    def public_value(self, value: Any, size: int) -> Any:
        """Project solver-boundary values to physical-domain boundary values.

        The inverse of :meth:`solver_value`: a value sitting on a solver bound
        that stands in for an inclusive physical bound is stored publicly at
        the physical bound itself.
        """
        if self.public_proj_low is None and self.public_proj_high is None:
            if self.shape == 0 and isinstance(value, float) and value == value:
                return value
            if self.shape == 1 and isinstance(value, np.ndarray) and value.ndim == 1 and value.shape[0] == size and value.dtype == np.float64 and not np.isnan(value).any():
                return value
            return self.coerce(value, size)
        arr = np.asarray(self.coerce(value, size), dtype=float).copy()
        if self.public_proj_low is not None:
            s_lo, d_lo, atol = self.public_proj_low
            arr = np.where(np.isclose(arr, s_lo, rtol=0.0, atol=atol), d_lo, arr)
        if self.public_proj_high is not None:
            s_hi, d_hi, atol = self.public_proj_high
            arr = np.where(np.isclose(arr, s_hi, rtol=0.0, atol=atol), d_hi, arr)
        return float(arr) if arr.ndim == 0 else arr

    def check_solver_domain(self, value: Any, size: int) -> None:
        """Raise if ``value`` is outside this variable's solver domain."""
        lb, ub = self.solver_bounds
        arr = np.asarray(self.solver_value(value, size), dtype=float)
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Variable {self.name!r} initial value is not finite.")
        if np.isfinite(lb) and np.any(arr < lb):
            raise ValueError(f"Variable {self.name!r} initial value is below solver_domain lower bound {lb}.")
        if np.isfinite(ub) and np.any(arr > ub):
            raise ValueError(f"Variable {self.name!r} initial value is above solver_domain upper bound {ub}.")

    def candidate_valid(self, value: Any, size: int) -> bool:
        """Return whether a prospective candidate value is finite and inside the domain."""
        try:
            arr = np.asarray(value, dtype=float)
            if arr.size == 0 or not np.all(np.isfinite(arr)):
                return False
            return value_in_domain(self.public_value(value, size), self.domain, zero_tol=0.0)
        except Exception:
            return False

    # ── Tolerances, scales and movement ────────────────────────────────────
    # ``rel_tol``/``abs_tol`` arguments are the per-scenario resolved values
    # (a per-variable override or this spec's defaults).

    def tolerance_floor(self, rel_tol: float, abs_tol: float) -> float:
        """Return the finite scale floor implied by abs_tol / rel_tol."""
        if rel_tol > 0.0 and abs_tol > 0.0:
            return max(abs_tol / rel_tol, 1.0e-300)
        if abs_tol > 0.0:
            return max(abs_tol, 1.0e-300)
        return 1.0e-300

    def tolerance_width(self, scale: Any, rel_tol: float, abs_tol: float) -> np.ndarray:
        """Return the physical tolerance width for a given scale."""
        scl = np.maximum(np.asarray(scale, dtype=float), 1.0e-300)
        return np.maximum(abs_tol, rel_tol * scl)

    def scale_of(self, rel_tol: float, abs_tol: float, *values: Any) -> float:
        """Return the residual/movement scale for this variable.

        The scale is the largest of the abs_tol / rel_tol floor and the finite
        magnitudes of the supplied reference values.  It intentionally ignores
        physical and solver-domain bounds; relation-residual scaling and
        movement-penalty scaling share this one definition.
        """
        magnitudes = [self.tolerance_floor(rel_tol, abs_tol)]
        for value in values:
            if value is None:
                continue
            try:
                arr = np.asarray(value, dtype=float).reshape(-1)
            except Exception:
                continue
            finite = arr[np.isfinite(arr)]
            if finite.size:
                magnitudes.append(float(np.max(np.abs(finite))))
        return max(magnitudes)

    @property
    def movement_is_multiplicative(self) -> bool:
        """Whether movement is measured in decades rather than absolute units.

        True when the physical domain excludes zero from below -- ``(0, inf)``
        and friends -- so the quantity is strictly positive and ``log(x/x0)``
        is always defined.  It matters because the absolute metric is *bounded*
        for a collapse toward zero: ``|x - x0|`` can never exceed ``x0``, so
        driving a positive input to 1e-12 costs at most ``x0/width = 1/rel_tol``
        tolerance widths -- no more than doubling it, and with no gradient left
        to pull it back.  The log metric is unbounded there instead.

        Variables whose domain *includes* zero (``f_He4``, every ``c_*``) keep
        the absolute metric: zero is a legitimate value for them, and any
        metric that stays finite at zero necessarily bounds the collapse.
        """
        lower, _upper, lower_inclusive, _upper_inclusive = self.domain
        if lower is None:
            return False
        return lower > 0.0 or (lower == 0.0 and not lower_inclusive)

    def movement_log_width(self, width: float, reference: Any) -> float | None:
        """Deadzone half-width in log space, or None to use absolute movement.

        ``log1p(width / |reference|)`` is the log-distance spanned by one
        absolute tolerance width at the reference, so the two metrics agree to
        first order for small deviations and the deadzone boundary does not
        move; only the far field changes.
        """
        if not self.movement_is_multiplicative:
            return None
        ref = np.asarray(reference, dtype=float).reshape(-1)
        ref = ref[np.isfinite(ref)]
        if ref.size == 0 or np.any(ref <= 0.0):
            return None
        log_width = float(np.log1p(width / float(np.min(np.abs(ref)))))
        return log_width if log_width > 0.0 else None

    def movement_excess(self, current: Any, reference: Any, rel_tol: float, abs_tol: float) -> float:
        """Return this input's worst movement past its tolerance band.

        The deadzone excess ``max(distance - 1, 0)`` reduced over the
        variable's points to its worst point: zero while the input stays within
        tolerance, growing once it crosses.  This is the per-input quantity the
        reconcile objective drives toward zero for as many inputs as possible.

        ``distance`` is ``|value - input| / tolerance``, or the multiplicative
        ``|log(value / input)| / log_width`` for strictly-positive variables
        (see :attr:`movement_is_multiplicative`).
        """
        cur = np.asarray(current, dtype=float).reshape(-1)
        ref = np.asarray(reference, dtype=float).reshape(-1)
        if cur.size == 0 or cur.shape != ref.shape:
            return 0.0
        width = self.tolerance_width(self.scale_of(rel_tol, abs_tol, reference), rel_tol, abs_tol)
        log_width = self.movement_log_width(float(np.max(width)), reference)
        if log_width is not None and bool(np.all(cur > 0.0)):
            distance = np.abs(np.log(cur / ref)) / log_width
        else:
            distance = np.abs(cur - ref) / np.maximum(np.broadcast_to(width, cur.shape), 1.0e-300)
        return float(np.max(np.maximum(distance - 1.0, 0.0)))

    def domain_violation_rows(self, value: Any, rel_tol: float, abs_tol: float) -> list[np.ndarray]:
        """Return tolerance-normalized physical-domain violation rows.

        Zero inside the physical domain, positive outside it, normalized by the
        tolerance width.  A profile's feasibility is governed by its extremes,
        so the per-point penalty is reduced to the worst point -- exact for
        feasibility and one row per bound instead of one per grid point.  A
        non-numeric or non-finite value yields one large row per bound.
        """
        lower, upper, lower_inc, upper_inc = self.domain
        if lower is None and upper is None:
            return []
        sides = int(lower is not None) + int(upper is not None)
        try:
            arr = np.asarray(value, dtype=float).reshape(-1)
        except Exception:
            return [np.full(sides, 1.0e12, dtype=float)]
        if not np.all(np.isfinite(arr)):
            return [np.full(sides, 1.0e12, dtype=float)]
        tol = np.maximum(self.tolerance_width(np.maximum(np.abs(arr), self.tolerance_floor(rel_tol, abs_tol)), rel_tol, abs_tol), 1.0e-300)
        rows: list[np.ndarray] = []
        is_profile = arr.size > 1
        if lower is not None:
            boundary = float(lower) + (ZERO_TOL if not lower_inc else 0.0)
            viol = np.maximum(boundary - arr, 0.0) / tol
            rows.append(np.asarray([float(np.max(viol))]) if is_profile else viol)
        if upper is not None:
            boundary = float(upper) - (ZERO_TOL if not upper_inc else 0.0)
            viol = np.maximum(arr - boundary, 0.0) / tol
            rows.append(np.asarray([float(np.max(viol))]) if is_profile else viol)
        return rows


class VariableRegistry:
    """Registry of allowed variables and aliases.

    The registry only stores shared, process-wide metadata (one
    :class:`VariableSpec` per canonical name).  A scenario's declared values
    live on immutable :class:`~fusdb.variable.Variable` records at the
    boundary; a solve's working/solved values live on the
    :class:`~fusdb.relationsystem.RelationSystem` that ran it -- the registry
    itself never holds either.
    """

    def __init__(self, specs: Iterable[VariableSpec], *, rel_tol_default: float = 0.001, profile_size_default: int = 46) -> None:
        self.rel_tol_default = float(rel_tol_default)
        self.profile_size_default = int(profile_size_default)
        by_name: dict[str, VariableSpec] = {}
        alias_to_name: dict[str, str] = {}
        for spec in specs:
            if spec.name in by_name:
                raise ValueError(f"Duplicate variable {spec.name!r}.")
            by_name[spec.name] = spec
            alias_to_name[spec.name] = spec.name
            for alias in spec.aliases:
                if alias in alias_to_name:
                    raise ValueError(f"Alias {alias!r} is ambiguous.")
                alias_to_name[alias] = spec.name
        self._specs = MappingProxyType(by_name)
        self._alias_to_name = MappingProxyType(alias_to_name)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "VariableRegistry":
        """Load a registry from YAML."""
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = yaml.load(handle, Loader=_UniqueKeyLoader) or {}
        defaults = raw.pop("defaults", {}) if isinstance(raw, dict) else {}
        rel_tol_default = float(defaults.get("rel_tol", 0.001)) if isinstance(defaults, Mapping) else 0.001
        abs_tol_default = float(defaults.get("abs_tol", 0.0)) if isinstance(defaults, Mapping) else 0.0
        profile_size_default = int(defaults.get("profile_size", 46)) if isinstance(defaults, Mapping) else 46
        specs: list[VariableSpec] = []
        for name, entry in raw.items():
            if not isinstance(entry, Mapping):
                raise TypeError(f"Variable {name!r} must be a mapping.")
            aliases = unique_preserve_order(entry.get("aliases", ()) or ())
            unit = str(entry.get("default_unit", entry.get("unit", "dimensionless")))
            shape = int(entry.get("shape", entry.get("ndim", 0)) or 0)
            domain = parse_domain(entry.get("domain"))
            solver_domain = parse_domain(entry.get("solver_domain", entry.get("domain")))
            validate_solver_domain(str(name), domain, solver_domain)
            constraints = parse_constraint_specs(entry.get("constraints"))
            description = entry.get("description", "")
            if not isinstance(description, str):
                raise TypeError(f"Variable {name!r}: description must be a string, got {type(description).__name__}.")
            rel_tol = float(entry.get("rel_tol", entry.get("rel_tol_defaultpervar", rel_tol_default)))
            abs_tol = float(entry.get("abs_tol", abs_tol_default))
            default_relation = entry.get("default_relation", ()) or ()
            if isinstance(default_relation, str):
                default_relation = (default_relation,)
            # A seeding default is either a number (constant x0 seed) or the name
            # of another variable (copy that variable's value at seed time).  It
            # is applied only when the variable is neither supplied nor derivable
            # from supplied data, and is never enforced -- a relation that
            # determines the variable moves it off the seed.
            default = entry.get("default")
            if default is not None and not isinstance(default, str):
                default = float(default)
            # An optional gate: the default is only applied (seeded) when this
            # other variable is itself available.  Used for composition that is
            # only meaningful given a precondition (the He ash fractions need a
            # particle confinement time ``tau_p`` for the balance to pin them).
            default_requires = entry.get("default_requires")
            nominal = entry.get("nominal")
            average_variable = entry.get("average_variable")
            specs.append(
                VariableSpec(
                    name=str(name),
                    aliases=aliases,
                    unit=unit,
                    shape=shape,
                    domain=domain,
                    solver_domain=solver_domain,
                    constraints=constraints,
                    description=description,
                    rel_tol=rel_tol,
                    abs_tol=abs_tol,
                    average_variable=None if average_variable is None else str(average_variable),
                    default_relation=tuple(str(item) for item in default_relation),
                    default=default,
                    default_requires=None if default_requires is None else str(default_requires),
                    nominal=None if nominal is None else float(nominal),
                )
            )
        return cls(specs, rel_tol_default=rel_tol_default, profile_size_default=profile_size_default)

    def average_of(self, name: str) -> str | None:
        """Return the scalar-average variable controlling a profile, or ``None``.

        Resolves the ``average_variable`` metadata, falling back to the
        ``<name>_avg`` alias convention.  This is the single source of truth for
        the profile -> average mapping; consumers must not re-implement the
        convention.
        """
        if name not in self:
            return None
        spec = self.get(name)
        if spec.average_variable:
            return self.resolve(spec.average_variable)
        alias = f"{name}_avg"
        if alias in self:
            return self.resolve(alias)
        return None

    def uniform_profile_grid(self, size: int) -> np.ndarray:
        """Return the canonical uniform profile coordinate grid of ``size`` points."""
        return np.linspace(0.0, 1.0, int(size))

    def resolve(self, name: str) -> str:
        """Resolve a canonical name or alias. Raises for unknown names."""
        try:
            return self._alias_to_name[str(name)]
        except KeyError as exc:
            raise KeyError(f"Unknown variable {name!r}.") from exc

    def get(self, name: str) -> VariableSpec:
        """Return one variable spec by name or alias."""
        return self._specs[self.resolve(name)]

    def __getitem__(self, name: str) -> VariableSpec:
        return self.get(name)

    def __contains__(self, name: object) -> bool:
        return str(name) in self._alias_to_name

    def __iter__(self):
        return iter(self._specs.values())

    def __len__(self) -> int:
        return len(self._specs)


_DEFAULT_PATH = Path(__file__).with_name("variables.yaml")
VARIABLES = VariableRegistry.from_yaml(_DEFAULT_PATH) if _DEFAULT_PATH.exists() else VariableRegistry(())
