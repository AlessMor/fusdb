"""Scenario variable object."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .relation import Relation, constraint_from_expression
from .registry import VARIABLES, VariableSpec, convert_value
from .utils import (
    ZERO_TOL,
    coerce_numeric_value,
    coerce_to_shape,
    domain_bounds_for_solver,
    parse_constraint_specs,
    scipy_bounds,
    value_in_domain,
)


@dataclass
class Variable:
    """One active scalar or profile variable.

    A ``Variable`` is its registry :class:`VariableSpec` (the immutable
    definition: name, aliases, unit, shape, domain, tolerances) plus the
    per-scenario state (``value``/``input_value``/``fixed``).  Definition
    metadata is read through ``self.spec``; it is never copied onto the
    instance, so a variable cannot drift out of sync with its registry.

    Args:
        name: Canonical variable name or alias.
        value: Scalar, one-dimensional profile, or None.
        unit: Unit of the supplied ``value``. If omitted, the registry default is assumed.
        rel_tol: Relative tolerance override.
        fixed: Whether solve modes may change this value.
        size: Profile length for one-dimensional variables.
        constraints: Additional local constraints or applicability guards.
    """

    name: str
    value: Any = None
    unit: str | None = None
    rel_tol: float | None = None
    abs_tol: float | None = None
    fixed: bool = False
    size: int | None = None
    constraints: Any = None
    spec: VariableSpec = field(default=None, init=False)
    input_value: Any = field(default=None, init=False)
    relations: tuple[Relation, ...] = field(default_factory=tuple, init=False)

    @property
    def aliases(self) -> tuple[str, ...]:
        """Registry aliases for this variable."""
        return self.spec.aliases

    @property
    def shape(self) -> int:
        """Registry shape: 0 for scalars, 1 for profiles."""
        return self.spec.shape

    def __post_init__(self) -> None:
        """Resolve registry metadata and normalize the value."""
        self.spec = spec = VARIABLES.get(self.name)
        self.name = spec.name
        self.rel_tol = spec.rel_tol if self.rel_tol is None else float(self.rel_tol)
        self.abs_tol = spec.abs_tol if self.abs_tol is None else float(self.abs_tol)
        self.value = coerce_numeric_value(self.value)
        if self.value is not None:
            self.value = convert_value(self.value, from_unit=self.unit or spec.unit, to_unit=spec.unit)
        self.unit = spec.unit  # value is now in canonical units
        self.input_value = self._copy_value(self.value)

        # Validate profile shape and physical domain.
        if self.size is not None:
            self.size = int(self.size)
            if self.size <= 0:
                raise ValueError(f"Variable {self.name!r} size must be positive.")
        if self.shape == 0 and self.size is not None:
            raise ValueError(f"Scalar variable {self.name!r} cannot define a profile size.")
        if self.value is not None and not value_in_domain(self.value, spec.domain):
            raise ValueError(f"Variable {self.name!r} value is outside domain {spec.domain!r}.")
        if self.shape == 1 and self.value is not None:
            self.value, self.size = coerce_to_shape(self.name, self.value, is_profile=True, size=self.size)

        # Variable constraints are relation guards attached to the variable.
        built: list[Relation] = []
        for index, (text, enforce) in enumerate(parse_constraint_specs(spec.constraints)):
            built.append(
                constraint_from_expression(
                    text,
                    name=f"{self.name}_registry_constraint_{index}",
                    enforce=enforce,
                    source_kind="variable",
                    source_name=self.name,
                )
            )
        for index, (text, enforce) in enumerate(parse_constraint_specs(self.constraints)):
            built.append(
                constraint_from_expression(
                    text,
                    name=f"{self.name}_constraint_{index}",
                    enforce=enforce,
                    source_kind="variable",
                    source_name=self.name,
                )
            )
        self.relations = tuple(built)

    def clone(self, **changes: Any) -> "Variable":
        """Return a fresh variable with selected field overrides.

        Args:
            **changes: Constructor field overrides.

        Returns:
            New Variable instance.
        """
        data = {
            "name": self.name,
            "value": self._copy_value(self.input_value),
            "unit": self.spec.unit,
            "rel_tol": self.rel_tol,
            "abs_tol": self.abs_tol,
            "fixed": self.fixed,
            "size": self.size,
            "constraints": self.constraints,
        }
        data.update(changes)
        return Variable(**data)

    def _normalize_value(self, value: Any) -> Any:
        """Normalize a canonical-unit value to this variable shape."""
        if value is None:
            return None
        coerced, self.size = coerce_to_shape(
            self.name, value, is_profile=self.shape == 1, size=self.size
        )
        return coerced

    def set_input(self, value: Any) -> None:
        """Set the user/input value in canonical units.

        Args:
            value: New canonical-unit value.
        """
        normalized = self._normalize_value(value)
        if normalized is not None and not value_in_domain(normalized, self.spec.domain, zero_tol=0.0):
            raise ValueError(f"Variable {self.name!r} value is outside domain {self.spec.domain!r}.")
        self.input_value = self._copy_value(normalized)
        self.value = self._copy_value(normalized)

    def set_value(self, value: Any) -> None:
        """Set the current public value in canonical units.

        Args:
            value: New canonical-unit value.
        """
        normalized = self._normalize_value(value)
        if normalized is not None and not value_in_domain(normalized, self.spec.domain, zero_tol=0.0):
            raise ValueError(f"Variable {self.name!r} value is outside domain {self.spec.domain!r}.")
        self.value = self._copy_value(normalized)

    def _copy_value(self, value: Any) -> Any:
        """Copy a scalar/array value.

        Args:
            value: Value to copy.

        Returns:
            Independent copy where appropriate.
        """
        if isinstance(value, np.ndarray):
            return value.copy()
        return value

    # ── Value forms, dimension, scales and tolerances ──────────────────────
    # Per-variable numerics owned by the variable itself: everything below
    # reads only the registry spec plus this instance's size/tolerances, so a
    # RelationSystem consults these instead of re-deriving them per name.

    @property
    def dim(self) -> int:
        """Number of scalar elements: 1 for scalars, the grid size for profiles."""
        if self.shape != 1:
            return 1
        return int(self.size or VARIABLES.profile_size_default)

    def coerce_shape(self, value: Any) -> Any:
        """Return ``value`` with this variable's registry shape.

        Scalar variables must remain scalar.  A profile-shaped value for a
        scalar variable is a relation/planning error and is rejected instead of
        being displayed or stored as a fake scalar result.  Profile variables
        may receive a scalar, which is broadcast to the profile grid.
        """
        size = self.dim if self.shape == 1 else None
        coerced, _size = coerce_to_shape(
            self.name, value, is_profile=self.shape == 1, size=size, squeeze_scalar=True, reject_nan=True
        )
        return coerced

    def solver_value(self, value: Any) -> Any:
        """Convert a public value to canonical solver shape.

        Values lying exactly on a physical-domain boundary are projected onto
        the corresponding solver-domain boundary; this is the inverse of
        :meth:`public_value`, so a profile edge stored publicly as ``T = 0`` is
        evaluated at the numerically safe solver bound (for example 1e-12)
        instead of hitting singular physics formulas.  Only values within
        ``ZERO_TOL`` of the boundary are projected: interior values and real
        domain violations are never clipped.
        """
        arr = np.asarray(self.coerce_shape(value), dtype=float)
        out = arr.astype(float, copy=True)
        d_lo, d_hi, d_lo_inc, d_hi_inc = self.spec.domain
        s_lo, s_hi = scipy_bounds(self.spec.solver_domain, zero_tol=ZERO_TOL)
        if d_lo is not None:
            lo = float(d_lo)
            if np.isfinite(s_lo) and s_lo > lo:
                target = float(s_lo)
            elif not d_lo_inc:
                target = lo + ZERO_TOL
            else:
                target = None
            if target is not None:
                out = np.where((out >= lo - ZERO_TOL) & (out <= lo + ZERO_TOL), target, out)
        if d_hi is not None:
            hi = float(d_hi)
            if np.isfinite(s_hi) and s_hi < hi:
                target = float(s_hi)
            elif not d_hi_inc:
                target = hi - ZERO_TOL
            else:
                target = None
            if target is not None:
                out = np.where((out >= hi - ZERO_TOL) & (out <= hi + ZERO_TOL), target, out)
        return float(out) if out.ndim == 0 else out

    def public_value(self, value: Any) -> Any:
        """Project solver-boundary values to physical-domain boundary values.

        The inverse of :meth:`solver_value`: a value sitting on a solver bound
        that stands in for an inclusive physical bound is stored publicly at
        the physical bound itself.
        """
        d_lo, d_hi, d_lo_inc, d_hi_inc = self.spec.domain
        s_lo, s_hi = domain_bounds_for_solver(self.spec.solver_domain, zero_tol=ZERO_TOL)
        arr = np.asarray(self.coerce_shape(value), dtype=float).copy()
        if s_lo is not None and d_lo is not None and d_lo_inc and not np.isclose(float(s_lo), float(d_lo), rtol=0.0, atol=ZERO_TOL):
            arr = np.where(np.isclose(arr, s_lo, rtol=0.0, atol=max(ZERO_TOL, abs(s_lo) * 1e-10)), d_lo, arr)
        if s_hi is not None and d_hi is not None and d_hi_inc and not np.isclose(float(s_hi), float(d_hi), rtol=0.0, atol=ZERO_TOL):
            arr = np.where(np.isclose(arr, s_hi, rtol=0.0, atol=max(ZERO_TOL, abs(s_hi) * 1e-10)), d_hi, arr)
        return float(arr) if arr.ndim == 0 else arr

    def check_solver_domain(self, value: Any) -> None:
        """Raise if ``value`` is outside this variable's solver domain."""
        lb, ub = scipy_bounds(self.spec.solver_domain, zero_tol=ZERO_TOL)
        arr = np.asarray(self.solver_value(value), dtype=float)
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Variable {self.name!r} initial value is not finite.")
        if np.isfinite(lb) and np.any(arr < lb):
            raise ValueError(f"Variable {self.name!r} initial value is below solver_domain lower bound {lb}.")
        if np.isfinite(ub) and np.any(arr > ub):
            raise ValueError(f"Variable {self.name!r} initial value is above solver_domain upper bound {ub}.")

    def candidate_valid(self, value: Any) -> bool:
        """Return whether a prospective candidate value is finite and inside the domain."""
        try:
            arr = np.asarray(value, dtype=float)
            if arr.size == 0 or not np.all(np.isfinite(arr)):
                return False
            return value_in_domain(self.public_value(value), self.spec.domain, zero_tol=0.0)
        except Exception:
            return False

    def tolerance_floor(self) -> float:
        """Return the finite scale floor implied by abs_tol / rel_tol."""
        rel_tol = float(self.rel_tol) if self.rel_tol is not None else float(VARIABLES.rel_tol_default)
        abs_tol = float(self.abs_tol) if self.abs_tol is not None else 0.0
        if rel_tol > 0.0 and abs_tol > 0.0:
            return max(abs_tol / rel_tol, 1.0e-300)
        if abs_tol > 0.0:
            return max(abs_tol, 1.0e-300)
        return 1.0e-300

    def scale(self, *values: Any) -> float:
        """Return the residual/movement scale for this variable.

        The scale is the largest of the abs_tol / rel_tol floor and the finite
        magnitudes of the supplied reference values.  It intentionally ignores
        physical and solver-domain bounds, including unbounded or artificial
        large bounds.  Both relation-residual scaling and movement-penalty
        scaling use this same definition.
        """
        magnitudes = [self.tolerance_floor()]
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

    def tolerance_width(self, scale: Any) -> np.ndarray:
        """Return the physical tolerance width for a given scale."""
        rel_tol = float(self.rel_tol or VARIABLES.rel_tol_default)
        abs_tol = float(self.abs_tol or 0.0)
        scl = np.maximum(np.asarray(scale, dtype=float), 1.0e-300)
        return np.maximum(abs_tol, rel_tol * scl)

    def movement_excess(self, current: Any, reference: Any) -> float:
        """Return this input's worst movement past its tolerance band.

        The deadzone excess ``max(|value - input| / tolerance - 1, 0)`` reduced
        over the variable's points to its worst point, so it is zero while the
        input stays within tolerance and grows once it crosses.  This is the
        per-input quantity the reconcile objective drives toward zero for as
        many inputs as possible.

        Args:
            current: Solved value (scalar or profile).
            reference: Supplied input value in the same solver units.
        """
        cur = np.asarray(current, dtype=float).reshape(-1)
        ref = np.asarray(reference, dtype=float).reshape(-1)
        if cur.size == 0 or cur.shape != ref.shape:
            return 0.0
        tol = np.maximum(np.broadcast_to(self.tolerance_width(self.scale(reference)), cur.shape), 1.0e-300)
        return float(np.max(np.maximum(np.abs(cur - ref) / tol - 1.0, 0.0)))

    def domain_violation_rows(self, value: Any) -> list[np.ndarray]:
        """Return tolerance-normalized physical-domain violation rows.

        Zero inside the physical domain, positive outside it, normalized by the
        tolerance width.  A profile's feasibility is governed by its extremes,
        not by each of its grid points: it lies in-domain iff its minimum
        clears the lower bound and its maximum clears the upper bound, so the
        per-point penalty is reduced to the worst point -- exact for
        feasibility and one row per bound instead of one per grid point.  A
        non-numeric or non-finite value yields one large row per bound.
        """
        lower, upper, lower_inc, upper_inc = self.spec.domain
        if lower is None and upper is None:
            return []
        sides = int(lower is not None) + int(upper is not None)
        try:
            arr = np.asarray(value, dtype=float).reshape(-1)
        except Exception:
            return [np.full(sides, 1.0e12, dtype=float)]
        if not np.all(np.isfinite(arr)):
            return [np.full(sides, 1.0e12, dtype=float)]
        tol = np.maximum(self.tolerance_width(np.maximum(np.abs(arr), self.tolerance_floor())), 1.0e-300)
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

    def moved_from_input(self, value: Any) -> bool:
        """Whether a candidate value moved off this variable's supplied input.

        Compares in solver form with a tight absolute tolerance scaled to the
        input magnitude; used to reject candidate solves that changed a fixed
        variable.
        """
        old = np.asarray(self.solver_value(self.input_value), dtype=float).reshape(-1)
        new = np.asarray(value, dtype=float).reshape(-1)
        atol = max(ZERO_TOL, 1e-10 * max(1.0, float(np.max(np.abs(old))) if old.size else 1.0))
        return old.shape != new.shape or not np.allclose(old, new, rtol=0.0, atol=atol)

    def movement_reference(self, fallback: Any, index: int | None = None) -> Any:
        """Return one supplied-input element for movement scaling, or ``fallback``."""
        if self.input_value is not None:
            try:
                arr = np.asarray(self.solver_value(self.input_value), dtype=float).reshape(-1)
                if arr.size:
                    return float(arr[min(index or 0, arr.size - 1)])
            except Exception:
                pass
        return fallback
