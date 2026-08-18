"""Acausal relation object and ``@relation`` decorator."""

from __future__ import annotations

import ast
import functools
import inspect
import operator
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from numbers import Real
from typing import Any, Callable

import numpy as np
from scipy.optimize import least_squares, root_scalar

from .numerics import compare_numeric, domain_bounds_for_solver, normalize_tags, parse_constraint_specs, signed_scalar_grid, unique_preserve_order, value_in_domain

REGISTERED_RELATIONS: dict[str, "Relation"] = {}
_ALLOWED_OPS = {"==", "<", "<=", ">", ">="}

# Discretisation coordinates are framework state, not physical unknowns.  A
# coordinate is never a solved variable: it is not registered, is classified as
# a relation *constant* regardless of its signature (so it drops out of the
# arity count and the structural graph), and is supplied by the framework at
# evaluation.  A standalone solve refuses to invert one, and forward standalone
# calls fall back to the default grid below when the framework does not supply
# one.
COORDINATE_NAMES = ("rho",)

# Lower-bound stand-in for a single-relation inverse solve whose lower bound is
# exactly zero.  The bounds are read with ``zero_tol=0.0`` on purpose -- an
# inclusive physical bound is not an open one and must not be nudged -- but
# handing a root-finder a lower bound of exactly zero for a variable appearing
# as 1/x or log(x) makes the bracket unusable, and the global packer already
# starts those variables at 1e-12.  Only an exact zero is replaced: a smaller
# declared floor (L_int at 1e-45, beam_stopping_cross_section at 1e-30) is
# deliberate and must survive.  This floors the *search box* alone; it never
# alters a declared domain.
INVERSE_BOUND_FLOOR = 1.0e-12


def _floored_lower_bound(lower: float | None) -> float:
    """Return an inverse-solve lower bound, replacing an exact zero."""
    if lower is None:
        return -np.inf
    return INVERSE_BOUND_FLOOR if float(lower) == 0.0 else float(lower)

# Standalone default grid for coordinate constants (the framework overrides it
# with its own ``profile_size`` grid via the evaluation namespace).  Matches
# ``VariableRegistry.uniform_profile_grid`` at the default ``profile_size`` of
# 46, so a forward standalone call with profile inputs of that length works.
_DEFAULT_COORDINATE_GRID = np.linspace(0.0, 1.0, 46)

# Shared variable registry, resolved lazily (importing it at module load would
# cycle through fusdb.registry) and cached for the hot tolerance/domain paths.
_VARIABLE_REGISTRY = None


def _variable_registry():
    global _VARIABLE_REGISTRY
    if _VARIABLE_REGISTRY is None:
        from .registry.variable_registry import VARIABLES

        _VARIABLE_REGISTRY = VARIABLES
    return _VARIABLE_REGISTRY


class RelationSolveError(ValueError):
    """Raised when a relation cannot be solved or verified."""


class RelationUnderdeterminedError(RelationSolveError):
    """Raised when too few variable values are supplied to a standalone relation."""


class RelationVerificationError(RelationSolveError):
    """Raised when a solved value does not verify against the canonical relation."""


class RelationNotInvertibleError(RelationSolveError):
    """Raised when a standalone inverse direction is not a well-posed request.

    Coordinates (the discretisation grid) and inequality relations refuse
    inversion: a coordinate is not an unknown, and an inequality determines a
    feasible interval rather than a value.
    """


@dataclass
class Relation:
    """One equation or inequality over FusDB variables.

    Args:
        name: User-facing relation name.
        func: Python implementation.
        input_names: Function input variable names.
        outputs: Declared output variable names.
        op: Comparison operator for outputless numeric residuals.
        rhs: Right side for outputless numeric residuals.
        tags: Descriptive/applicability tags.
        enforce: Whether this relation is solver-enforced.
        constraints: Relation-local constraints or applicability guards.
        source_kind: Diagnostic source category.
        source_name: Diagnostic source name.
        constant_names: Function parameters with defaults.
        dependency: Dependency hint used for graph reports.
        function_name: Decorated Python function name.
    """

    name: str
    func: Callable[..., Any]
    input_names: tuple[str, ...]
    outputs: tuple[str, ...] = ()
    op: str = "=="
    rhs: Any = 0.0
    tags: tuple[str, ...] = ()
    enforce: bool = True
    constraints: Any = None
    source_kind: str = "relation"
    source_name: str = ""
    constant_names: tuple[str, ...] = ()
    dependency: str = "dense"
    function_name: str = ""
    argument_names: tuple[str, ...] = ()
    constraint_relations: tuple["Relation", ...] = field(default_factory=tuple, init=False)
    _signature: inspect.Signature = field(init=False, repr=False)
    _constant_defaults: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    # Derived metadata cached once at construction.  ``input_names``/``outputs``
    # are normalized in ``__post_init__`` and never mutated afterwards (alias
    # canonicalization builds new Relation objects), so these stay valid for the
    # object's whole life.  Both are pure functions of those immutable fields, so
    # a Relation shared across RelationSystems (the registry singletons) yields
    # identical values everywhere -- caching them adds no cross-system state and
    # is safe under parallel runs.
    _variables: tuple[str, ...] = field(init=False, repr=False, compare=False)
    _implicit: bool = field(init=False, repr=False, compare=False)
    _arg_pairs: tuple[tuple[str, str], ...] = field(init=False, repr=False, compare=False)
    # Names coerced to solver shape before a relation evaluation (inputs plus
    # constants, disjoint by construction).  Cached so the per-evaluation
    # namespace build does not rebuild the set on every residual call.
    _coerce_names: tuple[str, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Normalize metadata and build local constraint relations."""
        self.name = str(self.name)
        if not self.name:
            raise ValueError("Relation name cannot be empty.")
        if self.op not in _ALLOWED_OPS:
            raise ValueError(f"Unsupported relation operator {self.op!r}.")
        self.input_names = tuple(str(name) for name in self.input_names)
        self.outputs = tuple(str(name) for name in self.outputs)
        self.tags = normalize_tags(self.tags)
        self.enforce = bool(self.enforce)
        self.source_name = str(self.source_name or self.name)
        self.source_kind = str(self.source_kind or "relation")
        self.constant_names = tuple(str(name) for name in self.constant_names)
        self.dependency = str(self.dependency or "dense")
        self.function_name = str(self.function_name or getattr(self.func, "__name__", self.name))
        self.argument_names = tuple(self.argument_names or self.input_names)
        if len(self.argument_names) != len(self.input_names):
            raise ValueError(f"Relation {self.name!r} argument_names and input_names must have the same length.")
        self._signature = inspect.signature(self.func)
        self._constant_defaults = {}
        for name in self.constant_names:
            parameter = self._signature.parameters.get(name)
            if parameter is not None and parameter.default is not inspect.Parameter.empty:
                self._constant_defaults[name] = parameter.default
        # A coordinate constant with no signature default still needs a value
        # for a standalone forward call (the framework supplies the real grid
        # via the namespace at solve time).
        for name in COORDINATE_NAMES:
            if name in self.constant_names and name not in self._constant_defaults:
                self._constant_defaults[name] = _DEFAULT_COORDINATE_GRID
        self._variables = unique_preserve_order((*self.input_names, *self.outputs))
        self._implicit = bool(set(self.outputs) & set(self.input_names))
        self._arg_pairs = tuple(zip(self.argument_names, self.input_names))
        self._coerce_names = (*self.input_names, *self.constant_names)

        # Local constraints are themselves relations. enforce=False means checked-only applicability.
        self.constraint_relations = build_constraint_relations(
            self.constraints,
            name_prefix=f"{self.name}_constraint",
            source_kind="relation",
            source_name=self.name,
        )

    @property
    def optional_variable_names(self) -> tuple[str, ...]:
        """Registry variables this relation reads as OPTIONAL contributors.

        A signature default on a parameter that names a registry variable does
        not mean "constant" -- it means *optional contributor*: :meth:`evaluate`
        uses the namespace value when the scenario provides one and falls back
        to the default otherwise, so ``def total(a=0.0, b=0.0)`` reads "whichever
        of these you tell me about; the rest are zero".

        The group is inherently RELATION-scoped, which is the point: the same
        variable may be an optional contributor here and a required input
        elsewhere, and nothing has to be declared on the variable itself.

        Coordinates are excluded -- they are framework constants with a grid
        default, and the coordinate providers (``rho_minor``, ``v_norm``,
        ``w_V``) are precisely the relations with no required inputs at all.
        """
        registry = _variable_registry()
        return tuple(
            name for name in self.constant_names
            if name in self._constant_defaults
            and name not in COORDINATE_NAMES
            and name in registry
        )

    @property
    def all_contributors_optional(self) -> bool:
        """Whether this relation has no required inputs but does have optional ones.

        Such a relation says nothing until at least one contributor exists, so
        the forward closure must not fire it on the vacuous truth of
        ``all(inp in known for inp in ())``.
        """
        return bool(self.optional_variable_names) and not self.input_names

    @property
    def output_names(self) -> tuple[str, ...]:
        """Declared output names."""
        return self.outputs

    @property
    def variables(self) -> tuple[str, ...]:
        """Variables touched by the relation (cached at construction)."""
        return self._variables

    @property
    def implicit(self) -> bool:
        """Whether an output also appears as an input (cached at construction)."""
        return self._implicit

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Use the relation with standalone acausal semantics.

        Let ``n`` be the number of relation variables, i.e. declared inputs plus
        declared outputs. Constants with defaults are not counted.

        * If all ``n`` variables are supplied, return ``True``/``False`` from
          canonical verification.
        * If exactly ``n - 1`` variables are supplied, compute the single
          missing variable and return its value.
        * If fewer than ``n - 1`` variables are supplied, raise a clear
          underdetermined error.

        Positional arguments map to the decorated function's declared inputs,
        followed by its optional constants.  This keeps the common forward call
        as simple as ``relation(a, b)``; inverse solves and explicit output
        verification remain unambiguous keyword calls.

        The returned value from an inverse solve is accepted only after the
        canonical relation and local relation constraints verify within the
        registry tolerances.
        """
        if not args:
            return self.solve(kwargs)
        positional_names = (*self.input_names, *self.constant_names)
        if len(args) > len(positional_names):
            raise TypeError(
                f"{self.function_name}() takes at most {len(positional_names)} positional "
                f"arguments but {len(args)} were given"
            )
        values = self._canonicalize_standalone_values(kwargs)
        for name, value in zip(positional_names, args):
            if name in values:
                raise TypeError(f"{self.function_name}() got multiple values for argument {name!r}")
            values[name] = value
        return self.solve(values)

    def evaluate(self, namespace: Mapping[str, Any]) -> Any:
        """Evaluate the implementation function in its declared direction.

        Args:
            namespace: Mapping of variable names to values.

        Returns:
            Raw function return value.
        """
        args = {arg: namespace[name] for arg, name in self._arg_pairs}
        for name in self.constant_names:
            if name in namespace and namespace[name] is not None:
                args[name] = namespace[name]
            elif name in self._constant_defaults:
                args[name] = self._constant_defaults[name]
        return self.func(**args)

    def output_map(self, result: Any) -> dict[str, Any]:
        """Map a raw implementation result to declared outputs.

        Args:
            result: Raw function result.

        Returns:
            Output name/value mapping.
        """
        if not self.outputs:
            return {}
        if isinstance(result, Mapping):
            missing = [name for name in self.outputs if name not in result]
            extras = [name for name in result if name not in self.outputs]
            if missing or extras:
                raise ValueError(f"Relation {self.name!r} returned mismatched keys; missing={missing}, extra={extras}.")
            return {name: result[name] for name in self.outputs}
        if len(self.outputs) == 1:
            return {self.outputs[0]: result}
        if not isinstance(result, (tuple, list)) or len(result) != len(self.outputs):
            raise ValueError(f"Relation {self.name!r} expected {len(self.outputs)} outputs.")
        return dict(zip(self.outputs, result))

    def comparisons(self, namespace: Mapping[str, Any]) -> list[tuple[Any, str, Any, str | None]]:
        """Return comparison tuples ``(lhs, op, rhs, output_name)``.

        Args:
            namespace: Variable values.

        Returns:
            Comparison tuples used by verification and residual compilation.
        """
        value = self.evaluate(namespace)
        if self.outputs:
            mapped = self.output_map(value)
            return [(namespace[name], "==", mapped[name], name) for name in self.outputs]
        if isinstance(value, tuple) and len(value) == 3 and str(value[0]) in _ALLOWED_OPS:
            return [(value[1], str(value[0]), value[2], None)]
        if isinstance(value, (bool, np.bool_)):
            return [(0.0 if bool(value) else 1.0, "==", 0.0, None)]
        return [(value, self.op, self.rhs, None)]

    def solve(self, values: Mapping[str, Any] | None = None) -> Any:
        """Evaluate, verify, or invert a standalone relation.

        Inverse directions follow the target's registry shape:

        * a scalar target is solved as a scalar, even when profiles are
          supplied (the residual may then be overdetermined -- consistent
          pointwise data still recovers the exact value);
        * a profile target is recovered pointwise when the relation provides
          a profile-sized residual; when the relation reduces to a scalar
          equation (an average or integral), the solved level is returned as
          a **flat profile** on the supplied grid -- the representative
          member of the solution family, not a unique shape;
        * a coordinate (``rho``) or an inequality relation refuses with
          :class:`RelationNotInvertibleError`.

        Args:
            values: Supplied relation variable values. Constants may also be
                supplied, but they are not counted as relation variables.

        Returns:
            ``bool`` when all relation variables are supplied, otherwise the
            single missing variable value.
        """
        ns = self._canonicalize_standalone_values(values or {})
        known = [name for name in self.variables if name in ns and ns[name] is not None]
        missing = [name for name in self.variables if name not in ns or ns[name] is None]

        if not missing:
            self._check_all_domains(ns, names=self.variables, use_solver_domain=False)
            self._check_local_constraints(ns)
            return bool(self.verify_status(ns)["verified"])

        if len(missing) > 1:
            raise RelationUnderdeterminedError(
                f"Relation {self.name!r} needs at least {len(self.variables) - 1} of "
                f"{len(self.variables)} variables. Supplied {len(known)}; missing {missing}."
            )

        target = missing[0]
        if target in COORDINATE_NAMES:
            raise RelationNotInvertibleError(
                f"Variable {target!r} in relation {self.name!r} is a coordinate grid, not an unknown; "
                "supply it instead of solving for it."
            )
        if self.op != "==":
            raise RelationNotInvertibleError(
                f"Relation {self.name!r} is an inequality; it constrains {target!r} to a feasible "
                "interval and cannot determine a single value."
            )
        self._check_all_domains(ns, names=known, use_solver_domain=False)

        # Fast canonical direction: all inputs are available and the single
        # missing variable is one declared output.
        if target in self.outputs and all(name in ns and ns[name] is not None for name in self.input_names):
            mapped = self.output_map(self.evaluate(ns))
            if target not in mapped:
                raise RelationSolveError(f"Relation {self.name!r} did not return output {target!r}.")
            ns[target] = mapped[target]
            self._verify_solved_namespace(ns, target)
            return ns[target]

        value, _info = self._solve_one_missing(target, ns)
        ns[target] = value
        self._verify_solved_namespace(ns, target)
        return value

    def _scaled_comparison(
        self,
        lhs: Any,
        op: str,
        rhs: Any,
        out: str | None,
        *,
        scales: Mapping[str, Any] | None,
        rel_tols: Mapping[str, float] | None,
        abs_tols: Mapping[str, float] | None,
    ) -> tuple[bool, np.ndarray, np.ndarray]:
        """Resolve scale and tolerances for one comparison and evaluate it.

        Shared by ``residual_vector`` and ``verify_status`` so the scale and
        tolerance derivation lives in one place; returns ``compare_numeric``'s
        ``(ok, residual, violation)``.  Scalar equality comparisons -- the
        bulk of a full-space residual vector -- take a plain-float path with
        arithmetic identical to :func:`compare_numeric`.
        """
        if out is not None and scales is not None:
            base_scale = scales.get(out, 1.0)
        else:
            magnitudes = [1.0]
            for value in (lhs, rhs):
                try:
                    arr = np.asarray(value, dtype=float).reshape(-1)
                except Exception:
                    continue
                finite = arr[np.isfinite(arr)]
                if finite.size:
                    magnitudes.append(float(np.max(np.abs(finite))))
            base_scale = max(magnitudes)
        if out is not None and rel_tols and out in rel_tols:
            tol = float(rel_tols[out])
        else:
            tol = self._variable_tolerance(out)[0]
        if out is not None and abs_tols and out in abs_tols:
            atol = float(abs_tols[out])
        else:
            atol = self._variable_tolerance(out)[1]
        if op == "==" and isinstance(lhs, float) and isinstance(rhs, float) and isinstance(base_scale, float):
            scl = max(abs(lhs), abs(rhs), base_scale, 1.0e-300)
            tol_width = max(atol, tol * scl, 1.0e-300)
            diff = lhs - rhs
            violation = abs(diff)
            return violation <= tol_width, np.asarray([diff / tol_width]), np.asarray([violation])
        scale = np.maximum(np.maximum(np.abs(np.asarray(lhs, dtype=float)), np.abs(np.asarray(rhs, dtype=float))), base_scale)
        return compare_numeric(lhs, op, rhs, scale=scale, rel_tol=tol, abs_tol=atol)

    def residual_vector(
        self,
        ns: Mapping[str, Any],
        *,
        scales: Mapping[str, Any] | None = None,
        rel_tols: Mapping[str, float] | None = None,
        abs_tols: Mapping[str, float] | None = None,
        safe: bool = False,
    ) -> np.ndarray:
        """Return a finite scaled residual vector.

        Args:
            ns: Variable namespace.
            scales: Optional variable scale mapping.
            rel_tols: Optional variable relative tolerance mapping.
            abs_tols: Optional variable absolute tolerance mapping.
            safe: Convert evaluation failures/non-finite values into large finite residuals.

        Returns:
            One-dimensional residual vector.
        """
        rows: list[np.ndarray] = []
        try:
            for lhs, op, rhs, out in self.comparisons(ns):
                _ok, residual, _violation = self._scaled_comparison(
                    lhs, op, rhs, out, scales=scales, rel_tols=rel_tols, abs_tols=abs_tols
                )
                rows.append(residual.reshape(-1))
            out = np.concatenate(rows) if rows else np.empty(0, dtype=float)
            if not np.all(np.isfinite(out)):
                raise FloatingPointError("non-finite residual")
            return out
        except Exception:
            if safe:
                return np.asarray([1.0e12], dtype=float)
            raise

    def verify_status(
        self,
        ns: Mapping[str, Any],
        *,
        scales: Mapping[str, Any] | None = None,
        rel_tols: Mapping[str, float] | None = None,
        abs_tols: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        """Verify one relation and its local constraints.

        The status half of :meth:`status_and_residual`.

        Args:
            ns: Variable namespace.
            scales: Optional variable scale mapping.
            rel_tols: Optional variable relative tolerance mapping.
            abs_tols: Optional variable absolute tolerance mapping.

        Returns:
            Diagnostic dictionary.
        """
        status, _residual = self.status_and_residual(ns, scales=scales, rel_tols=rel_tols, abs_tols=abs_tols)
        return status

    def status_and_residual(
        self,
        ns: Mapping[str, Any],
        *,
        scales: Mapping[str, Any] | None = None,
        rel_tols: Mapping[str, float] | None = None,
        abs_tols: Mapping[str, float] | None = None,
    ) -> tuple[dict[str, Any], np.ndarray]:
        """Verify the relation and build its residual vector from one evaluation.

        Final certification needs both the diagnostic status dict and the scaled
        residual vector for each enforced relation.  Computing them separately
        evaluates the implementation function (and its comparisons) twice; this
        method walks :meth:`comparisons` once and derives both, returning
        ``(status, residual_vector)``.

        The residual vector contains only the relation's own comparison
        residuals (local guards are verified for the status but never contribute
        residual rows, matching :meth:`residual_vector`).  Evaluation failures or
        non-finite residuals are folded into a large finite residual and a
        ``verified=False`` status rather than raised, so a single broken relation
        certifies cleanly as failed instead of aborting the whole certificate.
        """
        errors: list[str] = []
        warnings: list[str] = []
        residuals: list[float] = []
        rows: list[np.ndarray] = []
        max_violation = 0.0
        ok = True
        eval_failed = False
        try:
            for lhs, op, rhs, out in self.comparisons(ns):
                passed, residual, violation = self._scaled_comparison(
                    lhs, op, rhs, out, scales=scales, rel_tols=rel_tols, abs_tols=abs_tols
                )
                rows.append(residual.reshape(-1))
                residuals.extend(float(item) for item in residual)
                if violation.size:
                    max_violation = max(max_violation, float(np.max(violation)))
                ok = ok and passed
        except Exception as exc:
            ok = False
            eval_failed = True
            errors.append(str(exc))
        if eval_failed:
            residual_vector = np.asarray([1.0e12], dtype=float)
        else:
            residual_vector = np.concatenate(rows) if rows else np.empty(0, dtype=float)
            if not np.all(np.isfinite(residual_vector)):
                ok = False
                residual_vector = np.asarray([1.0e12], dtype=float)
        for guard in self.constraint_relations:
            try:
                status = guard.verify_status(ns, scales=scales, rel_tols=rel_tols, abs_tols=abs_tols)
                if not status["verified"]:
                    ok = False
                    message = f"{guard.name}: {status.get('errors') or 'constraint failed'}"
                    if guard.enforce:
                        errors.append(message)
                    else:
                        warnings.append(f"applicability failed: {message}")
            except Exception as exc:
                ok = False
                if guard.enforce:
                    errors.append(str(exc))
                else:
                    warnings.append(f"applicability failed: {exc}")
        status = {
            "relation": self.name,
            "verified": bool(ok),
            "enforced": bool(self.enforce),
            "errors": errors,
            "warnings": warnings,
            "residuals": residuals,
            "max_abs_scaled_residual": max((abs(item) for item in residuals), default=0.0),
            "max_physical_violation": max_violation,
        }
        return status, residual_vector

    def _canonicalize_standalone_values(self, values: Mapping[str, Any]) -> dict[str, Any]:
        allowed = set(self.variables) | set(self.constant_names)
        registry = _variable_registry()
        out: dict[str, Any] = {}
        unknown: list[str] = []
        for key, value in dict(values).items():
            text = str(key)
            if text in self.constant_names:
                out[text] = value
                continue
            resolved = registry.resolve(text) if text in registry else text
            if resolved not in allowed:
                unknown.append(text)
                continue
            out[resolved] = value
        if unknown:
            raise TypeError(f"Unknown keyword(s) for relation {self.function_name}: {sorted(unknown)}")
        return out

    def _variable_spec(self, name: str):
        registry = _variable_registry()
        if name not in registry:
            return None
        return registry.get(name)

    def _variable_tolerance(self, name: str | None) -> tuple[float, float]:
        if name is not None:
            spec = self._variable_spec(str(name))
            if spec is not None:
                return float(spec.rel_tol), float(spec.abs_tol)
        # Outputless relations (steady-state balances, inequality guards) have no
        # output variable to borrow a tolerance from.  Normalise them by the
        # registry's default relative tolerance rather than a machine-tight
        # 1e-8: at 1e-8 a small physical imbalance becomes a residual ~1e6x
        # larger than every output relation (which use ~0.001), which dominates
        # the least-squares cost and stalls the solve.
        return float(_variable_registry().rel_tol_default), 0.0

    def _check_all_domains(self, ns: Mapping[str, Any], *, names: Iterable[str], use_solver_domain: bool) -> None:
        for name in names:
            if name not in ns or ns[name] is None:
                continue
            spec = self._variable_spec(str(name))
            if spec is None:
                continue
            domain = spec.solver_domain if use_solver_domain else spec.domain
            if not value_in_domain(ns[name], domain, zero_tol=0.0):
                kind = "solver_domain" if use_solver_domain else "domain"
                raise RelationSolveError(f"Variable {name!r} in relation {self.name!r} violates {kind} {domain}.")

    def _bounds_for_target(self, target: str, template: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        spec = self._variable_spec(target)
        shape = np.asarray(template, dtype=float).shape
        if spec is None:
            return np.full(shape, -np.inf), np.full(shape, np.inf)
        lb, ub = domain_bounds_for_solver(spec.solver_domain, zero_tol=0.0)
        lower = _floored_lower_bound(lb)
        upper = np.inf if ub is None else float(ub)
        return np.full(shape, lower, dtype=float), np.full(shape, upper, dtype=float)

    def _verify_solved_namespace(self, ns: Mapping[str, Any], target: str) -> None:
        self._check_all_domains(ns, names=self.variables, use_solver_domain=False)
        self._check_local_constraints(ns)
        status = self.verify_status(ns)
        if not status["verified"]:
            raise RelationVerificationError(
                f"Relation {self.name!r} solved {target!r}, but canonical verification failed; "
                f"max residual={status.get('max_abs_scaled_residual')}."
            )

    def _check_local_constraints(self, ns: Mapping[str, Any]) -> None:
        for guard in self.constraint_relations:
            status = guard.verify_status(ns)
            if not status["verified"]:
                label = "constraint" if guard.enforce else "applicability guard"
                raise RelationSolveError(f"Relation {self.name!r} failed local {label} {guard.name!r}.")

    def _solve_one_missing(self, target: str, ns: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
        spec = self._variable_spec(target)
        if spec is not None and spec.shape:
            return self._solve_profile_target(target, ns)
        scalar = self._solve_one_missing_scalar_scan(target, ns)
        if scalar is not None:
            return scalar
        return self._solve_one_missing_least_squares(target, ns, template=self._scalar_template_for(ns))

    def _solve_profile_target(self, target: str, ns: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
        """Invert a profile-valued target.

        Pointwise inversion when the relation provides a profile-sized
        residual; otherwise (a reducing relation -- one scalar equation over
        many grid points) the solved level is returned as a flat profile on
        the supplied grid, the documented standalone semantics.
        """
        size = self._grid_size(ns)
        if size is not None:
            try:
                return self._solve_one_missing_least_squares(target, ns, template=np.ones(size))
            except RelationSolveError:
                pass  # reduction: profile-sized template is underdetermined
        scalar = self._solve_one_missing_scalar_scan(target, ns)
        if scalar is None:
            scalar = self._solve_one_missing_least_squares(target, ns, template=self._scalar_template_for(ns))
        value, info = scalar
        level = float(np.asarray(value, dtype=float).reshape(-1)[0])
        if size is None:
            return level, info
        return np.full(size, level), info

    @staticmethod
    def _grid_size(ns: Mapping[str, Any]) -> int | None:
        """Profile length implied by the supplied values (``rho`` preferred)."""
        rho = ns.get("rho")
        if rho is not None:
            arr = np.asarray(rho)
            if arr.ndim == 1 and arr.size > 1:
                return int(arr.size)
        for value in ns.values():
            arr = np.asarray(value)
            if arr.ndim >= 1 and arr.shape[-1] > 1:
                return int(arr.shape[-1])
        return None

    def _solve_one_missing_scalar_scan(self, target: str, ns: Mapping[str, Any]) -> tuple[Any, dict[str, Any]] | None:
        lower, upper = self._scalar_bounds_for_target(target)
        points = signed_scalar_grid(lower, upper, decades=240, step=6, dense=True)
        if not points:
            return None

        def residual_at(value: float) -> float:
            trial = dict(ns)
            trial[target] = float(value)
            residual = self.residual_vector(trial)
            if residual.size != 1:
                raise RelationSolveError("scalar bracketing requires exactly one residual")
            return float(residual[0])

        evaluated: list[tuple[float, float]] = []
        for point in points:
            try:
                residual = residual_at(point)
            except Exception:
                continue
            if not np.isfinite(residual):
                continue
            evaluated.append((float(point), float(residual)))
            if abs(residual) <= 1e-6:
                return float(point), {"method": "grid_exact", "residual": np.asarray([residual]), "success": True}
        evaluated.sort(key=lambda item: item[0])
        for (left, r_left), (right, r_right) in zip(evaluated[:-1], evaluated[1:]):
            if np.sign(r_left) == np.sign(r_right) or left == right:
                continue
            try:
                sol = root_scalar(residual_at, bracket=(left, right), method="brentq", xtol=1e-12, rtol=1e-12)
            except Exception:
                continue
            if sol.converged:
                root = float(sol.root)
                final = residual_at(root)
                if abs(final) <= 1e-6:
                    return root, {"method": "brentq", "residual": np.asarray([final]), "success": True}
        return None

    def _solve_one_missing_least_squares(
        self, target: str, ns: Mapping[str, Any], *, template: np.ndarray
    ) -> tuple[Any, dict[str, Any]]:
        flat0 = np.asarray(template, dtype=float).reshape(-1)
        if flat0.size == 0 or not np.all(np.isfinite(flat0)):
            raise RelationSolveError(f"No finite initial guess is available for {target!r} in {self.name!r}.")
        lb_template, ub_template = self._bounds_for_target(target, np.asarray(template, dtype=float))
        lb = np.broadcast_to(lb_template.reshape(flat0.shape), flat0.shape).copy()
        ub = np.broadcast_to(ub_template.reshape(flat0.shape), flat0.shape).copy()
        if np.any(np.isfinite(lb) & (flat0 < lb)) or np.any(np.isfinite(ub) & (flat0 > ub)):
            raise RelationSolveError(
                f"Initial guess for {target!r} in relation {self.name!r} is outside solver bounds."
            )
        offset = flat0.copy()
        scale = np.maximum(np.abs(offset), 1.0)
        lower = np.where(np.isfinite(lb), (lb - offset) / scale, -np.inf)
        upper = np.where(np.isfinite(ub), (ub - offset) / scale, np.inf)
        shape = np.asarray(template).shape

        def namespace_from_x(x: np.ndarray) -> dict[str, Any]:
            out = dict(ns)
            actual = offset + scale * np.asarray(x, dtype=float)
            out[target] = float(actual[0]) if shape == () else actual.reshape(shape)
            return out

        def residual(x: np.ndarray) -> np.ndarray:
            return self.residual_vector(namespace_from_x(x), safe=True)

        probe = residual(np.zeros_like(offset))
        if probe.size < offset.size:
            raise RelationSolveError(f"Relation {self.name!r} is underdetermined for {target!r}.")
        sol = least_squares(residual, np.zeros_like(offset), bounds=(lower, upper), method="trf", x_scale=np.ones_like(offset), max_nfev=200)
        value = namespace_from_x(sol.x)[target]
        final = residual(sol.x)
        if final.size and float(np.max(np.abs(final))) > 1e-6:
            raise RelationSolveError(f"Inverse solve for {target!r} in relation {self.name!r} did not verify.")
        return value, {"method": "least_squares", "success": bool(sol.success), "residual": final, "nfev": int(sol.nfev)}

    def _scalar_bounds_for_target(self, target: str) -> tuple[float, float]:
        spec = self._variable_spec(target)
        if spec is None:
            return -np.inf, np.inf
        lower, upper = domain_bounds_for_solver(spec.solver_domain, zero_tol=0.0)
        return _floored_lower_bound(lower), (np.inf if upper is None else float(upper))

    def _scalar_template_for(self, ns: Mapping[str, Any]) -> np.ndarray:
        """Scalar initial guess: the geometric mean of the supplied magnitudes.

        The inverse template follows the *target's* registry shape, never the
        shape of whichever supplied value happens to come first -- a scalar
        target solved against profile-sized residuals is simply
        overdetermined, and consistent data recovers it exactly.
        """
        positive = []
        for value in ns.values():
            arr = np.asarray(value, dtype=float).reshape(-1)
            positive.extend(float(v) for v in arr if np.isfinite(v) and v > 0)
        if positive:
            return np.asarray(np.exp(np.mean(np.log(np.asarray(positive, dtype=float)))))
        return np.asarray(1.0)

    @classmethod
    def from_function(
        cls,
        func: Callable[..., Any],
        *,
        outputs: Any = None,
        name: str | None = None,
        tags: Iterable[str] | None = None,
        enforce: bool = True,
        constraints: Any = None,
        dependency: str = "dense",
        h_factor: str | None = None,
    ) -> "Relation":
        """Build a relation from a decorated Python function.

        Args:
            func: Python implementation function.
            outputs: Explicit output name or names.
            name: Optional user-facing relation name.
            tags: Relation tags.
            enforce: Whether relation is solver-enforced.
            constraints: Local constraints/applicability guards.
            dependency: Dependency hint.

        Returns:
            Relation object.
        """
        if h_factor is not None:
            func = _with_h_factor(func, h_factor)
        inputs: list[str] = []
        constants: list[str] = []
        for parameter in inspect.signature(func).parameters.values():
            if parameter.kind in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.VAR_POSITIONAL}:
                raise ValueError(f"Relation {func.__name__!r} cannot use positional-only parameters or *args.")
            if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                continue
            # Coordinates are always framework-supplied constants, never solved
            # inputs, even without a signature default (S3).
            if parameter.default is inspect.Parameter.empty and parameter.name not in COORDINATE_NAMES:
                inputs.append(parameter.name)
            else:
                constants.append(parameter.name)
        if outputs is None:
            output_names: tuple[str, ...] = ()
        elif isinstance(outputs, str):
            output_names = (outputs,)
        else:
            output_names = tuple(str(item) for item in outputs)
        return cls(
            name=str(name or func.__name__),
            func=func,
            input_names=tuple(inputs),
            outputs=output_names,
            tags=tuple(tags or ()),
            enforce=enforce,
            constraints=constraints,
            source_kind="relation",
            source_name=str(name or func.__name__),
            constant_names=tuple(constants),
            dependency=dependency,
            function_name=func.__name__,
            argument_names=tuple(inputs),
        )


def is_default_relation(rel: "Relation") -> bool:
    """Whether a relation is a weak default (fallback provider / x0 seed)."""
    return "default" in set(rel.tags) or str(rel.source_kind).startswith("default")


def _with_h_factor(func: Callable[..., Any], h_name: str) -> Callable[..., Any]:
    """Wrap a confinement scaling so its result is H-enhanced.

    Energy-confinement scalings are published as a *raw* fit; the confinement
    time a device actually achieves is that fit times an enhancement factor H.
    Rather than write the multiplication into 50 relation bodies, the scaling
    declares ``h_factor="H98_y2"`` and this wrapper injects two optional
    constants and multiplies them in:

    * ``h_name`` -- the H defined against *this particular* scaling
      (``H98_y2`` for IPB98(y,2), ``H_iter_89p`` for ITER89-P, ...), which is
      what published design points quote; and
    * ``H_factor`` -- a generic multiplier that applies whichever scaling is
      active, matching PROCESS's ``hfact`` and cfspopcon's
      ``confinement_time_scalar``.

    Both default to 1.0, so they compose multiplicatively and each is a no-op
    when absent.  Supplying only the generic H enhances any scaling; supplying
    only the scaling-specific H enhances it exactly where it belongs, and is
    simply unused if a different scaling wins the provider slot.  Supplying
    both applies the product.

    Making the conditionality structural like this -- rather than coupling the
    two with an ``H_factor = H98_y2`` relation -- is deliberate: fusdb activates
    relations on variable availability, not on which sibling relation won a
    provider slot, so such a coupling would force a published ``H98_y2`` onto
    whatever scaling happened to be active.
    """
    signature = inspect.signature(func)
    extra = [
        inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=1.0, annotation=float)
        for name in (h_name, "H_factor")
        if name not in signature.parameters
    ]

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        scale = 1.0
        for parameter in extra:
            scale = scale * kwargs.pop(parameter.name, 1.0)
        return scale * func(*args, **kwargs)

    wrapper.__signature__ = signature.replace(parameters=[*signature.parameters.values(), *extra])
    return wrapper


def relation(
    _func: Callable[..., Any] | None = None,
    *,
    outputs: Any | None = None,
    name: str | None = None,
    tags: Iterable[str] | None = None,
    enforce: bool = True,
    constraints: Any = None,
    dependency: str = "dense",
    h_factor: str | None = None,
) -> Callable[[Callable[..., Any]], Relation] | Relation:
    """Decorate a function as a FusDB relation.

    Args:
        _func: Function when used as ``@relation`` without parentheses.
        outputs: Explicit output name or names.
        name: User-facing relation name.
        tags: Relation tags.
        enforce: Whether the relation is enforced.
        constraints: Local constraints or applicability guards.
        dependency: Dependency hint.

    Returns:
        Relation object or decorator.
    """

    def decorator(func: Callable[..., Any]) -> Relation:
        built = Relation.from_function(func, outputs=outputs, name=name, tags=tags, enforce=enforce, constraints=constraints, dependency=dependency, h_factor=h_factor)
        if built.name in REGISTERED_RELATIONS:
            raise ValueError(f"Duplicate relation {built.name!r}.")
        REGISTERED_RELATIONS[built.name] = built
        return built

    if _func is not None:
        return decorator(_func)
    return decorator


def canonicalize_relation_names(rel: "Relation", variable_registry: Any) -> "Relation":
    """Return ``rel`` with input/output variable names resolved to canonical names.

    Pure canonicalization: aliases are mapped through ``variable_registry`` and a
    new :class:`Relation` is returned only when a name actually changed.  No
    validation is performed -- see :func:`canonicalize_relation` for the variant
    that also rejects alias-degenerate relations.
    """
    inputs = tuple(variable_registry.get(name).canonical_name for name in rel.input_names)
    outputs = tuple(variable_registry.get(name).canonical_name for name in rel.outputs)
    if inputs == rel.input_names and outputs == rel.outputs:
        return rel
    return replace(rel, input_names=inputs, outputs=outputs)


def canonicalize_relation(rel: "Relation", variable_registry: Any) -> "Relation":
    """Canonicalize ``rel`` and reject alias-degenerate relations.

    A relation whose declared outputs collapse onto one of its own inputs after
    alias resolution (for example ``n_e_avg = n_avg`` when ``n_e_avg`` is an
    alias of ``n_avg``) is a tautology for this registry: it determines nothing,
    and acausal seeding would otherwise "solve" the identity to an arbitrary
    value.  Because this depends only on ``(rel, variable_registry)`` and not on
    any scenario, it is an authoring error and is raised here -- at registry /
    system build time -- rather than being silently dropped per system.
    """
    resolved = canonicalize_relation_names(rel, variable_registry)
    if resolved.implicit and not rel.implicit:
        raise ValueError(
            f"Relation {rel.name!r} is alias-degenerate: declared outputs ("
            + ", ".join(sorted(rel.outputs))
            + ") resolve to the same canonical variable as an input."
        )
    return resolved


_COMPARE_OPS = {ast.Eq: "==", ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">="}
_BINARY_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _compile_expression(node: ast.AST, names: list[str]) -> Callable[[Mapping[str, Any]], Any]:
    if isinstance(node, ast.Name):
        names.append(node.id)
        return lambda ns, name=node.id: ns[name]
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, Real):
            raise ValueError("Only real numeric constants are allowed in constraints.")
        value = float(node.value)
        return lambda ns, value=value: value
    if isinstance(node, ast.UnaryOp):
        if type(node.op) not in _UNARY_OPS:
            raise ValueError("Only unary + and - are supported in constraints.")
        operand = _compile_expression(node.operand, names)
        op = _UNARY_OPS[type(node.op)]
        return lambda ns, operand=operand, op=op: op(operand(ns))
    if isinstance(node, ast.BinOp):
        if type(node.op) not in _BINARY_OPS:
            raise ValueError("Only +, -, *, /, and ** are supported in constraints.")
        left = _compile_expression(node.left, names)
        right = _compile_expression(node.right, names)
        op = _BINARY_OPS[type(node.op)]
        return lambda ns, left=left, right=right, op=op: op(left(ns), right(ns))
    raise ValueError(f"Unsupported constraint expression element {type(node).__name__}.")


def constraint_from_expression(
    text: str,
    *,
    name: str | None = None,
    enforce: bool = True,
    tags: Iterable[str] | None = None,
    source_kind: str = "constraint",
    source_name: str = "",
) -> Relation:
    """Parse a simple comparison into an outputless relation.

    Args:
        text: Constraint expression such as ``x <= y``.
        name: Optional relation name.
        enforce: Whether the relation is solver-enforced.
        tags: Optional tags.
        source_kind: Diagnostic source kind.
        source_name: Diagnostic source name.

    Returns:
        Relation object.
    """
    tree = ast.parse(str(text), mode="eval")
    body = tree.body
    if not isinstance(body, ast.Compare) or len(body.ops) != 1 or len(body.comparators) != 1:
        raise ValueError(f"Constraint {text!r} must be a single comparison.")
    op_type = type(body.ops[0])
    if op_type not in _COMPARE_OPS:
        raise ValueError(f"Unsupported comparison in {text!r}.")
    names: list[str] = []
    left = _compile_expression(body.left, names)
    right = _compile_expression(body.comparators[0], names)
    inputs = unique_preserve_order(names)

    def func(**kwargs: Any) -> Any:
        return (_COMPARE_OPS[op_type], left(kwargs), right(kwargs))

    safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(text)).strip("_")
    return Relation(
        name=str(name or f"constraint_{safe}"),
        func=func,
        input_names=inputs,
        outputs=(),
        op=_COMPARE_OPS[op_type],
        rhs=0.0,
        tags=tuple(tags or ()),
        enforce=enforce,
        source_kind=source_kind,
        source_name=str(source_name or name or text),
        function_name=str(name or f"constraint_{safe}"),
        argument_names=inputs,
    )


def build_constraint_relations(
    constraints: Any,
    *,
    name_prefix: str,
    source_kind: str,
    source_name: str,
) -> tuple[Relation, ...]:
    """Normalize a constraint declaration into ordinary Relation objects."""
    return tuple(
        constraint_from_expression(
            text,
            name=f"{name_prefix}_{index}",
            enforce=enforce,
            source_kind=source_kind,
            source_name=source_name,
        )
        for index, (text, enforce) in enumerate(parse_constraint_specs(constraints))
    )
