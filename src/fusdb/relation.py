"""Acausal relation object and ``@relation`` decorator."""

from __future__ import annotations

import ast
import inspect
import operator
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Callable

import numpy as np
from scipy.optimize import least_squares, root_scalar

from .utils import compare_numeric, domain_bounds_for_solver, normalize_tags, parse_constraint_specs, safe_max_abs, unique_preserve_order, value_in_domain

REGISTERED_RELATIONS: dict[str, "Relation"] = {}
REGISTERED_RELATIONS_BY_FUNCTION: dict[str, "Relation"] = {}
_ALLOWED_OPS = {"==", "<", "<=", ">", ">="}


class RelationSolveError(ValueError):
    """Raised when a relation cannot be solved or verified."""


class RelationUnderdeterminedError(RelationSolveError):
    """Raised when too few variable values are supplied to a standalone relation."""


class RelationVerificationError(RelationSolveError):
    """Raised when a solved value does not verify against the canonical relation."""


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

        # Local constraints are themselves relations. enforce=False means checked-only applicability.
        built: list[Relation] = []
        for index, (text, enforce) in enumerate(parse_constraint_specs(self.constraints)):
            built.append(
                constraint_from_expression(
                    text,
                    name=f"{self.name}_constraint_{index}",
                    enforce=enforce,
                    source_kind="relation",
                    source_name=self.name,
                )
            )
        self.constraint_relations = tuple(built)

    @property
    def output_names(self) -> tuple[str, ...]:
        """Declared output names."""
        return self.outputs

    @property
    def variables(self) -> tuple[str, ...]:
        """Variables touched by the relation."""
        return unique_preserve_order((*self.input_names, *self.outputs))

    @property
    def implicit(self) -> bool:
        """Whether an output also appears as an input."""
        return bool(set(self.outputs) & set(self.input_names))

    def __call__(self, **kwargs: Any) -> Any:
        """Use the relation with strict standalone acausal semantics.

        Let ``n`` be the number of relation variables, i.e. declared inputs plus
        declared outputs. Constants with defaults are not counted.

        * If all ``n`` variables are supplied, return ``True``/``False`` from
          canonical verification.
        * If exactly ``n - 1`` variables are supplied, compute the single
          missing variable and return its value.
        * If fewer than ``n - 1`` variables are supplied, raise a clear
          underdetermined error.

        The returned value from an inverse solve is accepted only after the
        canonical relation and local relation constraints verify within the
        registry tolerances.
        """
        return self.solve(kwargs)

    def evaluate(self, namespace: Mapping[str, Any]) -> Any:
        """Evaluate the implementation function in its declared direction.

        Args:
            namespace: Mapping of variable names to values.

        Returns:
            Raw function return value.
        """
        args = {arg: namespace[name] for arg, name in zip(self.argument_names, self.input_names)}
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
            comparisons = self.comparisons(ns)
            for lhs, op, rhs, out in comparisons:
                base_scale = 1.0
                if out is not None and scales is not None:
                    base_scale = scales.get(out, 1.0)
                else:
                    base_scale = max(safe_max_abs(lhs), safe_max_abs(rhs), 1.0)
                scale = np.maximum(np.maximum(np.abs(np.asarray(lhs, dtype=float)), np.abs(np.asarray(rhs, dtype=float))), base_scale)
                if out is not None and rel_tols and out in rel_tols:
                    tol = float(rel_tols[out])
                else:
                    tol = self._variable_tolerance(out)[0]
                if out is not None and abs_tols and out in abs_tols:
                    atol = float(abs_tols[out])
                else:
                    atol = self._variable_tolerance(out)[1]
                _ok, residual, _violation = compare_numeric(lhs, op, rhs, scale=scale, rel_tol=tol, abs_tol=atol)
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

        Args:
            ns: Variable namespace.
            scales: Optional variable scale mapping.
            rel_tols: Optional variable relative tolerance mapping.
            abs_tols: Optional variable absolute tolerance mapping.

        Returns:
            Diagnostic dictionary.
        """
        errors: list[str] = []
        warnings: list[str] = []
        residuals: list[float] = []
        max_violation = 0.0
        ok = True
        try:
            for lhs, op, rhs, out in self.comparisons(ns):
                if out is not None and rel_tols and out in rel_tols:
                    tol = float(rel_tols[out])
                else:
                    tol = self._variable_tolerance(out)[0]
                base_scale = scales.get(out, 1.0) if out is not None and scales else max(safe_max_abs(lhs), safe_max_abs(rhs), 1.0)
                scale = np.maximum(np.maximum(np.abs(np.asarray(lhs, dtype=float)), np.abs(np.asarray(rhs, dtype=float))), base_scale)
                if out is not None and abs_tols and out in abs_tols:
                    atol = float(abs_tols[out])
                else:
                    atol = self._variable_tolerance(out)[1]
                passed, residual, violation = compare_numeric(lhs, op, rhs, scale=scale, rel_tol=tol, abs_tol=atol)
                residuals.extend(float(item) for item in residual)
                if violation.size:
                    max_violation = max(max_violation, float(np.max(violation)))
                ok = ok and passed
        except Exception as exc:
            ok = False
            errors.append(str(exc))
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
        return {
            "relation": self.name,
            "verified": bool(ok),
            "enforced": bool(self.enforce),
            "errors": errors,
            "warnings": warnings,
            "residuals": residuals,
            "max_abs_scaled_residual": max((abs(item) for item in residuals), default=0.0),
            "max_physical_violation": max_violation,
        }

    def _default_registry(self):
        """Return the shared variable registry without making it relation-owned."""
        try:
            from .registry.variable_registry import VARIABLES
        except Exception:
            return None
        return VARIABLES

    def _canonicalize_standalone_values(self, values: Mapping[str, Any]) -> dict[str, Any]:
        allowed = set(self.variables) | set(self.constant_names)
        registry = self._default_registry()
        out: dict[str, Any] = {}
        unknown: list[str] = []
        for key, value in dict(values).items():
            text = str(key)
            if text in self.constant_names:
                out[text] = value
                continue
            resolved = text
            if registry is not None and text in registry:
                resolved = registry.resolve(text)
            if resolved not in allowed:
                unknown.append(text)
                continue
            out[resolved] = value
        if unknown:
            raise TypeError(f"Unknown keyword(s) for relation {self.function_name}: {sorted(unknown)}")
        return out

    def _variable_spec(self, name: str):
        registry = self._default_registry()
        if registry is None or name not in registry:
            return None
        return registry.get(name)

    def _variable_tolerance(self, name: str | None) -> tuple[float, float]:
        if name is None:
            return 1.0e-8, 0.0
        spec = self._variable_spec(str(name))
        if spec is None:
            return 1.0e-8, 0.0
        return float(spec.rel_tol), float(spec.abs_tol)

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
        lower = -np.inf if lb is None else float(lb)
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
        scalar = self._solve_one_missing_scalar_scan(target, ns)
        if scalar is not None:
            return scalar
        return self._solve_one_missing_least_squares(target, ns)

    def _solve_one_missing_scalar_scan(self, target: str, ns: Mapping[str, Any]) -> tuple[Any, dict[str, Any]] | None:
        lower, upper = self._scalar_bounds_for_target(target)
        points = self._signed_scalar_grid(lower, upper)
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

    def _solve_one_missing_least_squares(self, target: str, ns: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
        template = self._initial_template_for(ns)
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
        return (-np.inf if lower is None else float(lower)), (np.inf if upper is None else float(upper))

    def _signed_scalar_grid(self, lower: float, upper: float, *, decades: int = 240, step: int = 6) -> list[float]:
        points: list[float] = []

        def add(value: float) -> None:
            if np.isfinite(value) and lower <= value <= upper:
                points.append(float(value))
        if lower <= 0.0 <= upper:
            add(0.0)
        if np.isfinite(lower):
            add(lower)
        if np.isfinite(upper):
            add(upper)
        if np.isfinite(lower) and np.isfinite(upper) and upper > lower:
            for value in np.linspace(lower, upper, 21):
                add(float(value))
        for exponent in range(-decades, decades + 1, step):
            magnitude = float(10.0 ** exponent)
            add(magnitude)
            add(-magnitude)
        return sorted(set(points))

    def _initial_template_for(self, ns: Mapping[str, Any]) -> np.ndarray:
        for value in ns.values():
            arr = np.asarray(value)
            if arr.ndim > 0 and arr.size > 1:
                return np.ones_like(arr, dtype=float)
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
        inputs: list[str] = []
        constants: list[str] = []
        for parameter in inspect.signature(func).parameters.values():
            if parameter.kind in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.VAR_POSITIONAL}:
                raise ValueError(f"Relation {func.__name__!r} cannot use positional-only parameters or *args.")
            if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                continue
            if parameter.default is inspect.Parameter.empty:
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


def relation(
    _func: Callable[..., Any] | None = None,
    *,
    outputs: Any | None = None,
    name: str | None = None,
    tags: Iterable[str] | None = None,
    enforce: bool = True,
    constraints: Any = None,
    dependency: str = "dense",
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
        built = Relation.from_function(func, outputs=outputs, name=name, tags=tags, enforce=enforce, constraints=constraints, dependency=dependency)
        if built.name in REGISTERED_RELATIONS:
            raise ValueError(f"Duplicate relation {built.name!r}.")
        REGISTERED_RELATIONS[built.name] = built
        REGISTERED_RELATIONS_BY_FUNCTION[built.function_name] = built
        return built

    if _func is not None:
        return decorator(_func)
    return decorator


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
