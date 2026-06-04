"""RelationSystem: simultaneous numeric residual solving with SciPy."""

from __future__ import annotations

import inspect
from collections.abc import Iterable
from typing import Any, Callable

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .relation import Relation, constraint_from_expression
from .registry import VARIABLES, VariableRegistry
from .utils import (
    compare_numeric,
    domain_bounds_for_solver,
    finite_array,
    parse_constraint_specs,
    scipy_bounds,
)
from .variable import Variable

DEFAULT_PROFILE_SIZE = 46
LEAST_SQUARES_SUPPORTS_WORKERS = "workers" in inspect.signature(least_squares).parameters


class RelationSystem:
    """Selected variables and relations ready for numeric execution.

    RelationSystem treats decorated relations as the source of truth.  Verify mode
    evaluates all residuals at the current state.  Reconcile and optimize modes
    solve all active residuals simultaneously with SciPy; ordered mode remains a
    forward-evaluation bridge for cfspopcon-like workflows.
    """

    def __init__(
        self,
        variables: Iterable[Variable],
        relations: Iterable[Relation],
        *,
        constraints: Any = None,
        name: str | None = None,
        verbose: bool = False,
        variable_registry: VariableRegistry = VARIABLES,
        targets: Iterable[str] | None = None,
        solve_for: Iterable[str] | None = None,
    ) -> None:
        """Build a numeric relation system from the reachable relation subgraph.

        The constructor accepts a broad relation list, but it does not activate all
        of it.  It first grows a structural closure from variables that already
        have values, plus any explicit ``solve_for`` variables.  Explicit output
        relations are activated when their inputs are reachable; their outputs
        then become reachable too.  Unconnected variables and relations remain
        undecided and are not packed into the SciPy vector.
        """
        self.name = str(name or "relation_system")
        self.verbose = bool(verbose)
        self.variable_registry = variable_registry
        self.zero_tol = 1e-8
        self.targets = self._resolve_names(targets or ())
        self.solve_for = self._resolve_names(solve_for or ())

        self.variables = list(variables)
        self.variables_by_name: dict[str, Variable] = {}
        for var in self.variables:
            if var.name in self.variables_by_name:
                raise ValueError(f"Duplicate variable {var.name!r}.")
            self.variables_by_name[var.name] = var

        self.candidate_primary_relations = [self._resolve_relation_names(rel) for rel in relations]
        self.system_constraint_relations = [
            self._resolve_relation_names(
                constraint_from_expression(
                    text,
                    name=f"system_constraint_{index}",
                    enforce=enforce,
                    source_kind="system",
                    source_name=self.name,
                )
            )
            for index, (text, enforce) in enumerate(parse_constraint_specs(constraints))
        ]

        self.profile_size = self._infer_profile_size()
        self._add_or_repair_rho_if_needed()
        self._add_uniform_profile_relation_candidates()
        self._broadcast_profile_values()
        self._plan_active_relations()
        self._broadcast_profile_values()
        self._seed_missing_values_from_relations()

        self._last_vector_spans: list[tuple[str, int, int, np.ndarray, np.ndarray]] = []
        self._last_solver_values: dict[str, Any] = {}

    def verify(self, **options: Any) -> dict[str, Any]:
        """Run verify mode."""
        from .modes.verify import run

        return run(self, **options)

    def reconcile(self, **options: Any) -> dict[str, Any]:
        """Run reconcile mode."""
        from .modes.reconcile import run

        return run(self, **options)

    def optimize(self, **options: Any) -> dict[str, Any]:
        """Run optimize mode."""
        from .modes.optimize import run

        return run(self, **options)

    def ordered(self, **options: Any) -> dict[str, Any]:
        """Run ordered evaluation mode."""
        from .modes.ordered import run

        return run(self, **options)

    def run(self, mode: str = "verify", **options: Any) -> dict[str, Any]:
        """Dispatch a mode by name.

        Args:
            mode: One of ``verify``, ``reconcile``, ``optimize``, or ``ordered``.
            **options: Mode-specific options.

        Returns:
            Result dictionary.
        """
        if mode == "verify":
            return self.verify(**options)
        if mode == "reconcile":
            return self.reconcile(**options)
        if mode == "optimize":
            return self.optimize(**options)
        if mode == "ordered":
            return self.ordered(**options)
        raise ValueError(f"Unknown mode {mode!r}.")

    def verify_current(self) -> dict[str, Any]:
        """Evaluate all active relations without changing variables.

        Returns:
            Result dictionary with relation and variable statuses.
        """
        result = self._new_result("verify")
        values = self._values_from_variables(for_solver=True)
        relation_status, residuals, errors, warnings = self._evaluate_relation_residuals(
            values,
            include_warning=True,
            relation_weight=1.0,
            strict=False,
        )
        result["relation_status"] = relation_status
        result["residuals"] = residuals.tolist()
        result["errors"].extend(errors)
        result["warnings"].extend(warnings)
        result["variable_status"] = self._classify_variables(relation_status)
        result["termination"] = "verification evaluated"
        result["success"] = self._relations_verified(relation_status)
        result["variables"] = self.variables_by_name
        result["relations"] = self.primary_relations
        return result

    def solve_mode(
        self,
        mode: str,
        *,
        objective: str | Callable[[dict[str, Any]], Any] | None = None,
        sense: str = "minimize",
        max_nfev: int | None = None,
        movement_weight: float = 0.1,
        relation_weight: float = 10.0,
        objective_weight: float = 1.0,
        workers: Any = None,
        verbose: int = 0,
    ) -> dict[str, Any]:
        """Solve reconcile or optimize mode with SciPy least_squares.

        Args:
            mode: ``reconcile`` or ``optimize``.
            objective: Variable name or callable objective for optimize mode.
            sense: ``minimize`` or ``maximize`` for the objective residual.
            max_nfev: Optional maximum function evaluations.
            movement_weight: Weight for movement-from-reference residuals.
            relation_weight: Weight for enforced relation residuals during solve.
            objective_weight: Weight for optimize objective residual.
            workers: Optional map-like callable for parallel finite differences.
                Ignored when the installed SciPy does not support it.
            verbose: SciPy verbosity level.

        Returns:
            Result dictionary.
        """
        if mode not in {"reconcile", "optimize"}:
            raise ValueError("solve_mode only supports 'reconcile' and 'optimize'.")
        result = self._new_result(mode)

        try:
            x0, lower, upper, x_scale, spans = self._pack_free_variables()
        except Exception as exc:
            result["errors"].append(str(exc))
            result["termination"] = "initialization failed"
            return result
        self._last_vector_spans = spans

        if x0.size == 0:
            validation = self.verify_current()
            validation["mode"] = mode
            validation["termination"] = "no free variables; validation only"
            return validation

        def residual_function(x: np.ndarray) -> np.ndarray:
            """Return the full residual vector for one SciPy candidate state."""
            values = self._values_from_vector(x, spans)
            relation_status, relation_residuals, errors, warnings = self._evaluate_relation_residuals(
                values,
                include_warning=False,
                relation_weight=relation_weight,
                strict=True,
                solver_residuals=True,
            )
            if errors:
                raise ValueError("; ".join(errors))
            if warnings and self.verbose:
                print("; ".join(warnings))

            blocks = [relation_residuals]
            movement = self._movement_residuals(values, spans, weight=movement_weight)
            if movement.size:
                blocks.append(movement)
            if mode == "optimize" and objective is not None:
                blocks.append(
                    self._objective_residual(
                        values,
                        objective=objective,
                        sense=sense,
                        weight=objective_weight,
                    )
                )
            return np.concatenate([block.ravel() for block in blocks if block.size])

        try:
            # Build the sparsity pattern once so SciPy can exploit structure.
            jac_sparsity = self._build_jac_sparsity(
                x0,
                spans,
                include_movement=True,
                include_objective=mode == "optimize" and objective is not None,
            )

            # Validate the sparsity shape against the actual residual length at x0.
            if jac_sparsity is not None:
                residual_size = int(np.asarray(residual_function(x0), dtype=float).size)
                expected_shape = (residual_size, int(x0.size))
                if tuple(jac_sparsity.shape) != expected_shape:
                    result["warnings"].append(
                        "Ignoring jac_sparsity because its shape does not match the current "
                        f"residual layout: got {tuple(jac_sparsity.shape)}, expected {expected_shape}."
                    )
                    jac_sparsity = None

            # Assemble solver kwargs and only pass workers on SciPy builds that expose it.
            solve_kwargs: dict[str, Any] = {
                "bounds": (lower, upper),
                "method": "trf",
                "x_scale": x_scale,
                "jac_sparsity": jac_sparsity,
                "max_nfev": max_nfev,
                "verbose": int(verbose),
            }
            if workers is not None and LEAST_SQUARES_SUPPORTS_WORKERS:
                solve_kwargs["workers"] = workers
            elif workers is not None:
                result["warnings"].append(
                    "Installed SciPy does not support least_squares(workers=...); "
                    "running solve without parallel finite differences."
                )

            # Run the nonlinear least-squares solve with the compatible kwargs.
            solve_result = least_squares(
                residual_function,
                x0,
                **solve_kwargs,
            )
        except Exception as exc:
            result["errors"].append(f"SciPy solve failed: {exc}")
            result["termination"] = "solver error"
            return result

        # Pull numerical values back into variables before final validation.
        solved_values = self._values_from_vector(solve_result.x, spans)
        self._last_solver_values = solved_values
        self._store_solved_values(solved_values, source=mode)

        validation_values = self._values_from_variables(for_solver=True)
        relation_status, residuals, errors, warnings = self._evaluate_relation_residuals(
            validation_values,
            include_warning=True,
            relation_weight=1.0,
            strict=False,
            solver_residuals=False,
        )
        result["relation_status"] = relation_status
        result["variable_status"] = self._classify_variables(relation_status)
        result["residuals"] = residuals.tolist()
        result["errors"].extend(errors)
        result["warnings"].extend(warnings)
        result["termination"] = str(solve_result.message)
        result["solver"] = {
            "success": bool(solve_result.success),
            "status": int(solve_result.status),
            "cost": float(solve_result.cost),
            "optimality": float(solve_result.optimality),
            "nfev": int(solve_result.nfev),
            "active_mask": solve_result.active_mask.tolist(),
        }
        if not solve_result.success:
            result["warnings"].append(f"SciPy stopped before convergence: {solve_result.message}")
        result["success"] = self._relations_verified(relation_status) and not errors
        result["likely_culprits"] = self._rank_input_culprits(
            relation_status,
            result["variable_status"],
        )
        if mode == "reconcile" and result["likely_culprits"]:
            culprit_names = ", ".join(item["name"] for item in result["likely_culprits"][:5])
            result["warnings"].append(
                "Likely non-fixed input culprits: "
                f"{culprit_names}."
            )
        result["variables"] = self.variables_by_name
        result["relations"] = self.primary_relations
        return result

    def ordered_evaluate(self, order: Iterable[Any] | None = None, *, passes: int = 1) -> dict[str, Any]:
        """Evaluate relations sequentially; later outputs overwrite earlier values.

        Args:
            order: Optional explicit relation order.
            passes: Number of passes through the relation list.

        Returns:
            Result dictionary.
        """
        result = self._new_result("ordered")
        rels = self.primary_relations if order is None else [
            self.relations_by_name[str(item)] if not isinstance(item, Relation) else item for item in order
        ]
        values = self._values_from_variables(for_solver=True, skip_missing=True)
        for _ in range(int(passes)):
            for rel in rels:
                missing = [name for name in rel.input_names if name not in values or values[name] is None]
                if missing:
                    result["errors"].append(f"Cannot evaluate {rel.name!r}; missing {missing}.")
                    result["termination"] = "ordered evaluation stopped"
                    return result
                try:
                    if rel.output_names:
                        raw = rel.evaluate(values)
                        for out, value in rel.output_map(raw).items():
                            values[out] = value
                            if out in self.variables_by_name:
                                self.variables_by_name[out].set_value(self._public_value(out, value), source="computed")
                    else:
                        for lhs, op, rhs, _ in rel.comparisons(values):
                            tol = self._relation_tolerance(rel, None)
                            ok = compare_numeric(lhs, op, rhs, rel_tol=tol, zero_tol=self.zero_tol)
                            if ok is False:
                                raise ValueError(f"constraint {rel.name!r} violated")
                except Exception as exc:
                    result["errors"].append(f"Relation {rel.name!r} failed: {exc}")
                    result["termination"] = "ordered evaluation stopped"
                    return result
        validation = self.verify_current()
        result.update(validation)
        result["mode"] = "ordered"
        result["relations"] = rels
        result["termination"] = "ordered evaluation completed" if not result["errors"] else result["termination"]
        return result

    def compatibility_report(self) -> dict[str, dict[str, str]]:
        """Return a simple numeric compatibility report for active relations.

        Returns:
            Mapping from relation name to backend label.  The SciPy backend uses
            numeric evaluation for every relation.
        """
        return {rel.name: {"backend": "numeric"} for rel in self.relations}

    def _resolve_relation_names(self, rel: Relation) -> Relation:
        """Resolve relation variable aliases through the variable registry."""
        inputs = tuple(self.variable_registry.resolve(name) if name in self.variable_registry else name for name in rel.input_names)
        outputs = tuple(self.variable_registry.resolve(name) if name in self.variable_registry else name for name in rel.output_names)
        if inputs == rel.input_names and outputs == rel.output_names:
            return rel
        return Relation(
            name=rel.name,
            func=rel.func,
            input_names=inputs,
            outputs=outputs,
            op=rel.op,
            rhs=rel.rhs,
            tags=rel.tags,
            enforce=rel.enforce,
            constraints=None,
            source_kind=rel.source_kind,
            source_name=rel.source_name,
            constant_names=rel.constant_names,
        )

    def _infer_profile_size(self) -> int:
        """Infer the common profile size from loaded profile variables."""
        sizes: set[int] = set()
        for var in self.variables:
            if var.shape != 1:
                continue
            if var.size is not None:
                sizes.add(var.size)
            elif isinstance(var.value, np.ndarray) and var.value.ndim == 1:
                sizes.add(int(var.value.shape[0]))
        if len(sizes) > 1:
            raise ValueError(f"Profile sizes are incompatible: {sorted(sizes)}.")
        return next(iter(sizes), DEFAULT_PROFILE_SIZE)

    def _resolve_names(self, names: Iterable[str]) -> set[str]:
        """Resolve variable aliases for a user supplied name list."""
        resolved: set[str] = set()
        for name in names:
            text = str(name)
            resolved.add(self.variable_registry.resolve(text) if text in self.variable_registry else text)
        return resolved

    def _add_or_repair_rho_if_needed(self) -> None:
        """Create a fixed normalized radial grid when any candidate uses rho."""
        if "rho" not in self.variable_registry:
            return
        uses_rho = any("rho" in rel.variables for rel in self.candidate_primary_relations)
        has_profile = any(var.shape == 1 for var in self.variables_by_name.values())
        if not uses_rho and not has_profile:
            return

        value = np.linspace(0.0, 1.0, self.profile_size)
        if "rho" not in self.variables_by_name:
            rho = Variable("rho", value=value, size=self.profile_size, fixed=True)
            rho.source = "defaulted"
            rho.freedom = "fixed"
            self.variables.append(rho)
            self.variables_by_name["rho"] = rho
            return

        rho = self.variables_by_name["rho"]
        if rho.value is None:
            rho.size = self.profile_size
            rho.set_value(value, source="defaulted")
        rho.fixed = True
        rho.freedom = "fixed"

    def _add_uniform_profile_relation_candidates(self) -> None:
        """Add fallback profile relations only for missing profile variables."""
        for name, var in list(self.variables_by_name.items()):
            if var.shape != 1 or var.value is not None:
                continue
            spec = self.variable_registry.get(name)
            avg = spec.average_variable
            if not avg or spec.default_relation or avg not in self.variables_by_name:
                continue

            def make_uniform_profile(avg_name: str) -> Callable[..., Any]:
                def uniform_profile(**kwargs: Any) -> Any:
                    return kwargs[avg_name]

                return uniform_profile

            self.candidate_primary_relations.append(
                Relation(
                    name=f"Uniform profile {name} from {avg}",
                    func=make_uniform_profile(avg),
                    input_names=(avg,),
                    outputs=(name,),
                    source_kind="default_profile",
                    source_name=name,
                )
            )

    def _plan_active_relations(self) -> None:
        """Select the reachable relation subgraph before any SciPy solve."""
        for name in sorted(self.targets | self.solve_for):
            self._ensure_variable_exists(name)

        reachable = {
            name
            for name, var in self.variables_by_name.items()
            if var.value is not None
        }
        # Explicit solve variables are allowed to start unknown.  They do not
        # cause unrelated high-arity relations to activate unless those relations
        # are otherwise structurally connected.
        reachable.update(self.solve_for)

        active_primary: list[Relation] = []
        active_names: set[str] = set()
        changed = True
        while changed:
            changed = False
            for rel in self.candidate_primary_relations:
                if rel.name in active_names:
                    continue
                inputs = set(rel.input_names)
                outputs = set(rel.output_names)
                variables = set(rel.variables)

                activate = False
                if rel.output_names and not rel.implicit:
                    # Forward structural reachability: outputs become unknowns
                    # once all inputs are already reachable.
                    activate = inputs <= reachable
                elif rel.output_names and rel.implicit:
                    # Implicit relations are only safe when all variables are
                    # reachable, or when the unknown outputs were explicitly
                    # requested as solve variables/targets.
                    activate = variables <= reachable or (
                        bool(outputs & (self.solve_for | self.targets))
                        and (inputs - outputs) <= reachable
                    )
                else:
                    # Outputless constraints do not create variables.  They only
                    # check an already reachable part of the graph.
                    activate = variables <= reachable

                if not activate:
                    continue

                for name in variables:
                    self._ensure_variable_exists(name)
                active_primary.append(rel)
                active_names.add(rel.name)
                reachable.update(outputs)
                changed = True

        active_relations: list[Relation] = list(active_primary)
        active_relation_names = {rel.name for rel in active_relations}
        active_variables: set[str] = set()
        for rel in active_relations:
            active_variables.update(rel.variables)

        # Relation-local guards follow their parent relation.
        for rel in active_primary:
            for guard in rel.constraint_relations:
                guard = self._resolve_relation_names(guard)
                if guard.name in active_relation_names:
                    continue
                if set(guard.variables) <= active_variables:
                    active_relations.append(guard)
                    active_relation_names.add(guard.name)

        # Variable guards apply only to variables touched by the active graph.
        for name in sorted(active_variables):
            var = self.variables_by_name.get(name)
            if var is None:
                continue
            for guard in var.relations:
                guard = self._resolve_relation_names(guard)
                if guard.name in active_relation_names:
                    continue
                if set(guard.variables) <= active_variables:
                    active_relations.append(guard)
                    active_relation_names.add(guard.name)

        # System constraints are active only when all variables are reachable.
        for rel in self.system_constraint_relations:
            if rel.name in active_relation_names:
                continue
            if set(rel.variables) <= active_variables:
                active_relations.append(rel)
                active_relation_names.add(rel.name)

        self.active_primary_relations = active_primary
        self.primary_relations = active_primary
        self.relations = active_relations
        self.active_variable_names = active_variables
        self.reachable_variables = reachable
        self.blocked_relations = [
            rel for rel in self.candidate_primary_relations if rel.name not in active_names
        ]
        self.undecided_variables = {
            name
            for name, var in self.variables_by_name.items()
            if name not in active_variables and var.value is None
        }
        self.relations_by_name = {
            rel.name: rel
            for rel in [*self.candidate_primary_relations, *self.system_constraint_relations, *active_relations]
        }

    def _ensure_variable_exists(self, raw_name: str) -> Variable:
        """Create one missing variable only when the active graph needs it."""
        name = self.variable_registry.resolve(raw_name) if raw_name in self.variable_registry else str(raw_name)
        if name in self.variables_by_name:
            return self.variables_by_name[name]
        if name not in self.variable_registry:
            raise ValueError(f"Relation requires unknown variable {name!r}.")
        spec = self.variable_registry.get(name)
        var = Variable(name, size=self.profile_size if spec.shape == 1 else None)
        var.source = "undecided"
        self.variables.append(var)
        self.variables_by_name[name] = var
        return var

    def _broadcast_profile_values(self) -> None:
        """Broadcast scalar profile values and validate all profile lengths."""
        for var in self.variables_by_name.values():
            if var.shape != 1:
                continue
            if var.size is None:
                var.size = self.profile_size
            if var.value is None:
                continue
            arr = np.asarray(var.value)
            if arr.ndim == 0:
                var.value = np.full(var.size, float(arr))
            elif arr.ndim == 1 and arr.shape[0] != var.size:
                raise ValueError(f"Profile {var.name!r} has length {arr.shape[0]}, expected {var.size}.")

    def _seed_missing_values_from_relations(self) -> None:
        """Seed missing active outputs from directly evaluable explicit relations."""
        changed = True
        while changed:
            changed = False
            values = self._values_from_variables(for_solver=True, skip_missing=True)
            for rel in self.relations:
                if rel.implicit or not rel.output_names:
                    continue
                if any(name not in values for name in rel.input_names):
                    continue
                if all(self.variables_by_name[name].value is not None for name in rel.output_names):
                    continue
                try:
                    mapped = rel.output_map(rel.evaluate(values))
                except Exception:
                    continue
                for name, value in mapped.items():
                    var = self.variables_by_name.get(name)
                    if var is None or var.value is not None:
                        continue
                    var.set_value(self._public_value(name, value), source="seeded")
                    changed = True

    def _pack_free_variables(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple[str, int, int, np.ndarray, np.ndarray]]]:
        """Pack active non-fixed variables into a normalized SciPy vector."""
        x0: list[float] = []
        lower: list[float] = []
        upper: list[float] = []
        x_scale: list[float] = []
        spans: list[tuple[str, int, int, np.ndarray, np.ndarray]] = []

        for name, var in self.variables_by_name.items():
            if name not in self.active_variable_names:
                continue
            if var.fixed:
                if var.value is None:
                    raise ValueError(f"Fixed variable {name!r} has no value.")
                continue

            spec = self.variable_registry.get(name)
            lb, ub = scipy_bounds(spec.solver_domain, zero_tol=self.zero_tol)
            size = var.size if var.shape == 1 else 1
            if var.shape == 1 and size is None:
                size = self.profile_size
                var.size = size

            start = len(x0)
            offsets: list[float] = []
            scales: list[float] = []
            for i in range(int(size)):
                init = self._initial_value(var, index=i if var.shape == 1 else None)
                if var.value is None:
                    step = 1e-2 * (len(x0) + 1) * max(abs(init), 1.0)
                    if np.isfinite(ub) and init + step > ub:
                        init = max(lb, init - step) if np.isfinite(lb) else init - step
                    else:
                        init = init + step
                    init = float(np.clip(init, lb, ub))
                scale = max(abs(init), 1.0)
                x0.append(0.0)
                lower.append((lb - init) / scale if np.isfinite(lb) else -np.inf)
                upper.append((ub - init) / scale if np.isfinite(ub) else np.inf)
                x_scale.append(1.0)
                offsets.append(init)
                scales.append(scale)

            spans.append(
                (
                    name,
                    start,
                    len(x0),
                    np.asarray(offsets, dtype=float),
                    np.asarray(scales, dtype=float),
                )
            )

        return (
            np.asarray(x0, dtype=float),
            np.asarray(lower, dtype=float),
            np.asarray(upper, dtype=float),
            np.asarray(x_scale, dtype=float),
            spans,
        )

    def _initial_value(self, var: Variable, index: int | None = None) -> float:
        """Return one finite solver-space initial value for a variable."""
        spec = self.variable_registry.get(var.name)
        lb, ub = scipy_bounds(spec.solver_domain, zero_tol=self.zero_tol)
        if var.value is not None:
            solver_value = self._solver_value(var.name, var.value)
            arr = np.asarray(solver_value, dtype=float)
            value = float(arr[index]) if var.shape == 1 and arr.ndim == 1 and index is not None else float(arr.ravel()[0])
            return float(np.clip(value, lb, ub))
        if np.isfinite(lb) and np.isfinite(ub):
            return float((lb + ub) / 2.0)
        if np.isfinite(lb):
            return float(lb + 1.0)
        if np.isfinite(ub):
            return float(ub - 1.0)
        return 1.0  # NOTE: this may be fragile for badly scaled variables.

    def _values_from_variables(self, *, for_solver: bool, skip_missing: bool = False) -> dict[str, Any]:
        """Return a namespace from the current Variable objects.

        Args:
            for_solver: If true, map physical boundary values to solver_domain.
            skip_missing: If true, omit missing variables instead of assigning None.

        Returns:
            Mapping of variable names to values.
        """
        values: dict[str, Any] = {}
        for name, var in self.variables_by_name.items():
            if var.value is None:
                if not skip_missing:
                    values[name] = None
                continue
            values[name] = self._solver_value(name, var.value) if for_solver else var.value
        return values

    def _values_from_vector(
        self,
        x: np.ndarray,
        spans: list[tuple[str, int, int, np.ndarray, np.ndarray]],
    ) -> dict[str, Any]:
        """Build a full variable namespace from a SciPy vector."""
        values = self._values_from_variables(for_solver=True, skip_missing=True)
        for name, start, stop, offsets, scales in spans:
            var = self.variables_by_name[name]
            raw = np.asarray(x[start:stop], dtype=float)
            actual = offsets + scales * raw
            values[name] = actual.copy() if var.shape == 1 else float(actual[0])
        return values

    def _solver_value(self, name: str, value: Any) -> Any:
        """Map a public physical value into the finite solver domain when needed."""
        spec = self.variable_registry.get(name)
        lb, ub = scipy_bounds(spec.solver_domain, zero_tol=self.zero_tol)
        try:
            arr = np.asarray(value, dtype=float)
        except Exception as exc:
            raise ValueError(f"Variable {name!r} value is not numeric.") from exc

        out = arr.astype(float, copy=True)
        if np.any(np.isnan(out)):
            raise ValueError(f"Variable {name!r} contains nan.")
        if np.any(np.isposinf(out)):
            if np.isfinite(ub):
                out = np.where(np.isposinf(out), ub, out)
            else:
                raise ValueError(f"Variable {name!r} contains +inf but has no finite solver upper bound.")
        if np.any(np.isneginf(out)):
            if np.isfinite(lb):
                out = np.where(np.isneginf(out), lb, out)
            else:
                raise ValueError(f"Variable {name!r} contains -inf but has no finite solver lower bound.")

        # Public boundary values are allowed; the solver uses the interior solver_domain.
        if np.isfinite(lb):
            out = np.maximum(out, lb)
        if np.isfinite(ub):
            out = np.minimum(out, ub)
        if out.ndim == 0:
            return float(out)
        return out

    def _public_value(self, name: str, value: Any) -> Any:
        """Map solver-boundary values back to physical/API boundaries."""
        spec = self.variable_registry.get(name)
        d_lo, d_hi, _, _ = spec.domain
        s_lo, s_hi = domain_bounds_for_solver(spec.solver_domain, zero_tol=self.zero_tol)
        arr = finite_array(value, name=name)
        out = arr.copy()
        if s_lo is not None and (d_lo is None or s_lo > d_lo):
            mapped = -np.inf if d_lo is None else d_lo
            out = np.where(arr <= s_lo + self.zero_tol, mapped, out)
        if s_hi is not None and (d_hi is None or s_hi < d_hi):
            mapped = np.inf if d_hi is None else d_hi
            out = np.where(arr >= s_hi - self.zero_tol, mapped, out)
        if out.ndim == 0:
            return float(out)
        return out.astype(float)

    def _store_solved_values(self, values: dict[str, Any], *, source: str) -> None:
        """Store solved active namespace values back into Variable objects."""
        for name, value in values.items():
            if name not in self.active_variable_names or value is None:
                continue
            if name not in self.variables_by_name:
                continue
            var = self.variables_by_name[name]
            if var.fixed:
                continue
            var.set_value(self._public_value(name, value), source=source)

    def _evaluate_relation_residuals(
        self,
        values: dict[str, Any],
        *,
        include_warning: bool,
        relation_weight: float,
        strict: bool,
        solver_residuals: bool = False,
    ) -> tuple[dict[str, dict[str, Any]], np.ndarray, list[str], list[str]]:
        """Evaluate all relation residuals and collect diagnostics.

        Args:
            values: Variable namespace in solver units.
            include_warning: Whether warning-only relations are evaluated.
            relation_weight: Multiplicative residual weight.
            strict: If true, evaluation failures are returned as errors suitable
                for stopping the solver.
            solver_residuals: If true, use solve-oriented residual scaling that
                preserves gradient information far from the target.

        Returns:
            Relation status mapping, residual vector, errors, and warnings.
        """
        status: dict[str, dict[str, Any]] = {}
        residual_blocks: list[np.ndarray] = []
        errors: list[str] = []
        warnings: list[str] = []

        for rel in self.relations:
            if not rel.enforce and not include_warning:
                continue
            missing = [name for name in rel.variables if name not in values or values[name] is None]
            if missing:
                record = self._relation_record(rel, "missing", missing=missing)
                status[rel.name] = record
                bucket = errors if rel.enforce else warnings
                bucket.append(f"Relation {rel.name!r} missing variables: {missing}.")
                continue
            try:
                comparisons = rel.comparisons(values)
            except Exception as exc:
                record = self._relation_record(rel, "invalid", message=str(exc))
                status[rel.name] = record
                bucket = errors if rel.enforce or strict else warnings
                bucket.append(f"Relation {rel.name!r} could not be evaluated: {exc}")
                continue

            relation_scaled: list[np.ndarray] = []
            relation_raw_max = 0.0
            relation_count = 0
            try:
                for lhs, op, rhs, output in comparisons:
                    lhs_arr, rhs_arr = self._broadcast_pair(lhs, rhs, rel, output)
                    raw = lhs_arr - rhs_arr
                    allowed = self._allowed_error(
                        rel,
                        output,
                        lhs_arr,
                        rhs_arr,
                        solver_residuals=solver_residuals,
                    )
                    scaled = self._scaled_residual(raw, op, allowed)
                    if not np.all(np.isfinite(scaled)):
                        raise ValueError("residual contains nan or inf")
                    relation_scaled.append(np.asarray(scaled, dtype=float).ravel())
                    relation_raw_max = max(relation_raw_max, float(np.max(np.abs(raw))))
                    relation_count += int(np.asarray(scaled).size)
            except Exception as exc:
                record = self._relation_record(rel, "invalid", message=str(exc))
                status[rel.name] = record
                bucket = errors if rel.enforce or strict else warnings
                bucket.append(f"Relation {rel.name!r} produced invalid residuals: {exc}")
                continue

            block = np.concatenate(relation_scaled) if relation_scaled else np.asarray([], dtype=float)
            max_scaled = float(np.max(np.abs(block))) if block.size else 0.0
            relation_ok = max_scaled <= 1.0
            status[rel.name] = self._relation_record(
                rel,
                "verified" if relation_ok else "violated",
                max_scaled_residual=max_scaled,
                max_abs_residual=relation_raw_max,
                residual_count=relation_count,
            )
            if rel.enforce:
                residual_blocks.append(relation_weight * block)
            if not relation_ok:
                bucket = errors if rel.enforce and not strict else warnings
                bucket.append(f"Relation {rel.name!r} violated; max scaled residual={max_scaled:.3g}.")

        residuals = np.concatenate(residual_blocks) if residual_blocks else np.asarray([], dtype=float)
        return status, residuals, errors, warnings

    def _broadcast_pair(self, lhs: Any, rhs: Any, rel: Relation, output: str | None) -> tuple[np.ndarray, np.ndarray]:
        """Return finite broadcast-compatible lhs/rhs arrays."""
        lhs_arr = finite_array(lhs, name=f"{rel.name} lhs")
        rhs_arr = finite_array(rhs, name=f"{rel.name} rhs")
        try:
            left, right = np.broadcast_arrays(lhs_arr, rhs_arr)
        except ValueError as exc:
            raise ValueError(
                f"Relation {rel.name!r} shape mismatch for output {output!r}: "
                f"lhs={lhs_arr.shape}, rhs={rhs_arr.shape}."
            ) from exc
        return left.astype(float), right.astype(float)

    def _allowed_error(
        self,
        rel: Relation,
        output: str | None,
        lhs: np.ndarray,
        rhs: np.ndarray,
        *,
        solver_residuals: bool,
    ) -> np.ndarray:
        """Return pointwise absolute tolerance implied by rel_tol.

        Args:
            rel: Active relation.
            output: Output variable name for output relations, else ``None``.
            lhs: Left-hand side values.
            rhs: Right-hand side values.
            solver_residuals: Whether the scale is being computed for the SciPy
                solve instead of user-facing diagnostics.

        Returns:
            Pointwise absolute tolerance.
        """
        tol = self._relation_tolerance(rel, output)
        # Diagnostic mode reports relative mismatch against the larger side of the
        # equation.  Solve mode must not do that for output relations because a
        # huge rhs can flatten the residual at roughly +/-100 and kill the local
        # gradient that least_squares needs to move the variable.
        if solver_residuals and output is not None:
            scale = np.maximum(np.abs(lhs), 1.0)
        else:
            scale = np.maximum(np.maximum(np.abs(lhs), np.abs(rhs)), 1.0)
        if output in self.variables_by_name:
            ref = self.variables_by_name[output].reference_value
            if ref is not None:
                try:
                    ref_arr = np.broadcast_to(np.asarray(self._solver_value(output, ref), dtype=float), scale.shape)
                    scale = np.maximum(scale, np.abs(ref_arr))
                except Exception:
                    pass
        return max(float(tol), 1e-15) * scale

    def _relation_tolerance(self, rel: Relation, output: str | None) -> float:
        """Return variable-level tolerance for one comparison."""
        if output in self.variables_by_name:
            return float(self.variables_by_name[output].rel_tol or self.variable_registry.rel_tol_default)
        involved = [self.variables_by_name[name].rel_tol for name in rel.variables if name in self.variables_by_name]
        if involved:
            return float(max(value or self.variable_registry.rel_tol_default for value in involved))
        return float(self.variable_registry.rel_tol_default)

    def _scaled_residual(self, raw: np.ndarray, op: str, allowed: np.ndarray) -> np.ndarray:
        """Scale equality or inequality residuals into tolerance units."""
        if op == "==":
            return raw / allowed
        if op in {">", ">="}:
            return np.minimum(raw, 0.0) / allowed
        if op in {"<", "<="}:
            return np.maximum(raw, 0.0) / allowed
        raise ValueError(f"Unsupported operator {op!r}.")

    def _movement_residuals(
        self,
        values: dict[str, Any],
        spans: list[tuple[str, int, int, np.ndarray, np.ndarray]],
        *,
        weight: float,
    ) -> np.ndarray:
        """Return scaled movement-from-reference residuals for free variables."""
        blocks: list[np.ndarray] = []
        for name, *_ in spans:
            var = self.variables_by_name[name]
            if var.reference_value is None:
                continue
            current = finite_array(values[name], name=name)
            reference = finite_array(self._solver_value(name, var.reference_value), name=f"{name} reference")
            current, reference = np.broadcast_arrays(current, reference)
            scale = np.maximum(np.maximum(np.abs(current), np.abs(reference)), 1.0)
            allowed = max(float(var.rel_tol or self.variable_registry.rel_tol_default), 1e-15) * scale
            blocks.append(weight * ((current - reference) / allowed).ravel())
        return np.concatenate(blocks) if blocks else np.asarray([], dtype=float)

    def _objective_residual(
        self,
        values: dict[str, Any],
        *,
        objective: str | Callable[[dict[str, Any]], Any],
        sense: str,
        weight: float,
    ) -> np.ndarray:
        """Return a one-element residual that nudges an optimization objective.

        The SciPy backend remains a residual solver.  Positive minimization
        objectives are pushed toward zero; positive maximization objectives are
        pushed upward through an inverse residual.  For exact constrained scalar
        optimization, a future backend should use scipy.optimize.minimize.
        """
        if isinstance(objective, str):
            name = self.variable_registry.resolve(objective)
            value = values[name]
        else:
            value = objective(values)
        arr = finite_array(value, name="objective")
        scalar = float(np.sum(arr))
        scale = max(abs(scalar), 1.0)
        if str(sense).lower().startswith("max"):
            if scalar <= 0.0:
                return np.asarray([weight * 1e6], dtype=float)
            return np.asarray([weight / max(scalar / scale, 1e-12)], dtype=float)
        return np.asarray([weight * scalar / scale], dtype=float)

    def _build_jac_sparsity(
        self,
        x0: np.ndarray,
        spans: list[tuple[str, int, int, np.ndarray, np.ndarray]],
        *,
        include_movement: bool,
        include_objective: bool,
    ) -> Any:
        """Build a conservative relation-level finite-difference sparsity pattern."""
        if x0.size < 2:
            return None
        values = self._values_from_vector(x0, spans)
        col_by_name: dict[str, list[int]] = {}
        for name, start, stop, *_ in spans:
            col_by_name[name] = list(range(start, stop))

        row_dependencies: list[list[int]] = []
        for rel in self.relations:
            if not rel.enforce:
                continue
            if any(name not in values or values[name] is None for name in rel.variables):
                continue
            try:
                comparisons = rel.comparisons(values)
            except Exception:
                continue
            cols: list[int] = []
            for name in rel.jacobian_variables:
                cols.extend(col_by_name.get(name, []))
            if not cols:
                continue
            for lhs, _op, rhs, output in comparisons:
                try:
                    left, right = self._broadcast_pair(lhs, rhs, rel, output)
                    count = int(np.broadcast(left, right).size)
                except Exception:
                    count = 1
                for _ in range(count):
                    row_dependencies.append(cols)

        if include_movement:
            for name, start, stop, *_ in spans:
                if self.variables_by_name[name].reference_value is None:
                    continue
                for col in range(start, stop):
                    row_dependencies.append([col])
        if include_objective:
            row_dependencies.append(list(range(x0.size)))
        if not row_dependencies:
            return None

        mat = lil_matrix((len(row_dependencies), x0.size), dtype=bool)
        for row, cols in enumerate(row_dependencies):
            for col in cols:
                mat[row, col] = True
        return mat.tocsr()

    def _relation_record(self, rel: Relation, status: str, **extra: Any) -> dict[str, Any]:
        """Return a normalized relation-status record."""
        record = {
            "name": rel.name,
            "status": status,
            "enforce": rel.enforce,
            "source_kind": rel.source_kind,
            "source_name": rel.source_name,
            "variables": rel.variables,
            "outputs": rel.output_names,
        }
        record.update(extra)
        return record

    def _relations_verified(self, relation_status: dict[str, dict[str, Any]]) -> bool:
        """Return whether all enforced relations are verified."""
        for record in relation_status.values():
            if record.get("enforce") and record.get("status") != "verified":
                return False
        return True

    def _classify_variables(self, relation_status: dict[str, dict[str, Any]]) -> dict[str, str]:
        """Infer variable statuses from active relation status and movement."""
        touched: dict[str, list[dict[str, Any]]] = {name: [] for name in self.variables_by_name}
        for record in relation_status.values():
            for name in record.get("variables", ()):
                if name in touched:
                    touched[name].append(record)

        variable_status: dict[str, str] = {}
        for name, var in self.variables_by_name.items():
            records = touched.get(name, [])
            if name not in self.active_variable_names:
                state = "unused" if var.value is not None else "undecided"
            elif not records:
                state = "active" if var.value is not None else "unresolved"
            elif any(r.get("status") in {"missing", "invalid"} for r in records):
                state = "unresolved"
            elif any(r.get("enforce") and r.get("status") == "violated" for r in records):
                state = "fixed_inconsistent" if var.fixed else "suspect"
            elif self._variable_moved(var):
                state = "adjusted"
            else:
                state = "consistent"
            var.validity = state
            variable_status[name] = state
        return variable_status

    def _rank_input_culprits(
        self,
        relation_status: dict[str, dict[str, Any]],
        variable_status: dict[str, str],
    ) -> list[dict[str, Any]]:
        """Rank likely problematic non-fixed input variables.

        Args:
            relation_status: Final relation diagnostics for the run.
            variable_status: Final variable statuses for the run.

        Returns:
            Ranked culprit records for supplied non-fixed inputs only.
        """
        # Group enforced non-verified relations by the variables they touch.
        touched: dict[str, list[dict[str, Any]]] = {name: [] for name in self.variables_by_name}
        for record in relation_status.values():
            if not record.get("enforce"):
                continue
            if record.get("status") == "verified":
                continue
            for name in record.get("variables", ()):
                if name in touched:
                    touched[name].append(record)

        ranked: list[dict[str, Any]] = []
        for name, var in self.variables_by_name.items():
            # Fixed quantities are constraints, not candidate culprits.
            if var.fixed or var.reference_value is None:
                continue

            relation_records = touched.get(name, [])
            movement_score = self._variable_movement_score(var)
            if not relation_records and movement_score <= 0.0:
                continue

            relation_score = sum(self._culprit_relation_score(record) for record in relation_records)
            max_relation_residual = max(
                (float(record.get("max_scaled_residual", 0.0)) for record in relation_records),
                default=0.0,
            )
            top_relations = tuple(
                record["name"]
                for record in sorted(
                    relation_records,
                    key=lambda item: self._culprit_relation_score(item),
                    reverse=True,
                )[:5]
            )
            ranked.append(
                {
                    "name": name,
                    "score": float(relation_score + movement_score),
                    "status": variable_status.get(name, "unresolved"),
                    "movement_score": float(movement_score),
                    "relation_count": len(relation_records),
                    "max_relation_residual": float(max_relation_residual),
                    "top_relations": top_relations,
                }
            )

        ranked.sort(
            key=lambda item: (
                -item["score"],
                -item["max_relation_residual"],
                -item["movement_score"],
                item["name"],
            )
        )
        return ranked

    def _culprit_relation_score(self, record: dict[str, Any]) -> float:
        """Return one scalar severity score for culprit ranking.

        Args:
            record: One relation-status record.

        Returns:
            Non-negative severity score in tolerance units.
        """
        if record.get("status") == "violated":
            return float(record.get("max_scaled_residual", 0.0))
        if record.get("status") in {"missing", "invalid"}:
            return 1e3
        return 0.0

    def _variable_moved(self, var: Variable) -> bool:
        """Return whether a variable moved more than its tolerance from reference."""
        return bool(self._variable_movement_score(var) > 1.0)

    def _variable_movement_score(self, var: Variable) -> float:
        """Return movement in units of the variable tolerance.

        Args:
            var: Variable to compare against its reference value.

        Returns:
            Maximum scaled movement. Zero means no measurable movement.
        """
        if var.reference_value is None or var.value is None:
            return 0.0
        try:
            # Compare in solver space so clipped boundary values stay consistent.
            current = finite_array(self._solver_value(var.name, var.value), name=var.name)
            ref = finite_array(self._solver_value(var.name, var.reference_value), name=f"{var.name} reference")
            current, ref = np.broadcast_arrays(current, ref)
            scale = np.maximum(np.maximum(np.abs(current), np.abs(ref)), 1.0)
            allowed = max(float(var.rel_tol or self.variable_registry.rel_tol_default), 1e-15) * scale
            return float(np.max(np.abs(current - ref) / allowed))
        except Exception:
            return 0.0

    def _new_result(self, mode: str) -> dict[str, Any]:
        """Create the common result dictionary."""
        return {
            "mode": mode,
            "success": False,
            "variables": self.variables_by_name,
            "relations": self.primary_relations,
            "candidate_relations": self.candidate_primary_relations,
            "blocked_relations": self.blocked_relations,
            "undecided_variables": sorted(self.undecided_variables),
            "active_variable_names": sorted(self.active_variable_names),
            "termination": "not run",
            "errors": [],
            "warnings": [],
            "relation_status": {},
            "variable_status": {},
            "likely_culprits": [],
            "residuals": [],
        }
