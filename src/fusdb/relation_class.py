"""Relation object used by the solver."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping
import inspect

import sympy as sp

from .relation_util import build_symbolic_model, try_sympify_expression


@dataclass(eq=False, slots=True, frozen=True)
class Relation:
    """Immutable relation metadata and evaluation helpers."""

    name: str
    variables: dict[str, sp.Symbol]
    numeric_functions: dict[str, tuple[tuple[str, ...], Callable]]
    _preferred_target: str | None = field(default=None, repr=False)
    tags: tuple[str, ...] = field(default_factory=tuple)
    rel_tol_default: float | None = None
    abs_tol_default: float | None = None
    constraints: tuple[str, ...] = field(default_factory=tuple)
    soft_constraints: tuple[str, ...] = field(default_factory=tuple)
    initial_guesses: dict[str, Callable] = field(default_factory=dict)
    inverse_functions: dict[str, Callable] = field(default_factory=dict)
    sympy_expression: sp.Expr | None = None

    def __post_init__(self) -> None:
        """Normalize mutable inputs into stable runtime containers."""
        from .registry import canonical_variable_name

        canonical_vars: dict[str, sp.Symbol] = {}
        for raw_name, symbol in (self.variables or {}).items():
            c_name = canonical_variable_name(str(raw_name))
            if c_name in canonical_vars:
                continue
            if isinstance(symbol, sp.Symbol) and symbol.name == c_name:
                canonical_vars[c_name] = symbol
            else:
                canonical_vars[c_name] = sp.Symbol(c_name, real=True)

        canonical_numeric: dict[str, tuple[tuple[str, ...], Callable]] = {}
        for raw_target, spec in (self.numeric_functions or {}).items():
            try:
                arg_names, fn = spec
            except Exception as exc:
                raise ValueError(
                    f"Relation '{self.name}' has invalid numeric_functions entry for "
                    f"target '{raw_target}': expected (inputs, callable)"
                ) from exc
            c_target = canonical_variable_name(str(raw_target))
            c_args = tuple(canonical_variable_name(str(name)) for name in arg_names)
            if not callable(fn):
                raise ValueError(
                    f"Relation '{self.name}' numeric function for target '{c_target}' is not callable"
                )
            canonical_numeric[c_target] = (c_args, fn)
            canonical_vars.setdefault(c_target, sp.Symbol(c_target, real=True))
            for name in c_args:
                canonical_vars.setdefault(name, sp.Symbol(name, real=True))

        target = None if self._preferred_target is None else canonical_variable_name(str(self._preferred_target))
        if target is not None:
            canonical_vars.setdefault(target, sp.Symbol(target, real=True))

        canonical_overrides = {
            canonical_variable_name(str(name)): fn
            for name, fn in (self.inverse_functions or {}).items()
        }

        object.__setattr__(self, "variables", canonical_vars)
        object.__setattr__(self, "numeric_functions", canonical_numeric)
        object.__setattr__(self, "_preferred_target", target)
        object.__setattr__(self, "tags", tuple(self.tags))
        object.__setattr__(self, "constraints", tuple(self.constraints))
        object.__setattr__(self, "soft_constraints", tuple(self.soft_constraints))
        object.__setattr__(self, "initial_guesses", dict(self.initial_guesses or {}))
        object.__setattr__(self, "inverse_functions", canonical_overrides)

    @property
    def preferred_target(self) -> str | None:
        """Return the preferred numeric target for this relation."""
        if self._preferred_target is not None:
            return self._preferred_target
        return next(iter(self.numeric_functions), None)

    def required_inputs(self, output: str | None = None) -> tuple[str, ...]:
        """Return ordered input names for the requested output variable."""
        out = self.preferred_target if output is None else output
        if out is not None and out in self.numeric_functions:
            return self.numeric_functions[out][0]
        if out is None:
            return tuple(self.variables)
        return tuple(name for name in self.variables if name != out)

    @classmethod
    def from_callable(
        cls,
        *,
        name: str,
        func: Callable,
        target: str | None = None,
        variables: Iterable[str] | None = None,
        inputs: Iterable[str] | None = None,
        tags: Iterable[str] = (),
        rel_tol_default: float | None = None,
        abs_tol_default: float | None = None,
        constraints: Iterable[str] = (),
        soft_constraints: Iterable[str] = (),
        initial_guesses: dict[str, Callable] | None = None,
        inverse_functions: dict[str, Callable] | None = None,
        numeric_functions: dict[str, tuple[tuple[str, ...], Callable]] | None = None,
        strict_symbolic: bool = False,
    ) -> "Relation":
        """Create a relation from a python callable and derive symbolic metadata."""
        relation_name = name
        if inputs is None:
            inputs = tuple(inspect.signature(func).parameters)
        else:
            inputs = tuple(inputs)

        base_vars: list[str] = list(variables or ())
        for input_name in inputs:
            if input_name not in base_vars:
                base_vars.append(input_name)
        if target is not None and target not in base_vars:
            base_vars.append(target)

        expr, symbols = build_symbolic_model(
            func,
            inputs,
            target,
            relation_name=relation_name,
            strict=strict_symbolic,
        )
        symbols_map = (
            symbols
            if symbols is not None
            else {name: sp.Symbol(name, real=True) for name in base_vars}
        )
        for var_name in base_vars:
            symbols_map.setdefault(var_name, sp.Symbol(var_name, real=True))

        numeric_map: dict[str, tuple[tuple[str, ...], Callable]] = dict(numeric_functions or {})
        if target is not None and target not in numeric_map:
            numeric_map[target] = (tuple(inputs), func)

        constraints_tuple = tuple(constraints)
        soft_constraints_tuple = tuple(soft_constraints)
        for expr_str in constraints_tuple:
            try_sympify_expression(
                str(expr_str),
                local_symbols=symbols_map,
                context=f"relation '{relation_name}' hard constraints",
                strict=strict_symbolic,
            )
        for expr_str in soft_constraints_tuple:
            try_sympify_expression(
                str(expr_str),
                local_symbols=symbols_map,
                context=f"relation '{relation_name}' soft constraints",
                strict=strict_symbolic,
            )

        return cls(
            name=relation_name,
            variables=symbols_map,
            numeric_functions=numeric_map,
            _preferred_target=target,
            tags=tuple(tags),
            rel_tol_default=rel_tol_default,
            abs_tol_default=abs_tol_default,
            constraints=constraints_tuple,
            soft_constraints=soft_constraints_tuple,
            initial_guesses=initial_guesses or {},
            inverse_functions=inverse_functions or {},
            sympy_expression=expr,
        )

    def evaluate(self, values: Mapping[str, object], target: str | None = None) -> object:
        """Evaluate a numeric function for the requested target (or default)."""
        eval_target = target
        if eval_target is None:
            eval_target = self.preferred_target
        if eval_target is None:
            raise ValueError(f"Relation '{self.name}' has no preferred target")
        spec = self.numeric_functions.get(eval_target)
        if spec is None:
            raise KeyError(f"Relation '{self.name}' has no numeric function for target '{eval_target}'")
        ordered, fn = spec
        return fn(*(values[name] for name in ordered))

    def inverse_solver(self, unknown: str) -> Callable | None:
        """Return numeric solver callable for the requested unknown."""
        if unknown not in self.variables:
            return None

        if unknown in self.numeric_functions:
            return self.numeric_functions[unknown][1]

        if self.sympy_expression is None:
            return None

        candidate_symbol = self.variables.get(unknown)
        if candidate_symbol is None:
            return None

        try:
            solutions = sp.solve(self.sympy_expression, candidate_symbol)
        except Exception:
            return None
        if not solutions:
            return None

        ordered = tuple(name for name in self.variables if name != unknown)
        try:
            args = [self.variables[name] for name in ordered]
            solver = sp.lambdify(args, solutions[0], modules=["numpy", "sympy"])
        except Exception:
            return None

        self.numeric_functions[unknown] = (ordered, solver)
        return solver

    def solve_for_value(self, unknown: str, values: Mapping[str, object]) -> object | None:
        """Solve for one variable from a mapping of known values."""
        if unknown not in self.variables:
            return None

        override_fn = self.inverse_functions.get(unknown)
        if override_fn is not None:
            try:
                return override_fn(values)
            except Exception:
                return None

        if unknown not in self.numeric_functions:
            if self.inverse_solver(unknown) is None:
                return None

        spec = self.numeric_functions.get(unknown)
        if spec is None:
            return None

        ordered, fn = spec
        try:
            args = tuple(values[name] for name in ordered)
        except KeyError:
            return None

        try:
            result = fn(*args)
        except Exception:
            return None

        try:
            scalar = float(result)
        except Exception:
            scalar = None
        return scalar if scalar is not None else result
