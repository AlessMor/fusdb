"""Relation object used by the solver."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping

import sympy as sp

from .relation_util import (
    apply_relation,
    build_relation_from_callable,
    evaluate_relation,
    inverse_solver_for_relation,
    normalize_relation_definition,
    solve_relation_for_value,
)


@dataclass(eq=False, slots=True, frozen=True)
class Relation:
    """Immutable relation metadata and evaluation helpers."""

    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    forward: Callable
    tags: tuple[str, ...] = field(default_factory=tuple)
    rel_tol_default: float | None = None
    abs_tol_default: float | None = None
    constraints: tuple[str, ...] = field(default_factory=tuple)
    soft_constraints: tuple[str, ...] = field(default_factory=tuple)
    initial_guesses: dict[str, Callable] = field(default_factory=dict)
    inverse_functions: dict[str, Callable] = field(default_factory=dict)
    sympy_expression: sp.Expr | None = None
    symbols: dict[str, sp.Symbol] = field(default_factory=dict, repr=False)
    solvers: dict[str, tuple[tuple[str, ...], Callable]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """Normalize mutable inputs into stable runtime containers."""
        normalize_relation_definition(self)

    @property
    def is_multi_output(self) -> bool:
        """Return True when the relation writes more than one output."""
        return len(self.outputs) > 1

    @property
    def is_forward_only(self) -> bool:
        """Return True when the relation should not participate in inverse solving."""
        return self.is_multi_output

    def __call__(self, *args, **kwargs):
        """Delegate direct calls to the forward function."""
        return self.forward(*args, **kwargs)

    @classmethod
    def from_callable(
        cls,
        *,
        name: str,
        func: Callable,
        target: str | None = None,
        outputs: Iterable[str] | None = None,
        symbols: Iterable[str] | Mapping[str, object] | None = None,
        inputs: Iterable[str] | None = None,
        tags: Iterable[str] = (),
        rel_tol_default: float | None = None,
        abs_tol_default: float | None = None,
        constraints: Iterable[str] = (),
        soft_constraints: Iterable[str] = (),
        initial_guesses: dict[str, Callable] | None = None,
        inverse_functions: dict[str, Callable] | None = None,
        solvers: dict[str, tuple[tuple[str, ...], Callable]] | None = None,
        strict_symbolic: bool = False,
    ) -> "Relation":
        """Create a relation from a python callable and derive symbolic metadata."""
        return build_relation_from_callable(
            name=name,
            func=func,
            target=target,
            outputs=outputs,
            symbols=symbols,
            inputs=inputs,
            tags=tags,
            rel_tol_default=rel_tol_default,
            abs_tol_default=abs_tol_default,
            constraints=constraints,
            soft_constraints=soft_constraints,
            initial_guesses=initial_guesses,
            inverse_functions=inverse_functions,
            solvers=solvers,
            strict_symbolic=strict_symbolic,
        )

    def evaluate(self, values: Mapping[str, object], target: str | None = None) -> object:
        """Evaluate a numeric function for the requested target (or default)."""
        return evaluate_relation(self, values, target)

    def apply(self, values: Mapping[str, object]) -> dict[str, object]:
        """Evaluate the forward mapping and return output assignments."""
        return apply_relation(self, values)

    def inverse_solver(self, unknown: str) -> Callable | None:
        """Return numeric solver callable for the requested unknown."""
        return inverse_solver_for_relation(self, unknown)

    def solve_for_value(self, unknown: str, values: Mapping[str, object]) -> object | None:
        """Solve for one variable from a mapping of known values."""
        return solve_relation_for_value(self, unknown, values)
