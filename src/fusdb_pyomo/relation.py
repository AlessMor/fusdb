"""Relation object and ``@relation`` decorator for numeric residual solving."""

from __future__ import annotations

import ast
import inspect
import operator
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Callable

from .utils import normalize_tags, parse_constraint_specs, unique_preserve_order

REGISTERED_RELATIONS: dict[str, "Relation"] = {}
_ALLOWED_OPS = {"==", "<", "<=", ">", ">="}


@dataclass
class Relation:
    """One physical equation, inequality, or warning relation.

    Args:
        name: Stable relation name.
        func: Python function implementing the right-hand side or residual.
        input_names: Canonical-ish function argument names. Final alias resolution is
            performed by ``RelationSystem`` so relations can be imported before registries.
        outputs: Declared output names. Empty output tuple means ``func`` returns an lhs
            residual that is compared with ``rhs`` using ``op``.
        op: Comparison operator for outputless constraints.
        tags: Descriptive/selective relation tags.
        enforce: False means warning-only.
        constraints: Relation-local validity constraints.
        source_kind: Source label for diagnostics.
        source_name: Source name for diagnostics.
        constant_names: Function parameters with defaults.
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
    constraint_relations: tuple["Relation", ...] = field(default_factory=tuple, init=False)
    jacobian_variables: tuple[str, ...] = field(default_factory=tuple, init=False)

    def __post_init__(self) -> None:
        """Normalize metadata and attach constraint relations."""
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

        owned: list[Relation] = []
        for index, (text, enforce) in enumerate(parse_constraint_specs(self.constraints)):
            owned.append(
                constraint_from_expression(
                    text,
                    name=f"{self.name}_constraint_{index}",
                    enforce=enforce,
                    source_kind="relation",
                    source_name=self.name,
                )
            )
        self.constraint_relations = tuple(owned)
        self.jacobian_variables = unique_preserve_order((*self.input_names, *self.outputs))

    @property
    def output_names(self) -> tuple[str, ...]:
        """Return declared output names."""
        return self.outputs

    @property
    def variables(self) -> tuple[str, ...]:
        """Return all variable names referenced by this relation."""
        return unique_preserve_order((*self.input_names, *self.outputs))

    @property
    def implicit(self) -> bool:
        """Return whether outputs also appear among function inputs."""
        return bool(set(self.outputs) & set(self.input_names))

    def __call__(self, **kwargs: Any) -> Any:
        """Evaluate the relation function by keyword names."""
        return self.evaluate(kwargs)

    def evaluate(self, namespace: Mapping[str, Any]) -> Any:
        """Evaluate the raw relation function.

        Args:
            namespace: Mapping from relation input names to values.

        Returns:
            Raw function result.
        """
        args: dict[str, Any] = {}
        sig = inspect.signature(self.func)
        for name in self.input_names:
            args[name] = namespace[name]
        for name in self.constant_names:
            if name in namespace:
                args[name] = namespace[name]
            else:
                default = sig.parameters[name].default
                if default is not inspect.Parameter.empty:
                    args[name] = default
        return self.func(**args)

    def output_map(self, result: Any) -> dict[str, Any]:
        """Map a relation result to declared outputs."""
        if not self.outputs:
            return {}
        if isinstance(result, Mapping):
            missing = [name for name in self.outputs if name not in result]
            extras = [name for name in result if name not in self.outputs]
            if missing or extras:
                raise ValueError(
                    f"Relation {self.name!r} returned keys that do not match outputs; "
                    f"missing={missing}, extra={extras}."
                )
            return {name: result[name] for name in self.outputs}
        if len(self.outputs) == 1:
            return {self.outputs[0]: result}
        if not isinstance(result, (tuple, list)):
            raise ValueError(f"Relation {self.name!r} must return tuple/list/dict for multiple outputs.")
        if len(result) != len(self.outputs):
            raise ValueError(f"Relation {self.name!r} expected {len(self.outputs)} outputs, got {len(result)}.")
        return dict(zip(self.outputs, result))

    def comparisons(self, namespace: Mapping[str, Any]) -> list[tuple[Any, str, Any, str | None]]:
        """Return comparison tuples ``(lhs, op, rhs, output_name)``.

        For output relations, the system convention is always ``output == func(inputs)``.
        For outputless constraints, the function value is compared against ``rhs``.
        """
        value = self.evaluate(namespace)
        if self.outputs:
            mapped = self.output_map(value)
            return [(namespace[name], "==", mapped[name], name) for name in self.outputs]
        return [(value, self.op, self.rhs, None)]

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
    ) -> "Relation":
        """Build a relation from a Python function signature."""
        input_names: list[str] = []
        constant_names: list[str] = []
        for parameter in inspect.signature(func).parameters.values():
            if parameter.kind in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.VAR_POSITIONAL}:
                raise ValueError(f"Relation {func.__name__!r} cannot use positional-only parameters or *args.")
            if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                continue
            if parameter.default is inspect.Parameter.empty:
                input_names.append(parameter.name)
            else:
                constant_names.append(parameter.name)
        if outputs is None:
            output_names: tuple[str, ...] = ()
        elif isinstance(outputs, str):
            output_names = (outputs,)
        else:
            output_names = tuple(str(item) for item in outputs)
        return cls(
            name=str(name or func.__name__),
            func=func,
            input_names=tuple(input_names),
            outputs=output_names,
            tags=tuple(tags or ()),
            enforce=enforce,
            constraints=constraints,
            source_kind="relation",
            source_name=str(name or func.__name__),
            constant_names=tuple(constant_names),
        )


def relation(
    _func: Callable[..., Any] | None = None,
    *,
    outputs: Any | None = None,
    name: str | None = None,
    tags: Iterable[str] | None = None,
    enforce: bool = True,
    constraints: Any = None,
) -> Callable[[Callable[..., Any]], Relation] | Relation:
    """Decorate a function as a FusDB relation.

    Args:
        outputs: Output variable name or names. Omit only for outputless constraints.
        name: Optional stable relation name.
        tags: Relation tags.
        enforce: False makes the relation warning-only.
        constraints: Additional relation-local validity checks.

    Returns:
        Decorator or built ``Relation``.
    """
    def decorator(func: Callable[..., Any]) -> Relation:
        built = Relation.from_function(
            func,
            outputs=outputs,
            name=name,
            tags=tags,
            enforce=enforce,
            constraints=constraints,
        )
        REGISTERED_RELATIONS[built.name] = built
        return built

    if _func is not None:
        return decorator(_func)
    return decorator


_COMPARE_OPS = {
    ast.Eq: "==",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
}
_BINARY_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _compile_expression(node: ast.AST, names: list[str]) -> Callable[[Mapping[str, Any]], Any]:
    if isinstance(node, ast.Name):
        name = node.id
        names.append(name)
        return lambda ns, name=name: ns[name]
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, Real):
            raise ValueError("Only real numeric constants are allowed in constraints.")
        value = float(node.value)
        return lambda ns, value=value: value
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise ValueError("Only unary + and - are supported in constraints.")
        operand = _compile_expression(node.operand, names)
        op = _UNARY_OPS[op_type]
        return lambda ns, operand=operand, op=op: op(operand(ns))
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BINARY_OPS:
            raise ValueError("Only +, -, *, /, and ** are supported in constraints.")
        left = _compile_expression(node.left, names)
        right = _compile_expression(node.right, names)
        op = _BINARY_OPS[op_type]
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
    """Parse ``lhs op rhs`` into an outputless relation."""
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
        return left(kwargs) - right(kwargs)

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
    )
