"""Relation object and relation registry utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping
import inspect
import logging
import types
import warnings

import sympy as sp

from .utils import normalize_tag, normalize_tags_to_tuple

_RELATION_REGISTRY: list["Relation"] = []


def try_sympify_expression(
    expression: str,
    *,
    local_symbols: Mapping[str, object] | None = None,
    context: str | None = None,
    strict: bool = False,
) -> sp.Expr | None:
    """Parse an expression with sympy.

    Args:
        expression: Candidate expression string.
        local_symbols: Optional local symbol mapping for parsing.
        context: Optional context label for errors.
        strict: Raise instead of warning when parsing fails.

    Returns:
        Parsed sympy expression or ``None`` when parsing fails.
    """
    # Parse once and keep failures explicit.
    try:
        return sp.sympify(expression, locals=dict(local_symbols or {}))
    except Exception as exc:
        where = f" for {context}" if context else ""
        msg = (
            f"Sympy could not parse expression{where}: {expression!r} "
            f"({type(exc).__name__}: {exc})"
        )
        if strict:
            raise ValueError(msg) from exc
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        return None


def _build_symbolic_model(
    func: Callable,
    arg_names: Iterable[str],
    preferred_target: str | None,
    *,
    relation_name: str | None = None,
    strict: bool = False,
) -> tuple[sp.Expr | None, dict[str, sp.Symbol] | None]:
    """Build implicit equation ``preferred_target - f(arg_names)``.

    Args:
        func: Forward callable.
        arg_names: Ordered callable argument names.
        preferred_target: Target variable name.
        relation_name: Optional relation label for diagnostics.
        strict: Raise instead of warning on symbolic conversion failures.

    Returns:
        Tuple ``(sympy_expression_or_none, symbols_or_none)``.
    """
    # Skip symbolic generation for relations without one scalar target.
    if preferred_target is None:
        return None, None

    # Build one canonical symbol map used by both direct and proxy calls.
    arg_tuple = tuple(arg_names)
    symbols = {name: sp.Symbol(name, real=True) for name in (*arg_tuple, preferred_target)}
    symbol_inputs = {name: symbols[name] for name in arg_tuple}

    # Try direct symbolic call first for simple numeric callables.
    direct_exc: Exception | None = None
    try:
        expr = func(**symbol_inputs)
        return symbols[preferred_target] - expr, symbols
    except Exception as exc:
        direct_exc = exc

    # Patch math/numpy globals to sympy equivalents and retry once.
    patched_globals = dict(func.__globals__)
    patched_globals.setdefault("__builtins__", func.__globals__.get("__builtins__", __builtins__))
    sympy_names: dict[str, object] = {
        "pi": sp.pi,
        "e": sp.E,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "asin": sp.asin,
        "acos": sp.acos,
        "atan": sp.atan,
        "arcsin": sp.asin,
        "arccos": sp.acos,
        "arctan": sp.atan,
        "sinh": sp.sinh,
        "cosh": sp.cosh,
        "tanh": sp.tanh,
        "asinh": sp.asinh,
        "acosh": sp.acosh,
        "atanh": sp.atanh,
        "sqrt": sp.sqrt,
        "exp": sp.exp,
        "log": sp.log,
        "log10": sp.log,
        "log2": sp.log,
        "power": sp.Pow,
        "abs": sp.Abs,
        "fabs": sp.Abs,
        "floor": sp.floor,
        "ceil": getattr(sp, "ceil", sp.ceiling),
    }
    module_proxy = types.SimpleNamespace(
        **{
            **{name: getattr(sp, name) for name in dir(sp)},
            **sympy_names,
        }
    )
    for module_name in ("math", "np", "numpy"):
        if module_name in patched_globals:
            patched_globals[module_name] = module_proxy
    for name, sympy_obj in sympy_names.items():
        if name in patched_globals:
            patched_globals[name] = sympy_obj

    # Rebuild callable with patched globals and retry.
    proxy_exc: Exception | None = None
    try:
        patched_func = types.FunctionType(
            func.__code__,
            patched_globals,
            func.__name__,
            func.__defaults__,
            func.__closure__,
        )
        patched_func.__kwdefaults__ = getattr(func, "__kwdefaults__", None)
        expr = patched_func(**symbol_inputs)
        return symbols[preferred_target] - expr, symbols
    except Exception as exc:
        proxy_exc = exc

    # Emit one clear diagnostic when both symbolic strategies fail.
    rel_label = relation_name or getattr(func, "__name__", "<unknown>")
    direct_label = "n/a" if direct_exc is None else f"{type(direct_exc).__name__}: {direct_exc}"
    proxy_label = "n/a" if proxy_exc is None else f"{type(proxy_exc).__name__}: {proxy_exc}"
    msg = (
        f"Could not convert relation '{rel_label}' to sympy expression for target "
        f"'{preferred_target}'. direct={direct_label}; proxy={proxy_label}"
    )
    if strict:
        raise ValueError(msg) from (proxy_exc or direct_exc)
    warnings.warn(msg, RuntimeWarning, stacklevel=2)
    return None, symbols


def _solve_for_policy(spec: object) -> tuple[bool, Callable | None, tuple[str, ...] | None]:
    """Return solve_for policy triplet from one solve_for entry.

    Args:
        spec: One solve_for entry payload.

    Returns:
        Tuple ``(enabled, callable_or_none, inputs_or_none)``.
    """
    # Normalize all accepted solve_for entry formats.
    if isinstance(spec, bool):
        return spec, None, None
    if callable(spec):
        return True, spec, None
    if isinstance(spec, Mapping):
        enabled = bool(spec.get("enabled", True))
        fn = spec.get("fn")
        inputs = spec.get("inputs")
        if inputs is not None:
            inputs = tuple(str(name) for name in inputs)
        return enabled, fn if callable(fn) else None, inputs
    return True, None, None


def _call_target_solver(spec: Mapping[str, object], values: Mapping[str, object]) -> object:
    """Evaluate one solve_for target entry."""
    # Dispatch explicit solver by positional inputs when requested.
    fn = spec.get("fn")
    if not callable(fn):
        raise KeyError("solve_for entry has no callable fn")
    ordered = spec.get("inputs")
    if ordered:
        return fn(*(values[name] for name in tuple(ordered)))
    return fn(values)


@dataclass(eq=False, slots=True, frozen=True)
class Relation:
    """Immutable relation metadata and evaluation helpers."""

    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    forward: Callable
    tags: tuple[str, ...] = field(default_factory=tuple)
    constraints: tuple[str, ...] = field(default_factory=tuple)
    initial_guesses: dict[str, Callable] = field(default_factory=dict)
    solve_for: dict[str, object] = field(default_factory=dict)
    sympy_expression: sp.Expr | None = field(init=False, default=None, repr=False)
    symbols: dict[str, sp.Symbol] = field(init=False, default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """Normalize mutable inputs into stable runtime containers.

        Args:
            None.

        Returns:
            None.
        """
        from .registry import canonical_variable_name

        # Canonicalize explicit input/output names.
        canonical_inputs = tuple(canonical_variable_name(str(name)) for name in (self.inputs or ()))
        canonical_outputs = tuple(canonical_variable_name(str(name)) for name in (self.outputs or ()))
        if not canonical_outputs:
            raise ValueError(f"Relation '{self.name}' must define at least one output")
        if not callable(self.forward):
            raise ValueError(f"Relation '{self.name}' forward callable is not callable")

        # Build canonical symbol map covering all declared variables.
        canonical_symbols: dict[str, sp.Symbol] = {}
        for raw_name, symbol in (self.symbols or {}).items():
            c_name = canonical_variable_name(str(raw_name))
            if c_name in canonical_symbols:
                continue
            if isinstance(symbol, sp.Symbol) and symbol.name == c_name:
                canonical_symbols[c_name] = symbol
            else:
                canonical_symbols[c_name] = sp.Symbol(c_name, real=True)
        for name in (*canonical_inputs, *canonical_outputs):
            canonical_symbols.setdefault(name, sp.Symbol(name, real=True))

        # Normalize initial guesses keyed by canonical target names.
        canonical_initial_guesses: dict[str, Callable] = {}
        for raw_target, fn in (self.initial_guesses or {}).items():
            c_target = canonical_variable_name(str(raw_target))
            if not callable(fn):
                raise ValueError(
                    f"Relation '{self.name}' initial guess for target '{c_target}' is not callable"
                )
            canonical_initial_guesses[c_target] = fn
            canonical_symbols.setdefault(c_target, sp.Symbol(c_target, real=True))

        # Normalize solve_for metadata and validate payloads.
        canonical_solve_for: dict[str, dict[str, object]] = {}
        for raw_target, spec in (self.solve_for or {}).items():
            c_target = canonical_variable_name(str(raw_target))
            if c_target not in canonical_symbols:
                raise ValueError(
                    f"Relation '{self.name}' solve_for target '{c_target}' is not part of relation variables"
                )
            canonical_symbols.setdefault(c_target, sp.Symbol(c_target, real=True))

            normalized_spec: dict[str, object] = {
                "enabled": True,
                "fn": None,
                "inputs": None,
                "guess": None,
                "source": "user",
            }
            if isinstance(spec, bool):
                normalized_spec["enabled"] = spec
            elif callable(spec):
                normalized_spec["fn"] = spec
            elif isinstance(spec, Mapping):
                enabled_raw = spec.get("enabled", True)
                if not isinstance(enabled_raw, bool):
                    raise ValueError(
                        f"Relation '{self.name}' solve_for[{c_target!r}].enabled must be bool"
                    )
                normalized_spec["enabled"] = enabled_raw

                fn = spec.get("fn")
                if fn is not None and not callable(fn):
                    raise ValueError(
                        f"Relation '{self.name}' solve_for[{c_target!r}].fn must be callable when provided"
                    )
                if fn is not None:
                    normalized_spec["fn"] = fn

                raw_inputs = spec.get("inputs")
                if raw_inputs is not None:
                    if isinstance(raw_inputs, str):
                        raw_inputs = (raw_inputs,)
                    try:
                        c_inputs = tuple(canonical_variable_name(str(name)) for name in raw_inputs)
                    except Exception as exc:
                        raise ValueError(
                            f"Relation '{self.name}' solve_for[{c_target!r}].inputs must be iterable of names"
                        ) from exc
                    normalized_spec["inputs"] = c_inputs
                    for c_name in c_inputs:
                        canonical_symbols.setdefault(c_name, sp.Symbol(c_name, real=True))

                guess = spec.get("guess")
                if guess is not None:
                    if not callable(guess):
                        raise ValueError(
                            f"Relation '{self.name}' solve_for[{c_target!r}].guess must be callable when provided"
                        )
                    normalized_spec["guess"] = guess
            else:
                raise ValueError(
                    f"Relation '{self.name}' solve_for[{c_target!r}] must be bool, callable, or mapping"
                )

            guess_fn = normalized_spec.get("guess")
            if callable(guess_fn):
                canonical_initial_guesses.setdefault(c_target, guess_fn)
            canonical_solve_for[c_target] = normalized_spec

        # Merge standalone initial guesses into solve_for entries.
        for target_name, guess_fn in canonical_initial_guesses.items():
            spec = canonical_solve_for.setdefault(
                target_name,
                {
                    "enabled": True,
                    "fn": None,
                    "inputs": None,
                    "guess": None,
                    "source": "auto",
                },
            )
            if spec.get("guess") is None:
                spec["guess"] = guess_fn

        # Ensure all known symbols have a solve_for direction entry.
        for target_name in tuple(canonical_symbols):
            canonical_solve_for.setdefault(
                target_name,
                {
                    "enabled": True,
                    "fn": None,
                    "inputs": None,
                    "guess": canonical_initial_guesses.get(target_name),
                    "source": "auto",
                },
            )

        # Enforce explicit inversion policy for multi-output relations.
        if len(canonical_outputs) > 1:
            if self.sympy_expression is not None:
                raise ValueError(
                    f"Relation '{self.name}' with multiple outputs cannot define a symbolic expression"
                )
            for target, spec in canonical_solve_for.items():
                enabled, fn, _inputs = _solve_for_policy(spec)
                if target in canonical_outputs:
                    continue
                if spec.get("source") == "auto":
                    spec["enabled"] = False
                    continue
                if enabled and fn is None:
                    raise ValueError(
                        f"Relation '{self.name}' with multiple outputs must provide an explicit "
                        f"solve_for callable for target '{target}'"
                    )

        # Commit normalized immutable fields.
        object.__setattr__(self, "inputs", canonical_inputs)
        object.__setattr__(self, "outputs", canonical_outputs)
        object.__setattr__(self, "tags", tuple(self.tags))
        object.__setattr__(self, "constraints", tuple(self.constraints))
        object.__setattr__(self, "initial_guesses", canonical_initial_guesses)
        object.__setattr__(self, "solve_for", canonical_solve_for)
        object.__setattr__(self, "symbols", canonical_symbols)

    def __call__(self, *args, **kwargs):
        """Delegate direct calls to the forward function.

        Args:
            *args: Positional arguments for forward callable.
            **kwargs: Keyword arguments for forward callable.

        Returns:
            Forward callable result.
        """
        return self.forward(*args, **kwargs)

    @classmethod
    def from_callable(
        cls,
        *,
        name: str,
        func: Callable,
        target: str | None = None,
        outputs: Iterable[str] | None = None,
        inputs: Iterable[str] | None = None,
        tags: Iterable[str] = (),
        constraints: Iterable[str] = (),
        initial_guesses: dict[str, Callable] | None = None,
        solve_for: dict[str, object] | None = None,
        strict_symbolic: bool = False,
    ) -> "Relation":
        """Create a relation from a callable and derive symbolic metadata.

        Args:
            name: Relation name.
            func: Forward callable.
            target: Single output name for scalar-output relations.
            outputs: Output names for multi-output relations.
            inputs: Optional explicit input names.
            tags: Optional relation tags.
            constraints: Optional validation constraints.
            initial_guesses: Optional initial guesses by target name.
            solve_for: Optional solve_for metadata.
            strict_symbolic: Raise when symbolic conversion fails.

        Returns:
            Initialized relation object.
        """
        from .registry import canonical_variable_name

        # Resolve callable signature against declared inputs.
        relation_name = name
        signature = inspect.signature(func)
        input_tuple = tuple(signature.parameters) if inputs is None else tuple(inputs)
        input_tuple = tuple(canonical_variable_name(str(name)) for name in input_tuple)
        params = signature.parameters
        if not any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
            invalid = [
                name
                for name in input_tuple
                if name not in params
                or params[name].kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.VAR_POSITIONAL)
            ]
            if invalid:
                raise TypeError(
                    f"Relation '{relation_name}' forward callable cannot accept declared inputs by name: {invalid!r}"
                )

        # Resolve output tuple from either target or outputs.
        if outputs is None:
            output_tuple = () if target is None else (target,)
        else:
            output_tuple = tuple(outputs)
        output_tuple = tuple(canonical_variable_name(str(name)) for name in output_tuple)

        if len(output_tuple) > 1 and target is not None:
            raise ValueError(
                f"Relation '{relation_name}' cannot define both target={target!r} "
                f"and multiple outputs={output_tuple!r}"
            )
        if not output_tuple:
            raise ValueError(f"Relation '{relation_name}' must define at least one output")

        # Build symbolic expression for single-output relations.
        preferred_target = output_tuple[0]
        multi_output = len(output_tuple) > 1
        base_vars: list[str] = []
        for input_name in input_tuple:
            if input_name not in base_vars:
                base_vars.append(input_name)
        for output_name in output_tuple:
            if output_name not in base_vars:
                base_vars.append(output_name)

        if multi_output:
            expr = None
            symbols_map = {name: sp.Symbol(name, real=True) for name in base_vars}
        else:
            expr, symbols = _build_symbolic_model(
                func,
                input_tuple,
                preferred_target,
                relation_name=relation_name,
                strict=strict_symbolic,
            )
            symbols_map = symbols if symbols is not None else {}
        for var_name in base_vars:
            symbols_map.setdefault(var_name, sp.Symbol(var_name, real=True))

        # Validate relation constraints at construction time.
        constraints_tuple = tuple(constraints)
        for expr_str in constraints_tuple:
            try_sympify_expression(
                str(expr_str),
                local_symbols=symbols_map,
                context=f"relation '{relation_name}' constraints",
                strict=strict_symbolic,
            )

        # Build relation object and attach parsed symbolic expression.
        relation_obj = cls(
            name=relation_name,
            inputs=input_tuple,
            outputs=output_tuple,
            forward=func,
            tags=tuple(tags),
            constraints=constraints_tuple,
            initial_guesses=initial_guesses or {},
            solve_for=solve_for or {},
        )
        object.__setattr__(relation_obj, "sympy_expression", expr)
        return relation_obj

    def input_names(self, output: str | None = None) -> tuple[str, ...]:
        """Return ordered input names for one numeric target.

        Args:
            output: Optional output/target name.

        Returns:
            Ordered input names for the chosen target.
        """
        target = self.outputs[0] if output is None else output
        if target in self.outputs:
            return self.inputs
        spec = self.solve_for.get(target)
        if isinstance(spec, Mapping):
            inputs = spec.get("inputs")
            if inputs:
                return tuple(str(name) for name in inputs)
        return tuple(name for name in self.symbols if name != target)

    def apply(self, values: Mapping[str, object]) -> dict[str, object]:
        """Evaluate forward mapping and return output assignments.

        Args:
            values: Known values mapping.

        Returns:
            Output assignments keyed by canonical output names.
        """
        from .registry import canonical_variable_name

        # Evaluate forward callable with declared inputs.
        result = self.forward(**{name: values[name] for name in self.inputs})
        target = self.outputs[0]

        # Validate mapping output payloads.
        if isinstance(result, Mapping):
            canonical_result = {
                canonical_variable_name(str(name)): value
                for name, value in result.items()
            }
            if len(self.outputs) > 1:
                missing = [name for name in self.outputs if name not in canonical_result]
                if missing:
                    raise KeyError(
                        f"Relation '{self.name}' did not return outputs {missing}"
                    )
                return {name: canonical_result[name] for name in self.outputs}
            if target not in canonical_result:
                raise KeyError(
                    f"Relation '{self.name}' did not return output '{target}'"
                )
            return {target: canonical_result[target]}

        # Enforce mapping-return contract for multi-output relations.
        if len(self.outputs) > 1:
            raise TypeError(
                f"Relation '{self.name}' with multiple outputs must return a mapping"
            )
        return {target: result}

    def evaluate(self, values: Mapping[str, object], target: str | None = None) -> object:
        """Evaluate numeric function for one target direction.

        Args:
            values: Known values mapping.
            target: Optional target direction.

        Returns:
            Scalar output value or mapping for multi-output forward evaluation.
        """
        # Route direct forward evaluation first.
        eval_target = self.outputs[0] if target is None else target
        if len(self.outputs) > 1:
            if target is None:
                return self.apply(values)
            return self.apply(values)[eval_target]
        if eval_target in self.outputs:
            return self.apply(values)[eval_target]
        if eval_target not in self.symbols:
            raise KeyError(f"Relation '{self.name}' has no numeric function for target '{eval_target}'")

        # Validate target solve_for policy.
        enabled, fn, _inputs = _solve_for_policy(self.solve_for.get(eval_target))
        if not enabled:
            raise KeyError(f"Relation '{self.name}' solve_for target '{eval_target}' is disabled")
        if fn is None and self.inverse_solver(eval_target) is None:
            raise KeyError(f"Relation '{self.name}' has no numeric function for target '{eval_target}'")

        spec = self.solve_for.get(eval_target)
        if not isinstance(spec, Mapping):
            raise KeyError(f"Relation '{self.name}' has invalid solve_for entry for target '{eval_target}'")
        return _call_target_solver(spec, values)

    def inverse_solver(self, unknown: str) -> Callable | None:
        """Return numeric inverse solver callable for an unknown.

        Args:
            unknown: Target unknown variable name.

        Returns:
            Callable solver or ``None`` when unavailable.
        """
        from .registry import canonical_variable_name

        # Normalize unknown name and policy metadata.
        unknown = canonical_variable_name(str(unknown))
        if unknown not in self.symbols:
            return None
        enabled, explicit_fn, explicit_inputs = _solve_for_policy(self.solve_for.get(unknown))
        if not enabled:
            return None

        # Prefer explicit user-provided inverse solver.
        if explicit_fn is not None:
            if explicit_inputs:
                return explicit_fn
            ordered = tuple(name for name in self.symbols if name != unknown)

            def _wrapped_solver(*args):
                values_map = dict(zip(ordered, args, strict=False))
                return explicit_fn(values_map)

            return _wrapped_solver

        # Forward callable already solves declared output directly.
        if unknown in self.outputs:
            if len(self.outputs) == 1:
                return self.forward
            return None
        if len(self.outputs) > 1:
            return None
        if self.sympy_expression is None:
            return None

        # Build one symbolic inverse for scalar single-output relation.
        candidate_symbol = self.symbols.get(unknown)
        if candidate_symbol is None:
            return None

        try:
            solutions = sp.solve(self.sympy_expression, candidate_symbol)
        except Exception:
            return None
        if not solutions:
            return None

        ordered = tuple(name for name in self.symbols if name != unknown)
        try:
            args = [self.symbols[name] for name in ordered]
            solver = sp.lambdify(args, solutions[0], modules=["numpy", "sympy"])
        except Exception:
            return None

        # Cache generated solver in solve_for for reuse.
        solve_for = dict(self.solve_for)
        current_spec = solve_for.get(unknown)
        if isinstance(current_spec, Mapping):
            merged_spec = dict(current_spec)
        else:
            merged_spec = {}
        merged_spec.setdefault("enabled", True)
        merged_spec["fn"] = solver
        merged_spec["inputs"] = ordered
        merged_spec.setdefault("guess", self.initial_guesses.get(unknown))
        merged_spec.setdefault("source", "auto")
        solve_for[unknown] = merged_spec
        object.__setattr__(self, "solve_for", solve_for)
        return solver

    def solve_for_value(self, unknown: str, values: Mapping[str, object]) -> object | None:
        """Solve relation for one unknown from known values.

        Args:
            unknown: Unknown variable to solve.
            values: Known values mapping.

        Returns:
            Solved value or ``None`` when no valid solve path exists.
        """
        from .registry import canonical_variable_name

        # Normalize unknown and check solve_for policy.
        unknown = canonical_variable_name(str(unknown))
        if unknown not in self.symbols:
            return None
        enabled, explicit_fn, explicit_inputs = _solve_for_policy(self.solve_for.get(unknown))
        if not enabled:
            return None

        # Run explicit inverse function when available.
        if explicit_fn is not None:
            try:
                if explicit_inputs:
                    result = explicit_fn(*(values[name] for name in explicit_inputs))
                else:
                    result = explicit_fn(values)
            except Exception:
                return None
            try:
                scalar = float(result)
            except Exception:
                scalar = None
            return scalar if scalar is not None else result

        # Forward target can be solved directly by evaluate.
        if unknown in self.outputs:
            try:
                return self.evaluate(values, target=unknown)
            except Exception:
                return None
        if len(self.outputs) > 1:
            return None

        # Build symbolic inverse lazily when possible.
        if self.inverse_solver(unknown) is None:
            return None

        spec = self.solve_for.get(unknown)
        if not isinstance(spec, Mapping):
            return None

        try:
            result = _call_target_solver(spec, values)
        except Exception:
            return None

        try:
            scalar = float(result)
        except Exception:
            scalar = None
        return scalar if scalar is not None else result


def relation(
    *,
    name: str | None = None,
    output: str | None = None,
    outputs: Iterable[str] | None = None,
    inputs: Iterable[str] | None = None,
    tags: Iterable[str] | str | None = None,
    constraints: Iterable[str] | str | None = None,
    initial_guesses: dict[str, Callable] | None = None,
    solve_for: dict[str, object] | None = None,
):
    """Decorator that builds and registers relation objects.

    Args:
        name: Optional relation name.
        output: Single output name.
        outputs: Multi-output names.
        inputs: Optional explicit input names.
        tags: Optional tags.
        constraints: Optional validation constraints.
        initial_guesses: Optional initial guess callables.
        solve_for: Optional solve_for metadata.

    Returns:
        Decorator returning a registered relation object.
    """

    def decorator(func: Callable) -> Relation:
        """Create one relation object from ``func`` and register it."""
        # Enforce one explicit output declaration mode.
        if (output is None) == (outputs is None):
            raise ValueError("relation() requires exactly one of output= or outputs=")

        # Normalize constraint payloads to tuples.
        constraints_tuple = (
            () if constraints is None else (constraints,) if isinstance(constraints, str) else tuple(constraints)
        )

        # Build relation from callable and append to global registry.
        relation_obj = Relation.from_callable(
            name=name or func.__name__,
            target=output,
            outputs=outputs,
            func=func,
            inputs=inputs,
            tags=normalize_tags_to_tuple(tags),
            constraints=constraints_tuple,
            initial_guesses=initial_guesses or {},
            solve_for=solve_for or {},
        )
        _RELATION_REGISTRY.append(relation_obj)
        return relation_obj

    return decorator


def get_filtered_relations(
    reactor_tags: Iterable[str] | str | None,
    variable_methods: Iterable[str] | None,
    variable_names: Iterable[str] | None = None,
    verbose: bool = False,
    extra_relations: Iterable[object] | None = None,
):
    """Get relation objects matching tags and method overrides.

    Args:
        reactor_tags: Reactor/domain tags.
        variable_methods: Relation methods selected by variables.
        variable_names: Optional seed variable names used to keep only
            relations connected to the reactor variable subgraph.
        verbose: Enable filtering logs.
        extra_relations: Extra relation objects appended to registry relations.

    Returns:
        Filtered relation list in registry order.
    """
    from . import relations
    from .registry import canonical_variable_name, load_allowed_tags

    logger = logging.getLogger(__name__)

    # Ensure registry is populated before any filtering is applied.
    relations.import_relations()

    # Load allowed tag groups used by filtering semantics.
    allowed_tags = load_allowed_tags()
    solving_order = allowed_tags.get("solving_order", {}) or {}
    allowed_domains = set(solving_order)
    allowed_specific = (
        set(allowed_tags.get("reactor_configurations", ()) or ())
        | set(allowed_tags.get("confinement_modes", ()) or ())
        | set(allowed_tags.get("reactor_families", ()) or ())
    )

    # Normalize all filter inputs once for deterministic matching.
    tags_normalized = tuple(normalize_tags_to_tuple(reactor_tags or ()))
    method_names = {name for name in (variable_methods or ()) if name}
    variable_names_set = {canonical_variable_name(str(name)) for name in (variable_names or ()) if name}
    domain_filters = {tag for tag in tags_normalized if tag in allowed_domains}
    reactor_tags_normalized = tuple(tag for tag in tags_normalized if tag not in allowed_domains)

    # Build working relation list from registry plus optional extras.
    relations_list = list(_RELATION_REGISTRY)
    if extra_relations:
        seen = {id(rel) for rel in relations_list}
        for rel in extra_relations:
            if id(rel) in seen:
                continue
            relations_list.append(rel)
            seen.add(id(rel))

    # Return fast when no filters are requested.
    if not reactor_tags_normalized and not domain_filters and not method_names and not variable_names_set:
        if verbose:
            logger.info("No filters applied; returning all %d relations", len(relations_list))
        return list(relations_list)

    results = []
    tags_set = set(reactor_tags_normalized)

    # Build method override output set to suppress alternative relations.
    override_outputs: set[str] = set()
    if method_names:
        for rel in relations_list:
            if rel.name in method_names:
                for target in rel.outputs:
                    override_outputs.add(canonical_variable_name(target))

    # Apply domain/specific-tag/method filtering in one deterministic pass.
    for rel in relations_list:
        relation_tags = set(rel.tags)
        domains = relation_tags.intersection(allowed_domains)
        reactor_specific = relation_tags.intersection(allowed_specific)

        if domain_filters:
            if relation_tags.intersection(domain_filters):
                pass
            elif domains and not domains.intersection(domain_filters):
                if verbose:
                    logger.info(
                        "Rejecting %s: domain %s not in tags",
                        rel.name,
                        sorted(domain_filters),
                    )
                continue
            elif not domains and rel.name not in domain_filters:
                if verbose:
                    logger.info(
                        "Rejecting %s: no domain match for %s",
                        rel.name,
                        sorted(domain_filters),
                    )
                continue

        if reactor_specific and not reactor_specific.issubset(tags_set):
            if verbose:
                missing = reactor_specific - tags_set
                logger.info("Rejecting %s: missing tags %s", rel.name, sorted(missing))
            continue

        if override_outputs:
            relation_outputs = {canonical_variable_name(name) for name in rel.outputs}
            if relation_outputs.intersection(override_outputs) and rel.name not in method_names:
                if verbose:
                    logger.info("Rejecting %s: method override", rel.name)
                continue

        results.append(rel)

    # Restrict relations to the variable-connected subgraph when requested.
    if variable_names_set:
        connected_vars = set(variable_names_set)
        connected_relations: list[object] = []
        connected_relation_ids: set[int] = set()
        changed = True
        while changed:
            changed = False
            for rel in results:
                rel_id = id(rel)
                if rel_id in connected_relation_ids:
                    continue
                rel_vars = {canonical_variable_name(name) for name in getattr(rel, "symbols", ()) if name}
                if not rel_vars or rel_vars.isdisjoint(connected_vars):
                    continue
                connected_relation_ids.add(rel_id)
                connected_relations.append(rel)
                new_vars = rel_vars - connected_vars
                if new_vars:
                    connected_vars.update(new_vars)
                    changed = True
        results = connected_relations

    if verbose:
        tags_label = ", ".join(reactor_tags_normalized) if reactor_tags_normalized else "none"
        domain_label = ", ".join(sorted(domain_filters)) if domain_filters else "none"
        variable_label = ", ".join(sorted(variable_names_set)) if variable_names_set else "none"
        logger.info(
            "Filtered relations: %d of %d (domain=%s, tags=%s, variables=%s)",
            len(results),
            len(_RELATION_REGISTRY),
            domain_label,
            tags_label,
            variable_label,
        )
    return results
