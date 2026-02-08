"""Relation class and decorator for physics relations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

from .utils import normalize_tags_to_tuple
from .relation_util import (
    build_inverse_solver,
    build_sympy_expr,
    evaluate_constraints,
    function_inputs,
)

# Central relation registry - populated by @Relation decorator
_RELATION_REGISTRY: list['Relation'] = []


@dataclass(eq=False)
class Relation:
    """Container for a single physical relation.

    Args:
        name: Human-readable relation name.
        output: Canonical output variable name.
        func: Explicit function returning output from inputs.
        inputs: Ordered input variable names.
        tags: Relation tags for filtering.
        rel_tol_default: Default relative tolerance for checks.
        abs_tol_default: Default absolute tolerance for checks.
        constraints: Constraint expressions evaluated on variables.
        initial_guesses: Optional initial guess callables per variable.
        solve_for: Optional explicit solvers for variables.
    """

    name: str
    output: str
    func: Callable
    inputs: list[str]
    tags: tuple[str, ...] = field(default_factory=tuple)
    rel_tol_default: float | None = None
    abs_tol_default: float | None = None
    constraints: tuple[str, ...] = field(default_factory=tuple)
    initial_guesses: dict[str, Callable] = field(default_factory=dict)
    solve_for: dict[str, Callable] | None = None
    sympy_expr: object | None = None
    _sympy_symbols: dict[str, object] | None = None
    _inverse_cache: dict[str, Callable] = field(default_factory=dict, repr=False)

    @property
    def variables(self) -> tuple[str, ...]:
        """Return ordered variables (inputs + output). Args: none. Returns: tuple."""
        return tuple([*self.inputs, self.output])

    def evaluate(self, **kwargs: object) -> object:
        """Evaluate the relation function with provided input values.
        
        This method computes the output of the relation by calling the underlying
        function with the provided input values. It extracts values from kwargs
        in the correct order based on self.inputs.

        Args:
            **kwargs: Input variable values as keyword arguments.
                     Must contain all variables in self.inputs.

        Returns:
            The computed output value from the relation function.
            
        Raises:
            KeyError: If any required input variable is missing from kwargs.
            
        Example:
            >>> # For a relation: output = input1 + input2
            >>> result = relation.evaluate(input1=5.0, input2=3.0)
            >>> # result = 8.0
        """
        # Extract input values in the correct order as specified by self.inputs
        args = [kwargs[name] for name in self.inputs]
        
        # Call the underlying physics function with ordered arguments
        return self.func(*args)

    def constraint_violations(self, values: dict[str, object]) -> list[str]:
        """Check which constraints are violated for given variable values.
        
        Constraints are expressions that must be satisfied for the relation to be
        physically valid. For example, constraints might enforce that certain
        variables must be positive or within specific ranges.
        
        Args:
            values: Dictionary mapping variable names to their values.
        
        Returns:
            List of constraint expressions that evaluated to False.
            Empty list if all constraints are satisfied.
            
        Example:
            >>> # Relation with constraint "x > 0"
            >>> violations = relation.constraint_violations({"x": -1.0})
            >>> # violations = ["x > 0"]
        """
        # Delegate to utility function that evaluates each constraint expression
        return evaluate_constraints(self.constraints, values)
    
    def check_satisfied(
        self, 
        values: dict[str, object],
        *,
        rel_tol: float | None = None,
        abs_tol: float | None = None,
        check_constraints: bool = True
    ) -> tuple[bool, str, float | None]:
        """Check if relation is satisfied with given values.
        
        Convenience wrapper around check_relation_satisfied utility.
        
        Args:
            values: Variable values to check
            rel_tol: Relative tolerance (uses rel_tol_default if None)
            abs_tol: Absolute tolerance (uses abs_tol_default if None)
            check_constraints: If True, check constraints before evaluating
            
        Returns:
            Tuple of (satisfied, status, residual) where:
            - satisfied: True if relation is satisfied
            - status: "SAT", "VIOLATED", or "UNDECIDABLE"
            - residual: actual - expected value, or None if undecidable
            
        Example:
            >>> rel = some_relation
            >>> values = {"var1": 1.0, "var2": 2.0, "output": 3.0}
            >>> satisfied, status, residual = rel.check_satisfied(values)
            >>> print(f"Status: {status}, Residual: {residual}")
        """
        from .relation_util import check_relation_satisfied
        return check_relation_satisfied(self, values, 
                                       rel_tol=rel_tol, 
                                       abs_tol=abs_tol,
                                       check_constraints=check_constraints)
    
    def get_residual(self, values: dict[str, object]) -> float | None:
        """Get residual (actual - expected) for this relation.
        
        Args:
            values: Variable values
            
        Returns:
            Residual value or None if relation cannot be evaluated
            
        Example:
            >>> rel = some_relation
            >>> values = {"var1": 1.0, "var2": 2.0, "output": 3.1}
            >>> residual = rel.get_residual(values)
            >>> print(f"Residual: {residual}")  # 0.1
        """
        _, _, residual = self.check_satisfied(values, check_constraints=False)
        return residual

    def inverse_solver(self, unknown: str) -> Callable | None:
        """Get or build a symbolic inverse solver for a specific variable.
        
        This method creates a callable that can solve the relation for the specified
        unknown variable, given all other variables. The solver is built once using
        symbolic mathematics (SymPy) and then cached for future use.
        
        Args:
            unknown: Name of the variable to solve for.
        
        Returns:
            Callable that takes all other variables and returns the unknown value,
            or None if symbolic inversion is not possible.
            
        Example:
            >>> # For relation: c = a + b
            >>> solver_a = relation.inverse_solver('a')  # Returns function: a = c - b
            >>> result = solver_a(b=3.0, c=8.0)  # result = 5.0
        """
        # Check if we already built and cached this inverse solver
        if unknown in self._inverse_cache:
            return self._inverse_cache[unknown]
        
        # Can't build inverse without symbolic expression
        if self.sympy_expr is None or self._sympy_symbols is None:
            return None
        
        # Build the symbolic inverse solver using SymPy
        solver = build_inverse_solver(self.sympy_expr, self._sympy_symbols, unknown, self.variables)
        
        # Cache for future use to avoid rebuilding
        self._inverse_cache[unknown] = solver
        return solver

    def solve_for_value(self, unknown: str, values: dict[str, object]) -> object | None:
        """Compute the value of an unknown variable from known variables.
        
        This method attempts to solve the relation for the specified unknown variable
        using either an explicit solver (if provided) or automatic symbolic inversion.
        It's the primary method used by the solver system to compute missing values.

        Args:
            unknown: Name of the variable to compute.
            values: Dictionary of known variable values.

        Returns:
            The computed value for the unknown variable, or None if:
            - Required variables are missing from values
            - Solving fails (e.g., division by zero, invalid domain)
            - No solver is available (neither explicit nor symbolic)
            
        Example:
            >>> # For relation: c = a + b
            >>> result = relation.solve_for_value('a', {'b': 3.0, 'c': 8.0})
            >>> # result = 5.0
        """
        # First, try using an explicit solver if one was provided
        # Explicit solvers are hand-written functions for complex inversions
        if self.solve_for and unknown in self.solve_for:
            try:
                return self.solve_for[unknown](values)
            except Exception:
                # Explicit solver failed, fall through to symbolic
                return None
        
        # No explicit solver, try building/using symbolic inverse
        solver = self.inverse_solver(unknown)
        if solver is None:
            # Symbolic inversion not possible
            return None
        
        # Collect all known variable values in order (excluding the unknown)
        args: list[object] = []
        for name in self.variables:
            if name == unknown:
                continue  # Skip the variable we're solving for
            if name not in values:
                # Missing a required variable, can't solve
                return None
            args.append(values[name])
        
        # Attempt to evaluate the symbolic solver
        try:
            result = solver(*args)
        except Exception:
            # Solver failed (e.g., sqrt of negative, division by zero)
            return None
        
        # Try to convert result to float for numerical consistency
        try:
            scalar = float(result)
        except Exception:
            # Result is symbolic or can't be converted
            scalar = None
        
        # Return float if possible, otherwise return the raw result
        return scalar if scalar is not None else result


def Relation_decorator(
    *,
    name: str | None = None,
    output: str,
    tags: Iterable[str] | str | None = None,
    rel_tol_default: float | None = None,
    abs_tol_default: float | None = None,
    constraints: Iterable[str] | None = None,
    initial_guesses: dict[str, Callable] | None = None,
    solve_for: dict[str, Callable] | None = None,
):
    """Decorator factory to transform a physics function into a Relation object.
    
    This decorator wraps a regular Python function and creates a Relation object
    that can be used by the solver system. It automatically extracts function inputs,
    builds symbolic representations for automatic inversion, and registers the
    relation in the global registry.
    
    Args:
        name: Human-readable name for the relation. If None, uses function name.
        output: Name of the output variable that this relation computes.
        tags: Classification tags for filtering (e.g., "geometry", "fusion_power").
              Can be a single string or iterable of strings.
        rel_tol_default: Default relative tolerance for satisfaction checks.
                         Used when checking if relation is satisfied.
        abs_tol_default: Default absolute tolerance for satisfaction checks.
                         Used when checking if relation is satisfied.
        constraints: Physical constraints that must be satisfied (e.g., "x > 0").
                    Each is a string expression evaluated with variable values.
        initial_guesses: Dict of callables providing initial guesses for variables
                        when using iterative solvers.
        solve_for: Dict mapping variable names to explicit inverse solver functions.
                  Used when symbolic inversion is not possible or inefficient.
    
    Returns:
        Decorator function that transforms a function into a Relation.
        
    Example:
        >>> @Relation(
        ...     name="Sum relation",
        ...     output="c",
        ...     tags="arithmetic",
        ...     constraints=("a > 0", "b > 0")
        ... )
        ... def sum_relation(a: float, b: float) -> float:
        ...     return a + b
    """

    def decorator(func: Callable) -> Relation:
        """Inner decorator that actually creates the Relation object."""
        
        # Extract input parameter names from function signature
        inputs = function_inputs(func)
        
        # Build symbolic expression for automatic inversion using SymPy
        # This allows solving for any variable given the others
        expr, symbols = build_sympy_expr(func, inputs, output)
        
        # Create the Relation object with all metadata
        # Handle constraints: if it's a string, wrap it; if iterable, convert to tuple
        if constraints is None:
            constraints_tuple = ()
        elif isinstance(constraints, str):
            constraints_tuple = (constraints,)
        else:
            constraints_tuple = tuple(constraints)
        
        relation = Relation(
            name=name or func.__name__,  # Use provided name or fall back to function name
            output=output,
            func=func,
            inputs=inputs,
            tags=normalize_tags_to_tuple(tags),  # Convert tags to normalized tuple
            rel_tol_default=rel_tol_default,
            abs_tol_default=abs_tol_default,
            constraints=constraints_tuple,  # Ensure constraints is a tuple
            initial_guesses=initial_guesses or {},
            solve_for=solve_for,
            sympy_expr=expr,  # Symbolic representation for inversion
            _sympy_symbols=symbols,  # Symbol mapping for SymPy
        )
        
        # Preserve original function metadata for better debugging/introspection
        relation.__name__ = func.__name__
        relation.__doc__ = func.__doc__
        
        # Auto-register in global registry so it can be discovered by solver
        _RELATION_REGISTRY.append(relation)
        
        return relation

    return decorator
