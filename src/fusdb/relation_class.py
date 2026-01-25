"""
Relation class and solver for fusdb.

This module provides the core infrastructure for defining and solving
implicit relations between reactor parameters. A Relation represents
a mathematical equation (as a zero-residual sympy expression) that
connects multiple physical variables. The RelationSystem class orchestrates
the solving of multiple interdependent relations, tracking parameter
provenance, handling conflicts, and issuing warnings when constraints
cannot be satisfied.

Key concepts:
- Relation: A mathematical equation connecting multiple variables (e.g., P = I² R)
- RelationSystem: A solver that resolves a set of relations given known inputs
- Parameter Records: Track the origin and status of each resolved parameter
"""
from __future__ import annotations
import math
import warnings
import re
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence
import networkx as nx
import sympy as sp
from sympy.core.relational import Relational
from fusdb.relation_util import (
    REL_TOL_DEFAULT, compile_constraints, constraints_ok, extract_constraint_bounds,
    numeric_value, relation_residual, solve_for_variable, solve_linear_system,
    solve_numeric_system, symbol, update_value,
)

# Type alias for warning callback functions
WarnFunc = Callable[[str, type[Warning] | None], None]


def _rec(inp: float | None = None, deps: set | None = None) -> dict:
    """
    Create a new parameter record dictionary.
    
    Parameter records track the provenance and status of each resolved parameter:
    - input_value: The original value provided as input (None if computed)
    - final_value: The value after solving (may differ from input if overridden)
    - computed_value: Alternative computed value when conflicts occur
    - explicit_deps: Set of explicit input parameters this value depends on
    - status: One of 'consistent', 'constraint', 'conflict', 'inconsistent'
    - constraint_limit: The constraint bound that was violated (if any)
    
    Args:
        inp: The input value (if this parameter was explicitly provided)
        deps: Set of explicit parameters this value depends on
        
    Returns:
        A dictionary with the parameter record fields initialized
    """
    return {
        "input_value": inp,
        "final_value": None,
        "computed_value": None,
        "explicit_deps": deps or set(),
        "status": "consistent",
        "constraint_limit": None
    }


@dataclass(frozen=False, eq=False)
class Relation:
    """
    Represents an implicit relation defined by a zero-residual Sympy expression.
    
    A Relation encapsulates a mathematical equation connecting multiple variables.
    The equation is represented in implicit form: f(x₁, x₂, ..., xₙ) = 0.
    For example, the power relation P = V × I becomes: P - V × I = 0.
    
    Attributes:
        name: Human-readable name for the relation (e.g., "Ohm's Law")
        variables: Tuple of variable names involved in the relation
        expr: Sympy expression representing the zero-residual equation
        rel_tol: Relative tolerance for considering the relation satisfied
        solve_for: Tuple of variables that can be solved for (defaults to all)
        initial_guesses: Mapping of variable names to initial guess values
        constraints: Tuple of constraints (e.g., "P >= 0") on the variables
        
    Computed attributes (set in __post_init__):
        solve_targets: Variables the solver should try to compute
        syms: Tuple of Sympy symbols for each variable
        residual_fn: Compiled function for fast residual evaluation
        constraints_compiled: Pre-compiled constraints for fast checking
        
    Example:
        >>> rel = Relation(
        ...     name="Power equation",
        ...     variables=("P", "V", "I"),
        ...     expr=sp.Symbol("P") - sp.Symbol("V") * sp.Symbol("I"),
        ...     constraints=("P >= 0",)
        ... )
    """
    name: str
    variables: tuple[str, ...]
    expr: sp.Expr
    rel_tol: float = REL_TOL_DEFAULT
    solve_for: tuple[str, ...] | None = None
    initial_guesses: Mapping[str, float | Callable[[Mapping[str, float]], float]] | None = None
    constraints: tuple[Relational | str, ...] = ()
    
    # Computed fields (populated in __post_init__)
    solve_targets: tuple[str, ...] = field(init=False)
    syms: tuple[sp.Symbol, ...] = field(init=False)
    residual_fn: Callable[..., object] | None = field(init=False)
    constraints_compiled: tuple[tuple[tuple[str, ...], Callable[..., object] | None, Relational], ...] = field(init=False)
    
    def __hash__(self):
        """Use object identity for hashing (enables use in sets/dicts)."""
        return id(self)
    
    def __post_init__(self):
        """
        Initialize computed fields after dataclass construction.
        
        This method:
        1. Converts variables to a tuple (ensuring immutability)
        2. Sympifies the expression (if it was passed as a string)
        3. Sets up solve targets (defaults to all variables)
        4. Creates Sympy symbols for each variable
        5. Compiles the residual function for fast evaluation
        6. Compiles constraints for fast checking
        """
        # Ensure variables is a tuple for immutability
        self.variables = tuple(self.variables)
        
        # Convert expression to Sympy form
        self.expr = sp.sympify(self.expr)
        
        # Set solve targets (which variables can be computed)
        self.solve_targets = self.solve_for or self.variables
        
        # Create Sympy symbols for each variable
        self.syms = tuple(symbol(n) for n in self.variables)
        
        # Try to compile a fast residual evaluation function
        try:
            self.residual_fn = sp.lambdify(self.syms, self.expr, "math")
        except Exception:
            self.residual_fn = None
        
        # Compile constraints for fast evaluation
        self.constraints_compiled = compile_constraints(self.constraints, self.variables)


class RelationSystem:
    """
    Solve a set of relations, tracking parameter provenance and conflicts.
    
    RelationSystem orchestrates the resolution of multiple interdependent
    Relation objects. It:
    - Iteratively solves for unknown variables
    - Tracks which explicit inputs each computed value depends on
    - Detects and reports conflicts when relations are inconsistent
    - Handles constraint violations gracefully
    - Records parameter provenance for debugging
    
    The solver supports two modes:
    - override_input: Allows relations to override explicit inputs when necessary
    - locked_input: Raises an error if relations cannot be satisfied with given inputs
    
    Attributes:
        relations: The relations to solve (current stage)
        all_relations: All registered relations (for alternative value computation)
        rel_tol: Relative tolerance for considering equations satisfied
        _warn: Callback function for issuing warnings
        _warnings_issued: Set of warning messages already issued (deduplication)
        _explicit_deps: Maps each variable to the set of explicit inputs it depends on
        _records: Parameter records tracking provenance and status
        
    Example:
        >>> sys = RelationSystem([rel1, rel2, rel3])
        >>> values = sys.solve({"P": 100.0}, mode="override_input")
        >>> print(values)
        {'P': 100.0, 'V': 10.0, 'I': 10.0}
    """
    
    def __init__(
        self,
        relations: Sequence[Relation],
        *,
        rel_tol: float = REL_TOL_DEFAULT,
        warn: WarnFunc = warnings.warn,
        warnings_issued: set[str] | None = None,
        all_relations: Sequence[Relation] | None = None
    ):
        """
        Initialize the RelationSystem.
        
        Args:
            relations: Sequence of Relation objects to solve (current stage)
            rel_tol: Relative tolerance for equation satisfaction
            warn: Function to call for issuing warnings
            warnings_issued: Set of already-issued warnings (for deduplication)
            all_relations: All registered relations (for alternative computations)
        """
        self.relations = tuple(relations)
        self.all_relations = tuple(all_relations or relations)
        self.rel_tol = rel_tol
        self._warn = warn
        self._warnings_issued = warnings_issued if warnings_issued is not None else set()
        self._explicit_deps: dict[str, set[str]] = {}
        self._records: dict[str, dict] = {}

    def solve(
        self,
        values: Mapping[str, float] | None = None,
        *,
        mode: str = "override_input",
        tol: float | None = None,
        max_iter: int = 50,
        explicit: set[str] | None = None
    ) -> dict[str, float]:
        """
        Solve the system of relations for unknown variables.
        
        This is the main entry point for solving. Given a set of known values,
        it iteratively solves for unknown variables using the registered relations.
        
        Args:
            values: Dictionary of known variable values
            mode: Either 'override_input' (allow overriding inputs) or 
                  'locked_input' (error if inputs can't be satisfied)
            tol: Tolerance for equation satisfaction (defaults to rel_tol)
            max_iter: Maximum number of iteration passes
            explicit: Set of variable names that are considered explicit inputs
                     (defaults to all provided values)
                     
        Returns:
            Dictionary of all resolved variable values
            
        Raises:
            ValueError: If mode is invalid or locked_input mode fails
        """
        # Convert input values to floats
        vals = {k: float(v) for k, v in (values or {}).items()}
        
        # Determine which parameters are explicit inputs
        exp_set = set(vals) if explicit is None else set(explicit) & set(vals)
        exp_vals = {n: vals[n] for n in exp_set}
        
        # Validate mode parameter
        if mode not in {"override_input", "locked_input"}:
            raise ValueError("mode must be 'override_input' or 'locked_input'")
        override = mode == "override_input"
        tol_use = self.rel_tol if tol is None else float(tol)
        
        # In override mode, nothing is locked; in locked mode, explicit inputs are locked
        locked = set() if override else exp_set
        
        # Initialize dependency tracking for explicit inputs
        self._explicit_deps = {n: {n} for n in exp_set}
        
        # Initialize parameter records for explicit inputs
        self._records = {n: _rec(exp_vals[n], {n}) for n in exp_set}
        
        # Get all variables involved in the relations
        all_vars = sorted({v for r in self.relations for v in r.variables})
        
        # First pass: solve iteratively, suppressing warnings in override mode
        self._solve_iter(all_vars, vals, locked, max_iter, exp_set, suppress=override)
        
        # Check for constraint violations
        viols = self._validate(vals, tol_use, exp_set)
        
        # In override mode, if violations exist, try removing conflicting explicit values
        if override and viols:
            for rel, _ in viols:
                # Remove one explicit value from the violating relation
                for t in (rel.solve_for or rel.variables):
                    if t in vals and t in exp_set:
                        del vals[t]
                        break
            # Re-solve with the explicit values removed
            self._solve_iter(all_vars, vals, set(), max_iter, exp_set, suppress=False)
        
        # Final validation
        viols = self._validate(vals, tol_use, exp_set)
        
        if viols:
            if not override:
                # In locked mode, raise an error with details
                lines = ["Unable to satisfy relations with locked inputs:"]
                for rel, det in viols:
                    rel_exp = [n for n in rel.variables if n in exp_set]
                    exp_str = ', '.join(f"{n}={exp_vals.get(n, '?')}" for n in rel_exp) or 'none'
                    lines.append(f"- {rel.name}: {det}")
                    lines.append(f"  explicit inputs: {exp_str}")
                raise ValueError("\n".join(lines))
            
            # In override mode, issue warnings and update records
            for rel, det in viols:
                if det not in self._warnings_issued:
                    self._warnings_issued.add(det)
                    self._warn(det, UserWarning)
                self._update_records(rel, det, vals, exp_set)
        
        # Warn if explicit values were overridden
        if override:
            for n, ev in exp_vals.items():
                sv = vals.get(n)
                if sv and abs(sv - ev) > tol_use * max(abs(ev), abs(sv), 1.0):
                    self._warn(f"Explicit {n} overridden: explicit={ev}, solved={sv}", UserWarning)
        
        # Finalize parameter records
        self._finalize(vals, exp_set)
        
        return vals

    def get_parameter_records(self) -> dict[str, dict]:
        """
        Get the parameter records after solving.
        
        Returns:
            Dictionary mapping variable names to their parameter records
        """
        return dict(self._records)

    def _solve_iter(self, all_vars, vals, locked, max_iter, exp_set, *, suppress):
        """
        Iteratively solve for unknown variables.
        
        This method repeatedly calls _solve_sz with increasing block sizes
        until no more progress can be made. It starts with single-variable
        solutions (sz=1) and progresses to larger blocks when needed.
        
        Args:
            all_vars: List of all variable names in the system
            vals: Current dictionary of known values (modified in place)
            locked: Set of variable names that cannot be modified
            max_iter: Maximum number of outer iteration passes
            exp_set: Set of explicit input variable names
            suppress: If True, suppress constraint warnings
        """
        for _ in range(max_iter):
            prog = False
            # Try solving blocks of increasing size (1 to 7 variables)
            for sz in range(1, 8):
                while (p := self._solve_sz(all_vars, vals, locked, sz, max_iter, exp_set, suppress)):
                    prog = True
                    # If we made progress with sz > 1, restart from sz=1
                    sz > 1 and None
                if prog and sz > 1:
                    break
            if not prog:
                break

    def _solve_sz(self, all_vars, vals, locked, sz, max_iter, exp_set, suppress) -> bool:
        """
        Attempt to solve blocks of 'sz' relations with 'sz' unknowns.
        
        This method finds relations where exactly 'sz' variables are unknown
        and tries to solve them as a system.
        
        Args:
            all_vars: List of all variable names
            vals: Current dictionary of known values
            locked: Set of locked variable names
            sz: Block size (number of unknowns to solve simultaneously)
            max_iter: Maximum iterations for numeric solving
            exp_set: Set of explicit input names
            suppress: If True, suppress constraint warnings
            
        Returns:
            True if at least one variable was solved
        """
        # Build mapping from relation to its unknown variables
        rel_unk = {
            r: [v for v in r.variables if v in all_vars and v not in vals and v not in locked]
            for r in self.relations
        }
        rel_unk = {r: u for r, u in rel_unk.items() if u}
        
        if not rel_unk:
            return False
        
        # For single-variable case, solve directly
        if sz == 1:
            prog = False
            for r, unk in rel_unk.items():
                if len(unk) == 1:
                    res = self._solve1(r, unk[0], vals, suppress)
                    if res is not None:
                        vals[unk[0]] = res
                        self._track(unk[0], r, vals, exp_set)
                        prog = True
            return prog
        
        # For small systems, try all combinations
        prog = False
        rels = list(rel_unk.keys())
        
        if sz <= 3 and len(rels) <= 20:
            from itertools import combinations
            for sub in combinations(rels, sz):
                uvars = set()
                [uvars.update(rel_unk[r]) for r in sub]
                if len(uvars) == sz and self._solve_blk(list(sub), list(uvars), vals, max_iter):
                    prog = True
        else:
            # For larger systems, use graph-based component detection
            g = nx.Graph()
            for r, u in rel_unk.items():
                g.add_node(r, bipartite=0)
                [g.add_edge(r, v) for v in u]
            for comp in nx.connected_components(g):
                cr = [n for n in comp if isinstance(n, Relation)]
                cv = [n for n in comp if isinstance(n, str)]
                if len(cr) == sz == len(cv):
                    if self._solve_blk(cr, cv, vals, max_iter):
                        prog = True
        
        return prog

    def _solve1(self, rel, tgt, vals, suppress) -> float | None:
        """
        Solve a single relation for one unknown variable.
        
        First attempts symbolic solution, then falls back to numeric methods
        if needed.
        
        Args:
            rel: The Relation to solve
            tgt: The target variable name to solve for
            vals: Current dictionary of known values
            suppress: If True, suppress constraint warnings
            
        Returns:
            The computed value, or None if solution failed
        """
        # Try symbolic solution with constraints
        res = solve_for_variable(rel.expr, tgt, vals, rel.variables, rel.constraints_compiled)
        if res is not None:
            return res
        
        # Try symbolic solution without constraints (to detect violations)
        uc = solve_for_variable(rel.expr, tgt, vals, rel.variables, [], check_constraints=False)
        if uc is not None and not suppress:
            self._constraint_warn(rel, tgt, uc, vals)
        
        # Fall back to numeric solution
        return self._num_solve1(rel, tgt, vals)

    def _constraint_warn(self, rel, tgt, computed, vals):
        """
        Issue a warning when a constraint is violated.
        
        Called when symbolic solution succeeds but violates constraints.
        Updates the parameter record and issues a descriptive warning.
        
        Args:
            rel: The Relation that produced the violation
            tgt: The target variable that violates constraints
            computed: The computed value that violates constraints
            vals: Current dictionary of known values
        """
        # Build test values dictionary
        test = dict(vals)
        test[tgt] = computed
        failed = []
        lim = None
        
        # Check each constraint
        for names, fn, con in rel.constraints_compiled:
            if tgt not in names:
                continue
            try:
                if fn and fn(*[test.get(n, 0) for n in names]) is False:
                    s = str(con)
                    failed.append(s)
                    # Extract the constraint limit
                    for op in ('<=', '>='):
                        if m := re.search(rf'{tgt}\s*{op}\s*([\d.e+-]+)', s):
                            lim = float(m.group(1))
                            break
            except Exception:
                pass
        
        if not failed:
            return
        
        # Update parameter record
        if tgt in self._records:
            self._records[tgt]["status"] = "constraint"
            self._records[tgt]["constraint_limit"] = lim or self._records[tgt]["constraint_limit"]
        
        # Build and issue warning message
        inp = [v for v in rel.variables if v != tgt and v in vals]
        inp_s = f" from inputs ({', '.join(f'{v}={vals[v]:.4g}' for v in inp)})" if inp else ""
        msg = f"{rel.name}: cannot solve for {tgt} = {computed:.4g}{inp_s} (violates constraints: {', '.join(failed)})"
        
        if msg not in self._warnings_issued:
            self._warnings_issued.add(msg)
            self._warn(msg, UserWarning)

    def _num_solve1(self, rel, tgt, vals) -> float | None:
        """
        Numerically solve a single relation for one variable.
        
        Uses scipy.optimize.root with multiple initial guesses to find
        a solution that satisfies the relation and its constraints.
        
        Args:
            rel: The Relation to solve
            tgt: The target variable name
            vals: Current dictionary of known values
            
        Returns:
            The computed value, or None if no valid solution found
        """
        from scipy.optimize import root
        
        # Define residual function for root finder
        def rfn(x):
            t = dict(vals)
            t[tgt] = x[0]
            r = relation_residual(t, rel.variables, rel.residual_fn, rel.expr)
            return [r if r is not None else 1e10]
        
        # Extract constraint bounds for the target variable
        lo, hi = extract_constraint_bounds(rel.constraints_compiled, tgt)
        
        # Build list of initial guesses
        gs = [1.0]
        
        # Use provided initial guess if available
        if rel.initial_guesses and tgt in rel.initial_guesses:
            g = rel.initial_guesses[tgt]
            try:
                gs[0] = float(g(dict(vals)) if callable(g) else g)
            except Exception:
                pass
        
        # Add guesses based on constraint bounds
        if math.isfinite(lo) and math.isfinite(hi):
            gs.extend([(lo + hi) / 2, lo + (hi - lo) * 0.25, lo + (hi - lo) * 0.75])
        elif math.isfinite(lo):
            gs.extend([lo + 1, lo + 10])
        elif math.isfinite(hi):
            gs.extend([hi - 1, hi - 10])
        else:
            gs.extend([0.1, 10.0, 100.0])
        
        # Try each initial guess
        for g in gs:
            if not math.isfinite(g):
                continue
            try:
                r = root(rfn, [g], method='hybr')
                if r.success and math.isfinite(float(r.x[0])):
                    t = dict(vals)
                    t[tgt] = float(r.x[0])
                    if constraints_ok(rel.constraints_compiled, t, focus_names={tgt}):
                        return float(r.x[0])
            except Exception:
                pass
        
        return None

    def _solve_blk(self, rels, unk, vals, max_iter) -> bool:
        """
        Solve a block of relations for multiple unknowns simultaneously.
        
        First attempts linear algebraic solution, then falls back to
        numeric methods for nonlinear systems.
        
        Args:
            rels: List of Relation objects to solve together
            unk: List of unknown variable names
            vals: Current dictionary of known values
            max_iter: Maximum iterations for numeric solving
            
        Returns:
            True if the block was successfully solved
        """
        if not unk:
            return False
        
        # Handle trivial case of single unknown
        if len(unk) == 1 == len(rels):
            r = self._solve1(rels[0], unk[0], vals, True)
            if r is not None:
                vals[unk[0]] = r
                return True
            return False
        
        # Create symbols and substitute known values
        usym = [symbol(n) for n in unk]
        uset = set(usym)
        ksubs = {
            symbol(n): sp.Float(vals[n])
            for rel in rels for n in rel.variables
            if n not in unk and n in vals
        }
        
        # Build substituted equations
        eqs = [
            e for rel in rels
            if (e := rel.expr.subs(ksubs)) != 0 and e.free_symbols and e.free_symbols.issubset(uset)
        ]
        
        if len(eqs) < len(usym):
            return False
        
        # Try linear algebraic solution first
        sol = solve_linear_system(eqs, usym)
        
        # Fall back to numeric solution for small nonlinear systems
        if sol is None and len(usym) <= 6:
            gs = self._guesses(rels, unk, vals)
            sv = solve_numeric_system(eqs, usym, gs, max_iter=max_iter)
            if sv and len(sv) == len(usym):
                sol = dict(zip(usym, map(sp.Float, sv)))
        
        if sol is None:
            return False
        
        # Extract numeric values from solution
        solved = {}
        for sym, expr in sol.items():
            if expr.free_symbols:
                return False
            n = numeric_value(expr)
            if n is None or not math.isfinite(n):
                return False
            solved[sym.name] = float(n)
        
        # Verify constraints are satisfied
        for rel in rels:
            t = {**{n: vals[n] for n in rel.variables if n in vals}, **solved}
            if not constraints_ok(rel.constraints_compiled, t, focus_names=set(solved)):
                return False
        
        # Update values
        for n, v in solved.items():
            update_value(vals, n, v)
        
        return True

    def _guesses(self, rels, unk, vals) -> list[float]:
        """
        Generate initial guesses for numeric solving.
        
        Collects initial guesses from relation definitions and
        current known values.
        
        Args:
            rels: List of Relation objects being solved
            unk: List of unknown variable names
            vals: Current dictionary of known values
            
        Returns:
            List of initial guess values (one per unknown)
        """
        # Start with known values if available
        gv = {n: float(vals[n]) for n in unk if n in vals and math.isfinite(vals[n])}
        
        # Add guesses from relation definitions
        for rel in rels:
            if not rel.initial_guesses:
                continue
            ctx = {k: float(v) for k, v in vals.items() if math.isfinite(v)}
            ctx.update(gv)
            for n, gs in rel.initial_guesses.items():
                if n in unk and n not in gv:
                    try:
                        g = gs(ctx) if callable(gs) else gs
                        gv[n] = float(g) if isinstance(g, (int, float)) and math.isfinite(float(g)) else 1.0
                    except Exception:
                        pass
        
        return [gv.get(n, 1.0) for n in unk]

    def _track(self, tgt, rel, vals, exp_set):
        """
        Track the explicit dependencies for a computed variable.
        
        Updates _explicit_deps to record which explicit input parameters
        the computed value depends on (transitively).
        
        Args:
            tgt: The target variable that was computed
            rel: The Relation used to compute it
            vals: Current dictionary of known values
            exp_set: Set of explicit input variable names
        """
        deps = set()
        for v in rel.variables:
            if v == tgt:
                continue
            # If v is explicit, add it directly; otherwise add its dependencies
            if v in exp_set:
                deps.add(v)
            else:
                deps.update(self._explicit_deps.get(v, set()))
        self._explicit_deps[tgt] = deps

    def _validate(self, vals, tol, exp_set) -> list[tuple[Relation, str]]:
        """
        Validate all relations and report violations.
        
        Checks each relation to ensure it is satisfied within tolerance
        and that all constraints are met.
        
        Args:
            vals: Dictionary of computed values
            tol: Tolerance for residual check
            exp_set: Set of explicit input variable names
            
        Returns:
            List of (Relation, detail_message) tuples for violations
        """
        viols = []
        for rel in self.relations:
            # Skip if not all variables are available
            if any(n not in vals for n in rel.variables):
                continue
            
            # Compute residual
            res = relation_residual(vals, rel.variables, rel.residual_fn, rel.expr)
            rt = rel.rel_tol if rel.rel_tol is not None else tol
            sc = max(max(abs(vals[n]) for n in rel.variables), 1.0)
            
            # Check constraints
            cok = constraints_ok(rel.constraints_compiled, vals)
            
            # Record violation if residual too large or constraints violated
            if res is None or abs(res) > rt * sc or not cok:
                viols.append((rel, self._fmt_viol(rel, vals, res, rt * sc, cok, exp_set)))
        
        return viols

    def _fmt_viol(self, rel, vals, res, tol, cok, exp_set) -> str:
        """
        Format a human-readable violation message.
        
        Creates a message that directly points to the explicit input
        parameters that are in conflict, rather than intermediate
        computed values.
        
        Args:
            rel: The Relation that is violated
            vals: Dictionary of computed values
            res: The computed residual (or None)
            tol: Tolerance threshold
            cok: Whether constraints are satisfied
            exp_set: Set of explicit input variable names
            
        Returns:
            Human-readable violation message
        """
        # Find all explicit sources that this relation depends on
        srcs = set()
        for v in rel.variables:
            if v in exp_set:
                srcs.add(v)
            else:
                srcs.update(self._explicit_deps.get(v, set()))
        
        # If constraints violated, provide specific constraint info
        if not cok:
            for names, fn, con in rel.constraints_compiled:
                try:
                    if fn and fn(*[vals.get(n, 0) for n in names]) is False:
                        for n in names:
                            if n in exp_set:
                                return f"{n}={vals.get(n, 0):.3e} violates constraint: {con}"
                            elif n in self._explicit_deps:
                                ds = sorted(self._explicit_deps[n])[:2]
                                return f"{n}={vals.get(n, 0):.3e} (from {', '.join(f'{d}={vals.get(d, 0):.3e}' for d in ds)}) violates: {con}"
                except Exception:
                    pass
        
        # Format message pointing to explicit inputs
        if len(srcs) >= 2:
            return f"Inputs {', '.join(f'{s}={vals.get(s, 0):.3e}' for s in sorted(srcs)[:4])} are inconsistent"
        if srcs:
            s = list(srcs)[0]
            return f"{s}={vals.get(s, 0):.3e} inconsistent with {rel.name}"
        
        return f"{rel.name} not satisfied ({', '.join(f'{v}={vals.get(v, 0):.3e}' for v in rel.variables[:4] if v in vals)})"

    def _update_records(self, rel, det, vals, exp_set):
        """
        Update parameter records when a violation is detected.
        
        Parses the violation message to identify affected variables
        and updates their status appropriately.
        
        Args:
            rel: The Relation that is violated
            det: The detail message describing the violation
            vals: Dictionary of computed values
            exp_set: Set of explicit input variable names
        """
        is_con = "violates" in det.lower()
        mentioned = set()
        clim = None
        
        # Extract variable names from the detail message
        for vm in re.finditer(r'(\w+)=[\d.e+-]+', det):
            mentioned.add(vm.group(1))
        
        # Extract constraint limit if present
        for op in ('<=', '>='):
            if m := re.search(rf'(\w+)\s*{op}\s*([\d.e+-]+)', det):
                mentioned.add(m.group(1))
                clim = float(m.group(2))
                break
        
        # Update records for mentioned explicit variables
        for v in mentioned & exp_set:
            if v not in self._records:
                self._records[v] = _rec(vals.get(v), self._explicit_deps.get(v, set()))
            r = self._records[v]
            if is_con:
                r["status"] = "constraint"
                r["constraint_limit"] = clim or r["constraint_limit"]
            elif r["status"] != "constraint":
                r["status"] = "conflict"
                self._alt(v, vals, exp_set)

    def _alt(self, var, vals, exp_set):
        """
        Compute an alternative value for a conflicting variable.
        
        When a conflict is detected, this method tries to compute what
        value the variable would have if derived from other relations,
        storing it in the parameter record for diagnostic purposes.
        
        This is purely diagnostic and should not block the solver.
        We skip complex expressions to avoid expensive symbolic solving.
        
        Args:
            var: The variable to compute an alternative for
            vals: Dictionary of computed values
            exp_set: Set of explicit input variable names
        """
        # Build independent values (excluding var and its dependents)
        ind = {
            k: v for k, v in vals.items()
            if k != var and (k in exp_set or var not in self._explicit_deps.get(k, set()))
        }
        
        # Try direct computation from current stage relations
        # Only attempt SIMPLE relations to avoid expensive symbolic solving
        attempts = 0
        max_attempts = 5  # Limit attempts to avoid slow diagnostics
        
        for rel in self.relations:
            if attempts >= max_attempts:
                break
            if var not in rel.variables:
                continue
            # Skip relations where var is not a preferential solve target
            if rel.solve_for and var not in rel.solve_for:
                continue
            # Skip relations with many variables (likely complex)
            if len(rel.variables) > 4:
                continue
            if all(v in ind for v in rel.variables if v != var):
                attempts += 1
                try:
                    c = solve_for_variable(rel.expr, var, ind, rel.variables, [], check_constraints=False)
                    if c is not None and abs(c - vals.get(var, 0)) > 1e-6 * max(abs(c), 1):
                        self._records[var]["computed_value"] = c
                        return
                except Exception:
                    pass  # Skip if solving fails

    def _finalize(self, vals, exp_set):
        """
        Finalize parameter records after solving.
        
        Ensures all computed variables have records and final values
        are properly set.
        
        Args:
            vals: Dictionary of final computed values
            exp_set: Set of explicit input variable names
        """
        # Update final values for explicit inputs
        for n in exp_set:
            if n in self._records:
                self._records[n]["final_value"] = vals.get(n)
        
        # Create records for derived variables
        for v, val in vals.items():
            if v not in exp_set and v not in self._records:
                self._records[v] = _rec()
                self._records[v]["final_value"] = val
                self._records[v]["explicit_deps"] = self._explicit_deps.get(v, set())
