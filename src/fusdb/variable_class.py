"""Variable dataclass for tracking values, metadata, and computation history.

This module provides the Variable class which stores a variable's values across
solver passes, tracks its history, and manages metadata like tolerances and units.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import math

_INPUT_MISSING = object()


@dataclass
class Variable:
    """Container for variable metadata, values, and computation history.
    
    Variables track their values across multiple solver passes, maintaining
    a complete history of updates with reasons and sources. They also store
    metadata like units, tolerances, and whether they're fixed or input values.
    
    Attributes:
        name: Canonical variable name (e.g., "R", "a", "P_fus").
        values: List of values across solver passes (chronological).
        value_passes: Pass IDs corresponding to each value (parallel to values).
        history: Complete change history with old/new values and reasons.
        unit: Physical unit string (e.g., "m", "W", "T").
        rel_tol: Relative tolerance for value change detection.
        abs_tol: Absolute tolerance for value change detection.
        method: Computation method description.
        input_source: "explicit" if value came from user input, "default" if set by defaults, else None.
        fixed: True if value should not be modified by solver.
    """

    name: str
    values: list[Any] = field(default_factory=list)
    value_passes: list[int | None] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    unit: str | None = None
    rel_tol: float | None = None
    abs_tol: float | None = None
    method: str | None = None
    input_source: str | None = None
    fixed: bool = False
    _input_value_cache: object = field(default_factory=lambda: _INPUT_MISSING, repr=False)

    @property
    def current_value(self) -> Any | None:
        """Get the most recent value for this variable.
        
        Returns the last element from the values list, or None if no values
        have been recorded yet.
        
        Returns:
            The latest value, or None if values list is empty.
        """
        return self.values[-1] if self.values else None

    @property
    def input_value(self) -> Any | None:
        """Get the original input value for this variable.
        
        Searches the history to find the value that was explicitly provided
        as input or set as a default. This is different from current_value
        which returns the latest computed value.
        
        Returns:
            The input/default value, or None if this is not an input variable
            or no input value was recorded.
        """
        if self._input_value_cache is not _INPUT_MISSING:
            return self._input_value_cache
        
        # Non-input variables don't have an input value
        if self.input_source is None:
            self._input_value_cache = None
            return None
        
        # Search history in reverse order for input/default entries
        # (more recent entries take precedence)
        for entry in reversed(self.history):
            if entry.get("reason") in ("input", "default"):
                self._input_value_cache = entry.get("new")
                return self._input_value_cache
        
        # Fallback: return the very first value if it exists
        self._input_value_cache = self.values[0] if self.values else None
        return self._input_value_cache

    def get_value_at_pass(self, pass_id: int) -> Any | None:
        """Retrieve the variable value at a specific solver pass.
        
        This method looks up what value the variable had during a particular
        solver pass. It finds the most recent value that was set at or before
        the requested pass.
        
        Args:
            pass_id: The solver pass number to query.
        
        Returns:
            The value at or before the specified pass, or None if:
            - pass_id is None
            - No values have been recorded
            - No value exists at or before the requested pass
        """
        # Can't retrieve without a valid pass_id or values
        if pass_id is None or not self.values:
            return None
        
        # If we're tracking pass IDs, use them for precise lookup
        if self.value_passes and len(self.value_passes) == len(self.values):
            # Find the latest value with pass_id <= requested pass
            for val, pid in zip(reversed(self.values), reversed(self.value_passes)):
                if pid is not None and pid <= pass_id:
                    return val
            return None
        
        # Fallback: treat pass_id as direct index into values array
        return self.values[pass_id] if 0 <= pass_id < len(self.values) else None

    def add_value(
        self,
        value: object,
        *,
        pass_id: int | None = None,
        reason: str | None = None,
        relation: str | list[str] | None = None,
        default_rel_tol: float = 0.0,
    ) -> bool:
        """Add a new value if it differs from the current value.
        
        This method intelligently compares the new value with the current value
        using tolerances. It only appends if the value has actually changed,
        preventing duplicate entries and reducing memory usage.
        
        Handles multiple value types:
        - Numeric values (compared with rel_tol/abs_tol)
        - NumPy arrays (extracted to scalars when appropriate)
        - SymPy symbolic expressions (compared symbolically or numerically)
        - Generic objects (compared with ==)
        
        Args:
            value: The new value to add.
            pass_id: Optional solver pass number.
            reason: Human-readable reason for the update (e.g., "computed", "input").
            relation: Name(s) of relation(s) that computed this value.
            default_rel_tol: Default relative tolerance if not set on variable.
        
        Returns:
            True if the value was added (it changed), False if unchanged.
        """
        # Get the current last value for comparison
        last = self.values[-1] if self.values else None
        
        # Only check for changes if we have a previous value
        if self.values:
            # Use tolerances for comparison
            rel_tol = self.rel_tol if self.rel_tol is not None else default_rel_tol
            abs_tol = self.abs_tol or 0.0
            
            # Check if values are equal within tolerance
            values_equal = False
            
            # Handle None - both None is equal
            if last is None or value is None:
                values_equal = (last is value)
            else:
                from .utils import safe_float, within_tolerance
                # Try numeric comparison first (fast path)
                try:
                    lv = safe_float(last)
                    vv = safe_float(value)
                    if lv is not None and vv is not None:
                        values_equal = within_tolerance(lv, vv, rel_tol=rel_tol, abs_tol=abs_tol)
                except Exception:
                    pass

                if not values_equal:
                    try:
                        import numpy as np
                        if isinstance(last, np.ndarray) or isinstance(value, np.ndarray):
                            values_equal = bool(np.array_equal(last, value))
                    except Exception:
                        pass

                if not values_equal:
                    try:
                        import sympy as sp
                        if isinstance(last, sp.Basic) or isinstance(value, sp.Basic):
                            try:
                                if sp.simplify(last - value) == 0:
                                    values_equal = True
                            except Exception:
                                pass
                            if not values_equal:
                                try:
                                    lv = safe_float(last.evalf() if isinstance(last, sp.Basic) else last)
                                    vv = safe_float(value.evalf() if isinstance(value, sp.Basic) else value)
                                    if lv is not None and vv is not None:
                                        values_equal = within_tolerance(lv, vv, rel_tol=rel_tol, abs_tol=abs_tol)
                                except Exception:
                                    pass
                    except Exception:
                        pass

                if not values_equal:
                    try:
                        values_equal = (last == value)
                    except Exception:
                        pass
            
            if values_equal:
                return False  # Value hasn't changed
        
        # Value is new or different - append it
        self.values.append(value)
        
        # Maintain parallel pass_id tracking if needed
        if pass_id is not None or self.value_passes:
            # Backfill with None if we're starting to track pass IDs mid-stream
            if not self.value_passes:
                self.value_passes.extend([None] * (len(self.values) - 1))
            elif len(self.value_passes) < len(self.values) - 1:
                self.value_passes.extend([None] * ((len(self.values) - 1) - len(self.value_passes)))
            self.value_passes.append(pass_id)
        
        # Record this change in the history with full metadata
        entry: dict[str, object] = {
            "pass_id": pass_id,
            "old": last,
            "new": value,
            "reason": reason or "set"
        }
        if relation is not None:
            entry["relation"] = relation
        self.history.append(entry)
        if reason in ("input", "default"):
            self._input_value_cache = value
        
        return True
    
    @classmethod
    def get_from_dict(
        cls,
        variables_dict: dict[str, Variable],
        name: str,
        pass_id: int | None = None,
        allow_override: bool = False,
        mode: str = "current"
    ) -> Any | None:
        """Get variable value from a variables dictionary with flexible retrieval modes.
        
        This is a convenience class method for retrieving variable values from
        a variables dictionary (as used in Reactor and RelationSystem). It handles
        variable name normalization and provides three retrieval modes.
        
        Args:
            variables_dict: Dictionary mapping variable names to Variable objects.
            name: Variable name to look up (aliases will be normalized to canonical name).
            pass_id: Pass index for mode="pass" - which solver pass to query.
            allow_override: When mode="current", return latest non-None computed value
                          instead of preferring the input value.
            mode: Retrieval mode:
                - "input": Return the original input/default value only.
                - "current": Return input value if available, else latest computed.
                - "pass": Return value at specific solver pass (requires pass_id).
        
        Returns:
            The variable value according to the specified mode, or None if:
            - Variable doesn't exist in the dictionary
            - No value is available for the requested mode
            
        Example:
            >>> var_dict = {"R": Variable(name="R", values=[6.2], input_source="explicit")}
            >>> Variable.get_from_dict(var_dict, "R", mode="input")
            6.2
            >>> Variable.get_from_dict(var_dict, "R", mode="current")
            6.2
        """
        from .registry import canonical_variable_name
        
        # Normalize variable name to canonical form (handles aliases)
        name = canonical_variable_name(name)
        
        # Look up the variable in the dictionary
        var = variables_dict.get(name)
        if var is None:
            return None
        
        # Mode: "input" - return original input value only
        if mode == "input":
            return var.input_value
        
        # Mode: "pass" - return value at specific solver pass
        if mode == "pass":
            return var.get_value_at_pass(pass_id)
        
        # Mode: "current" - most recent value with various behaviors
        if allow_override:
            # Return latest non-None value (ignoring whether it's input or computed)
            for value in reversed(var.values):
                if value is not None:
                    return value
            return None
        
        # Default "current" behavior: prefer input value, fall back to computed
        input_val = var.input_value
        return input_val if input_val is not None else var.current_value
