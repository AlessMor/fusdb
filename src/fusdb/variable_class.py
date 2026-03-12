"""Variable containers with scalar/profile semantics.

- `Variable` is an abstract-ish base class with common metadata and value management.
- `Variable0D` and `Variable1D` are concrete subclasses for scalar and profile variables, respectively.
"""
#TODO(low): Add history tracking of values with pass_id and reason for better debugging and analysis.
#TODO(low): Consider adding validation for unit consistency and method applicability in add_value.
#TODO(low): Consider storing pint unit instead of string and adding unit conversion utilities.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .utils import as_profile_array


@dataclass(slots=True)
class Variable:
    """Abstract base variable class with metadata and value management.

    Runtime state stores only numeric values:
    - `ndim=0`: `float`
    - `ndim=1`: `np.ndarray` profile values
    """

    name: str
    unit: str | None = None
    ndim: int = 0
    rel_tol: float | None = None
    abs_tol: float | None = None
    method: str | None = None
    input_source: str | None = None
    fixed: bool = False

    input_value: Any | None = field(default=None, init=False)
    current_value: Any | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Validate dimensionality and abstract-ish base instantiation."""
        if self.__class__ is Variable:
            raise TypeError(
                "Variable is abstract. Use Variable0D, Variable1D, or variable_util.make_variable()."
            )
        if self.ndim not in (0, 1):
            raise ValueError("Variable ndim must be 0 or 1.")

@dataclass(slots=True)
class Variable0D(Variable):
    """Scalar variable type."""

    ndim: int = 0

    def add_value(
        self,
        value: float | None,
        *,
        pass_id: int | None = None,
        reason: str | None = None,
        as_input: bool = False,
    ) -> bool:
        """Set scalar current value.

        Args:
            value: Candidate scalar value.
            pass_id: Optional solver pass identifier.
            reason: Optional solver reason label.
            as_input: When True, also store first accepted value as input.

        Returns:
            True when value changed and was stored, else False.

        Raises:
            ValueError: If value is invalid for scalar variables.
        """
        # NOTE: pass_id/reason could be used for saving history
        _ = pass_id
        _ = reason

        if value is None: # early exit for None values
            return False
        try:
            new_value = float(value)
            if not np.isfinite(new_value):
                raise ValueError
        except Exception:
            raise ValueError(f"Scalar variable '{self.name}' requires a finite numeric scalar.")

        current = self.current_value  # last stored scalar value
        same = isinstance(current, (int, float, np.floating)) and float(current) == new_value  # skip no-op write
        if not same:
            self.current_value = new_value  # commit only when value actually changed
        if as_input and self.input_value is None and self.current_value is not None:
            self.input_value = self.current_value  # snapshot first accepted value as input baseline
        return not same


@dataclass(slots=True)
class Variable1D(Variable):
    """Profile variable type storing direct numeric profile arrays.

    The profile x-axis is always normalized to ``[0, 1]`` and interpreted
    as normalized over ``coord`` (for example ``coord='a'`` means x is over
    minor radius ``a`` when ``a`` is available in solver values).
    """

    ndim: int = 1
    coord: str = "a"  # coordinate name used to de-normalize x in [0, 1]
    profile_size: int = 51

    @property
    def current_value_mean(self) -> float | None:
        """Return mean of current profile value."""
        arr = as_profile_array(self.current_value)
        return float(np.mean(arr)) if arr is not None else None

    @property
    def input_value_mean(self) -> float | None:
        """Return mean of input profile value."""
        arr = as_profile_array(self.input_value)
        return float(np.mean(arr)) if arr is not None else None

    def add_value(
        self,
        value: float | np.ndarray | None,
        *,
        pass_id: int | None = None,
        reason: str | None = None,
        as_input: bool = False,
    ) -> bool:
        """Set profile current value as a validated numeric array.

        Args:
            value: Candidate scalar mean or 1D NumPy array.
            pass_id: Optional solver pass identifier.
            reason: Optional solver reason label.
            as_input: When True, also store first accepted value as input.

        Returns:
            True when value changed and was stored, else False.

        Raises:
            ValueError: If value is invalid for profile variables.
        """
        # NOTE: pass_id/reason could be used for saving history.
        _ = pass_id
        _ = reason

        if value is None: # early exit for None values
            return False
        try: 
        # value can be a scalar mean or a profile array
            if isinstance(value, np.ndarray):
            # try to use directly a provided profile array
                new_value = as_profile_array(value)
                if new_value is None: # invalid array (non-finite, non-1D, or empty)
                    raise ValueError
            else: 
            # try to broadcast a provided scalar mean to a full profile array
                if self.profile_size < 1: # sanity check for profile size
                    raise ValueError
                scalar = float(value)
                if not np.isfinite(scalar): # sanity check for scalar value
                    raise ValueError
                new_value = np.full(int(self.profile_size), scalar, dtype=float)
        except Exception:
            raise ValueError(
                f"Profile variable '{self.name}' requires a finite float or 1D finite NumPy array."
            )

        current = self.current_value  # last stored profile value
        if isinstance(current, np.ndarray):
            same = current.shape == new_value.shape and np.array_equal(current, new_value)  # fast path for arrays
        else:
            current_arr = as_profile_array(current)  # normalize any array-like previous value for comparison
            same = (
                current_arr is not None
                and current_arr.shape == new_value.shape
                and np.array_equal(current_arr, new_value)
            )
        if not same:
            self.current_value = new_value  # commit only when profile actually changed
        if as_input and self.input_value is None and self.current_value is not None:
            self.input_value = self.current_value  # snapshot first accepted profile as input baseline
        return not same
