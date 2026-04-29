"""Variable data container with explicit scalar/profile semantics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .utils import as_profile_array


@dataclass(slots=True)
class Variable:
    """Store one variable definition and current/input values.

    Attributes:
        name: Canonical variable name.
        unit: Unit label for human inspection.
        ndim: Variable dimensionality (0 scalar, 1 profile).
        rel_tol: Relative tolerance used by solver/diagnostics.
        constraints: Hard validation constraints for this variable.
        method: Optional method tag associated with the value source.
        input_source: Optional provenance label for input values.
        fixed: Whether the solver must treat this value as immutable.
        coord: Coordinate name for profile variables.
        profile_size: Broadcast size when a profile receives one scalar input.
        input_value: First accepted value marked as input baseline.
        current_value: Latest accepted runtime value.
    """

    name: str
    unit: str | None = None
    ndim: int = 0
    rel_tol: float | None = None
    constraints: tuple[str, ...] = field(default_factory=tuple)
    method: str | None = None
    input_source: str | None = None
    fixed: bool = False
    coord: str = "a"
    profile_size: int = 51

    input_value: Any | None = field(default=None, init=False)
    current_value: Any | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Validate dimensionality and canonicalize variable name.

        Args:
            None.

        Returns:
            None.

        Raises:
            ValueError: If ``ndim`` is not 0 or 1.
        """
        from .registry import allowed_variable_constraints, canonical_variable_name

        # Normalize variable identity once so all lookups use canonical names.
        self.name = canonical_variable_name(self.name)

        # Keep dimensionality strict and explicit.
        if self.ndim not in (0, 1):
            raise ValueError("Variable ndim must be 0 or 1.")

        # Attach registry constraints when the caller did not provide explicit ones.
        if self.constraints:
            self.constraints = tuple(str(item) for item in self.constraints)
        else:
            self.constraints = tuple(allowed_variable_constraints(self.name))

    @classmethod
    def make(cls, *, ndim: int = 0, **kwargs: object) -> "Variable":
        """Create one variable instance with normalized dimensionality.

        Args:
            ndim: Variable dimensionality (0 scalar, 1 profile).
            **kwargs: Additional Variable dataclass fields.

        Returns:
            One initialized Variable object.

        Raises:
            ValueError: If dimensionality is unsupported or inconsistent.
        """
        # Normalize explicit ndim first.
        try:
            normalized_ndim = int(ndim)
        except Exception as exc:
            raise ValueError(
                f"Unsupported variable ndim={ndim}. Supported values are 0 and 1."
            ) from exc

        # Resolve possible duplicate ndim passed through kwargs.
        payload = dict(kwargs)
        provided_ndim = payload.pop("ndim", normalized_ndim)
        try:
            provided_ndim = int(provided_ndim)
        except Exception as exc:
            raise ValueError(
                f"Unsupported variable ndim={provided_ndim}. Supported values are 0 and 1."
            ) from exc
        if provided_ndim != normalized_ndim:
            raise ValueError(
                f"Conflicting ndim values: ndim={normalized_ndim} but kwargs['ndim']={provided_ndim}."
            )

        # Build one unified Variable instance.
        return cls(**payload, ndim=normalized_ndim)

    def add_value(
        self,
        value: float | np.ndarray | None,
        *,
        pass_id: int | None = None,
        reason: str | None = None,
        as_input: bool = False,
    ) -> bool:
        """Set current value with explicit ndim-aware validation.

        Args:
            value: Candidate scalar or profile payload.
            pass_id: Optional solver pass identifier.
            reason: Optional solver reason label.
            as_input: When ``True``, store first accepted value in ``input_value``.

        Returns:
            ``True`` when value changed, ``False`` otherwise.

        Raises:
            ValueError: If ``value`` cannot be validated for this variable.
        """
        # Keep solver context arguments available for future history tracking.
        _ = pass_id
        _ = reason

        # Ignore missing writes explicitly.
        if value is None:
            return False

        # Validate and normalize the incoming payload in one ndim-aware branch.
        if self.ndim == 0:
            try:
                new_value = float(value)
                if not np.isfinite(new_value):
                    raise ValueError
            except Exception as exc:
                raise ValueError(
                    f"Scalar variable '{self.name}' requires a finite numeric scalar."
                ) from exc

            # Detect no-op scalar rewrites before mutating state.
            current = self.current_value
            same = (
                isinstance(current, (int, float, np.floating))
                and float(current) == new_value
            )
        else:
            arr = as_profile_array(value)
            if arr is not None:
                new_value = arr
            else:
                try:
                    scalar = float(value)
                    if not np.isfinite(scalar):
                        raise ValueError
                    if self.profile_size < 1:
                        raise ValueError
                except Exception as exc:
                    raise ValueError(
                        f"Profile variable '{self.name}' requires a finite float or 1D finite NumPy array."
                    ) from exc
                new_value = np.full(int(self.profile_size), scalar, dtype=float)

            # Compare profiles without implicit reductions before mutating state.
            current = self.current_value
            if isinstance(current, np.ndarray):
                same = current.shape == new_value.shape and np.array_equal(current, new_value)
            else:
                current_arr = as_profile_array(current)
                same = (
                    current_arr is not None
                    and current_arr.shape == new_value.shape
                    and np.array_equal(current_arr, new_value)
                )

        # Commit only real state changes.
        if not same:
            self.current_value = new_value
        changed = not same

        # Snapshot first accepted value as input baseline when requested.
        if as_input and self.input_value is None and self.current_value is not None:
            self.input_value = self.current_value

        return changed
