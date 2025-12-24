import math
from typing import Any, Callable

WarnFunc = Callable[[str, type[Warning] | None], None]
REL_TOL_DEFAULT = 1e-2  # relative tolerance for relation checks


def require_nonzero(value: float, field_name: str, context: str = "relation checks") -> None:
    """Validate that a value is non-zero.
    
    Args:
        value: The numeric value to check.
        field_name: The name of the field being validated.
        context: A description of the context where this check is performed.
        
    Raises:
        ValueError: If the value is zero.
    """
    if value == 0:
        raise ValueError(f"{field_name} must be non-zero for {context}")


def coerce_number(value: Any, field_name: str) -> float | None:
    """Convert a value to a float, ensuring it is finite.
    
    Args:
        value: The value to convert. Can be int, float, or None.
        field_name: The name of the field for error reporting.
        
    Returns:
        The value as a float, or None if the input is None.
        
    Raises:
        ValueError: If the value is not numeric, not finite, or cannot be converted.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{field_name} must be finite for relation checks")
        return number
    raise ValueError(f"{field_name} must be numeric for relation checks; got {value!r}")
