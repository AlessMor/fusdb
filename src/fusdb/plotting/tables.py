"""Table rendering helpers for multi-reactor comparison views."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import numbers

import numpy as np
import pandas as pd

from fusdb.reactor_class import Reactor
from fusdb.utils import as_profile_array, safe_float, within_tolerance

DEFAULT_METADATA_FIELDS: tuple[str, ...] = (
    "id",
    "name",
    "organization",
    "country",
    "tags",
    "solve_status",
    "year",
    "doi",
    "notes",
)


def _normalize_reactors(
    reactors: Mapping[str, Reactor] | Iterable[Reactor],
) -> list[tuple[str, Reactor]]:
    """Return one normalized ``[(reactor_id, reactor), ...]`` list.

    Args:
        reactors: Mapping or iterable of Reactor objects.

    Returns:
        Ordered list of reactor identifier/object pairs.
    """
    # Keep mapping keys when provided explicitly.
    if isinstance(reactors, Mapping):
        return [(str(name), reactor) for name, reactor in reactors.items()]

    # Otherwise derive IDs from reactor metadata.
    out: list[tuple[str, Reactor]] = []
    for idx, reactor in enumerate(reactors):
        rid = reactor.id or reactor.name or f"reactor_{idx + 1}"
        out.append((str(rid), reactor))
    return out


def _ordered_variable_names(payloads: Mapping[str, dict[str, object]]) -> list[str]:
    """Return variable row names in registry order with unknown extras appended.

    Args:
        payloads: Per-reactor table payload mapping.

    Returns:
        Ordered variable names.
    """
    # Collect all variable names used by the provided reactors.
    present_names = {
        name
        for payload in payloads.values()
        for name in (payload.get("variables", {}) or {})
    }
    if not present_names:
        return []

    # Read canonical registry order once and keep only present names.
    from fusdb.registry import load_allowed_variables

    allowed_variables, _, _ = load_allowed_variables()
    ordered = [name for name in allowed_variables if name in present_names]
    ordered_set = set(ordered)

    # Append non-registry names deterministically at the end.
    extras = sorted(
        (name for name in present_names if name not in ordered_set),
        key=lambda name: (name.lower(), name),
    )
    return [*ordered, *extras]


def _numeric_value(value: object) -> float | None:
    """Return one comparable scalar value for tolerance checks.

    Args:
        value: Scalar/profile payload.

    Returns:
        Comparable scalar value or ``None`` when unavailable.
    """
    # Reduce profiles explicitly to their mean value for comparisons.
    profile = as_profile_array(value)
    if profile is not None:
        return float(np.mean(profile))

    # Convert scalar-like payloads when possible.
    scalar = safe_float(value)
    if scalar is not None:
        return scalar

    # Accept symbolic numeric scalars when available.
    try:
        import sympy as sp
    except Exception:
        sp = None
    if sp is not None and isinstance(value, sp.Expr) and value.is_number:
        return safe_float(value.evalf())
    return None


def _values_match(left: object, right: object, *, eps: float = 1e-12) -> bool:
    """Return whether two scalar/profile payloads are numerically identical.

    Args:
        left: First payload.
        right: Second payload.
        eps: Relative epsilon for scalar/profile comparisons.

    Returns:
        ``True`` when payloads match within epsilon.
    """
    # Compare profile payloads element-wise.
    left_profile = as_profile_array(left)
    right_profile = as_profile_array(right)
    if left_profile is not None or right_profile is not None:
        if left_profile is None or right_profile is None:
            return False
        if left_profile.shape != right_profile.shape:
            return False
        return bool(np.allclose(left_profile, right_profile, rtol=eps, atol=eps))

    # Compare scalar payloads with one relative epsilon.
    left_num = _numeric_value(left)
    right_num = _numeric_value(right)
    if left_num is None or right_num is None:
        return False
    scale = max(abs(left_num), abs(right_num), 1.0)
    return abs(left_num - right_num) <= eps * scale


def _values_equal_exact(left: object, right: object) -> bool:
    """Return whether two payloads are exactly equal.

    Args:
        left: First payload.
        right: Second payload.

    Returns:
        ``True`` when both payloads are exactly equal.
    """
    # Compare profile payloads with exact element equality.
    left_profile = as_profile_array(left)
    right_profile = as_profile_array(right)
    if left_profile is not None or right_profile is not None:
        if left_profile is None or right_profile is None:
            return False
        if left_profile.shape != right_profile.shape:
            return False
        return bool(np.array_equal(left_profile, right_profile))

    # Compare scalar payloads exactly.
    left_num = _numeric_value(left)
    right_num = _numeric_value(right)
    if left_num is None or right_num is None:
        return False
    return left_num == right_num


def _preferred_final_value(variable_payload: Mapping[str, object]) -> object | None:
    """Return the best available computed value for one variable.

    Args:
        variable_payload: Payload emitted by ``Reactor.to_table_payload``.

    Returns:
        Preferred final value, or ``None`` when unavailable.
    """
    # Read baseline and current runtime value.
    current_value = variable_payload.get("current_value")
    return current_value


def _input_status(variable_payload: Mapping[str, object]) -> tuple[str, object | None]:
    """Classify one variable payload into display status categories.

    Args:
        variable_payload: Payload emitted by ``Reactor.to_table_payload``.

    Returns:
        Tuple ``(status, final_value)`` where status is one of:
        ``EXACT``, ``WITHIN_TOL``, ``INCONSISTENT``, ``NON_INPUT``.
    """
    # Classify non-input variables first.
    input_source = variable_payload.get("input_source")
    input_value = variable_payload.get("input_value")
    if input_source is None or input_value is None:
        return "NON_INPUT", _preferred_final_value(variable_payload)

    # Resolve one preferred computed value for comparison and optional arrow text.
    final_value = _preferred_final_value(variable_payload)
    diag_status = variable_payload.get("diag_status")

    # Diagnostics inconsistency always wins when available.
    if diag_status == "INCONSISTENT":
        return "INCONSISTENT", final_value

    # Without a computed value, keep the explicit input as exact by default.
    if final_value is None:
        return "EXACT", None

    # Exact equality is green.
    if _values_equal_exact(input_value, final_value):
        return "EXACT", final_value

    # Tolerance equality is orange.
    input_num = _numeric_value(input_value)
    final_num = _numeric_value(final_value)
    rel_tol = variable_payload.get("rel_tol")
    rel = 0.01 if rel_tol is None else float(rel_tol)
    if input_num is not None and final_num is not None:
        if within_tolerance(input_num, final_num, rel_tol=rel):
            return "WITHIN_TOL", final_value

    # Everything else is inconsistent.
    return "INCONSISTENT", final_value


def _format_profile(value: np.ndarray) -> str:
    """Return one compact profile summary string.

    Args:
        value: 1D profile array.

    Returns:
        Compact profile summary string.
    """
    # Render profile cardinality and basic statistics.
    mean_val = float(np.mean(value))
    min_val = float(np.min(value))
    max_val = float(np.max(value))
    mean_text = f"{mean_val:.3g}".replace("e+0", "e").replace("e+", "e")
    min_text = f"{min_val:.3g}".replace("e+0", "e").replace("e+", "e")
    max_text = f"{max_val:.3g}".replace("e+0", "e").replace("e+", "e")
    return f"profile(n={value.size}, mean={mean_text}, min={min_text}, max={max_text})"


def _to_display_value(value: object) -> object:
    """Return one compact display-ready value.

    Args:
        value: Raw scalar/profile payload.

    Returns:
        Scalar or compact string representation.
    """
    # Convert profile payloads to a compact summary.
    profile = as_profile_array(value)
    if profile is not None:
        return _format_profile(profile)

    # Keep scalar numpy arrays compact.
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return float(value.item())
        return f"array({value.size})"

    # Convert symbolic numeric values when possible.
    try:
        import sympy as sp
    except Exception:
        sp = None
    if sp is not None and isinstance(value, sp.Expr) and value.is_number:
        scalar = safe_float(value.evalf())
        return scalar if scalar is not None else value
    return value


def _format_value(value: object) -> str:
    """Return one human-friendly table string.

    Args:
        value: Display-ready scalar/string payload.

    Returns:
        Cell text.
    """
    # Keep explicit strings untouched.
    if isinstance(value, str):
        return value

    # Render tag-like metadata lists as one readable comma-separated string.
    if isinstance(value, (list, tuple, set)):
        values = list(value)
        if isinstance(value, set):
            values = sorted(values, key=lambda item: str(item))
        return ", ".join(str(item) for item in values)

    # Use scientific notation for very small/large scalar numbers.
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        if value != 0 and (abs(value) >= 1e4 or abs(value) <= 1e-4):
            return f"{value:.2e}"

    # Render empty cells as blank strings.
    if value is None:
        return ""
    return str(value)


def _cell_text(variable_payload: dict[str, object]) -> str:
    """Return one table cell string from one variable payload.

    Args:
        variable_payload: Payload emitted by ``Reactor.to_table_payload``.

    Returns:
        HTML-capable cell text.
    """
    # Resolve baseline and solved values once.
    input_value = variable_payload.get("input_value")
    final_value = _preferred_final_value(variable_payload)

    # Keep empty cells blank when no values are present.
    if input_value is None and final_value is None:
        return ""

    status, _ = _input_status(variable_payload)
    input_text = _format_value(_to_display_value(input_value))
    final_text = _format_value(_to_display_value(final_value))

    # Non-input values display their final value only.
    if status == "NON_INPUT":
        return final_text if final_value is not None else input_text

    # Red cells optionally show baseline -> corrected value.
    if status == "INCONSISTENT":
        if final_value is not None and not _values_match(input_value, final_value):
            return f"{input_text} -> {final_text}"
        return input_text

    # Green/orange cells display the baseline input value only.
    return input_text


def _style_cells(
    data: pd.DataFrame,
    *,
    payloads: Mapping[str, dict[str, object]],
    metadata_fields: set[str],
) -> pd.DataFrame:
    """Return one style matrix for reactor comparison cells.

    Args:
        data: Display DataFrame.
        payloads: Per-reactor table payload mapping.
        metadata_fields: Metadata row names.

    Returns:
        DataFrame containing CSS style strings.
    """
    # Initialize one blank style frame with matching index/columns.
    styles = pd.DataFrame("", index=data.index, columns=data.columns)

    # Style every variable cell from diagnostics-aware status.
    for row_name in styles.index:
        for reactor_id in styles.columns:
            parts = ["white-space: pre-line;"]
            if row_name in metadata_fields:
                styles.loc[row_name, reactor_id] = " ".join(parts)
                continue

            reactor_payload = payloads.get(reactor_id, {})
            var_payload = (reactor_payload.get("variables", {}) or {}).get(row_name)
            if var_payload is None:
                styles.loc[row_name, reactor_id] = " ".join(parts)
                continue

            status, _final_value = _input_status(var_payload)

            # Apply one explicit 4-state color policy.
            if status == "EXACT":
                parts.append("background-color: #d9f2d9; color: #111111;")
            elif status == "WITHIN_TOL":
                parts.append("background-color: #ffe8c2; color: #111111;")
            elif status == "INCONSISTENT":
                parts.append("background-color: #ffd6d6; color: #111111;")
            else:
                parts.append("background-color: #ffffff; color: #111111;")

            styles.loc[row_name, reactor_id] = " ".join(parts)
    return styles


def build_reactor_comparison_table(
    reactors: Mapping[str, Reactor] | Iterable[Reactor],
    *,
    metadata_fields: Sequence[str] = DEFAULT_METADATA_FIELDS,
    include_diagnostics: bool = True,
) -> dict[str, object]:
    """Build one diagnostics-aware multi-reactor comparison table.

    Args:
        reactors: Mapping or iterable of Reactor objects.
        metadata_fields: Metadata row names shown before variable rows.
        include_diagnostics: Whether to attach diagnostics status to cells.

    Returns:
        Mapping with ``dataframe``, ``styled``, ``warnings``, and ``payloads``.
    """
    # Normalize reactor input to a stable identifier/object list.
    reactor_pairs = _normalize_reactors(reactors)
    metadata_fields_tuple = tuple(metadata_fields)
    metadata_field_set = set(metadata_fields_tuple)

    # Build per-reactor payloads and keep any diagnostics warnings.
    payloads: dict[str, dict[str, object]] = {}
    warnings: dict[str, list[str]] = {}
    for rid, reactor in reactor_pairs:
        warnings[rid] = []
        try:
            payload = reactor.to_table_payload(
                reactor_id=rid,
                metadata_fields=metadata_fields_tuple,
                include_diagnostics=include_diagnostics,
            )
        except Exception as exc:
            warnings[rid].append(f"to_table_payload failed: {exc}")
            payload = reactor.to_table_payload(
                reactor_id=rid,
                metadata_fields=metadata_fields_tuple,
                include_diagnostics=False,
            )
        payloads[rid] = payload

    # Collect all variable names in canonical registry order.
    variable_names = _ordered_variable_names(payloads)
    row_names = [*metadata_fields_tuple, *variable_names]

    # Materialize one column per reactor for metadata + variable rows.
    table_data: dict[str, dict[str, object]] = {}
    for rid, _reactor in reactor_pairs:
        payload = payloads[rid]
        metadata = payload.get("metadata", {}) or {}
        variables = payload.get("variables", {}) or {}
        column_data: dict[str, object] = {}
        for row_name in row_names:
            if row_name in metadata_field_set:
                column_data[row_name] = _format_value(metadata.get(row_name))
                continue
            var_payload = variables.get(row_name)
            column_data[row_name] = "" if var_payload is None else _cell_text(var_payload)
        table_data[rid] = column_data

    # Build one dataframe and one styled HTML view.
    dataframe = pd.DataFrame.from_dict(table_data, orient="columns")
    dataframe = dataframe.reindex(index=row_names, columns=[rid for rid, _ in reactor_pairs])
    styled = dataframe.style.apply(
        _style_cells,
        axis=None,
        payloads=payloads,
        metadata_fields=metadata_field_set,
    )
    styled = styled.format(escape=None)

    return {
        "dataframe": dataframe,
        "styled": styled,
        "warnings": warnings,
        "payloads": payloads,
    }
