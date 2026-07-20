"""Save and load mode results as HDF5.

One archive format for every mode: a reconcile result and a popcon grid go to
the same ``.h5`` layout, so downstream analysis never branches on how the file
was produced.  Nested dicts become HDF5 groups, numeric arrays become
datasets, scalars/strings become attributes, and anything else (lists of
dicts, ``None``) is stored as JSON.  Loading inverts the mapping back into
plain dicts/arrays -- results are archived *data*, not resurrected live
objects (tuples come back as lists, ints inside JSON stay ints).

h5py is an optional dependency: ``pip install fusdb[io]``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

_FORMAT_ATTR = "fusdb_result_format"
_FORMAT_VERSION = 1
_JSON_KIND = "json"
_NULL_TOKEN = "__fusdb_null__"


def _h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise ImportError(
            "Saving/loading results requires h5py; install it with `pip install fusdb[io]`."
        ) from exc
    return h5py


def _escape(key: Any) -> str:
    # "/" nests groups in HDF5 names; relation names may contain it.
    return str(key).replace("/", "\\u2044")


def _unescape(name: str) -> str:
    return name.replace("\\u2044", "/")


def _write_json(group: Any, name: str, value: Any) -> None:
    data = group.create_dataset(name, data=json.dumps(value, default=str))
    data.attrs["kind"] = _JSON_KIND


def _write_entry(group: Any, key: Any, value: Any) -> None:
    name = _escape(key)
    if value is None:
        group.attrs[name] = _NULL_TOKEN
    elif isinstance(value, Mapping):
        _write_group(group.create_group(name), value)
    elif isinstance(value, np.ndarray):
        if value.dtype.kind in "bifu":
            group.create_dataset(name, data=value)
        else:
            _write_json(group, name, value.tolist())
    elif isinstance(value, (bool, int, float, str, np.bool_, np.integer, np.floating)):
        group.attrs[name] = value
    elif isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
        # Non-empty all-numeric sequences round-trip as arrays; anything else
        # (names, dicts, empty lists) keeps its type through JSON.
        if items:
            try:
                arr = np.asarray(items, dtype=float)
                if arr.dtype.kind in "if" and arr.ndim >= 1:
                    group.create_dataset(name, data=arr)
                    return
            except (TypeError, ValueError):
                pass
        _write_json(group, name, items)
    else:
        _write_json(group, name, value)


def _write_group(group: Any, mapping: Mapping[str, Any]) -> None:
    for key, value in mapping.items():
        _write_entry(group, key, value)


def _read_group(group: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, value in group.attrs.items():
        if name == _FORMAT_ATTR:
            continue
        if isinstance(value, str) and value == _NULL_TOKEN:
            out[_unescape(name)] = None
        elif isinstance(value, bytes):
            out[_unescape(name)] = value.decode("utf-8")
        elif isinstance(value, (np.bool_, np.integer, np.floating)):
            out[_unescape(name)] = value.item()
        else:
            out[_unescape(name)] = value
    for name, item in group.items():
        key = _unescape(name)
        if hasattr(item, "keys") and not hasattr(item, "dtype"):  # subgroup
            out[key] = _read_group(item)
        elif item.attrs.get("kind") == _JSON_KIND:
            raw = item[()]
            out[key] = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else str(raw))
        else:
            out[key] = np.asarray(item[()])
    return out


def save_result(result: Mapping[str, Any], path: str | Path) -> Path:
    """Write one mode result dict to ``path`` as HDF5; returns the path.

    Args:
        result: A mode result (or any mapping of plain data).
        path: Target file path; overwritten if it exists.
    """
    h5py = _h5py()
    target = Path(path)
    with h5py.File(target, "w") as handle:
        handle.attrs[_FORMAT_ATTR] = _FORMAT_VERSION
        _write_group(handle, result)
    return target


def load_result(path: str | Path) -> dict[str, Any]:
    """Load a result saved by :func:`save_result` back into plain dicts/arrays."""
    h5py = _h5py()
    with h5py.File(Path(path), "r") as handle:
        return _read_group(handle)
