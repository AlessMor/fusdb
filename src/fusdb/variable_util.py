"""Utility helpers for :class:`~.Variable` objects.

This module contains the canonical :func:`~.make_variable` factory for concrete
:class:`~.Variable` construction.
"""
from __future__ import annotations

def make_variable(*, ndim: int = 0, **kwargs: object):
    """Create a :class:`~.Variable` instance.

    Factory function that instantiates either :class:`~.Variable0D` or
    :class:`~.Variable1D` based on the dimensionality parameter.

    Args:
        ndim (int): Variable dimensionality (0 for scalar, 1 for profile).
        **kwargs: Additional :class:`~.Variable` constructor arguments.

    Returns:
        Union[:class:`~.Variable0D`, :class:`~.Variable1D`]: A variable instance.

    Raises:
        ValueError: If ``ndim`` is not 0 or 1, or if conflicting ``ndim`` values are provided.
    """
    try:
        ndim = int(ndim)
    except Exception as exc:
        raise ValueError(f"Unsupported variable ndim={ndim}. Supported values are 0 and 1.") from exc

    kwargs = dict(kwargs)
    provided_ndim = kwargs.pop("ndim", ndim)
    if int(provided_ndim) != ndim:
        raise ValueError(f"Conflicting ndim values: ndim={ndim} but kwargs['ndim']={provided_ndim}.")

    if ndim == 0:
        from .variable_class import Variable0D

        return Variable0D(**kwargs, ndim=0)
    elif ndim == 1:
        from .variable_class import Variable1D

        return Variable1D(**kwargs, ndim=1)
    else:
        raise ValueError(f"Unsupported variable ndim={ndim}. Supported values are 0 and 1.")
