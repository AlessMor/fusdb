"""Plasma geometry relations defined once and solved generically."""

from __future__ import annotations

import sympy as sp

from fusdb.reactors_class import Reactor
from fusdb.relation_class import PRIORITY_STRICT


@Reactor.relation(
    "geometry",
    name="Major radius",
    output="R",
    priority=PRIORITY_STRICT,
    initial_guesses={
        "R": lambda v: (v["R_max"] + v["R_min"]) / 2,
        "R_max": lambda v: 2 * v["R"] - v["R_min"],
        "R_min": lambda v: 2 * v["R"] - v["R_max"],
    },
)
def major_radius(R_max: float, R_min: float) -> float:
    """Return the average major radius from inboard/outboard extents."""
    return (R_max + R_min) / 2


@Reactor.relation(
    "geometry",
    name="Aspect ratio",
    output="A",
    priority=PRIORITY_STRICT,
    constraints=("a != 0", "R > a"),
    initial_guesses={
        "A": lambda v: v["R"] / v["a"],
        "R": lambda v: v["A"] * v["a"],
        "a": lambda v: v["R"] / v["A"],
    },
)
def aspect_ratio(R: float, a: float) -> float:
    """Return aspect ratio from major and minor radius."""
    return R / a


@Reactor.relation(
    "geometry",
    name="Elongation",
    output="kappa",
    constraints=("R_max - R_min != 0",),
    initial_guesses={
        "kappa": lambda v: (v["Z_max"] - v["Z_min"]) / (v["R_max"] - v["R_min"]),
        "Z_max": lambda v: v["Z_min"] + v["kappa"] * (v["R_max"] - v["R_min"]),
        "Z_min": lambda v: v["Z_max"] - v["kappa"] * (v["R_max"] - v["R_min"]),
        "R_max": lambda v: v["R_min"] + (v["Z_max"] - v["Z_min"]) / v["kappa"],
        "R_min": lambda v: v["R_max"] - (v["Z_max"] - v["Z_min"]) / v["kappa"],
    },
)
def elongation(Z_max: float, Z_min: float, R_max: float, R_min: float) -> float:
    """Return elongation from vertical and horizontal extents."""
    return (Z_max - Z_min) / (R_max - R_min)

# NOTE: check if "95" relations are valid for L,I and H-modes
@Reactor.relation(
    ("geometry", "tokamak"),
    name="Elongation 95%",
    output="kappa",
    initial_guesses={"kappa": lambda v: 1.12 * v["kappa_95"], "kappa_95": lambda v: v["kappa"] / 1.12},
)
def elongation_95(kappa_95: float) -> float:
    """Return core elongation from kappa_95.
    N.A. Uckan and ITER Physics Group, ITER Physics Design Guidelines: 1989, ITER Documentation Series, No. 10, IAEA/ITER/DS/10 (1990)
    """
    return 1.12 * kappa_95


# NOTE: check if "95" relations are valid for L,I and H-modes
@Reactor.relation(
    ("geometry", "tokamak"),
    name="Triangularity 95%",
    output="delta",
    initial_guesses={"delta": lambda v: 1.5 * v["delta_95"], "delta_95": lambda v: v["delta"] / 1.5},
)
def triangularity_95(delta_95: float) -> float:
    """Return core triangularity from delta_95.
    N.A. Uckan and ITER Physics Group, ITER Physics Design Guidelines: 1989, ITER Documentation Series, No. 10, IAEA/ITER/DS/10 (1990)
    """
    return 1.5 * delta_95


@Reactor.relation(
    ("geometry", "tokamak"),
    name="Tokamak volume",
    output="V_p",
    variables=("a", "R", "kappa", "delta", "squareness"),
    priority=PRIORITY_STRICT,
    initial_guesses={
        "V_p": lambda v: plasma_volume(v["a"], v["R"], v["kappa"], v["delta"], v["squareness"]),
        "a": lambda v: max(1e-3, (abs(v.get("V_p", 1.0)) ** (1 / 3)) / max(v.get("kappa", 1.0), 1e-3)),
        "R": lambda v: max(1e-3, (abs(v.get("V_p", 1.0)) ** (1 / 3)) * max(v.get("kappa", 1.0), 1e-3)),
    },
)
def plasma_volume(a: float, R: float, kappa: float, delta: float, xi: float) -> float:
    """Return tokamak plasma volume from geometric shaping parameters."""
    epsilon = a / R
    theta07 = sp.asin(0.7) + sp.Piecewise(
        (0, sp.Eq(xi, 0)),
        ((1 - sp.sqrt(1 + 8 * xi**2)) / (4 * xi), True),
    )
    w_07 = sp.cos(theta07 - xi * sp.sin(2 * theta07)) / sp.sqrt(0.51) * (1 - 0.49 / 2 * delta**2)
    S_phi = sp.pi * a**2 * kappa * (1 + 0.52 * (w_07 - 1))
    return 2 * sp.pi * R * (1 - 0.25 * delta * epsilon) * S_phi


@Reactor.relation(
    ("geometry", "tokamak"),
    name="Tokamak surface",
    output="S_p",
    variables=("a", "R", "kappa", "delta", "squareness"),
    priority=PRIORITY_STRICT,
    initial_guesses={
        "S_p": lambda v: plasma_surface_area(v["a"], v["R"], v["kappa"], v["delta"], v["squareness"]),
        "a": lambda v: max(1e-3, (abs(v.get("S_p", 1.0)) ** (1 / 2)) / max(v.get("kappa", 1.0), 1e-3)),
        "R": lambda v: max(1e-3, (abs(v.get("S_p", 1.0)) ** (1 / 2))),
    },
)
def plasma_surface_area(a: float, R: float, kappa: float, delta: float, xi: float) -> float:
    """Return tokamak plasma surface area from geometric shaping parameters."""
    epsilon = a / R
    theta07 = sp.asin(0.7) + sp.Piecewise(
        (0, sp.Eq(xi, 0)),
        ((1 - sp.sqrt(1 + 8 * xi**2)) / (4 * xi), True),
    )
    w_07 = sp.cos(theta07 - xi * sp.sin(2 * theta07)) / sp.sqrt(0.51) * (1 - 0.49 / 2 * delta**2)
    Lp = 2 * sp.pi * a * (1 + 0.55 * (kappa - 1)) * (1 + 0.08 * delta**2) * (1 + 0.2 * (w_07 - 1))
    return 2 * sp.pi * R * (1 - 0.32 * delta * epsilon) * Lp


# Configuration-specific geometry guidance (simplified from PROCESS/STAR/ITER sources).
# For spherical tokamaks, see Menard et al., Nucl. Fusion 2016 and PROCESS Issue #1439/#1086.
@Reactor.relation(
    ("geometry", "spherical_tokamak"),
    name="ST elongation vs aspect ratio",
    output="kappa",
    initial_guesses={"kappa": lambda v: 2.0},
)
def st_elongation_from_aspect_ratio(A: float) -> float:
    """Return spherical tokamak elongation from aspect ratio."""
    return 0.95 * (1.9 + 1.9 / (A ** 1.4))


@Reactor.relation(
    ("geometry", "spherical_tokamak"),
    name="ST triangularity vs aspect ratio",
    output="delta",
    initial_guesses={"delta": lambda v: 0.3},
)
def st_triangularity_from_aspect_ratio(A: float) -> float:
    """Return spherical tokamak triangularity from aspect ratio."""
    return 0.53 * (1 + 0.77 * (1 / A) ** 3) / 1.50
