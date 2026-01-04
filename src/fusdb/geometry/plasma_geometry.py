"""Plasma geometry relations defined once and solved generically."""

from __future__ import annotations

import math

from fusdb.reactors_class import Reactor
from fusdb.relation_class import PRIORITY_STRICT
from fusdb.relations_util import require_nonzero


@Reactor.relation(
    "geometry",
    name="Major radius",
    output="R",
    solve_for=("R", "R_max", "R_min"),
    priority=PRIORITY_STRICT,
    initial_guesses={
        "R": lambda v: (v["R_max"] + v["R_min"]) / 2,
        "R_max": lambda v: 2 * v["R"] - v["R_min"],
        "R_min": lambda v: 2 * v["R"] - v["R_max"],
    },
)
def major_radius(R_max: float, R_min: float) -> float:
    return (R_max + R_min) / 2


@Reactor.relation(
    "geometry",
    name="Aspect ratio",
    output="A",
    solve_for=("A", "R", "a"),
    priority=PRIORITY_STRICT,
    initial_guesses={
        "A": lambda v: v["R"] / v["a"],
        "R": lambda v: v["A"] * v["a"],
        "a": lambda v: v["R"] / v["A"],
    },
)
def aspect_ratio(R: float, a: float) -> float:
    require_nonzero(a, "a", "geometry relations")
    return R / a


@Reactor.relation(
    "geometry",
    name="Elongation",
    output="kappa",
    solve_for=("kappa", "Z_max", "Z_min", "R_max", "R_min"),
    initial_guesses={
        "kappa": lambda v: (v["Z_max"] - v["Z_min"]) / (v["R_max"] - v["R_min"]),
        "Z_max": lambda v: v["Z_min"] + v["kappa"] * (v["R_max"] - v["R_min"]),
        "Z_min": lambda v: v["Z_max"] - v["kappa"] * (v["R_max"] - v["R_min"]),
        "R_max": lambda v: v["R_min"] + (v["Z_max"] - v["Z_min"]) / v["kappa"],
        "R_min": lambda v: v["R_max"] - (v["Z_max"] - v["Z_min"]) / v["kappa"],
    },
)
def elongation(Z_max: float, Z_min: float, R_max: float, R_min: float) -> float:
    require_nonzero(R_max - R_min, "R_span", "geometry relations")
    return (Z_max - Z_min) / (R_max - R_min)


@Reactor.relation(
    ("geometry", "tokamak"),
    name="Elongation 95%",
    output="kappa",
    solve_for=("kappa", "kappa_95"),
    initial_guesses={"kappa": lambda v: 1.12 * v["kappa_95"], "kappa_95": lambda v: v["kappa"] / 1.12},
)
def elongation_95(kappa_95: float) -> float:
    return 1.12 * kappa_95


@Reactor.relation(
    ("geometry", "tokamak"),
    name="Triangularity 95%",
    output="delta",
    solve_for=("delta", "delta_95"),
    initial_guesses={"delta": lambda v: 1.5 * v["delta_95"], "delta_95": lambda v: v["delta"] / 1.5},
)
def triangularity_95(delta_95: float) -> float:
    return 1.5 * delta_95


@Reactor.relation(
    ("geometry", "tokamak"),
    name="Tokamak volume",
    output="V_p",
    variables=("a", "R", "kappa", "delta_95", "squareness"),
    solve_for=("V_p", "a", "R"),
    priority=PRIORITY_STRICT,
    initial_guesses={
        "V_p": lambda v: plasma_volume(v["a"], v["R"], v["kappa"], v["delta_95"], v["squareness"]),
        "a": lambda v: max(1e-3, (abs(v.get("V_p", 1.0)) ** (1 / 3)) / max(v.get("kappa", 1.0), 1e-3)),
        "R": lambda v: max(1e-3, (abs(v.get("V_p", 1.0)) ** (1 / 3)) * max(v.get("kappa", 1.0), 1e-3)),
    },
)
def plasma_volume(a: float, R: float, kappa: float, delta: float, xi: float) -> float:
    epsilon = a / R
    theta07 = math.asin(0.7) if xi == 0 else math.asin(0.7) + (1 - math.sqrt(1 + 8 * xi**2)) / (4 * xi)
    w_07 = math.cos(theta07 - xi * math.sin(2 * theta07)) / math.sqrt(0.51) * (1 - 0.49 / 2 * delta**2)
    S_phi = math.pi * a**2 * kappa * (1 + 0.52 * (w_07 - 1))
    return 2 * math.pi * R * (1 - 0.25 * delta * epsilon) * S_phi


@Reactor.relation(
    ("geometry", "tokamak"),
    name="Tokamak surface",
    output="S_p",
    variables=("a", "R", "kappa", "delta_95", "squareness"),
    solve_for=("S_p", "a", "R"),
    priority=PRIORITY_STRICT,
    initial_guesses={
        "S_p": lambda v: plasma_surface_area(v["a"], v["R"], v["kappa"], v["delta_95"], v["squareness"]),
        "a": lambda v: max(1e-3, (abs(v.get("S_p", 1.0)) ** (1 / 2)) / max(v.get("kappa", 1.0), 1e-3)),
        "R": lambda v: max(1e-3, (abs(v.get("S_p", 1.0)) ** (1 / 2))),
    },
)
def plasma_surface_area(a: float, R: float, kappa: float, delta: float, xi: float) -> float:
    epsilon = a / R
    theta07 = math.asin(0.7) if xi == 0 else math.asin(0.7) + (1 - math.sqrt(1 + 8 * xi**2)) / (4 * xi)
    w_07 = math.cos(theta07 - xi * math.sin(2 * theta07)) / math.sqrt(0.51) * (1 - 0.49 / 2 * delta**2)
    Lp = 2 * math.pi * a * (1 + 0.55 * (kappa - 1)) * (1 + 0.08 * delta**2) * (1 + 0.2 * (w_07 - 1))
    return 2 * math.pi * R * (1 - 0.32 * delta * epsilon) * Lp


# Configuration-specific geometry guidance (simplified from PROCESS/STAR/ITER sources).
# For spherical tokamaks, see Menard et al., Nucl. Fusion 2016 and PROCESS Issue #1439/#1086.
@Reactor.relation(
    ("geometry", "spherical_tokamak"),
    name="ST elongation vs aspect ratio",
    output="kappa",
    solve_for=("kappa",),
    initial_guesses={"kappa": lambda v: 2.0},
)
def st_elongation_from_aspect_ratio(A: float) -> float:
    return 0.95 * (1.9 + 1.9 / (A ** 1.4))


@Reactor.relation(
    ("geometry", "spherical_tokamak"),
    name="ST triangularity vs aspect ratio",
    output="delta_95",
    solve_for=("delta_95",),
    initial_guesses={"delta_95": lambda v: 0.3},
)
def st_triangularity_from_aspect_ratio(A: float) -> float:
    return 0.53 * (1 + 0.77 * (1 / A) ** 3) / 1.50
